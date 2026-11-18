#!/usr/bin/env python3
"""Retrieval quality evaluation for the RAG stack.

Reads a JSONL question set, retrieves candidates, and reports recall@k, MRR@10,
and nDCG@10. Two retrieval modes are supported:

  search   - `POST /search` on the embedding API (port 8000), `min_score: null`,
            so the score threshold does not hide results.
  gateway  - `POST /v1/chat/completions` on the gateway (port 8001) with the
            conversation `history`, reading the retrieved sources back from the
            `X-RAG-Sources` response header. This is the only way to measure the
            gateway's own retrieval behavior (follow-up rewriting, injection).

`--sweep` applies a `top_k` x `min_score` grid to the candidates already
retrieved in one pass, so tuning does not cost one API run per configuration.
`--rerank on` asks the cross-encoder to score the candidate pool, and
`--sweep-rerank-top-n 30,50` runs one pass per pool, so the P1.1 sweep compares
pool sizes against the unlabelled fused baseline. `--rerank off` forces
reranking off, which measures on/off against one deployed configuration.
`--corpus` validates the ground truth offline against the sample corpus, so a
question set does not rot silently when the documents change. In gateway mode
`--corpus` is also what makes `text_contains` usable: the harness slices the
corpus file at `char_start:char_end` to recover the text of each retrieved chunk.

All requests reuse one keep-alive connection: on Windows a fresh TCP connection
to a Docker-published port costs ~20 s, so opening one connection per question
turns a 60-question run into 20 minutes of connection setup.

See `eval/README.md` for the dataset format. The script is stdlib-only.

Usage:
    python scripts/eval_retrieval.py --questions eval/questions.jsonl
    python scripts/eval_retrieval.py --questions eval/questions.jsonl --corpus eval/corpus
    python scripts/eval_retrieval.py --questions eval/questions.jsonl \
        --sweep --json-out eval/baseline.json
    python scripts/eval_retrieval.py --questions eval/questions.jsonl --mode gateway \
        --api-key sk-local --sweep
    python scripts/eval_retrieval.py --questions eval/questions.jsonl \
        --rerank on --rerank-top-n 30 --json-out eval/rerank-on.json
    python scripts/eval_retrieval.py --questions eval/questions.jsonl --mode gateway \
        --api-key sk-local --rerank on --sweep-rerank-top-n 30,50 \
        --sweep --sweep-md eval/sweep-rerank.md --min-recall 0.70 --min-mrr 0.35
"""

from __future__ import annotations

import argparse
import base64
import http.client
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit

DEFAULT_QUESTIONS = Path("eval/questions.jsonl")
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_GATEWAY_URL = "http://localhost:8001"
DEFAULT_TOP_K = 50
DEFAULT_GATEWAY_TOP_K = 20
DEFAULT_RECALL_K = "5,10"
DEFAULT_SWEEP_TOP_K = "3,5,8,10"
DEFAULT_SWEEP_MIN_SCORE = "0.3,0.4,0.5,0.6,off"
DEFAULT_SWEEP_RERANK_TOP_N = "30"

# A threshold below every cosine score, i.e. "no threshold" for the gateway.
NO_THRESHOLD = -1.0
# The api's cap on rerank_top_n (RERANK_CANDIDATE_CAP), mirrored so the sweep asks for real values.
MAX_RERANK_TOP_N = 100

_ALLOWED_EXPECTED_FIELDS = {"file", "chunk", "text_contains", "section"}


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


class HttpError(RuntimeError):
    """A non-2xx response, reported with its status and a truncated body."""

    def __init__(self, status: int, body: str) -> None:
        self.status = status
        self.body = body
        super().__init__(f"HTTP {status}: {body}")


class Session:
    """One keep-alive connection reused for the whole run.

    On Windows a fresh TCP connection to a Docker-published port costs ~20 s
    (the port proxy wakes up) while a request on an open connection costs tens
    of milliseconds. `urllib` opens one connection per request, so a 60-question
    run spends 20 minutes in connection setup; reusing one connection makes the
    same run take seconds.
    """

    def __init__(self, base_url: str, timeout: float) -> None:
        parts = urlsplit(base_url)
        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise SystemExit(f"invalid base URL: {base_url!r} (use http://host:port)")
        self.base_url = base_url
        self.host = parts.hostname
        self.port = parts.port or (443 if parts.scheme == "https" else 80)
        self.timeout = timeout
        self._https = parts.scheme == "https"
        self._conn: Optional[http.client.HTTPConnection] = None

    def _connect(self) -> http.client.HTTPConnection:
        if self._conn is None:
            factory = http.client.HTTPSConnection if self._https else http.client.HTTPConnection
            self._conn = factory(self.host, self.port, timeout=self.timeout)
        return self._conn

    def _reset(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None

    def close(self) -> None:
        self._reset()

    def request(self, path: str, payload: Optional[Dict[str, Any]],
               headers: Dict[str, str]) -> Tuple[Any, Dict[str, str]]:
        """Send one request and return the decoded body plus lowercased headers.

        Uvicorn delivers header names lowercased, so they are normalized here.
        A connection closed by the server is reopened once.
        """
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        method = "POST" if payload is not None else "GET"
        for attempt in (1, 2):
            connection = self._connect()
            try:
                connection.request(method, path, body=body, headers=headers)
                response = connection.getresponse()
                raw = response.read()
            except (http.client.HTTPException, OSError) as exc:
                self._reset()
                if attempt == 2:
                    raise ConnectionError(f"cannot reach {self.base_url}: {exc}") from exc
                continue
            if response.status >= 400:
                raise HttpError(response.status, raw[:500].decode("utf-8", "replace"))
            return json.loads(raw), {key.lower(): value for key, value in response.getheaders()}
        raise ConnectionError(f"cannot reach {self.base_url}")

    def post(self, path: str, payload: Dict[str, Any],
             headers: Optional[Dict[str, str]] = None) -> Tuple[Any, Dict[str, str]]:
        return self.request(path, payload, {"Content-Type": "application/json", **(headers or {})})

    def get(self, path: str, headers: Optional[Dict[str, str]] = None) -> Tuple[Any, Dict[str, str]]:
        return self.request(path, None, dict(headers or {}))


def decode_sources(header: Optional[str]) -> List[Dict[str, Any]]:
    """Decode `X-RAG-Sources` (unpadded base64url JSON) into a list of hits."""
    if not header:
        return []
    padded = header + "=" * (-len(header) % 4)
    payload = json.loads(base64.urlsafe_b64decode(padded))
    return [hit for hit in payload if isinstance(hit, dict)]


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


def _fail(where: str, message: str) -> None:
    raise SystemExit(f"{where}: {message}")


def _validate_expected(entry: Any, where: str, gateway: bool, corpus_available: bool = False) -> None:
    if not isinstance(entry, dict):
        _fail(where, "'relevant' entries must be JSON objects")
    unknown = set(entry) - _ALLOWED_EXPECTED_FIELDS
    if unknown:
        _fail(where, f"unknown relevant field(s): {', '.join(sorted(unknown))}")
    if not entry:
        _fail(where, "a 'relevant' entry must have at least one field")

    if "file" in entry and not isinstance(entry["file"], str):
        _fail(where, "'file' must be a string")
    if "text_contains" in entry and not isinstance(entry["text_contains"], str):
        _fail(where, "'text_contains' must be a string")
    if gateway and "text_contains" in entry and not corpus_available:
        # The gateway reports sources in a header without the chunk text. With
        # `--corpus` the text is read back offline from the corpus file through
        # the `char_start`/`char_end` offsets, so 'text_contains' works there.
        _fail(where, "'text_contains' is not available in gateway mode without --corpus; use 'file' and 'chunk'")
    if "chunk" in entry:
        try:
            chunk = int(entry["chunk"])
        except (TypeError, ValueError):
            _fail(where, "'chunk' must be an integer")
        if chunk < 0:
            _fail(where, "'chunk' must be >= 0")
        entry["chunk"] = chunk


def load_questions(path: Path, gateway: bool = False, corpus_available: bool = False) -> List[Dict[str, Any]]:
    if not path.is_file():
        _fail(
            str(path),
            "question file not found. Copy eval/questions.example.jsonl and "
            "replace the examples with real questions.",
        )

    questions: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            where = f"{path}:{line_number}"
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                _fail(where, f"invalid JSON: {exc}")

            if not isinstance(item, dict):
                _fail(where, "each line must be a JSON object")
            question = item.get("question")
            relevant = item.get("relevant")
            if not isinstance(question, str) or not question.strip():
                _fail(where, "'question' must be a non-empty string")
            if not isinstance(relevant, list) or not relevant:
                _fail(where, "'relevant' must be a non-empty array")
            for entry in relevant:
                _validate_expected(entry, where, gateway, corpus_available)

            history = item.get("history", [])
            if not isinstance(history, list) or not all(isinstance(turn, str) for turn in history):
                _fail(where, "'history' must be an array of strings (previous user turns)")

            question_id = item.get("id")
            if question_id is None:
                question_id = f"q{len(questions) + 1:03d}"
            questions.append(
                {
                    "id": str(question_id),
                    "question": question.strip(),
                    "relevant": relevant,
                    "history": [turn.strip() for turn in history if turn.strip()],
                }
            )

    if not questions:
        _fail(str(path), "no questions found")
    return questions


def check_corpus(questions: List[Dict[str, Any]], corpus_dir: Path) -> List[str]:
    """Validate ground truth against the corpus files, offline (no API calls).

    Every `file` must exist in the corpus. `text_contains` must occur in it as a
    substring; `section` is a heading path, so each of its segments must occur in
    the file. Used to catch a question set that refers to documents that were
    renamed, edited, or never committed.
    """
    by_name: Dict[str, List[Path]] = {}
    for path in sorted(corpus_dir.rglob("*")):
        if path.is_file():
            by_name.setdefault(path.name, []).append(path)

    problems: List[str] = []
    for question in questions:
        for expected in question["relevant"]:
            name = expected.get("file")
            if not name:
                continue
            candidates = by_name.get(Path(name).name, [])
            if not candidates:
                problems.append(f"{question['id']}: corpus file '{name}' not found in {corpus_dir}")
                continue
            texts = [p.read_text(encoding="utf-8").casefold() for p in candidates]

            needle = expected.get("text_contains")
            if needle is not None and not any(needle.casefold() in text for text in texts):
                problems.append(f"{question['id']}: text_contains {needle!r} not in '{name}'")

            needle = expected.get("section")
            if needle is not None:
                # the section path is assembled from heading lines, so the joined
                # string never appears in the file: check each segment
                segments = [s.strip().casefold() for s in needle.split(">") if s.strip()]
                if not any(all(s in text for s in segments) for text in texts):
                    problems.append(f"{question['id']}: section {needle!r} not in '{name}'")
    return problems


def corpus_texts(corpus_dir: Path) -> Dict[str, str]:
    """Map each corpus file name to its normalized text (CRLF/CR converted to LF).

    The chunker records character offsets in the normalized text, so slicing this
    text with `char_start`/`char_end` reproduces the chunk exactly. That is how
    `text_contains` works in gateway mode, where the source header carries no text.
    """
    texts: Dict[str, str] = {}
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        texts.setdefault(path.name, raw.replace("\r\n", "\n").replace("\r", "\n"))
    return texts


# --------------------------------------------------------------------------- #
# Matching and metrics
# --------------------------------------------------------------------------- #


def _same_file(actual: str, expected: str) -> bool:
    if actual == expected:
        return True
    # Accept a host-side path (data/documents/x.txt) and a container-side path
    # (/data/documents/x.txt) as the same document.
    try:
        return Path(actual).name == Path(expected).name
    except (OSError, ValueError):
        return False


def _matches(hit: Dict[str, Any], expected: Dict[str, Any]) -> bool:
    if "file" in expected and not _same_file(str(hit.get("file", "")), expected["file"]):
        return False
    if "chunk" in expected:
        try:
            actual_chunk = int(hit.get("chunk"))
        except (TypeError, ValueError):
            return False
        if actual_chunk != expected["chunk"]:
            return False
    if "text_contains" in expected:
        haystack = str(hit.get("text", "")).casefold()
        if expected["text_contains"].casefold() not in haystack:
            return False
    if "section" in expected:
        # The section path survives in `X-RAG-Sources`, so it is the stable
        # criterion in gateway mode: it does not depend on chunk indices.
        haystack = str(hit.get("section", "")).casefold()
        if expected["section"].casefold() not in haystack:
            return False
    return True


def _is_relevant(hit: Dict[str, Any], relevant: List[Dict[str, Any]]) -> bool:
    return any(_matches(hit, expected) for expected in relevant)


def rank_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The order the injected chunks appear in, as the gateway sends them.

    Reranking orders candidates by `rerank_score` and `X-RAG-Sources` carries that
    order, so a reranked pass has to be scored in it, not in the order the fused
    list arrived in. A `None` rerank_score means reranking did not run, so the
    returned order is kept; ties keep the returned order, the same rule
    `api/rerank.py` applies, so the same request scored twice gives the same numbers.
    """
    if all(not isinstance(hit.get("rerank_score"), (int, float)) for hit in hits):
        return list(hits)
    return sorted(hits, key=lambda hit: (
        hit["rerank_score"] if isinstance(hit.get("rerank_score"), (int, float)) else float("-inf"),
    ), reverse=True)


def _first_relevant_rank(hits: List[Dict[str, Any]], relevant: List[Dict[str, Any]]) -> Optional[int]:
    for rank, hit in enumerate(rank_hits(hits), 1):
        if _is_relevant(hit, relevant):
            return rank
    return None


def _ndcg_at_k(hits: List[Dict[str, Any]], relevant: List[Dict[str, Any]], k: int) -> float:
    dcg = 0.0
    for rank, hit in enumerate(rank_hits(hits)[:k], 1):
        if _is_relevant(hit, relevant):
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def score_metrics(hits: List[Dict[str, Any]], relevant: List[Dict[str, Any]],
                 top_k: int, min_score: float) -> Dict[str, float]:
    """Metrics for one configuration, as the gateway applies them.

    Candidates are cut at `top_k` first, then the score threshold removes hits
    below `min_score`, exactly like `POST /search` with `score_threshold` and the
    gateway's own filter. The order is the one `rank_hits` gives, so a reranked
    pass is measured on the order the gateway would have injected.
    """
    kept = rank_hits([hit for hit in hits[:top_k] if float(hit.get("score", 0.0)) >= min_score])
    first_rank = None
    for rank, hit in enumerate(kept, 1):
        if _is_relevant(hit, relevant):
            first_rank = rank
            break
    return {
        "injected": float(len(kept)),
        "recall": 1.0 if first_rank is not None and first_rank <= top_k else 0.0,
        "zero_context": 1.0 if not kept else 0.0,
        "mrr": 1.0 / first_rank if first_rank is not None and first_rank <= 10 else 0.0,
    }


# --------------------------------------------------------------------------- #
# Retrieval modes
# --------------------------------------------------------------------------- #


def _normalize_search_hit(hit: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(hit, dict):
        return None
    score = hit.get("score")
    if not isinstance(score, (int, float)):
        return None
    rerank_score = hit.get("rerank_score")
    if isinstance(rerank_score, (int, float)):
        rerank_score = float(rerank_score)
    else:
        # Reranking did not run, or the api has no reranker loaded.
        rerank_score = None
    return {
        "file": str(hit.get("file", "")),
        "chunk": int(hit.get("chunk", 0)),
        "text": str(hit.get("text", "")),
        "section": str(hit.get("section") or ""),
        "score": float(score),
        "rerank_score": rerank_score,
    }


def evaluate_search(session: Session, questions: List[Dict[str, Any]], top_k: int,
                    retrieval_mode: Optional[str] = None,
                    sparse_weight: Optional[float] = None,
                    sparse_min_relative: Optional[float] = None,
                    rerank: Optional[bool] = None,
                    rerank_top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """Retrieve through `POST /search`.

    `rerank` is the per-request flag: `True` asks the api to score the candidate
    pool with the cross-encoder, `False` forces reranking off for this request even
    when `RERANK_ENABLED=true` on the service, `None` (the default) sends no field
    and uses the service's configured default, so a baseline run is byte-identical
    to a build without reranking. `rerank_top_n` is the pool the model sees.
    """
    results: List[Dict[str, Any]] = []

    for question in questions:
        start = time.perf_counter()
        request = {"query": question["question"], "top_k": top_k, "min_score": None}
        if retrieval_mode:
            # "hybrid" (dense + BM25, RRF) or "dense". Omitted = the api default.
            request["mode"] = retrieval_mode
        if sparse_weight is not None:
            request["sparse_weight"] = sparse_weight
        if sparse_min_relative is not None:
            request["sparse_min_relative"] = sparse_min_relative
        if rerank is not None:
            # `false` is a real answer, not an absent field: it lets the harness
            # compare on/off against one deployed configuration.
            request["rerank"] = bool(rerank)
        if rerank_top_n is not None:
            request["rerank_top_n"] = int(rerank_top_n)
        payload, _ = session.post("/search", request)
        latency = time.perf_counter() - start
        if not isinstance(payload, list):
            raise RuntimeError(
                f"{question['id']}: /search returned {type(payload).__name__}, expected a list"
            )
        hits = [h for h in (_normalize_search_hit(hit) for hit in payload) if h is not None]
        results.append(_build_result(question, hits, latency))

    return results


def _build_result(question: Dict[str, Any], hits: List[Dict[str, Any]],
                  latency: float) -> Dict[str, Any]:
    relevant = question["relevant"]
    first_rank = _first_relevant_rank(hits, relevant)
    first_relevant_score = (
        round(hits[first_rank - 1]["score"], 4) if first_rank is not None else None
    )
    scores = [hit["rerank_score"] for hit in hits if isinstance(hit.get("rerank_score"), (int, float))]
    return {
        "id": question["id"],
        "question": question["question"],
        "expected": relevant,
        "returned": len(hits),
        "first_rank": first_rank,
        "first_relevant_score": first_relevant_score,
        "latency_s": round(latency, 3),
        "scores": [round(hit["score"], 4) for hit in hits],
        # The reranker ran for this question when every returned hit carries a
        # score. A fallback pass (load failure, timeout, bad score shape) returns
        # the fused list with no scores, which must show up in the run record.
        "reranked": bool(scores) and len(scores) == len(hits),
        "rerank_scores": [round(score, 4) for score in scores],
        "hits": hits,
    }


def _resolve_model(session: Session, api_key: str, model: Optional[str]) -> str:
    if model:
        return model
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    listing, _ = session.get("/v1/models", headers)
    entries = listing.get("data") if isinstance(listing, dict) else None
    if not isinstance(entries, list) or not entries:
        raise RuntimeError(
            f"no models at {session.base_url}/v1/models - set UPSTREAM_BASE_URL on "
            "the gateway or pass --model"
        )
    first = entries[0].get("id") if isinstance(entries[0], dict) else None
    if not first:
        raise RuntimeError(f"{session.base_url}/v1/models returned an unexpected entry")
    return str(first)


def read_gateway_defaults(session: Session, api_key: str) -> Dict[str, Any]:
    """The gateway's effective RAG defaults, read from `/health`.

    A committed eval run has to say which configuration produced its numbers:
    the rewrite mode lives in the gateway process, not in the harness, and it
    changes retrieval quality on follow-up questions. Returns {} when the
    gateway does not report defaults, so a run against an older gateway still
    completes.
    """
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    try:
        body, _ = session.get("/health", headers)
    except (HttpError, ConnectionError, OSError, ValueError):
        return {}
    defaults = body.get("defaults") if isinstance(body, dict) else None
    return defaults if isinstance(defaults, dict) else {}


def evaluate_gateway(session: Session, api_key: str, model: str,
                     questions: List[Dict[str, Any]], top_k: int,
                     texts: Optional[Dict[str, str]] = None,
                     rerank: Optional[bool] = None,
                     rerank_top_n: Optional[int] = None,
                     rerank_query: Optional[str] = None) -> List[Dict[str, Any]]:
    """Retrieve through the gateway and read `X-RAG-Sources` from the response.

    The request carries the `history` turns plus the question, exactly as an agent
    would send them, so follow-up rewriting and the "fresh user turn" rule are
    part of what is measured. `X-RAG-TOP-K` is raised to the gateway cap and
    `X-RAG-MIN-SCORE` is disabled, so the sweep can apply any configuration
    offline to the raw candidate list.

    `rerank` sends `X-RAG-RERANK: on|off` (omitted = the gateway's `RAG_RERANK`),
    `rerank_top_n` sends `X-RAG-RERANK-TOP-N`, `rerank_query` sends
    `X-RAG-RERANK-QUERY: retrieval|question` - which text the cross-encoder scores
    the passages against. The sources header carries `rerank_score`, which is the
    order the context block was built in.

    `texts` maps corpus file names to normalized text. The header has no chunk
    text, so each hit's text is sliced from the corpus by `char_start`/`char_end`;
    that makes `text_contains` usable in gateway mode.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-RAG": "on",
        "X-RAG-TOP-K": str(top_k),
        "X-RAG-MIN-SCORE": str(NO_THRESHOLD),
    }
    if rerank is not None:
        headers["X-RAG-RERANK"] = "on" if rerank else "off"
    if rerank_top_n is not None:
        headers["X-RAG-RERANK-TOP-N"] = str(int(rerank_top_n))
    if rerank_query is not None:
        headers["X-RAG-RERANK-QUERY"] = rerank_query
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    results: List[Dict[str, Any]] = []
    for question in questions:
        messages = [{"role": "user", "content": turn} for turn in question["history"]]
        messages.append({"role": "user", "content": question["question"]})
        start = time.perf_counter()
        _, response_headers = session.post("/v1/chat/completions", {
            "model": model,
            "messages": messages,
            "stream": False,
            "max_tokens": 64,
            "temperature": 0,
        }, headers)
        latency = time.perf_counter() - start
        hits: List[Dict[str, Any]] = []
        for hit in sources_from_headers(response_headers):
            name = Path(str(hit.get("file", ""))).name
            normalized = (texts or {}).get(name)
            start_offset = hit.get("char_start")
            end_offset = hit.get("char_end")
            chunk_text = ""
            if normalized is not None and isinstance(start_offset, int) and isinstance(end_offset, int):
                chunk_text = normalized[start_offset:end_offset]
            rerank_score = hit.get("rerank_score")
            hits.append({
                "file": str(hit.get("file", "")),
                "chunk": int(hit.get("chunk", 0)),
                "text": chunk_text,
                "section": str(hit.get("section") or ""),
                "score": float(hit.get("score", 0.0)),
                "rerank_score": float(rerank_score) if isinstance(rerank_score, (int, float)) else None,
            })
        results.append(_build_result(question, hits, latency))

    return results


def sources_from_headers(response_headers: Dict[str, str]) -> List[Dict[str, Any]]:
    """Read `X-RAG-Sources` from normalized (lowercased) response headers."""
    return decode_sources(response_headers.get("x-rag-sources"))


# --------------------------------------------------------------------------- #
# Aggregation and sweep
# --------------------------------------------------------------------------- #


def summarise(results: List[Dict[str, Any]], recall_ks: List[int], top_k: int,
              recall_key: str = "first_rank") -> Dict[str, Any]:
    total = len(results)
    summary: Dict[str, Any] = {"questions": total, "top_k": top_k}
    for k in recall_ks:
        summary[f"recall@{k}"] = sum(
            1.0 if result["first_rank"] is not None and result["first_rank"] <= k else 0.0
            for result in results
        ) / total
    summary["mrr@10"] = sum(
        1.0 / result["first_rank"] if result["first_rank"] is not None and result["first_rank"] <= 10 else 0.0
        for result in results
    ) / total
    summary["ndcg@10"] = sum(
        _ndcg_at_k(result["hits"], result["expected"], 10) for result in results
    ) / total
    summary["avg_candidates"] = sum(result["returned"] for result in results) / total
    summary["rerank_rate"] = sum(1 for result in results if result.get("reranked")) / total
    latencies = sorted(result["latency_s"] for result in results if result["latency_s"] is not None)
    if latencies:
        summary["avg_latency_s"] = round(sum(latencies) / len(latencies), 3)
        summary["p50_latency_s"] = latencies[len(latencies) // 2]
        p95 = int(math.ceil(0.95 * len(latencies))) - 1
        summary["p95_latency_s"] = latencies[min(len(latencies) - 1, p95)]
    return summary


def sweep(results: List[Dict[str, Any]], top_ks: List[int], min_scores: List[float],
          rerank_top_ns: Sequence[int] = (None,)) -> Dict[str, Any]:
    """Apply the (top_k, min_score) grid to the candidates of one retrieval pass.

    Each row is one (rerank_top_n, min_score) configuration: `top_k` cuts the
    list, `min_score` drops hits below the threshold, `rerank_top_n` is the pool
    the cross-encoder scored, so the P1.1 sweep is `30` against `50` against the
    pool reranking was never asked for. `rerank_rate` in a row is the fraction of
    questions that were actually reranked in that pass, so a pass where the model
    failed to load is visible as a row of `off` numbers, not as a silent baseline.
    """
    grid: Dict[str, Any] = {
        "top_ks": top_ks,
        "min_scores": min_scores,
        "rerank_top_ns": list(rerank_top_ns),
        "rows": [],
    }

    for rerank_top_n in rerank_top_ns:
        subset = [result for result in results if result.get("rerank_top_n") == rerank_top_n]
        total = len(subset)
        if not total:
            continue
        for min_score in min_scores:
            row: Dict[str, Any] = {
                "min_score": min_score,
                "rerank_top_n": rerank_top_n,
                "rerank_rate": sum(1 for result in subset if result.get("reranked")) / total,
            }
            for k in top_ks:
                recall = 0.0
                injected = 0.0
                zero = 0
                mrr = 0.0
                for result in subset:
                    metrics = score_metrics(result["hits"], result["expected"], k, min_score)
                    recall += metrics["recall"]
                    injected += metrics["injected"]
                    zero += metrics["zero_context"]
                    mrr += metrics["mrr"]
                row[f"recall@{k}"] = recall / total
                row[f"injected@{k}"] = injected / total
                row[f"mrr@{k}"] = mrr / total
                row[f"zero_context@{k}"] = zero / total
            grid["rows"].append(row)

    return grid


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def print_report(summary: Dict[str, Any], results: List[Dict[str, Any]],
                recall_ks: List[int], show_failures: int) -> None:
    print(f"Questions: {summary['questions']}  (top_k={summary['top_k']})")
    for k in recall_ks:
        print(f"  recall@{k}: {summary[f'recall@{k}']:.3f}")
    print(f"  MRR@10:   {summary['mrr@10']:.3f}")
    print(f"  nDCG@10:  {summary['ndcg@10']:.3f}")
    if "avg_latency_s" in summary:
        print(f"  latency:  avg {summary['avg_latency_s']:.2f} s  "
              f"p50 {summary['p50_latency_s']:.2f} s  p95 {summary['p95_latency_s']:.2f} s")

    max_k = max(recall_ks)
    failures = [
        result
        for result in results
        if result["first_rank"] is None or result["first_rank"] > max_k
    ]
    print(f"Failures in top {max_k}: {len(failures)}/{len(results)}")

    if show_failures and failures:
        print()
        for result in failures[: max(0, show_failures)]:
            rank = result["first_rank"] if result["first_rank"] is not None else "-"
            print(f"  {result['id']}: rank={rank}  {result['question'][:100]}")
            print(f"    expected: {json.dumps(result['expected'], ensure_ascii=False)}")


def _row_label(row: Dict[str, Any]) -> str:
    """The row label: `min_score`, plus the rerank pool when the grid has that axis.

    With two pools in the grid the same `min_score` appears twice, so the label
    has to say which pool the cross-encoder scored. `off` means reranking was not
    asked for, and a pool with `rerank_rate=0` in the row means the model was asked
    for and did not run, so the numbers are the fused baseline.
    """
    score = "off" if row["min_score"] <= NO_THRESHOLD else f"{row['min_score']:.2f}"
    pool = row.get("rerank_top_n")
    return score if pool is None else f"{score} rerank={pool}"


def print_sweep(grid: Dict[str, Any], top_ks: List[int]) -> None:
    """Print recall, injected-context, and zero-context tables for the grid."""
    blocks = (
        ("recall", "Recall of the relevant chunk among the injected context"),
        ("injected", "Average number of injected chunks per question"),
        ("zero_context", "Questions that get no context at all"),
    )
    print()
    for metric, title in blocks:
        print(title)
        header = "config " + " ".join(f"{f'{metric}@{k}':>12}" for k in top_ks)
        print(header)
        print("-" * len(header))
        for row in grid["rows"]:
            label = _row_label(row)
            cells = " ".join(f"{row[f'{metric}@{k}']:>12.3f}" for k in top_ks)
            print(f"{label:>18}    {cells}")
        print()


def sweep_markdown(grid: Dict[str, Any], top_ks: List[int], generated_at: str, md_path: str) -> str:
    """Markdown tables of the grid: recall, MRR, injected chunks, zero-context rate."""
    blocks = (
        ("recall", "Recall of the relevant chunk among the injected context"),
        ("mrr", "MRR within the injected context (ranked within `top_k` only)"),
        ("injected", "Average number of injected chunks per question"),
        ("zero_context", "Fraction of questions that get no context at all"),
    )
    lines = [
        "# Tuning sweep",
        "",
        "Generated by `python scripts/eval_retrieval.py --sweep --sweep-md %s`" % md_path,
        f"(generated at {generated_at}).",
        "",
        "The grid is applied offline to the candidates of one retrieval pass: `top_k` cuts",
        "the list, `min_score` drops hits below the threshold, exactly as `POST /search` with",
        "`score_threshold` and the gateway apply them. All numbers are averages over the",
        "questions of the dataset.",
        "",
        "Rows labelled `rerank=N` are a pass that asked the cross-encoder to score a pool of",
        "`N` candidates and ranked the injected chunks by `rerank_score`; unlabelled rows use",
        "the fused order. A `rerank=N` row whose reranker did not run (`rerank_rate=0` in the",
        "JSON) is the fused baseline: the reranker is never a hard dependency.",
        "",
    ]
    for metric, title in blocks:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| min_score | " + " | ".join(f"{metric}@{k}" for k in top_ks) + " |")
        lines.append("|" + "---|" * (len(top_ks) + 1))
        for row in grid["rows"]:
            label = _row_label(row)
            cells = " | ".join(f"{row[f'{metric}@{k}']:.3f}" for k in top_ks)
            lines.append(f"| {label} | {cells} |")
        lines.append("")
    return "\n".join(lines)


def _parse_ks(value: str) -> List[int]:
    try:
        ks = sorted({int(part) for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --recall-k: {value}") from exc
    if not ks or ks[0] < 1:
        raise argparse.ArgumentTypeError("--recall-k must contain positive integers")
    return ks


def _parse_scores(value: str) -> List[float]:
    scores: List[float] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        scores.append(NO_THRESHOLD if part.lower() in {"off", "none", "-1"} else float(part))
    if not scores:
        raise argparse.ArgumentTypeError("--sweep-min-score must contain numbers or 'off'")
    return sorted(scores)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--mode", choices=("search", "gateway"), default="search",
                        help="retrieve through the embedding API (default) or the gateway")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="embedding API base URL")
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL, help="gateway base URL")
    parser.add_argument("--api-key", default="", help="bearer key for the gateway")
    parser.add_argument("--retrieval-mode", choices=("hybrid", "dense"), default=None,
                        help="send `mode` to /search; omitted = the api's SEARCH_MODE default")
    parser.add_argument("--sparse-weight", type=float, default=None,
                        help="RRF weight of the BM25 branch, sent to /search (tuning)")
    parser.add_argument("--sparse-min-relative", type=float, default=None,
                        help="drop BM25 hits below this fraction of the best BM25 score")
    parser.add_argument("--rerank", choices=("on", "off", "default"), default="default",
                        help="cross-encoder reranking: `on` asks for it, `off` forces it off "
                             "for every request, `default` sends no field and uses RERANK_ENABLED")
    parser.add_argument("--rerank-top-n", type=int, default=None,
                        help=f"candidates the cross-encoder scores per request (1..{MAX_RERANK_TOP_N})")
    parser.add_argument("--rerank-query", choices=("retrieval", "question"), default=None,
                        help="which text the cross-encoder scores the passages against: "
                             "`retrieval` (the text retrieval was asked with, what the gateway "
                             "sends today) or `question` (the user's message, so a long rewrite "
                             "is judged against the question it was built from). Omitted = no "
                             "field is sent, so the service default applies. Gateway mode only: "
                             "in search mode the retrieval query is the question, so the pair "
                             "query is the same text")
    parser.add_argument("--sweep-rerank-top-n", default=DEFAULT_SWEEP_RERANK_TOP_N,
                        help="rerank pools to run as extra retrieval passes with --sweep --rerank on")
    parser.add_argument("--model", default=None, help="model id for gateway mode")
    parser.add_argument("--top-k", type=int, default=None,
                        help=f"candidates per question (default {DEFAULT_TOP_K} for search, "
                             f"{DEFAULT_GATEWAY_TOP_K} for gateway)")
    parser.add_argument("--recall-k", type=_parse_ks, default=_parse_ks(DEFAULT_RECALL_K))
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--show-failures", type=int, default=20)
    parser.add_argument("--corpus", default=None,
                        help="validate ground truth against corpus files, offline")
    parser.add_argument("--sweep", action="store_true",
                        help="apply the top_k x min_score grid to the retrieved candidates")
    parser.add_argument("--sweep-top-k", default=DEFAULT_SWEEP_TOP_K)
    parser.add_argument("--sweep-min-score", default=DEFAULT_SWEEP_MIN_SCORE)
    parser.add_argument("--sweep-md", default=None,
                        help="write the sweep tables as Markdown to this file (requires --sweep)")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--min-recall", type=float, default=None,
                        help="exit 1 if recall@max(recall-k) is below this value")
    parser.add_argument("--min-mrr", type=float, default=None,
                        help="exit 1 if MRR@10 is below this value")
    parser.add_argument("--validate-only", action="store_true",
                        help="validate the question file and exit without calling the API")
    args = parser.parse_args()

    gateway_mode = args.mode == "gateway"
    top_k = args.top_k if args.top_k is not None else (
        DEFAULT_GATEWAY_TOP_K if gateway_mode else DEFAULT_TOP_K
    )
    if top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.sweep_md and not args.sweep:
        parser.error("--sweep-md requires --sweep")

    # `default` sends no rerank field at all, so a baseline run stays byte-identical
    # to a build without reranking. `off` is an explicit answer: the harness can
    # compare on/off against one deployed configuration.
    rerank = None if args.rerank == "default" else args.rerank == "on"
    pools: List[Optional[int]] = [args.rerank_top_n]
    if args.sweep:
        pools = _parse_ks(args.sweep_rerank_top_n)
    for pool in pools:
        if pool is not None and not 1 <= pool <= MAX_RERANK_TOP_N:
            parser.error(f"rerank_top_n must be in 1..{MAX_RERANK_TOP_N}, got {pool}")
    if args.rerank_top_n is not None and rerank is False:
        parser.error("--rerank-top-n with --rerank off asks the model to score nothing")
    if len(pools) > 1 and rerank is not True:
        parser.error("--sweep-rerank-top-n runs one retrieval pass per pool, so it needs --rerank on")
    if args.rerank_query is not None and rerank is False:
        parser.error("--rerank-query with --rerank off asks the model to score nothing")
    if args.rerank_query is not None and not gateway_mode:
        parser.error("--rerank-query is a gateway knob: in search mode the retrieval query is "
                     "the question, so the pair query is the same text and the field changes nothing")

    corpus_dir = Path(args.corpus) if args.corpus else None
    questions = load_questions(Path(args.questions), gateway=gateway_mode,
                               corpus_available=corpus_dir is not None)
    if args.limit is not None:
        questions = questions[: max(0, args.limit)]
        if not questions:
            _fail(str(args.questions), "--limit removed all questions")

    corpus_problems: List[str] = []
    if corpus_dir:
        if not corpus_dir.is_dir():
            _fail(str(corpus_dir), "corpus directory not found")
        corpus_problems = check_corpus(questions, corpus_dir)
        for problem in corpus_problems:
            print(f"  {problem}")
        if not corpus_problems:
            print(f"Ground truth matches the corpus ({len(questions)} questions)")

    if args.validate_only:
        print(f"{len(questions)} question(s) OK: {args.questions}")
        return 1 if corpus_problems else 0

    base_url = args.gateway_url if gateway_mode else args.api_url
    session = Session(base_url, args.timeout)
    gateway_defaults: Dict[str, Any] = {}
    results: List[Dict[str, Any]] = []
    try:
        if gateway_mode:
            model = _resolve_model(session, args.api_key, args.model)
            gateway_defaults = read_gateway_defaults(session, args.api_key)
        for pool in pools:
            # One retrieval pass per rerank pool. `rerank_top_n` is the size of the
            # candidate set the cross-encoder sees, so `30` and `50` are two passes
            # over the same questions, not one pass with two thresholds.
            if gateway_mode:
                batch = evaluate_gateway(session, args.api_key, model, questions, top_k,
                                         corpus_texts(corpus_dir) if corpus_dir else None,
                                         rerank, pool, args.rerank_query)
            else:
                batch = evaluate_search(session, questions, top_k, args.retrieval_mode,
                                        args.sparse_weight, args.sparse_min_relative,
                                        rerank, pool)
            for result in batch:
                # The pool this pass asked the model to score, so the sweep grid can
                # tell the passes apart. `None` = reranking was not asked for.
                result["rerank_top_n"] = pool
            results.extend(batch)
    except HttpError as exc:
        print(f"HTTP {exc.status} from {base_url}: {exc.body}")
        return 1
    except (ConnectionError, OSError) as exc:
        print(f"Cannot reach {base_url}: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - report and exit non-zero
        print(f"Evaluation failed: {exc}")
        return 1
    finally:
        session.close()

    # The headline numbers describe the first pass; the sweep rows carry every pass.
    headline = [result for result in results if result["rerank_top_n"] == pools[0]]
    summary = summarise(headline, args.recall_k, top_k)
    print_report(summary, headline, args.recall_k, args.show_failures)
    if rerank is False:
        print("rerank: forced off for every request")
    elif rerank is True:
        print(f"rerank: on (pool {pools[0]}), reranked {summary['rerank_rate']:.2f} of questions")
    else:
        print(f"rerank: service default, reranked {summary['rerank_rate']:.2f} of questions")
    if gateway_mode and gateway_defaults.get("query_rewrite"):
        print(f"gateway rewrite: {gateway_defaults['query_rewrite']} "
              f"(history={gateway_defaults.get('rewrite_history', '?')})")
    if gateway_mode and args.rerank_query is not None:
        print(f"rerank pair query: {args.rerank_query}")

    grid = sweep(results, _parse_ks(args.sweep_top_k), _parse_scores(args.sweep_min_score),
                pools) if args.sweep else None
    if grid is not None:
        print_sweep(grid, grid["top_ks"])

    if args.sweep_md:
        md_path = Path(args.sweep_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        md_path.write_text(sweep_markdown(grid, grid["top_ks"], stamp, args.sweep_md) + "\n", encoding="utf-8")
        print(f"Wrote {md_path}")

    if args.json_out:
        serialisable = [{k: v for k, v in result.items() if k != "hits"} for result in results]
        output = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": str(Path(args.questions)),
            "mode": args.mode,
            "base_url": base_url,
            # what the retrieval passes asked for, so a rerank-*.json file is
            # comparable against eval/baseline.json line by line
            "rerank": args.rerank,
            "rerank_top_n": pools[0],
            "rerank_pools": pools,
            "rerank_query": args.rerank_query,
            "retrieval_mode": args.retrieval_mode,
            **summary,
            "results": serialisable,
        }
        if gateway_defaults:
            # which gateway configuration produced these numbers
            output["gateway_defaults"] = gateway_defaults
        if grid is not None:
            output["sweep"] = grid
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Wrote {output_path}")

    max_k = max(args.recall_k)
    failed = bool(corpus_problems)
    if args.min_recall is not None and summary[f"recall@{max_k}"] < args.min_recall:
        print(
            f"recall@{max_k}={summary[f'recall@{max_k}']:.3f} "
            f"is below --min-recall={args.min_recall:.3f}"
        )
        failed = True
    if args.min_mrr is not None and summary["mrr@10"] < args.min_mrr:
        print(f"MRR@10={summary['mrr@10']:.3f} is below --min-mrr={args.min_mrr:.3f}")
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
