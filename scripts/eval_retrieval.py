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
`--corpus` validates the ground truth offline against the sample corpus, so a
question set does not rot silently when the documents change.

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
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

DEFAULT_QUESTIONS = Path("eval/questions.jsonl")
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_GATEWAY_URL = "http://localhost:8001"
DEFAULT_TOP_K = 50
DEFAULT_GATEWAY_TOP_K = 20
DEFAULT_RECALL_K = "5,10"
DEFAULT_SWEEP_TOP_K = "3,5,8,10"
DEFAULT_SWEEP_MIN_SCORE = "0.3,0.4,0.5,0.6,off"

# A threshold below every cosine score, i.e. "no threshold" for the gateway.
NO_THRESHOLD = -1.0

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


def _validate_expected(entry: Any, where: str, gateway: bool) -> None:
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
    if gateway and "text_contains" in entry:
        # The gateway reports sources in a header without the chunk text, so
        # ground truth in gateway mode can only use file / chunk.
        _fail(where, "'text_contains' is not available in gateway mode; use 'file' and 'chunk'")
    if "chunk" in entry:
        try:
            chunk = int(entry["chunk"])
        except (TypeError, ValueError):
            _fail(where, "'chunk' must be an integer")
        if chunk < 0:
            _fail(where, "'chunk' must be >= 0")
        entry["chunk"] = chunk


def load_questions(path: Path, gateway: bool = False) -> List[Dict[str, Any]]:
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
                _validate_expected(entry, where, gateway)

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


def _first_relevant_rank(hits: List[Dict[str, Any]], relevant: List[Dict[str, Any]]) -> Optional[int]:
    for rank, hit in enumerate(hits, 1):
        if _is_relevant(hit, relevant):
            return rank
    return None


def _ndcg_at_k(hits: List[Dict[str, Any]], relevant: List[Dict[str, Any]], k: int) -> float:
    dcg = 0.0
    for rank, hit in enumerate(hits[:k], 1):
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
    gateway's own filter.
    """
    kept = [hit for hit in hits[:top_k] if float(hit.get("score", 0.0)) >= min_score]
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
    return {
        "file": str(hit.get("file", "")),
        "chunk": int(hit.get("chunk", 0)),
        "text": str(hit.get("text", "")),
        "section": str(hit.get("section") or ""),
        "score": float(score),
    }


def evaluate_search(session: Session, questions: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for question in questions:
        start = time.perf_counter()
        payload, _ = session.post(
            "/search",
            {"query": question["question"], "top_k": top_k, "min_score": None},
        )
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
    return {
        "id": question["id"],
        "question": question["question"],
        "expected": relevant,
        "returned": len(hits),
        "first_rank": first_rank,
        "first_relevant_score": first_relevant_score,
        "latency_s": round(latency, 3),
        "scores": [round(hit["score"], 4) for hit in hits],
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


def evaluate_gateway(session: Session, api_key: str, model: str,
                     questions: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    """Retrieve through the gateway and read `X-RAG-Sources` from the response.

    The request carries the `history` turns plus the question, exactly as an agent
    would send them, so follow-up rewriting and the "fresh user turn" rule are
    part of what is measured. `X-RAG-TOP-K` is raised to the gateway cap and
    `X-RAG-MIN-SCORE` is disabled, so the sweep can apply any configuration
    offline to the raw candidate list.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-RAG": "on",
        "X-RAG-TOP-K": str(top_k),
        "X-RAG-MIN-SCORE": str(NO_THRESHOLD),
    }
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
        hits = [
            {
                "file": str(hit.get("file", "")),
                "chunk": int(hit.get("chunk", 0)),
                "text": "",
                "section": str(hit.get("section") or ""),
                "score": float(hit.get("score", 0.0)),
            }
            for hit in sources_from_headers(response_headers)
        ]
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
    latencies = sorted(result["latency_s"] for result in results if result["latency_s"] is not None)
    if latencies:
        summary["avg_latency_s"] = round(sum(latencies) / len(latencies), 3)
        summary["p50_latency_s"] = latencies[len(latencies) // 2]
        p95 = int(math.ceil(0.95 * len(latencies))) - 1
        summary["p95_latency_s"] = latencies[min(len(latencies) - 1, p95)]
    return summary


def sweep(results: List[Dict[str, Any]], top_ks: List[int], min_scores: List[float]) -> Dict[str, Any]:
    """Apply the (top_k, min_score) grid to the candidates of one retrieval pass."""
    total = len(results)
    grid: Dict[str, Any] = {"top_ks": top_ks, "min_scores": min_scores, "rows": []}

    for min_score in min_scores:
        row: Dict[str, Any] = {"min_score": min_score}
        for k in top_ks:
            recall = 0.0
            injected = 0.0
            zero = 0
            mrr = 0.0
            for result in results:
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
        header = "min_score " + " ".join(f"{f'{metric}@{k}':>12}" for k in top_ks)
        print(header)
        print("-" * len(header))
        for row in grid["rows"]:
            label = "off" if row["min_score"] <= NO_THRESHOLD else f"{row['min_score']:.2f}"
            cells = " ".join(f"{row[f'{metric}@{k}']:>12.3f}" for k in top_ks)
            print(f"{label:>9}    {cells}")
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
    ]
    for metric, title in blocks:
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| min_score | " + " | ".join(f"{metric}@{k}" for k in top_ks) + " |")
        lines.append("|" + "---|" * (len(top_ks) + 1))
        for row in grid["rows"]:
            label = "off" if row["min_score"] <= NO_THRESHOLD else f"{row['min_score']:.2f}"
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

    questions = load_questions(Path(args.questions), gateway=gateway_mode)
    if args.limit is not None:
        questions = questions[: max(0, args.limit)]
        if not questions:
            _fail(str(args.questions), "--limit removed all questions")

    corpus_problems: List[str] = []
    if args.corpus:
        corpus_dir = Path(args.corpus)
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
    try:
        if gateway_mode:
            model = _resolve_model(session, args.api_key, args.model)
            results = evaluate_gateway(session, args.api_key, model, questions, top_k)
        else:
            results = evaluate_search(session, questions, top_k)
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

    summary = summarise(results, args.recall_k, top_k)
    print_report(summary, results, args.recall_k, args.show_failures)

    grid = sweep(results, _parse_ks(args.sweep_top_k), _parse_scores(args.sweep_min_score)) if args.sweep else None
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
            **summary,
            "results": serialisable,
        }
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
