#!/usr/bin/env python3
"""Print what the retrieval path returned for one question, in the order it injected.

The harness reports one number per question: `first_rank`. A measurement that moves
a question's expected chunk says that it moved, not what moved it, and `--json-out`
does not store the chunk text. This script prints every returned candidate with its
`score` and `rerank_score`, its file, section, chunk index, the text it carries, and
a `*` where the question's `relevant` entry matches, so a ranking change can be read
as a cause: which chunk came first, and what number put it there.

    python scripts/diagnose_rerank.py --question-id mt-003
    python scripts/diagnose_rerank.py --question-id q-022 --mode search --rerank off
    python scripts/diagnose_rerank.py --question-id mt-003 --rerank on \
        --rerank-top-n 30 --top-k 20
    python scripts/diagnose_rerank.py --question-id mt-003 --mode search \
        --retrieval-mode dense --retrieval-query concat --rerank off

`search` asks `api` directly (`POST /search`) and prints the order that endpoint
returns. `gateway` asks the gateway (`POST /v1/chat/completions`) and reads
`X-RAG-Sources`, which is the order the context block was built in, so the follow-up
rewrite (`RAG_QUERY_REWRITE`) is part of the request being measured. Gateway mode
has no chunk text in the header, so text is sliced from the corpus by
`char_start`/`char_end` the same way `scripts/eval_retrieval.py` does it — that is
what makes `text_contains` and the section match work in both modes.

Stdlib only. The stack must be running.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_retrieval as ev  # noqa: E402

DEFAULT_GATEWAY_URL = "http://localhost:8001"
DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_CORPUS = ROOT / "eval" / "corpus"
TEXT_HEAD = 78
# The gateway's `concat` rewrite caps (`RAG_REWRITE_HISTORY`, per-turn and total
# characters), so `--retrieval-query concat` in search mode sends the same string
# the shipped gateway default sends.
CONCAT_HISTORY = 3
CONCAT_TURN_CHARS = 500
CONCAT_QUERY_CHARS = 2000


def annotate(hits: List[Dict[str, Any]], relevant: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The returned order, scored against the question's expected entries.

    `ev.rank_hits` is the harness's rule: the order reranking produced when every
    hit carries a `rerank_score`, otherwise the order the service returned. `match`
    is the harness's own relevance test, so the `*` here is the same criterion that
    produced `first_rank` in the run.
    """
    rows: List[Dict[str, Any]] = []
    for rank, hit in enumerate(ev.rank_hits(hits), 1):
        rows.append({
            "rank": rank,
            "file": Path(str(hit.get("file", ""))).name,
            "section": str(hit.get("section") or ""),
            "chunk": int(hit.get("chunk", 0)),
            "score": float(hit.get("score", 0.0)),
            "rerank_score": hit.get("rerank_score"),
            "match": ev._is_relevant(hit, relevant),
            "text": str(hit.get("text") or ""),
        })
    return rows


def print_report(question: Dict[str, Any], rows: List[Dict[str, Any]], settings: Dict[str, Any],
                text_head: int) -> None:
    print(f"{question['id']}  {question['question']}")
    if question.get("history"):
        print(f"  history: {len(question['history'])} turn(s): {question['history'][0]}")
    print(f"  expected: " + " | ".join(
        f"{entry.get('file', '')} " + (f"section '{entry['section']}'" if "section" in entry
                                       else f"chunk {entry['chunk']}" if "chunk" in entry
                                       else f"contains '{entry['text_contains']}'")
        for entry in question["relevant"]
    ))
    matched = [row["rank"] for row in rows if row["match"]]
    first_rank = matched[0] if matched else None
    print(f"  returned {len(rows)}  rerank={settings['rerank']} "
          f"pool={settings['rerank_top_n']}  first_rank={first_rank}"
          + (f"  matches at {matched}" if len(matched) > 1 else ""))
    # Which request produced this order. The rank is a property of the query text
    # and the fusion, so a comparison is only read if both are printed.
    mode = settings.get("retrieval_mode")
    query_setting = settings.get("retrieval_query")
    pair_setting = settings.get("rerank_query")
    if mode is not None or query_setting is not None or pair_setting is not None:
        print(f"  request: mode={mode or 'service default'} query={query_setting}"
              f"  {settings.get('query', '')[:80]!r}")
        # Which text the cross-encoder scored: `retrieval` is the text above,
        # `question` is the user's message. A pair-query reading is only read if it
        # is printed, because the ranking is a property of the pair, not of the pool.
        print(f"  pair query: {pair_setting or 'retrieval (service default)'}")
    print()
    print(f"{'rank':>4}  {'score':>6}  {'rerank':>7}  {'match':>5}  "
          f"{'file':<17} {'chunk':>4}  section")
    for row in rows:
        rerank = f"{row['rerank_score']:.4f}" if isinstance(row["rerank_score"], (int, float)) else "-"
        print(f"{row['rank']:>4}  {row['score']:.4f}  {rerank:>7}  "
              f"{'*' if row['match'] else '':>5}  {row['file']:<17} {row['chunk']:>4}  "
              f"{row['section'][:34]}")
        if row["match"] and text_head:
            # the matched chunk is the answer to the question, so show what it says
            print(f"       {'':<2} {row['text'][:text_head].replace(chr(10), ' | ')}")


def concat_query(history: List[str], query: str) -> str:
    """The gateway's `concat` rewrite, so `--mode search` measures the same text.

    The retrieval query is what decides rank, and the shipped gateway default is
    `concat`: the last `RAG_REWRITE_HISTORY` user turns, each capped, then the
    question, capped. `mt-003` is exactly a case where that matters - its question
    is Russian and its history is English, so the query the model sees is mixed
    language. Search mode has to send the same string as the gateway does,
    otherwise a dense-vs-hybrid rank reading is about a different request.
    """
    parts = [text[-CONCAT_TURN_CHARS:] for text in history[-CONCAT_HISTORY:]]
    parts.append(query)
    return "\n".join(parts)[-CONCAT_QUERY_CHARS:]


def fetch_search(session: ev.Session, query: str, top_k: int,
                 min_score: float, rerank: Optional[bool], rerank_top_n: Optional[int],
                 retrieval_mode: Optional[str] = None,
                 rerank_query: Optional[str] = None) -> List[Dict[str, Any]]:
    """`POST /search` on `api`, the same request shape the harness sends."""
    request: Dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "min_score": None if min_score == ev.NO_THRESHOLD else min_score,
    }
    # `mode` is the per-request dense/hybrid knob `eval_retrieval.py` uses, so the
    # same tool can show where a chunk sat before fusion and after it.
    if retrieval_mode:
        request["mode"] = retrieval_mode
    if rerank is not None:
        request["rerank"] = bool(rerank)
    if rerank_top_n is not None:
        request["rerank_top_n"] = int(rerank_top_n)
    # `rerank_query` is the pair text the model judges: set it when the retrieval
    # query is a long rewrite, so the pairs are built from the question the rewrite
    # came from. Sent only with reranking on, the same rule as the pool key.
    if rerank_query is not None and rerank is not False:
        request["rerank_query"] = rerank_query
    payload, _ = session.post("/search", request)
    if not isinstance(payload, list):
        raise RuntimeError(f"/search returned {type(payload).__name__}, expected a list")
    return [h for h in (ev._normalize_search_hit(hit) for hit in payload) if h is not None]


def fetch_gateway(session: ev.Session, question: Dict[str, Any], model: str, top_k: int,
                  rerank: Optional[bool], rerank_top_n: Optional[int],
                  texts: Dict[str, str], rerank_query_mode: Optional[str] = None) -> List[Dict[str, Any]]:
    """Ask the gateway and read `X-RAG-Sources`, the order the context block used.

    The answers are not part of the measurement: the gateway's retrieval is what is
    being printed, so the upstream only has to answer. `scripts/upstream_stub.py`
    answers instantly, which keeps the request cost on the retrieval path.
    """
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-RAG": "on",
        "X-RAG-TOP-K": str(top_k),
        "X-RAG-MIN-SCORE": str(ev.NO_THRESHOLD),
    }
    if rerank is not None:
        headers["X-RAG-RERANK"] = "on" if rerank else "off"
    if rerank_top_n is not None:
        headers["X-RAG-RERANK-TOP-N"] = str(int(rerank_top_n))
    if rerank_query_mode is not None:
        headers["X-RAG-RERANK-QUERY"] = rerank_query_mode

    messages = [{"role": "user", "content": turn} for turn in question.get("history", [])]
    messages.append({"role": "user", "content": question["question"]})
    _, response_headers = session.post("/v1/chat/completions", {
        "model": model,
        "messages": messages,
        "stream": False,
        "max_tokens": 64,
        "temperature": 0,
    }, headers)

    hits: List[Dict[str, Any]] = []
    for hit in ev.sources_from_headers(response_headers):
        name = Path(str(hit.get("file", ""))).name
        normalized = texts.get(name)
        start, end = hit.get("char_start"), hit.get("char_end")
        chunk_text = ""
        if normalized is not None and isinstance(start, int) and isinstance(end, int):
            chunk_text = normalized[start:end]
        rerank_score = hit.get("rerank_score")
        hits.append({
            "file": str(hit.get("file", "")),
            "chunk": int(hit.get("chunk", 0)),
            "text": chunk_text,
            "section": str(hit.get("section") or ""),
            "score": float(hit.get("score", 0.0)),
            "rerank_score": float(rerank_score) if isinstance(rerank_score, (int, float)) else None,
        })
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=str(ROOT / "eval" / "questions.multiturn.jsonl"))
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--mode", choices=("gateway", "search"), default="gateway")
    parser.add_argument("--base-url", default=None,
                        help=f"gateway mode: {DEFAULT_GATEWAY_URL}; search mode: {DEFAULT_API_URL}")
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS),
                        help="corpus directory used to slice gateway-mode chunk text")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-score", type=float, default=ev.NO_THRESHOLD,
                        help="threshold sent to the service; -1 sends none")
    parser.add_argument("--rerank", choices=("on", "off", "default"), default="default",
                        help="cross-encoder on the request path: `on` asks for it, "
                             "`off` forces it off, `default` sends no field")
    parser.add_argument("--rerank-top-n", type=int, default=None,
                        help=f"candidates the cross-encoder scores (1..{ev.MAX_RERANK_TOP_N})")
    parser.add_argument("--model", default=None,
                        help="gateway mode: model name to ask for; resolved from /v1/models if omitted")
    parser.add_argument("--show-text", type=int, default=TEXT_HEAD,
                        help="characters of the matched chunk to print (0 = no text)")
    parser.add_argument("--retrieval-mode", choices=("hybrid", "dense"), default=None,
                        help="search mode: ask api for dense-only, or hybrid (the service default). "
                             "Shows where a chunk ranked before BM25 joined the fusion")
    parser.add_argument("--retrieval-query", choices=("question", "concat"), default="question",
                        help="search mode: send the question alone, or the gateway's `concat` "
                             "rewrite of history + question, so both modes measure the same request")
    parser.add_argument("--rerank-query", choices=("retrieval", "question"), default=None,
                        help="which text the cross-encoder scores: `retrieval` (the text above) "
                             "or `question` (the user's message, so a long rewrite is judged "
                             "against the question it came from). Omitted = send no field")
    args = parser.parse_args()

    if args.rerank_query is not None and args.rerank == "off":
        parser.error("--rerank-query with --rerank off asks the model to score nothing")

    if args.retrieval_mode is not None and args.mode != "search":
        parser.error("--retrieval-mode is a /search knob; gateway mode has no way to ask for it")

    if args.rerank_top_n is not None and not 1 <= args.rerank_top_n <= ev.MAX_RERANK_TOP_N:
        parser.error(f"--rerank-top-n must be in 1..{ev.MAX_RERANK_TOP_N}, got {args.rerank_top_n}")
    if args.rerank_top_n is not None and args.rerank == "off":
        parser.error("--rerank-top-n with --rerank off asks the model to score nothing")

    rerank = None if args.rerank == "default" else args.rerank == "on"
    base_url = args.base_url or (DEFAULT_GATEWAY_URL if args.mode == "gateway" else DEFAULT_API_URL)
    questions = ev.load_questions(Path(args.questions), gateway=args.mode == "gateway",
                                  corpus_available=True)
    question = next((q for q in questions if q["id"] == args.question_id), None)
    if question is None:
        parser.error(f"--question-id {args.question_id} is not in {args.questions}")

    # The concat rewrite needs turns to prepend, and in gateway mode the gateway does
    # the rewriting itself, so asking for it there measures nothing.
    if args.retrieval_query == "concat" and args.mode == "search" and not question.get("history"):
        parser.error("--retrieval-query concat needs a history to prepend")

    query_text = (concat_query(question["history"], question["question"])
                  if args.retrieval_query == "concat" and args.mode == "search"
                  else question["question"])
    # The pair text the model judges. In search mode the tool sends the text itself,
    # so `question` + `--retrieval-query concat` measures the fix without a gateway:
    # retrieve with the rewrite, score against the question.
    pair_text = question["question"] if args.rerank_query == "question" else None

    session = ev.Session(base_url, 60.0)
    try:
        if args.mode == "search":
            hits = fetch_search(session, query_text, args.top_k, args.min_score,
                                rerank, args.rerank_top_n, args.retrieval_mode,
                                pair_text)
            texts: Dict[str, str] = {}
        else:
            texts = ev.corpus_texts(Path(args.corpus))
            model = args.model
            if not model:
                listing, _ = session.get("/v1/models")
                entries = listing.get("data") if isinstance(listing, dict) else None
                if not isinstance(entries, list) or not entries:
                    raise RuntimeError(f"no models at {base_url}/v1/models - pass --model")
                model = str(entries[0].get("id"))
            hits = fetch_gateway(session, question, model, args.top_k, rerank,
                                 args.rerank_top_n, texts, args.rerank_query)
    finally:
        session.close()

    settings = {"rerank": args.rerank, "rerank_top_n": args.rerank_top_n,
                "retrieval_mode": args.retrieval_mode, "retrieval_query": args.retrieval_query,
                "rerank_query": args.rerank_query, "query": query_text}
    print_report(question, annotate(hits, question["relevant"]), settings, args.show_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
