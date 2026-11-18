#!/usr/bin/env python3
"""Latency probe for the embedding API under indexing load.

Bulk indexing and user queries share one GPU model, so the question P2.4 asks is
how much a user has to wait while the indexer is working. The probe answers it by
measuring `POST /search` latency in three situations:

  idle        - the stack is not indexing anything;
  ingest      - synthetic documents are written to the watched folder and the
                indexer is busy embedding them;
  settled     - indexing finished, the same requests again.

`POST /embed` is measured as well: that is the endpoint queued by
`EMBEDDING_MAX_CONCURRENCY`.

Requests reuse one keep-alive connection per worker through
`eval_retrieval.Session`. On Windows a fresh TCP connection to a
Docker-published port costs seconds, which would swamp the numbers; connection
setup is not what is being measured here.

p50 / p95 / p99 are reported for each phase, together with the number of chunks
the indexer added and the latency multiplier under ingest. `--rerank on` / `--rerank off`
measure the same path with and without the cross-encoder, and `--rerank-compare` runs
both on one deployment, so the reranker's latency cost is a number and not a guess.
Stdlib only. The probe writes synthetic files only into the documents folder it is
pointed at, and removes them at the end.

Usage:
    python scripts/latency_probe.py
    python scripts/latency_probe.py --files 12 --concurrency 4 --requests 40
    python scripts/latency_probe.py --skip-ingest --json-out perf.json
    python scripts/latency_probe.py --rerank on --rerank-top-n 30 --rerank-compare \
        --json-out eval/perf-rerank.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_retrieval as ev  # noqa: E402

DEFAULT_API_URL = "http://localhost:8000"
DEFAULT_DOCS_DIR = Path("data/documents")
DEFAULT_QUERIES = [
    "embedding model",
    "RAG_TOP_K",
    "chunk size",
    "Qdrant snapshot",
    "gateway authentication",
]
BULK_MARK = "perf-"

SAMPLE_SENTENCES = [
    "The gateway exposes an OpenAI-compatible chat endpoint.",
    "Retrieval runs before the request is forwarded upstream.",
    "Documents are chunked by headings, paragraphs, and sentences.",
    "Each chunk carries a section path and character offsets.",
    "The index is a derived representation; the files stay authoritative.",
    "Sources are returned in a response header for citations.",
]


# --------------------------------------------------------------------------- #
# Latency statistics
# --------------------------------------------------------------------------- #


def percentile(values: List[float], point: float) -> float:
    """Nearest-rank percentile on a list of latencies (seconds).

    `ceil(p * n)`-th smallest value, the standard nearest-rank definition: the
    p50 of 1..100 is 50, not the average of 50 and 51.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(point * len(ordered)) - 1))
    return ordered[index]


def summarise(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
    return {
        "count": len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "mean": statistics.fmean(values),
    }


def mode_name(rerank: Optional[bool]) -> str:
    """Which search path a phase measured, in words a reader can use.

    Naming the counterpart of `--rerank` `idle_rerank` is right when the primary is
    fused and wrong when the primary is rerank: the phase named `idle_rerank` then
    measured the fused path, and the printed multiplier said scoring the pool cost
    7% of a fused request. Every phase says which mode it ran, so the pair is read by
    mode and the run is unambiguous whichever side `--rerank` was asked for.
    """
    if rerank is None:
        return "default"
    return "rerank" if rerank else "fused"


def report(name: str, stats: Dict[str, float], budget: Budget,
           mode: Optional[str] = None) -> Dict[str, float]:
    """Record a phase, and say it out loud as it finishes.

    A probe that prints only the table at the end looks like a hanging shell. Each
    phase says its numbers and the time the budget has left, so a run that overruns
    is diagnosed at the phase that overran, not watched.
    """
    stats["budget_left"] = round(budget.left(), 2)
    if mode:
        stats["mode"] = mode
    print(
        f"  {name}: n={stats['count']} p50={stats['p50']:.2f}s p95={stats['p95']:.2f}s "
        f"budget left {stats['budget_left']:.1f}s"
    )
    return stats


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #


class Budget:
    """Wall-clock limit for the whole probe run.

    The owner's rule is that a measurement must not run for more than a minute, and
    a budget that is only checked between phases does not hold: the ingest phase
    alone used to wait 240s for the indexer. Every loop in this file asks the budget
    before starting another request, so the run ends at the deadline and reports what
    it measured up to that point. `left()` is printed with each phase, so an
    over-budget run is visible as a number, not as a hanging shell.
    """

    def __init__(self, seconds: float) -> None:
        self.deadline = time.monotonic() + seconds

    def left(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def spent(self) -> bool:
        return self.left() <= 0.0


def measure_search(sessions: List[ev.Session], queries: List[str], requests: int,
                   duration: Optional[float] = None, rerank: Optional[bool] = None,
                   rerank_top_n: Optional[int] = None,
                   budget: Optional[Budget] = None) -> List[float]:
    """Latency of `POST /search`, one connection per parallel worker.

    With `duration` the loop runs for that many seconds instead of a fixed
    number of requests, which is how the ingest phase is kept overlapping the
    indexer's work. Round-robin over sessions keeps every connection busy.

    `rerank` and `rerank_top_n` are sent as the fields `api` accepts, so one probe
    measures the fused path (`--rerank off`) and the reranked path
    (`--rerank on --rerank-top-n 30`). Scoring the pool costs more GPU time than
    embedding the query, and the numbers say whether it is worth paying it.
    `default` sends no field, so the probe reports what a client that asks for
    nothing gets.
    """
    latencies: List[float] = []
    index = 0
    # A duration window is shortened to the time the budget still allows, so the
    # ingest phase cannot outlive the run it belongs to.
    window = min(duration, budget.left()) if duration is not None and budget else duration
    deadline = time.monotonic() + window if window is not None else None
    while True:
        if budget is not None and budget.spent():
            break
        if deadline is not None:
            if time.monotonic() >= deadline:
                break
        elif index >= requests:
            break
        session = sessions[index % len(sessions)]
        query = queries[index % len(queries)]
        request: Dict[str, Any] = {"query": query, "top_k": 5, "min_score": None}
        if rerank is not None:
            # `false` is an answer, not an absent field: `off` measures the fused
            # path on a service that has a reranker loaded.
            request["rerank"] = bool(rerank)
        if rerank_top_n is not None:
            request["rerank_top_n"] = int(rerank_top_n)
        start = time.perf_counter()
        try:
            session.post("/search", request)
            latencies.append(time.perf_counter() - start)
        except Exception as exc:  # noqa: BLE001 - the probe reports, not raises
            print(f"  search request failed: {exc}", file=sys.stderr)
        index += 1
    return latencies


def measure_encode(session: ev.Session, texts: List[str], calls: int,
                   budget: Optional[Budget] = None) -> List[float]:
    """Latency of `POST /embed` for one batch, `calls` times in sequence."""
    latencies: List[float] = []
    for _ in range(calls):
        if budget is not None and budget.spent():
            break
        start = time.perf_counter()
        try:
            session.post("/embed", {"texts": texts})
            latencies.append(time.perf_counter() - start)
        except Exception as exc:  # noqa: BLE001
            print(f"  embed request failed: {exc}", file=sys.stderr)
    return latencies


def warm_up(sessions: List[ev.Session], query: str, rerank: Optional[bool],
           rerank_top_n: Optional[int]) -> float:
    """Put every connection, and the GPU, into the state the probe is meant to measure.

    Measured on the live stack: the first request on a connection that has not been
    used, and the first request after the GPU has been idle, costs ~20s, while the
    same request on a warmed connection costs ~1s. That cost is connection setup and
    CUDA wake-up, not the queue the task is about, so it is paid once here and never
    inside a timed phase. Warming the rerank path warms the reranker too, so the
    fused phase measured after it does not pay the model.
    """
    request: Dict[str, Any] = {"query": query, "top_k": 5, "min_score": None}
    if rerank is not None:
        request["rerank"] = bool(rerank)
    if rerank_top_n is not None:
        request["rerank_top_n"] = int(rerank_top_n)
    start = time.perf_counter()
    for session in sessions:
        try:
            session.post("/search", request)
        except Exception as exc:  # noqa: BLE001 - the probe reports, not raises
            print(f"  warm-up request failed: {exc}", file=sys.stderr)
    return time.perf_counter() - start


def points_count(session: ev.Session) -> int:
    body, _headers = session.get("/collection/info")
    return int(body.get("points_count") or 0)


# --------------------------------------------------------------------------- #
# Bulk corpus
# --------------------------------------------------------------------------- #


def write_bulk_files(docs_dir: Path, files: int, paragraphs: int) -> List[Path]:
    """Create synthetic documents in the watched folder, marked with `BULK_MARK`."""
    created: List[Path] = []
    for index in range(files):
        text = f"# Performance probe document {index}\n\n"
        for section in range(paragraphs):
            text += f"## Section {section}\n\n"
            sentences = [random.choice(SAMPLE_SENTENCES) for _ in range(8)]
            text += " ".join(sentences) + "\n\n"
        path = docs_dir / f"{BULK_MARK}{index:02d}.txt"
        path.write_text(text, encoding="utf-8")
        created.append(path)
    return created


def remove_bulk_files(paths: List[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> Dict[str, Any]:
    queries = args.queries.split(",") if args.queries else DEFAULT_QUERIES
    sessions = [ev.Session(args.api_url, args.timeout) for _ in range(max(1, args.concurrency))]
    budget = Budget(args.budget)
    results: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "docs_dir": str(args.docs_dir),
        "concurrency": len(sessions),
        "budget_seconds": args.budget,
        "request_timeout": args.timeout,
        "phases": {},
    }
    print(f"budget {args.budget:.0f}s wall clock, request timeout {args.timeout:.0f}s")

    try:
        health, _ = sessions[0].get("/health")
        results["max_concurrency"] = health.get("max_concurrency")
        results["device"] = health.get("device")
        results["dtype"] = health.get("dtype")
        results["rerank_enabled"] = health.get("rerank_enabled")
        results["rerank_model"] = health.get("rerank_model")
        print(
            f"api: device={results['device']} dtype={results['dtype']} "
            f"max_concurrency={results['max_concurrency']} "
            f"rerank={'on' if results['rerank_enabled'] else 'off'} "
            f"model={results['rerank_model']}"
        )

        batch: Optional[List[str]] = [random.choice(SAMPLE_SENTENCES) for _ in range(args.batch_size)]
        encode_latencies = measure_encode(sessions[0], batch, args.encode_calls, budget)
        results["phases"]["encode_batch"] = summarise(encode_latencies)
        results["phases"]["encode_batch"]["batch_size"] = args.batch_size
        results["phases"]["encode_batch"]["budget_left"] = round(budget.left(), 2)

        # `default` sends no rerank field, so the probe measures what a client that
        # asks for nothing gets; `off` is an explicit answer, so on/off is measured
        # against one deployed service, not against two builds.
        rerank = None if args.rerank == "default" else args.rerank == "on"
        results["rerank"] = args.rerank
        results["rerank_top_n"] = args.rerank_top_n

        # Every connection is warmed, so no timed phase pays connection setup: the
        # measured number is the queue. Measured on the live stack, the first request
        # on a connection that has not been used costs ~20s and the same request on
        # a warmed connection ~1s, so warming is the price of admission and is paid
        # outside the measurement.
        results["warm_up_seconds"] = round(
            warm_up(sessions, queries[0], rerank, args.rerank_top_n), 2
        ) if not budget.spent() else 0.0
        print(f"  warm-up {results['warm_up_seconds']}s, budget left {budget.left():.1f}s")

        primary = mode_name(rerank)
        results["primary_mode"] = primary
        results["phases"][f"idle_{primary}"] = report(f"idle_{primary}", summarise(
            measure_search(sessions, queries, args.requests, None, rerank,
                           args.rerank_top_n, budget)
        ), budget, primary)

        if args.rerank_compare:
            # The counterpart on the same GPU queue, so the difference in the table is
            # the cost of scoring the pool. Scoring 30 candidates costs more GPU time
            # than embedding the query, and this is the number that says whether it
            # is worth paying it on the request path.
            other = True if rerank is None else not rerank
            other_name = mode_name(other)
            warm_up(sessions, queries[0], other, args.rerank_top_n)
            results["phases"][f"idle_{other_name}"] = report(f"idle_{other_name}", summarise(
                measure_search(sessions, queries, args.requests, None, other,
                               args.rerank_top_n, budget)
            ), budget, other_name)

        if not args.skip_ingest:
            docs_dir = Path(args.docs_dir)
            if not docs_dir.is_dir():
                raise SystemExit(f"documents folder not found: {docs_dir}")

            before = points_count(sessions[0])
            files = write_bulk_files(docs_dir, args.files, args.paragraphs)
            expected = args.files * args.paragraphs

            # the indexer starts after the debounce, so the search loop keeps
            # running while it is embedding the new files. The window and the wait
            # for the indexer are both cut to the budget: this phase used to wait
            # 240s for points that may never arrive, and that is where hours went.
            window = min(args.ingest_seconds, budget.left())
            loading = measure_search(sessions, queries, args.requests, window,
                                     rerank, args.rerank_top_n, budget)
            # Points are sampled at the end of the window, not waited for: waiting for
            # the indexer to catch up (measured: 48 chunks did not appear in 20s) ends
            # the overlap the phase is about, and the numbers read as idle. Sampling at
            # the boundary says how many chunks the indexer put in *while* search ran,
            # which is the question P2.4 asks.
            added = points_count(sessions[0]) - before
            stats = report("ingest", summarise(loading), budget, primary)
            stats["chunks_added"] = added
            stats["chunks_expected"] = expected
            stats["ingest_seconds"] = window
            results["phases"]["ingest"] = stats
            # The point of the phase is that the indexer was busy while search ran.
            # `chunks_added` of 0 means the search loop finished before the indexer
            # started, so the numbers say nothing about contention and the phase has
            # to be read as idle, not as ingest.
            print(f"  ingest: indexer added {added} of {expected} chunks during the "
                  f"{window:.0f}s window"
                  + ("  << no overlap, phase measures idle" if added == 0 else ""))

            results["phases"]["settled"] = report("settled", summarise(
                measure_search(sessions, queries, args.requests, None, rerank,
                               args.rerank_top_n, budget)
            ), budget, primary)
            if args.rerank_compare:
                other_name = mode_name(True if rerank is None else not rerank)
                results["phases"][f"settled_{other_name}"] = report(
                    f"settled_{other_name}", summarise(
                        measure_search(sessions, queries, args.requests, None,
                                       True if rerank is None else not rerank,
                                       args.rerank_top_n, budget)
                    ), budget, other_name
                )

            if args.keep:
                print(f"keeping {len(files)} probe files in {docs_dir}")
            else:
                remove_bulk_files(files)
                print(f"probe files removed ({len(files)}); the indexer deletes their vectors")
            results["points_before"] = before
        results["budget_exceeded"] = budget.spent()
        if budget.spent():
            print(f"budget of {args.budget:.0f}s used up: the phases after it were skipped")
        return results
    finally:
        for session in sessions:
            session.close()


def print_table(results: Dict[str, Any]) -> None:
    header = f"{'phase':>12} {'count':>6} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8}"
    print()
    print(header)
    print("-" * len(header))
    for name, stats in results["phases"].items():
        row = f"{name:>12} {stats['count']:>6} {stats['p50']:>8.3f} {stats['p95']:>8.3f} " \
              f"{stats['p99']:>8.3f} {stats['mean']:>8.3f}"
        if "chunks_added" in stats:
            row += f"  chunks_added={stats['chunks_added']}"
        if "ingest_seconds" in stats:
            row += f"  window={stats['ingest_seconds']:.0f}s"
        if "batch_size" in stats:
            row += f"  batch_size={stats['batch_size']}"
        print(row)
    print()
    phases = results["phases"]
    primary = results.get("primary_mode")
    idle, ingest = phases.get(f"idle_{primary}"), phases.get("ingest")
    if idle and ingest:
        p50 = idle["p50"] or 1e-9
        p95 = idle["p95"] or 1e-9
        print(
            f"p50 under ingest: x{ingest['p50'] / p50:.2f}   "
            f"p95 under ingest: x{ingest['p95'] / p95:.2f}   (mode {primary})"
        )

    # The reranker's cost, measured on the same GPU queue: the pair is the phase that
    # ran `fused` and the phase that ran `rerank`, found by mode, so the multiplier is
    # the price of scoring the pool and not an artefact of which side `--rerank` asked
    # for. The gate P1.1 sets is that ranking improves by more than this.
    idle_by_mode = {stats.get("mode"): stats for name, stats in phases.items()
                    if name.startswith("idle_")}
    fused, reranked = idle_by_mode.get("fused"), idle_by_mode.get("rerank")
    if fused and reranked:
        base = fused["p50"] or 1e-9
        base95 = fused["p95"] or 1e-9
        print(
            f"rerank cost vs fused: p50 x{reranked['p50'] / base:.2f}   "
            f"p95 x{reranked['p95'] / base95:.2f}  "
            f"(pool {results.get('rerank_top_n')}, model {results.get('rerank_model')})"
        )
    elif reranked:
        print("rerank measured with no fused counterpart: add --rerank-compare to get "
              "the price of scoring the pool")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR),
                        help="host path of the watched documents folder")
    # Defaults are the size of one minute: the probe measures the situation it is
    # about (search staying responsive while the indexer is busy), not a full corpus,
    # and the whole run is capped by `--budget`. Measured on the live stack: a rerank
    # request costs ~1s when the connection is warm, ~20s on a cold one, and the
    # indexer takes seconds for a handful of chunks. Six documents at 250 requests a
    # second is not what the task is about.
    parser.add_argument("--files", type=int, default=12, help="synthetic documents to write")
    parser.add_argument("--paragraphs", type=int, default=4, help="sections per document")
    parser.add_argument("--batch-size", type=int, default=32, help="texts per /embed call")
    parser.add_argument("--encode-calls", type=int, default=2)
    parser.add_argument("--requests", type=int, default=4, help="search requests per phase")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="parallel search connections. 1 reads the serialized GPU queue; "
                             "more add queue waiting, and pay the cold-connection cost each")
    parser.add_argument("--queries", default="", help="comma-separated search queries")
    parser.add_argument("--ingest-seconds", type=float, default=20.0,
                        help="how long the search loop runs while the indexer is adding chunks")
    parser.add_argument("--timeout", type=float, default=15.0, help="per-request timeout")
    parser.add_argument("--budget", type=float, default=60.0,
                        help="wall-clock limit for the whole run: phases stop when it is used up")
    parser.add_argument("--skip-ingest", action="store_true", help="measure idle + encode only")
    parser.add_argument("--rerank", choices=("on", "off", "default"), default="default",
                        help="cross-encoder reranking on the search path: `on` asks for it, "
                             "`off` forces it off, `default` sends no field (RERANK_ENABLED decides)")
    parser.add_argument("--rerank-top-n", type=int, default=None,
                        help=f"candidates the cross-encoder scores per request (1..{ev.MAX_RERANK_TOP_N})")
    parser.add_argument("--rerank-compare", action="store_true",
                        help="run the counterpart of --rerank in extra phases, so the reranker's "
                             "latency cost is measured on one deployment (idle_rerank / settled_rerank)")
    parser.add_argument("--keep", action="store_true",
                        help="keep the probe files in the documents folder")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    if args.rerank_top_n is not None and not 1 <= args.rerank_top_n <= ev.MAX_RERANK_TOP_N:
        parser.error(f"--rerank-top-n must be in 1..{ev.MAX_RERANK_TOP_N}, got {args.rerank_top_n}")
    if args.rerank_top_n is not None and args.rerank == "off":
        parser.error("--rerank-top-n with --rerank off asks the model to score nothing")

    try:
        results = run(args)
    except ev.HttpError as exc:
        print(f"HTTP {exc.status} from {args.api_url}: {exc.body}")
        return 1
    except (ConnectionError, OSError) as exc:
        print(f"Cannot reach {args.api_url}: {exc}")
        return 1
    except SystemExit as exc:
        print(str(exc))
        return 1

    print_table(results)

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {out}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(130)
