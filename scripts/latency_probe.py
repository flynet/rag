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
the indexer added and the latency multiplier under ingest. Stdlib only. The probe
writes synthetic files only into the documents folder it is pointed at, and
removes them at the end.

Usage:
    python scripts/latency_probe.py
    python scripts/latency_probe.py --files 12 --concurrency 4 --requests 40
    python scripts/latency_probe.py --skip-ingest --json-out perf.json
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


# --------------------------------------------------------------------------- #
# Probes
# --------------------------------------------------------------------------- #


def measure_search(sessions: List[ev.Session], queries: List[str], requests: int,
                   duration: Optional[float] = None) -> List[float]:
    """Latency of `POST /search`, one connection per parallel worker.

    With `duration` the loop runs for that many seconds instead of a fixed
    number of requests, which is how the ingest phase is kept overlapping the
    indexer's work. Round-robin over sessions keeps every connection busy.
    """
    latencies: List[float] = []
    index = 0
    deadline = time.monotonic() + duration if duration is not None else None
    while True:
        if deadline is not None:
            if time.monotonic() >= deadline:
                break
        elif index >= requests:
            break
        session = sessions[index % len(sessions)]
        query = queries[index % len(queries)]
        start = time.perf_counter()
        try:
            session.post("/search", {"query": query, "top_k": 5, "min_score": None})
            latencies.append(time.perf_counter() - start)
        except Exception as exc:  # noqa: BLE001 - the probe reports, not raises
            print(f"  search request failed: {exc}", file=sys.stderr)
        index += 1
    return latencies


def measure_encode(session: ev.Session, texts: List[str], calls: int) -> List[float]:
    """Latency of `POST /embed` for one batch, `calls` times in sequence."""
    latencies: List[float] = []
    for _ in range(calls):
        start = time.perf_counter()
        try:
            session.post("/embed", {"texts": texts})
            latencies.append(time.perf_counter() - start)
        except Exception as exc:  # noqa: BLE001
            print(f"  embed request failed: {exc}", file=sys.stderr)
    return latencies


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


def wait_for_indexing(session: ev.Session, expected_delta: int, timeout: float) -> int:
    """Poll `points_count` until `expected_delta` points appear, or time out."""
    start = points_count(session)
    target = start + expected_delta
    deadline = time.monotonic() + timeout
    last = start
    while time.monotonic() < deadline:
        current = points_count(session)
        if current >= target:
            return current - start
        last = current
        time.sleep(1.0)
    print(
        f"  indexing did not reach {target} points (last seen {last}) in {timeout:.0f}s",
        file=sys.stderr,
    )
    return last - start


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def run(args: argparse.Namespace) -> Dict[str, Any]:
    queries = args.queries.split(",") if args.queries else DEFAULT_QUERIES
    sessions = [ev.Session(args.api_url, args.timeout) for _ in range(max(1, args.concurrency))]
    results: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "docs_dir": str(args.docs_dir),
        "concurrency": len(sessions),
        "phases": {},
    }

    try:
        health, _ = sessions[0].get("/health")
        results["max_concurrency"] = health.get("max_concurrency")
        results["device"] = health.get("device")
        results["dtype"] = health.get("dtype")
        print(
            f"api: device={results['device']} dtype={results['dtype']} "
            f"max_concurrency={results['max_concurrency']}"
        )

        batch: Optional[List[str]] = [random.choice(SAMPLE_SENTENCES) for _ in range(args.batch_size)]
        encode_latencies = measure_encode(sessions[0], batch, args.encode_calls)
        results["phases"]["encode_batch"] = summarise(encode_latencies)
        results["phases"]["encode_batch"]["batch_size"] = args.batch_size

        results["phases"]["idle"] = summarise(
            measure_search(sessions, queries, args.requests)
        )

        if not args.skip_ingest:
            docs_dir = Path(args.docs_dir)
            if not docs_dir.is_dir():
                raise SystemExit(f"documents folder not found: {docs_dir}")

            before = points_count(sessions[0])
            files = write_bulk_files(docs_dir, args.files, args.paragraphs)
            expected = args.files * args.paragraphs

            # the indexer starts after the debounce, so the search loop keeps
            # running while it is embedding the new files
            loading = measure_search(sessions, queries, args.requests, args.ingest_seconds)
            added = wait_for_indexing(sessions[0], expected, args.ingest_timeout)
            results["phases"]["ingest"] = summarise(loading)
            results["phases"]["ingest"]["chunks_added"] = added
            results["phases"]["ingest"]["ingest_seconds"] = args.ingest_seconds

            results["phases"]["settled"] = summarise(
                measure_search(sessions, queries, args.requests)
            )

            if args.keep:
                print(f"keeping {len(files)} probe files in {docs_dir}")
            else:
                remove_bulk_files(files)
                print(f"probe files removed ({len(files)}); the indexer deletes their vectors")
            results["points_before"] = before
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
    idle, ingest = results["phases"].get("idle"), results["phases"].get("ingest")
    if idle and ingest:
        p50 = idle["p50"] or 1e-9
        p95 = idle["p95"] or 1e-9
        print(
            f"p50 under ingest: x{ingest['p50'] / p50:.2f}   "
            f"p95 under ingest: x{ingest['p95'] / p95:.2f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR),
                        help="host path of the watched documents folder")
    parser.add_argument("--files", type=int, default=10, help="synthetic documents to write")
    parser.add_argument("--paragraphs", type=int, default=6, help="sections per document")
    parser.add_argument("--batch-size", type=int, default=32, help="texts per /embed call")
    parser.add_argument("--encode-calls", type=int, default=8)
    parser.add_argument("--requests", type=int, default=25, help="search requests per phase")
    parser.add_argument("--concurrency", type=int, default=3, help="parallel search connections")
    parser.add_argument("--queries", default="", help="comma-separated search queries")
    parser.add_argument("--ingest-timeout", type=float, default=240.0,
                        help="how long to wait for the indexer to catch up")
    parser.add_argument("--ingest-seconds", type=float, default=60.0,
                        help="how long the search loop runs during the ingest phase")
    parser.add_argument("--timeout", type=float, default=60.0, help="per-request timeout")
    parser.add_argument("--skip-ingest", action="store_true", help="measure idle + encode only")
    parser.add_argument("--keep", action="store_true",
                        help="keep the probe files in the documents folder")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

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
