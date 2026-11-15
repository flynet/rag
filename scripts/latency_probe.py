#!/usr/bin/env python3
"""Latency probe for the embedding API under indexing load.

Bulk indexing and user queries share one GPU model, so the question P2.5 asks is
how much a user has to wait while the indexer is working. The probe answers it by
measuring `POST /search` latency in three situations:

  idle        - the stack is not indexing anything;
  ingest      - synthetic documents are written to the watched folder and the
                indexer is busy embedding them;
  settled     - indexing finished, the same requests again.

`POST /embed` is measured as well: that is the queue serialized by
`EMBEDDING_MAX_CONCURRENCY`.

p50 / p95 / p99 are reported for each phase, together with the number of chunks
the indexer added. The probe is stdlib-only and writes nothing outside the
documents folder it is pointed at.

Usage:
    python scripts/latency_probe.py
    python scripts/latency_probe.py --files 12 --concurrency 4 --requests 40
    python scripts/latency_probe.py --skip-ingest --json-out perf.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
# HTTP
# --------------------------------------------------------------------------- #


class Timer:
    def __init__(self):
        self.start = time.perf_counter()
        self.elapsed: Optional[float] = None

    def stop(self) -> float:
        self.elapsed = time.perf_counter() - self.start
        return self.elapsed


def percentile(values: List[float], point: float) -> float:
    """Nearest-rank percentile on a list of latencies (seconds)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(point * (len(ordered) - 1)))))
    return ordered[index]


def summarise(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p95": 0.0, "mean": 0.0}
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


async def measure_search(api_url: str, queries: List[str], requests: int,
                         concurrency: int) -> List[float]:
    """Latency of `POST /search` for `requests` calls, `concurrency` at a time."""
    url = f"{api_url.rstrip('/')}/search"
    latencies: List[float] = []
    queue: asyncio.Queue = asyncio.Queue()
    for index in range(requests):
        queue.put_nowait(queries[index % len(queries)])

    workers = []
    for _ in range(max(1, concurrency)):
        workers.append(asyncio.create_task(_search_worker(url, queue, latencies)))
    await asyncio.gather(*workers)
    return latencies


async def _search_worker(url: str, queue: asyncio.Queue, latencies: List[float]) -> None:
    loop = asyncio.get_running_loop()
    while True:
        try:
            query = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        body = json.dumps({"query": query, "top_k": 5, "min_score": None}).encode("utf-8")
        timer = Timer()
        try:
            request = urllib.request.Request(
                url, data=body, method="POST", headers={"Content-Type": "application/json"}
            )
            await loop.run_in_executor(None, lambda: urllib.request.urlopen(request, timeout=60).read())
            latencies.append(timer.stop())
        except Exception as exc:  # noqa: BLE001 - the probe reports, not raises
            timer.stop()
            print(f"  search request failed: {exc}", file=sys.stderr)


async def measure_encode(api_url: str, texts: List[str], calls: int) -> List[float]:
    """Latency of `POST /embed` for one batch, `calls` times in sequence."""
    url = f"{api_url.rstrip('/')}/embed"
    body = json.dumps({"texts": texts}).encode("utf-8")
    loop = asyncio.get_running_loop()
    latencies: List[float] = []
    for _ in range(calls):
        timer = Timer()
        request = urllib.request.Request(
            url, data=body, method="POST", headers={"Content-Type": "application/json"}
        )
        await loop.run_in_executor(None, lambda: urllib.request.urlopen(request, timeout=120).read())
        latencies.append(timer.stop())
    return latencies


async def points_count(api_url: str) -> int:
    url = f"{api_url.rstrip('/')}/collection/info"
    loop = asyncio.get_running_loop()
    payload = json.loads(await loop.run_in_executor(None, _get, url))
    return int(payload.get("points_count") or 0)


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read()


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


async def wait_for_indexing(api_url: str, expected_delta: int, timeout: float) -> int:
    """Poll `points_count` until `expected_delta` points appear, or time out."""
    start = await points_count(api_url)
    target = start + expected_delta
    deadline = time.monotonic() + timeout
    last = start
    while time.monotonic() < deadline:
        current = await points_count(api_url)
        if current >= target:
            return current - start
        last = current
        await asyncio.sleep(1.0)
    print(f"  indexing did not reach {target} points (last seen {last}) in {timeout:.0f}s",
          file=sys.stderr)
    return last - start


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


async def run(args: argparse.Namespace) -> Dict[str, Any]:
    queries = args.queries.split(",") if args.queries else DEFAULT_QUERIES
    results: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "api_url": args.api_url,
        "docs_dir": str(args.docs_dir),
        "max_concurrency": None,
        "phases": {},
    }

    health = json.loads(_get(f"{args.api_url.rstrip('/')}/health"))
    results["max_concurrency"] = health.get("max_concurrency")
    results["device"] = health.get("device")
    results["dtype"] = health.get("dtype")

    print(f"api: device={results['device']} dtype={results['dtype']} "
          f"max_concurrency={results['max_concurrency']}")

    batch = [random.choice(SAMPLE_SENTENCES) for _ in range(args.batch_size)]
    encode_latencies = await measure_encode(args.api_url, batch, args.encode_calls)
    results["phases"]["encode_batch"] = summarise(encode_latencies)
    results["phases"]["encode_batch"]["batch_size"] = args.batch_size

    idle = await measure_search(args.api_url, queries, args.requests, args.concurrency)
    results["phases"]["idle"] = summarise(idle)

    if not args.skip_ingest:
        docs_dir = Path(args.docs_dir)
        if not docs_dir.is_dir():
            raise SystemExit(f"documents folder not found: {docs_dir}")

        before = await points_count(args.api_url)
        files = write_bulk_files(docs_dir, args.files, args.paragraphs)
        expected = args.files * args.paragraphs

        # the indexer starts on the debounce, so the search loop runs while it is
        # embedding the new files
        loading = await measure_search(args.api_url, queries, args.requests, args.concurrency)
        added = await wait_for_indexing(args.api_url, expected, args.ingest_timeout)
        results["phases"]["ingest"] = summarise(loading)
        results["phases"]["ingest"]["chunks_added"] = added

        settled = await measure_search(args.api_url, queries, args.requests, args.concurrency)
        results["phases"]["settled"] = summarise(settled)

        if args.keep:
            print(f"keeping {len(files)} probe files in {docs_dir}")
        else:
            remove_bulk_files(files)
            print(f"probe files removed ({len(files)}); the indexer deletes their vectors")

    return results


def print_table(results: Dict[str, Any]) -> None:
    header = f"{'phase':>12} {'count':>6} {'p50':>8} {'p95':>8} {'p99':>8} {'mean':>8}"
    print()
    print(header)
    print("-" * len(header))
    for name, stats in results["phases"].items():
        print(f"{name:>12} {stats['count']:>6} {stats['p50']:>8.3f} {stats['p95']:>8.3f} "
              f"{stats['p99']:>8.3f} {stats['mean']:>8.3f}"
              + (f"  chunks_added={stats['chunks_added']}" if "chunks_added" in stats else "")
              + (f"  batch_size={stats['batch_size']}" if "batch_size" in stats else ""))
    print()
    idle, ingest = results["phases"].get("idle"), results["phases"].get("ingest")
    if idle and ingest:
        p50 = idle["p50"] or 1e-9
        p95 = idle["p95"] or 1e-9
        print(f"p50 under ingest: x{ingest['p50'] / p50:.2f}   p95 under ingest: x{ingest['p95'] / p95:.2f}")


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
    parser.add_argument("--concurrency", type=int, default=3, help="parallel search requests")
    parser.add_argument("--queries", default="", help="comma-separated search queries")
    parser.add_argument("--ingest-timeout", type=float, default=240.0)
    parser.add_argument("--skip-ingest", action="store_true", help="measure idle + encode only")
    parser.add_argument("--keep", action="store_true", help="keep the probe files in the documents folder")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    try:
        results = asyncio.run(run(args))
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code} from {args.api_url}: {exc.read()[:400].decode('utf-8', 'replace')}")
        return 1
    except urllib.error.URLError as exc:
        print(f"Cannot reach {args.api_url}: {exc.reason}")
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
