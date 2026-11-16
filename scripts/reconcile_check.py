#!/usr/bin/env python3
"""Index/filesystem reconciliation check (stdlib only, no dependencies).

The indexer deletes vectors of files that disappear, and removes stale vectors on
startup (`VectorDB.reconcile_files`). This script verifies the result from outside:
the files under the watched folder must match the `payload.file` values in Qdrant,
and a marker query must find (or not find) a given file in the top hits.

Usage:
    python scripts/reconcile_check.py audit
    python scripts/reconcile_check.py count --file NAME
    python scripts/reconcile_check.py search --query "text" --expect NAME --absent NAME

The container mount matters: `./data` is mounted as `/data`, so a host file
`data/documents/report.md` is indexed as `/data/documents/report.md`. `--prefix`
is the container path of `--documents`, and it is used to build index paths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
DEFAULT_API_URL = os.environ.get("API_URL", "http://localhost:8000")
DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "documents")
DEFAULT_DOCUMENTS = os.environ.get("DOCUMENTS_DIR", os.path.join("data", "documents"))
DEFAULT_PREFIX = "/data/documents"


def post(url: str, body: dict):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read())


def points_count(qdrant_url: str, collection: str, file_path: str | None = None) -> int:
    body: dict = {"exact": True}
    if file_path is not None:
        body["filter"] = {"must": [{"key": "file", "match": {"value": file_path}}]}
    return post(f"{qdrant_url}/collections/{collection}/points/count", body)["result"]["count"]


def indexed_files(qdrant_url: str, collection: str) -> set[str]:
    """Distinct `payload.file` values, scrolled without vectors."""
    found: set[str] = set()
    offset = None
    while True:
        body = {
            "limit": 100,
            "with_payload": ["file"],
            "with_vector": False,
            "offset": offset,
        }
        result = post(f"{qdrant_url}/collections/{collection}/points/scroll", body)["result"]
        for point in result.get("points", []):
            file_name = (point.get("payload") or {}).get("file")
            if file_name:
                found.add(file_name)
        offset = result.get("next_page_offset")
        if offset is None:
            return found


def disk_files(documents: str, prefix: str) -> set[str]:
    root = os.path.abspath(documents)
    prefix = prefix.rstrip("/")
    paths = set()
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            full = os.path.abspath(os.path.join(dirpath, name))
            rel = os.path.relpath(full, root).replace("\\", "/")
            paths.add(f"{prefix}/{rel}")
    return paths


def search_hits(api_url: str, query: str, top_k: int) -> list[tuple[str, float]]:
    response = post(f"{api_url}/search", {"query": query, "top_k": top_k, "min_score": None})
    hits = response if isinstance(response, list) else (response.get("results") or [])
    out = []
    for hit in hits:
        file_name = hit.get("file") or (hit.get("payload") or {}).get("file")
        out.append((file_name, round(hit.get("score", 0), 3)))
    return out


def index_path(prefix: str, name: str) -> str:
    """The indexed path of a file name under the watched folder."""
    return f"{prefix.rstrip('/')}/{name}"


def compare_paths(on_disk: set[str], in_index: set[str]) -> tuple[list[str], list[str]]:
    """(files on disk that are not indexed, indexed paths that are gone from disk)."""
    return sorted(on_disk - in_index), sorted(in_index - on_disk)


def check_presence(hits: list[tuple[str, float]], prefix: str, name: str, want: bool) -> bool:
    """True when the indexed path of `name` appears in `hits` as required."""
    return (index_path(prefix, name) in [hit_name for hit_name, _ in hits]) == want


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["audit", "count", "search"])
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--documents", default=DEFAULT_DOCUMENTS)
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    parser.add_argument("--file", help="file name under --documents, for count and search")
    parser.add_argument("--query", help="search text")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--expect", help="file that must appear in the hits")
    parser.add_argument("--absent", help="file that must NOT appear in the hits")
    args = parser.parse_args()

    if args.mode == "count":
        if not args.file:
            parser.error("count needs --file")
        path = index_path(args.prefix, args.file)
        print(f"{path} {points_count(args.qdrant_url, args.collection, path)}")
        return 0

    if args.mode == "search":
        if not args.query:
            parser.error("search needs --query")
        hits = search_hits(args.api_url, args.query, args.top_k)
        for name, score in hits:
            print(f"{name} {score}")
        ok = True
        for flag, must, want in (("--expect", args.expect, True), ("--absent", args.absent, False)):
            if not must:
                continue
            present = check_presence(hits, args.prefix, must, want)
            print(f"{'OK' if present else 'FAIL'} {flag} {index_path(args.prefix, must)}")
            ok = ok and present
        return 0 if ok else 1

    on_disk = disk_files(args.documents, args.prefix)
    in_index = indexed_files(args.qdrant_url, args.collection)
    missing, orphaned = compare_paths(on_disk, in_index)
    print(f"files on disk: {len(on_disk)}   indexed file paths: {len(in_index)}")
    print(f"total points: {points_count(args.qdrant_url, args.collection)}")
    for path in missing:
        print(f"  NOT INDEXED  {path}")
    for path in orphaned:
        print(f"  ORPHANED     {path}")
    if not missing and not orphaned:
        print("index matches the watched folder")
        return 0
    print(f"FAILED: {len(missing)} not indexed, {len(orphaned)} orphaned")
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        print(f"HTTP error: {exc.code} {exc.read()[:500]!r}")
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Cannot reach the stack: {exc.reason}")
        sys.exit(1)
