#!/usr/bin/env python3
"""Rehearsal for a Qdrant version change (stdlib only, no dependencies).

Upgrading Qdrant is a deliberate action: the collection schema and the storage
format are written by the running server, so a version change must be checked
against the data the current stack produced. This script checks that without
touching the live volume and without stopping the stack:

    python scripts/qdrant_upgrade_check.py --image qdrant/qdrant:v1.20.1
    python scripts/qdrant_upgrade_check.py --image qdrant/qdrant:latest --snapshot backups/documents-....snapshot

Checks:
 1. `volume` - copy the live volume into a scratch volume, start the target
    image on the copy, and compare `status`, `points`, `dim`, the distinct
    `payload.file` values and a self-vector query with the live collection;
 2. `restore` - restore a snapshot into a fresh volume of the target image and
    compare the same numbers.

A pass means the target version opens the data the current version wrote and
answers vector search on it. A fail on `volume` means the storage is not
readable by that version, so the upgrade needs a snapshot restore; a fail on
`restore` means the snapshot is not readable by that version.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qdrant_snapshot as qs  # noqa: E402

COLLECTION = qs.DEFAULT_COLLECTION
LIVE_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
CONTAINER = "rag_qdrant_upgrade_check"
SCRATCH_VOLUME = "rag_qdrant_upgrade_check"


def docker(*args: str) -> str:
    """Run a docker command and return stdout; stop on a non-zero exit."""
    result = subprocess.run(["docker", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(f"docker {' '.join(args)} failed: {(result.stderr or result.stdout).strip()}")
    return result.stdout.strip()


def docker_soft(*args: str) -> None:
    """Run a docker command and ignore failure (cleanup of leftovers must not stop the check)."""
    subprocess.run(["docker", *args], capture_output=True, text=True)


def cleanup() -> None:
    """Remove the sandbox container and volumes; also clears leftovers from an interrupted run."""
    docker_soft("rm", "-f", CONTAINER)
    for volume in (f"{SCRATCH_VOLUME}_copy", f"{SCRATCH_VOLUME}_fresh"):
        docker_soft("volume", "rm", volume)


def find_live_volume(live_volume: str | None) -> str:
    """Compose names its volumes `<project>_<volume>`, so match the suffix."""
    if live_volume:
        return live_volume
    names = [line.split()[-1] for line in docker("volume", "ls").splitlines() if "qdrant" in line]
    matches = [name for name in names if name.endswith("qdrant_data")]
    if not matches:
        raise SystemExit("no qdrant_data volume found; pass --live-volume")
    return matches[0]


def http(method: str, base: str, path: str, body: dict | None = None, timeout: int = 60) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{base}{path}", data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read())
    return payload.get("result", payload)


def collection_stats(base: str, collection: str = COLLECTION) -> dict:
    """`status`, `points`, `dim` and the distinct `payload.file` values."""
    info = http("GET", base, f"/collections/{collection}")
    points = info.get("points_count")
    if points is None:
        return {"status": "missing", "points": 0, "dim": 0, "files": 0}
    dim = (info.get("config") or {}).get("params", {}).get("vectors", {}).get("size")
    files: set[str] = set()
    offset = None
    while True:
        result = http(
            "POST",
            base,
            f"/collections/{collection}/points/scroll",
            {"limit": 100, "with_payload": ["file"], "with_vector": False, "offset": offset},
        )
        for point in result.get("points", []):
            file_name = (point.get("payload") or {}).get("file")
            if file_name:
                files.add(file_name)
        offset = result.get("next_page_offset")
        if offset is None:
            break
    return {"status": info.get("status"), "points": points, "dim": dim, "files": len(files)}


def self_query(base: str, collection: str = COLLECTION) -> tuple[int, int, float] | None:
    """Embed search without an embedding model: take one stored vector, query with
    it, and check that the point itself is the top hit (score 1.0 for cosine)."""
    points = http(
        "POST",
        base,
        f"/collections/{collection}/points/scroll",
        {"limit": 1, "with_payload": ["file"], "with_vector": True},
    ).get("points", [])
    if not points:
        return None
    point = points[0]
    hits = http(
        "POST",
        base,
        f"/collections/{collection}/points/query",
        {"query": point["vector"], "limit": 1, "with_payload": ["file"]},
    ).get("points", [])
    if not hits:
        return None
    return (point["id"], hits[0]["id"], round(hits[0]["score"], 6))


def start(image: str, volume: str, port: int, fresh: bool) -> None:
    """Start the target image on a volume and wait for the REST API."""
    if fresh:
        docker("volume", "create", volume)
    docker(
        "run",
        "-d",
        "--name",
        CONTAINER,
        "-p",
        f"{port}:6333",
        "-v",
        f"{volume}:/qdrant/storage",
        image,
    )
    base = f"http://localhost:{port}"
    for _ in range(30):
        try:
            # A freshly published Docker port on Windows can take a while to accept a
            # connection, so the probe uses a short timeout and polls.
            http("GET", base, "/collections", timeout=5)
            return
        except urllib.error.URLError:
            time.sleep(2)
    raise SystemExit(f"{image} did not answer on port {port}")


def stop() -> None:
    cleanup()


def report(label: str, live: dict, target: dict, query) -> bool:
    same = (
        target["status"] == live["status"]
        and target["points"] == live["points"]
        and target["dim"] == live["dim"]
        and target["files"] == live["files"]
    )
    print(f"{label}: status={target['status']} points={target['points']} dim={target['dim']} files={target['files']}")
    print(f"baseline:      status={live['status']} points={live['points']} dim={live['dim']} files={live['files']}")
    top_id, hit_id, score = query
    query_ok = top_id == hit_id and score > 0.99
    print(f"self-vector query: point {top_id} top hit {hit_id} score {score}")
    print(f"{'PASS' if same and query_ok else 'FAIL'} {label}")
    return same and query_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="image to check, e.g. qdrant/qdrant:v1.20.1")
    parser.add_argument("--live-url", default=LIVE_URL, help=f"live Qdrant REST URL (default {LIVE_URL})")
    parser.add_argument("--live-volume", default=None, help="compose volume name (default: the one ending in qdrant_data)")
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--port", type=int, default=6335)
    parser.add_argument("--check", choices=["volume", "restore", "both"], default="volume")
    parser.add_argument("--snapshot", default=None, help="snapshot file to restore, for --check restore|both")
    args = parser.parse_args()

    cleanup()  # a run interrupted by Ctrl+C or a timeout leaves a sandbox container and volumes
    live = collection_stats(args.live_url, args.collection)
    if live["status"] == "missing":
        raise SystemExit(f"collection '{args.collection}' is missing on the live stack")
    base = f"http://localhost:{args.port}"
    results = []

    if args.check in ("volume", "both"):
        volume = find_live_volume(args.live_volume)
        scratch = f"{SCRATCH_VOLUME}_copy"
        docker("volume", "create", scratch)
        docker(
            "run",
            "--rm",
            "--entrypoint",
            "sh",
            "-v",
            f"{volume}:/src",
            "-v",
            f"{scratch}:/dst",
            args.image,
            "-c",
            "cp -a /src/. /dst/",
        )
        start(args.image, scratch, args.port, fresh=False)
        try:
            results.append(report("volume copy", live, collection_stats(base, args.collection), self_query(base, args.collection)))
        finally:
            stop()

    if args.check in ("restore", "both"):
        if not args.snapshot:
            parser.error("--check restore|both needs --snapshot")
        fresh = f"{SCRATCH_VOLUME}_fresh"
        start(args.image, fresh, args.port, fresh=True)
        qs.BASE_URL = base
        try:
            qs.cmd_restore(SimpleNamespace(collection=args.collection, file=args.snapshot, yes=True))
            results.append(report("snapshot restore", live, collection_stats(base, args.collection), self_query(base, args.collection)))
        finally:
            stop()

    return 0 if all(results) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        print(f"HTTP error: {exc.code} {exc.read()[:500]!r}")
        sys.exit(1)
