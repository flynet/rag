#!/usr/bin/env python3
"""Qdrant snapshot tool (stdlib only, no dependencies).

Snapshots let you restore the index without re-embedding the corpus: the
re-index cost of a full `data/documents` scan (GPU time for every chunk) is
replaced by uploading one file.

Usage (run from the repository root):
    python scripts/qdrant_snapshot.py list
    python scripts/qdrant_snapshot.py create
    python scripts/qdrant_snapshot.py create --keep 5
    python scripts/qdrant_snapshot.py prune --keep 5
    python scripts/qdrant_snapshot.py download --snapshot NAME
    python scripts/qdrant_snapshot.py delete --snapshot NAME
    python scripts/qdrant_snapshot.py status
    python scripts/qdrant_snapshot.py restore --file backups/documents-....snapshot --yes

Defaults: `QDRANT_URL` (http://localhost:6333) and `COLLECTION_NAME`
(`documents`). Snapshots stored on the Qdrant node live under
`qdrant_data/snapshots/<collection>/`, downloaded copies go to `backups/`.

Notes:
- `restore` is destructive: it overwrites the collection data on the node. It
  requires an explicit `--yes`.
- `create` also downloads the snapshot to `backups/`, so the copy survives a
  Qdrant container rebuild.
- `--keep N` prunes the oldest node snapshots after a successful create.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333").rstrip("/")
DEFAULT_COLLECTION = os.environ.get("COLLECTION_NAME", "documents")
DEFAULT_OUT = "backups"

# Qdrant REST base, overridden by --url in main()
BASE_URL = DEFAULT_URL

# Snapshot creation is synchronous (wait=true) and a small collection is fast;
# a large corpus can take minutes, so the read timeout is generous.
TIMEOUT = 300


def request(
    method: str,
    path: str,
    params: dict | None = None,
    body: bytes | None = None,
    content_type: str = "application/json",
    timeout: int = TIMEOUT,
):
    """Raw HTTP call against the Qdrant REST API; returns the response bytes."""
    url = f"{BASE_URL}{path}"
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, data=body, method=method)
    if body is not None:
        req.add_header("Content-Type", content_type)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


def api_json(method: str, path: str, params: dict | None = None, body: bytes | None = None) -> dict:
    raw = request(method, path, params, body)
    if not raw:
        return {}
    payload = json.loads(raw)
    if "result" in payload:
        return payload["result"]
    return payload


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def multipart(field: str, filename: str, data: bytes, boundary: str) -> bytes:
    """Minimal multipart/form-data body for the Qdrant snapshot upload endpoint."""
    return (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()


def format_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def snapshot_sort_key(snapshot: dict) -> tuple:
    """Sort snapshots oldest first; creation_time is ISO-8601, name is a fallback."""
    return (str(snapshot.get("creation_time") or ""), str(snapshot.get("name") or ""))


def pick_deletions(snapshots: list[dict], keep: int) -> list[dict]:
    """Snapshots to remove so that at most `keep` newest ones stay on the node."""
    if keep <= 0:
        return []
    ordered = sorted(snapshots, key=snapshot_sort_key)
    return ordered[: max(0, len(ordered) - keep)]


def collection_info(collection: str) -> dict | None:
    """Collection metadata, or None when the collection does not exist."""
    try:
        return api_json("GET", f"/collections/{collection}")
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None  # the collection does not exist
        raise  # unreachable/auth/server errors must fail the command, not print "missing"


def print_collection(collection: str, info: dict | None) -> None:
    if info is None:
        print(f"collection '{collection}': missing")
        return
    vectors = ((info.get("config") or {}).get("params") or {}).get("vectors")
    size = None
    if isinstance(vectors, dict):
        size = vectors.get("size")
    print(
        f"collection '{collection}': status={info.get('status')} "
        f"points={info.get('points_count')} dim={size}"
    )


def cmd_status(args: argparse.Namespace) -> int:
    print_collection(args.collection, collection_info(args.collection))
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    snapshots = api_json("GET", f"/collections/{args.collection}/snapshots")
    if not isinstance(snapshots, list):
        print(f"Unexpected snapshot list: {snapshots}", file=sys.stderr)
        return 1
    for snapshot in sorted(snapshots, key=snapshot_sort_key):
        print(
            f"{snapshot.get('name')}  {format_size(int(snapshot.get('size', 0)))}  "
            f"{snapshot.get('creation_time')}"
        )
    if not snapshots:
        print(f"no snapshots for collection '{args.collection}'")
    return 0


def save_snapshot(collection: str, name: str, data: bytes, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, name)
    with open(target, "wb") as file:
        file.write(data)
    return target


def cmd_create(args: argparse.Namespace) -> int:
    created = api_json("POST", f"/collections/{args.collection}/snapshots", {"wait": "true"})
    name = created.get("name")
    if not name:
        print(f"Snapshot creation returned no name: {created}", file=sys.stderr)
        return 1
    print(f"created on node: {name} ({format_size(int(created.get('size', 0)))})")

    data = request("GET", f"/collections/{args.collection}/snapshots/{name}")
    if not data:
        print("Downloaded snapshot is empty - the copy is not usable", file=sys.stderr)
        return 1
    target = save_snapshot(args.collection, name, data, args.out)
    print(f"downloaded: {target} ({format_size(len(data))}, sha256={sha256(data)[:16]}...)")

    if args.keep:
        snapshots = api_json("GET", f"/collections/{args.collection}/snapshots")
        victims = pick_deletions(snapshots if isinstance(snapshots, list) else [], args.keep)
        for snapshot in victims:
            api_json("DELETE", f"/collections/{args.collection}/snapshots/{snapshot['name']}")
            print(f"pruned on node: {snapshot['name']}")
        print(f"node snapshots kept: {max(0, len(snapshots) - len(victims))}")
    return 0


def cmd_download(args: argparse.Namespace) -> int:
    data = request("GET", f"/collections/{args.collection}/snapshots/{args.snapshot}")
    if not data:
        print("Downloaded snapshot is empty", file=sys.stderr)
        return 1
    target = save_snapshot(args.collection, args.snapshot, data, args.out)
    print(f"saved: {target} ({format_size(len(data))}, sha256={sha256(data)[:16]}...)")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    api_json("DELETE", f"/collections/{args.collection}/snapshots/{args.snapshot}")
    print(f"deleted on node: {args.snapshot}")
    return 0


def cmd_prune(args: argparse.Namespace) -> int:
    snapshots = api_json("GET", f"/collections/{args.collection}/snapshots")
    if not isinstance(snapshots, list):
        print(f"Unexpected snapshot list: {snapshots}", file=sys.stderr)
        return 1
    victims = pick_deletions(snapshots, args.keep)
    for snapshot in victims:
        api_json("DELETE", f"/collections/{args.collection}/snapshots/{snapshot['name']}")
        print(f"pruned on node: {snapshot['name']}")
    print(f"node snapshots kept: {len(snapshots) - len(victims)} (keep={args.keep})")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    if not args.yes:
        print(
            "restore overwrites the collection data on the node; "
            "rerun with --yes to confirm",
            file=sys.stderr,
        )
        return 2

    if not os.path.isfile(args.file):
        print(f"snapshot file not found: {args.file}", file=sys.stderr)
        return 1
    with open(args.file, "rb") as file:
        data = file.read()
    checksum = sha256(data)

    print(f"local file: {args.file} ({format_size(len(data))}, sha256={checksum[:16]}...)")
    existing = collection_info(args.collection)
    print_collection(args.collection, existing)
    if existing is not None and existing.get("points_count"):
        print(f"WARNING: points of '{args.collection}' will be replaced by the snapshot")

    boundary = "qdrant-snapshot-upload"
    request(
        "POST",
        f"/collections/{args.collection}/snapshots/upload",
        {"wait": "true", "checksum": checksum},
        multipart("snapshot", os.path.basename(args.file), data, boundary),
        f"multipart/form-data; boundary={boundary}",
    )

    print_collection(args.collection, collection_info(args.collection))
    print(
        "Verify search before serving traffic: "
        "python scripts/smoke_test.py  (and check api /health + embedder logs)"
    )
    return 0


def main() -> int:
    global BASE_URL
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Qdrant REST URL (default {DEFAULT_URL})")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help=f"collection name (default {DEFAULT_COLLECTION})")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"local snapshot directory (default {DEFAULT_OUT})")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="list snapshots stored on the Qdrant node")
    create = sub.add_parser("create", help="create a snapshot on the node and download it")
    create.add_argument("--keep", type=int, default=5, help="keep this many newest snapshots on the node (0 = no pruning)")
    download = sub.add_parser("download", help="download a node snapshot into the local directory")
    download.add_argument("--snapshot", required=True)
    delete = sub.add_parser("delete", help="delete a snapshot on the node")
    delete.add_argument("--snapshot", required=True)
    prune = sub.add_parser("prune", help="delete the oldest node snapshots")
    prune.add_argument("--keep", type=int, default=5)
    sub.add_parser("status", help="show collection status, points and dimension")
    restore = sub.add_parser("restore", help="upload a local snapshot and replace the collection data")
    restore.add_argument("--file", required=True)
    restore.add_argument("--yes", action="store_true", help="confirm the destructive overwrite")

    args = parser.parse_args()
    BASE_URL = args.url.rstrip("/")

    handlers = {
        "list": cmd_list,
        "create": cmd_create,
        "download": cmd_download,
        "delete": cmd_delete,
        "prune": cmd_prune,
        "status": cmd_status,
        "restore": cmd_restore,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:400].decode(errors="replace")
        print(f"Qdrant HTTP error {exc.code}: {detail}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Cannot reach Qdrant at {BASE_URL}: {exc.reason}", file=sys.stderr)
        sys.exit(1)
