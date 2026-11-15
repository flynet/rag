#!/usr/bin/env python3
"""Smoke test for the RAG gateway (stdlib only, no dependencies).

Usage:
    python scripts/smoke_test.py
    python scripts/smoke_test.py --base-url http://localhost:8001 --api-key sk-local
    GATEWAY_URL=http://host:8001 GATEWAY_KEY=sk-local python scripts/smoke_test.py

Checks: /health, /v1/models, /v1/embeddings, /v1/chat/completions (stream + non-stream),
including the X-RAG-* response headers that report retrieved sources.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import urllib.error
import urllib.request

DEFAULT_BASE_URL = os.environ.get("GATEWAY_URL", "http://localhost:8001")
DEFAULT_API_KEY = os.environ.get("GATEWAY_KEY", "")

failures: list[str] = []


def request(
    method: str,
    url: str,
    api_key: str,
    payload: dict | None = None,
    extra_headers: dict | None = None,
):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    for name, value in (extra_headers or {}).items():
        req.add_header(name, value)
    return urllib.request.urlopen(req, timeout=120)


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  OK    {name}")
    else:
        print(f"  FAIL  {name}{': ' + detail if detail else ''}")
        failures.append(name)


def decode_sources(headers) -> list:
    raw = headers.get("X-RAG-Sources")
    if not raw:
        return []
    padded = raw + "=" * (-len(raw) % 4)
    try:
        return json.loads(base64.urlsafe_b64decode(padded))
    except Exception:
        return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--model", default=None, help="model id (autodetected from /v1/models)")
    parser.add_argument("--prompt", default="What is in the knowledge base?")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    key = args.api_key

    print(f"Gateway: {base}\n")

    print("[1/5] GET /health")
    with request("GET", f"{base}/health", key) as response:
        payload = json.loads(response.read())
    check("health returns 200", response.status == 200)
    check("health status is ok|degraded", payload.get("status") in {"ok", "degraded"}, str(payload))
    print(f"        status={payload.get('status')} auth_required={payload.get('auth_required')}")
    dependencies = payload.get("dependencies") or {}
    check("health reports upstream", "upstream" in dependencies, str(dependencies))

    print("[2/5] GET /v1/models")
    with request("GET", f"{base}/v1/models", key) as response:
        models = json.loads(response.read()).get("data", [])
    check("models listed", isinstance(models, list))
    model = args.model or (models[0]["id"] if models else None)
    check(
        "at least one model available",
        bool(model),
        "check UPSTREAM_BASE_URL: the gateway lists models from your OpenAI-compatible server",
    )
    print(f"        models={[m['id'] for m in models]}")

    print("[3/5] POST /v1/embeddings")
    with request("POST", f"{base}/v1/embeddings", key, {"input": ["hello world"]}) as response:
        embeddings = json.loads(response.read())
    vector = embeddings["data"][0]["embedding"] if embeddings.get("data") else []
    check("one embedding returned", len(embeddings.get("data", [])) == 1)
    check("embedding is a non-empty float list", bool(vector) and isinstance(vector[0], float))
    print(f"        dim={len(vector)}")

    if not model:
        print("\nSkipping chat checks: no model available from the upstream server.")
        return 1 if failures else 0

    print("[4/5] POST /v1/chat/completions (non-stream)")
    with request(
        "POST",
        f"{base}/v1/chat/completions",
        key,
        {"model": model, "messages": [{"role": "user", "content": args.prompt}]},
    ) as response:
        completion = json.loads(response.read())
    message = (completion.get("choices") or [{}])[0].get("message", {})
    check("assistant message returned", bool(message.get("content")))
    sources = decode_sources(response.headers)
    print(f"        X-RAG-Hits={response.headers.get('X-RAG-Hits')} sources={sources}")
    check("RAG sources header present", response.headers.get("X-RAG-Hits") is not None)

    print("[5/5] POST /v1/chat/completions (stream, X-RAG: off)")
    chunks = 0
    done = False
    with request(
        "POST",
        f"{base}/v1/chat/completions",
        key,
        {"model": model, "messages": [{"role": "user", "content": args.prompt}], "stream": True},
        extra_headers={"X-RAG": "off"},
    ) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            chunks += 1
    check("stream produced SSE chunks", chunks > 0, f"chunks={chunks}")
    check("stream terminated with [DONE]", done)
    check(
        "X-RAG: off disables retrieval",
        response.headers.get("X-RAG-Hits") == "0",
        f"got X-RAG-Hits={response.headers.get('X-RAG-Hits')}",
    )

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s) - {', '.join(failures)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except urllib.error.HTTPError as exc:
        print(f"HTTP error: {exc.code} {exc.read()[:500]!r}")
        sys.exit(1)
    except urllib.error.URLError as exc:
        print(f"Cannot reach gateway: {exc.reason}")
        sys.exit(1)
