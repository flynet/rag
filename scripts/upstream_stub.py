#!/usr/bin/env python3
"""A stand-in OpenAI-compatible upstream for gateway-mode retrieval measurement.

Gateway mode measures the gateway's own retrieval path: follow-up rewriting, the
"fresh user turn" rule, injection, and the `X-RAG-Sources` header it returns. The
answers are never evaluated, so the upstream LLM is only in the loop to make the
gateway answer. Pointing `UPSTREAM_BASE_URL` at this stub isolates retrieval latency
and lets a gateway run complete when the real LLM server is not reachable on the
network — the numbers describe retrieval, which is what is being measured.

The stub answers instantly, so gateway-mode latency measured against it is the cost
of retrieval and rewriting, not of generating an answer. A run against the real
upstream is a different number and is recorded as such.

Usage:
    python scripts/upstream_stub.py --port 5011
    # gateway env: UPSTREAM_BASE_URL=http://host.docker.internal:5011/v1
"""

from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL_ID = "stub"
ANSWER = "stub answer: the upstream server is a test double, this run measures retrieval"


def completion(body: dict) -> dict:
    model = body.get("model") or MODEL_ID
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": ANSWER},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


class Handler(BaseHTTPRequestHandler):
    # keep-alive is what the gateway's client pool relies on
    protocol_version = "HTTP/1.1"

    def _reply(self, payload: dict, status: int = 200) -> None:
        raw = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path.rstrip("/").endswith("/models"):
            self._reply({
                "object": "list",
                "data": [{"id": MODEL_ID, "object": "model", "created": 0, "owned_by": "stub"}],
            })
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._reply({"error": {"message": "request body is not JSON"}}, 400)
            return
        if not isinstance(body, dict):
            self._reply({"error": {"message": "request body is not an object"}}, 400)
            return

        payload = completion(body)
        if body.get("stream"):
            # The harness asks for `stream: false`; streaming is answered so a
            # streaming client gets the same answer as a non-streaming one.
            chunks = []
            for index, text in enumerate(payload["choices"]):
                chunk = {
                    "id": payload["id"], "object": "chat.completion.chunk",
                    "created": payload["created"], "model": payload["model"],
                    "choices": [{"index": index, "delta": {"content": text["message"]["content"]},
                                  "finish_reason": None}],
                }
                chunks.append(f"data: {json.dumps(chunk)}\n\n")
            chunks.append("data: [DONE]\n\n")
            raw = "".join(chunks).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self._reply(payload)

    def log_message(self, format, *args):  # keep the run output clean
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5011)
    parser.add_argument("--host", default="0.0.0.0",
                        help="bind address; 0.0.0.0 so the gateway container can reach it")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"stub upstream on http://{args.host}:{args.port}/v1  model={MODEL_ID}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
