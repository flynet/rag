"""Unit tests for the stub upstream used by gateway-mode measurement.

Run from the repository root:

    python -m unittest discover -s tests -p "test_*.py"

Gateway mode measures the gateway's retrieval path, and the harness needs a gateway
that answers. The stub is the test double for that: it must speak enough of the
OpenAI schema for the gateway to forward, and the routes the gateway builds
(`/models`, `/chat/completions`) have to match the routes the stub serves.
"""

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import upstream_stub as stub  # noqa: E402
import gateway.main as gw  # noqa: E402


class _LiveStub:
    """A stub server on 127.0.0.1, with the request helpers the tests need."""

    def __init__(self):
        self.server = stub.ThreadingHTTPServer(("127.0.0.1", 0), stub.Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def get(self, path):
        with urllib.request.urlopen(f"{self.base}{path}", timeout=10) as response:
            return json.loads(response.read())

    def post(self, path, body, raw: bool = False):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        request = urllib.request.Request(
            f"{self.base}{path}", data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            text = response.read().decode()
            return text if raw else json.loads(text)


class StubAnswerTests(unittest.TestCase):
    def setUp(self):
        self.stub = _LiveStub()

    def tearDown(self):
        self.stub.stop()

    def test_models_lists_one_model(self):
        # the harness resolves `--model` from this list, so it must be OpenAI-shaped
        payload = self.stub.get("/v1/models")
        self.assertEqual(payload["object"], "list")
        self.assertTrue(payload["data"][0]["id"])

    def test_a_completion_is_answered(self):
        payload = self.stub.post("/v1/chat/completions", {
            "model": "some-model", "messages": [{"role": "user", "content": "hi"}],
            "stream": False,
        })
        self.assertEqual(payload["object"], "chat.completion")
        # the gateway forwards the body verbatim, so the model name travels back
        self.assertEqual(payload["model"], "some-model")
        self.assertEqual(payload["choices"][0]["message"]["role"], "assistant")
        self.assertTrue(payload["choices"][0]["message"]["content"])
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")

    def test_streaming_is_answered_as_event_streams(self):
        text = self.stub.post("/v1/chat/completions", {
            "model": "stub", "messages": [{"role": "user", "content": "hi"}], "stream": True,
        }, raw=True)
        frames = [line[len("data: "):] for line in text.split("\n") if line.startswith("data: ")]
        self.assertTrue(frames)
        self.assertEqual(frames[-1], "[DONE]")
        self.assertTrue(json.loads(frames[0])["choices"][0]["delta"]["content"])

    def test_a_body_that_is_not_json_is_reported(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.stub.post("/v1/chat/completions", b"{not json")
        self.assertEqual(caught.exception.code, 400)

    def test_answers_are_the_same_text_with_different_ids(self):
        # a stub that answers instantly isolates retrieval cost from answer cost,
        # so the content is fixed and the ids are not
        first = self.stub.post("/v1/chat/completions", {"model": "stub", "messages": []})
        second = self.stub.post("/v1/chat/completions", {"model": "stub", "messages": []})
        self.assertEqual(first["choices"][0]["message"]["content"],
                         second["choices"][0]["message"]["content"])
        self.assertNotEqual(first["id"], second["id"])


class GatewayRoutesTests(unittest.TestCase):
    """The routes the gateway builds have to be the routes the stub serves."""

    def setUp(self):
        self.stub = _LiveStub()

    def tearDown(self):
        self.stub.stop()

    def test_the_gateway_does_not_double_the_v1_prefix(self):
        with patch.object(gw, "UPSTREAM_BASE_URL", f"{self.stub.base}/v1"):
            self.assertTrue(gw._upstream_url("/models").endswith("/v1/models"))
            self.assertTrue(gw._upstream_url("/chat/completions").endswith("/v1/chat/completions"))

    def test_the_gateway_does_not_double_the_v1_prefix_when_base_has_no_v1(self):
        with patch.object(gw, "UPSTREAM_BASE_URL", self.stub.base):
            self.assertTrue(gw._upstream_url("/models").endswith("/v1/models"))

    def test_the_stub_answers_both_gateway_routes(self):
        # the URL the gateway builds has to be the URL the stub serves, so the
        # request is made on the stub's own port with the path the gateway produced
        with patch.object(gw, "UPSTREAM_BASE_URL", f"{self.stub.base}/v1"):
            models_url = gw._upstream_url("/models")
            completion_url = gw._upstream_url("/chat/completions")
            self.assertTrue(self.stub.get(models_url[len(self.stub.base):])["data"])
            payload = self.stub.post(completion_url[len(self.stub.base):],
                                     {"model": "stub", "messages": [], "stream": False})
            self.assertTrue(payload["choices"])

    def test_a_model_server_that_is_not_configured_is_reported(self):
        # the stub is only there to answer; the gate stays: no upstream, 503
        with patch.object(gw, "UPSTREAM_BASE_URL", ""):
            from starlette.exceptions import HTTPException as StarletteHTTPException
            with self.assertRaises(StarletteHTTPException):
                gw._upstream_url("/models")


class ContractTests(unittest.TestCase):
    def test_the_stub_is_documented_for_gateway_runs(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("scripts/upstream_stub.py", readme)
        eval_readme = (ROOT / "eval" / "README.md").read_text(encoding="utf-8")
        self.assertIn("host.docker.internal", eval_readme)

    def test_the_stub_needs_no_third_party_packages(self):
        source = (ROOT / "scripts" / "upstream_stub.py").read_text(encoding="utf-8")
        self.assertIn("from http.server import", source)
        self.assertNotIn("import httpx", source)


if __name__ == "__main__":
    unittest.main()
