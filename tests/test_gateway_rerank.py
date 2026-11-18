"""Unit tests for reranking on the gateway side (no network, no GPU, no Docker).

Run from the repository root:

    python -m unittest discover -s tests -p "test_*.py"

The gateway does not own the reranker: `api` does. What the gateway has to get
right is the request flag (`X-RAG-RERANK`, and the `RAG_RERANK` default), keeping
the pre-P1.1 payload byte-identical when nothing is asked for, sending an explicit
`rerank: false` when the client refuses reranking (so on/off is measurable on one
deployment), and passing the reranker's number through to the citation headers and
the injected context.

`min_score` is the measured dense cosine threshold from P0, so it keeps filtering
on `score`, not on `rerank_score`: a reranker that prefers a chunk the threshold
would have removed must not resurrect it.
"""

import asyncio
import base64
import json
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gateway.main as gw  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402

RELEVANT = {
    "file": "/data/documents/TODO.md",
    "chunk": 3,
    "score": 0.62,
    "text": "P1.1 Reranker",
    "title": "RAG Gateway",
    "section": "P1",
    "char_start": 400,
    "char_end": 500,
}

NOISY = {
    "file": "/data/documents/README.md",
    "chunk": 11,
    "score": 0.41,
    "text": "BM25 term overlap",
    "title": "RAG Gateway",
    "section": "Features",
    "char_start": 900,
    "char_end": 1000,
}


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _StubClient:
    """A stand-in for `httpx.AsyncClient` that records the request body."""

    def __init__(self, captured, payload):
        self.captured = captured
        self.payload = payload

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def post(self, url, json):  # noqa: A002 - the httpx parameter name
        self.captured.append((url, json))
        return _Response(self.payload)


def stub_client(payload):
    """A client factory that answers with `payload` and records what it sent."""
    captured: list = []

    def factory(**kwargs):
        return _StubClient(captured, payload)

    return factory, captured


def search(hits, rerank, min_score=0.3, top_k=8, pool=None):
    factory, captured = stub_client(hits)
    with patch.object(gw.httpx, "AsyncClient", factory):
        results = asyncio.run(gw._rag_search("the query", top_k, min_score, rerank, pool))
    return results, captured


class FlagHeaderTests(unittest.TestCase):
    def test_no_header_is_no_opinion(self):
        self.assertIsNone(gw._flag_header(None, "X-RAG-RERANK"))

    def test_on_and_off_are_accepted(self):
        for value in ("on", "ON", "true", "1", "yes"):
            self.assertTrue(gw._flag_header(value, "X-RAG-RERANK"))
        for value in ("off", "OFF", "false", "0", "no"):
            self.assertFalse(gw._flag_header(value, "X-RAG-RERANK"))

    def test_a_bad_flag_is_a_400(self):
        with self.assertRaises(StarletteHTTPException) as ctx:
            gw._flag_header("maybe", "X-RAG-RERANK")
        self.assertEqual(ctx.exception.status_code, 400)


class SettingsTests(unittest.TestCase):
    def test_no_header_keeps_the_service_default(self):
        with patch.object(gw, "ENABLE_RAG", True), patch.object(gw, "RAG_TOP_K", 8), \
            patch.object(gw, "RAG_MAX_TOP_K", 20), patch.object(gw, "RAG_MIN_SCORE", 0.3), \
            patch.object(gw, "RAG_RERANK", False), patch.object(gw, "RAG_RERANK_TOP_N", 0):
            settings = gw._rag_settings(None, None, None, None, None)
        # `None` is no opinion, not `False`: the gateway sends no field, so
        # `api`'s RERANK_ENABLED answers, and the payload stays the pre-P1.1 one.
        self.assertEqual(settings[:4], (True, 8, 0.3, None))
        # 0 means "ask for no pool": `api`'s RERANK_TOP_N applies, and the gateway
        # has no model of its own to pick a pool size by.
        self.assertIsNone(settings[4])

    def test_the_header_overrides_the_default_on_one_request(self):
        # An agent can ask for reranking on one call and refuse it on the next.
        # `off` is carried as `False`, not dropped, so a deployment with the model
        # loaded can be measured on its fused path.
        with patch.object(gw, "RAG_RERANK", False):
            self.assertTrue(gw._rag_settings(None, None, None, "on", None)[3])
        with patch.object(gw, "RAG_RERANK", True):
            self.assertFalse(gw._rag_settings(None, None, None, "off", None)[3])

    def test_the_pool_header_overrides_the_service_default(self):
        with patch.object(gw, "RAG_RERANK_TOP_N", 0):
            self.assertEqual(gw._rag_settings(None, None, None, "on", "50")[4], 50)
        with patch.object(gw, "RAG_RERANK_TOP_N", 30):
            self.assertEqual(gw._rag_settings(None, None, None, "on", None)[4], 30)

    def test_a_pool_header_out_of_range_is_a_400(self):
        # `api` answers with the fused list when it cannot use a pool, so a bad pool
        # is reported here, not paid for on the GPU.
        for value in ("0", "101", "abc"):
            with self.assertRaises(StarletteHTTPException):
                gw._rag_settings(None, None, None, "on", value)

    def test_the_other_flags_keep_their_own_parsing(self):
        with patch.object(gw, "ENABLE_RAG", True), patch.object(gw, "RAG_TOP_K", 8), \
            patch.object(gw, "RAG_MAX_TOP_K", 20), patch.object(gw, "RAG_MIN_SCORE", 0.3), \
            patch.object(gw, "RAG_RERANK", False), patch.object(gw, "RAG_RERANK_TOP_N", 0):
            settings = gw._rag_settings("off", "5", "0.6", "on", None)
        self.assertEqual(settings[:4], (False, 5, 0.6, True))

    def test_a_bad_rerank_flag_is_a_400_not_a_silent_off(self):
        with self.assertRaises(StarletteHTTPException):
            gw._rag_settings(None, None, None, "maybe", None)


class RequestBodyTests(unittest.TestCase):
    def test_no_request_sends_the_pre_p11_payload(self):
        # A request that asks for nothing is byte-identical to the request an `api`
        # without reranking would accept, and `RERANK_ENABLED` on `api` decides.
        _, captured = search([RELEVANT], None)
        self.assertEqual(captured[0][1], {"query": "the query", "top_k": 8, "min_score": 0.3})

    def test_rerank_off_is_sent_as_an_explicit_answer(self):
        # `off` is a request, not an absence. On a deployment with
        # `RERANK_ENABLED=true`, the only way to measure the fused path is to send
        # `false`, so the eval harness compares on/off against one running service.
        _, captured = search([RELEVANT], False)
        self.assertEqual(captured[0][1]["rerank"], False)
        self.assertNotIn("rerank_top_n", captured[0][1])

    def test_rerank_on_is_sent_as_a_flag(self):
        _, captured = search([RELEVANT], True)
        # the default pool is `api`'s RERANK_TOP_N, so no pool key is sent at all
        self.assertNotIn("rerank_top_n", captured[0][1])
        self.assertEqual(captured[0][1]["rerank"], True)

    def test_an_asked_pool_is_sent(self):
        _, captured = search([RELEVANT], True, pool=50)
        self.assertEqual(captured[0][1]["rerank_top_n"], 50)

    def test_a_pool_is_never_sent_without_reranking(self):
        # asking a pool while forcing rerank off asks the model to score nothing,
        # and a pool with no rerank field spends GPU time on a service the client
        # never asked to use, so the pool travels only with `rerank: true`
        _, captured = search([RELEVANT], False, pool=50)
        self.assertNotIn("rerank_top_n", captured[0][1])
        self.assertEqual(captured[0][1]["rerank"], False)

        _, captured = search([RELEVANT], None, pool=50)
        self.assertNotIn("rerank_top_n", captured[0][1])
        self.assertNotIn("rerank", captured[0][1])

    def test_the_rerank_score_of_a_hit_is_kept(self):
        hit = dict(RELEVANT, rerank_score=2.71828)
        results, _ = search([hit], True)
        self.assertEqual(results[0]["rerank_score"], 2.7183)

    def test_a_hit_without_a_rerank_score_has_no_key(self):
        results, _ = search([RELEVANT], True)
        self.assertNotIn("rerank_score", results[0])

    def test_the_threshold_still_filters_on_score(self):
        # min_score is the dense cosine number measured in P0, so a reranker
        # preference cannot resurrect a hit the threshold removes.
        noisy = dict(NOISY, rerank_score=9.9)
        results, _ = search([noisy, RELEVANT], True, min_score=0.5)
        self.assertEqual([hit["file"] for hit in results], ["/data/documents/TODO.md"])

    def test_the_answer_is_cut_to_top_k(self):
        hits = [dict(RELEVANT, chunk=i, score=0.9 - i / 100) for i in range(10)]
        results, _ = search(hits, True, min_score=-1.0, top_k=3)
        self.assertEqual(len(results), 3)

    def test_the_gateway_contract_is_backward_compatible(self):
        # An `api` that answers without `rerank_score` (an older build, or a
        # service with no reranker) is a valid answer.
        results, _ = search([RELEVANT], True)
        self.assertEqual(results[0]["score"], 0.62)
        self.assertEqual(results[0]["file"], "/data/documents/TODO.md")


class ContextBlockTests(unittest.TestCase):
    def test_the_rerank_score_is_shown_for_audit(self):
        block = gw._format_context_block([dict(RELEVANT, rerank_score=2.71)])
        self.assertIn('relevance="0.62"', block)
        self.assertIn('rerank="2.71"', block)

    def test_no_rerank_score_is_no_extra_attribute(self):
        self.assertNotIn("rerank=", gw._format_context_block([RELEVANT]))

    def test_the_rerank_score_is_in_the_citation_header(self):
        headers = gw._rag_headers([dict(RELEVANT, rerank_score=2.71)])
        payload = json.loads(base64.urlsafe_b64decode(headers["X-RAG-Sources"] + "=="))
        self.assertEqual(payload[0]["rerank_score"], 2.71)
        self.assertEqual(payload[0]["score"], 0.62)

    def test_the_header_shape_is_unchanged_without_reranking(self):
        headers = gw._rag_headers([RELEVANT])
        payload = json.loads(base64.urlsafe_b64decode(headers["X-RAG-Sources"] + "=="))
        self.assertEqual(
            sorted(payload[0]),
            ["char_end", "char_start", "chunk", "file", "score", "section", "title"],
        )


class ContractTests(unittest.TestCase):
    def test_the_api_accepts_the_flag_the_gateway_sends(self):
        source = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
        self.assertIn("rerank: Optional[bool] = None", source)
        self.assertIn("rerank_top_n: Optional[int] =", source)
        self.assertIn("rerank_score: Optional[float] = None", source)

    def test_the_gateway_header_is_documented(self):
        source = (ROOT / "gateway" / "main.py").read_text(encoding="utf-8")
        self.assertIn("X-RAG-RERANK: off|on", source)
        self.assertIn("X-RAG-RERANK-TOP-N: 30", source)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("X-RAG-RERANK-TOP-N", readme)

    def test_the_default_is_off_in_every_plumbing(self):
        # Reranking costs GPU time, so nothing on the request path enables it by
        # accident: gateway default, compose, env template, api default.
        gateway = (ROOT / "gateway" / "main.py").read_text(encoding="utf-8")
        self.assertIn('_env_bool("RAG_RERANK", False)', gateway)
        compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
        self.assertIn("RAG_RERANK=${RAG_RERANK:-false}", compose)
        env_template = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("RAG_RERANK=false", env_template)
        # 0 = ask `api` for its own RERANK_TOP_N, so a deployment that never talks
        # about pool sizes sends no pool key at all.
        self.assertIn('_env_int("RAG_RERANK_TOP_N", 0)', gateway)
        self.assertIn("RAG_RERANK_TOP_N=${RAG_RERANK_TOP_N:-0}", compose)
        self.assertIn("RAG_RERANK_TOP_N=0", env_template)

    def test_the_version_bump_is_recorded(self):
        # The request and response contracts changed, so both services report the
        # new minor version. `2.2.0` added `rerank` and `rerank_top_n`; `2.3.0` adds
        # `rerank_query` and `X-RAG-RERANK-QUERY`.
        api_source = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
        gateway_source = (ROOT / "gateway" / "main.py").read_text(encoding="utf-8")
        self.assertIn('version="2.3.0"', api_source)
        self.assertIn('version="2.3.0"', gateway_source)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("2.3.0", readme)

    def test_the_fallback_is_recorded_in_the_readme(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("RERANK_ENABLED", readme)
        self.assertIn("rerank_score", readme)
        # The contract statement the acceptance criteria are checked against.
        self.assertTrue(
            re.search(r"reranker[^.\n]*fused[^.\n]*order", readme, re.IGNORECASE),
            "the README must state that a failed reranker returns the fused order",
        )


class PairQueryTests(unittest.TestCase):
    """`X-RAG-RERANK-QUERY`: which text the cross-encoder judges."""

    def search_pair(self, rerank, mode, question, pool=None):
        factory, captured = stub_client([RELEVANT])
        with patch.object(gw.httpx, "AsyncClient", factory):
            asyncio.run(gw._rag_search("the rewritten retrieval text", 8, 0.3,
                                       rerank, pool, mode, question))
        return captured[0][1]

    def test_no_header_is_no_opinion(self):
        self.assertIsNone(gw._rerank_query_header(None))
        self.assertIsNone(gw._rerank_query_header("   "))

    def test_both_settings_are_accepted(self):
        for value in ("retrieval", "RETRIEVAL", "question", "Question"):
            self.assertIn(gw._rerank_query_header(value), ("retrieval", "question"))

    def test_a_bad_pair_query_is_a_400(self):
        # A pair query is a request to spend GPU time on a specific text, so a typo
        # is reported on the request, not paid for by the model.
        for value in ("answer", "chunk", "off"):
            with self.assertRaises(StarletteHTTPException) as ctx:
                gw._rerank_query_header(value)
            self.assertEqual(ctx.exception.status_code, 400)

    def test_no_header_uses_the_service_setting(self):
        with patch.object(gw, "RAG_RERANK_QUERY", "question"):
            self.assertEqual(gw._rag_settings(None, None, None, "on", None, None)[5], "question")
        with patch.object(gw, "RAG_RERANK_QUERY", "retrieval"):
            self.assertEqual(gw._rag_settings(None, None, None, "on", None, None)[5], "retrieval")

    def test_the_header_overrides_the_service_setting(self):
        with patch.object(gw, "RAG_RERANK_QUERY", "retrieval"):
            self.assertEqual(gw._rag_settings(None, None, None, "on", None, "question")[5],
                             "question")

    def test_retrieval_sends_no_pair_query(self):
        # `retrieval` is what `api` already does with the text it was given, so the
        # default path sends no key at all and the payload stays the 2.2.0 one.
        body = self.search_pair(True, "retrieval", "the question")
        self.assertNotIn("rerank_query", body)
        self.assertEqual(body["rerank"], True)

    def test_the_question_is_sent_when_asked(self):
        # The measured lever: retrieve with the rewrite, score against the question.
        body = self.search_pair(True, "question", "Сколько измерений у её векторов?")
        self.assertEqual(body["rerank_query"], "Сколько измерений у её векторов?")

    def test_a_pair_query_never_travels_without_reranking(self):
        # Same rule as the pool key: asking the model to score a text while forcing
        # rerank off asks it to score nothing.
        body = self.search_pair(False, "question", "the question")
        self.assertNotIn("rerank_query", body)
        self.assertEqual(body["rerank"], False)

    def test_a_long_question_is_a_head_not_a_rejection(self):
        # `api` caps `rerank_query`, so sending a whole chat message as the pair query
        # is a request `api` answers with 422. The gateway sends the head.
        body = self.search_pair(True, "question", "测量" * 3000)
        self.assertEqual(len(body["rerank_query"]), gw.RERANK_QUERY_MAX_CHARS)


if __name__ == "__main__":
    unittest.main()
