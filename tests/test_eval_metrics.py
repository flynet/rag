"""Unit tests for the retrieval evaluation harness (stdlib only).

Run from the repository root:

    python -m unittest discover -s tests -p "test_*.py"

No GPU, API, or Docker is involved: the metric tests work on synthetic hit
lists, the latency statistics on plain lists, and the HTTP session and latency
probe tests run against a local stub server on 127.0.0.1.
"""

import base64
import http.server
import json
import math
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_retrieval as ev  # noqa: E402
import latency_probe as lp  # noqa: E402


def hit(file: str, chunk: int, text: str, score: float):
    return {"file": file, "chunk": chunk, "text": text, "score": score}


HITS = [
    hit("/data/documents/a.txt", 0, "Intro about Alpha", 0.90),
    hit("/data/documents/b.txt", 1, "Unrelated Beta", 0.80),
    hit("/data/documents/a.txt", 3, "Detail about gamma", 0.55),
    hit("/data/documents/c.txt", 0, "Gamma note", 0.40),
    hit("/data/documents/a.txt", 5, "Tail of Alpha", 0.20),
]

EXPECTED = [{"file": "a.txt", "text_contains": "Detail about gamma"}]


class MatchingTests(unittest.TestCase):
    def test_basename_matches_container_path(self):
        self.assertTrue(ev._same_file("/data/documents/a.txt", "a.txt"))
        self.assertFalse(ev._same_file("/data/documents/a.txt", "b.txt"))

    def test_text_contains_is_case_insensitive(self):
        self.assertTrue(ev._matches(HITS[2], {"file": "a.txt", "text_contains": "GAMMA"}))
        self.assertFalse(ev._matches(HITS[1], {"file": "a.txt", "text_contains": "gamma"}))

    def test_all_expected_fields_must_match(self):
        self.assertFalse(ev._matches(HITS[2], {"file": "b.txt", "chunk": 3}))
        self.assertTrue(ev._matches(HITS[2], {"file": "a.txt", "chunk": 3}))

    def test_section_matches_in_gateway_mode(self):
        # gateway hits carry no text, so `section` is the criterion that survives
        hit = {"file": "README.md", "chunk": 4, "text": "",
               "section": "Gateway variables > RAG", "score": 0.61}
        self.assertTrue(ev._matches(hit, {"file": "README.md", "section": "RAG"}))
        self.assertFalse(ev._matches(hit, {"file": "README.md", "section": "Qdrant"}))
        # `text_contains` cannot match an empty text, which is why it is rejected
        self.assertFalse(ev._matches(hit, {"file": "README.md", "text_contains": "RAG"}))

    def test_first_relevant_rank_is_one_based(self):
        self.assertEqual(ev._first_relevant_rank(HITS, EXPECTED), 3)
        self.assertIsNone(ev._first_relevant_rank(HITS, [{"file": "z.txt"}]))

    def test_ndcg_at_k(self):
        # ideal = 1 relevant hit, so nDCG is 1/log2(1+3) = 1/2
        self.assertAlmostEqual(ev._ndcg_at_k(HITS, EXPECTED, 10), 0.5, places=6)
        self.assertEqual(ev._ndcg_at_k(HITS, [{"file": "z.txt"}], 10), 0.0)

    def test_ndcg_ideal_places_all_relevant_first(self):
        hits = [
            hit("a.txt", 0, "x", 0.9),
            hit("a.txt", 1, "y", 0.8),
            hit("b.txt", 0, "z", 0.7),
        ]
        relevant = [{"file": "a.txt"}, {"file": "b.txt"}]
        # the ideal ranking has 2 relevant hits: 1/log2(2) + 1/log2(3)
        idcg = 1.0 + 1.0 / math.log2(3)
        # actual: ranks 1, 2, 3 all relevant
        dcg = 1.0 + 1.0 / math.log2(3) + 1.0 / math.log2(4)
        self.assertAlmostEqual(ev._ndcg_at_k(hits, relevant, 10), dcg / idcg, places=6)
        # a cut at 2 keeps ranks 1 and 2, ideal stays at 2
        self.assertAlmostEqual(ev._ndcg_at_k(hits, relevant, 2), 1.0, places=6)


class ScoreMetricsTests(unittest.TestCase):
    def test_no_threshold_keeps_everything(self):
        metrics = ev.score_metrics(HITS, EXPECTED, 10, ev.NO_THRESHOLD)
        self.assertEqual(metrics["injected"], 5.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["zero_context"], 0.0)
        self.assertEqual(metrics["mrr"], 1 / 3)

    def test_threshold_drops_the_relevant_hit_and_recall(self):
        metrics = ev.score_metrics(HITS, EXPECTED, 10, 0.6)
        self.assertEqual(metrics["injected"], 2.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["mrr"], 0.0)

    def test_threshold_below_the_relevant_hit_keeps_recall(self):
        metrics = ev.score_metrics(HITS, EXPECTED, 10, 0.5)
        self.assertEqual(metrics["injected"], 3.0)
        self.assertEqual(metrics["recall"], 1.0)

    def test_top_k_cap_limits_context_and_recall(self):
        metrics = ev.score_metrics(HITS, EXPECTED, 2, ev.NO_THRESHOLD)
        self.assertEqual(metrics["injected"], 2.0)
        self.assertEqual(metrics["recall"], 0.0)

    def test_threshold_can_remove_all_context(self):
        metrics = ev.score_metrics(HITS, EXPECTED, 3, 0.95)
        self.assertEqual(metrics["injected"], 0.0)
        self.assertEqual(metrics["zero_context"], 1.0)


class SweepTests(unittest.TestCase):
    def setUp(self):
        self.results = [
            {
                "id": "q1",
                "expected": EXPECTED,
                "hits": HITS,
                "returned": len(HITS),
                "first_rank": 3,
            },
            {
                "id": "q2",
                "expected": [{"file": "b.txt"}],
                "hits": HITS,
                "returned": len(HITS),
                "first_rank": 2,
            },
        ]

    def test_grid_shape(self):
        grid = ev.sweep(self.results, [3, 5], [0.5, 0.9])
        self.assertEqual(grid["top_ks"], [3, 5])
        self.assertEqual(len(grid["rows"]), 2)

    def test_recall_and_context_are_averaged_over_questions(self):
        grid = ev.sweep(self.results, [3], [0.5])
        row = grid["rows"][0]
        # q1: relevant at rank 3 with score 0.55 -> injected; q2: relevant at rank 2
        self.assertAlmostEqual(row["recall@3"], 1.0)
        self.assertAlmostEqual(row["injected@3"], 3.0)
        self.assertAlmostEqual(row["zero_context@3"], 0.0)
        self.assertAlmostEqual(row["mrr@3"], (1 / 3 + 1 / 2) / 2)

    def test_high_threshold_leaves_questions_without_context(self):
        grid = ev.sweep(self.results, [5], [0.95])
        row = grid["rows"][0]
        # nothing survives the threshold, so every question loses its context
        self.assertAlmostEqual(row["zero_context@5"], 1.0)
        self.assertAlmostEqual(row["injected@5"], 0.0)
        self.assertAlmostEqual(row["recall@5"], 0.0)

    def test_threshold_can_keep_one_question_relevant(self):
        grid = ev.sweep(self.results, [5], [0.8])
        row = grid["rows"][0]
        # q2's b.txt hit (0.80) survives and is relevant; q1's relevant hit (0.55) is cut
        self.assertAlmostEqual(row["recall@5"], 0.5)
        self.assertAlmostEqual(row["injected@5"], 2.0)
        self.assertAlmostEqual(row["mrr@5"], (0.0 + 1 / 2) / 2)
        self.assertAlmostEqual(row["zero_context@5"], 0.0)

    def test_markdown_renders_every_metric_table(self):
        grid = ev.sweep(self.results, [3, 5], [0.5, ev.NO_THRESHOLD])
        text = ev.sweep_markdown(grid, grid["top_ks"], "2026-01-01T00:00:00+00:00", "eval/sweep-multiturn.md")
        for metric in ("recall@3", "recall@5", "mrr@3", "injected@5", "zero_context@3"):
            self.assertIn(metric, text)
        self.assertIn("| off |", text)
        self.assertIn("| 0.50 |", text)
        self.assertEqual(text.count("## "), 4)
        self.assertIn("--sweep-md eval/sweep-multiturn.md", text)


class SourceHeaderTests(unittest.TestCase):
    def test_decode_unpadded_base64url(self):
        payload = [{"file": "/data/documents/a.txt", "chunk": 2, "score": 0.61}]
        raw = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode()
        ).decode().rstrip("=")
        self.assertEqual(ev.decode_sources(raw), payload)

    def test_missing_header_is_empty(self):
        self.assertEqual(ev.decode_sources(None), [])
        self.assertEqual(ev.decode_sources(""), [])

    def test_normalized_headers_are_read_case_insensitively(self):
        payload = [{"file": "/data/documents/a.txt", "chunk": 2, "score": 0.61}]
        raw = base64.urlsafe_b64encode(
            json.dumps(payload, separators=(",", ":")).encode()
        ).decode().rstrip("=")
        # uvicorn delivers header names lowercased
        self.assertEqual(ev.sources_from_headers({"x-rag-sources": raw}), payload)
        # normalize the way _post_chat does, then read
        self.assertEqual(
            ev.sources_from_headers({k.lower(): v for k, v in {"X-RAG-Sources": raw}.items()}),
            payload,
        )
        self.assertEqual(ev.sources_from_headers({}), [])


class DatasetTests(unittest.TestCase):
    def _write(self, lines):
        tmp = Path(tempfile.mkdtemp()) / "questions.jsonl"
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return tmp

    def test_ids_are_numbered_when_absent(self):
        path = self._write([
            json.dumps({"question": "alpha", "relevant": [{"file": "a.txt"}]}),
            json.dumps({"question": "beta", "relevant": [{"file": "b.txt", "chunk": 1}]}),
        ])
        questions = ev.load_questions(path)
        self.assertEqual([q["id"] for q in questions], ["q001", "q002"])
        self.assertEqual(questions[1]["relevant"][0]["chunk"], 1)

    def test_history_is_read_as_user_turns(self):
        path = self._write([
            json.dumps({"question": "and the second one?", "history": ["first topic", "second topic"],
                        "relevant": [{"file": "a.txt"}]}),
        ])
        self.assertEqual(ev.load_questions(path)[0]["history"], ["first topic", "second topic"])

    def test_text_contains_is_rejected_in_gateway_mode(self):
        path = self._write([
            json.dumps({"question": "alpha", "relevant": [{"file": "a.txt", "text_contains": "x"}]}),
        ])
        with self.assertRaises(SystemExit):
            ev.load_questions(path, gateway=True)

    def test_unknown_field_is_rejected(self):
        path = self._write([
            json.dumps({"question": "alpha", "relevant": [{"page": 3}]}),
        ])
        with self.assertRaises(SystemExit):
            ev.load_questions(path)

    def test_bad_history_is_rejected(self):
        path = self._write([
            json.dumps({"question": "alpha", "relevant": [{"file": "a.txt"}], "history": "not a list"}),
        ])
        with self.assertRaises(SystemExit):
            ev.load_questions(path)


class CorpusCheckTests(unittest.TestCase):
    def setUp(self):
        self.corpus = Path(tempfile.mkdtemp())
        (self.corpus / "README.md").write_text("# Guide\n\nAlpha details\n\n## Setup\n", encoding="utf-8")

    def test_ground_truth_matches(self):
        questions = [{"id": "q1", "relevant": [{"file": "README.md", "text_contains": "alpha"}]}]
        self.assertEqual(ev.check_corpus(questions, self.corpus), [])

    def test_section_path_is_checked_segment_by_segment(self):
        # the joined path never appears in the file, its segments do
        questions = [{"id": "q1", "relevant": [{"file": "README.md", "section": "Guide > Setup"}]}]
        self.assertEqual(ev.check_corpus(questions, self.corpus), [])

    def test_section_segment_missing_from_the_file_is_reported(self):
        questions = [{"id": "q2", "relevant": [{"file": "README.md", "section": "Guide > Missing"}]}]
        problems = ev.check_corpus(questions, self.corpus)
        self.assertEqual(len(problems), 1)
        self.assertIn("section", problems[0])

    def test_missing_text_is_reported(self):
        questions = [{"id": "q1", "relevant": [{"file": "README.md", "text_contains": "gamma"}]}]
        problems = ev.check_corpus(questions, self.corpus)
        self.assertEqual(len(problems), 1)
        self.assertIn("not in", problems[0])

    def test_missing_file_is_reported(self):
        questions = [{"id": "q1", "relevant": [{"file": "gone.md", "text_contains": "alpha"}]}]
        problems = ev.check_corpus(questions, self.corpus)
        self.assertTrue(any("not found" in problem for problem in problems))


class _StubServer(http.server.ThreadingHTTPServer):
    """Local stub API: counts connections and bodies, no GPU, API, or Docker.

    Threading so several keep-alive connections can be served at the same time,
    the way the real API does it.
    """

    def __init__(self, handler_class):
        super().__init__(("127.0.0.1", 0), handler_class)
        self.connections = 0
        self.paths = []
        self.bodies = []

    def process_request(self, request, client_address):
        self.connections += 1
        return super().process_request(request, client_address)

    def handle_error(self, request, client_address):
        pass


class _StubHandler(http.server.BaseHTTPRequestHandler):
    # keep-alive is what the session relies on, so speak HTTP/1.1
    protocol_version = "HTTP/1.1"
    SOURCES = [{"file": "/data/documents/a.txt", "chunk": 2, "score": 0.61,
               "section": "Guide > Setup", "char_start": 0, "char_end": 18}]

    def _reply(self, payload, extra_headers=None):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        self.server.bodies.append(body)
        self.server.paths.append(self.path)

        # a slow query lets the tests check that a time window stops the loop
        if isinstance(body, dict) and body.get("query") == "slow":
            time.sleep(0.15)

        if self.path == "/boom":
            self.send_response(422)
            self.send_header("Content-Length", "17")
            self.end_headers()
            self.wfile.write(b'{"detail":"failed"}')
        elif self.path == "/v1/chat/completions":
            raw = base64.urlsafe_b64encode(
                json.dumps(self.SOURCES).encode()
            ).decode().rstrip("=")
            self._reply(b"{}", {"X-RAG-Sources": raw})
        else:
            hit = {"file": "/data/documents/a.txt", "chunk": 2,
                   "text": "Detail about gamma", "section": "Guide > Setup",
                   "score": 0.61}
            self._reply(json.dumps([hit]).encode())

    def do_GET(self):
        self.server.paths.append(self.path)
        if self.path == "/health":
            defaults = {"top_k": 8, "min_score": 0.3,
                        "query_rewrite": "concat", "rewrite_history": 3}
            self._reply(json.dumps({"status": "ok", "defaults": defaults}).encode())
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def log_message(self, format, *args):  # keep the test output clean
        pass


class SessionTests(unittest.TestCase):
    def setUp(self):
        self.server = _StubServer(_StubHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.session = ev.Session(f"http://127.0.0.1:{self.server.server_port}", 10.0)

    def tearDown(self):
        self.session.close()
        self.server.shutdown()
        self.server.server_close()

    def test_one_connection_serves_every_request(self):
        for _ in range(3):
            body, _headers = self.session.post(
                "/search", {"query": "alpha", "top_k": 10, "min_score": None}
            )
            self.assertEqual(len(body), 1)
        self.assertEqual(len(self.server.bodies), 3)
        self.assertEqual(self.server.connections, 1)

    def test_evaluate_search_reuses_the_connection(self):
        questions = [{"id": "q1", "question": "alpha", "history": [],
                      "relevant": [{"file": "a.txt", "text_contains": "gamma"}]}]
        results = ev.evaluate_search(self.session, questions, 10)
        self.assertEqual(results[0]["first_rank"], 1)
        self.assertEqual(self.server.connections, 1)

    def test_response_headers_are_normalized(self):
        _body, headers = self.session.post(
            "/v1/chat/completions", {"model": "stub", "messages": []}
        )
        self.assertIn("x-rag-sources", headers)
        self.assertEqual(ev.sources_from_headers(headers), _StubHandler.SOURCES)

    def test_non_2xx_reports_status_and_body(self):
        with self.assertRaises(ev.HttpError) as raised:
            self.session.post("/boom", {})
        self.assertEqual(raised.exception.status, 422)
        self.assertIn("failed", raised.exception.body)

    def test_unreachable_host_is_reported(self):
        dead = ev.Session("http://127.0.0.1:1", 2.0)
        with self.assertRaises(ConnectionError):
            dead.post("/search", {"query": "alpha", "top_k": 1, "min_score": None})

    def test_gateway_defaults_are_recorded(self):
        defaults = ev.read_gateway_defaults(self.session, "")
        self.assertEqual(defaults["query_rewrite"], "concat")
        self.assertEqual(defaults["top_k"], 8)

    def test_gateway_without_defaults_is_not_fatal(self):
        # an older gateway, or /health behind a route that 404s, must not stop the run
        with patch.object(self.session, "get", side_effect=ev.HttpError(404, "not found")):
            self.assertEqual(ev.read_gateway_defaults(self.session, "sk-test"), {})


class GatewayCorpusTests(unittest.TestCase):
    """Gateway mode reads the chunk text back from the corpus via char offsets."""

    def setUp(self):
        self.server = _StubServer(_StubHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.session = ev.Session(f"http://127.0.0.1:{self.server.server_port}", 10.0)
        self.tmp = tempfile.TemporaryDirectory()
        self.corpus = Path(self.tmp.name)
        (self.corpus / "a.txt").write_bytes(b"Detail about gamma\nRest of the file\r\n")

    def tearDown(self):
        self.session.close()
        self.server.shutdown()
        self.server.server_close()
        self.tmp.cleanup()

    def test_corpus_text_is_normalized_like_the_chunker(self):
        texts = ev.corpus_texts(self.corpus)
        self.assertEqual(texts["a.txt"], "Detail about gamma\nRest of the file\n")

    def test_text_contains_is_accepted_in_gateway_mode_with_a_corpus(self):
        path = self.corpus / "q.jsonl"
        path.write_text(json.dumps({
            "id": "q1", "question": "alpha",
            "relevant": [{"file": "a.txt", "text_contains": "Detail about gamma"}],
        }) + "\n", encoding="utf-8")
        questions = ev.load_questions(path, gateway=True, corpus_available=True)
        self.assertEqual(len(questions), 1)

    def test_gateway_hit_is_matched_through_the_corpus_slice(self):
        questions = [{"id": "q1", "question": "alpha", "history": [],
                      "relevant": [{"file": "a.txt", "text_contains": "Detail about gamma"}]}]
        results = ev.evaluate_gateway(self.session, "", "stub", questions, 20,
                                      ev.corpus_texts(self.corpus))
        hit = results[0]["hits"][0]
        self.assertEqual(hit["text"], "Detail about gamma")
        self.assertEqual(results[0]["first_rank"], 1)

    def test_a_hit_from_a_file_outside_the_corpus_has_no_text(self):
        questions = [{"id": "q1", "question": "alpha", "history": [],
                      "relevant": [{"file": "a.txt", "text_contains": "Detail about gamma"}]}]
        results = ev.evaluate_gateway(self.session, "", "stub", questions, 20, {})
        self.assertEqual(results[0]["hits"][0]["text"], "")
        self.assertIsNone(results[0]["first_rank"])

    def test_an_expected_file_outside_the_corpus_does_not_match(self):
        questions = [{"id": "q1", "question": "alpha", "history": [],
                      "relevant": [{"file": "z.txt", "text_contains": "Detail about gamma"}]}]
        results = ev.evaluate_gateway(self.session, "", "stub", questions, 20,
                                      ev.corpus_texts(self.corpus))
        self.assertIsNone(results[0]["first_rank"])


class LatencyTests(unittest.TestCase):
    """The latency probe must reuse connections: connection setup on Windows
    costs seconds, and measuring it would bury the number being measured."""

    def setUp(self):
        self.server = _StubServer(_StubHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()

    def test_percentile_is_nearest_rank(self):
        values = [float(i) for i in range(1, 101)]
        self.assertEqual(lp.percentile(values, 0.50), 50.0)
        self.assertEqual(lp.percentile(values, 0.95), 95.0)
        self.assertEqual(lp.percentile(values, 0.99), 99.0)
        self.assertEqual(lp.percentile([], 0.95), 0.0)

    def test_summarise_reports_the_core_stats(self):
        stats = lp.summarise([1.0, 2.0, 3.0])
        self.assertEqual(stats["count"], 3)
        self.assertEqual(stats["p50"], 2.0)
        self.assertEqual(stats["p95"], 3.0)
        self.assertAlmostEqual(stats["mean"], 2.0)
        self.assertEqual(lp.summarise([])["count"], 0)

    def test_measure_search_reuses_one_connection_per_worker(self):
        sessions = [ev.Session(self.url, 10.0) for _ in range(2)]
        try:
            latencies = lp.measure_search(sessions, ["alpha", "beta"], 6)
            self.assertEqual(len(latencies), 6)
            self.assertEqual(len(self.server.bodies), 6)
            # six requests, two workers: two connections, not six
            self.assertEqual(self.server.connections, 2)
            self.assertTrue(all(value > 0 for value in latencies))
        finally:
            for session in sessions:
                session.close()

    def test_duration_phase_stops_at_the_deadline(self):
        session = ev.Session(self.url, 10.0)
        try:
            # the stub answers a "slow" query in 0.15s, so 1000 of them would
            # take 150s; a 0.3s window must stop the loop after a few requests
            latencies = lp.measure_search([session], ["slow"], 1000, duration=0.3)
            self.assertGreater(len(latencies), 0)
            self.assertLess(len(latencies), 5)
            self.assertGreater(latencies[0], 0.1)
        finally:
            session.close()


if __name__ == "__main__":
    unittest.main()
