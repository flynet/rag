"""Unit tests for hybrid retrieval (stdlib only).

Run from the repository root:

    python -m unittest discover -s tests -p "test_*.py"

`api/hybrid.py` has no third-party imports, so fusion and the threshold rules are
tested without Qdrant, GPU, Docker, torch, or fastembed. The sparse wire format
from `POST /embed` is tested against a local stub server.
"""

import asyncio
import http.server
import json
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
sys.path.insert(0, str(ROOT / "embedder"))

import hybrid  # noqa: E402


class Point:
    """A stand-in for a qdrant point match."""

    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload


def hits(*items):
    return hybrid.normalize_hits(
        [Point(i, s, {"file": f"/data/documents/{i}.txt", "chunk": i, "text": i}) for i, s in items]
    )


class NormalizeTests(unittest.TestCase):
    def test_objects_and_dicts(self):
        raw = hits(("a", 0.9), ("b", 0.5)) + [{"id": "c", "score": 0.2, "payload": {"file": "f"}}]
        out = hybrid.normalize_hits(raw)
        self.assertEqual([h["id"] for h in out], ["a", "b", "c"])
        self.assertEqual(out[2]["payload"]["file"], "f")

    def test_bad_entries_are_dropped(self):
        out = hybrid.normalize_hits([Point(None, 0.5, {}), Point("x", None, {}), Point("y", 0.1, None)])
        self.assertEqual([h["id"] for h in out], ["y"])
        self.assertEqual(out[0]["payload"], {})


class FuseTests(unittest.TestCase):
    def test_rrf_math(self):
        dense = hits(("a", 0.9), ("b", 0.8))
        sparse = hits(("b", 5.0), ("c", 4.0))
        fused = hybrid.fuse(dense, sparse, top_k=3)
        k = hybrid.RRF_K
        # `b` is 2nd in dense and 1st in sparse: it is the only hit in both lists.
        self.assertEqual(fused[0]["id"], "b")
        self.assertAlmostEqual(fused[0]["fusion_score"], 1 / (k + 2) + 1 / (k + 1))
        self.assertAlmostEqual(fused[1]["fusion_score"], 1 / (k + 1))
        self.assertEqual([h["id"] for h in fused[1:]], ["a", "c"])

    def test_dense_and_sparse_scores_kept(self):
        fused = hybrid.fuse(hits(("a", 0.9)), hits(("b", 3.5)), top_k=2)
        by_id = {h["id"]: h for h in fused}
        self.assertEqual(by_id["a"]["dense_score"], 0.9)
        self.assertIsNone(by_id["a"]["sparse_score"])
        self.assertEqual(by_id["b"]["sparse_score"], 3.5)
        self.assertIsNone(by_id["b"]["dense_score"])
        self.assertEqual(by_id["b"]["ranks"], {"sparse": 1})

    def test_top_k_limit(self):
        fused = hybrid.fuse(hits(("a", 0.9), ("b", 0.8), ("c", 0.7)), [], top_k=2)
        self.assertEqual(len(fused), 2)
        self.assertEqual([h["id"] for h in fused], ["a", "b"])

    def test_tie_broken_by_dense_score_then_id(self):
        # Rank 1 in one list each: `a` (dense) and `c` (sparse) share the fused
        # score. The hit with a dense cosine score comes first, then `c`.
        dense = hits(("a", 0.6), ("b", 0.7))
        sparse = hits(("c", 5.0), ("d", 5.0))
        fused = hybrid.fuse(dense, sparse, top_k=4)
        self.assertEqual([h["id"] for h in fused], ["a", "c", "b", "d"])

    def test_equal_scores_fall_back_to_id(self):
        # Rank 1 in each list, and the dense score is 0.0, so the tie is broken
        # by the id: the order is reproducible across runs.
        dense = [{"id": "d", "score": 0.0, "payload": {}}]
        sparse = [{"id": "c", "score": 0.0, "payload": {}}]
        fused = hybrid.fuse(dense, sparse, top_k=2)
        self.assertEqual([h["id"] for h in fused], ["c", "d"])

    def test_empty_lists(self):
        self.assertEqual(hybrid.fuse([], [], top_k=5), [])


class RelativeFilterTests(unittest.TestCase):
    def test_weak_term_overlap_is_dropped(self):
        entries = hybrid.normalize_hits(hits(("a", 10.0), ("b", 6.0), ("c", 4.0)))
        kept = hybrid.filter_relative(entries, 0.5)
        self.assertEqual([h["id"] for h in kept], ["a", "b"])

    def test_ratio_zero_keeps_everything(self):
        entries = hybrid.normalize_hits(hits(("a", 10.0), ("b", 0.5)))
        self.assertEqual(len(hybrid.filter_relative(entries, 0.0)), 2)

    def test_empty_list(self):
        self.assertEqual(hybrid.filter_relative([], 0.5), [])

    def test_all_equal_scores_survive(self):
        entries = hybrid.normalize_hits(hits(("a", 3.0), ("b", 3.0)))
        self.assertEqual(len(hybrid.filter_relative(entries, 1.0)), 2)


class ThresholdTests(unittest.TestCase):
    def test_threshold_applies_to_dense_cosine_only(self):
        fused = hybrid.fuse(hits(("a", 0.55), ("b", 0.25)), hits(("c", 3.0)), top_k=3)
        kept = hybrid.apply_threshold(fused, 0.3)
        self.assertEqual([h["id"] for h in kept], ["a", "c"])

    def test_bm25_only_hits_survive(self):
        # 0.9 strips every dense hit; the exact-term hit has no cosine score.
        fused = hybrid.fuse(hits(("a", 0.2)), hits(("b", 4.0)), top_k=2)
        kept = hybrid.apply_threshold(fused, 0.9)
        self.assertEqual([h["id"] for h in kept], ["b"])

    def test_no_threshold_keeps_everything(self):
        fused = hybrid.fuse(hits(("a", 0.1)), hits(("b", 1.0)), top_k=2)
        self.assertEqual(len(hybrid.apply_threshold(fused, None)), 2)


class _StubHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length).decode())
        texts = request["texts"]
        n = len(texts)
        # The term id depends on the text, so a batch handling bug shows up as
        # sparse vectors that do not line up with the texts they came from.
        body = {
            "dim": 2,
            "vectors": [[0.1, 0.2] for _ in range(n)],
            "sparse": [
                {"indices": [100 + ord(text)], "values": [1.5]} for text in texts
            ],
        }
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # keep the test output clean
        pass


class _SparseShortHandler(_StubHandler):
    """A stub that returns one sparse vector fewer than the texts it was sent."""

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length).decode())
        n = len(request["texts"])
        body = {
            "dim": 2,
            "vectors": [[0.1, 0.2] for _ in range(n)],
            "sparse": [{"indices": [7], "values": [1.5]} for _ in range(n - 1)],
        }
        payload = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class _DenseOnlyHandler(_StubHandler):
    """A stub for an `api` without BM25: no `sparse` key in the response."""

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length).decode())
        n = len(request["texts"])
        payload = json.dumps({"dim": 2, "vectors": [[0.1, 0.2] for _ in range(n)]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


def start_stub(handler_class):
    server = http.server.HTTPServer(("127.0.0.1", 0), handler_class)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


class RemoteEmbedderTests(unittest.TestCase):
    """`POST /embed` returns dense vectors and BM25 sparse vectors, batch-aligned."""

    def test_dense_and_sparse_aligned_across_batches(self):
        server = start_stub(_StubHandler)
        try:
            from embedder import RemoteEmbedder

            client = RemoteEmbedder(f"http://127.0.0.1:{server.server_address[1]}", batch_size=2)
            texts = ["a", "b", "c", "d", "e"]

            async def call():
                return await client.embed_with_sparse(texts)

            dense, sparse = asyncio.run(call())
            self.assertEqual(dense.shape, (5, 2))
            # Each sparse vector corresponds to its own text, in order, even when
            # the texts were sent in batches.
            self.assertEqual([s["indices"] for s in sparse], [[197], [198], [199], [200], [201]])
            self.assertEqual(client.dim, 2)

            async def embed_only():
                return await client.embed(texts)

            self.assertEqual(asyncio.run(embed_only()).shape, (5, 2))
        finally:
            server.shutdown()
            server.server_close()

    def test_response_without_sparse_is_dense_only(self):
        server = start_stub(_DenseOnlyHandler)
        try:
            from embedder import RemoteEmbedder

            client = RemoteEmbedder(f"http://127.0.0.1:{server.server_address[1]}", batch_size=2)

            async def call():
                return await client.embed_with_sparse(["a", "b"])

            dense, sparse = asyncio.run(call())
            self.assertEqual(dense.shape, (2, 2))
            self.assertEqual(sparse, [])
        finally:
            server.shutdown()
            server.server_close()

    def test_mismatched_sparse_count_is_a_hard_error(self):
        server = start_stub(_SparseShortHandler)
        try:
            from embedder import RemoteEmbedder

            client = RemoteEmbedder(f"http://127.0.0.1:{server.server_address[1]}", batch_size=2)

            async def call():
                return await client.embed_with_sparse(["a", "b", "c"])

            with self.assertRaises(RuntimeError):
                asyncio.run(call())
        finally:
            server.shutdown()
            server.server_close()

    def test_empty_texts(self):
        from embedder import RemoteEmbedder

        client = RemoteEmbedder("http://127.0.0.1:1", batch_size=2)
        client.dim = 2

        async def call():
            return await client.embed_with_sparse([])

        dense, sparse = asyncio.run(call())
        self.assertEqual(dense.shape, (0, 2))
        self.assertEqual(sparse, [])


class SparseWireFormatTests(unittest.TestCase):
    def test_sparse_vectors_are_positive_int_indices(self):
        # The contract Qdrant takes: indices must be ints, values floats.
        item = {"indices": [1234567890, 42], "values": [1.25, 0.5]}
        self.assertTrue(all(isinstance(i, int) and i > 0 for i in item["indices"]))
        self.assertEqual(len(item["indices"]), len(item["values"]))


if __name__ == "__main__":
    unittest.main()
