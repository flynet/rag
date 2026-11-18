"""Unit tests for the query embedding cache (`Embedder.embed_query`).

Run from the repository root:

    python -m unittest discover -s tests -p "test_*.py"

`Embedder` is built against a fake model written here: `sentence_transformers` is
imported inside `Embedder.__init__` (the convention `SparseEmbedder` uses for an
optional heavy dependency), so a fake module in `sys.modules` is enough to build a
real `Embedder` on a machine with torch and no sentence-transformers, no GPU, no
Docker, and no model download. The fake model counts forward passes, which is the
thing P3.2 has to prove: a repeated question must not make one.
"""

import asyncio
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Loaded from the file path under a name that is not `embedder`: `embedder.py` exists
# in `api/` and in `embedder/`, and `sys.modules` has one slot. `test_hybrid.py` wants
# `embedder/embedder.py` under that name, so importing this file through the path is
# what keeps both test files working in one run.
_spec = importlib.util.spec_from_file_location("api_embedder", ROOT / "api" / "embedder.py")
emb_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(emb_mod)

DIM = 8
MODEL_PATH = "models/fake"


class FakeModel:
    """Stand-in for SentenceTransformer: deterministic vectors, counted calls.

    A vector is a pure function of the text it was made from, which is what makes a
    cache answer correct: same text -> the same vector, different text -> a different
    one. Lengths are used as the signal, so the texts in these tests are made to have
    distinct lengths and `self.calls` records every forward pass, in order.
    """

    def __init__(self, model_path, **kwargs):
        self.model_path = model_path
        self.max_seq_length = 512
        self.prompts = {"query": "Instruct: {q}"}
        self.calls = []

    def get_sentence_embedding_dimension(self):
        return DIM

    def encode(self, texts, batch_size=32, convert_to_numpy=True,
              normalize_embeddings=True, show_progress_bar=False, prompt_name=None):
        self.calls.append((tuple(texts), prompt_name))
        vectors = []
        for text in texts:
            marker = float(len(text) * 10 + (1 if prompt_name else 0))
            vectors.append(np.full(DIM, marker, dtype=np.float32))
        return np.vstack(vectors)


def make_embedder(model_path=MODEL_PATH, query_prompt="query", cache_size=256):
    """Build a real `Embedder` on the fake model."""
    return emb_mod.Embedder(
        model_path,
        device="cpu",
        dtype="float32",
        normalize=False,
        query_prompt=query_prompt,
        cache_size=cache_size,
    )


def run(coro):
    return asyncio.run(coro)


def expected(text: str) -> list:
    """The vector the fake model makes for `text` (with the query prompt applied)."""
    return np.full(DIM, len(text) * 10 + 1, dtype=np.float32).tolist()


class CacheTests(unittest.TestCase):
    def setUp(self):
        module = types.ModuleType("sentence_transformers")
        module.SentenceTransformer = FakeModel
        self._saved = sys.modules.get("sentence_transformers")
        sys.modules["sentence_transformers"] = module
        self.embedder = make_embedder()

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("sentence_transformers", None)
        else:
            sys.modules["sentence_transformers"] = self._saved

    def test_a_repeated_question_costs_no_forward_pass(self):
        first = run(self.embedder.embed_query("how many dimensions"))
        self.assertEqual(len(self.embedder.model.calls), 1)

        second = run(self.embedder.embed_query("how many dimensions"))
        self.assertEqual(len(self.embedder.model.calls), 1,
                         "the repeat re-encoded: the cache did not answer")
        self.assertEqual(first.tolist(), second.tolist())
        self.assertEqual((self.embedder.cache_hits, self.embedder.cache_misses), (1, 1))

    def test_a_hit_is_a_vector_of_the_full_dimension(self):
        """/search takes `vectors[0]` as the vector, so shape is the contract."""
        vectors = run(self.embedder.embed_query("alpha"))
        self.assertEqual(vectors.shape, (1, DIM))
        self.assertEqual(len(vectors[0]), DIM)

        cached = run(self.embedder.embed_query("alpha"))
        self.assertEqual(cached.shape, (1, DIM),
                         "a cache hit must return the same shape a miss returns")

    def test_the_cache_answers_the_vector_for_the_same_text(self):
        run(self.embedder.embed_query("long question"))
        run(self.embedder.embed_query("short"))
        # Distinct texts get distinct vectors, so a hit answering the wrong text is visible.
        repeat = run(self.embedder.embed_query("long question"))
        self.assertEqual(repeat[0].tolist(), expected("long question"))

    def test_whitespace_around_a_question_is_not_part_of_the_key(self):
        run(self.embedder.embed_query("  how many dimensions  "))
        run(self.embedder.embed_query("how many dimensions"))
        self.assertEqual(len(self.embedder.model.calls), 1)
        self.assertEqual(self.embedder.cache_hits, 1)

    def test_case_is_kept_because_lowercasing_would_change_the_embedding(self):
        run(self.embedder.embed_query("Alpha"))
        run(self.embedder.embed_query("alpha"))
        self.assertEqual(len(self.embedder.model.calls), 2,
                         "case-folding made two different strings one cache entry")

    def test_an_all_whitespace_query_is_not_handed_to_the_model_as_empty(self):
        run(self.embedder.embed_query("   "))
        texts, _ = self.embedder.model.calls[-1]
        self.assertEqual(texts, (" ",), "an empty string was sent to the model")

    def test_the_caller_cannot_reach_the_cached_vector(self):
        vectors = run(self.embedder.embed_query("how many dimensions"))
        vectors[0][0] = -1.0
        repeat = run(self.embedder.embed_query("how many dimensions"))
        self.assertEqual(repeat[0].tolist(), expected("how many dimensions"),
                         "the mutation reached the cache")

    def test_the_cache_is_bounded_by_entry_count(self):
        self.embedder.cache_max = 2
        for text in ("aa", "bbb", "cccc"):
            run(self.embedder.embed_query(text))
        self.assertLessEqual(len(self.embedder.cache), 2)
        self.assertEqual(len(self.embedder.cache), 2)

        # The oldest entry is gone, so the first text encodes again: 3 misses in the
        # loop, plus this one.
        run(self.embedder.embed_query("aa"))
        self.assertEqual(len(self.embedder.model.calls), 4)

    def test_the_newest_entry_survives_eviction(self):
        self.embedder.cache_max = 2
        for text in ("aa", "bbb", "cccc"):
            run(self.embedder.embed_query(text))
        run(self.embedder.embed_query("cccc"))
        self.assertEqual(len(self.embedder.model.calls), 3,
                         "LRU evicted the most recently used entry")

    def test_cache_size_zero_encodes_every_query(self):
        self.embedder.cache_max = 0
        run(self.embedder.embed_query("how many dimensions"))
        run(self.embedder.embed_query("how many dimensions"))
        self.assertEqual(len(self.embedder.model.calls), 2)
        self.assertEqual(len(self.embedder.cache), 0)
        self.assertEqual(self.embedder.cache_stats()["enabled"], False)

    def test_a_vector_from_a_different_model_is_not_served(self):
        run(self.embedder.embed_query("how many dimensions"))
        self.embedder.model_path = "models/other"
        run(self.embedder.embed_query("how many dimensions"))
        self.assertEqual(len(self.embedder.model.calls), 2,
                         "a redeployed model served the previous model's vector")

    def test_turning_the_retrieval_prompt_off_is_not_a_cache_hit(self):
        run(self.embedder.embed_query("how many dimensions"))
        self.embedder.query_prompt = None
        run(self.embedder.embed_query("how many dimensions"))
        self.assertEqual(len(self.embedder.model.calls), 2,
                         "the cached vector was made with a prompt the query is not using")

    def test_a_cache_hit_does_not_wait_behind_a_bulk_ingest(self):
        """The lookup happens before the encode queue, which is P2.4's win."""
        run(self.embedder.embed_query("how many dimensions"))

        async def probe():
            # The queue is held by a bulk ingest: a miss would block here, a hit answers
            # while it is held.
            async with self.embedder.encode_queue:
                await asyncio.sleep(0.3)
                return "the queue was free, which is not the point of this test"

        async def check():
            holder = asyncio.create_task(probe())
            hit = await asyncio.wait_for(self.embedder.embed_query("how many dimensions"), 0.1)
            holder.cancel()
            return hit

        hit = asyncio.run(check())
        self.assertEqual(hit[0].tolist(), expected("how many dimensions"))

    def test_documents_do_not_go_through_the_cache(self):
        run(self.embedder.embed(["chunk one", "chunk two"]))
        self.assertEqual(len(self.embedder.cache), 0)

    def test_counters_are_reported_for_health(self):
        run(self.embedder.embed_query("alpha"))
        run(self.embedder.embed_query("alpha"))
        stats = self.embedder.cache_stats()
        self.assertEqual(stats["enabled"], True)
        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["max"], 256)
        self.assertEqual((stats["hits"], stats["misses"]), (1, 1))
        self.assertEqual(stats["vector_dim"], DIM)
        self.assertEqual(stats["approx_bytes"], 1 * (DIM * 4 + 112))

    def test_approx_bytes_stays_at_the_bound(self):
        for i in range(40):
            run(self.embedder.embed_query("q" * (i + 2)))
        stats = self.embedder.cache_stats()
        bound = stats["max"] * (stats["vector_dim"] * 4 + 112)
        self.assertLessEqual(stats["entries"], stats["max"])
        self.assertLessEqual(stats["approx_bytes"], bound)


if __name__ == "__main__":
    unittest.main()
