"""Collection-schema tests for the indexer (stdlib only).

`embedder/vector_db.py` imports `qdrant_client`, which is not installed on the
host (it lives in the containers). The stub installed here is the minimum the
module touches: a fake client that records `create_collection`,
`delete_collection`, `create_payload_index` and `upsert` calls, and a collection
description with the same shape Qdrant reports.

Run from the repository root:

    python -m unittest discover -s tests -p "test_*.py"
"""

import numpy as np
import sys
import types
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "embedder"))

_client = None


def _install_qdrant_stub():
    """Register a fake `qdrant_client` before `vector_db` is imported."""
    package = types.ModuleType("qdrant_client")
    models = types.ModuleType("qdrant_client.models")

    class Distance:
        COSINE = "cosine"

    class Modifier:
        IDF = "idf"

    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    models.Distance = Distance
    models.Modifier = Modifier
    for name in (
        "VectorParams",
        "SparseVectorParams",
        "SparseVector",
        "PointStruct",
        "KeywordIndexParams",
        "FieldCondition",
        "Filter",
        "FilterSelector",
        "MatchValue",
    ):
        setattr(models, name, Params)

    def QdrantClient(*args, **kwargs):
        return _client

    package.QdrantClient = QdrantClient
    package.models = models
    sys.modules["qdrant_client"] = package
    sys.modules["qdrant_client.models"] = models


_install_qdrant_stub()

import vector_db  # noqa: E402


def collection(vectors, sparse=None):
    """A collection description shaped like Qdrant's `get_collection` answer."""
    return types.SimpleNamespace(
        config=types.SimpleNamespace(
            params=types.SimpleNamespace(vectors=vectors, sparse_vectors=sparse)
        )
    )


def named(dim=1024, sparse=True):
    vectors = {"dense": types.SimpleNamespace(size=dim)}
    sparse_config = {"bm25": types.SimpleNamespace(modifier="idf")} if sparse else None
    return collection(vectors, sparse_config)


def unnamed(dim=1024):
    """The pre-hybrid schema: one vector with no name."""
    return collection(types.SimpleNamespace(size=dim), None)


class FakeClient:
    def __init__(self):
        self.collection = None
        self.calls = []
        self.points = []

    def get_collection(self, collection_name):
        if self.collection is None:
            raise ValueError("collection is missing")
        return self.collection

    def create_collection(self, collection_name, vectors_config, sparse_vectors_config=None):
        self.calls.append(("create", vectors_config, sparse_vectors_config))
        self.collection = collection(vectors_config, sparse_vectors_config)

    def delete_collection(self, collection_name):
        self.calls.append(("delete", collection_name))
        self.collection = None

    def create_payload_index(self, collection_name, field_name, field_schema):
        self.calls.append(("index", field_name))

    def upsert(self, collection_name, points):
        self.points.extend(points)


def chunk(index=0, text="chunk text", section="Section"):
    return vector_db.TextChunk(
        text=text, section=section, char_start=index * 10, char_end=index * 10 + len(text)
    )


def vectors(rows, dim=1024):
    """A (n, dim) float array, the shape `upsert` takes."""
    return np.asarray([[value for value in row] + [0.0] * (dim - len(row)) for row in rows])


class SchemaTests(unittest.TestCase):
    def setUp(self):
        global _client
        _client = FakeClient()
        self.client = _client
        self.db = vector_db.VectorDB()

    def created(self):
        creates = [c for c in self.client.calls if c[0] == "create"]
        return creates[-1] if creates else None

    def test_missing_collection_is_created_with_named_vectors(self):
        self.db.ensure_collection(dim=1024)
        vectors, sparse = self.created()[1], self.created()[2]
        self.assertEqual(list(vectors), ["dense"])
        self.assertEqual(vectors["dense"].size, 1024)
        self.assertEqual(list(sparse), ["bm25"])
        # A payload index on `file` keeps the per-file filters cheap.
        self.assertIn(("index", "file"), self.client.calls)

    def test_legacy_unnamed_collection_is_recreated(self):
        self.client.collection = unnamed()
        self.db.ensure_collection(dim=1024)
        self.assertIn(("delete", "documents"), self.client.calls)
        vectors = self.created()[1]
        self.assertEqual(list(vectors), ["dense"])
        self.assertEqual(list(self.created()[2]), ["bm25"])

    def test_matching_hybrid_schema_is_untouched(self):
        self.client.collection = named()
        self.db.ensure_collection(dim=1024)
        self.assertEqual(self.client.calls, [])

    def test_missing_sparse_vector_triggers_reindex(self):
        self.client.collection = named(sparse=False)
        self.db.ensure_collection(dim=1024, sparse_enabled=True)
        self.assertIn(("delete", "documents"), self.client.calls)

    def test_bm25_disabled_keeps_a_dense_only_collection(self):
        self.client.collection = named(sparse=False)
        self.db.ensure_collection(dim=1024, sparse_enabled=False)
        self.assertEqual(self.client.calls, [])

    def test_dimension_change_recreates_the_collection(self):
        self.client.collection = named(dim=768)
        self.db.ensure_collection(dim=1024)
        self.assertIn(("delete", "documents"), self.client.calls)
        self.assertEqual(self.created()[1]["dense"].size, 1024)


class UpsertTests(unittest.TestCase):
    def setUp(self):
        global _client
        _client = FakeClient()
        self.client = _client
        self.db = vector_db.VectorDB()

    def test_both_named_vectors_are_written(self):
        self.db.upsert(
            "/data/documents/a.txt",
            [chunk()],
            vectors([[0.1, 0.2] * 512]),
            sparse=[{"indices": [10, 20], "values": [1.0, 2.0]}],
        )
        point = self.client.points[0]
        self.assertEqual(sorted(point.vector), ["bm25", "dense"])
        self.assertEqual(len(point.vector["dense"]), 1024)
        self.assertEqual(point.vector["bm25"].indices, [10, 20])

    def test_missing_sparse_vector_is_dense_only(self):
        self.db.upsert("/data/documents/a.txt", [chunk()], vectors([[0.1] * 1024]), sparse=[])
        point = self.client.points[0]
        self.assertEqual(list(point.vector), ["dense"])

    def test_sparse_count_mismatch_is_a_hard_error(self):
        with self.assertRaises(ValueError):
            self.db.upsert(
                "/data/documents/a.txt",
                [chunk(), chunk(1)],
                vectors([[0.1] * 1024, [0.2] * 1024]),
                sparse=[{"indices": [1], "values": [1.0]}],
            )

    def test_payload_keeps_citation_and_signature_fields(self):
        meta = {"mtime": 1, "size": 2, "model": "Qwen/Qwen3-Embedding-0.6B", "title": "T"}
        self.db.upsert(
            "/data/documents/a.txt",
            [chunk()],
            vectors([[0.1] * 1024]),
            sparse=[{"indices": [5], "values": [1.0]}],
            meta=meta,
        )
        payload = self.client.points[0].payload
        self.assertEqual(payload["file"], "/data/documents/a.txt")
        self.assertEqual(payload["chunk"], 0)
        self.assertEqual(payload["section"], "Section")
        self.assertEqual(payload["char_start"], 0)
        self.assertEqual(payload["char_end"], 10)
        self.assertEqual(payload["model"], "Qwen/Qwen3-Embedding-0.6B")

    def test_ids_are_stable_for_the_same_file_and_chunk(self):
        ids = []
        for _ in range(2):
            self.db.upsert("/data/documents/a.txt", [chunk()], vectors([[0.1] * 1024]), sparse=[])
            ids.append(self.client.points[-1].id)
        self.assertEqual(ids[0], ids[1])

        # A second chunk of the same file is a different point.
        self.db.upsert(
            "/data/documents/a.txt",
            [chunk(), chunk(1)],
            vectors([[0.1] * 1024, [0.2] * 1024]),
            sparse=[],
        )
        chunk_ids = [p.id for p in self.client.points[-2:]]
        self.assertNotEqual(chunk_ids[0], chunk_ids[1])
        self.assertEqual(chunk_ids[0], ids[0])


if __name__ == "__main__":
    unittest.main()
