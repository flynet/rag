# vector_db.py
import hashlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    KeywordIndexParams,
    MatchValue,
    Modifier,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

from chunking import TextChunk

log = logging.getLogger("embedder.db")

# Named vectors: `dense` is the embedding model, `bm25` is the sparse BM25
# representation. Named vectors are required for hybrid search, because a query
# has to address each branch by name (`using="dense"` / `using="bm25"`).
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


class VectorDB:
    def __init__(self, host="qdrant", port=6333, collection_name="documents"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    # ------------------------------------------------------------------ #
    # Collection
    # ------------------------------------------------------------------ #

    def _collection_info(self):
        try:
            return self.client.get_collection(self.collection_name)
        except Exception:
            return None

    def _schema_ok(self, dim: int, sparse_enabled: bool) -> Tuple[bool, str]:
        """Does the collection match the named-vector schema hybrid search needs?"""
        info = self._collection_info()
        if info is None:
            return (False, "collection is missing")

        vectors = info.config.params.vectors
        if not isinstance(vectors, dict):
            return (False, "collection stores one unnamed vector (pre-hybrid schema)")

        dense = vectors.get(DENSE_VECTOR)
        if dense is None:
            return (False, f"collection has no '{DENSE_VECTOR}' vector")

        existing = getattr(dense, "size", None)
        if existing != dim:
            return (
                False,
                f"collection dense dim={existing} but the model produces dim={dim} "
                "(EMBEDDING_MODEL changed?)",
            )

        if sparse_enabled:
            sparse = info.config.params.sparse_vectors
            if not isinstance(sparse, dict) or SPARSE_VECTOR not in sparse:
                return (False, f"collection has no '{SPARSE_VECTOR}' sparse vector (hybrid schema)")

        return (True, "ok")

    def ensure_collection(
        self,
        dim: int,
        sparse_enabled: bool = True,
        distance: Distance = Distance.COSINE,
    ) -> None:
        """Create the collection, recreating it when the schema does not fit.

        The pre-hybrid collection (one unnamed vector) cannot hold BM25 sparse
        vectors, so switching to hybrid is a breaking change: the collection is
        dropped and every watched file is re-indexed by the startup scan. The
        source documents on disk are the source of truth, so that is safe; a
        snapshot taken before the change (see `scripts/qdrant_snapshot.py`) is
        the rollback path.
        """
        info = self._collection_info()
        ok, reason = self._schema_ok(dim, sparse_enabled)
        if ok:
            return

        if info is not None:
            log.error(
                "Collection '%s' does not fit the hybrid schema: %s - recreating it, "
                "all watched documents will be re-indexed",
                self.collection_name,
                reason,
            )
            self.client.delete_collection(self.collection_name)

        sparse_config = (
            {SPARSE_VECTOR: SparseVectorParams(modifier=Modifier.IDF)} if sparse_enabled else None
        )
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={DENSE_VECTOR: VectorParams(size=dim, distance=distance)},
            sparse_vectors_config=sparse_config,
        )
        self._ensure_payload_index("file")
        log.info(
            "Created collection '%s' (named dense dim=%d, sparse=%s)",
            self.collection_name,
            dim,
            SPARSE_VECTOR if sparse_enabled else "disabled",
        )

    def _ensure_payload_index(self, field: str) -> None:
        """A keyword index on `field` so `file` / `section` filters are cheap."""
        try:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field,
                field_schema=KeywordIndexParams(),
            )
        except Exception as exc:
            # The index is an optimization; Qdrant filters correctly without it.
            log.warning("Could not create a payload index on %r: %s", field, exc)

    # ------------------------------------------------------------------ #
    # Points
    # ------------------------------------------------------------------ #

    def _generate_id(self, file_path: str, chunk_index: int) -> int:
        """Generate unique integer ID from file path and chunk index"""
        text = f"{file_path}::chunk::{chunk_index}"
        hash_obj = hashlib.md5(text.encode())
        # Convert first 8 bytes to int (positive)
        return int.from_bytes(hash_obj.digest()[:8], "big")

    def upsert(
        self,
        file_path: str,
        chunks: List[TextChunk],
        vectors: np.ndarray,
        meta: Optional[Dict[str, Any]] = None,
        sparse: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """upsert chunks->vectors. vectors: numpy array shape (n, dim)

        `meta` is copied into every payload so the indexer can later tell whether
        a file is still up to date (mtime/size/chunking config/model). Chunk
        metadata (section and character offsets) is stored alongside the text.

        `sparse` is the per-chunk BM25 sparse vector, `{"indices":[...],
        "values":[...]}`. When it is missing (an empty list: `api` has BM25
        disabled), the point is written with the dense vector only: a chunk
        without a sparse vector is found by semantic search but not by exact-term
        search, which is the degradation the sparse branch is designed for.
        """
        if len(chunks) == 0:
            return
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(vectors)} vectors for {file_path}"
            )
        # An empty `sparse` list means "no BM25 for this run", not a partial list,
        # so it is not a length mismatch.
        if sparse and len(sparse) != len(chunks):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(sparse)} sparse vectors for {file_path}"
            )

        dim = vectors.shape[1]
        sparse_enabled = bool(sparse)
        # ensure collection
        self.ensure_collection(dim=dim, sparse_enabled=sparse_enabled)

        extra = dict(meta or {})
        points = []
        # Use hash-based integer ID
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            pid = self._generate_id(file_path, i)
            named = {DENSE_VECTOR: vec.tolist()}
            if sparse_enabled and sparse[i]:
                named[SPARSE_VECTOR] = SparseVector(
                    indices=sparse[i]["indices"], values=sparse[i]["values"]
                )
            points.append(
                PointStruct(
                    id=pid,
                    payload={
                        **extra,
                        "file": file_path,
                        "chunk": i,
                        "text": chunk.text,
                        "section": chunk.section,
                        "char_start": chunk.char_start,
                        "char_end": chunk.char_end,
                    },
                    vector=named,
                )
            )

        # batch upsert
        self.client.upsert(collection_name=self.collection_name, points=points)

    def delete_by_file(self, file_path: str) -> None:
        """Delete every point whose payload.file equals file_path."""
        flt = Filter(must=[FieldCondition(key="file", match=MatchValue(value=file_path))])
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=flt),
        )

    # ------------------------------------------------------------------ #
    # Startup reconciliation
    # ------------------------------------------------------------------ #

    def reconcile_files(self, root: Path, batch_size: int = 256) -> Tuple[int, int]:
        """Remove vectors of files that are no longer present under `root`.

        The watcher only sees changes while this process is running. Files
        deleted or renamed while the stack was down would otherwise stay in
        Qdrant and keep appearing in search results. Returns
        ``(number of files, number of chunks)`` removed.
        """
        root = root.resolve()
        counts: Dict[str, int] = {}

        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=["file"],
                with_vectors=False,
            )
            for point in points:
                payload = point.payload or {}
                file_path = payload.get("file")
                if isinstance(file_path, str) and file_path:
                    counts[file_path] = counts.get(file_path, 0) + 1
            if offset is None:
                break

        missing = {
            file_path: count
            for file_path, count in counts.items()
            if not self._is_indexed_file(file_path, root)
        }
        for file_path in missing:
            self.delete_by_file(file_path)

        return len(missing), sum(missing.values())

    @staticmethod
    def _is_indexed_file(file_path: str, root: Path) -> bool:
        """True when `file_path` is an existing file under the watched root."""
        try:
            path = Path(file_path)
            if not path.is_file():
                return False
            return path.resolve().is_relative_to(root)
        except (OSError, RuntimeError):
            return False

    # ------------------------------------------------------------------ #
    # Change detection
    # ------------------------------------------------------------------ #

    def _first_payload(self, file_path: str) -> Optional[Dict[str, Any]]:
        flt = Filter(must=[FieldCondition(key="file", match=MatchValue(value=file_path))])
        points, _next_page = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=flt,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        if not points:
            return None
        return points[0].payload or {}

    def is_up_to_date(self, file_path: str, signature: Dict[str, Any]) -> bool:
        """True if every signature field matches what is stored in the payload."""
        if not signature:
            return False
        try:
            payload = self._first_payload(file_path)
        except Exception as exc:
            log.warning("Cannot read index state for %s: %s", file_path, exc)
            return False

        if payload is None:
            return False

        return all(payload.get(key) == value for key, value in signature.items())

    def file_exists(self, file_path: str) -> bool:
        """Whether the database already has vectors for this file."""
        try:
            return self._first_payload(file_path) is not None
        except Exception as exc:
            log.warning("Error checking file existence for %s: %s", file_path, exc)
            return False
