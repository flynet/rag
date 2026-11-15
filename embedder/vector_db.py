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
    MatchValue,
    PointStruct,
    VectorParams,
)

from chunking import TextChunk

log = logging.getLogger("embedder.db")


class VectorDB:
    def __init__(self, host="qdrant", port=6333, collection_name="documents"):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    # ------------------------------------------------------------------ #
    # Collection
    # ------------------------------------------------------------------ #

    def _collection_dim(self) -> Optional[int]:
        try:
            info = self.client.get_collection(self.collection_name)
        except Exception:
            return None
        return getattr(info.config.params.vectors, "size", None)

    def ensure_collection(self, dim: int, distance: Distance = Distance.COSINE) -> None:
        """Create the collection if it is missing; recreate it when the dimension changes."""
        existing = self._collection_dim()

        if existing is None:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=distance),
            )
            log.info("Created collection '%s' (dim=%d)", self.collection_name, dim)
            return

        if existing != dim:
            log.error(
                "Collection '%s' uses dim=%s but the model produces dim=%d "
                "(probably EMBEDDING_MODEL changed) - recreating the collection, "
                "all documents will be re-indexed",
                self.collection_name,
                existing,
                dim,
            )
            self.client.delete_collection(self.collection_name)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=distance),
            )
            log.info("Recreated collection '%s' (dim=%d)", self.collection_name, dim)

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
    ) -> None:
        """upsert chunks->vectors. vectors: numpy array shape (n, dim)

        `meta` is copied into every payload so the indexer can later tell whether
        a file is still up to date (mtime/size/chunking config/model). Chunk
        metadata (section and character offsets) is stored alongside the text.
        """
        if len(chunks) == 0:
            return
        if len(chunks) != len(vectors):
            raise ValueError(
                f"Got {len(chunks)} chunks but {len(vectors)} vectors for {file_path}"
            )

        dim = vectors.shape[1]
        # ensure collection
        self.ensure_collection(dim=dim)

        extra = dict(meta or {})
        points = []
        # Use hash-based integer ID
        for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
            pid = self._generate_id(file_path, i)
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
                    vector=vec.tolist(),
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
