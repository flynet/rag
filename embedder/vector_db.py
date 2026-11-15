# vector_db.py
import hashlib
import logging
from typing import Any, Dict, List, Optional

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
        """Создаём коллекцию, если её нет; пересоздаём при смене размерности."""
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
        chunks: List[str],
        vectors: np.ndarray,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """upsert chunks->vectors. vectors: numpy array shape (n, dim)

        `meta` is copied into every payload so the indexer can later tell whether
        a file is still up to date (mtime/size/chunking config/model).
        """
        if len(chunks) == 0:
            return

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
                    payload={"file": file_path, "chunk": i, "text": chunk, **extra},
                    vector=vec.tolist(),
                )
            )

        # batch upsert
        self.client.upsert(collection_name=self.collection_name, points=points)

    def delete_by_file(self, file_path: str) -> None:
        """Удаляем все точки, у которых payload.file == file_path"""
        flt = Filter(must=[FieldCondition(key="file", match=MatchValue(value=file_path))])
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=FilterSelector(filter=flt),
        )

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
        """True, если все поля signature совпадают с сохранёнными в payload."""
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
        """Проверяем, есть ли уже векторы для этого файла в базе"""
        try:
            return self._first_payload(file_path) is not None
        except Exception as exc:
            log.warning("Error checking file existence for %s: %s", file_path, exc)
            return False
