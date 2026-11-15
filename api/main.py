# main.py
"""Embedding + retrieval service.

Internal service: the only HTTP client that talks to it is `gateway`.

Public (gateway-facing) endpoints:
    POST /search          - semantic search over the document collection
    POST /v1/embeddings   - OpenAI-compatible embeddings
    GET  /collection/info - collection statistics
    GET  /collection/sample - sample points
    GET  /health
"""

import base64
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [api] %(message)s",
)
log = logging.getLogger("api")

# Добавляем путь к embedder модулю (если нужно)
sys.path.append("/app")

# Импортируем embedder из соседнего сервиса (будет через volume или копирование)
try:
    from embedder import Embedder
except ImportError as exc:  # pragma: no cover
    if os.environ.get("ALLOW_EMBEDDER_FALLBACK", "false").lower() in {"1", "true", "yes"}:
        log.error("Embedder not found (%s) - using DUMMY embeddings", exc)

        # Fallback для локальных тестов без GPU/модели.
        class Embedder:
            def __init__(self, model_path, device="cpu", dtype=None, batch_size=32, normalize=True, max_seq_length=None):
                self.dim = 768
                log.warning("Dummy embedder initialized (dim=%d)", self.dim)

            async def embed(self, texts: List[str]):
                return np.random.randn(len(texts), self.dim).astype(np.float32)
    else:
        raise

# Настройки
MODEL_PATH = os.environ.get("MODEL_PATH", "google/embeddinggemma-300m")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "documents")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "rag-embeddings")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "256"))

# Устройство и точность: один экземпляр модели на всю систему (см. README)
EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cuda")
EMBEDDING_DTYPE = os.environ.get("EMBEDDING_DTYPE", "float16")
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_NORMALIZE = os.environ.get("EMBEDDING_NORMALIZE", "true").lower() in {"1", "true", "yes"}
EMBEDDING_MAX_SEQ_LENGTH = int(os.environ.get("EMBEDDING_MAX_SEQ_LENGTH", "0")) or None

# Глобальные объекты (инициализируются при старте)
embedder = None
qdrant_client = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global embedder, qdrant_client

    log.info("Connecting to Qdrant at %s:%s", QDRANT_HOST, QDRANT_PORT)
    qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    log.info("Loading embedder from %s", MODEL_PATH)
    embedder = Embedder(
        MODEL_PATH,
        device=EMBEDDING_DEVICE,
        dtype=EMBEDDING_DTYPE,
        batch_size=EMBEDDING_BATCH_SIZE,
        normalize=EMBEDDING_NORMALIZE,
        max_seq_length=EMBEDDING_MAX_SEQ_LENGTH,
    )

    log.info("Startup complete")
    yield


app = FastAPI(title="RAG Search API", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Qdrant compatibility helpers
# --------------------------------------------------------------------------- #


def _collection_exists() -> bool:
    exists = getattr(qdrant_client, "collection_exists", None)
    if callable(exists):
        return bool(exists(COLLECTION_NAME))
    try:
        qdrant_client.get_collection(COLLECTION_NAME)
        return True
    except Exception:
        return False


def _query(vector: List[float], limit: int, min_score: Optional[float]):
    """Run a nearest-neighbour search across qdrant-client versions.

    `QdrantClient.search` was removed in favour of `query_points`; keep both.
    """
    kwargs: Dict[str, Any] = {
        "collection_name": COLLECTION_NAME,
        "limit": limit,
        "with_payload": True,
    }
    if min_score is not None:
        kwargs["score_threshold"] = min_score

    query_points = getattr(qdrant_client, "query_points", None)
    if callable(query_points):
        return list(query_points(query=vector, **kwargs).points)
    return list(qdrant_client.search(query_vector=vector, **kwargs))


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #


async def _encode(texts: List[str]) -> np.ndarray:
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedding model is not loaded yet")
    return np.asarray(await embedder.embed(texts), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    min_score: Optional[float] = None


class SearchResult(BaseModel):
    score: float
    text: str
    file: str
    chunk: int


@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """Поиск релевантных документов по запросу."""
    if not _collection_exists():
        return []

    # 1. Получаем embedding для запроса
    query_vectors = await _encode([request.query])
    query_vector = query_vectors[0].astype(float).tolist()

    # 2. Ищем в Qdrant
    hits = _query(query_vector, request.top_k, request.min_score)

    # Дублируем фильтр локально, чтобы контракт /search не зависел от бэкенда
    if request.min_score is not None:
        hits = [hit for hit in hits if hit.score >= request.min_score]

    # 3. Формируем ответ
    response: List[SearchResult] = []
    for hit in hits:
        payload = hit.payload or {}
        response.append(
            SearchResult(
                score=float(hit.score),
                text=payload.get("text", ""),
                file=payload.get("file", "unknown"),
                chunk=payload.get("chunk", 0),
            )
        )

    return response


# --------------------------------------------------------------------------- #
# Internal encoding endpoint (used by the embedder service)
# --------------------------------------------------------------------------- #


class EmbedRequest(BaseModel):
    texts: List[str] = Field(min_length=1)


@app.post("/embed")
async def embed_texts(request: EmbedRequest):
    """Внутренний эндпоинт: батч текстов -> векторы.

    Единственный способ, которым `embedder` получает эмбеддинги, - поэтому
    модель загружена ровно один раз на всю систему.
    """
    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"'texts' contains {len(request.texts)} items, maximum is {MAX_BATCH_SIZE}",
        )
    if not all(isinstance(text, str) for text in request.texts):
        raise HTTPException(status_code=400, detail="'texts' must contain only strings")

    vectors = await _encode([text if text.strip() else " " for text in request.texts])

    return {
        "dim": int(vectors.shape[1]),
        "vectors": vectors.astype(float).tolist(),
    }


# --------------------------------------------------------------------------- #
# OpenAI-compatible embeddings
# --------------------------------------------------------------------------- #


class EmbeddingsRequest(BaseModel):
    input: Union[str, List[Any]]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"
    dimensions: Optional[int] = None
    user: Optional[str] = None


def _approx_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token); only used for the usage block."""
    return max(1, len(text) // 4)


@app.post("/v1/embeddings")
async def create_embeddings(request: EmbeddingsRequest):
    texts = [request.input] if isinstance(request.input, str) else list(request.input)

    if not texts:
        raise HTTPException(status_code=400, detail="'input' must not be empty")
    if len(texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"'input' contains {len(texts)} items, maximum is {MAX_BATCH_SIZE}",
        )
    if not all(isinstance(text, str) for text in texts):
        raise HTTPException(
            status_code=400,
            detail="'input' must contain only strings (token arrays are not supported)",
        )

    # Пустая строка ломает некоторые модели - подменяем пробелом
    texts = [text if text.strip() else " " for text in texts]

    vectors = await _encode(texts)

    if request.encoding_format == "base64":
        data = [
            {
                "object": "embedding",
                "index": index,
                "embedding": base64.b64encode(
                    vector.astype(np.float32).tobytes()
                ).decode(),
            }
            for index, vector in enumerate(vectors)
        ]
    else:
        data = [
            {
                "object": "embedding",
                "index": index,
                "embedding": vector.astype(float).tolist(),
            }
            for index, vector in enumerate(vectors)
        ]

    tokens = sum(_approx_tokens(text) for text in texts)

    return {
        "object": "list",
        "data": data,
        "model": request.model or EMBEDDING_MODEL_NAME,
        "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
    }


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


@app.get("/health")
async def health():
    try:
        collection_exists = _collection_exists() if qdrant_client is not None else False
    except Exception:  # Qdrant недоступен - health должен всё равно ответить
        collection_exists = False

    return {
        "status": "ok",
        "embedder": "ready" if embedder is not None else "not_loaded",
        "model_path": MODEL_PATH,
        "device": EMBEDDING_DEVICE,
        "dtype": EMBEDDING_DTYPE,
        "dim": getattr(embedder, "dim", None),
        "collection": COLLECTION_NAME,
        "collection_exists": collection_exists,
    }


@app.get("/collection/info")
async def collection_info():
    """Статистика коллекции: количество векторов, размерность и т.д."""
    try:
        if not _collection_exists():
            return {"collection_name": COLLECTION_NAME, "exists": False}

        info = qdrant_client.get_collection(COLLECTION_NAME)
        vectors_config = info.config.params.vectors

        return {
            "collection_name": COLLECTION_NAME,
            "exists": True,
            "points_count": info.points_count,
            "status": str(info.status),
            "vector_size": getattr(vectors_config, "size", None),
        }
    except Exception as exc:
        return {"error": str(exc), "collection_name": COLLECTION_NAME}


@app.get("/collection/sample")
async def collection_sample(limit: int = 5):
    """Несколько точек коллекции для проверки индексации."""
    try:
        if not _collection_exists():
            return {"total_points": 0, "points": []}

        points, _next_page = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        payload_points = []
        for point in points:
            payload = point.payload or {}
            text = payload.get("text", "")
            payload_points.append(
                {
                    "id": point.id,
                    "file": payload.get("file", "unknown"),
                    "chunk": payload.get("chunk", 0),
                    "text_preview": f"{text[:100]}..." if text else "",
                }
            )

        return {"total_points": len(payload_points), "points": payload_points}
    except Exception as exc:
        return {"error": str(exc)}
