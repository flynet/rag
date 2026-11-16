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
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Union

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [api] %(message)s",
)
log = logging.getLogger("api")

from embedder import Embedder, SparseEmbedder  # noqa: E402
from hybrid import (  # noqa: E402
    DENSE_VECTOR,
    SPARSE_VECTOR,
    SEARCH_MODES,
    apply_threshold,
    filter_relative,
    fuse,
    normalize_hits,
)

# Model configuration: one model instance for the whole system (see README)
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen3-Embedding-0.6B")
QDRANT_HOST = os.environ.get("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "documents")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "rag-embeddings")
MAX_BATCH_SIZE = int(os.environ.get("MAX_BATCH_SIZE", "256"))

EMBEDDING_DEVICE = os.environ.get("EMBEDDING_DEVICE", "cuda")
EMBEDDING_DTYPE = os.environ.get("EMBEDDING_DTYPE", "float16")
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_NORMALIZE = os.environ.get("EMBEDDING_NORMALIZE", "true").lower() in {"1", "true", "yes"}
EMBEDDING_MAX_SEQ_LENGTH = int(os.environ.get("EMBEDDING_MAX_SEQ_LENGTH", "8192")) or None
# Query prompt name: Qwen3-Embedding has "query" with a retrieval instruction.
# An empty value encodes queries exactly like documents.
EMBEDDING_QUERY_PROMPT = os.environ.get("EMBEDDING_QUERY_PROMPT", "query").strip() or None
# How many encode calls may run at once. 1 serializes them on the GPU, which
# keeps VRAM and p95 latency predictable while indexing and chat share the model.
EMBEDDING_MAX_CONCURRENCY = max(1, int(os.environ.get("EMBEDDING_MAX_CONCURRENCY", "1")))

# BM25 sparse search (FastEmbed, CPU). The model is tiny and adds no GPU load.
SPARSE_ENABLED = os.environ.get("SPARSE_ENABLED", "true").lower() in {"1", "true", "yes"}
SPARSE_MODEL = os.environ.get("SPARSE_MODEL", "Qdrant/bm25")
BM25_THREADS = int(os.environ.get("BM25_THREADS", "0")) or None

# "hybrid" runs dense + BM25 and fuses with RRF; "dense" is the old behaviour.
# A collection without sparse vectors always falls back to dense (see /search).
SEARCH_MODE = os.environ.get("SEARCH_MODE", "hybrid").strip().lower()
if SEARCH_MODE not in SEARCH_MODES:
    log.warning("Unknown SEARCH_MODE %r, using 'hybrid'", SEARCH_MODE)
    SEARCH_MODE = "hybrid"

# Weight of the BM25 branch in RRF (the dense branch is fixed at 1.0). Below 1.0
# a BM25-only hit cannot outrank a chunk the dense branch placed near the top.
SPARSE_WEIGHT = float(os.environ.get("SPARSE_WEIGHT", "1.0"))

# Drop BM25 hits below this fraction of the best BM25 score for the query. A weak
# term overlap (one shared common word) is ranking noise; 0.0 keeps every sparse
# hit.
SPARSE_MIN_RELATIVE = float(os.environ.get("SPARSE_MIN_RELATIVE", "0.0"))

# Global objects (initialized at startup)
embedder = None
sparse_embedder = None
qdrant_client = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global embedder, sparse_embedder, qdrant_client

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
        query_prompt=EMBEDDING_QUERY_PROMPT,
        max_concurrency=EMBEDDING_MAX_CONCURRENCY,
    )

    if SPARSE_ENABLED:
        try:
            sparse_embedder = SparseEmbedder(SPARSE_MODEL, threads=BM25_THREADS)
        except Exception as exc:  # noqa: BLE001 - dense search must keep working
            log.error("BM25 model failed to load, search falls back to dense-only: %s", exc)
            sparse_embedder = None
    else:
        log.info("BM25 sparse search is disabled (SPARSE_ENABLED=false)")

    log.info("Startup complete (search mode: %s)", SEARCH_MODE)
    yield


app = FastAPI(title="RAG Search API", version="2.1.0", lifespan=lifespan)


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


def _query(vector: List[float], limit: int, min_score: Optional[float], using: Optional[str] = None):
    """Run a nearest-neighbour search across qdrant-client versions.

    `QdrantClient.search` was removed in favour of `query_points`; keep both.
    `using` selects a named vector (`dense`), which the hybrid schema needs.
    """
    kwargs: Dict[str, Any] = {
        "collection_name": COLLECTION_NAME,
        "limit": limit,
        "with_payload": True,
    }
    if min_score is not None:
        kwargs["score_threshold"] = min_score
    if using:
        kwargs["using"] = using

    query_points = getattr(qdrant_client, "query_points", None)
    if callable(query_points):
        return list(query_points(query=vector, **kwargs).points)
    return list(qdrant_client.search(query_vector=vector, **kwargs))


def _query_sparse(sparse: Dict[str, Any], limit: int):
    """BM25 sparse search: dot product of the query terms against stored weights."""
    query_points = getattr(qdrant_client, "query_points", None)
    if not callable(query_points):
        return []
    points = query_points(
        query=SparseVector(indices=sparse["indices"], values=sparse["values"]),
        collection_name=COLLECTION_NAME,
        limit=limit,
        using=SPARSE_VECTOR,
        with_payload=True,
    )
    return list(points.points)


# The schema decides what a request can do, and `POST /search` runs on every
# request: reading it once per 10 seconds is enough to notice a re-index.
SCHEMA_TTL = 10.0
_schema_cache: Dict[str, Any] = {"names": {}, "checked": None}


def _vector_names() -> Dict[str, str]:
    """`{vector_name: "dense" | "sparse"}` of the collection, cached for `SCHEMA_TTL`.

    A collection written before the hybrid schema has one unnamed vector, so it
    has no `dense` name to query and no `bm25` sparse vector.
    """
    if _schema_cache["checked"] is not None and (time.monotonic() - _schema_cache["checked"]) < SCHEMA_TTL:
        return _schema_cache["names"]

    names: Dict[str, str] = {}
    try:
        info = qdrant_client.get_collection(COLLECTION_NAME)
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            names.update({key: "dense" for key in vectors})
        sparse = info.config.params.sparse_vectors
        if isinstance(sparse, dict):
            names.update({key: "sparse" for key in sparse})
    except Exception:  # Qdrant down / collection missing
        names = {}

    _schema_cache["names"] = names
    _schema_cache["checked"] = time.monotonic()
    return names


def _sparse_available() -> bool:
    """Does the collection carry `bm25` sparse vectors?"""
    return _vector_names().get(SPARSE_VECTOR) == "sparse"


def _dense_using() -> Optional[str]:
    """The named dense vector, or `None` for a pre-hybrid collection.

    A collection with one unnamed vector cannot be queried with
    `using="dense"`, so the name is passed only when the schema has it.
    """
    return DENSE_VECTOR if _vector_names().get(DENSE_VECTOR) == "dense" else None


# --------------------------------------------------------------------------- #
# Encoding helpers
# --------------------------------------------------------------------------- #


async def _encode(texts: List[str], query: bool = False) -> np.ndarray:
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedding model is not loaded yet")
    if query:
        vectors = await embedder.embed_query(texts)
    else:
        vectors = await embedder.embed(texts)
    return np.asarray(vectors, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=100)
    min_score: Optional[float] = None
    # "hybrid" (dense + BM25, RRF) or "dense" (embeddings only). Falls back to
    # dense when the collection has no sparse vectors.
    mode: Optional[str] = None
    # RRF weight of the BM25 branch (the dense branch is fixed at 1.0). Tuning
    # knob for `scripts/eval_retrieval.py`; `SPARSE_WEIGHT` is the service default.
    sparse_weight: Optional[float] = Field(default=None, ge=0.0, le=10.0)
    # Keep BM25 hits within this fraction of the best BM25 score. Tuning knob for
    # `scripts/eval_retrieval.py`; `SPARSE_MIN_RELATIVE` is the service default.
    sparse_min_relative: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    score: float
    text: str
    file: str
    chunk: int
    title: Optional[str] = None
    section: Optional[str] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    fusion_score: Optional[float] = None
    match: Optional[str] = None


def _result(payload: Dict[str, Any], score: float, fusion_score: Optional[float], match: Optional[str]):
    return SearchResult(
        score=float(score),
        text=payload.get("text", ""),
        file=payload.get("file", "unknown"),
        chunk=payload.get("chunk", 0),
        title=payload.get("title"),
        section=payload.get("section"),
        char_start=payload.get("char_start"),
        char_end=payload.get("char_end"),
        fusion_score=float(fusion_score) if fusion_score is not None else None,
        match=match,
    )


def _search_dense(query_vector: List[float], request: SearchRequest) -> List[SearchResult]:
    hits = _query(query_vector, request.top_k, request.min_score, using=_dense_using())
    # Filter locally as well, so the /search contract does not depend on the backend
    if request.min_score is not None:
        hits = [hit for hit in hits if hit.score >= request.min_score]
    return [_result(hit.payload or {}, hit.score, None, "dense") for hit in hits]


async def _search_hybrid(query_vector: List[float], request: SearchRequest) -> List[SearchResult]:
    # Each branch returns more candidates than `top_k`, so RRF has material to
    # rank: a chunk ranked 12th by BM25 can enter the final top 8 once fused.
    candidates = min(100, max(request.top_k, request.top_k * 2))

    dense_hits = normalize_hits(
        _query(query_vector, candidates, request.min_score, using=_dense_using())
    )

    sparse_query = (await sparse_embedder.embed_query([request.query]))[0]
    ratio = request.sparse_min_relative if request.sparse_min_relative is not None else SPARSE_MIN_RELATIVE
    sparse_hits = filter_relative(normalize_hits(_query_sparse(sparse_query, candidates)), ratio)

    weight = request.sparse_weight if request.sparse_weight is not None else SPARSE_WEIGHT
    fused = apply_threshold(
        fuse(dense_hits, sparse_hits, request.top_k, sparse_weight=weight),
        request.min_score,
    )

    results: List[SearchResult] = []
    for hit in fused:
        ranks = hit["ranks"]
        match = "both" if "dense" in ranks and "sparse" in ranks else ("dense" if "dense" in ranks else "sparse")
        # `score` stays a relevance number the gateway and the eval harness can
        # filter on: cosine when the dense branch found the chunk, otherwise the
        # BM25 weight of the exact-term match. `fusion_score` is the RRF rank
        # score, which has no cosine meaning on its own.
        score = hit["dense_score"] if hit["dense_score"] is not None else hit["sparse_score"]
        results.append(_result(hit["payload"], score, hit["fusion_score"], match))

    return results


@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """Document search over the collection: dense, or dense + BM25 fused with RRF."""
    if not _collection_exists():
        return []

    mode = (request.mode or SEARCH_MODE).strip().lower()
    if mode not in SEARCH_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"'mode' must be one of: {', '.join(SEARCH_MODES)}",
        )

    hybrid = mode == "hybrid"
    if hybrid and sparse_embedder is None:
        log.info("BM25 model is not loaded - dense-only search for this request")
        hybrid = False
    if hybrid and not _sparse_available():
        log.info("Collection has no '%s' sparse vectors - dense-only search", SPARSE_VECTOR)
        hybrid = False

    # Encode the query with the model's retrieval instruction
    # (Qwen3-Embedding: prompt_name="query").
    query_vectors = await _encode([request.query], query=True)
    query_vector = query_vectors[0].astype(float).tolist()

    if hybrid:
        return await _search_hybrid(query_vector, request)
    return _search_dense(query_vector, request)


# --------------------------------------------------------------------------- #
# Internal encoding endpoint (used by the embedder service)
# --------------------------------------------------------------------------- #


class EmbedRequest(BaseModel):
    texts: List[str] = Field(min_length=1)


@app.post("/embed")
async def embed_texts(request: EmbedRequest):
    """Internal endpoint: a batch of texts -> vectors.

    This is the only way `embedder` gets embeddings, so the model is loaded
    exactly once for the whole system.
    """
    if len(request.texts) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"'texts' contains {len(request.texts)} items, maximum is {MAX_BATCH_SIZE}",
        )
    if not all(isinstance(text, str) for text in request.texts):
        raise HTTPException(status_code=400, detail="'texts' must contain only strings")

    cleaned = [text if text.strip() else " " for text in request.texts]
    vectors = await _encode(cleaned)

    body: Dict[str, Any] = {
        "dim": int(vectors.shape[1]),
        "vectors": vectors.astype(float).tolist(),
    }
    if sparse_embedder is not None:
        # BM25 weights per text, in the same order: the indexer stores them as
        # the `bm25` named sparse vector of each chunk.
        body["sparse"] = await sparse_embedder.embed(cleaned)

    return body


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

    # An empty string breaks some models - replace it with a space
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
    except Exception:  # Qdrant is down - /health still answers
        collection_exists = False

    return {
        "status": "ok",
        "embedder": "ready" if embedder is not None else "not_loaded",
        "model_path": MODEL_PATH,
        "device": EMBEDDING_DEVICE,
        "dtype": EMBEDDING_DTYPE,
        "dim": getattr(embedder, "dim", None),
        "query_prompt": getattr(embedder, "query_prompt", EMBEDDING_QUERY_PROMPT),
        "max_concurrency": getattr(embedder, "max_concurrency", EMBEDDING_MAX_CONCURRENCY),
        "sparse_enabled": sparse_embedder is not None,
        "sparse_model": SPARSE_MODEL if sparse_embedder is not None else None,
        "search_mode": SEARCH_MODE,
        "collection": COLLECTION_NAME,
        "collection_exists": collection_exists,
    }


@app.get("/collection/info")
async def collection_info():
    """Collection statistics: number of vectors, dimension, and so on."""
    try:
        if not _collection_exists():
            return {"collection_name": COLLECTION_NAME, "exists": False}

        info = qdrant_client.get_collection(COLLECTION_NAME)
        vectors_config = info.config.params.vectors
        if isinstance(vectors_config, dict):
            dense = vectors_config.get(DENSE_VECTOR)
            vector_size = getattr(dense, "size", None)
            named = sorted(vectors_config)
        else:
            vector_size = getattr(vectors_config, "size", None)
            named = []

        sparse = info.config.params.sparse_vectors
        return {
            "collection_name": COLLECTION_NAME,
            "exists": True,
            "points_count": info.points_count,
            "status": str(info.status),
            "vector_size": vector_size,
            "vectors": named,
            "sparse_vectors": sorted(sparse) if isinstance(sparse, dict) else [],
        }
    except Exception as exc:
        return {"error": str(exc), "collection_name": COLLECTION_NAME}


@app.get("/collection/sample")
async def collection_sample(limit: int = 5):
    """A few collection points, to check what indexing produced."""
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
                    "title": payload.get("title"),
                    "section": payload.get("section"),
                    "char_start": payload.get("char_start"),
                    "char_end": payload.get("char_end"),
                    "text_preview": f"{text[:100]}..." if text else "",
                }
            )

        return {"total_points": len(payload_points), "points": payload_points}
    except Exception as exc:
        return {"error": str(exc)}
