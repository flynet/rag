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

import asyncio
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
from rerank import (  # noqa: E402
    RERANK_CANDIDATE_CAP,
    candidate_limit,
    candidate_passages,
    pair_query,
    rerank_results,
    resolve_rerank,
    resolve_rerank_instruct,
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
# Distinct query embeddings kept for reuse, by entry count. A query vector is a pure
# function of (model, prompt, text), so a repeated question costs no forward pass.
# 0 disables the cache, which is what a deployment without memory headroom uses.
EMBEDDING_CACHE_SIZE = max(0, int(os.environ.get("EMBEDDING_CACHE_SIZE", "256")))

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

# Cross-encoder reranking (P1.1). Off by default: reranking `RERANK_TOP_N`
# candidates costs more GPU work than embedding one query, so it is enabled per
# deployment after measuring MRR, p50/p95, and VRAM on the eval set. A reranker
# that fails to load, times out, or answers with an unusable shape leaves the
# fused order untouched, so enabling it can never break search.
RERANK_ENABLED = os.environ.get("RERANK_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
RERANK_MODEL = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3").strip()
# Same GPU and dtype as the embedding model by default: a 0.6B/0.5B cross-encoder
# in fp16 fits next to Qwen3-Embedding-0.6B on an 8 GB card (measured on the
# live stack before enabling reranking by default).
RERANK_DEVICE = os.environ.get("RERANK_DEVICE", EMBEDDING_DEVICE).strip() or EMBEDDING_DEVICE
RERANK_DTYPE = os.environ.get("RERANK_DTYPE", EMBEDDING_DTYPE).strip() or EMBEDDING_DTYPE
# Token window per (query, passage) pair. Chunks are 1024 characters (~250
# tokens) plus the title/section prefix, so 512 covers a chunk with headroom.
RERANK_MAX_LENGTH = int(os.environ.get("RERANK_MAX_LENGTH", "512"))
# Character budget of the passage text before tokenization (the chunk size).
RERANK_MAX_CHARS = int(os.environ.get("RERANK_MAX_CHARS", "1024"))
# Candidates the cross-encoder sees per request. The P1.1 scope asks for 30-50.
RERANK_TOP_N = max(1, min(RERANK_CANDIDATE_CAP, int(os.environ.get("RERANK_TOP_N", "30"))))
# Requests per second budget: 1 serializes predictions on the GPU.
RERANK_BATCH_SIZE = max(1, int(os.environ.get("RERANK_BATCH_SIZE", "16")))
RERANK_MAX_CONCURRENCY = max(1, int(os.environ.get("RERANK_MAX_CONCURRENCY", "1")))
# Time a rerank request may spend on the model. Overrunning it returns the fused
# list, so a slow model degrades quality, never the request.
RERANK_TIMEOUT = float(os.environ.get("RERANK_TIMEOUT", "10.0"))
# raw | sigmoid | softmax - see api/reranker.py
RERANK_TRANSFORM = os.environ.get("RERANK_TRANSFORM", "raw").strip().lower()
# Instruction in front of the query: `qwen3` is the wording the Qwen3-Reranker
# family is trained on, any other text is the instruction itself, empty = plain
# pairs. A pair model trained with a prompt scores low without it, so this is a
# quality knob: the measured probe on plain pairs for Qwen3-Reranker-0.6B was
# `-2.0625` for a pair that is obviously a match.
RERANK_QUERY_INSTRUCT = resolve_rerank_instruct(os.environ.get("RERANK_QUERY_INSTRUCT", ""))

# Global objects (initialized at startup)
embedder = None
sparse_embedder = None
qdrant_client = None
reranker = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global embedder, sparse_embedder, qdrant_client, reranker

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
        cache_size=EMBEDDING_CACHE_SIZE,
    )

    if SPARSE_ENABLED:
        try:
            sparse_embedder = SparseEmbedder(SPARSE_MODEL, threads=BM25_THREADS)
        except Exception as exc:  # noqa: BLE001 - dense search must keep working
            log.error("BM25 model failed to load, search falls back to dense-only: %s", exc)
            sparse_embedder = None
    else:
        log.info("BM25 sparse search is disabled (SPARSE_ENABLED=false)")

    if RERANK_ENABLED:
        # Imported here so the service starts without the reranker installed.
        from reranker import Reranker

        try:
            reranker = Reranker(
                RERANK_MODEL,
                device=RERANK_DEVICE,
                dtype=RERANK_DTYPE,
                max_length=RERANK_MAX_LENGTH,
                batch_size=RERANK_BATCH_SIZE,
                transform=RERANK_TRANSFORM,
                max_concurrency=RERANK_MAX_CONCURRENCY,
                query_instruct=RERANK_QUERY_INSTRUCT,
            )
        except Exception as exc:  # noqa: BLE001 - the fused order is a valid answer
            log.error("Reranker failed to load, search keeps the fused order: %s", exc)
            reranker = None

    log.info(
        "Startup complete (search mode: %s, rerank: %s)",
        SEARCH_MODE,
        "on" if reranker is not None else "off",
    )
    yield


app = FastAPI(title="RAG Search API", version="2.3.0", lifespan=lifespan)


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


async def _encode(texts: List[str]) -> np.ndarray:
    """Encode document texts. Not cached: the indexer encodes each chunk once."""
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedding model is not loaded yet")
    return np.asarray(await embedder.embed(texts), dtype=np.float32)


async def _encode_query(query: str) -> np.ndarray:
    """Encode one retrieval query, answering from the cache when it can.

    Returns shape `(1, dim)`: `/search` takes `vectors[0]` as the vector, so a scalar
    return would read the first component as the whole vector and silently score
    garbage. `/search` has exactly one query text, which is why this takes a string.
    """
    if embedder is None:
        raise HTTPException(status_code=503, detail="Embedding model is not loaded yet")
    return np.asarray(await embedder.embed_query(query), dtype=np.float32)


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


class SearchRequest(BaseModel):
    # The query is the key of the embedding cache (`Embedder.embed_query`), so its
    # length is what makes the cache's memory bound real: the cap times `EMBEDDING_CACHE_SIZE`
    # is the text the cache can hold. 4000 is the repo's own length cap (`rerank_query`
    # below, `RERANK_QUERY_MAX_CHARS` in the gateway), and the gateway's `concat` rewrite
    # is capped at 2000, so a rewrite fits with headroom.
    query: str = Field(min_length=1, max_length=4000)
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
    # Cross-encoder reranking of the candidate list. `null` = the service default
    # (`RERANK_ENABLED`); `false` forces it off for this request, so the harness
    # can compare rerank on/off against one deployed configuration. A reranker
    # that is not loaded, times out, or fails never reranks.
    rerank: Optional[bool] = None
    # Candidates the cross-encoder sees (`RERANK_TOP_N` is the service default,
    # `1..100`). Ignored when reranking is off.
    rerank_top_n: Optional[int] = Field(default=None, ge=1, le=RERANK_CANDIDATE_CAP)
    # The text the cross-encoder scores the passages against. `null` = the retrieval
    # query, which is the pre-P1.1 behavior. Set it when the retrieval query is a
    # long rewrite: the model should judge the question the user asked, not the
    # history injected into the rewrite. Measured on `eval/questions.multiturn.jsonl`
    # with `Qwen/Qwen3-Reranker-0.6B`: pairs built from the `concat` rewrite rank
    # term-noise chunks above the answer, and `mt-003`'s expected chunk moves from
    # dense rank 9 to rank 20, out of the list in hybrid. Pairs built from the
    # question keep the answer's terms in the pair, which is what the reranker is for.
    rerank_query: Optional[str] = Field(default=None, min_length=1, max_length=4000)


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
    # Cross-encoder relevance score, `null` when reranking did not run. `score`
    # stays the pre-rerank relevance number (dense cosine, or the BM25 weight of
    # an exact-term hit), so `min_score` keeps its measured meaning with reranking
    # on; `rerank_score` is what the reranker decided.
    rerank_score: Optional[float] = None


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


def _candidates_dense(query_vector: List[float], request: SearchRequest, pool: int) -> List[SearchResult]:
    """The dense branch, as many candidates as the request asked for."""
    hits = _query(query_vector, pool, request.min_score, using=_dense_using())
    # Filter locally as well, so the /search contract does not depend on the backend
    if request.min_score is not None:
        hits = [hit for hit in hits if hit.score >= request.min_score]
    return [_result(hit.payload or {}, hit.score, None, "dense") for hit in hits]


async def _candidates_hybrid(query_vector: List[float], request: SearchRequest, pool: int) -> List[SearchResult]:
    """Both branches, fused with RRF, in candidate-pool size.

    Each branch returns more items than `top_k` so RRF has material to rank: a
    chunk ranked 12th by BM25 can enter the final top 8 once fused. With
    reranking the pool is `rerank_top_n`, so the cross-encoder sees the intruders
    RRF put at a fixed high rank.
    """
    dense_hits = normalize_hits(
        _query(query_vector, pool, request.min_score, using=_dense_using())
    )

    sparse_query = (await sparse_embedder.embed_query([request.query]))[0]
    ratio = request.sparse_min_relative if request.sparse_min_relative is not None else SPARSE_MIN_RELATIVE
    sparse_hits = filter_relative(normalize_hits(_query_sparse(sparse_query, pool)), ratio)

    weight = request.sparse_weight if request.sparse_weight is not None else SPARSE_WEIGHT
    fused = apply_threshold(
        fuse(dense_hits, sparse_hits, pool, sparse_weight=weight),
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


async def _rank_and_cut(
    candidates: List[SearchResult], request: SearchRequest, rerank: bool
) -> List[SearchResult]:
    """Rerank the candidates and cut to `top_k`, keeping the fused order on failure.

    Reranking is a quality improvement, never a hard dependency: a timeout, a
    model error, or a score count that does not match the candidates returns the
    fused list as it is, so the caller always gets an answer.
    """
    if not rerank:
        return candidates[: request.top_k]

    passages = candidate_passages(candidates, RERANK_MAX_CHARS)
    # `rerank_query` is the text the model judges; a long retrieval rewrite scores
    # against the short question it was built from when the caller sets it.
    pair_text = pair_query(request.rerank_query, request.query)
    try:
        scores = await asyncio.wait_for(
            reranker.score(pair_text, passages), RERANK_TIMEOUT
        )
    except Exception as exc:  # noqa: BLE001 - search must survive a broken reranker
        log.warning(
            "Reranker failed (%s), keeping the fused order for %d candidates",
            exc,
            len(candidates),
        )
        return candidates[: request.top_k]

    ordered = rerank_results(candidates, scores, request.top_k)
    if ordered is None:
        log.warning(
            "Reranker returned %d scores for %d candidates, keeping the fused order",
            len(scores),
            len(candidates),
        )
        return candidates[: request.top_k]
    return ordered


@app.post("/search", response_model=List[SearchResult])
async def search(request: SearchRequest):
    """Document search over the collection: dense, dense + BM25 fused with RRF,
    and optionally reranked with a cross-encoder."""
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

    # A rerank request against a service without a loaded reranker is answered
    # with the fused order, not an error: the gateway contract stays valid.
    rerank = resolve_rerank(request.rerank, RERANK_ENABLED, reranker is not None)
    if request.rerank and reranker is None:
        log.info("Rerank requested with no reranker loaded - fused order kept")

    pool = candidate_limit(
        request.top_k,
        rerank,
        hybrid,
        request.rerank_top_n if request.rerank_top_n is not None else RERANK_TOP_N,
    )

    # Encode the query with the model's retrieval instruction
    # (Qwen3-Embedding: prompt_name="query"). A repeated question is answered from the
    # cache without entering the encode queue.
    query_vectors = await _encode_query(request.query)
    query_vector = query_vectors[0].astype(float).tolist()

    if hybrid:
        candidates = await _candidates_hybrid(query_vector, request, pool)
    else:
        candidates = _candidates_dense(query_vector, request, pool)

    return await _rank_and_cut(candidates, request, rerank)


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
        # The cache is a measured behavior, so its counters are on the running
        # service: `hits` growing on a repeated request is what closes P3.2.
        "embedder_cache": embedder.cache_stats() if embedder is not None else {
            "enabled": EMBEDDING_CACHE_SIZE > 0, "entries": 0, "max": EMBEDDING_CACHE_SIZE,
            "hits": 0, "misses": 0, "vector_dim": None, "approx_bytes": 0,
        },
        "sparse_enabled": sparse_embedder is not None,
        "sparse_model": SPARSE_MODEL if sparse_embedder is not None else None,
        "search_mode": SEARCH_MODE,
        "rerank_enabled": reranker is not None,
        "rerank_model": getattr(reranker, "model_path", None),
        "rerank_top_n": RERANK_TOP_N,
        "rerank_transform": getattr(reranker, "transform", RERANK_TRANSFORM),
        # The instruction the query is labelled with, so a quality change can be
        # read from the health endpoint instead of the container's env file.
        "rerank_query_instruct": getattr(reranker, "query_instruct", RERANK_QUERY_INSTRUCT),
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
