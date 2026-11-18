# embedder.py
"""Sentence-Transformers wrapper for the search API.

The model is loaded exactly once per deployment. The indexing service
(`embedder`) is a thin HTTP client and has no torch / CUDA at all, so a single
machine does not keep two copies of the weights and two CUDA contexts.

Retrieval models can be asymmetric: Qwen3-Embedding expects an instruction on
the query side only, while documents are encoded as plain text. Hence the two
entry points below: `embed()` for documents and `embed_query()` for queries.
"""

import asyncio
import logging
from collections import OrderedDict
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# `sentence_transformers` is imported inside `Embedder.__init__`, the way `fastembed` is
# imported in `SparseEmbedder` below: the cache rules are unit-tested against a fake
# model on a machine that has torch but not sentence-transformers, and the dense path
# should load a model at the point of use, not at import time.

log = logging.getLogger("api.embedder")

_DTYPE_ALIASES = {
    "": None,
    "auto": None,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
    "float": torch.float32,
}


def resolve_dtype(name: Optional[str], device: str):
    """Map EMBEDDING_DTYPE to a torch dtype (None = keep the model default)."""
    key = (name or "auto").strip().lower()
    if key not in _DTYPE_ALIASES:
        raise ValueError(
            f"Unknown EMBEDDING_DTYPE '{name}', expected one of: "
            f"{', '.join(sorted(k for k in _DTYPE_ALIASES if k))}"
        )

    dtype = _DTYPE_ALIASES[key]
    if dtype is not None and device.startswith("cpu") and dtype in (torch.float16, torch.bfloat16):
        log.warning("fp16/bf16 on CPU is slow and unsupported by many ops - using float32")
        return torch.float32
    return dtype


def _embedding_dim(model) -> Optional[int]:
    """Read the embedding dimension across sentence-transformers versions."""
    # v5.4+ renamed the getter; the old name still exists but is deprecated.
    for name in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, name, None)
        if callable(getter):
            return getter()
    return None


class Embedder:
    """Async wrapper around SentenceTransformer.

    expose:
        async embed(list[str]) -> np.ndarray (n, dim)          # documents
        async embed_query(str) -> np.ndarray (1, dim)          # one retrieval query
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: Optional[str] = "float16",
        batch_size: int = 32,
        normalize: bool = True,
        max_seq_length: Optional[int] = None,
        query_prompt: Optional[str] = "query",
        max_concurrency: int = 1,
        cache_size: int = 256,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = resolve_dtype(dtype, device)
        self.batch_size = max(1, batch_size)
        self.normalize = normalize
        self.query_prompt = (query_prompt or "").strip() or None
        # One GPU model serializes encode calls; a queue of 1 keeps VRAM and
        # latency predictable. Raising it overlaps requests but uses more memory.
        self.max_concurrency = max(1, max_concurrency)
        self.encode_queue = asyncio.Semaphore(self.max_concurrency)
        # A query embedding is a pure function of (model, prompt, text), so a
        # repeated question does not need a forward pass. The cache is bounded by
        # entry count, and an entry is exactly one vector: a 1024-dim float32 array is
        # 4 KB plus ~112 B of array object, so 256 entries is ~1 MB. The key carries the
        # query text, capped by `SearchRequest.query`, so keys add at most
        # `cache_max * max_query_chars` bytes. `cache_size=0` turns it off. Documents
        # are not cached: the indexer encodes each chunk once, so caching the ingest
        # path only grows the footprint.
        self.cache_max = max(0, cache_size)
        self.cache: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self.cache_hits = 0
        self.cache_misses = 0

        log.info(
            "Loading model %s (device=%s, dtype=%s, batch=%d, cache=%d)",
            model_path,
            device,
            self.dtype or "model-default",
            self.batch_size,
            self.cache_max,
        )

        # Imported here, not at module level: this file's own convention for an
        # optional heavy dependency (see `SparseEmbedder`), and it lets the cache rules
        # be tested against a fake model on a machine without sentence-transformers.
        from sentence_transformers import SentenceTransformer

        kwargs = {"device": device, "trust_remote_code": True}
        if self.dtype is not None:
            kwargs["model_kwargs"] = {"torch_dtype": self.dtype}

        self.model = SentenceTransformer(model_path, **kwargs)

        if max_seq_length:
            self.model.max_seq_length = max_seq_length

        self.dim = int(_embedding_dim(self.model) or 0)
        log.info("Model ready (dim=%d, max_seq_length=%s)", self.dim, self.model.max_seq_length)

        # Qwen3-Embedding stores a retrieval instruction under the "query"
        # prompt. If the model has none, queries are encoded as plain text.
        prompts = getattr(self.model, "prompts", None) or {}
        if self.query_prompt:
            prompt_text = prompts.get(self.query_prompt)
            if prompt_text is None:
                log.warning(
                    "Model has no %r prompt (available: %s) - queries will be "
                    "encoded without an instruction",
                    self.query_prompt,
                    ", ".join(sorted(prompts)) or "none",
                )
                self.query_prompt = None
            elif not prompt_text.strip():
                log.info(
                    "Model's %r prompt is empty - queries will be encoded "
                    "without an instruction",
                    self.query_prompt,
                )
                self.query_prompt = None
            else:
                log.info("Using query prompt %r: %r", self.query_prompt, prompt_text)
        else:
            log.info("Query prompt disabled - queries and documents are encoded identically")

    def _encode_sync(self, texts: List[str], prompt_name: Optional[str] = None):
        """Synchronous embedding call. Returns np.ndarray shape=(len(texts), dim)."""
        kwargs = {}
        if prompt_name:
            kwargs["prompt_name"] = prompt_name
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            **kwargs,
        )

    async def embed(self, texts: List[str], prompt_name: Optional[str] = None):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        loop = asyncio.get_running_loop()
        # the encode queue serializes requests on the GPU, so indexing and user
        # queries do not compete for VRAM at the same time
        async with self.encode_queue:
            vectors = await loop.run_in_executor(None, self._encode_sync, texts, prompt_name)
        return np.asarray(vectors, dtype=np.float32)

    def _cache_key(self, norm: str, prompt_name: Optional[str]) -> tuple:
        """The key a query embedding is a function of.

        Model and prompt are in the key, so a redeploy with a different model, or with
        the retrieval prompt switched off, cannot serve a vector the previous one
        produced. The text is the string that was encoded, not a normalized variant of
        it: lowercasing would change the embedding, so the key is exactly the encoded
        text and nothing the cache answers is a vector for a different string.
        """
        return (self.model_path, prompt_name, norm)

    def cache_stats(self) -> Dict[str, Any]:
        """Cache state, reported by `/health`: the counters are what P3.2 validates.

        `approx_bytes` is the bound the task asks for, read from the running service
        rather than from the size of the dict in the source. An entry is one vector
        (dim x 4 bytes) plus the array object (~112 B), and the query text in the key is
        capped by `SearchRequest.query`, so the bound is the memory the container holds.
        """
        return {
            "enabled": self.cache_max > 0,
            "entries": len(self.cache),
            "max": self.cache_max,
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "vector_dim": self.dim,
            "approx_bytes": len(self.cache) * (self.dim * 4 + 112),
        }

    async def embed_query(self, query: str) -> np.ndarray:
        """Encode one retrieval query with the model's query prompt (when it has one).

        Returns shape `(1, dim)`, which is what `/search` indexes: `vectors[0]` is the
        vector, not the first component of it. A query embedding is a pure function of
        (model, prompt, text), so a repeated question costs no forward pass, and a hit
        answers **before the encode queue** - a cached query does not wait behind a bulk
        ingest. Documents go through `embed`, which is not cached, and `/v1/embeddings`
        goes through `embed` as well, so it is not cached: its texts are document-side
        and are encoded without the retrieval prompt.
        """
        # Whitespace around a question is not part of it, which is what makes "  q  "
        # and "q" one cache entry. An all-whitespace query becomes a space, the guard
        # `/embed` and `/v1/embeddings` already use: an empty text is not something to
        # hand to the model.
        norm = query.strip() or " "
        key = self._cache_key(norm, self.query_prompt)

        if self.cache_max and key in self.cache:
            self.cache.move_to_end(key)
            self.cache_hits += 1
            log.debug("query cache hit %r (entries=%d, hits=%d, misses=%d)",
                      norm[:80], len(self.cache), self.cache_hits, self.cache_misses)
            return self.cache[key]

        self.cache_misses += 1
        log.debug("query cache miss %r (entries=%d, hits=%d, misses=%d)",
                  norm[:80], len(self.cache), self.cache_hits, self.cache_misses)
        vector = await self.embed([norm], prompt_name=self.query_prompt)
        if self.cache_max:
            # A copy, not a view: the cached vector is its own 4 KB, so no batch array
            # stays alive behind it, and nothing the caller mutates can reach the cache.
            self.cache[key] = np.array(vector, dtype=np.float32)
            if len(self.cache) > self.cache_max:
                self.cache.popitem(last=False)
        return vector


def sparse_to_dict(vector):
    """Turn a fastembed sparse vector into the dict Qdrant takes.

    `{"indices": [int], "values": [float]}`; indices must be positive integers.
    """
    return {
        "indices": [int(i) for i in vector.indices.tolist()],
        "values": [float(v) for v in vector.values.tolist()],
    }


class SparseEmbedder:
    """BM25 sparse vectors (FastEmbed, CPU ONNX - no torch, no CUDA).

    expose:
        async embed(list[str]) -> list[dict]        # document side, BM25 weights
        async embed_query(list[str]) -> list[dict]  # query side, term frequency

    The Qdrant BM25 recipe: documents carry BM25 term weights, queries carry raw
    term frequency, and the collection's sparse index has `modifier=IDF`, so IDF
    is computed from the real corpus (better than the static estimate fastembed
    bakes into `embed()`).
    """

    def __init__(self, model_name: str = "Qdrant/bm25", threads: Optional[int] = None):
        # Imported here so the dense path works when fastembed is not installed.
        from fastembed import SparseTextEmbedding

        self.model_name = model_name
        log.info("Loading BM25 sparse model %s (CPU, threads=%s)", model_name, threads or "auto")
        self.model = SparseTextEmbedding(model_name=model_name, threads=threads)
        # Smoke test: one tiny encode confirms the ONNX session works.
        probe = self.model.embed(["bm25 probe"])
        sample = list(probe)[0]
        log.info(
            "BM25 model ready (%d terms in the probe text, term id space ~%d)",
            len(sample.indices),
            int(max(sample.indices)) if len(sample.indices) else 0,
        )

    def _encode_sync(self, texts: List[str], query: bool):
        if query:
            vectors = self.model.query_embed(texts)
        else:
            vectors = self.model.embed(texts)
        return [sparse_to_dict(v) for v in vectors]

    async def embed(self, texts: List[str]):
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode_sync, texts, False)

    async def embed_query(self, texts: List[str]):
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode_sync, texts, True)
