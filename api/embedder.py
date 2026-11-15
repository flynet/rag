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
from typing import List, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

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


def _embedding_dim(model: SentenceTransformer) -> Optional[int]:
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
        async embed_query(list[str]) -> np.ndarray (n, dim)    # queries
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
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = resolve_dtype(dtype, device)
        self.batch_size = max(1, batch_size)
        self.normalize = normalize
        self.query_prompt = (query_prompt or "").strip() or None

        log.info(
            "Loading model %s (device=%s, dtype=%s, batch=%d)",
            model_path,
            device,
            self.dtype or "model-default",
            self.batch_size,
        )

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
        # Run embedding in a background thread to not block the event loop
        vectors = await loop.run_in_executor(None, self._encode_sync, texts, prompt_name)
        return np.asarray(vectors, dtype=np.float32)

    async def embed_query(self, texts: List[str]):
        """Encode queries with the model's query prompt (when it has one)."""
        return await self.embed(texts, prompt_name=self.query_prompt)
