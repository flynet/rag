# embedder.py
"""Sentence-Transformers wrapper for the search API.

The model is loaded exactly once per deployment. The indexing service
(`embedder`) is a thin HTTP client and has no torch / CUDA at all, so a single
machine does not keep two copies of the weights and two CUDA contexts.
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


class Embedder:
    """Async wrapper around SentenceTransformer.

    expose: async embed(list[str]) -> np.ndarray (n, dim)
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: Optional[str] = "float16",
        batch_size: int = 32,
        normalize: bool = True,
        max_seq_length: Optional[int] = None,
    ):
        self.model_path = model_path
        self.device = device
        self.dtype = resolve_dtype(dtype, device)
        self.batch_size = max(1, batch_size)
        self.normalize = normalize

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

        self.dim = int(self.model.get_sentence_embedding_dimension() or 0)
        log.info("Model ready (dim=%d)", self.dim)

    def _encode_sync(self, texts: List[str]):
        """Synchronous embedding call. Returns np.ndarray shape=(len(texts), dim)."""
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
        )

    async def embed(self, texts: List[str]):
        loop = asyncio.get_running_loop()
        # Run embedding in a background thread to not block the event loop
        vectors = await loop.run_in_executor(None, self._encode_sync, texts)
        return np.asarray(vectors, dtype=np.float32)
