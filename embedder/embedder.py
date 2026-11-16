# embedder.py
"""Thin HTTP client for the embedding service (`api`).

The model lives in `api` and is loaded once per deployment. This service only
watches the documents folder and asks `api` for vectors, so it ships without
torch, CUDA and the weights themselves.
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np

log = logging.getLogger("embedder.client")


class RemoteEmbedder:
    """Drop-in replacement for the old local model wrapper.

    expose: async embed(list[str]) -> np.ndarray (n, dim)
    """

    def __init__(
        self,
        api_url: str,
        batch_size: int = 32,
        timeout: float = 300.0,
        retries: int = 3,
        retry_delay: float = 2.0,
    ):
        self.api_url = api_url.rstrip("/")
        self.batch_size = max(1, batch_size)
        self.timeout = timeout
        self.retries = max(1, retries)
        self.retry_delay = retry_delay
        self.dim: Optional[int] = None

    async def health(self) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{self.api_url}/health")
            response.raise_for_status()
            return response.json()

    async def sparse_enabled(self) -> bool:
        """Does `api` report a BM25 model? The collection schema depends on it."""
        try:
            payload = await self.health()
        except Exception as exc:  # noqa: BLE001 - fall back to the default schema
            log.warning("Could not read api /health, assuming BM25 is enabled: %s", exc)
            return True
        flag = payload.get("sparse_enabled")
        return True if flag is None else bool(flag)

    async def wait_ready(self, attempts: int = 300, delay: float = 2.0) -> int:
        """Block until `api` reports a loaded model. Returns the vector dimension.

        The default gives ~10 minutes: the first start has to download and load
        the embedding model (Qwen3-Embedding-0.6B is ~1.2 GB) before `api`
        answers `/health`.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                payload = await self.health()
                if payload.get("embedder") == "ready":
                    self.dim = int(payload.get("dim") or 0) or None
                    if self.dim:
                        log.info("Embedding service is ready (dim=%d)", self.dim)
                        return self.dim
                    last_error = RuntimeError("api did not report the embedding dimension")
                else:
                    last_error = RuntimeError(f"model not loaded yet: {payload.get('embedder')}")
            except Exception as exc:  # httpx errors, connection refused, etc.
                last_error = exc

            if attempt == attempts:
                break
            if attempt in (1, 5) or attempt % 15 == 0:
                log.info("Waiting for embedding service at %s (attempt %d/%d)", self.api_url, attempt, attempts)
            await asyncio.sleep(delay)

        raise RuntimeError(
            f"Embedding service at {self.api_url} did not become ready in time: {last_error}"
        )

    async def _post(self, path: str, payload: dict) -> dict:
        last_error: Optional[Exception] = None

        for attempt in range(1, self.retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(f"{self.api_url}{path}", json=payload)
                    response.raise_for_status()
                    return response.json()
            except httpx.HTTPStatusError as exc:
                # 4xx means we sent something wrong - retrying will not help
                if exc.response.status_code < 500 and exc.response.status_code != 429:
                    raise RuntimeError(
                        f"Embedding request rejected ({exc.response.status_code}): "
                        f"{exc.response.text[:300]}"
                    ) from exc
                last_error = exc
            except (httpx.HTTPError, ValueError) as exc:
                last_error = exc

            if attempt < self.retries:
                wait = self.retry_delay * attempt
                log.warning(
                    "Embedding request failed (attempt %d/%d), retrying in %.1fs: %s",
                    attempt,
                    self.retries,
                    wait,
                    last_error,
                )
                await asyncio.sleep(wait)

        raise RuntimeError(f"Embedding request failed after {self.retries} attempts: {last_error}")

    async def embed_with_sparse(self, texts: List[str]):
        """Dense vectors + BM25 sparse vectors, batch-aligned with `texts`.

        `api` returns `sparse` when its BM25 model is enabled. Responses without
        it (BM25 disabled, or an older `api`) give an empty sparse list, and the
        caller stores dense-only points: semantic search keeps working, exact-term
        search does not.
        """
        if not texts:
            return np.zeros((0, self.dim or 0), dtype=np.float32), []

        dense_batches: List[np.ndarray] = []
        sparse_batches: List[List[Dict[str, Any]]] = []

        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            payload = await self._post("/embed", {"texts": batch})

            if payload.get("dim"):
                self.dim = int(payload["dim"])
            dense_batches.append(np.asarray(payload.get("vectors") or [], dtype=np.float32))
            sparse_batches.append(list(payload.get("sparse") or []))

        if not dense_batches:
            return np.zeros((0, self.dim or 0), dtype=np.float32), []

        dense = np.vstack(dense_batches)
        sparse = [item for batch in sparse_batches for item in batch]
        if sparse and len(sparse) != len(texts):
            raise RuntimeError(
                f"api returned {len(sparse)} sparse vectors for {len(texts)} texts"
            )
        return dense, sparse

    async def embed(self, texts: List[str]) -> np.ndarray:
        dense, _ = await self.embed_with_sparse(texts)
        return dense
