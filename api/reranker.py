# reranker.py
"""Cross-encoder reranker wrapper for the search API.

Placement decided in P1.1: inside `api`, on the same GPU as the embedding model.
That keeps the deployment single-service (no second model owner, no new network
hop, no new healthcheck), and the wrapper serializes predictions the same way
`Embedder` does, so indexing, chat retrieval, and reranking never hold the GPU
at the same time.

`predict` gives one number per (query, passage) pair:
- a classifier-head reranker (`BAAI/bge-reranker-v2-m3`,
  `jina-reranker-v2-base-multilingual`, `cross-encoder/ms-marco-*`) returns raw
  logits, so `RERANK_TRANSFORM=raw` is enough; ranking only cares about order;
- `sigmoid` maps those logits to (0, 1) for a score that is comparable across
  requests, without changing the order;
- the Qwen3-Reranker family scores by the logits of the "yes" and "no" tokens,
  so `RERANK_TRANSFORM=softmax` reads a two-column output as P(yes). Measured
  2026-11-18: `sentence-transformers` gives this model one logit per pair, so the
  two-column path is not taken and the probe score was `-2.0625` for an obvious
  match;
- `RERANK_QUERY_INSTRUCT` puts the instruction the family is trained on in front of
  the query (`Instruct: ...\nQuery: ...`). Plain pairs are what produced that probe
  score, so the instruction is the quality knob for prompt-trained models.

A model that fails to load, times out, or returns an unusable shape is handled
by the caller (`api/main.py`): the fused order is returned unchanged, so the
reranker can never break search.
"""

import asyncio
import logging
from typing import List, Optional

from sentence_transformers import CrossEncoder

from rerank import apply_transform, format_query, resolve_transform

log = logging.getLogger("api.reranker")


class Reranker:
    """Async wrapper around a cross-encoder.

    expose:
        async score(query, list[str]) -> list[float]   # one score per passage
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: Optional[str] = "float16",
        max_length: int = 512,
        batch_size: int = 16,
        transform: str = "raw",
        max_concurrency: int = 1,
        query_instruct: Optional[str] = None,
    ):
        # Imported lazily so `api` keeps starting when the reranker is not used.
        from embedder import resolve_dtype

        self.model_path = model_path
        self.device = device
        self.dtype = resolve_dtype(dtype, device)
        self.max_length = max(1, max_length)
        self.batch_size = max(1, batch_size)
        self.transform = resolve_transform(transform)
        self.max_concurrency = max(1, max_concurrency)
        # The instruction goes in front of the query, once per request, not once
        # per pair: the model labels the query, and every passage is judged
        # against the same labelled question. `api/main.py` resolves the env value;
        # here it is the text itself, or `None`.
        self.query_instruct = (query_instruct or "").strip() or None
        self.predict_queue = asyncio.Semaphore(self.max_concurrency)

        log.info(
            "Loading reranker %s (device=%s, dtype=%s, max_length=%d, batch=%d, transform=%s, instruct=%s)",
            model_path,
            device,
            self.dtype or "model-default",
            self.max_length,
            self.batch_size,
            self.transform,
            repr(self.query_instruct) if self.query_instruct else "none",
        )

        # `trust_remote_code` and the dtype go in `model_kwargs`, which
        # `CrossEncoder` passes to `AutoModel.from_pretrained`. This is the form
        # both the 5.x and 6.x line accept.
        model_kwargs = {"trust_remote_code": True}
        if self.dtype is not None:
            model_kwargs["torch_dtype"] = self.dtype

        self.model = CrossEncoder(
            model_path,
            device=device,
            max_length=self.max_length,
            model_kwargs=model_kwargs,
        )

        # Smoke test: one tiny pair confirms the model predicts and warms the
        # weights, so a bad model id fails at startup, not on the first request.
        probe = self.model.predict(
            [["which cat is the best pet?", "Cats are popular pets."]],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = apply_transform(probe, self.transform)
        log.info(
            "Reranker ready (probe score=%s, transform=%s)",
            round(scores[0], 4) if scores else None,
            self.transform,
        )

    def _predict_sync(self, pairs: List[List[str]]):
        return self.model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )

    async def score(self, query: str, passages: List[str]) -> List[float]:
        """One relevance score per passage, in the order they were given.

        Serialized on the GPU the same way `Embedder` serializes encode calls,
        so a rerank burst and a bulk ingest do not compete for VRAM.
        """
        if not passages:
            return []

        pairs = [[format_query(query, self.query_instruct), passage] for passage in passages]
        loop = asyncio.get_running_loop()
        async with self.predict_queue:
            values = await loop.run_in_executor(None, self._predict_sync, pairs)
        return apply_transform(values, self.transform)
