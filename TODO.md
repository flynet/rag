# TODO / Roadmap

Actionable plan for the RAG stack. Items are ordered by expected value, not by implementation cost. Every quality change is expected to be validated against the retrieval evaluation set from **P0.1**.

**Status legend:** `[ ]` todo · `[~]` in progress · `[x]` done

## Priority overview

| ID | Item | Priority | Effort | Why |
|---|---|---|---|---|
| P0.1 | Retrieval evaluation harness | P0 | S | Without metrics every other tuning is guesswork |
| P0.2 | Startup reconciliation of stale vectors | P0 | S | Real correctness bug: deleted files stay searchable |
| P0.3 | Tune `RAG_TOP_K` / `RAG_MIN_SCORE` | P0 | S | Cheap quality win once P0.1 exists |
| P1.1 | Reranker | P1 | M | Usually the largest retrieval-quality gain |
| P1.2 | Structure-aware chunking + metadata | P1 | M | Better recall and citations |
| P1.3 | Multi-turn query rewriting | P1 | M | Fixes elliptical follow-up questions |
| P2.1 | Hybrid BM25 + dense (RRF) | P2 | L | Exact terms, identifiers, names |
| P2.2 | Healthchecks, pinned Qdrant, retries | P2 | S | Removes startup races and upgrade risk |
| P2.3 | Qdrant snapshots / restore | P2 | S | Recover without a full reindex |
| P2.4 | Indexing vs query contention | P2 | M | Keeps search responsive during bulk ingest |
| P3.1 | TEI/vLLM embedding backend | P3 | L | Throughput under concurrent load |
| P3.2 | Query embedding cache | P3 | S | Repeated-question latency |
| P3.3 | Quantization / payload index | P3 | S | Only matters as the corpus grows |
| P3.4 | Larger embedding model (4B/8B) | P3 | L | Quality, but needs more VRAM |
| P3.5 | PostgreSQL + pgvector migration | P3 | L | Optional infrastructure consolidation |

---

## P0 — Correctness and measurement

### [~] P0.1 Retrieval evaluation harness

**Why:** there is currently no way to tell whether a change (model, chunking, reranker, prompt) improves retrieval.

**Scope:**
- [ ] Create `eval/questions.jsonl` with 50–200 real questions.
- [ ] Ground truth per question: `file` plus either `chunk` index or a text substring unique to the relevant chunk.
- [x] Add `scripts/eval_retrieval.py` (stdlib) that calls `POST /search` with `min_score: null` and `top_k` 10 and 50.
- [x] Metrics: `recall@5`, `recall@10`, `MRR@10`, `nDCG@10`.
- [x] Print per-question failures (missed / wrong rank) and a summary table.
- [ ] Save the baseline for `Qwen3-Embedding-0.6B` to `eval/baseline.json`.
- [x] Document the dataset format and usage in `eval/README.md`.

**Acceptance criteria:**
- One command runs the whole set and exits non-zero on a configurable metric regression.
- Baseline numbers for the current model are committed.
- Questions cover: exact terms/identifiers, paraphrases, Russian/English, and multi-chunk answers.

**Notes:** the harness, schema, example dataset, and `--min-recall`/`--min-mrr` gates are in place. Remaining work: add real questions and commit the baseline. Keep the dataset small but representative; 50 well-chosen questions are better than 500 synthetic ones. Do not tune on the same questions you report as final — keep a held-out subset.

### [~] P0.2 Startup reconciliation of stale vectors

**Why:** files deleted or renamed while the stack was stopped leave their chunks in Qdrant and remain searchable. The startup scan only queues files that exist.

**Scope:**
- [x] On startup, read distinct `payload.file` values from Qdrant (scroll with `with_payload=["file"]`, `with_vectors=False`, batched).
- [x] Compare with the files present under `WATCH_PATH`.
- [x] Delete vectors for missing files (`delete_by_file`).
- [x] Handle renamed files as delete + index (no duplicate paths).
- [x] Log a summary: `removed N orphaned files / M chunks`.

**Acceptance criteria:**
- Stop the stack, delete a file, start the stack: its chunks are gone from `/collection/sample` and from search results.
- Rename a file while stopped: only the new path is searchable.
- Unchanged files are not re-embedded.

**Notes:** implemented in `VectorDB.reconcile_files()` and called before the startup scan. Needs end-to-end verification with Docker (daemon was not running during implementation). Keep this O(number of distinct files); for very large collections, batch the scroll and avoid loading texts.

### [ ] P0.3 Tune `RAG_TOP_K` / `RAG_MIN_SCORE`

**Why:** `3` and `0.5` are inherited defaults, not measured values. Qwen3-Embedding has a different score distribution than embeddinggemma.

**Scope:**
- [ ] Run the evaluation harness for `top_k` ∈ {3, 5, 8, 10}.
- [ ] Run it for `min_score` ∈ {0.3, 0.4, 0.5, 0.6} (and with the threshold disabled).
- [ ] Check query-prompt on/off (`EMBEDDING_QUERY_PROMPT=query` vs empty) on the same set.
- [ ] Record the chosen values in `.env.example` and README.

**Acceptance criteria:**
- Chosen values are justified by the evaluation table, not intuition.
- README states that the threshold is model-dependent and how to re-tune it.

---

## P1 — Retrieval quality

### [ ] P1.1 Reranker

**Why:** a cross-encoder reordering the top candidates usually gives the largest precision gain, especially for paraphrased or long queries.

**Scope:**
- [ ] Retrieve 30–50 candidates from Qdrant (dense), rerank them, pass 5–8 to the LLM.
- [ ] Evaluate `Qwen3-Reranker-0.6B` and `BAAI/bge-reranker-v2-m3` (or `jina-reranker-v2-base-multilingual`).
- [ ] Decide placement: inside `api` (same GPU, simplest) or a separate `reranker` service (isolated VRAM/latency).
- [ ] Add a `/rerank` contract or a `rerank=true` flag on `/search`; keep the gateway backwards compatible.
- [ ] Add a fallback: if the reranker fails, return dense results unchanged.
- [ ] Report latency impact (p50/p95) and VRAM usage.

**Acceptance criteria:**
- `recall@10` does not regress; `MRR@10` / precision at the injected `k` improves on the evaluation set.
- Search still works with the reranker disabled and when it is down.
- VRAM fits alongside the embedding model on 8 GB.

**Notes:** reranking 50 candidates with a 0.6B cross-encoder costs more than embedding one query; measure before enabling by default. Consider running the reranker on CPU if VRAM is tight.

### [~] P1.2 Structure-aware chunking + citation metadata

**Why:** fixed 1024-character chunks cut through headings and sentences; payload had only `file` and `chunk`.

**Scope:**
- [x] Split on headings/paragraphs first, then fall back to a size limit with overlap.
- [x] Prepend the document title (file name or first heading) to each chunk before embedding.
- [x] Store `title`, `section`, `char_start`, `char_end` in the payload.
- [x] Keep deterministic chunk IDs; include the chunking version in the file signature.
- [x] Update docs and expose the metadata through `/search`, `/collection/sample`, and `X-RAG-Sources`.
- [ ] Re-index the corpus and re-run the evaluation.

**Acceptance criteria:**
- Chunks no longer start/end mid-word in the common case.
- Citations can point to a section and character range.
- Evaluation improves or stays flat; indexing time is acceptable.

**Notes:** implemented in `embedder/chunking.py` (`CHUNKER_VERSION = 2`). The title is prepended only to the embedding input; the payload keeps the original chunk text and offsets. `file_signature` includes `chunking_version`, so existing documents are re-indexed on the next start. Remaining work: run with Docker, re-index, and compare against the evaluation baseline. The chunker is covered by synthetic tests (sections, offsets, word boundaries, overlap, CRLF, long words) but not yet by the live pipeline.

### [~] P1.3 Multi-turn query rewriting

**Why:** the gateway embeds only the last user message. A follow-up like "and the second one?" produces a meaningless query.

**Scope:**
- [x] Detect follow-up turns (short/elliptical last message or existing history).
- [x] Option A (cheap): concatenate the last 2–3 user messages as the retrieval query.
- [x] Option B (better): rewrite the query with the upstream LLM using the last N turns; cache and time-box it.
- [x] Add an env flag (`RAG_QUERY_REWRITE=off|concat|llm`).
- [x] Keep the current behavior inside tool loops (last message is `tool`/`assistant`).
- [ ] Extend the evaluation set with multi-turn follow-ups.

**Acceptance criteria:**
- Follow-up questions retrieve the correct passages in the evaluation set.
- Single-turn questions do not regress.
- Added latency is bounded and documented.

**Notes:** implemented in `gateway/main.py` (`_prepare_retrieval_query`, `_is_follow_up`, `_concat_query`, `_rewrite_query_llm`). Default is `off` until the evaluation set covers multi-turn cases; `concat` is deterministic, `llm` adds one upstream call with `RAG_REWRITE_TIMEOUT` (10s) and falls back to `concat`. Rewrites are cached in memory (`RAG_REWRITE_CACHE_SIZE`, default 256). The evaluation harness currently sends a single `question` to `/search`, so it cannot measure gateway-level rewriting yet; that is the remaining work together with the dataset.

---

## P2 — Retrieval breadth and reliability

### [ ] P2.1 Hybrid search: BM25 + dense with RRF

**Why:** dense embeddings miss exact identifiers, product codes, and rare names; BM25 covers them.

**Scope:**
- [ ] Migrate the collection to named vectors: `dense` (1024) + `bm25` (sparse).
- [ ] Compute sparse vectors with FastEmbed `Qdrant/bm25` (add the dependency to both services that write vectors).
- [ ] Query with Qdrant `prefetch` for both branches and `FusionQuery(fusion=RRF)`.
- [ ] Re-index the corpus; keep the model/chunker signature in the payload.
- [ ] Evaluate on identifier-heavy questions vs dense-only.

**Acceptance criteria:**
- Exact-term questions improve on the evaluation set.
- Dense-only quality does not regress for paraphrased questions.
- Collection schema and re-index procedure are documented.

**Notes:** this is a breaking collection change; plan a re-index window. Keep the dense branch as a fallback if sparse vectors are missing.

### [~] P2.2 Reliability: healthchecks, pinned versions, retries

**Why:** Compose starts services before they are ready; `api` may still be downloading weights, and Qdrant was pulled as `latest`.

**Scope:**
- [x] Pin `qdrant/qdrant` to a specific version (`v1.20.1`, overridable via `QDRANT_VERSION`).
- [x] Add an `api` healthcheck that requires `embedder == "ready"` (plain `/health` returns 200 even while loading).
- [x] Add a Qdrant healthcheck. The official image has no `curl`/`wget`, so it uses a TCP check (`bash /dev/tcp`).
- [x] Add `depends_on` conditions: `api` and `embedder` wait for Qdrant; `embedder` waits for `api`; `gateway` starts early and degrades gracefully.
- [x] Add retry/backoff around Qdrant connection setup in the indexer.
- [x] Document the startup sequence and expected first-boot duration (~1.2 GB download + model load).

**Acceptance criteria:**
- Fresh `docker compose up` reaches a working state without a crash loop.
- `docker compose ps` shows meaningful health states.
- Upgrading Qdrant is a deliberate, tested action.

**Notes:** implemented in `docker-compose.yaml` (`QDRANT_VERSION`, healthchecks, `depends_on`) and `embedder/main.py` (`wait_for_qdrant`). There was no `qdrant_data` on this machine, so there was no previously running tag to preserve; `v1.20.1` was the latest stable release at implementation time. E2E (`docker compose up` with the GPU stack) is still pending because Docker Desktop is not running here.

### [ ] P2.3 Qdrant snapshots and restore

**Why:** the index can be rebuilt, but that costs GPU time and a full re-index.

**Scope:**
- [ ] Document and script a snapshot procedure (`POST /collections/{name}/snapshots` or stop-and-copy of `qdrant_data`).
- [ ] Define retention (e.g. keep the last N snapshots).
- [ ] Perform and document a restore test.
- [ ] Optionally add a scheduled snapshot job.

**Acceptance criteria:**
- A snapshot can be restored into a fresh Qdrant instance.
- The README documents the exact commands and caveats.

### [ ] P2.4 Indexing vs query contention

**Why:** bulk indexing and user queries share one GPU and one model; a large ingest delays search.

**Scope:**
- [ ] Measure search latency during a bulk ingest.
- [ ] Options: lower indexer batch size, sleep between batches, a priority queue in `api`, a separate embedding instance for indexing, or CPU embedding for the indexer.
- [ ] Add a concurrency limit around `model.encode` so concurrent requests do not pile up.
- [ ] Expose queue/latency metrics (at least in logs).

**Acceptance criteria:**
- Search p95 latency stays within an agreed budget during indexing.
- Indexing still completes in a reasonable time.

---

## P3 — Scale and performance

### [ ] P3.1 TEI/vLLM embedding backend

**Why:** the custom FastAPI service has no dynamic batching; each request runs `model.encode` in a worker thread.

**Scope:**
- [ ] Benchmark the current `/v1/embeddings` and `/search` under concurrent load.
- [ ] Evaluate Text Embeddings Inference (TEI) or vLLM with Qwen3-Embedding-0.6B.
- [ ] Keep a thin FastAPI layer for `/search`, `/embed`, and Qdrant access; delegate encoding to the backend.
- [ ] Preserve the existing OpenAI-compatible `/v1/embeddings` contract.

**Acceptance criteria:**
- Throughput improves materially under concurrency.
- Latency and quality do not regress for single requests.
- Rollback to the in-process model is documented.

### [ ] P3.2 Query embedding cache

**Why:** repeated questions still run a full forward pass.

**Scope:**
- [ ] Add an LRU cache keyed by `(model, prompt_name, normalized_text)`.
- [ ] Bound memory (vectors are ~4 KB each at 1024 dims).
- [ ] Add hit/miss counters to `/health` or logs.
- [ ] Consider caching at the gateway for full user messages.

**Acceptance criteria:**
- Repeated identical queries hit the cache and skip `model.encode`.
- Memory stays bounded under a long-running workload.

### [ ] P3.3 Qdrant quantization and payload indexes

**Why:** only relevant as the corpus grows; at 10k chunks the savings are negligible.

**Scope:**
- [ ] Add a payload index on `file` (helps deletes and scroll filters).
- [ ] Evaluate scalar quantization (int8) with rescoring on a copy of the collection.
- [ ] Measure recall and memory before enabling.
- [ ] Consider `on_disk` vectors/payload for large corpora.

**Acceptance criteria:**
- Deletion/scroll latency improves or memory drops, without a measurable recall regression.
- Changes are justified by corpus size, not applied by default.

### [ ] P3.4 Larger embedding model (4B / 8B)

**Why:** Qwen3-Embedding-4B scores ~5 points higher than 0.6B on MTEB multilingual; 8B ~6 points higher.

**Scope:**
- [ ] Confirm GPU capacity: 4B is ~8 GB in fp16, 8B ~16 GB — neither fits on an 8 GB card with activations.
- [ ] If the GPU is upgraded, benchmark quality, latency, and indexing time.
- [ ] Account for the dimension change (4B: 2560; 8B: 4096) and a full re-index.
- [ ] Update `EMBEDDING_MAX_SEQ_LENGTH`, batch size, and memory docs.

**Acceptance criteria:**
- A documented VRAM/latency/quality trade-off for the target hardware.
- Rollback to 0.6B is a one-variable change plus re-index.

### [ ] P3.5 PostgreSQL + pgvector migration (optional)

**Why:** consolidate metadata and vectors in one service; simpler backups and SQL filtering.

**Scope:**
- [ ] Add `pgvector/pgvector:pg16` to Compose.
- [ ] Create a `documents` table with `embedding vector(1024)` (not 2304 — that was an old model).
- [ ] Add HNSW/IVFFlat indexes and a `file` index.
- [ ] Replace `qdrant-client` in `embedder/vector_db.py` with `psycopg`/`asyncpg`.
- [ ] Port `/search` to the `<=>` cosine operator.
- [ ] Migrate the corpus and re-run the evaluation.

**Acceptance criteria:**
- Search quality is equivalent to Qdrant on the evaluation set.
- Backup/restore is simpler than the current setup.
- The migration and rollback are documented.

**Notes:** this is a strategic option, not a priority. Qdrant is a better fit until there is a concrete operational reason to consolidate.

---

## Done

- [x] Replace `google/embeddinggemma-300m` with `Qwen/Qwen3-Embedding-0.6B`.
- [x] Add the query-side retrieval instruction (`prompt_name="query"`), with a safe fallback for models without it.
- [x] Pin `transformers>=4.51,<5` and `sentence-transformers>=5.0,<6`.
- [x] Cap embedding input at 8192 tokens and raise the readiness wait to ~10 minutes.
- [x] Expose `query_prompt` and the effective dimension in `api` `/health`.
- [x] Document the current state, limitations, and roadmap in English (`README.md`, `TODO.md`).

## Conventions

- One item at a time; update this file as work progresses.
- Any retrieval-quality change must include evaluation numbers (before/after).
- Any schema or model change must state the re-index cost.
- Prefer changes that are reversible with one environment variable or one restart.
