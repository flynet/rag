# TODO / Roadmap

Actionable plan for the RAG stack. Items are ordered by expected value, not by implementation cost. Every quality change is expected to be validated against the retrieval evaluation set from **P0.1**.

**Status legend:** `[ ]` todo · `[~]` in progress · `[x]` done

## Product direction — Local Knowledge Engine

**Target definition:** a local-first, model-agnostic knowledge and context service that indexes documents
and software repositories through specialized pipelines, organizes them into workspaces and collections,
retrieves information through hybrid and structure-aware search, and exposes that context to chatbots and
AI agents through REST, OpenAI-compatible APIs, and MCP, while keeping original files and Git as the
source of truth.

Three use cases share one platform:

1. **Knowledge base / Oracle** — grounded answers about a product or subject (docs, PDFs, manuals, text) with source references.
2. **Coding agent context engine** — relevant code, symbols, docs, dependencies, and tests for agents such as OpenCode working on large repositories.
3. **General local knowledge service** — the same index exposed to chatbots, local assistants, and custom Python/Java applications.

**Core architectural principle:** one platform and shared infrastructure, with specialized ingestion,
parsing, chunking, and retrieval strategies per data type. Do not build separate products for PDF RAG,
documentation RAG, Java RAG, and Python RAG, and do not force every format through one generic parser.

**Division of responsibilities:**

- **Knowledge Engine:** indexes sources, stores searchable representations, retrieves, assembles context, exposes APIs/tools.
- **Coding agent / harness (OpenCode, DSH):** plans work, decides what information it needs, reads and edits files, runs commands and tests, manages its tool loop.
- **Original files and Git:** authoritative for current content. The index is a derived searchable representation, never a replacement for files.
- **LLM / inference server:** generates answers or code; the engine works with any OpenAI-compatible endpoint.

Retrieved content is **untrusted data**: it must never override system instructions or tool policies. For
code tasks, the agent must be able to open the original file and verify current content before editing, so
every result carries provenance (path, line/page, revision, content hash).

Items **P4–P10** are this direction. **P4.3** (safe indexing) is the foundation to finish before pointing
the engine at whole repositories. P0–P3 stay the quality/reliability spine that every new phase depends on.

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
| P4.1 | Workspaces, collections, source registry, metadata schema | P4 | M | Multi-project isolation is the base for everything else |
| P4.2 | Ingestion contract with specialized pipelines | P4 | M | New formats without rewriting the core |
| P4.3 | Safe indexing (include/exclude, secrets, encodings, status) | P4 | M | Foundation before indexing whole repos |
| P5.1 | PDF/DOCX/HTML/TXT extraction with page provenance | P5 | M | Document Oracle needs real formats, not just Markdown |
| P5.2 | Citation data in retrieval responses | P5 | S | Grounded answers need traceable sources |
| P6.1 | Code pipeline: Tree-sitter chunking by symbol units | P6 | L | Symbol-sized chunks beat text windows |
| P6.2 | Symbol index and exact symbol lookup | P6 | M | `find_symbol` / `get_symbol` for agents |
| P6.3 | References, dependencies, related tests | P6 | L | Context assembly for code tasks |
| P6.4 | Git-aware freshness and invalidation | P6 | M | Index must track the repository revision |
| P7.1 | Metadata filtering in retrieval | P7 | M | Scoped search, no cross-collection leaks |
| P7.2 | Context assembly with token budget | P7 | M | Not a flat list of chunks |
| P7.3 | Exact lookup by metadata | P7 | S | Identifiers, paths, symbols |
| P7.4 | Evaluation set for code queries | P7 | M | Code retrieval needs its own quality gate |
| P7.5 | Embedding profiles per collection | P7 | M | Different models for docs and code |
| P8.1 | MCP server with retrieval tools | P8 | M | Agents decide when they need context |
| P8.2 | Gateway injection behavior spec | P8 | M | Tool loops must not re-inject blindly |
| P8.3 | Real coding-agent integration test | P8 | S | Proof that the contract works end to end |
| P9.1 | Index freshness (content hash, parser and index versions) | P9 | M | Stale records must be detectable |
| P9.2 | Retrieval cache with explicit invalidation | P9 | S | Repeated queries, no cross-scope leaks |
| P9.3 | Context/prompt cache rules | P9 | M | Validate freshness before reuse |
| P10.1 | Per-source indexing status, logs, and metrics | P10 | M | Operators need to see what is indexed and why not |
| P10.2 | Backup and restore for the metadata layer | P10 | S | The registry is as critical as the vectors |
| P10.3 | Configurable embedding and reranker providers | P10 | M | Model-agnostic by configuration |
| P10.4 | Access isolation for multiple clients | P10 | M | Per-client scope and key permissions |

---

## P0 — Correctness and measurement

### [x] P0.1 Retrieval evaluation harness

**Why:** there is currently no way to tell whether a change (model, chunking, reranker, prompt) improves retrieval.

**Scope:**
- [x] Create `eval/questions.jsonl` with 50–200 real questions.
- [x] Ground truth per question: `file` plus either `chunk` index or a text substring unique to the relevant chunk.
- [x] Add `scripts/eval_retrieval.py` (stdlib) that calls `POST /search` with `min_score: null` and `top_k` 10 and 50.
- [x] Metrics: `recall@5`, `recall@10`, `MRR@10`, `nDCG@10`.
- [x] Print per-question failures (missed / wrong rank) and a summary table.
- [x] Save the baseline for `Qwen3-Embedding-0.6B` to `eval/baseline.json`.
- [x] Document the dataset format and usage in `eval/README.md`.

**Acceptance criteria:**
- One command runs the whole set and exits non-zero on a configurable metric regression.
- Baseline numbers for the current model are committed.
- Questions cover: exact terms/identifiers, paraphrases, Russian/English, and multi-chunk answers.

**Notes:** 60 questions in `eval/questions.jsonl` over the corpus committed in `eval/corpus/` (202 chunks: `README.md` 68, `TODO.md` 80, `env_example.txt` 40, `eval_README.md` 13, `example.txt` 1). Coverage: exact identifiers and env keys/defaults, versions, ports, hardware, chunking, snapshot and restore commands, health endpoints, roadmap item names, paraphrases, Russian and English, multi-chunk answers. Ground truth was verified twice: offline against `eval/corpus` (`--corpus`) and against the indexed chunks, so every `text_contains` needle exists in the collection. Baseline committed in `eval/baseline.json`: `recall@5=0.683`, `recall@10=0.800`, `MRR@10=0.414`, `nDCG@10=0.557`; recorded gates `--min-recall 0.70 --min-mrr 0.35`, verified by a gate run against the live stack. 12 of 60 questions have no expected hit in the top 10, and they concentrate on short table rows (`Current state`, config tables, troubleshooting) — direct evidence for P2.1 hybrid search and P0.3 tuning. Caveats: the corpus is self-referential (project docs and the env template), so real user documents and a held-out subset are still to be added.

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

**Notes:** implemented in `VectorDB.reconcile_files()` and called before the startup scan. Live evidence from the running stack: deleting a file logged `File disappeared (deleted?)` and `Deleted vectors for /data/documents/Text Document.txt`; a queued index job raced the delete and left one orphaned chunk, which was removed with a filtered `points/delete` (`points_count` 193 → 192). The delete-while-stopped acceptance run is still to be done, together with the rename case. Keep this O(number of distinct files); for very large collections, batch the scroll and avoid loading texts.

### [x] P0.3 Tune `RAG_TOP_K` / `RAG_MIN_SCORE`

**Why:** `3` and `0.5` are inherited defaults, not measured values. Qwen3-Embedding has a different score distribution than embeddinggemma.

**Scope:**
- [x] Run the evaluation harness for `top_k` ∈ {3, 5, 8, 10}.
- [x] Run it for `min_score` ∈ {0.3, 0.4, 0.5, 0.6} (and with the threshold disabled).
- [x] Check query-prompt on/off (`EMBEDDING_QUERY_PROMPT=query` vs empty) on the same set.
- [x] Sweep the follow-up set (`eval/questions.multiturn.jsonl`) in gateway mode with the same grid.
- [x] Record the chosen values in `.env.example` and README.

**Acceptance criteria:**
- Chosen values are justified by the evaluation table, not intuition. *(met: `eval/sweep.md`, `eval/sweep-prompt-off.md`, `eval/sweep-multiturn.md`)*
- README states that the threshold is model-dependent and how to re-tune it.

**Notes:** chosen `RAG_TOP_K=8` / `RAG_MIN_SCORE=0.3`, applied to gateway defaults,
`.env.example` and README. Single-turn set (60 questions, 202 chunks): `top_k=3`
recovers only 0.500 of the answers, `top_k=8` recovers 0.783; `min_score=0.5` cuts
recall@8 to 0.483 and leaves 23% of questions with no context, `0.6` leaves 52%.
`0.3` costs nothing over the disabled threshold, so it is the safe operating point.
Query prompt on/off: on = recall@8 0.783 / MRR@10 0.414, off = recall@8 0.683 /
MRR@10 0.455; the prompt is kept on because coverage at `top_k=8` is what the
injection budget needs. Follow-up set (9 conversations, gateway, rewrite `concat`,
the committed sweep `eval/sweep-multiturn.md`): `min_score=0.5` leaves 11% of the
follow-ups with no context at all and recall@8 0.667, `0.6` leaves 67% and recall@8
0.333, `0.3` keeps recall@8 0.889 with 0% zero-context. With rewrite `off` the same
thresholds are far worse (78% and 100% zero-context), so a threshold tuned on
single-turn questions alone is not safe for conversations. Re-run `--sweep` after any
model, chunking, or query-prompt change; the numbers are not transferable.

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
- [x] Re-index the corpus and re-run the evaluation.

**Acceptance criteria:**
- Chunks no longer start/end mid-word in the common case.
- Citations can point to a section and character range.
- Evaluation improves or stays flat; indexing time is acceptable.

**Notes:** implemented in `embedder/chunking.py` (`CHUNKER_VERSION = 2`). The title is prepended only to the embedding input; the payload keeps the original chunk text and offsets. `file_signature` includes `chunking_version`, so existing documents are re-indexed on the next start. The live pipeline is verified through Docker: `embedder` indexed `example.txt` (`Upserted 1 chunks for /data/documents/example.txt`), `api` `/collection/info` reports `points=1 dim=1024`, `/collection/sample` and `X-RAG-Sources` carry `title`, `section`, `char_start`, `char_end`, and the chat answer cited `/data/documents/example.txt` from the injected passage. The corpus was re-indexed with `CHUNKER_VERSION = 2` (192 chunks over 5 files) and the evaluation now runs against it: P0.1 numbers are the baseline floor, recorded before any later change. There are no pre-chunker numbers to compare against, so the acceptance criterion "improves or stays flat" is measured from this point forward. Indexing 5 documents (about 86 KB of text) took under a minute. The chunker is covered by synthetic tests (sections, offsets, word boundaries, overlap, CRLF, long words).

### [x] P1.3 Multi-turn query rewriting

**Why:** the gateway embeds only the last user message. A follow-up like "and the second one?" produces a meaningless query.

**Scope:**
- [x] Detect follow-up turns (short/elliptical last message or existing history).
- [x] Option A (cheap): concatenate the last 2–3 user messages as the retrieval query.
- [x] Option B (better): rewrite the query with the upstream LLM using the last N turns; cache and time-box it.
- [x] Add an env flag (`RAG_QUERY_REWRITE=off|concat|llm`).
- [x] Keep the current behavior inside tool loops (last message is `tool`/`assistant`).
- [x] Extend the evaluation set with multi-turn follow-ups.

**Acceptance criteria:**
- Follow-up questions retrieve the correct passages in the evaluation set.
- Single-turn questions do not regress.
- Added latency is bounded and documented.

**Notes:** implemented in `gateway/main.py` (`_prepare_retrieval_query`, `_is_follow_up`, `_concat_query`, `_rewrite_query_llm`): requests without history and long non-elliptical questions are never rewritten, `llm` is time-boxed by `RAG_REWRITE_TIMEOUT` (10 s), falls back to `concat` on failure, and rewrites are cached in memory (`RAG_REWRITE_CACHE_SIZE`, default 256). The gateway retrieval path is verified live against a self-hosted OpenAI-compatible server (`UPSTREAM_BASE_URL=http://10.0.0.99:5001/v1`): non-stream and streaming chat both returned grounded answers citing the retrieved section, `X-RAG-Sources` decoded to valid entries, `X-RAG: off` produced `X-RAG-Hits=0`, and an unrelated question produced 0 hits. The default is `concat` in the gateway code, `docker-compose.yaml` and `.env.example`, measured on `eval/questions.multiturn.jsonl` (9 follow-up conversations, gateway mode, 20 candidates requested, threshold disabled, frozen corpus, one session):

| mode | recall@5 | recall@10 | MRR@10 | nDCG@10 | latency avg / p95 |
|---|---|---|---|---|---|
| `off` | 0.667 | 0.778 | 0.277 | 0.392 | 4.45 s / 4.94 s |
| `concat` (default) | 0.778 | 1.000 | 0.420 | 0.770 | 4.66 s / 4.95 s |
| `llm` | 0.889 | 1.000 | 0.514 | 0.781 | 4.09 s / 4.91 s |

Single-turn non-regression is measured, not assumed: 60 single-turn questions through the gateway give `recall@10 0.800` / `MRR@10 0.414` in both `off` and `concat` (`eval/gateway-singleturn-off.json`, `eval/gateway-singleturn-concat.json`), and the guarantee is also covered offline by `tests/test_gateway_rewrite.py` (16 tests: first turn untouched, `concat` joins recent user turns only, `llm` fallback and cache, no upstream call for non-follow-ups). Latency: upstream generation dominates (~4 s per question on the measured server) and the extra `llm` call was not separable from that noise in these runs (the three runs measured 4.09–6.08 s average, 4.91–7.01 s p95, `llm` the fastest), so the cost of `llm` is documented as one extra upstream call per rewritten follow-up, bounded by `RAG_REWRITE_TIMEOUT`, not as a measured millisecond figure. Threshold tuning matters more for follow-ups than for single-turn questions: with rewrite `off`, `RAG_MIN_SCORE=0.5` strips all context from 78% of the follow-ups and `0.6` from 100%; with `concat`, `0.5` strips 11% and `0.6` 67%; with `llm`, 22% and 67%. At `min_score=0.3` zero-context is 0% in every mode, and recall@8 at that threshold is 1.000 (`llm`), 0.889 (`concat`), 0.667 (`off`) — `RAG_MIN_SCORE=0.3` is kept, and `llm` is the mode that gains most from a wide `top_k`. The gateway reports its effective mode in `/health` (`defaults.query_rewrite` + `rewrite_history`), the harness records it in every gateway run as `gateway_defaults` and prints it before the run, so each committed result file says which mode produced it; `eval/README.md` documents that the mode is read at container creation, so a change needs `docker compose up -d --force-recreate gateway`, and that two eval sessions must not share one gateway container (a run measured against a container recreated by another session reports the mode it did not run with).

---

## P2 — Retrieval breadth and reliability

### [ ] P2.1 Hybrid search: BM25 + dense with RRF

**Why:** dense embeddings miss exact identifiers, product codes, and rare names; BM25 covers them.

**Scope:**
- [ ] Migrate the collection to named vectors: `dense` (1024) + `bm25` (sparse).
- [ ] Compute sparse vectors with FastEmbed `Qdrant/bm25` (add the dependency to both services that write vectors).
- [ ] Query with Qdrant `prefetch` for both branches and `FusionQuery(fusion=RRF)`.
- [ ] Apply workspace/collection/source/type filters to both branches (shared with P4.1).
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

**Notes:** implemented in `docker-compose.yaml` (`QDRANT_VERSION`, healthchecks, `depends_on`) and `embedder/main.py` (`wait_for_qdrant`). E2E verified live on this machine with the GPU stack: `docker compose up -d` reached a healthy state without a crash loop, `docker compose ps` showed `qdrant`, `api`, `gateway` healthy and `embedder` running, `api` logged `Startup complete`, `embedder` logged stale-vector reconciliation and the watcher (`Started watcher on /data/documents (debounce 1.0s)`), and `python scripts/smoke_test.py` passed all 5 checks against the gateway with a real LLM upstream. There was no `qdrant_data` on this machine, so there was no previously running tag to preserve; `v1.20.1` was the latest stable release at implementation time. Storage now lives on a named volume (see P2.3 notes).

### [x] P2.3 Qdrant snapshots and restore — done, restore verified live

**Why:** the index can be rebuilt, but that costs GPU time and a full re-index.

**Status:** `scripts/qdrant_snapshot.py` (stdlib only) implements `create`, `list`,
`download`, `delete`, `prune`, `status` and `restore` over the Qdrant REST API. `create`
downloads the snapshot into `./backups/` (git-ignored) so the copy survives a container
rebuild, and prunes the oldest node snapshots with `--keep` (default 5). `restore` uploads
the file with a sha256 checksum (`POST /collections/{name}/snapshots/upload`), prints the
collection state before and after, and refuses to run without `--yes` because it overwrites
the collection. The README documents the procedure, retention and caveats (version pin,
restore with `embedder` stopped, full snapshots restored via the `--snapshot` boot flag).

**Verified:** the protocol against a local stub Qdrant (create → download → prune → list →
restore into a fresh collection, restore guard without `--yes`, node checksum rejection,
missing file), and then against a real Qdrant container: 12 points seeded → `create` →
258 KB snapshot with a checksum on the node plus a copy in `backups/` → `docker compose
down -v` (volume removed) → `up -d qdrant` → `status` reports `missing` → `restore --file
... --yes` → `status` reports `green, points=12, dim=1024` → a scroll by `payload.file` and
a `points/query` search returned the restored chunk with score 1.0.

**Storage lesson baked into the compose file:** the collection snapshot contains the WAL
(`0/wal/first-index`). On a Windows bind mount (`./qdrant_data:/qdrant/storage`) that file
came out as NUL bytes, so the snapshot restored only into the same instance; Qdrant itself
warned there: `Unrecognized filesystem - cannot guarantee data safety`. On a Linux named
volume `first-index` is valid JSON and the snapshot restores into any instance. Storage is
therefore a named volume, and `./qdrant_snapshots` is bind-mounted so node snapshots persist.

**Scope:**
- [x] Document and script a snapshot procedure (`POST /collections/{name}/snapshots?wait=true`, download, prune)
- [x] Define retention (keep the last N snapshots on the node, `--keep`, default 5)
- [x] Keep a copy outside the volume (`backups/`, git-ignored) so it survives a container rebuild
- [x] Document the restore procedure (upload endpoint, checksum, `--yes` guard, caveats)
- [x] Perform and document a restore test (live run on `qdrant/qdrant:v1.20.1`)
- [ ] Optionally add a scheduled snapshot job with a checksum and an off-host copy

**Acceptance criteria:**
- A snapshot can be restored into a fresh Qdrant instance. *(met: restore into a volume-free instance, checked with `status`, scroll and search)*
- The README documents the exact commands and caveats.

### [~] P2.4 Indexing vs query contention

**Why:** bulk indexing and user queries share one GPU and one model; a large ingest delays search.

**Scope:**
- [x] Measure search latency during a bulk ingest.
- [ ] Options: lower indexer batch size, sleep between batches, a priority queue in `api`, a separate embedding instance for indexing, or CPU embedding for the indexer.
- [x] Add a concurrency limit around `model.encode` so concurrent requests do not pile up.
- [x] Expose queue/latency metrics (at least in logs).

**Acceptance criteria:**
- Search p95 latency stays within an agreed budget during indexing.
- Indexing still completes in a reasonable time.

**Notes:** `EMBEDDING_MAX_CONCURRENCY` (default `1`) wraps `model.encode` in an
`asyncio.Semaphore` in `api/embedder.py`, so encode calls queue one at a time
instead of competing for VRAM; the value is reported by `api` `/health`
(`max_concurrency`). `scripts/latency_probe.py` measures `POST /search`
p50/p95/p99 in three phases (idle, while synthetic documents are being indexed,
settled) plus a 32-text `POST /embed` batch, and prints the multiplier under
ingest. The first sample is in `eval/perf-probe.json`, measured while the
baseline and gateway eval runs were also loading the API, so its absolute
latencies are inflated; the idle-vs-ingest comparison on a quiet stack and the
budget decision are still to be done, and the `api` container has to be rebuilt
for the queue to take effect.

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
- [ ] For result caching (P9.2), keys must include retrieval configuration, workspace/collection filters, embedding profile, and index version/state.

**Acceptance criteria:**
- Repeated identical queries hit the cache and skip `model.encode`.
- Memory stays bounded under a long-running workload.
- Cached results are never returned for a different workspace/collection scope.

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

## P4 — Knowledge core (workspaces, collections, sources, pipelines)

### [ ] P4.1 Workspaces, collections, source registry, metadata schema

**Why:** one global collection `documents` cannot serve several products, projects, and repositories in one
installation; agents need scoped retrieval.

**Concepts:**

- **Workspace** — top-level organizational and access/filtering boundary.
- **Collection** — a logical knowledge set (one product documentation set, one code repository).
- **Source** — a registered file, directory, repository, or other ingestible origin.
- **Document/chunk** — extracted and indexed content derived from a source.
- **Index profile** — parser, chunking, and embedding configuration used for a source or collection.

**Scope:**
- [ ] Metadata baseline: `workspace_id`, `collection_id`, `source_id`, `source_path`, `source_type` (`document|code|text`), `file_type`, `language`, `chunk_type` (`section|page|class|method|function`), `content_hash`, `indexed_at`, `parser_version`, `embedding_profile`, plus page number and line range.
- [ ] Code payload additions: symbol name, qualified symbol, imports, repository revision when available.
- [ ] A metadata DB for source records, jobs, configuration, and index state. SQLite is the local-first starting point: one file, no server, trivial backup.
- [ ] `api` `/search` gains optional `workspace` / `collection` filters; add Qdrant payload indexes on these fields (P3.3).
- [ ] Source lifecycle: register, index, refresh, remove; removal deletes derived index entries.

**Acceptance criteria:**
- Two collections are managed and searched through one service with no cross-leak: a scoped query never returns results from another workspace/collection.
- Adding a collection does not require re-indexing the others.
- The logical Workspace/Collection abstraction stays independent from the physical Qdrant layout.

**Notes:** start with one shared physical collection plus mandatory metadata filters for compatible vector
configurations. Never mix incompatible vector dimensions or embedding models in one vector field: use
separate collections or named vector fields (P7.5). Keep original files where they are; do not duplicate
sources into Qdrant. Field names and indexing strategy are finalized during implementation design.

### [ ] P4.2 Ingestion contract with specialized pipelines

**Why:** PDF, Markdown, and code need different parsers and chunkers; one generic chunker pollutes the index.

**Scope:**
- [ ] Common ingestion contract as separate stages: discovery → extraction → normalization → chunking → embedding → storage.
- [ ] Processor registry keyed by `source_type` / `file_type`, with a generic text fallback for formats without a specialized parser.
- [ ] `parser_version` per processor; bumping it triggers re-indexing of affected sources (`CHUNKER_VERSION` in `embedder/chunking.py` becomes per-pipeline).
- [ ] Document and code pipelines share platform services (embedding, storage, retrieval) and keep specialized parsing.

**Acceptance criteria:**
- Adding a new format means adding a processor, not changing the core.
- Pipeline selection is recorded in metadata (`source_type`, `parser_version`) and visible per source.

### [ ] P4.3 Safe indexing foundation

**Why:** pointing the engine at a real repository without filters indexes secrets, build outputs, and
binaries. This is a high-priority foundation, before any bulk indexing.

**Scope:**
- [ ] Configurable include/exclude rules; honor `.gitignore` where appropriate, plus engine-specific rules.
- [ ] Exclude `.git`, dependency directories, build outputs, caches, virtual environments, generated files, and binaries by default.
- [ ] Detect file types instead of reading every file as UTF-8; handle encoding errors and oversized files safely (skip and report, never crash). Observed live: a Windows-1251 file was indexed as mojibake — its chunk text and title were unreadable `????` garbage, and the gateway would inject that into prompts.
- [ ] Do not index secrets by default: `.env`, private keys, credentials, configurable secret patterns.
- [ ] Make indexing idempotent; handle interrupted jobs and retries safely.
- [ ] Track indexing status and errors per source; deletion or exclusion removes derived entries. Observed live: deleting a file immediately after it was created left one orphaned chunk (the queued index job ran after the delete event); startup reconciliation does not cover a running watcher race.
- [ ] Treat retrieved content as untrusted data: it must never override system instructions or tool policies.

**Acceptance criteria:**
- Pointing at a real repository yields no secrets, no build outputs, no binary chunks, and no indexing crash.
- Removing a source removes its chunks; re-indexing the same source produces no duplicates.
- Per-source status and errors are queryable (P10.1).

---

## P5 — Document intelligence

### [ ] P5.1 Format extraction with provenance (PDF, DOCX, HTML, TXT)

**Why:** a documentation Oracle needs real formats, not only Markdown and text.

**Scope:**
- [ ] Page-aware PDF extraction; add OCR only where required.
- [ ] DOCX and HTML extraction where initial use cases need it; keep the TXT path.
- [ ] Preserve titles, headings, page numbers, and source locations in chunks (extend the P1.2 structure-aware chunking).
- [ ] Extract product, document version, and language from filenames and metadata where available.

**Acceptance criteria:**
- A PDF answer cites the page; a Markdown answer cites the section.
- Unsupported and binary files are skipped with a recorded reason, never a crash.

### [ ] P5.2 Citation data in retrieval responses

**Why:** grounded answers must show traceable sources.

**Scope:**
- [ ] Extend the retrieval response and `X-RAG-Sources` with `workspace_id`, `collection_id`, `source_path`, page/section, and content hash.
- [ ] The gateway labels each passage with file and location so the answer can cite it.

**Acceptance criteria:** an answer can be checked against the exact source location it cites.

---

## P6 — Code intelligence

### [ ] P6.1 Tree-sitter chunking by symbol units

**Why:** arbitrary text windows split functions and lose symbols; code retrieval quality depends on symbol-sized chunks.

**Scope:**
- [ ] Repository discovery with ignore rules; language and file-type detection.
- [ ] Syntax-aware parsing with Tree-sitter (or an equivalent parser framework) per language.
- [ ] Chunk by meaningful units: classes, methods, functions, interfaces, modules (not text windows alone).
- [ ] Preserve symbol name, qualified name, file path, line range, language, and imports.

**Acceptance criteria:**
- One language indexed end to end, with symbol and line-range provenance in every chunk.
- Code evaluation (P7.4) improves over text-window chunking.

### [ ] P6.2 Symbol index and exact symbol lookup

**Why:** agents ask for a class, method, or identifier, not for a semantic neighborhood.

**Scope:**
- [ ] Initial symbol index: name, qualified name, file, line range, kind.
- [ ] Exact lookup by symbol, qualified symbol, and identifier (`find_symbol`, `get_symbol`).

**Acceptance criteria:** an agent locates a class/method and gets full source locations without guessing file paths.

### [ ] P6.3 References, dependencies, related tests

**Why:** editing code needs its surroundings, not one isolated chunk.

**Scope:**
- [ ] Add reference and dependency relationships progressively.
- [ ] Assemble related interfaces/implementations, imports, usages, and tests into a context package.

**Acceptance criteria:** a code context package includes the relevant symbol plus dependencies and tests, each with path and line range.

### [ ] P6.4 Git-aware freshness

**Why:** the index must reflect the repository revision, not a stale snapshot.

**Scope:**
- [ ] Track repository identity and the indexed commit/revision where available.
- [ ] Detect changed, added, renamed, and deleted files; invalidate affected chunks, symbols, references, and retrieval-cache entries.
- [ ] Support a forced full re-index.
- [ ] Document whether the index reflects HEAD, the working tree, or both, and handle uncommitted changes where feasible.

**Acceptance criteria:** after a checkout or commit, the index reports the revision it reflects; changed files re-index and deleted files disappear from search.

---

## P7 — Retrieval engine v2

### [ ] P7.1 Metadata filtering in retrieval

**Why:** scope isolation is a core requirement, not a feature.

**Scope:**
- [ ] Workspace/collection/source/type filters applied before or during retrieval, in both dense and sparse branches (P2.1).
- [ ] Configurable result limits and score thresholds per request.

**Acceptance criteria:** a scoped query returns only in-scope results; unscoped behavior stays as today.

### [ ] P7.2 Context assembly with a token budget

**Why:** a flat list of similar chunks is not a usable context package.

**Scope:**
- [ ] Assemble from retrieved results: passages with section headings, document title/version, page, citation references; for code, symbols, dependencies, tests, line ranges, revision/content hashes for freshness validation.
- [ ] Respect a configurable token/size budget and drop near-duplicate chunks.

**Acceptance criteria:** assembled context fits the budget, contains no near-duplicates, and every item has provenance.

### [ ] P7.3 Exact lookup by metadata

**Scope:**
- [ ] Lookup by source path, symbol, qualified symbol, identifier, and other exact metadata.

**Acceptance criteria:** exact-term lookups return the source location without embedding similarity.

### [ ] P7.4 Evaluation set for code queries

**Why:** code retrieval has different ground truth than documentation.

**Scope:**
- [ ] Extend `eval/questions.jsonl` with code questions: symbol location, function behavior, dependency questions.
- [ ] Ground truth: file plus symbol/line range.

**Acceptance criteria:** code metrics (recall@k, MRR@10) are reported separately from document metrics and gated.

### [ ] P7.5 Embedding profiles per collection

**Why:** documentation and code may need different models and prompting.

**Scope:**
- [ ] Configurable embedding profile per collection (model, prompt name, dimension) recorded in metadata.
- [ ] Named vector fields per profile; never mix incompatible dimensions in one field.
- [ ] Cache keys include the embedding profile (P3.2, P9.2).

**Acceptance criteria:** two collections use different models; a query uses the profile of its scope; quality is measured per profile.

---

## P8 — Agent integration

### [ ] P8.1 MCP server with retrieval tools

**Why:** coding agents should decide when they need context, not receive blind injections.

**Scope:**
- [ ] MCP server exposing retrieval tools: `search_knowledge`, `search_code`, `find_symbol`, `get_symbol`, `find_references` (after reference indexing), `get_file_structure`, `get_source_metadata`.
- [ ] Tool results include source paths, line/page locations, content hashes or revision, and clear result types.
- [ ] Retrieval aids only: the agent keeps using its own filesystem and terminal tools for reading, editing, and testing.

**Acceptance criteria:** an MCP client can search and locate symbols in a registered repository and cite paths/lines from the results.

### [ ] P8.2 Gateway injection behavior spec

**Why:** tool-loop requests must not re-inject the same context on every iteration.

**Scope:**
- [ ] Document exactly when retrieval is triggered and how content is injected.
- [ ] Configurable retrieval scope (workspace, collection, source type).
- [ ] Decide retrieval on the initial user turn, on selected subsequent turns, or through explicit agent tools; preserve streaming; handle tool calls and tool-loop messages correctly.
- [ ] Keep the OpenAI-compatible gateway contract; integrate multi-turn rewriting (P1.3).

**Acceptance criteria:** documented behavior matches implementation and tests, including the tool-loop case.

### [ ] P8.3 Real coding-agent integration test

**Scope:**
- [ ] Drive one task with a coding agent (OpenCode) against a registered repository: locate code, read the original file, edit, run tests.

**Acceptance criteria:** a recorded session where the agent uses engine retrieval and verifies against original files.

---

## P9 — Caching and freshness

Three mechanisms solve different problems; do not conflate them.

### [ ] P9.1 Persistent index freshness

**Scope:**
- [ ] Content hashes plus parser and index versions identify stale records.
- [ ] Incremental updates when sources change; explicit stale/refresh reporting per source.

**Acceptance criteria:** a source whose content hash changed is marked stale and re-indexed automatically.

### [ ] P9.2 Retrieval cache

**Why:** repeated queries re-run embedding and search; caching helps only if keys cover the scope.

**Scope:**
- [ ] Cache search results for repeated queries; keys include query, retrieval configuration, workspace/collection filters, embedding profile, and index version/state.
- [ ] Invalidate or bypass the cache when affected content or index configuration changes; never return cached results from a different scope.

**Acceptance criteria:** repeated queries skip work; a re-index invalidates affected entries; cross-scope leakage is impossible.

### [ ] P9.3 Context/prompt cache rules

**Scope:**
- [ ] Reuse assembled context only after validating source freshness.
- [ ] Treat model-side prefix/prompt caching as an inference-server capability, not a RAG feature; do not claim Qdrant provides LLM prompt caching.

**Acceptance criteria:** reused context is freshness-checked; documentation does not overstate prompt caching.

---

## P10 — Operability and extensibility

### [ ] P10.1 Per-source status, logs, and metrics

**Why:** bulk indexing over many sources fails partially; silent failure is worse than a loud one.

**Scope:**
- [ ] Status, errors, and progress per source (registry + structured logs).
- [ ] Index size, staleness, cache hits, and latency in health/metrics endpoints.

**Acceptance criteria:** "what is indexed, what failed, what is stale" is answerable from the API.

### [ ] P10.2 Backup and restore for the metadata layer

**Why:** the source registry is as critical as the vectors.

**Scope:**
- [ ] Backup and restore the metadata DB together with Qdrant snapshots (P2.3): same version discipline, checksum, confirmation guard, off-host copy.

**Acceptance criteria:** a restore brings back registry and vectors consistently.

### [ ] P10.3 Configurable embedding and reranker providers

**Why:** the engine must be model-agnostic, not tied to one sentence-transformers path.

**Scope:**
- [ ] Embedding and reranking behind interfaces with multiple implementations: sentence-transformers, TEI/vLLM (P3.1), external OpenAI-compatible embedding endpoints, selected by configuration.

**Acceptance criteria:** switching a provider is a configuration change, not a code change; profiles recorded per collection.

### [ ] P10.4 Access isolation for multiple clients

**Scope:**
- [ ] Per-client keys with workspace/collection permissions, enforced in gateway and API.

**Acceptance criteria:** a client cannot retrieve outside its granted scope.

---

## Non-goals (initial scope)

- Multi-user SaaS, accounts, web UI, cloud sync, distributed scaling.
- Autonomous agent planning inside the engine: the agent plans, the engine supplies context.
- The index as a replacement for files, Git, `grep`, LSP, IDE tools, or the agent's own filesystem tools.
- Fine-tuning, knowledge graphs, multimodal indexing, multi-agent orchestration, full-text search servers, code execution/sandboxing, semantic memory with LLM summarization: deferred until the basic engine is proven.
- Treating prefix/prompt caching as a feature of the RAG service.
- Replacing the OpenAI-compatible gateway with an MCP-only architecture.
- Premature microservice splitting; local-first simplicity wins.

## Open design decisions

- Qdrant layout: shared collection with filters, per-collection collections, named vector fields.
- Metadata DB choice (SQLite first) and its schema; where source registry, index profiles, and jobs live.
- Exact metadata field names and payload indexing strategy.
- Context assembly format and token budget defaults.
- MCP tool surface naming and transport.
- Git working-tree handling and secret rule defaults.
- Embedding profiles per data type; reranker choice.
- Cache invalidation mechanism and cache storage location.
- Workspace/collection permission model for multiple clients.
- How index freshness is exposed to clients (API, MCP, response headers).

---

## Done

- [x] Replace `google/embeddinggemma-300m` with `Qwen/Qwen3-Embedding-0.6B`.
- [x] Add the query-side retrieval instruction (`prompt_name="query"`), with a safe fallback for models without it.
- [x] Pin `transformers>=4.51,<5` and `sentence-transformers>=5.0,<6`.
- [x] Cap embedding input at 8192 tokens and raise the readiness wait to ~10 minutes.
- [x] Expose `query_prompt` and the effective dimension in `api` `/health`.
- [x] Snapshot restore proven live: a snapshot taken on a Linux named volume restores into a
      fresh Qdrant instance (`status` + `points/query` verified); a snapshot created on a
      Windows bind mount carries a corrupted WAL `first-index` and cannot be relied on.
- [x] Document the current state, limitations, and roadmap in English (`README.md`, `TODO.md`).

## Conventions

- One item at a time; update this file as work progresses.
- Any retrieval-quality change must include evaluation numbers (before/after).
- Any schema or model change must state the re-index cost.
- Prefer changes that are reversible with one environment variable or one restart.
- Do not rewrite the whole project in one pass: propose small, reviewable changes and keep working functionality.
- Before changing storage schemas, explain migration and re-indexing implications.
- Add tests for each meaningful behavior and run the existing checks after changes.
- Prefer simple local-first components and explicit interfaces over unnecessary microservices.
- Do not add dependencies, services, or model requirements without explaining why they are needed.
- Do not invent APIs, configuration keys, or modules; verify names and behavior in the repository.
- Keep document and code pipelines specialized while sharing platform services.
- Keep original files and Git authoritative; make index freshness and provenance visible to clients.
- Treat security, file filtering, secret exclusion, scope isolation, and prompt-injection resistance as core requirements.
- At the end of each phase, report files changed, architectural decisions, tests run, known limitations, and the next proposed phase.
