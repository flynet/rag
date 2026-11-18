# TODO / Roadmap

Actionable plan for the RAG stack. Items are ordered by expected value, not by implementation cost. Every quality change is expected to be validated against the retrieval evaluation set from **P0.1**.

**Status legend:** `[ ]` todo · `[~]` in progress · `[x]` done

## Time budgets — hard rule (2026-11-18, recorded by the owner)

**A full test suite takes about 1 minute.** `python -m unittest discover -s tests -p "test_*.py"` runs 260 tests
in 26 seconds. **If a test run takes longer than a minute, it is not working correctly** — it is waiting,
retrying, loading models, or hanging; diagnose that instead of letting it run.

**Never launch a command without an explicit time limit.** Hours have been lost to commands with no timeout.
Every measured budget below must be signed off against the measured value; if a step exceeds its budget, stop
and diagnose, do not extend the budget.

| Step | Measured | Expect |
|---|---|---|
| Test suite (260 tests) | 26 s | ≤ 1 min |
| 60-question retrieval sweep, dense, no rerank | ~10 s | ≤ 1 min |
| 60-question sweep, Qwen3 rerank, 2 pools | ~5 min | ≤ 10 min |
| 14-question gateway multiturn, 2 pools | ~2.5 min | ≤ 5 min |
| Docker image rebuild (`docker compose build api`) | ~1 min when cached; ~10 min cold (installs `default-jdk` for `rank-bm25`) | ≤ 15 min |
| Container start + model load (`bge-small` + `Qwen3-0.6B`) | ~90 s | ≤ 4 min |
| Health check after recreate | 10 s, then every 10 s, **bounded to ~2 min** | ≤ 4 min |
| Latency probe, small volume (6 docs × 4 sections, 8 search requests, 2 connections, 30 s ingest window) | minutes | ≤ 5 min for fused + reranked together |
| User-visible gateway latency, live LLM, 6 requests | generation-bound | ≤ 5 min |

**Measure on a small volume that emulates the situation being fixed, not on the full
corpus.** The situation P2.4 fixes is search staying responsive while the indexer is
busy, so the probe needs an ingest window and a handful of concurrent search requests —
not a large corpus. Reading a multiplier from 8 requests is enough; the full corpus is
for quality grids, not for latency. Never let a probe run for hours: if it exceeds its
budget, stop and diagnose.

Health-check loops must use a condition that can actually match (`ready True` and the prompt text are
separated by the model name, so `-match 'ready True Given'` never matches and the loop sleeps 20 minutes).
Check health with `eval_retrieval.Session`, not `Invoke-WebRequest` (fails against `localhost` on this machine).

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

### [x] P0.2 Startup reconciliation of stale vectors

**Why:** files deleted or renamed while the stack was stopped leave their chunks in Qdrant and remain searchable. The startup scan only queues files that exist.

**Scope:**
- [x] On startup, read distinct `payload.file` values from Qdrant (scroll with `with_payload=["file"]`, `with_vectors=False`, batched).
- [x] Compare with the files present under `WATCH_PATH`.
- [x] Delete vectors for missing files (`delete_by_file`).
- [x] Handle renamed files as delete + index (no duplicate paths).
- [x] Log a summary: `removed N orphaned files / M chunks`.

**Acceptance criteria:**
- [x] Stop the stack, delete a file, start the stack: its chunks are gone from `/collection/sample` and from search results.
- [x] Rename a file while stopped: only the new path is searchable.
- [x] Unchanged files are not re-embedded.

**Notes:** implemented in `VectorDB.reconcile_files()`, called before the startup scan. The acceptance probes are committed as `scripts/reconcile_check.py` (`audit`, `count --file`, `search --query --expect --absent`), documented in the README. Runs on the live stack (5-file corpus, 202 chunks at baseline):

- Delete while stopped: marker file indexed to 1 chunk, stack stopped, file deleted, stack started. Startup logged `Reconciling index with /data/documents...` and `Removed stale vectors for 2 file(s) (2 chunks)`. `count --file` for the deleted file returned 0, and `audit` reported `index matches the watched folder` (no `ORPHANED`).
- Rename while stopped: `p02_check.txt` → `p02_check_v2.txt` with the stack down. After startup: `count` = 0 for the old path, 1 for the new path, `audit` matched 6 disk files against 6 indexed paths, and `search --expect p02_check_v2.txt --absent p02_check.txt` printed `OK` for both (the renamed chunk ranked first, score 0.885 in the earlier run). No duplicate paths.
- No re-embedding of unchanged files: the same startup logged `Unchanged, skipping` for all untouched files and upserted only the renamed one; `points_count` stayed at the baseline.
- Delete while running (watcher, as a control): `File disappeared (deleted?)` and `Deleted vectors for <path>`, count dropped to 0, `audit` back to a clean match.
- `audit` paginates Qdrant scroll with `next_page_offset`; reading `next_offset` stops after the first page (100 points) and reports indexed files as `NOT INDEXED` — this was the bug found while writing the probe on a 203-point collection.

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
- [x] Retrieve 30–50 candidates and rerank them, pass 5–8 to the LLM. The candidate
  list is the **fused hybrid list** (`POST /search` requests `2 × top_k` candidates,
  capped at 100), not dense-only: after P2.1 the items that push relevant chunks out
  of the top 10 come from the BM25 branch. Measured: pool 30 is enough — pool 50 gives
  the same grids at higher latency on both models.
- [x] Evaluate `Qwen3-Reranker-0.6B` and `BAAI/bge-reranker-v2-m3`. **`bge` does not
  meet the criterion** (follow-up MRR@10 `0.511` against the committed `0.770`);
  `Qwen3-Reranker-0.6B` is the better model on both sets.
- [x] Decide placement: inside `api` (same GPU, simplest) — one serialized GPU queue,
  no extra service, no extra network hop.
- [x] Add a `rerank=true` flag on `/search`; the gateway passes it through as a
  tri-state (`X-RAG-RERANK: on|off`, absent sends no field) and the pool key travels
  only with `rerank: true`. `rerank_score` is a new optional field; `score` keeps its
  measured P0 meaning, so `min_score` is unaffected.
- [x] Choose the pair text the model scores: `api` accepts `rerank_query` (capped
  4000 chars), the gateway sends `X-RAG-RERANK-QUERY: retrieval|question`, `retrieval`
  sends no field, and the pair travels only with `rerank: true`. Measured: the knob
  works on the case it was built for and loses on the set, so the default is
  `retrieval` (see Notes).
- [x] Add a fallback: if the reranker fails, return the fused order unchanged
  (missing model, load failure, `RERANK_TIMEOUT`, score-count mismatch — never an error).
- [x] Report latency impact (p50/p95) and VRAM usage.

**Acceptance criteria:**
- [x] `recall@10` does not regress; `MRR@10` / precision at the injected `k` improves on
  the evaluation set. Measured on the live stack: single-turn `recall@10 0.867 → 0.917`,
  `MRR@10 0.486 → 0.649`, `nDCG@10 0.642 → 0.804`; follow-ups `MRR@10 0.4529 → 0.7028`,
  `nDCG@10 0.7103 → 0.8747`, `recall@5 0.778` (fused `0.778`, `bge` `0.667`).
  `recall@10` on follow-ups is `0.889` against the dense-only committed `1.000` —
  that is the P2.1 fusion cost, not the reranker's, and it is the same at `rerank=off`
  on the same gateway request.
- [x] Search still works with the reranker disabled and when it is down (tested at the
  gateway: `off` sends an explicit `rerank: false`, no header sends no field, and
  `rerank=true` with no model loaded is answered as a fused result, never an error).
- [x] VRAM fits alongside the embedding model on 8 GB (`RTX 4060 Laptop 8188 MiB`:
  `7671 MiB` used / `286 MiB` free with `Qwen3-Embedding-0.6B` + `Qwen3-Reranker-0.6B`
  resident; `RERANK_ENABLED=false` frees the reranker's pages).

**Notes:** reranking 50 candidates with a 0.6B cross-encoder costs more than embedding
one query; measured p50 `1.49s` on `Qwen3-Reranker-0.6B` against `0.59s` on
`bge-reranker-v2-m3` and `0.087s` fused. Consider running the reranker on CPU if VRAM is
tight. RRF is rank-based, so an intruder from the BM25 branch sits at a fixed high rank
(16th with `top_k=8`), not just below the cut — a cross-encoder that re-scores content,
not position, is what removes it. The measured case is `mt-003` in **P2.1**: dense rank
9, hybrid rank 13. **The reranker does not fix `mt-003` on its own, and the query
prompt does not either** (`Qwen3` with `RERANK_QUERY_INSTRUCT=qwen3` leaves it outside
the top ten, and the instruction makes the single-turn grid worse: `recall@10 0.900` vs
`0.917`, `MRR@10 0.621` vs `0.649`). The reranker raises follow-up MRR by moving real
answers up; the intruder has to be removed by the fusion of P2.1, so the default is
`Qwen/Qwen3-Reranker-0.6B` with **plain pairs** (`RERANK_QUERY_INSTRUCT` empty), pool 30,
opt-in, and P1.1 does not claim the `mt-003` criterion.

Which text the model judges is the third knob, and it was measured: `X-RAG-RERANK-QUERY:
question` scores the passages against the user's message instead of the long `concat`
rewrite, which puts `mt-003`'s expected chunk at **rank 1** (it was rank 20 with the
rewrite as the pair, and outside the top ten in every plain and instructed pass). It is
a net loss on the set: `mt-001` rank `1 → 12`, `mt-006` rank `1 → 16`, follow-up
`recall@10 0.889 → 0.778`, `MRR@10 0.703 → 0.509`, `nDCG@10 0.875 → 0.513`. A short
question drops the topical anchor an elliptical follow-up needs, so the same history
terms that push `mt-003` down carry `mt-001` and `mt-006` up. `RAG_RERANK_QUERY`
stays `retrieval`; `question` is a per-request knob for the long-rewrite case.

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
- Follow-up questions retrieve the correct passages in the evaluation set. *(met: `concat` puts the expected section in the top ten for all 9 follow-ups, `recall@10` `1.000`; `off` leaves 2 outside)*
- Single-turn questions do not regress. *(met: 60 single-turn questions through the gateway give identical `recall@10 0.800` / `MRR@10 0.414` with `off` and `concat`, the same twelve top-10 failures as `eval/baseline.json`; the guarantee is also asserted offline in `tests/test_gateway_rewrite.py`)*
- Added latency is bounded and documented. *(met: `concat` adds no upstream call; `llm` adds exactly one, bounded by `RAG_REWRITE_TIMEOUT` (10 s) and cached by `RAG_REWRITE_CACHE_SIZE`; the measured run-to-run latency difference is upstream-dominated, documented in README and `eval/README.md`)*

**Notes:** implemented in `gateway/main.py` (`_prepare_retrieval_query`, `_is_follow_up`, `_concat_query`, `_rewrite_query_llm`): requests without history and long non-elliptical questions are never rewritten, `llm` is time-boxed by `RAG_REWRITE_TIMEOUT` (10 s), falls back to `concat` on failure, and rewrites are cached in memory (`RAG_REWRITE_CACHE_SIZE`, default 256). The gateway retrieval path is verified live against a self-hosted OpenAI-compatible server (`UPSTREAM_BASE_URL=http://10.0.0.99:5001/v1`): non-stream and streaming chat both returned grounded answers citing the retrieved section, `X-RAG-Sources` decoded to valid entries, `X-RAG: off` produced `X-RAG-Hits=0`, and an unrelated question produced 0 hits. The default is `concat` in the gateway code, `docker-compose.yaml` and `.env.example`, measured on `eval/questions.multiturn.jsonl` (9 follow-up conversations, gateway mode, 20 candidates requested, threshold disabled, frozen corpus, one session):

| mode | recall@5 | recall@10 | MRR@10 | nDCG@10 | latency avg / p95 |
|---|---|---|---|---|---|
| `off` | 0.667 | 0.778 | 0.277 | 0.392 | 4.45 s / 4.94 s |
| `concat` (default) | 0.778 | 1.000 | 0.420 | 0.770 | 4.66 s / 4.95 s |
| `llm` | 0.889 | 1.000 | 0.514 | 0.781 | 4.09 s / 4.91 s |

Single-turn non-regression is measured, not assumed: 60 single-turn questions through the gateway give `recall@10 0.800` / `MRR@10 0.414` in both `off` and `concat` (`eval/gateway-singleturn-off.json`, `eval/gateway-singleturn-concat.json`), and the guarantee is also covered offline by `tests/test_gateway_rewrite.py` (16 tests: first turn untouched, `concat` joins recent user turns only, `llm` fallback and cache, no upstream call for non-follow-ups). Latency: upstream generation dominates (~4 s per question on the measured server) and the extra `llm` call was not separable from that noise in these runs (the three runs measured 4.09–6.08 s average, 4.91–7.01 s p95, `llm` the fastest), so the cost of `llm` is documented as one extra upstream call per rewritten follow-up, bounded by `RAG_REWRITE_TIMEOUT`, not as a measured millisecond figure. Threshold tuning matters more for follow-ups than for single-turn questions: with rewrite `off`, `RAG_MIN_SCORE=0.5` strips all context from 78% of the follow-ups and `0.6` from 100%; with `concat`, `0.5` strips 11% and `0.6` 67%; with `llm`, 22% and 67%. At `min_score=0.3` zero-context is 0% in every mode, and recall@8 at that threshold is 1.000 (`llm`), 0.889 (`concat`), 0.667 (`off`) — `RAG_MIN_SCORE=0.3` is kept, and `llm` is the mode that gains most from a wide `top_k`. The gateway reports its effective mode in `/health` (`defaults.query_rewrite` + `rewrite_history`), the harness records it in every gateway run as `gateway_defaults` and prints it before the run, so each committed result file says which mode produced it; `eval/README.md` documents that the mode is read at container creation, so a change needs `docker compose up -d --force-recreate gateway`, and that two eval sessions must not share one gateway container (a run measured against a container recreated by another session reports the mode it did not run with).

---

## P2 — Retrieval breadth and reliability

### [x] P2.1 Hybrid search: BM25 + dense with RRF — done, measured gains, one multiturn recall loss

**Why:** dense embeddings miss exact identifiers, product codes, and rare names; BM25 covers them.

**Scope:**
- [x] Migrate the collection to named vectors: `dense` (1024) + `bm25` (sparse).
- [x] Compute sparse vectors with FastEmbed `Qdrant/bm25` (add the dependency to both services that write vectors).
- [x] Query with Qdrant `prefetch` for both branches and `FusionQuery(fusion=RRF)`.
- [ ] Apply workspace/collection/source/type filters to both branches (shared with P4.1).
- [x] Re-index the corpus; keep the model/chunker signature in the payload.
- [x] Evaluate on identifier-heavy questions vs dense-only.

**Acceptance criteria:**
- [x] Exact-term questions improve on the evaluation set.
- [x] Dense-only quality does not regress for paraphrased questions.
- [x] Collection schema and re-index procedure are documented.

**Notes:** this is a breaking collection change; plan a re-index window. Keep the dense branch as a fallback if sparse vectors are missing.

Implemented in `api/hybrid.py` (stdlib fusion), `api/main.py` (two Qdrant branches + `/embed` returns dense and BM25), `api/embedder.py` (`SparseEmbedder`), `embedder/vector_db.py` (named `dense` + sparse `bm25` with `modifier=IDF`, schema migration by recreating the collection), `embedder/embedder.py` (`embed_with_sparse`), `embedder/main.py` (`SPARSE_MODEL` in the file signature). The dependency lives where vectors are produced: `api/requirements.txt` has `fastembed>=0.3.7` (CPU ONNX); `embedder` receives BM25 vectors over `POST /embed`, so it still has no torch and no model. `SPARSE_WEIGHT`, `SPARSE_MIN_RELATIVE`, `SEARCH_MODE`, `SPARSE_ENABLED`, `SPARSE_MODEL`, `BM25_THREADS` are in `docker-compose.yaml` and `.env.example`.

The planned Qdrant-side fusion (`prefetch` + `FusionQuery(fusion=RRF)`) was verified live on a throwaway collection (the client model is `Prefetch`, not `PrefetchQuery`; `Fusion.RRF` works with two prefetches, dense `score_threshold` works inside a prefetch) but is **not** what ships: RRF scores are rank-based, so a `min_score=0.3` threshold applied to a fused score means something different from 0.3 cosine. Fusion therefore runs in `api`: each branch is queried separately (`using="dense"`, `using="bm25"`), and `apply_threshold` filters on the dense cosine score of every fused hit, keeping BM25-only hits (they have no cosine score and are exactly the exact-identifier hits the dense branch misses). `POST /search` exposes `mode`, `sparse_weight`, `sparse_min_relative` for the harness; `score` in the response is the dense cosine when the dense branch found the chunk, otherwise the BM25 weight, and `fusion_score` is the RRF rank score.

Measured on the live stack (202 points, 5 corpus files, dense `Qwen3-Embedding-0.6B` fp16 CUDA + FastEmbed `Qdrant/bm25`), 2026-11-16; the full grid with the source file of every number is [`eval/p21-sweep.md`](eval/p21-sweep.md), the runs are committed as `eval/p21-search-*.json` and `eval/p21-gateway-*.json`. Selected `SPARSE_WEIGHT=1.0`, `SPARSE_MIN_RELATIVE=0.7` (60 single-turn questions, `POST /search`, top_k=10, no threshold):

| | recall@5 | recall@10 | MRR@10 | nDCG@10 | p95 |
|---|---|---|---|---|---|
| dense-only | 0.683 | 0.800 | 0.414 | 0.557 | 0.064 s |
| hybrid (selected) | **0.750** | **0.883** | 0.478 | 0.644 | 0.100 s |

Per question: 21 improved, 27 unchanged, 12 regressed; 8 questions dense-only missed entirely are inside the top 10 (`q-003`, `q-009`, `q-028`, `q-029`, `q-032`, `q-039`, `q-054`, `q-055`), 4 dense-only hits dropped out (`q-004` 5→out, `q-008` 9→out, `q-012` 5→out, `q-037` 1→out). Subset by ground-truth shape: exact identifiers / code-like strings (n=37) `0.811→0.838` recall@10 and `0.451→0.461` MRR@10; paraphrased questions (n=23) `0.783→0.913` recall@10 and `0.355→0.480` MRR@10 — so the paraphrase subset does not regress, it improves. The identifier subset gains less than expected because the corpus is 5 documentation files whose "identifiers" (`RAG_TOP_K`, `QDRANT_VERSION`, `mem_limit`) are dense-retrievable already; the sparse branch mostly re-orders them. A corpus with product codes and ticket IDs is where BM25 should dominate, and P1.2 (structure-aware chunking acceptance) plus P4.1 (workspaces) are the inputs for that. Through the gateway, 60 single-turn questions at top_k=20 with rewrite `concat`: recall@10 `0.800→0.883`, MRR@10 `0.414→0.490`.

Follow-up questions are the cost: 9 conversations through the gateway, rewrite `concat`, top_k=20 — dense-only `recall@10 1.000` / `MRR@10 0.420`; selected hybrid `recall@10 0.889` / `MRR@10 0.453`. `mt-003` (a long rewritten follow-up) is pushed from rank 9 to 13 by BM25 term overlap with many chunks; `mt-009` improves 4→1 and `mt-004` 6→3, which is why MRR rises while recall@10 drops. A floor of 0.85 keeps recall@5 at 0.556 with MRR 0.350, and `SPARSE_WEIGHT=1.5` gives the same numbers as `1.0` on this set, so the floor is the knob that matters and 0.7 is the shipped compromise. The reranker (P1.1) is the intended fix for the long-rewritten follow-up case, and `llm` rewrite keeps `recall@10 1.000` with dense retrieval.

`tests/test_hybrid.py` (23 tests: RRF math, tie-breaking, top_k limit, the relative floor, the threshold keeping BM25-only hits, `POST /embed` sparse alignment across batches, dense-only responses, mismatched sparse count as a hard error) and `tests/test_vector_db_schema.py` (12 tests: named-vector creation, legacy unnamed collection recreated, missing sparse vector recreated, BM25 disabled keeps a dense-only collection, dimension change recreated, both vectors written, dense-only upsert, sparse length mismatch, citation and signature payload, stable point ids) run stdlib-only with a stubbed `qdrant_client`: 123 tests pass. Re-index procedure: [`README.md`](README.md) → *Collection schema (named dense + BM25)*, with the snapshot-before / re-index / `reconcile_check.py audit` / dense-vs-hybrid eval sequence. The live collection is migrated and verified: `dense` dim 1024 + `bm25` with `modifier=idf`, 202 of 202 points carry a sparse vector, BM25 queries on `RAG_TOP_K`, `QDRANT_VERSION`, `pgvector`, `BM25` return the expected chunks, and an unrelated phrase returns 0 hits. The shared workspace/collection/source/type filters are not implemented (P4.1): the search path has no filters today, so the "both branches" requirement is recorded here and lands with P4.1.

### [x] P2.2 Reliability: healthchecks, pinned versions, retries

**Why:** Compose starts services before they are ready; `api` may still be downloading weights, and Qdrant was pulled as `latest`.

**Scope:**
- [x] Pin `qdrant/qdrant` to a specific version (`v1.20.1`, overridable via `QDRANT_VERSION`).
- [x] Add an `api` healthcheck that requires `embedder == "ready"` (plain `/health` returns 200 even while loading).
- [x] Add a Qdrant healthcheck. The official image has no `curl`/`wget`, so it uses a TCP check (`bash /dev/tcp`).
- [x] Add `depends_on` conditions: `api` and `embedder` wait for Qdrant; `embedder` waits for `api`; `gateway` starts early and degrades gracefully.
- [x] Add retry/backoff around Qdrant connection setup in the indexer.
- [x] Document the startup sequence and expected first-boot duration (~1.2 GB download + model load).
- [x] Rehearse a Qdrant version change against the data the current stack wrote (`scripts/qdrant_upgrade_check.py`, volume copy + snapshot restore + self-vector query).

**Acceptance criteria:**
- [x] Fresh `docker compose up` reaches a working state without a crash loop.
- [x] `docker compose ps` shows meaningful health states.
- [x] Upgrading Qdrant is a deliberate, tested action.

**Notes:** implemented in `docker-compose.yaml` (`QDRANT_VERSION`, healthchecks, `depends_on`) and `embedder/main.py` (`wait_for_qdrant`). E2E verified live on this machine with the GPU stack: `docker compose up -d` reached a healthy state without a crash loop, `docker compose ps` showed `qdrant`, `api`, `gateway` healthy and `embedder` running, `api` logged `Startup complete`, `embedder` logged stale-vector reconciliation and the watcher (`Started watcher on /data/documents (debounce 1.0s)`), and `python scripts/smoke_test.py` passed all 5 checks against the gateway with a real LLM upstream. There was no `qdrant_data` on this machine, so there was no previously running tag to preserve; `v1.20.1` was the latest stable release at implementation time. Storage now lives on a named volume (see P2.3 notes).

Version changes are rehearsed with `scripts/qdrant_upgrade_check.py` (stdlib only, docker CLI): it copies the compose volume (`rag_qdrant_data`) into a scratch volume, starts the target image on the copy, restores a snapshot into a fresh volume of the target image, compares `status` / `points` / `dim` / distinct `payload.file` and a self-vector query (one stored vector queried back, its own point must be the top hit) with the live collection, then removes the sandbox. The live stack keeps running and its volume is never written. Measured on this machine against the live collection (`green`, 202 points, dim 1024, 5 file paths):

- `--image qdrant/qdrant:v1.20.1 --check volume` → `PASS`, numbers identical to the baseline, self-vector query score 1.0 (the pinned tag reads its own storage).
- `--image qdrant/qdrant:latest --check volume` (that tag is Qdrant **1.15.5**, about 13 months older than the pinned tag) → `PASS`: `green`, 202 points, dim 1024, 5 files, self-vector query score 1.0, no warnings in the container log. So for this schema (one dense cosine vector, no quantization, no sparse vectors, plain payload indexes) the older server opened storage written by `v1.20.1`. This is not a general guarantee for other schemas, so the check must be re-run before any bump.
- `--image qdrant/qdrant:latest --check restore --snapshot backups/documents-....snapshot` (3.9 MB, created by `v1.20.1` at 202 points) → `PASS`: the older server restored all 202 points, dim 1024, 5 file paths, self-vector query score 1.0. The snapshot is portable to that older version, which is what a rollback path needs.

`tests/test_qdrant_upgrade_check.py` (14 tests) covers collection stats with scroll pagination, the self-vector query, the PASS/FAIL verdict, the compose volume name lookup, the readiness poll budget, and sandbox cleanup. A run interrupted by a timeout left a sandbox container behind, so `cleanup()` now runs at the start of every run, and the readiness poll uses a short timeout because a freshly published Docker port on Windows takes a while to accept a connection.

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

### [x] P2.4 Indexing vs query contention

**Why:** bulk indexing and user queries share one GPU and one model; a large ingest delays search.

**Scope:**
- [x] Measure search latency during a bulk ingest.
- [x] Options: lower indexer batch size, sleep between batches, a priority queue in `api`, a separate embedding instance for indexing, or CPU embedding for the indexer. **Measured as unnecessary**: search did not degrade during bulk indexing, so none of these is shipped and the serialized queue stays at 1.
- [x] Add a concurrency limit around `model.encode` so concurrent requests do not pile up.
- [x] Expose queue/latency metrics (at least in logs).

**Acceptance criteria:**
- Search p95 latency stays within an agreed budget during indexing.
- Indexing still completes in a reasonable time.

**Budget, signed off against the measurement (2026-11-18):** rerank-on search
`p95 <= 1.5s` while the indexer is busy (measured `p95 1.05s` inside a 20s window with
60 chunks added), fused search `p95 <= 0.10s` (measured `0.094s` idle, `0.052s`
settled). Indexing throughput `~1.5 files/s` at `EMBED_BATCH_SIZE=32` (chunks upserted
every ~0.5s per file). Measured on one warmed connection, pool 30,
`Qwen/Qwen3-Reranker-0.6B`: **search did not degrade during bulk indexing** —
`p50 x0.97 / p95 x1.06` against its own idle latency — so no queue priority, smaller
batch, sleep between batches, second embedding instance, or CPU embedding for the
indexer is needed. The serialized queue (`EMBEDDING_MAX_CONCURRENCY=1`) is what
keeps it that way, and the price of keeping rerank on the request path is
`p50 x11 / p95 x10` against fused in the same run.

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

Measured 2026-11-18: that sample reads `idle p50 21.24s` / `ingest p50 21.27s` /
`settled p50 21.28s`, so the ingest multiplier is `x1.00` in absolute terms and the
number is connection setup, not the GPU queue — `measure_search` now reuses one
keep-alive connection per worker through `eval_retrieval.Session`, which is what
makes idle-vs-ingest a measurement of contention. Re-measuring on a quiet stack with
`--rerank-compare` (the reranker's cost and the ingest multiplier in one run), and the
budget is recorded as `p95 <= <value>` with the ingest throughput that paid for it.

Measured 2026-11-18 on a quiet stack, one connection, warmed, `--rerank-compare`,
pool 30, `Qwen/Qwen3-Reranker-0.6B`: fused `p50 0.047s / p95 0.059s`, rerank
`p50 0.855s / p95 0.930s` — **the price of scoring the pool is p50 ×18.3, p95 ×15.7**.
The probe is bounded by `--budget` (60s default): every loop, including the indexer
wait, stops at the deadline, and every phase prints its numbers and the time left, so
an over-budget run is diagnosed at the phase, not watched. Two mechanics that the
first bounded runs showed:

- **a connection that has not been used costs ~20s on the first request**, the same
  request on a warmed connection ~1s (measured: 22.15s, 22.17s, then 0.89s on the
  second request of the same connection). The probe warms every connection before
  anything is timed, so a phase reads the GPU queue and not connection setup.
- **the probe deletes its files at the end, which kills the overlap**: the indexer
  works through files at ~1.5 files/s (`Upserted 4 chunks` every ~0.5s), so files
  deleted by the probe before the indexer reaches them leave the ingest window with
  nothing to index, and `chunks_added=0` reads as idle. `--keep` holds the files, so
  the window overlaps real indexing; the points delta is sampled at the window
  boundary, not waited for.

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

### [x] P3.2 Query embedding cache

**Why:** repeated questions still run a full forward pass.

**Scope:**
- [x] Add an LRU cache keyed by `(model, prompt_name, normalized_text)`. **`normalized_text` is the query with surrounding whitespace stripped, not a case-folded string**: lowercasing changes the embedding, so a cached answer would be a vector for a different text.
- [x] Bound memory (vectors are ~4 KB each at 1024 dims).
- [x] Add hit/miss counters to `/health` or logs.
- [x] Consider caching at the gateway for full user messages. **Measured as not needed in P3.2**: `/search` has exactly one query text, so a gateway cache would be a second copy of the same lookup, and a gateway key must also carry conversation history. Deferred with P9.2 (result caching), whose key needs retrieval configuration, workspace/collection filters, embedding profile, and index version/state.
- [ ] For result caching (P9.2), keys must include retrieval configuration, workspace/collection filters, embedding profile, and index version/state.

**Acceptance criteria:**
- Repeated identical queries hit the cache and skip `model.encode`.
- Memory stays bounded under a long-running workload.
- Cached results are never returned for a different workspace/collection scope.

**Measured against the running stack (2026-11-18, `api` rebuilt, collection of 202
points, `Qwen3-Embedding-0.6B`, dim 1024, `EMBEDDING_CACHE_SIZE=256`):**
- Two identical `POST /search`: `hits +1, misses +0` — the second query cost no
  forward pass. Padded with spaces, the same question is one entry (`strip` is part
  of the key): `hits +2, misses +0`.
- 300 distinct queries with `rerank: false` in **29.3 s**: `entries` reached `256` and
  stayed there, `approx_bytes` `1,077,248` = exactly the bound
  `256 x (1024x4 + 112)`. The oldest entry (`probe query 000`) encoded again on a
  re-query (`misses +1`) and the newest was answered from the cache (`hits +1`), so
  LRU eviction is in effect, not just a counter.
- `4001`-char `query` is rejected with `422` before encoding, `4000` is accepted: the
  memory bound includes the key text.
- An all-whitespace query returns hits and is counted as one miss: the model receives
  a single space, not an empty string (the guard `/embed` and `/v1/embeddings` use).

**Budget, measured against the run (2026-11-18):** the whole suite is `298` tests in
`30.5s` (`<= 60s`); the cache file is `16` tests in `0.14s` (`2.6s` wall, the
`torch` import is the cost); `api` rebuild `~1min` cached; the live pair check `2.4s`;
the 300-query bound loop `29.3s` with `rerank: false` (`~0.94s` per request with
rerank on does not fit a 3-minute budget, so the loop measures the cache leg).
Health checks use `eval_retrieval.Session` (one warmed connection), not
`Invoke-WebRequest`.

**Notes:** the cache is in `Embedder.embed_query` in `api/embedder.py`, not in
`api/main.py`: the model path and the resolved prompt are known there and every query
call passes through one method. A hit is answered **before** the `EMBEDDING_MAX_CONCURRENCY`
semaphore, so a repeated question does not wait behind a bulk ingest — this is the
P2.4-relevant win. `embed_query` takes one string and returns shape `(1, dim)`
(`/search` indexes `vectors[0]`); the batch-miss logic is removed, because `/search`
has exactly one query text and nothing else calls this method. The cached value is a
copy (`np.array(..., float32)`), so the caller cannot mutate the cache and no batch
array stays alive behind an entry. `cache_size=0` disables it. Counters count
**positions**, not distinct texts, and `/health` reports `embedder_cache`
(`enabled`/`entries`/`max`/`hits`/`misses`/`vector_dim`/`approx_bytes`). Not cached:
documents (`POST /embed`), `POST /v1/embeddings`, and BM25 (`SparseEmbedder`). The
cache is one `api` process, cleared on restart; there is no shared cache and no TTL.
`SearchRequest.query` is capped at `max_length=4000` (the repo's own cap:
`rerank_query`, `RERANK_QUERY_MAX_CHARS`; the gateway `concat` rewrite is capped at
2000), which is what makes the memory bound real. `sentence_transformers` is imported
inside `Embedder.__init__` (the convention `SparseEmbedder` uses), so the cache rules
are tested against a fake model that counts forward passes, with no GPU, Docker, or
model download. `tests/test_embedding_cache.py` covers repeat, shape, whitespace keys,
case, the empty-string guard, copy-on-store, LRU bound and eviction order, `cache_size=0`,
model change, prompt change, a hit answering while the encode queue is held, documents
bypassing the cache, and the `/health` counters.

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
