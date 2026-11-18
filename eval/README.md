# Retrieval evaluation

`scripts/eval_retrieval.py` measures retrieval quality against a labelled
question set. Run it after every change to the model, prompt, chunking, or
retriever, and keep the numbers in the commit message.

Two retrieval modes:

| Mode | What it measures | Endpoint |
|---|---|---|
| `search` (default) | the embedding API itself, with no score threshold hiding results | `POST /search` on the API (port 8000) |
| `gateway` | the gateway's own retrieval path: follow-up rewriting, the "fresh user turn" rule, injection | `POST /v1/chat/completions` on the gateway (port 8001), sources read back from `X-RAG-Sources` |

## Dataset format

One JSON object per line (JSONL). Example:

```json
{"id": "q-001", "question": "Who is Alice Davince?", "relevant": [{"file": "example.txt", "text_contains": "Alice Davince"}]}
```

Fields:

| Field | Required | Description |
|---|---|---|
| `id` | no | stable identifier; defaults to `q001`, `q002`, ... |
| `question` | yes | the query sent for retrieval |
| `relevant` | yes | non-empty array of expected hits |
| `history` | no | array of strings: previous user turns, oldest first (gateway mode) |

Each expected hit may use any combination of these fields; all provided fields
must match:

| Field | Description |
|---|---|
| `file` | file path as stored in the payload, or just its basename (`example.txt`) |
| `chunk` | zero-based chunk index |
| `text_contains` | case-insensitive substring of the chunk text |
| `section` | case-insensitive substring of the section path (`Gateway variables > RAG`), present in both modes |

`text_contains` is usually the most stable criterion: chunk indices change when
the chunker changes, while a distinctive phrase does not. Use `file` alone when
any chunk of the document is acceptable, and `section` when the answer is tied to
a heading rather than to exact wording.

**`text_contains` in gateway mode needs `--corpus`**: `X-RAG-Sources` carries file,
chunk, score, title, section and character offsets, not the chunk text. With
`--corpus` the harness reads the file from the corpus snapshot, normalizes it the
way the chunker does (CRLF/CR to LF), and slices `char_start:char_end` to recover
the chunk text, so `text_contains` works in gateway mode as well. Without
`--corpus` the loader rejects `text_contains` in gateway mode; use `file` and
`chunk` there.

`eval/questions.example.jsonl` contains schema examples for the sample document.
`eval/questions.jsonl` is the real question set used for the baseline.
`eval/questions.multiturn.jsonl` is the follow-up set: each line carries `history`
(the previous user turns) plus a short follow-up question, and its ground truth
uses `file` + `section` only, so the same file works in both modes. It measures
what retrieval does when the user asks "and the min score?" without repeating the
topic — the gateway's follow-up rewrite (P1.3) is what changes it.

## Corpus

`eval/corpus/` holds the snapshot of the documents the baseline was measured on.
It is committed so a fresh clone reproduces the same numbers; `data/documents/`
is git-ignored, so without the corpus the question set refers to files that do
not exist.

```bash
# populate the watched folder from the committed corpus
cp eval/corpus/* data/documents/
```

On PowerShell:

```powershell
Copy-Item eval/corpus/* data/documents/
```

The corpus is a **snapshot**, not a symlink to `README.md` / `TODO.md`. After the
documents in the repository change, validate the question set against the corpus:

```bash
python scripts/eval_retrieval.py --questions eval/questions.jsonl --corpus eval/corpus --validate-only
```

This is offline (no API calls): it reports every `file` missing from the corpus
and every `text_contains` that no longer occurs in it. If validation fails,
either update the questions and re-snapshot the corpus, or the baseline is no
longer reproducible.

## Running

```bash
# validate the dataset (and the corpus) without calling the API
python scripts/eval_retrieval.py --questions eval/questions.jsonl --corpus eval/corpus --validate-only

# full run against the API (default: http://localhost:8000)
python scripts/eval_retrieval.py --questions eval/questions.jsonl

# save the baseline (aggregate metrics + per-question results + sweep grid)
python scripts/eval_retrieval.py --questions eval/questions.jsonl --sweep \
    --json-out eval/baseline.json

# fail the run if quality drops below the recorded gates
python scripts/eval_retrieval.py --questions eval/questions.jsonl \
    --min-recall 0.70 --min-mrr 0.35

# gateway behavior, including multi-turn follow-ups
python scripts/eval_retrieval.py --questions eval/questions.jsonl --mode gateway \
    --api-key sk-local --sweep

# follow-up questions with conversation history through the gateway
python scripts/eval_retrieval.py --questions eval/questions.multiturn.jsonl \
    --mode gateway --sweep --json-out eval/multiturn-gateway.json
```

Useful options:

| Option | Default | Description |
|---|---|---|
| `--mode` | `search` | `search` (embedding API) or `gateway` |
| `--api-url` | `http://localhost:8000` | embedding API base URL |
| `--gateway-url` | `http://localhost:8001` | gateway base URL |
| `--api-key` | — | bearer key for the gateway |
| `--model` | first from `/v1/models` | model id for gateway mode |
| `--top-k` | `50` (search) / `20` (gateway) | candidates requested per question |
| `--recall-k` | `5,10` | comma-separated cutoffs for recall |
| `--limit` | — | evaluate only the first N questions |
| `--timeout` | `120` | per-request timeout, seconds |
| `--corpus` | — | validate ground truth against a corpus directory |
| `--sweep` | off | apply the `top_k` × `min_score` grid to the retrieved candidates |
| `--sweep-top-k` | `3,5,8,10` | grid columns for `top_k` |
| `--sweep-min-score` | `0.3,0.4,0.5,0.6,off` | grid rows for `min_score` (`off` = no threshold) |
| `--json-out` | — | write aggregates, per-question results and the sweep grid |
| `--show-failures` | `20` | number of failing questions to print |
| `--retrieval-mode` | — (`api` default) | `hybrid` or `dense`, sent as `mode` to `/search` (search mode only) |
| `--sparse-weight` | — (`SPARSE_WEIGHT`) | RRF weight of the BM25 branch, sent to `/search` (search mode only) |
| `--sparse-min-relative` | — (`SPARSE_MIN_RELATIVE`) | drop BM25 hits below this fraction of the best BM25 score (search mode only) |
| `--rerank` | `default` | `on` asks for cross-encoder reranking, `off` forces it off, `default` sends no field at all (`RERANK_ENABLED` / `RAG_RERANK` decides) |
| `--rerank-top-n` | — (`RERANK_TOP_N`, 30) | candidates the cross-encoder scores per request, `1..100` |
| `--sweep-rerank-top-n` | `30` | pools to run as extra retrieval passes with `--sweep --rerank on` |
| `--min-recall` | — | exit 1 if `recall@max(recall-k)` is below the value |
| `--min-mrr` | — | exit 1 if `MRR@10` is below the value |

## Sweep: tuning `top_k` and `min_score`

`--sweep` applies the grid **to the candidates of a single retrieval pass**: the
API is called once per question, and every `(top_k, min_score)` combination is
computed offline, so a sweep does not cost one run per configuration. It
reproduces exactly what the gateway does: cut the candidate list at `top_k`, drop
hits below `min_score`, inject what is left.

Three tables are printed:

- **`recall@k`** — the relevant chunk is inside the injected context.
- **`injected@k`** — average number of chunks injected per question (context size).
- **`zero_context@k`** — share of questions that get no context at all (a
  threshold that is too high looks "safe" and quietly disables retrieval).

Reading the table: pick the smallest `top_k` whose `recall@k` is flat, then the
largest `min_score` that keeps `recall@k` and does not raise `zero_context`. The
sweep results go into `--json-out`, so the chosen values are justified by the
committed numbers, not by intuition.

`min_score` is model-dependent: a new embedding model needs a fresh sweep. The
query-side instruction is part of the configuration, so compare
`EMBEDDING_QUERY_PROMPT=query` with `EMBEDDING_QUERY_PROMPT=` (two runs, with
`docker compose up -d api` between them) and keep the value that wins.

### Sweep for the hybrid branch

The BM25 branch adds two ranking knobs that the `top_k` × `min_score` sweep does not
cover, so they are measured with per-request overrides on `POST /search`:

- `--sparse-weight` — the RRF weight of the BM25 branch (the dense branch is fixed at `1.0`). Above `1.0` an exact-term hit counts for more than a dense neighbour; below `1.0` a BM25-only hit cannot outrank a chunk the dense branch placed near the top.
- `--sparse-min-relative` — a relative floor on the BM25 score: hits below this fraction of the best BM25 score for the query are dropped, which removes weak term overlap (one shared common word). `min_score` is never applied to BM25: it filters the dense cosine score, and a BM25-only hit has no cosine score, so it always passes.

The gateway has no way to pass them, so a gateway run uses the service defaults (`SPARSE_WEIGHT`, `SPARSE_MIN_RELATIVE`), which are read when the container is created: a change needs `docker compose up -d --force-recreate api`, and `/health` reports the effective `search_mode`.

The grid is in [`p21-sweep.md`](p21-sweep.md) with the file each number came from. Reading it: the floor matters more than the weight (`0.7` is the best recall@10), and `weight=1.5` buys MRR/nDCG at the cost of recall, so `1.0` stays the default. The follow-up set is the binding one: BM25 gains on long rewritten questions with many common terms (`mt-003`: rank 9 → 13), which is what the reranker (P1.1) is for.

### Sweep for the reranker (P1.1)

The reranker changes the order, so the harness has to score the order the reranker
produced. `rank_hits` ranks the injected chunks by `rerank_score` when the response
carries it, and by `score` (the fused order) when it does not, so a reranked pass is
measured on the order the gateway injects, not on the RRF position the candidate came
in at. `rerank_top_n` is the pool the model sees, so pool size is a **retrieval pass**
axis, not a grid row: `--sweep-rerank-top-n 30,50` calls the API once per pool, and the
grid rows are labelled `0.30 rerank=30`, `0.30 rerank=50`, and unlabelled rows for the
pass that never asked for reranking.

```powershell
Copy-Item eval/corpus/* data/documents/
docker compose up -d --force-recreate api        # RERANK_ENABLED=true, RERANK_MODEL set
python scripts/eval_retrieval.py --questions eval/questions.jsonl --limit 1   # warm the connection
python scripts/eval_retrieval.py --questions eval/questions.jsonl --corpus eval/corpus \
    --rerank on --rerank-top-n 30 --json-out eval/rerank-on-30.json
python scripts/eval_retrieval.py --questions eval/questions.jsonl --corpus eval/corpus \
    --rerank on --sweep-rerank-top-n 30,50 --sweep --sweep-md eval/sweep-rerank.md \
    --json-out eval/rerank-sweep.json
python scripts/eval_retrieval.py --questions eval/questions.jsonl --corpus eval/corpus \
    --rerank off --json-out eval/rerank-off.json          # same service, rerank forced off
python scripts/eval_retrieval.py --questions eval/questions.multiturn.jsonl --mode gateway \
    --corpus eval/corpus --rerank on --sweep-rerank-top-n 30,50 --sweep \
    --json-out eval/rerank-multiturn.json
```

For a model trained on an instructed pair, measure it both ways, because the prompt is
a quality change with a latency cost of its own:

```powershell
$env:RERANK_QUERY_INSTRUCT = "qwen3"
docker compose build api                          # the new code, not the image built earlier
docker compose up -d --force-recreate api         # /health reports rerank_query_instruct
python scripts/eval_retrieval.py --questions eval/questions.jsonl --corpus eval/corpus \
    --rerank on --top-k 50 --sweep-rerank-top-n 30,50 --sweep \
    --sweep-md eval/sweep-qwen3-instruct.md --json-out eval/rerank-qwen3-instruct-sweep.json
python scripts/eval_retrieval.py --questions eval/questions.multiturn.jsonl --mode gateway \
    --corpus eval/corpus --rerank on --top-k 20 --sweep-rerank-top-n 30,50 --sweep \
    --json-out eval/rerank-qwen3-instruct-multiturn.json
```

Measure which text the model judges, because a long rewrite is a pair of two texts and
the history in it is term noise:

```powershell
# quality passes use the stub upstream: the gateway fills X-RAG-Sources before
# generation and the rewrite is concat, so a real LLM adds generation time only
# (a 4B model with ~5000 tokens of injected context) and no extra information.
python scripts/upstream_stub.py --port 5011              # answers instantly
$env:UPSTREAM_BASE_URL = "http://host.docker.internal:5011/v1"   # compose reads shell env first
docker compose up -d --force-recreate gateway            # .env keeps the live LLM URL
python scripts/eval_retrieval.py --questions eval/questions.multiturn.jsonl --mode gateway \
    --corpus eval/corpus --rerank on --top-k 20 --rerank-query question \
    --sweep-rerank-top-n 30,50 --sweep --sweep-md eval/sweep-qwen3-question-multiturn.md \
    --json-out eval/rerank-qwen3-question-multiturn.json
# the single case it fixes, read from the diagnosis tool (no gateway, no upstream):
python scripts/diagnose_rerank.py --question-id mt-003 --mode search \
    --retrieval-query concat --rerank on --rerank-top-n 30 --top-k 20 \
    --rerank-query question --show-text 0        # first_rank 1, was 20 with the rewrite as the pair
```

Three numbers say whether the reranker is worth the latency:

- `rerank_rate` — the share of questions that were actually reranked. `0` on a `rerank=N`
  row means the model was asked for and did not run (load failure, timeout over
  `RERANK_TIMEOUT`, a score count that did not match the candidates), so that row is the
  fused baseline, not a reranked result.
- `MRR@10` with `--rerank on` against `--rerank off` on the same deployment — the P1.1
  acceptance criterion is that ranking improves, and the single-turn set has to show no
  recall regression.
- `mt-003` — the BM25 term-noise follow-up that went from rank 9 to 13 when the hybrid branch arrived. It is
  the question the reranker exists for, so `--show-failures` output has to show it back inside the top ten.
  Measured 2026-11-18 with `BAAI/bge-reranker-v2-m3`: it is **not** in the top ten, at pool 30 or pool 50, so the
  criterion is not met by that model on this corpus. `scripts/diagnose_rerank.py` prints what beat it:

```powershell
python scripts/diagnose_rerank.py --question-id mt-003 --rerank on --rerank-top-n 30 --top-k 20
python scripts/diagnose_rerank.py --question-id mt-003 --rerank off --top-k 20
python scripts/diagnose_rerank.py --question-id mt-003 --mode search --retrieval-mode dense \
    --retrieval-query concat --rerank off --top-k 20
```

The first run prints the order the gateway injected with `score` and `rerank_score` side by side, and `*` marks
the chunk the question expects; the second prints the fused order for the same request. The rank change is read
as a cause: which chunk came first, and which number put it there. The third prints the **dense-only** order for
the *same request string*: `--retrieval-query concat` sends the gateway's `concat` rewrite in search mode, so the
comparison is about fusion, not about a shorter query. `--retrieval-mode` is a `/search` knob, so it is refused
in gateway mode.

`min_score` keeps filtering `score`, never `rerank_score`, so a reranker preference cannot
resurrect a hit the threshold removes; the sweep grid stays comparable with the committed
single-turn and multiturn tables. Gateway mode sends `X-RAG-RERANK: on|off` and
`X-RAG-RERANK-TOP-N`, and reads `rerank_score` back from `X-RAG-Sources`, so
`rerank_rate` is recorded in gateway runs too. Every `--json-out` file stores
`rerank`, `rerank_pools`, and per-question `rerank_scores`.

The latency side of the same decision is measured by `scripts/latency_probe.py`, on
one deployment, on the GPU queue the reranker shares with the embedding model. It
runs **small, on the situation being fixed** — search staying responsive while the
indexer is busy — and the whole run is capped by `--budget` (default 60s): every loop,
including the wait for the indexer, stops when the budget is used up, and every phase
prints the time it has left. The defaults are that size, so the command is short —
`--rerank-compare` measures fused and reranked on one card in one run:

```powershell
python scripts/latency_probe.py --rerank on --rerank-top-n 30 --rerank-compare --json-out eval/perf-rerank.json
```

The printed `rerank cost vs fused` multiplier is the price of scoring the pool, and
`ingest` / `settled` show what the price is while the indexer is busy. `chunks_added`
in the `ingest` phase is the overlap check: it counts the chunks the indexer inserted
**during** the search window, so `chunks_added=0` means the window ended before the
indexer started and the numbers say nothing about contention. Measured on the live
stack: rerank on pool 30 costs **p50 x15.4 / p95 x13.3** against fused search
(0.829s against 0.054s), and search under the ingest window stayed at **p50 x1.05 /
p95 x1.15** of its own idle latency. Two measured
facts about the queue, both built into the probe: a rerank request costs ~1s on a
connection that has been warmed and ~20s on one that has not, so **every connection
is warmed before anything is timed**; and the queue is serialized
(`max_concurrency=1`), so `--concurrency 1` reads the queue and more connections add
queue waiting. `/health` on `api` reports `rerank_enabled`, `rerank_model`,
`rerank_top_n`, `rerank_transform`; record those with the numbers, because a rerank
measurement is a property of the model and the dtype, not of the code. VRAM headroom
is read from `nvidia-smi` with both models resident — the reranker is a second model
on the same card as the embedding model, and that is what bounds `RERANK_TOP_N`.

The user-visible number is measured against the **live LLM**, on a small sample, after
the GPU probes finish (embedding and reranker are one serialized queue, so two runs at
once inflate both):

```powershell
python scripts/eval_retrieval.py --questions eval/questions.multiturn.jsonl --mode gateway \
    --corpus eval/corpus --rerank on --top-k 20 --limit 6 --json-out eval/live-latency.json
```


The sweeps that produced the committed values are kept in the repository:

| File | Configuration | Result |
|---|---|---|
| `eval/sweep.md` + `eval/baseline.json` | `EMBEDDING_QUERY_PROMPT=query`, 60 questions | `top_k=8` + `min_score=0.3`: recall@8 `0.783`, zero-context `0%` |
| `eval/sweep-prompt-off.md` + `eval/baseline-prompt-off.json` | `EMBEDDING_QUERY_PROMPT=` (no instruction), 60 questions | best recall@8 `0.683`, MRR@10 `0.455` — lower coverage, higher top-3 precision |
| `eval/gateway-singleturn-off.json` + `eval/gateway-singleturn-concat.json` | 60 single-turn questions, gateway mode, rewrite `off` vs `concat` | both `recall@10 0.800`, `MRR@10 0.414` — the rewrite does not touch single-turn requests |
| `eval/multiturn-off.json` | 9 follow-up conversations, gateway, rewrite `off` | recall@5 `0.667`, recall@10 `0.778`, MRR@10 `0.277`, nDCG@10 `0.392` — 2 of 9 follow-ups have no expected hit in the top ten |
| `eval/multiturn-gateway.json` + `eval/sweep-multiturn.md` | the same 9 conversations, rewrite `concat` (the shipped default) | recall@5 `0.778`, recall@10 `1.000`, MRR@10 `0.420`, nDCG@10 `0.770`; `min_score=0.3`: recall@8 `0.889`, zero-context `0%`; `0.5`: recall@8 `0.667`, zero-context `11%`; `0.6`: `67%` |
| `eval/multiturn-llm.json` + `eval/sweep-multiturn-llm.md` | the same 9 conversations, rewrite `llm` | recall@5 `0.889`, recall@10 `1.000`, MRR@10 `0.514`, nDCG@10 `0.781`; `min_score=0.3`: recall@8 `1.000`, zero-context `0%`; `0.5`: recall@8 `0.667`, zero-context `22%` — the best ranking, at one extra upstream call per follow-up |
| `eval/p21-sweep.md` + the `eval/p21-*.json` files | dense-only vs dense + BM25 (RRF), `SPARSE_WEIGHT` × `SPARSE_MIN_RELATIVE`, 60 single-turn (`/search`, top_k=10) and 9 follow-up (gateway, top_k=20) questions | selected `SPARSE_WEIGHT=1.0` / `SPARSE_MIN_RELATIVE=0.7`: single-turn recall@10 `0.883` (dense-only `0.800`), recall@5 `0.750`; gateway single-turn recall@10 `0.883`, MRR@10 `0.490`; follow-ups recall@10 `0.889` / MRR@10 `0.453` against dense-only `1.000` / `0.420` |
| `eval/sweep-rerank.md` + `eval/rerank-sweep.json` + `eval/rerank-off.json` | cross-encoder `BAAI/bge-reranker-v2-m3` (cuda, fp16, `RERANK_TRANSFORM=raw`), fused pool 30 vs 50, and the same service with rerank forced off, 60 single-turn questions through `/search` (top_k 50 candidates) | **rerank on, pool 30**: recall@5 `0.733`, recall@10 `0.883`, MRR@10 `0.620`, nDCG@10 `0.765`, `rerank_rate 1.0`, p50 `0.590s` / p95 `0.671s`. Baseline `--rerank off`: recall@5 `0.700`, recall@10 `0.867`, MRR@10 `0.486`, nDCG@10 `0.642`, p50 `0.087s` / p95 `0.135s`. Pool 50 scores the same grid (recall@10 `0.883`, MRR@10 `0.620`) at p50 `0.611s` / p95 `0.710s` — scoring 30 is enough, so `RERANK_TOP_N=30`. The reranker costs `+0.50s` p50 (x6.8) and buys `+0.134` MRR@10 and `+0.016` recall@10; gates `--min-recall 0.70 --min-mrr 0.35` pass on every pass. |
| `eval/rerank-multiturn.json` | 9 follow-up conversations, gateway mode with `scripts/upstream_stub.py` as the upstream, pools 30 vs 50, rewrite `concat`, `BAAI/bge-reranker-v2-m3` | **rerank on, pool 30**: recall@5 `0.667`, recall@10 `0.889`, MRR@10 `0.511`, nDCG@10 `0.754`, `rerank_rate 1.0`, p50 `0.451s` / p95 `0.528s` (stub upstream — retrieval and rewrite only, no answer generation). Pool 50 adds nothing at no-threshold (`MRR@10 0.511`) and `0.582` at `min_score 0.4` vs `0.526` for pool 30. **This file's `rerank off` rows are not a fused baseline**: the gateway sent `rerank: true` only, so `api` applied its `RERANK_ENABLED=true` default and `rerank_rate` was `1.0` on every row — the numbers below are the fused baseline re-measured after the tri-state fix. |
| `eval/rerank-off-multiturn.json` | the same 9 conversations, gateway mode, `X-RAG-RERANK: off` forced through the fixed gateway, pools labelled 30 | fused baseline: recall@5 `0.778`, recall@10 `0.889`, MRR@10 `0.4529`, nDCG@10 `0.7103`, `rerank_rate 0.0` (the model never ran), avg candidates 20, p50 `0.098s` / p95 `0.131s`. Gates `--min-recall 0.70 --min-mrr 0.35` pass. **`mt-003` fused `first_rank 13`** — already outside the top 10 before reranking, and out of the returned list after it. So on this corpus `BAAI/bge-reranker-v2-m3` buys MRR `0.4529 → 0.511` and nDCG `0.7103 → 0.754`, loses recall@5 `0.778 → 0.667`, and does not meet the P1.1 multiturn criterion. |
| `eval/rerank-off-multiturn-topk8.json` + `eval/rerank-multiturn-topk8.json` | the same 9 conversations at the shipped `RAG_TOP_K=8`, gateway, fused vs `rerank on` pool 30 | fused: recall@5 `0.778`, recall@10 `0.889`, MRR@10 `0.456`, nDCG@10 `0.655`, `mt-003` absent from the returned 8; rerank on: `0.667`, `0.889`, `0.511`, `0.694`, `mt-003` absent, p50 `0.45s`. **`top_k` is not the cause** — `mt-003` is outside the top ten before reranking at every `top_k`, which is the P2.1 hybrid regression recorded in the row above. The reranker has to pull it back, and `bge-reranker-v2-m3` does not: the expected chunks score below `0.118` and leave the pool |
| `eval/rerank-qwen3-sweep.json` + `eval/sweep-qwen3.md` + `eval/rerank-qwen3-multiturn.json` | `Qwen/Qwen3-Reranker-0.6B` (cuda, fp16), the same 60 single-turn (`/search`, top_k 50) and 9 multiturn (gateway, top_k 20, stub upstream, rewrite `concat`) questions, pools 30 vs 50, **plain pairs** (`RERANK_QUERY_INSTRUCT` off) | single-turn pool 30: recall@5 `0.850`, recall@10 `0.917`, MRR@10 `0.649`, nDCG@10 `0.804`, `rerank_rate 1.0`, p50 `1.49s` / p95 `1.77s`. Gateway multiturn pool 30: recall@5 `0.778`, recall@10 `0.889`, MRR@10 `0.703`, nDCG@10 `0.875`, p50 `0.94s` / p95 `1.18s`. **Against `bge-reranker-v2-m3` this model is better at every rank on the single-turn set** (recall@10 `0.917` vs `0.883`, MRR@10 `0.649` vs `0.620`) and on the follow-ups (MRR@10 `0.703` vs `0.511`, fused baseline `0.4529`), at 2.5x the latency. **`mt-003` is still outside the top ten**, so the criterion is not met by either model with plain pairs. The pass that decides the default is the one that puts `mt-003` back inside the top ten with no `recall@10` regression, at a latency the ingest probe can pay. Plain pairs scored this model `probe score=-2.0625` on an obvious match, and its top `rerank_score` values on `mt-003`'s pool (`7.9453`, `4.3594`, `3.7812`) are term-noise chunks, so the instruction is expected to move the numbers. **Measured plain pairs, pool 30:** single-turn recall@5 `0.850`, recall@10 `0.917`, MRR@10 `0.649`, nDCG@10 `0.804` (the best grid on the corpus); follow-ups MRR@10 `0.703`, nDCG@10 `0.875` |
| `eval/rerank-qwen3-instruct-sweep.json` + `eval/sweep-qwen3-instruct.md` + `eval/rerank-qwen3-instruct-multiturn.json` | the same two sets with `RERANK_QUERY_INSTRUCT=qwen3` (`Given a query, retrieve passages that answer the query.`), measured on the rebuilt image | **first pass was invalid**: `docker compose up -d --force-recreate api` runs the image built earlier, so the container had no `RERANK_QUERY_INSTRUCT` code, `/health` reported `rerank_query_instruct` as absent, and the two passes printed the same grid to the digit as the plain-pair pass. Signature of an invalid pass: grid identical to the digit to the pass it is supposed to differ from. Measured on the rebuilt image, single-turn pool 30: recall@5 `0.833`, recall@10 `0.900`, MRR@10 `0.621`, nDCG@10 `0.762`, p50 `1.39s` / p95 `1.60s`; pool 50 identical. **The instruction hurts the single-turn set** (`recall@10 0.900` vs `0.917`, `MRR@10 0.621` vs `0.649`, `nDCG@10 0.762` vs `0.804` for plain pairs). Gateway multiturn pool 30: recall@5 `0.778` (plain `0.733`, `bge` `0.667`, fused baseline `0.778`), recall@10 `0.889`, MRR@10 `0.700` (plain `0.703`), nDCG@10 `0.832` (plain `0.875`), p50 `0.98s`. Pool 50: recall@10 `0.889`, MRR@10 `0.644` — worse than pool 30. **`mt-003` is outside the top ten in both instructed and plain passes**, so the prompt is not the lever that fixes it; the lever is the fusion of P2.1 (see `mt-003` below) |
| `eval/rerank-qwen3-question-multiturn.json` + `eval/sweep-qwen3-question-multiturn.md` | 9 follow-up conversations, gateway, pools 30 vs 50, **pair query = the user's question** (`X-RAG-RERANK-QUERY: question`), plain pairs, `Qwen/Qwen3-Reranker-0.6B` | pool 30: recall@5 `0.556`, recall@10 `0.778`, MRR@10 `0.509`, nDCG@10 `0.513`, `rerank_rate 1.0`, p50 `1.03s` / p95 `1.11s`; pool 50 `MRR@10 0.507` (no gain). Per-question first rank, `question` vs `retrieval`: **`mt-003` `1` vs `99`** (the lever works on the case it was built for — scoring the injected history ranks its term overlap above the answer), `mt-001` `12` vs `1`, `mt-006` `16` vs `1`, `mt-005` `6` vs `8`. **Net loss**: `recall@10 0.778` vs `0.889`, `MRR@10 0.509` vs `0.703`, `nDCG@10 0.513` vs `0.875`. A short question drops the topical anchor an elliptical follow-up needs, so the same history terms that push `mt-003` down carry `mt-001` and `mt-006` up. **`RAG_RERANK_QUERY=retrieval` stays the default**; `question` is a per-request knob. The P1.1 `mt-003` criterion is met by the mechanism, not by a shippable configuration: no measured pass (bge, Qwen3 plain, Qwen3 instructed, pair=question) leaves `mt-003` inside the top ten **and** the aggregate unregressed |
| `mt-003` diagnosis: dense vs hybrid on the gateway's query | `scripts/diagnose_rerank.py --question-id mt-003 --mode search --retrieval-mode dense --retrieval-query concat --rerank off --top-k 20` | **the expected chunk is at dense rank 9** on the `concat` request (`Which embedding model does the stack use?\nСколько измерений у её векторов?`), at hybrid rank 13, and outside the reranked top 20. So fusion, not the reranker, is what removed it, and the reranker's scores on this pool (`7.9453`, `4.3594`, `3.7812` on `README.md` chunks 47/48/59) put term noise above it |

`RAG_TOP_K=8` / `RAG_MIN_SCORE=0.3` is the choice all the tables support, and the
follow-up sweep (`eval/sweep-multiturn.md`) is the binding one for the threshold:
an elliptical question ("and the min score?") embeds weakly, so a threshold tuned
on single-turn questions alone silently strips the context of real conversations.
With the rewrite `off`, `min_score=0.5` leaves `78%` of the follow-ups with no
context at all and `0.6` leaves `100%`; with `concat` the same thresholds leave
`11%` and `67%`, and at `0.3` zero-context is `0%` in every mode. So `min_score`
stays at `0.3`. `RAG_QUERY_REWRITE=concat` is the default for the mirror-image
reason: `off` misses the top ten on 2 of 9 follow-ups, `concat` closes both at no
extra upstream cost, and `llm` ranks higher still (`recall@8` `1.000` at
`min_score=0.3`) for one extra upstream call, so it stays opt-in.

Reproduce the record from a fresh clone:

```powershell
Copy-Item eval/corpus/* data/documents/
docker compose up -d                    # re-embeds the corpus, 202 chunks
python scripts/eval_retrieval.py --questions eval/questions.jsonl --sweep \
    --sweep-md eval/sweep.md --json-out eval/baseline.json
# then disable the query prompt and run again (no re-index: the prompt is
# query-side only). Set EMBEDDING_QUERY_PROMPT= in .env, then:
docker compose up -d api
python scripts/eval_retrieval.py --questions eval/questions.jsonl --sweep \
    --sweep-md eval/sweep-prompt-off.md --json-out eval/baseline-prompt-off.json
```

The rewrite comparison recreates the gateway per mode. `POST /health` reports the
mode the container picked up (`"query_rewrite"` in `defaults`); check it before
measuring, because a container that was only restarted keeps the environment it was
created with:

```powershell
$env:RAG_QUERY_REWRITE = "off"
docker compose up -d --force-recreate gateway
python scripts/eval_retrieval.py --questions eval/questions.jsonl --mode gateway \
    --corpus eval/corpus --json-out eval/gateway-singleturn-off.json
python scripts/eval_retrieval.py --questions eval/questions.multiturn.jsonl --mode gateway \
    --corpus eval/corpus --json-out eval/multiturn-off.json
$env:RAG_QUERY_REWRITE = "concat"
docker compose up -d --force-recreate gateway
python scripts/eval_retrieval.py --questions eval/questions.jsonl --mode gateway \
    --corpus eval/corpus --json-out eval/gateway-singleturn-concat.json
python scripts/eval_retrieval.py --questions eval/questions.multiturn.jsonl --mode gateway \
    --corpus eval/corpus --sweep --sweep-md eval/sweep-multiturn.md \
    --json-out eval/multiturn-gateway.json
$env:RAG_QUERY_REWRITE = "llm"
docker compose up -d --force-recreate gateway
python scripts/eval_retrieval.py --questions eval/questions.multiturn.jsonl --mode gateway \
    --corpus eval/corpus --sweep --sweep-md eval/sweep-multiturn-llm.md \
    --json-out eval/multiturn-llm.json
Remove-Item Env:RAG_QUERY_REWRITE
docker compose up -d --force-recreate gateway   # back to the shipped default (concat)
```

The hybrid grid is measured per request, so one container serves the whole sweep:

```powershell
python scripts/eval_retrieval.py --questions eval/questions.jsonl --mode search \
    --retrieval-mode dense --top-k 10 --json-out eval/p21-search-dense.json
foreach ($cfg in @("1.0:0.0","1.0:0.3","1.0:0.5","1.0:0.7","1.5:0.3","1.5:0.5")) {
    $w, $r = $cfg -split ":"
    python scripts/eval_retrieval.py --questions eval/questions.jsonl --mode search \
        --retrieval-mode hybrid --sparse-weight $w --sparse-min-relative $r \
        --top-k 10 --json-out "eval/p21-search-hybrid-w$w-r$r.json"
}
# the gateway runs use the service defaults, so recreate `api` with the chosen values
# committed file names: weight 1.0 / floor 0.0 is `p21-search-hybrid-plain.json`,
# the selected weight 1.0 / floor 0.7 run is `p21-search-selected.json`
$env:SPARSE_WEIGHT = "1.0"; $env:SPARSE_MIN_RELATIVE = "0.7"
docker compose up -d --force-recreate api
python scripts/eval_retrieval.py --questions eval/questions.jsonl --mode gateway \
    --top-k 20 --corpus data/documents --json-out eval/p21-gateway-singleturn.json
python scripts/eval_retrieval.py --questions eval/questions.multiturn.jsonl --mode gateway \
    --top-k 20 --corpus data/documents --json-out eval/p21-gateway-multiturn.json
Remove-Item Env:SPARSE_WEIGHT,Env:SPARSE_MIN_RELATIVE
```

Each gateway run prints the mode it measured against
(`gateway rewrite: concat (history=3)`, read from `/health`), so a run that got a
container created with a different mode is visible in its own output, and every
committed result file stores the mode in `gateway_defaults`. Warm up the connection
before a measured run (`--limit 1`): the first request after a container is created
pays the Windows connection setup (~20 s), which would otherwise land in the
latency numbers. Run the modes one after another in one session — a container
recreated by a parallel session, on a shared machine, is enough to make a measured
run report the wrong mode.

## Metrics

- **recall@k** — fraction of questions with at least one relevant hit in the top `k`.
- **MRR@10** — mean reciprocal rank of the first relevant hit within the top 10.
- **nDCG@10** — binary-relevance nDCG at 10; the ideal ranking places all expected hits first.

In `search` mode `min_score` is `null`, so metrics measure the retriever itself.
In `gateway` mode the candidates requested are `X-RAG-TOP-K=20` with
`X-RAG-MIN-SCORE` disabled, so the raw candidate list is recorded and thresholds
are applied by the sweep.

## Current baseline

The dataset is 60 questions (`eval/questions.jsonl`) over the committed corpus
(202 chunks: `README.md` 68, `TODO.md` 80, `env_example.txt` 40,
`eval_README.md` 13, `example.txt` 1). Ground truth is `file` + `text_contains`,
validated offline against `eval/corpus` and checked against the indexed chunks,
so every needle exists in the collection. The corpus is a **frozen snapshot** of
the project docs: editing `README.md` in the repository does not move the baseline,
and the reproduce step copies the snapshot into `data/documents`, so a needle that
stops existing in the snapshot has to be re-pointed in `eval/questions.jsonl`
(`q-056` cites the rewrite bullets in `env_example.txt`, where the snapshot marks
`off` as the default — the shipped default is now `concat`, and the snapshot is
frozen, so the needle stays as it is). Measured with `top_k=50`, `min_score: null`:

| Metric | Value | Recorded gate |
|---|---|---|
| recall@5 | 0.683 | — |
| recall@10 | 0.800 | `--min-recall 0.70` |
| MRR@10 | 0.414 | `--min-mrr 0.35` |
| nDCG@10 | 0.557 | — |

The gates sit just under the baseline, so they catch a regression without
encoding the current quality as a target. `q-058` uses two `relevant` entries,
which is what the acceptance criteria call a multi-chunk answer. 12 questions
have no expected hit in the top 10, and they concentrate on short table rows
(`Current state`, the configuration table, memory limits) — dense embeddings do
best on narrative and heading chunks, so this is the target of hybrid search
(P2.1) and threshold tuning (P0.3). The corpus is self-referential (project docs
and the env template); real user documents and a held-out subset are still to be
added.

## Building a good question set

- 50–200 real questions from actual users, not synthetic paraphrases.
- Cover exact identifiers/product codes, paraphrases, Russian and English, and
  answers spread across multiple chunks.
- Keep a held-out subset: tune on one part, report on the other.
- A change is worth keeping only if it improves the aggregate **and** does not
  regress the categories that matter to you.
- Re-baseline after an intentional change (new model, new chunking) and record
  the reason in the commit message.

## Notes

- `eval/baseline.json` (search mode: metrics, per-question ranks, sweep grid),
  `eval/gateway-singleturn-off.json` / `eval/gateway-singleturn-concat.json` (60
  single-turn questions through the gateway) and
  `eval/multiturn-off.json` / `eval/multiturn-gateway.json` /
  `eval/multiturn-llm.json` (9 follow-up conversations, one rewrite mode each) are
  generated by `--json-out`, not hand-written, together with
  `eval/sweep-multiturn.md` (the `concat` grid) and `eval/sweep-multiturn-llm.md`
  (the `llm` grid). A gateway run records the gateway's
  own defaults in `gateway_defaults` (read from `/health`), so every committed
  file says which rewrite mode produced its numbers. Commit them after a deliberate
  quality change so regressions are visible later.
- The harness reuses **one keep-alive connection** for the whole run. On Windows a
  fresh TCP connection to a Docker-published port costs ~20 s, so opening one per
  question turns a 60-question run into ~20 minutes of connection setup; reusing
  the connection makes the same run take ~30 s.
- The API must be running with the same embedding model, corpus and chunking as
  the deployment you are evaluating.
- The question set may contain domain data; review it before publishing the
  repository.
- Gateway mode calls the upstream LLM once per question (answers are not
  evaluated, only the retrieved sources), so it is slower and needs a working
  `UPSTREAM_BASE_URL`. `RAG_QUERY_REWRITE=llm` adds one more call per follow-up,
  bounded by `RAG_REWRITE_TIMEOUT`.
- When the LLM server is not reachable, `scripts/upstream_stub.py` answers in place
  of it (`UPSTREAM_BASE_URL=http://host.docker.internal:5011/v1` from the gateway
  container). The stub returns instantly, so gateway latency measured against it is
  the cost of retrieval and rewriting, not of generating an answer — record which
  upstream produced the numbers, and re-measure against the real server before
  quoting latency as the user-visible figure.

- A reranked gateway run reads `rerank_score` from `X-RAG-Sources`, so `rerank_rate`
  in `--json-out` is the share of questions the model actually scored. A run against
  a service with no reranker loaded (`RERANK_ENABLED=false`) reports `rerank_rate 0`:
  the request is answered with the fused order, which is the degradation P1.1 specifies,
  not an error. Record a rerank run with `RERANK_MODEL`, `RERANK_TRANSFORM`,
  `RERANK_DEVICE` and `RERANK_DTYPE` from `/health` in the commit message: the numbers
  are a property of the model, not of the code.
- Absolute latency depends on the upstream server, so compare modes in one session
  on a warmed connection: the first request after a container is created pays the
  connection setup (~20 s on Windows), and a loaded upstream inflates every run.
  Warm up with `--limit 1` before the measured run.
- The gateway picks up `RAG_QUERY_REWRITE` when the container is **created**, so
  an env change needs `docker compose up -d --force-recreate gateway`; check the
  mode in `/health` before measuring, and never run two eval sessions against the
  same container at the same time.

## Tests

The metric, sweep, matching, dataset and corpus-check logic is covered by unit
tests that need no GPU, API, or Docker, and the follow-up rewrite (the single-turn
guarantee, `concat` history joining, `llm` fallback and cache) by gateway tests
that import `gateway/main.py` without running it:

```bash
python -m unittest discover -s tests -p "test_*.py"
```
