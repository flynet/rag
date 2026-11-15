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

**Gateway mode cannot use `text_contains`**: `X-RAG-Sources` carries file, chunk,
score, title, section and character offsets, not the chunk text. Ground truth in
gateway mode must use `file`, `chunk` and `section` (the loader rejects
`text_contains` in that mode).

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
so every needle exists in the collection. Measured with `top_k=50`,
`min_score: null`:

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

- `eval/baseline.json` and `eval/sweep.md` are generated, not hand-written.
  Commit them after a deliberate quality change so regressions are visible later.
- The API must be running with the same embedding model, corpus and chunking as
  the deployment you are evaluating.
- The question set may contain domain data; review it before publishing the
  repository.
- Gateway mode calls the upstream LLM once per question (answers are not
  evaluated, only the retrieved sources), so it is slower and needs a working
  `UPSTREAM_BASE_URL`.

## Tests

The metric, sweep, matching, dataset and corpus-check logic is covered by unit
tests that need no GPU, API, or Docker:

```bash
python -m unittest discover -s tests -p "test_*.py"
```
