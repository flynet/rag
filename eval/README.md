# Retrieval evaluation

`scripts/eval_retrieval.py` measures retrieval quality against a labelled
question set. It calls `POST /search` on the **embedding API** (port 8000, not
the gateway), with `min_score: null`, so the score threshold does not hide
results. Run it after every change to the model, prompt, chunking, or retriever.

## Dataset format

One JSON object per line (JSONL). Example:

```json
{"id": "q-001", "question": "Who is Alice Davince?", "relevant": [{"file": "example.txt", "text_contains": "Alice Davince"}]}
```

Fields:

| Field | Required | Description |
|---|---|---|
| `id` | no | stable identifier; defaults to `q001`, `q002`, ... |
| `question` | yes | the query sent to `/search` |
| `relevant` | yes | non-empty array of expected hits |

Each expected hit may use any combination of these fields; all provided fields
must match:

| Field | Description |
|---|---|
| `file` | file path as stored in the payload, or just its basename (`example.txt`) |
| `chunk` | zero-based chunk index |
| `text_contains` | case-insensitive substring of the chunk text |

`text_contains` is usually the most stable criterion: chunk indices change when
the chunker changes, while a distinctive phrase does not. Use `file` alone when
any chunk of the document is acceptable.

`eval/questions.example.jsonl` contains schema examples for the sample
document. Copy it to `eval/questions.jsonl` and replace the examples with real
questions:

```bash
cp eval/questions.example.jsonl eval/questions.jsonl
```

## Running

```bash
# validate the dataset without calling the API
python scripts/eval_retrieval.py --questions eval/questions.jsonl --validate-only

# full run against the API (default: http://localhost:8000)
python scripts/eval_retrieval.py --questions eval/questions.jsonl

# save the baseline (aggregate metrics + per-question results)
python scripts/eval_retrieval.py --questions eval/questions.jsonl \
    --json-out eval/baseline.json

# fail the run if quality drops below a gate
python scripts/eval_retrieval.py --questions eval/questions.jsonl \
    --min-recall 0.85 --min-mrr 0.70
```

Useful options:

| Option | Default | Description |
|---|---|---|
| `--api-url` | `http://localhost:8000` | embedding API base URL |
| `--top-k` | `50` | candidates requested per question |
| `--recall-k` | `5,10` | comma-separated cutoffs for recall |
| `--limit` | — | evaluate only the first N questions |
| `--json-out` | — | write aggregates and per-question results |
| `--show-failures` | `20` | number of failing questions to print |
| `--min-recall` | — | exit 1 if `recall@max(recall-k)` is below the value |
| `--min-mrr` | — | exit 1 if `MRR@10` is below the value |

## Metrics

- **recall@k** — fraction of questions with at least one relevant hit in the top `k`.
- **MRR@10** — mean reciprocal rank of the first relevant hit within the top 10.
- **nDCG@10** — binary-relevance nDCG at 10; the ideal ranking places all expected hits first.

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

- `eval/baseline.json` is generated, not hand-written. Commit it after a
  deliberate quality change so regressions are visible later.
- The API must be running with the same embedding model and collection as the
  deployment you are evaluating.
- The question set may contain domain data; review it before publishing the
  repository.
