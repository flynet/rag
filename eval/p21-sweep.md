# P2.1 hybrid search: measured grid

Measured on 2026-11-16 on the live stack (`api`, `gateway`, `qdrant` on this
machine, collection `documents`: 202 points, dim 1024, named `dense` + sparse
`bm25`, 5 corpus files under `data/documents`). Dense branch:
`Qwen/Qwen3-Embedding-0.6B` (fp16, CUDA, query prompt "query"). Sparse branch:
FastEmbed `Qdrant/bm25`, collection sparse index `modifier=IDF`. Fusion is RRF
(`k=60`) computed in `api`, dense weight fixed at 1.0, BM25 weight
`SPARSE_WEIGHT`, BM25 hits below `SPARSE_MIN_RELATIVE` × the best BM25 score of
the query are dropped. `min_score` is applied to the dense cosine score, so
BM25-only hits always pass.

`weight`/`floor` columns are `SPARSE_WEIGHT` / `SPARSE_MIN_RELATIVE`.
Every number comes from the JSON file named in the last column.

## Single-turn, 60 questions, `POST /search`, top_k=10, min_score=null

| mode | weight | floor | recall@5 | recall@10 | MRR@10 | nDCG@10 | p95 latency | file |
|---|---|---|---|---|---|---|---|---|
| dense | – | – | 0.683 | 0.800 | 0.414 | 0.557 | 0.064 | `p21-search-dense.json` |
| hybrid | 1.0 | 0.0 | 0.717 | 0.850 | 0.463 | 0.643 | 0.086 | `p21-search-hybrid-plain.json` |
| hybrid | 0.5 | 0.0 | 0.733 | 0.817 | 0.438 | 0.602 | 0.083 | `p21-search-w0.5.json` |
| hybrid | 1.5 | 0.0 | 0.717 | 0.817 | 0.487 | 0.657 | 0.091 | `p21-search-w1.5.json` |
| hybrid | 1.0 | 0.3 | 0.717 | 0.867 | 0.464 | 0.649 | 0.091 | `p21-search-w1.0-r0.3.json` |
| hybrid | 1.0 | 0.5 | 0.750 | 0.867 | 0.469 | 0.651 | 0.086 | `p21-search-w1.0-r0.5.json` |
| **hybrid** | **1.0** | **0.7** | **0.750** | **0.883** | **0.478** | **0.644** | **0.100** | `p21-search-selected.json` |
| hybrid | 1.5 | 0.3 | 0.717 | 0.817 | 0.487 | 0.657 | 0.090 | `p21-search-w1.5-r0.3.json` |
| hybrid | 1.5 | 0.5 | 0.733 | 0.800 | 0.486 | 0.648 | 0.084 | `p21-search-w1.5-r0.5.json` |

Selected `weight=1.0, floor=0.7`: the best recall@10 (0.883, +8.3 points over
dense-only) and the best recall@5 among the hybrid runs. `weight=1.5` gives the
best MRR/nDCG (0.487/0.657) but loses recall, so it is not the default.

Per-question effect of the selected configuration against dense-only (first
relevant chunk, top 10): 21 improved, 12 regressed, 27 unchanged. 8 questions
that dense-only missed entirely are now inside the top 10 (`q-003`, `q-009`,
`q-028`, `q-029`, `q-032`, `q-039`, `q-054`, `q-055`); 4 dense-only hits dropped
out of the top 10 (`q-004` from 5, `q-008` from 9, `q-012` from 5, `q-037` from 1).
Subset split by ground-truth shape:

| subset | n | dense recall@10 / MRR@10 | hybrid recall@10 / MRR@10 |
|---|---|---|---|
| exact identifiers / code-like strings | 37 | 0.811 / 0.451 | 0.838 / 0.461 |
| paraphrased questions | 23 | 0.783 / 0.355 | 0.913 / 0.480 |

The exact-identifier subset gains less than the paraphrase subset: the corpus is
5 documentation files, so most "identifiers" (`RAG_TOP_K`, `QDRANT_VERSION`,
`mem_limit`) are dense-retrievable already, and the BM25 branch mostly re-orders
them. A corpus with product codes and ticket IDs is where the sparse branch is
expected to dominate (P4.1 workspaces and P1.2 structure-aware chunking both
feed into that).

## Multiturn, 9 follow-ups, gateway, top_k=20

The gateway default `RAG_QUERY_REWRITE=concat` (P1.3) is on in every row.

| rewrite | mode | weight | floor | recall@5 | recall@10 | MRR@10 | nDCG@10 | file |
|---|---|---|---|---|---|---|---|---|
| concat | dense | – | – | 0.778 | 1.000 | 0.420 | 0.770 | `multiturn-gateway.json` |
| concat | hybrid | 1.0 | 0.5 | 0.556 | 0.889 | 0.514 | 0.698 | `p21-gateway-multiturn-r0.5.json` |
| **concat** | **hybrid** | **1.0** | **0.7** | **0.778** | **0.889** | **0.453** | **0.710** | `p21-gateway-multiturn.json` |
| concat | hybrid | 1.0 | 0.85 | 0.556 | 0.889 | 0.350 | 0.669 | `p21-gateway-multiturn-r0.85.json` |
| concat | hybrid | 1.5 | 0.7 | 0.778 | 0.889 | 0.453 | 0.710 | `p21-gateway-multiturn-w1.5-r0.7.json` |
| llm | dense | – | – | 0.889 | 1.000 | 0.514 | 0.781 | `multiturn-llm.json` |

Per-question first relevant rank, concat-dense vs concat-hybrid (selected):
`mt-002` 1→1, `mt-006` 1→1, `mt-009` 4→1, `mt-004` 6→3, `mt-001` 3→5,
`mt-005` 3→7, `mt-007` 4→5, `mt-008` 3→5, `mt-003` 9→13.

`mt-003` is the recall@10 loss: a long rewritten follow-up matches many chunks on
common terms, so BM25 pushes the dense hit from rank 9 to rank 13. MRR improves on
the questions the dense branch already found (`mt-009` 4→1, `mt-004` 6→3), which is
why MRR@10 is above the dense baseline while recall@10 is below it. A reranker
(P1.1) is the intended fix for that ordering; `llm` rewrite is the opt-in mode
that keeps recall@10 at 1.000 with dense retrieval.

Gateway single-turn, 60 questions, top_k=20, rewrite `concat`:

| mode | recall@5 | recall@10 | MRR@10 | nDCG@10 | p95 latency | file |
|---|---|---|---|---|---|---|
| dense | 0.683 | 0.800 | 0.414 | 0.557 | – | `gateway-singleturn-concat.json` |
| hybrid (selected) | 0.700 | 0.883 | 0.490 | 0.654 | 4.631 | `p21-gateway-singleturn.json` |
| hybrid, floor 0.5 | 0.733 | 0.833 | 0.469 | 0.641 | 4.833 | `p21-gateway-singleturn-r0.5.json` |

Gateway latency is dominated by the upstream LLM call; the BM25 encode and the
second Qdrant query add ~20-35 ms p95 at `POST /search`.

## Reproduce

```
python scripts/eval_retrieval.py --questions eval/questions.jsonl --mode search \
    --retrieval-mode dense --top-k 10 --json-out eval/p21-search-dense.json
python scripts/eval_retrieval.py --questions eval/questions.jsonl --mode search \
    --retrieval-mode hybrid --sparse-weight 1.0 --sparse-min-relative 0.7 \
    --top-k 10 --json-out eval/p21-search-selected.json
python scripts/eval_retrieval.py --questions eval/questions.jsonl --mode gateway \
    --top-k 20 --corpus data/documents --json-out eval/p21-gateway-singleturn.json
python scripts/eval_retrieval.py --questions eval/questions.multiturn.jsonl --mode gateway \
    --top-k 20 --corpus data/documents --json-out eval/p21-gateway-multiturn.json
```

Gateway runs use the service defaults (`SEARCH_MODE`, `SPARSE_WEIGHT`,
`SPARSE_MIN_RELATIVE`), so changing them takes
`docker compose up -d --force-recreate api`. `--retrieval-mode`,
`--sparse-weight` and `--sparse-min-relative` are per-request overrides that only
`POST /search` accepts.
