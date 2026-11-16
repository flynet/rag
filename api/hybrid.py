"""Hybrid retrieval: dense + BM25 sparse, fused with RRF.

`POST /search` runs two Qdrant queries (the dense vector and the `bm25` sparse
vector) and merges the two lists here. Merging in the API is what keeps the
`min_score` threshold meaningful: RRF scores are rank-based, so a threshold of
0.3 on a fused score means something completely different from 0.3 cosine. Every
fused hit carries the dense cosine score (`dense_score`), and a BM25-only hit
(an exact term the embedding missed) carries `None` - which passes the threshold.
That is the point of the sparse branch.

Stdlib only, so fusion and the threshold rules are testable without Qdrant,
GPU or Docker.
"""

SEARCH_MODES = ("hybrid", "dense")

# Named vectors in the collection: `dense` is the embedding model, `bm25` is the
# sparse BM25 representation (the same names the indexer uses in
# `embedder/vector_db.py`).
DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"

# Reciprocal rank fusion constant. RRF only cares about rank, so `k` shapes the
# decay; 60 is the value from the RRF paper.
RRF_K = 60


def normalize_hits(hits):
    """Turn qdrant point matches (objects) or dicts into {"id","score","payload"}."""
    out = []
    for hit in hits:
        if isinstance(hit, dict):
            ident = hit.get("id", hit.get("point_id"))
            score = hit.get("score")
            payload = hit.get("payload")
        else:
            ident = getattr(hit, "id", None)
            score = getattr(hit, "score", None)
            payload = getattr(hit, "payload", None)
        if ident is None or not isinstance(score, (int, float)):
            continue
        out.append({"id": ident, "score": float(score), "payload": payload or {}})
    return out


def fuse(dense, sparse, top_k, k=RRF_K, sparse_weight=1.0):
    """Reciprocal rank fusion of two ranked lists.

    A hit's fused score is `sum(weight / (k + rank))` over the lists it appears
    in, where rank is 1-based and the dense weight is fixed at 1.0. With
    `sparse_weight` below 1.0, a BM25-only hit cannot outrank a hit that the
    dense branch placed near the top, and exact-term hits keep their edge: the
    sparse branch is a modifier of the ranking, not a competing list of equal
    size. Ties are broken by the dense score, then by id, so the order is
    deterministic. Returns at most `top_k` hits; each carries `fusion_score`,
    `dense_score` (`None` for BM25-only hits), `sparse_score` and `ranks`.
    """
    merged = {}
    for source, weight, hits in (("dense", 1.0, dense), ("sparse", sparse_weight, sparse)):
        for rank, hit in enumerate(hits, start=1):
            entry = merged.setdefault(
                hit["id"],
                {
                    "id": hit["id"],
                    "payload": hit["payload"],
                    "fusion_score": 0.0,
                    "dense_score": None,
                    "sparse_score": None,
                    "ranks": {},
                },
            )
            entry["fusion_score"] += weight / (k + rank)
            entry["ranks"][source] = rank
            if source == "dense":
                entry["dense_score"] = hit["score"]
            else:
                entry["sparse_score"] = hit["score"]

    def key(entry):
        dense = entry["dense_score"] if entry["dense_score"] is not None else 0.0
        return (-entry["fusion_score"], -dense, str(entry["id"]))

    return sorted(merged.values(), key=key)[: max(0, int(top_k))]


def filter_relative(hits, ratio):
    """Keep hits within `ratio` of the best score in the list.

    BM25 scores are not comparable to cosine scores, and a weak term overlap
    (say 2.0 against a best match of 10.0) adds ranking noise: the chunk shares
    one common word with the query. Dropping the weak tail keeps the exact-identifier
    hits, which are the point of the sparse branch.
    """
    if not hits or ratio is None or ratio <= 0.0:
        return list(hits)
    best = max(hit["score"] for hit in hits)
    floor = ratio * best
    return [hit for hit in hits if hit["score"] >= floor]


def apply_threshold(fused, min_score):
    """Filter on the dense cosine score, keeping BM25-only hits.

    A fused RRF score has no cosine meaning, so `min_score` is applied to the
    dense score. Hits with no dense score at all (found only by the sparse
    branch) are kept: dropping them would make the sparse branch useless, and
    those are exactly the exact-identifier hits the dense branch misses.
    """
    if min_score is None:
        return list(fused)
    return [
        hit
        for hit in fused
        if hit["dense_score"] is None or hit["dense_score"] >= min_score
    ]
