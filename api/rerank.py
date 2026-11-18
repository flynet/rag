"""Cross-encoder reranking: candidate pooling, passage text, and the ordering rule.

`POST /search` with `rerank=true` pulls a wider candidate list from both Qdrant
branches, scores every candidate with a cross-encoder, and keeps `top_k`. A
cross-encoder re-scores content, not rank position, which is the point: after
P2.1 an intruder from the BM25 branch sits at a fixed high RRF rank (16th with
`top_k=8`) no matter what it says, so no weight or floor removes it — the
measured case is `mt-003`, dense rank 9 becoming hybrid rank 13. Judging the text
moves it below the cut.

Stdlib and numpy only, so the pooling, passage, ordering, and score-transform
rules are testable without GPU, torch, Docker, or a reranker model.
`api/reranker.py` is the model wrapper.
"""

import numpy as np

# The same cap `POST /search` puts on `top_k`, so one request cannot make the
# service query an unbounded number of points.
RERANK_CANDIDATE_CAP = 100

# The `rerank_query` field is capped, so a long chat message becomes the head of the
# pair text instead of a request the service rejects.
RERANK_QUERY_MAX_CHARS = 4000


def pair_query(rerank_query, retrieval_query) -> str:
    """The text the cross-encoder judges the passages against.

    `rerank_query` is the caller's answer: the text to score. `None` (no field) is
    the retrieval query, which is the pre-P1.1 behavior. `question` matters on a
    long rewrite: `concat` puts the previous turns in the query, and a cross-encoder
    given that text scores the term overlap the history contributes - measured
    2026-11-18 on `eval/questions.multiturn.jsonl`, `mt-003`'s expected chunk went
    from dense rank 9 to rank 20 when the `concat` text was scored, and the top
    `rerank_score` values on its pool (`7.9453`, `4.3594`, `3.7812`) were chunks that
    share words with the history, not chunks that answer the question.
    """
    text = rerank_query if rerank_query not in (None, "") else retrieval_query
    return str(text).strip()[:RERANK_QUERY_MAX_CHARS]

# How the model's output becomes one relevance number per pair. `raw` keeps the
# classifier-head logits; `sigmoid` maps them to (0, 1); `softmax` reads a
# two-column output (the Qwen3-Reranker "no"/"yes" token logits) as P(yes).
TRANSFORMS = ("raw", "sigmoid", "softmax")

# A cross-encoder is trained on a labelled pair, and the Qwen3-Reranker family
# labels it: `Instruct: ...\nQuery: ...\nPassage: ...`. With plain pairs the model
# sees an unlabelled concatenation, and the measured symptom is the score itself:
# the startup probe for `Qwen/Qwen3-Reranker-0.6B` scored an obviously matching
# pair at `-2.0625`. `RERANK_QUERY_INSTRUCT` puts the instruction in front of the
# query, the same role `EMBEDDING_QUERY_PROMPT` plays for the embedding model.
QWEN3_RERANK_INSTRUCT = "Given a query, retrieve passages that answer the query."

# `RERANK_QUERY_INSTRUCT` values that name a known instruction. Any other non-empty
# value is used as the instruction text, so a model that ships its own wording can
# be configured without a code change.
KNOWN_INSTRUCTS = {"qwen3": QWEN3_RERANK_INSTRUCT}

# Values that mean "no prompt", spelled the way the other env knobs spell it.
OFF_INSTRUCTS = ("", "none", "off", "no", "false", "0")


def resolve_rerank_instruct(value):
    """The instruction to prefix the query with; `None` = plain pairs."""
    text = (value or "").strip()
    if text.lower() in OFF_INSTRUCTS:
        return None
    known = KNOWN_INSTRUCTS.get(text.lower())
    return known if known else text


def format_query(query, instruct):
    """`Instruct: ...\nQuery: ...`, the form the model was trained on."""
    if not instruct:
        return query
    return f"Instruct: {instruct}\nQuery: {query}"


def resolve_transform(name):
    """Validate `RERANK_TRANSFORM` and return the canonical spelling."""
    key = (name or "raw").strip().lower()
    if key not in TRANSFORMS:
        raise ValueError(
            f"Unknown RERANK_TRANSFORM '{name}', expected one of: {', '.join(TRANSFORMS)}"
        )
    return key


def apply_transform(values, transform):
    """Turn one model output per pair into one relevance number per pair.

    Monotone transforms only: `rerank_order` decides the ranking from these
    numbers, so a transform that is not monotone would change the answer. `raw`
    is enough for ranking; `sigmoid` and the softmax over the yes/no logits make
    `rerank_score` comparable between requests, which is what the citation
    headers show.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 2 and array.shape[1] == 2:
        # Two-column output: the Qwen3-Reranker "no"/"yes" token logits.
        exp = np.exp(array - array.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        return [float(value) for value in probs[:, 1]]
    flat = array.reshape(-1)
    if transform == "sigmoid":
        return [float(1.0 / (1.0 + np.exp(-value))) for value in flat]
    return [float(value) for value in flat]


def candidate_limit(top_k, rerank, hybrid=True, rerank_top_n=None):
    """How many candidates each Qdrant branch must return.

    Without reranking the hybrid branch needs `2 x top_k` items so RRF has
    material to rank (the P2.1 behaviour), and dense-only asks for `top_k`.
    With reranking both branches return `rerank_top_n` candidates: the list the
    cross-encoder sees is the candidate pool, so asking for a wide pool is what
    gives the model an intruder to judge. Capped at 100 and never below `top_k`,
    so enabling reranking can never shrink the answer.
    """
    top_k = max(1, int(top_k))
    if not rerank:
        return min(RERANK_CANDIDATE_CAP, top_k * 2) if hybrid else top_k

    pool = int(rerank_top_n) if rerank_top_n else top_k
    return min(RERANK_CANDIDATE_CAP, max(top_k, pool))


def resolve_rerank(request_flag, server_enabled, model_available):
    """Whether to rerank this request.

    The request flag wins over the service default (`RERANK_ENABLED`), so the
    evaluation harness can compare rerank on/off against one deployed
    configuration. A reranker that is not loaded — disabled, failed to load,
    or a model that the wrapper could not reach — never reranks: the fused
    order is kept, exactly as before reranking existed.
    """
    enabled = server_enabled if request_flag is None else bool(request_flag)
    return bool(enabled and model_available)


def passage_text(text, title=None, section=None, max_chars=None):
    """The passage handed to the cross-encoder.

    Chunk text with the document title and the section path as a prefix line, so
    the model sees the same citation context the gateway injects: a chunk that
    starts inside a long section reads differently once it says which section it
    is. Truncating keeps the total inside `max_chars`, so a long section path
    cannot push the body out of the model's input window.
    """
    prefix = " > ".join(
        part.strip() for part in (title, section) if part and part.strip()
    )
    body = text if text.strip() else " "

    if max_chars and max_chars > 0:
        if prefix:
            body = body[: max(0, max_chars - len(prefix) - 1)]
            return f"{prefix}\n{body}"
        return body[:max_chars]
    return f"{prefix}\n{body}" if prefix else body


def rerank_order(scores, top_k):
    """Indices of `scores`, best first, truncated to `top_k`.

    Ties keep the input order, which is the fused order: when the model scores
    two chunks identically, the RRF ranking decides, so the result stays
    deterministic and reproducible across runs. Negative scores (raw logits of a
    classifier-head model) order normally, so no transform is needed for ranking.
    """
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return order[: max(0, int(top_k))]


def _field(item, name, default=None):
    """Read a field of a candidate, from an object attribute or a dict key."""
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _set_score(item, value):
    """Write `rerank_score` on a candidate, in whatever form it is stored."""
    if isinstance(item, dict):
        item["rerank_score"] = value
    else:
        setattr(item, "rerank_score", value)


def candidate_passages(candidates, max_chars=None):
    """The passage text for every candidate, in the order they were given.

    Candidates arrive as the fused search results, so the fields are read
    whichever way they are stored. A chunk with no text still gets a passage: the
    cross-encoder scores the citation context, and the count of scores must match
    the count of candidates for the ordering to be valid.
    """
    return [
        passage_text(
            _field(item, "text", "") or "",
            _field(item, "title"),
            _field(item, "section"),
            max_chars,
        )
        for item in candidates
    ]


def rerank_results(candidates, scores, top_k):
    """Candidates ordered by the reranker, cut to `top_k`, with `rerank_score` set.

    Returns `None` when the score count does not match the candidates: the caller
    keeps the pre-rerank order, because a list that is shorter or longer than the
    pool cannot be reordered without dropping or duplicating a chunk. The input
    list is not modified, so the caller can return it unchanged after a failure.
    """
    if len(scores) != len(candidates):
        return None

    ordered = []
    for index in rerank_order(scores, top_k):
        item = candidates[index]
        _set_score(item, float(scores[index]))
        ordered.append(item)
    return ordered
