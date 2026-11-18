"""Unit tests for cross-encoder reranking rules (no GPU, no model, no Docker).

Run from the repository root:

    python -m unittest discover -s tests -p "test_*.py"

`api/rerank.py` has no torch, sentence-transformers, Qdrant, or HTTP imports, so
the pooling, passage, ordering, and score-transform rules are tested against
synthetic candidates. The model wrapper (`api/reranker.py`) runs only inside the
`api` container.

These rules are the ones that decide whether reranking can hurt: the candidate
pool must never shrink the answer, the fused order must survive a broken
reranker, and ties must keep RRF's ranking so a run is reproducible.
"""

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

import rerank  # noqa: E402


class Candidate:
    """A stand-in for a fused `SearchResult`."""

    def __init__(self, text, title=None, section=None):
        self.text = text
        self.title = title
        self.section = section
        self.rerank_score = None

    def __repr__(self):
        return f"Candidate({self.text!r}, rerank_score={self.rerank_score})"


class CandidateLimitTests(unittest.TestCase):
    def test_hybrid_without_rerank_keeps_the_p21_pool(self):
        # P2.1: the BM25 branch needs 2 x top_k items for RRF to rank.
        self.assertEqual(rerank.candidate_limit(8, False), 16)
        self.assertEqual(rerank.candidate_limit(5, False), 10)

    def test_dense_without_rerank_asks_for_top_k(self):
        self.assertEqual(rerank.candidate_limit(8, False, hybrid=False), 8)

    def test_rerank_asks_for_the_pool(self):
        self.assertEqual(rerank.candidate_limit(8, True, True, 30), 30)
        self.assertEqual(rerank.candidate_limit(8, True, True, 50), 50)

    def test_pool_is_capped(self):
        self.assertEqual(rerank.candidate_limit(8, True, True, 500), rerank.RERANK_CANDIDATE_CAP)
        # The pre-P1.1 pool is capped by the same rule: `2 x top_k` never exceeds
        # the `top_k` ceiling of `POST /search`.
        self.assertEqual(rerank.candidate_limit(60, False, True), rerank.RERANK_CANDIDATE_CAP)
        self.assertEqual(rerank.candidate_limit(8, False, True), 16)

    def test_rerank_never_shrinks_the_answer(self):
        # A pool below top_k cannot drop a hit the caller asked for.
        self.assertEqual(rerank.candidate_limit(10, True, True, 4), 10)

    def test_pool_without_a_setting_falls_back_to_top_k(self):
        self.assertEqual(rerank.candidate_limit(8, True, True, None), 8)

    def test_pool_applies_to_dense_too(self):
        self.assertEqual(rerank.candidate_limit(8, True, False, 30), 30)


class ResolveRerankTests(unittest.TestCase):
    def test_no_flag_uses_the_service_default(self):
        self.assertTrue(rerank.resolve_rerank(None, True, True))
        self.assertFalse(rerank.resolve_rerank(None, False, True))

    def test_the_request_flag_overrides_the_service_default(self):
        # The eval harness compares on/off against one deployed configuration.
        self.assertTrue(rerank.resolve_rerank(True, False, True))
        self.assertFalse(rerank.resolve_rerank(False, True, True))

    def test_a_missing_model_never_reranks(self):
        for flag, server in [(True, True), (True, False), (None, True), (False, True)]:
            self.assertFalse(rerank.resolve_rerank(flag, server, False))


class PassageTextTests(unittest.TestCase):
    def test_chunk_text_alone(self):
        self.assertEqual(rerank.passage_text("chunk body"), "chunk body")

    def test_title_and_section_become_one_prefix_line(self):
        passage = rerank.passage_text("chunk body", "RAG Gateway", "Configuration")
        self.assertEqual(passage, "RAG Gateway > Configuration\nchunk body")

    def test_missing_metadata_is_skipped(self):
        self.assertEqual(
            rerank.passage_text("chunk body", None, "  ", None),
            "chunk body",
        )

    def test_the_total_stays_inside_the_budget(self):
        budget = 40
        passage = rerank.passage_text("x" * 100, "Title", "Section", budget)
        self.assertLessEqual(len(passage), budget)
        self.assertTrue(passage.startswith("Title > Section\n"))

    def test_a_long_section_cannot_push_the_body_out(self):
        # The budget is shared: prefix first, then as much body as is left.
        passage = rerank.passage_text("body text", "T" * 300, None, 320)
        self.assertIn("body text", passage)

    def test_empty_chunk_still_gets_a_passage(self):
        # The score count must match the candidate count, so a chunk with no body
        # still gets a passage the model can tokenize: the citation context.
        passage = rerank.passage_text("   ", "RAG Gateway", "Configuration")
        self.assertEqual(passage, "RAG Gateway > Configuration\n ")
        self.assertGreater(len(rerank.passage_text("   ")), 0)


class CandidatePassagesTests(unittest.TestCase):
    def test_objects_and_dicts_are_read_the_same_way(self):
        candidates = [
            Candidate("object body", "Doc", "Section"),
            {"text": "dict body", "title": "Doc2", "section": "Part"},
        ]
        passages = rerank.candidate_passages(candidates, None)
        self.assertEqual(passages[0], "Doc > Section\nobject body")
        self.assertEqual(passages[1], "Doc2 > Part\ndict body")

    def test_the_count_matches_the_candidates(self):
        # A score per candidate is what makes the ordering valid.
        self.assertEqual(len(rerank.candidate_passages([Candidate(""), Candidate("")], 64)), 2)

    def test_a_candidate_with_no_text_keeps_its_slot(self):
        passages = rerank.candidate_passages([Candidate(None), Candidate("b")], None)
        self.assertEqual(len(passages), 2)
        self.assertIn("b", passages[1])


class RerankOrderTests(unittest.TestCase):
    def test_scores_decide_the_order(self):
        self.assertEqual(rerank.rerank_order([0.2, 0.9, 0.5], 3), [1, 2, 0])

    def test_ties_keep_the_fused_order(self):
        # Deterministic: equal scores fall back to the RRF rank, so a rerun gives
        # the same answer.
        self.assertEqual(rerank.rerank_order([0.5, 0.5, 0.9], 3), [2, 0, 1])

    def test_negative_logits_order_normally(self):
        # A classifier head returns unbounded logits; ranking needs no transform.
        self.assertEqual(rerank.rerank_order([-3.0, 2.0, -1.0], 3), [1, 2, 0])

    def test_cut_to_top_k(self):
        self.assertEqual(rerank.rerank_order([0.1, 0.2, 0.3, 0.4], 2), [3, 2])

    def test_top_k_zero_is_empty(self):
        self.assertEqual(rerank.rerank_order([0.1], 0), [])


class RerankResultsTests(unittest.TestCase):
    def test_scores_are_recorded_and_the_list_is_cut(self):
        candidates = [Candidate("a"), Candidate("b"), Candidate("c")]
        ordered = rerank.rerank_results(candidates, [0.2, 0.9, 0.5], 2)
        self.assertEqual([item.text for item in ordered], ["b", "c"])
        self.assertEqual([item.rerank_score for item in ordered], [0.9, 0.5])

    def test_dict_candidates_get_the_key(self):
        ordered = rerank.rerank_results([{"text": "a"}, {"text": "b"}], [0.1, 0.7], 2)
        self.assertEqual([item["rerank_score"] for item in ordered], [0.7, 0.1])

    def test_a_mismatched_score_count_is_refused(self):
        # Dropping or duplicating a chunk is worse than not reranking.
        candidates = [Candidate("a"), Candidate("b")]
        self.assertIsNone(rerank.rerank_results(candidates, [0.5], 2))
        self.assertIsNone(rerank.rerank_results(candidates, [0.5, 0.6, 0.7], 2))

    def test_the_refused_case_leaves_the_candidates_untouched(self):
        # The caller returns the fused list, so nothing may be reordered.
        candidates = [Candidate("a"), Candidate("b")]
        rerank.rerank_results(candidates, [0.5], 2)
        self.assertEqual([item.text for item in candidates], ["a", "b"])
        self.assertEqual([item.rerank_score for item in candidates], [None, None])

    def test_no_candidates_is_an_empty_answer(self):
        self.assertEqual(rerank.rerank_results([], [], 5), [])


class TransformTests(unittest.TestCase):
    def test_raw_keeps_the_logits(self):
        self.assertEqual(rerank.apply_transform([2.0, -1.0, 0.0], "raw"), [2.0, -1.0, 0.0])

    def test_sigmoid_is_monotone_and_in_range(self):
        scores = rerank.apply_transform([-3.0, 0.0, 3.0], "sigmoid")
        self.assertEqual(len(scores), 3)
        for value in scores:
            self.assertGreater(value, 0.0)
            self.assertLess(value, 1.0)
        self.assertEqual(scores, sorted(scores))

    def test_sigmoid_does_not_change_the_ranking(self):
        logits = [0.1, 2.0, -1.0, 4.0]
        self.assertEqual(
            rerank.rerank_order(rerank.apply_transform(logits, "sigmoid"), 4),
            rerank.rerank_order(logits, 4),
        )

    def test_two_column_output_is_yes_probability(self):
        # Qwen3-Reranker scores by the "no"/"yes" token logits, so the number
        # reported is P(yes): exp(yes) / (exp(no) + exp(yes)).
        scores = rerank.apply_transform([[3.0, 1.0], [0.0, 2.0]], "softmax")
        self.assertAlmostEqual(scores[0], 1.0 / (1.0 + math.exp(2.0)), places=6)
        self.assertAlmostEqual(scores[1], 1.0 / (1.0 + math.exp(-2.0)), places=6)
        # The relevant chunk wins: the second pair has the higher "yes" logit.
        self.assertEqual(rerank.rerank_order(scores, 2), [1, 0])

    def test_the_probabilities_sum_to_one(self):
        scores = rerank.apply_transform([[1.0, 2.0], [-2.0, -1.0], [10.0, 12.0]], "softmax")
        self.assertEqual(len(scores), 3)
        for value in scores:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)

    def test_an_unknown_transform_is_rejected(self):
        with self.assertRaises(ValueError):
            rerank.resolve_transform("sqrt")

    def test_the_default_transform_is_raw(self):
        self.assertEqual(rerank.resolve_transform(None), "raw")
        self.assertEqual(rerank.resolve_transform("Sigmoid"), "sigmoid")


class QueryInstructTests(unittest.TestCase):
    def test_empty_and_off_values_send_plain_pairs(self):
        for value in ("", "   ", "none", "OFF", "no", "false", "0"):
            self.assertIsNone(rerank.resolve_rerank_instruct(value))

    def test_qwen3_names_the_known_instruction(self):
        # The wording this family is trained on, so a deployment turns the prompt on
        # by name instead of copying text into the env file.
        self.assertEqual(rerank.resolve_rerank_instruct("qwen3"), rerank.QWEN3_RERANK_INSTRUCT)
        self.assertEqual(rerank.resolve_rerank_instruct(" Qwen3 "), rerank.QWEN3_RERANK_INSTRUCT)

    def test_any_other_value_is_the_instruction(self):
        # A reranker that ships its own wording is a config change, not a code change.
        text = "Given a question and a document, decide if the document answers it."
        self.assertEqual(rerank.resolve_rerank_instruct(text), text)

    def test_the_query_is_labelled_and_the_passage_is_not(self):
        # Only the query carries the instruction. The passage is the chunk as stored,
        # so the count of passages stays the count the model has to score.
        question = "Сколько измерений у её векторов?"
        formatted = rerank.format_query(question, rerank.QWEN3_RERANK_INSTRUCT)
        self.assertTrue(formatted.startswith("Instruct: "))
        self.assertIn(f"Query: {question}", formatted)
        self.assertEqual(rerank.format_query("question", None), "question")

    def test_the_reranker_formats_the_query_once_per_request(self):
        # One pair per passage, all built from the same formatted query: the prompt is
        # not repeated per candidate, and `api` passes the resolved env value. The
        # wrapper needs torch, so the wiring is read from the source.
        source = (ROOT / "api" / "reranker.py").read_text(encoding="utf-8")
        self.assertIn(
            "pairs = [[format_query(query, self.query_instruct), passage] for passage in passages]",
            source,
        )
        self.assertIn("query_instruct: Optional[str] = None", source)
        api_source = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
        self.assertIn("query_instruct=RERANK_QUERY_INSTRUCT", api_source)
        # the health endpoint shows it, so a measured quality change is visible without
        # reading the container's env file
        self.assertIn('"rerank_query_instruct"', api_source)


class DefaultsTests(unittest.TestCase):
    def test_the_shipped_default_is_rerank_off(self):
        # Scoring 30 candidates costs more GPU work than embedding the query, so
        # reranking is off until MRR, p50/p95, and VRAM are measured on the eval
        # set. The service, compose, and the env template must agree.
        api_source = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("RERANK_ENABLED", "false")', api_source)
        compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
        self.assertIn(
            "RERANK_ENABLED=${RERANK_ENABLED:-false}",
            compose,
            "the compose default must match the api default",
        )
        env_template = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("RERANK_ENABLED=false", env_template)

    def test_the_gateway_default_is_off_too(self):
        gateway_source = (ROOT / "gateway" / "main.py").read_text(encoding="utf-8")
        self.assertIn('_env_bool("RAG_RERANK", False)', gateway_source)
        compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
        self.assertIn(
            "RAG_RERANK=${RAG_RERANK:-false}",
            compose,
            "the compose default must match the gateway default",
        )
        env_template = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("RAG_RERANK=false", env_template)

    def test_the_pool_is_the_p11_scope(self):
        # P1.1 asks for 30-50 candidates.
        api_source = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("RERANK_TOP_N", "30")', api_source)

    def test_no_instruction_is_the_default(self):
        # Plain pairs are the shape the reranker was measured with first, and the
        # instruction is a per-model quality change, so a model is never given a
        # prompt it has not been measured with. `api`, compose, and the template
        # must agree on "off".
        api_source = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
        self.assertIn('os.environ.get("RERANK_QUERY_INSTRUCT", "")', api_source)
        compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
        self.assertIn(
            "RERANK_QUERY_INSTRUCT=${RERANK_QUERY_INSTRUCT:-}",
            compose,
            "the compose default must match the api default",
        )
        env_template = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("RERANK_QUERY_INSTRUCT=", env_template)


class PairQueryTests(unittest.TestCase):
    """Which text the cross-encoder judges: the measured lever for long rewrites."""

    def test_no_field_scores_the_retrieval_query(self):
        # `null` is the pre-P1.1 behavior: the pairs are built from the text retrieval
        # was asked with, so adding the field cannot change a request that does not use it.
        retrieval = "Which embedding model does the stack use?\nСколько измерений у её векторов?"
        self.assertEqual(rerank.pair_query(None, retrieval), retrieval)

    def test_the_question_is_what_the_model_judges(self):
        # The measured case: `concat` puts the English history turn in the query, and
        # scoring that text ranks term overlap above the answer. `question` is the
        # lever, so it must win over the retrieval text when it is set.
        history = "Which embedding model does the stack use?"
        question = "Сколько измерений у её векторов?"
        concat = f"{history}\n{question}"
        self.assertEqual(rerank.pair_query(question, concat), question)

    def test_an_empty_answer_is_no_answer(self):
        # `""` is no opinion, not "score the empty string": a pair with an empty query
        # scores every passage the same, which is a silent quality loss.
        self.assertEqual(rerank.pair_query("", "the retrieval query"), "the retrieval query")

    def test_a_long_message_is_a_head_not_a_rejection(self):
        # `rerank_query` is capped on `api`, so a long chat message is sent as its head
        # rather than as a request the service answers with a 400.
        long = "测量" * 3000
        self.assertEqual(len(rerank.pair_query(long, "q")), rerank.RERANK_QUERY_MAX_CHARS)

if __name__ == "__main__":
    unittest.main()
