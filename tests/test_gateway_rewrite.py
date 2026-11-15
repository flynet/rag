"""Unit tests for the gateway's follow-up query rewriting (no network, no GPU).

Run from the repository root:

    python -m unittest discover -s tests -p "test_*.py"

The rewrite decides which text the gateway embeds for a follow-up turn. These
tests cover the three modes on synthetic conversations: single-turn requests
keep the raw query, elliptical follow-ups get the recent user turns, and a
failing `llm` rewrite falls back to concatenation.
"""

import asyncio
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gateway.main as gw  # noqa: E402

TOP_K_TURN = "What is RAG_TOP_K in the gateway?"
FOLLOW_UP = "and the min score?"
LONG_QUESTION = "How do I set the embedding batch size and the indexer cache?"


def messages(*turns):
    """Chat messages: user turns in order, the last one is the current query."""
    return [{"role": "user", "content": turn} for turn in turns]


def prepare(mode, msgs, query, llm=None):
    """Run _prepare_retrieval_query with the mode patched, and no upstream call."""
    with patch.object(gw, "RAG_QUERY_REWRITE", mode), patch.object(
        gw, "RAG_REWRITE_HISTORY", 3
    ), patch.object(gw, "_rewrite_cache", {}):
        if llm is None:
            return asyncio.run(gw._prepare_retrieval_query(msgs, query, "test-model"))
        with patch.object(gw, "_rewrite_query_llm", llm):
            return asyncio.run(gw._prepare_retrieval_query(msgs, query, "test-model"))


def async_returns(value):
    async def call(*args):
        return value

    return call


def async_raises(exc):
    async def call(*args):
        raise exc

    return call


class FollowUpHeuristicTests(unittest.TestCase):
    def test_short_follow_up_is_detected(self):
        self.assertTrue(gw._is_follow_up(FOLLOW_UP))

    def test_anaphora_is_detected(self):
        self.assertTrue(gw._is_follow_up("what about into a fresh instance?"))

    def test_full_long_question_is_not_a_follow_up(self):
        self.assertFalse(gw._is_follow_up(LONG_QUESTION))

    def test_empty_message_is_not_a_follow_up(self):
        self.assertFalse(gw._is_follow_up("   "))


class SingleTurnTests(unittest.TestCase):
    def test_first_turn_is_untouched_in_concat(self):
        self.assertEqual(prepare("concat", messages(FOLLOW_UP), FOLLOW_UP), FOLLOW_UP)

    def test_first_turn_is_untouched_in_llm(self):
        self.assertEqual(
            prepare("llm", messages(FOLLOW_UP), FOLLOW_UP, async_returns("x")),
            FOLLOW_UP,
        )

    def test_long_question_with_history_is_untouched(self):
        msgs = messages(TOP_K_TURN, LONG_QUESTION)
        self.assertEqual(prepare("concat", msgs, LONG_QUESTION), LONG_QUESTION)

    def test_history_of_one_turn_is_the_current_query(self):
        self.assertEqual(gw._history_user_texts(messages(FOLLOW_UP), 3), [])


class ConcatModeTests(unittest.TestCase):
    def test_off_keeps_the_raw_follow_up(self):
        msgs = messages(TOP_K_TURN, FOLLOW_UP)
        self.assertEqual(prepare("off", msgs, FOLLOW_UP), FOLLOW_UP)

    def test_concat_prepends_the_previous_user_turn(self):
        msgs = messages(TOP_K_TURN, FOLLOW_UP)
        query = prepare("concat", msgs, FOLLOW_UP)
        self.assertTrue(query.startswith(TOP_K_TURN))
        self.assertTrue(query.endswith(FOLLOW_UP))

    def test_concat_uses_only_the_recent_turns(self):
        with patch.object(gw, "RAG_REWRITE_HISTORY", 1):
            msgs = messages("the oldest turn", TOP_K_TURN, FOLLOW_UP)
            query = asyncio.run(gw._prepare_retrieval_query(msgs, FOLLOW_UP, "test-model"))
        self.assertNotIn("the oldest turn", query)
        self.assertIn(TOP_K_TURN, query)

    def test_assistant_turns_are_not_prepended(self):
        msgs = [
            {"role": "user", "content": TOP_K_TURN},
            {"role": "assistant", "content": "RAG_TOP_K is 8."},
            {"role": "user", "content": FOLLOW_UP},
        ]
        query = prepare("concat", msgs, FOLLOW_UP)
        self.assertNotIn("RAG_TOP_K is 8.", query)


class LlmModeTests(unittest.TestCase):
    def test_standalone_query_is_used(self):
        msgs = messages(TOP_K_TURN, FOLLOW_UP)
        standalone = "What is RAG_MIN_SCORE in the gateway?"
        query = prepare("llm", msgs, FOLLOW_UP, async_returns(standalone))
        self.assertEqual(query, standalone)

    def test_failure_falls_back_to_concat(self):
        msgs = messages(TOP_K_TURN, FOLLOW_UP)
        query = prepare("llm", msgs, FOLLOW_UP, async_raises(ValueError("no rewrite")))
        self.assertTrue(query.startswith(TOP_K_TURN))
        self.assertTrue(query.endswith(FOLLOW_UP))

    def test_repeated_follow_up_is_cached(self):
        calls = []

        async def counting(*args):
            calls.append(args)
            return "standalone query"

        msgs = messages(TOP_K_TURN, FOLLOW_UP)
        # The cache must be patched once for both requests: patching a fresh dict
        # inside `prepare` would give each call an empty cache.
        with patch.object(gw, "RAG_QUERY_REWRITE", "llm"), patch.object(
            gw, "RAG_REWRITE_HISTORY", 3
        ), patch.object(gw, "_rewrite_cache", {}), patch.object(
            gw, "_rewrite_query_llm", counting
        ):
            first = asyncio.run(gw._prepare_retrieval_query(msgs, FOLLOW_UP, "test-model"))
            second = asyncio.run(gw._prepare_retrieval_query(msgs, FOLLOW_UP, "test-model"))
        self.assertEqual((first, second), ("standalone query", "standalone query"))
        self.assertEqual(len(calls), 1)

    def test_non_follow_up_never_costs_an_upstream_call(self):
        calls = []

        async def counting(*args):
            calls.append(args)
            return "standalone query"

        msgs = messages(TOP_K_TURN, LONG_QUESTION)
        self.assertEqual(prepare("llm", msgs, LONG_QUESTION, counting), LONG_QUESTION)
        self.assertEqual(calls, [])


class DefaultModeTests(unittest.TestCase):
    def test_shipped_default_is_concat(self):
        source = (ROOT / "gateway" / "main.py").read_text(encoding="utf-8")
        default = re.search(
            r'os\.environ\.get\("RAG_QUERY_REWRITE", "([a-z]+)"\)', source
        )
        self.assertIsNotNone(default, "the rewrite default is not read from the env")
        self.assertEqual(
            default.group(1),
            "concat",
            "the default mode must be the measured one: on "
            "eval/questions.multiturn.jsonl, off leaves 2 of 9 follow-ups with no "
            "expected hit in the top ten (recall@10 0.778, MRR@10 0.277) and concat "
            "closes both (recall@10 1.000, MRR@10 0.420) with no extra upstream call",
        )
        compose = (ROOT / "docker-compose.yaml").read_text(encoding="utf-8")
        self.assertIn(
            "RAG_QUERY_REWRITE=${RAG_QUERY_REWRITE:-concat}",
            compose,
            "the compose default must match the gateway default",
        )
        env_template = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn(
            "RAG_QUERY_REWRITE=concat",
            env_template,
            "the env template must document the same default",
        )


if __name__ == "__main__":
    unittest.main()
