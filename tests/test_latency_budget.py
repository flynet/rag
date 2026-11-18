"""The latency probe must fit inside its wall-clock budget (stdlib only, no GPU).

Run from the repository root:

    python -m unittest discover -s tests -p "test_*.py"

The probe used to run for hours: the indexer wait was 240s, and every request on a
connection that had not been used cost ~20s on the live stack, so a phase that
measured a 1s request read 20s. These tests check the two fixes on a fake session
that answers instantly, so they run in well under a second.
"""

import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import latency_probe as lp  # noqa: E402


class FakeSession:
    """Counts the requests the probe sends, and answers instantly."""

    def __init__(self) -> None:
        self.calls = 0

    def post(self, path, body):
        self.calls += 1
        return {}, {}


class BudgetTests(unittest.TestCase):
    def test_a_used_up_budget_sends_nothing(self):
        session = FakeSession()
        latencies = lp.measure_search([session], ["query"], 20, None, True, 30,
                                      lp.Budget(0.0))
        self.assertEqual(latencies, [])
        self.assertEqual(session.calls, 0, "no request is sent once the budget is gone")

    def test_a_duration_window_is_cut_to_the_time_left(self):
        # The phase asks for 10 seconds of search overlapping the indexer. The budget
        # allows 0.2, so the phase ends at 0.2: a window that outlives its run is what
        # made the probe take hours.
        session = FakeSession()
        budget = lp.Budget(0.2)
        start = time.monotonic()
        lp.measure_search([session], ["query"], 20, 10.0, True, 30, budget)
        self.assertLess(time.monotonic() - start, 1.0)
        self.assertGreater(session.calls, 0, "the shortened window does run requests")

    def test_the_encode_phase_stops_at_the_budget_too(self):
        session = FakeSession()
        latencies = lp.measure_encode(session, ["a", "b"], 5, lp.Budget(0.0))
        self.assertEqual(latencies, [])
        self.assertEqual(session.calls, 0)

    def test_a_phase_reports_the_time_it_has_left(self):
        # A run that overruns has to be diagnosed at the phase that overran, so each
        # phase says how much budget is left, not just the numbers it measured.
        stats = lp.report("idle", lp.summarise([1.0, 2.0]), lp.Budget(30.0))
        self.assertEqual(stats["count"], 2)
        self.assertAlmostEqual(stats["budget_left"], 30.0, delta=0.5)


class WarmUpTests(unittest.TestCase):
    def test_every_connection_is_warmed_once(self):
        # Measured on the live stack: the first request on a connection that has not
        # been used costs ~20s, the same request on a warmed connection ~1s. Warming
        # all of them is what lets a timed phase read the queue, not the connection.
        sessions = [FakeSession() for _ in range(3)]
        lp.warm_up(sessions, "query", True, 30)
        self.assertEqual([s.calls for s in sessions], [1, 1, 1])

    def test_the_warm_up_sends_the_rerank_request(self):
        # Warming the rerank path warms the reranker, so the fused phase measured
        # after it does not pay the model. The request the warm-up sends is the one
        # the phase will measure, so the pool key has to travel with it.
        session = FakeSession()
        seen = {}

        def post(path, body):
            session.calls += 1
            seen.update(body)
            return {}, {}

        session.post = post
        lp.warm_up([session], "query", True, 30)
        self.assertEqual(seen["rerank"], True)
        self.assertEqual(seen["rerank_top_n"], 30)


if __name__ == "__main__":
    unittest.main()
