"""Unit tests for the Qdrant upgrade rehearsal (no network, no GPU, no Docker).

Run from the repository root:

    python -m unittest discover -s tests -p "test_*.py"

`scripts/qdrant_upgrade_check.py` decides whether a target Qdrant image can read
the data the current stack wrote. These tests cover the parts that produce the
verdict: collection statistics, the self-vector query, the pass/fail report, the
compose volume name, and the readiness poll.
"""

import sys
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import qdrant_upgrade_check as uc  # noqa: E402

LIVE = {"status": "green", "points": 202, "dim": 1024, "files": 5}
GREEN_INFO = {"status": "green", "points_count": 202, "config": {"params": {"vectors": {"size": 1024}}}}


def stats_http(path, body):
    if path.endswith("/collections/documents"):
        return GREEN_INFO
    if body["offset"] is None:
        return {"points": [{"payload": {"file": "/data/documents/a.txt"}}], "next_page_offset": 7}
    return {"points": [{"payload": {"file": "/data/documents/b.txt"}}], "next_page_offset": None}


class StatsTests(unittest.TestCase):
    def test_stats_count_pages_and_distinct_files(self):
        with patch.object(uc, "http", lambda method, base, path, body=None, timeout=60: stats_http(path, body)):
            stats = uc.collection_stats("http://q")
        self.assertEqual(stats, {"status": "green", "points": 202, "dim": 1024, "files": 2})

    def test_missing_collection_is_reported_as_missing(self):
        with patch.object(uc, "http", lambda method, base, path, body=None, timeout=60: {}):
            self.assertEqual(uc.collection_stats("http://q")["status"], "missing")

    def test_query_uses_a_stored_vector_and_finds_the_same_point(self):
        def fake_http(method, base, path, body=None, timeout=60):
            if "scroll" in path:
                return {"points": [{"id": 5, "vector": [0.1, 0.2], "payload": {"file": "/data/documents/a.txt"}}]}
            return {"points": [{"id": 5, "score": 1.0, "payload": {"file": "/data/documents/a.txt"}}]}

        with patch.object(uc, "http", fake_http):
            self.assertEqual(uc.self_query("http://q"), (5, 5, 1.0))

    def test_query_without_hits_returns_none(self):
        def fake_http(method, base, path, body=None, timeout=60):
            if "scroll" in path:
                return {"points": [{"id": 5, "vector": [0.1], "payload": {"file": "a"}}]}
            return {"points": []}

        with patch.object(uc, "http", fake_http):
            self.assertIsNone(uc.self_query("http://q"))


class ReportTests(unittest.TestCase):
    def test_pass_when_the_target_sees_the_same_data(self):
        self.assertTrue(uc.report("volume copy", LIVE, dict(LIVE), (5, 5, 1.0)))

    def test_fail_when_points_are_lost(self):
        target = dict(LIVE, points=12)
        self.assertFalse(uc.report("volume copy", LIVE, target, (5, 5, 1.0)))

    def test_fail_when_the_self_query_hits_a_different_point(self):
        self.assertFalse(uc.report("volume copy", LIVE, dict(LIVE), (5, 6, 0.71)))

    def test_fail_when_the_collection_is_not_green(self):
        self.assertFalse(uc.report("volume copy", LIVE, dict(LIVE, status="red"), (5, 5, 1.0)))


class VolumeNameTests(unittest.TestCase):
    LISTING = "DRIVER    VOLUME NAME\nlocal     some_other_volume\nlocal     rag_qdrant_data"

    def test_compose_volume_is_found_by_suffix(self):
        with patch.object(uc, "docker", lambda *args: self.LISTING):
            self.assertEqual(uc.find_live_volume(None), "rag_qdrant_data")

    def test_explicit_volume_is_used_as_is(self):
        self.assertEqual(uc.find_live_volume("explicit_volume"), "explicit_volume")

    def test_no_qdrant_volume_is_an_error(self):
        with patch.object(uc, "docker", lambda *args: "DRIVER    VOLUME NAME\nlocal     other"):
            with self.assertRaises(SystemExit):
                uc.find_live_volume(None)


class ReadinessTests(unittest.TestCase):
    def test_start_polls_until_the_port_answers(self):
        calls = {"polls": 0}

        def fake_http(method, base, path, body=None, timeout=60):
            calls["polls"] += 1
            if calls["polls"] < 3:
                raise urllib.error.URLError("connection refused")
            return {"collections": []}

        runs = []
        with patch.object(uc, "docker", lambda *args: runs.append(args) or ""), \
             patch.object(uc, "http", fake_http), \
             patch.object(uc.time, "sleep", lambda seconds: None):
            uc.start("qdrant/qdrant:v1.20.1", "scratch", 6335, fresh=False)

        self.assertEqual(calls["polls"], 3)
        self.assertIn("qdrant/qdrant:v1.20.1", runs[0])
        self.assertIn("6335:6333", runs[0])

    def test_start_gives_up_after_the_poll_budget(self):
        def fake_http(method, base, path, body=None, timeout=60):
            raise urllib.error.URLError("connection refused")

        with patch.object(uc, "docker", lambda *args: ""), \
             patch.object(uc, "http", fake_http), \
             patch.object(uc.time, "sleep", lambda seconds: None):
            with self.assertRaises(SystemExit):
                uc.start("qdrant/qdrant:latest", "scratch", 6335, fresh=True)


class CleanupTests(unittest.TestCase):
    def test_cleanup_removes_container_and_both_scratch_volumes(self):
        calls = []
        with patch.object(uc, "docker_soft", lambda *args: calls.append(args)):
            uc.cleanup()
        self.assertIn(("rm", "-f", uc.CONTAINER), calls)
        self.assertIn(("volume", "rm", f"{uc.SCRATCH_VOLUME}_copy"), calls)
        self.assertIn(("volume", "rm", f"{uc.SCRATCH_VOLUME}_fresh"), calls)


if __name__ == "__main__":
    unittest.main()
