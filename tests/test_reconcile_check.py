"""Unit tests for the reconcile check tool (no network, no GPU, no Docker).

Run from the repository root:

    python -m unittest discover -s tests -p "test_*.py"

`scripts/reconcile_check.py` compares the watched folder with the `payload.file`
values in Qdrant and checks whether a file appears in the top hits of a query.
These tests cover the parts that decide the verdict: the host-to-container path
mapping, the two directions of the `audit` diff, the presence checks behind
`--expect` and `--absent`, the scroll pagination, and the request bodies.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import scripts.reconcile_check as rc  # noqa: E402

PREFIX = "/data/documents"
A = "/data/documents/a.txt"
B = "/data/documents/b.txt"
C = "/data/documents/c.txt"


class PathMappingTests(unittest.TestCase):
    def test_index_path_uses_the_container_prefix(self):
        self.assertEqual(rc.index_path("/data/documents/", "report.md"), "/data/documents/report.md")
        self.assertEqual(rc.index_path(PREFIX, "sub/report.md"), "/data/documents/sub/report.md")

    def test_disk_files_maps_nested_files_to_indexed_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.txt").write_text("a")
            (root / "sub").mkdir()
            (root / "sub" / "b.md").write_text("b")
            self.assertEqual(
                rc.disk_files(str(root), PREFIX),
                {"/data/documents/a.txt", "/data/documents/sub/b.md"},
            )

    def test_compare_paths_reports_both_directions(self):
        on_disk = {A, B}
        in_index = {B, C}
        missing, orphaned = rc.compare_paths(on_disk, in_index)
        self.assertEqual(missing, [A])
        self.assertEqual(orphaned, [C])

    def test_clean_match_reports_nothing(self):
        self.assertEqual(rc.compare_paths({A}, {A}), ([], []))


class PresenceTests(unittest.TestCase):
    hits = [(A, 0.9), (B, 0.4)]

    def test_expect_passes_when_the_file_is_in_the_hits(self):
        self.assertTrue(rc.check_presence(self.hits, PREFIX, "a.txt", True))

    def test_absent_passes_when_the_file_is_not_in_the_hits(self):
        self.assertTrue(rc.check_presence(self.hits, PREFIX, "c.txt", False))

    def test_expect_fails_for_a_deleted_file(self):
        self.assertFalse(rc.check_presence(self.hits, PREFIX, "c.txt", True))


class ScrollTests(unittest.TestCase):
    def test_scroll_paginates_with_next_page_offset(self):
        bodies = []

        def fake_post(url, body):
            bodies.append((url, body))
            if len(bodies) == 1:
                return {"result": {"points": [{"payload": {"file": A}}], "next_page_offset": 42}}
            return {"result": {"points": [{"payload": {"file": B}}], "next_page_offset": None}}

        with patch.object(rc, "post", fake_post):
            files = rc.indexed_files("http://q", "documents")

        self.assertEqual(files, {A, B})
        self.assertEqual([body["offset"] for _, body in bodies], [None, 42])
        for url, body in bodies:
            self.assertEqual(url, "http://q/collections/documents/points/scroll")
            self.assertFalse(body["with_vector"])
            self.assertEqual(body["with_payload"], ["file"])

    def test_scroll_stops_when_the_offset_is_absent(self):
        calls = []

        def fake_post(url, body):
            calls.append(body)
            return {"result": {"points": [{"payload": {"file": A}}]}}

        with patch.object(rc, "post", fake_post):
            files = rc.indexed_files("http://q", "documents")

        self.assertEqual(files, {A})
        self.assertEqual(len(calls), 1, "the scroll loop must not spin without an offset")

    def test_points_repeat_as_one_path(self):
        def fake_post(url, body):
            return {
                "result": {
                    "points": [{"payload": {"file": A}}, {"payload": {"file": A}}],
                    "next_page_offset": None,
                }
            }

        with patch.object(rc, "post", fake_post):
            self.assertEqual(rc.indexed_files("http://q", "documents"), {A})


class RequestBodyTests(unittest.TestCase):
    def test_count_sends_a_file_filter(self):
        seen = []

        def fake_post(url, body):
            seen.append((url, body))
            return {"result": {"count": 3}}

        with patch.object(rc, "post", fake_post):
            count = rc.points_count("http://q", "documents", A)

        self.assertEqual(count, 3)
        url, body = seen[0]
        self.assertEqual(url, "http://q/collections/documents/points/count")
        self.assertTrue(body["exact"])
        self.assertEqual(body["filter"]["must"][0]["key"], "file")
        self.assertEqual(body["filter"]["must"][0]["match"]["value"], A)

    def test_total_sends_no_filter(self):
        seen = []

        def fake_post(url, body):
            seen.append(body)
            return {"result": {"count": 202}}

        with patch.object(rc, "post", fake_post):
            self.assertEqual(rc.points_count("http://q", "documents"), 202)
        self.assertNotIn("filter", seen[0])

    def test_search_sends_the_query_parameters(self):
        seen = []

        def fake_post(url, body):
            seen.append((url, body))
            return []

        with patch.object(rc, "post", fake_post):
            rc.search_hits("http://a", "marker text", 12)

        url, body = seen[0]
        self.assertEqual(url, "http://a/search")
        self.assertEqual(body, {"query": "marker text", "top_k": 12, "min_score": None})

    def test_search_hits_accepts_the_list_response_of_api(self):
        with patch.object(rc, "post", lambda url, body: [{"score": 0.83, "file": A}]):
            self.assertEqual(rc.search_hits("http://a", "q", 20), [(A, 0.83)])

    def test_search_hits_accepts_a_results_dict_with_payload(self):
        response = {"results": [{"score": 0.5, "payload": {"file": B}}]}
        with patch.object(rc, "post", lambda url, body: response):
            self.assertEqual(rc.search_hits("http://a", "q", 20), [(B, 0.5)])


if __name__ == "__main__":
    unittest.main()
