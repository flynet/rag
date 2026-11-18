"""Unit tests for the structure-aware chunker (stdlib only).

Run from the repository root:

    python -m unittest discover -s tests -p "test_*.py"

`embedder/chunking.py` has no third-party imports, so the splitter, the heading
parser, the citation offsets, and the overlap rules are tested without Qdrant,
GPU, Docker, torch, or fastembed. The tests cover paragraph and sentence
preference, hard cuts for long words, section paths (`Guide > Installation`),
character offsets that refer to the normalized text, overlap aligned to word
starts, CRLF handling, and the title-prepending rule used before embedding.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "embedder"))

from chunking import (  # noqa: E402
    CHUNKER_VERSION,
    TextChunk,
    document_title,
    split_text_into_chunks,
)

SECTIONS = """# Guide

alpha beta gamma delta.

## Installation

install these steps.

## Usage

use it like this.
"""

# A fenced block hides its "#" line from the heading parser.
FENCED = (
    "# Real\n"
    "\n"
    "intro text.\n"
    "\n"
    "```text\n"
    "# Not a heading\n"
    "```\n"
    "\n"
    "after the block.\n"
)


class EmptyInputTests(unittest.TestCase):
    def test_blank_text_yields_no_chunks(self):
        self.assertEqual(split_text_into_chunks(""), [])
        self.assertEqual(split_text_into_chunks("   \n\t\n "), [])


class BoundaryTests(unittest.TestCase):
    def test_paragraph_break_is_preferred(self):
        text = "A" * 100 + "\n\n" + "B" * 100
        chunks = split_text_into_chunks(text, chunk_size=120, overlap=0)

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "A" * 100)
        self.assertEqual(chunks[1].text, "B" * 100)

    def test_sentence_end_is_used_when_no_paragraph_fits(self):
        text = "Word one. Word two. Word three."
        chunks = split_text_into_chunks(text, chunk_size=20, overlap=0)

        self.assertEqual(chunks[0].text, "Word one. Word two.")
        self.assertTrue(chunks[0].text.endswith("."))

    def test_long_word_without_whitespace_is_hard_cut(self):
        text = "x" * 2500
        chunks = split_text_into_chunks(text, chunk_size=1000, overlap=100)

        self.assertEqual([len(c.text) for c in chunks], [1000, 1000, 500])
        self.assertTrue(all(len(c.text) <= 1000 for c in chunks))

    def test_chunks_never_exceed_chunk_size(self):
        text = "word " * 900
        chunks = split_text_into_chunks(text, chunk_size=128, overlap=32)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(c.text) <= 128 for c in chunks))

    def test_chunks_are_trimmed(self):
        text = "\n\n  alpha beta gamma  \n\n  delta epsilon  \n\n"
        for chunk in split_text_into_chunks(text, chunk_size=32, overlap=0):
            self.assertEqual(chunk.text, chunk.text.strip())

    def test_overlap_is_at_most_the_configured_size(self):
        text = " ".join(f"w{index}" for index in range(400))
        chunks = split_text_into_chunks(text, chunk_size=100, overlap=20)

        self.assertGreater(len(chunks), 1)
        for previous, current in zip(chunks, chunks[1:]):
            shared = previous.char_end - current.char_start
            self.assertGreater(shared, 0)
            self.assertLessEqual(shared, 20)

    def test_overlap_starts_on_a_word_boundary(self):
        text = " ".join(f"token{index}" for index in range(400))
        chunks = split_text_into_chunks(text, chunk_size=100, overlap=40)

        for chunk in chunks[1:]:
            self.assertTrue(
                text[chunk.char_start - 1].isspace(),
                f"chunk starts inside a word at {chunk.char_start}",
            )
            self.assertFalse(chunk.text[0].isspace())

    def test_zero_overlap_produces_contiguous_chunks(self):
        text = " ".join(f"w{index}" for index in range(100))
        chunks = split_text_into_chunks(text, chunk_size=80, overlap=0)

        for previous, current in zip(chunks, chunks[1:]):
            self.assertGreaterEqual(current.char_start, previous.char_end)


class SectionTests(unittest.TestCase):
    def test_chunks_carry_the_nested_section_path(self):
        chunks = split_text_into_chunks(SECTIONS, chunk_size=40, overlap=0)

        self.assertEqual(
            [chunk.section for chunk in chunks],
            ["Guide", "Guide > Installation", "Guide > Usage"],
        )

    def test_chunk_does_not_swallow_the_next_heading(self):
        chunks = split_text_into_chunks(SECTIONS, chunk_size=40, overlap=0)

        self.assertNotIn("## Installation", chunks[0].text)
        self.assertTrue(chunks[1].text.startswith("## Installation"))

    def test_headings_inside_fenced_blocks_are_ignored(self):
        chunks = split_text_into_chunks(FENCED, chunk_size=1024, overlap=0)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].section, "Real")
        self.assertNotIn("Not a heading", chunks[0].section)

    def test_first_chunk_of_a_headingless_document_uses_the_title(self):
        chunks = split_text_into_chunks("plain text without headings", chunk_size=100)

        self.assertEqual(chunks[0].section, "plain text without headings")


class OffsetTests(unittest.TestCase):
    def test_offsets_slice_the_normalized_text(self):
        chunks = split_text_into_chunks(SECTIONS, chunk_size=40, overlap=0)

        for chunk in chunks:
            self.assertEqual(SECTIONS[chunk.char_start : chunk.char_end], chunk.text)

    def test_offsets_are_monotonic(self):
        chunks = split_text_into_chunks(SECTIONS, chunk_size=40, overlap=0)

        starts = [chunk.char_start for chunk in chunks]
        self.assertEqual(starts, sorted(starts))
        for chunk in chunks:
            self.assertLess(chunk.char_start, chunk.char_end)

    def test_crlf_is_normalized_before_offsets_are_taken(self):
        chunks = split_text_into_chunks("# T\r\n\r\nBody line.\r\n", chunk_size=1024)

        self.assertEqual(len(chunks), 1)
        self.assertNotIn("\r", chunks[0].text)
        self.assertEqual(chunks[0].text, "# T\n\nBody line.")

    def test_chunking_is_deterministic(self):
        first = split_text_into_chunks(SECTIONS, chunk_size=40, overlap=10)
        second = split_text_into_chunks(SECTIONS, chunk_size=40, overlap=10)

        self.assertEqual(first, second)


class TitleTests(unittest.TestCase):
    def test_title_prefers_the_first_heading(self):
        self.assertEqual(document_title(SECTIONS), "Guide")

    def test_title_falls_back_to_the_first_line(self):
        self.assertEqual(document_title("Just a line\nsecond line"), "Just a line")

    def test_title_falls_back_to_the_file_stem(self):
        self.assertEqual(document_title("   \n\n", Path("report.md")), "report")

    def test_embedding_text_prepends_the_title_once(self):
        chunk = TextChunk(text="Body text.", section="S", char_start=0, char_end=10)

        self.assertEqual(chunk.embedding_text("Doc"), "Doc\n\nBody text.")
        self.assertEqual(chunk.embedding_text(""), "Body text.")

    def test_embedding_text_does_not_duplicate_a_leading_title(self):
        chunk = TextChunk(
            text="# Doc\n\nBody text.", section="Doc", char_start=0, char_end=17
        )

        self.assertEqual(chunk.embedding_text("Doc"), "# Doc\n\nBody text.")

    def test_chunker_version_is_part_of_the_signature_contract(self):
        self.assertIsInstance(CHUNKER_VERSION, int)
        self.assertGreaterEqual(CHUNKER_VERSION, 2)


class CorpusShapeTests(unittest.TestCase):
    def test_large_document_is_fully_covered(self):
        paragraphs = [f"Paragraph {index} with a sentence in it. " * 3 for index in range(200)]
        text = "\n\n".join(paragraphs)
        chunks = split_text_into_chunks(text, chunk_size=1024, overlap=128)

        self.assertGreater(len(chunks), 10)
        self.assertTrue(all(len(chunk.text) <= 1024 for chunk in chunks))
        self.assertEqual(chunks[0].char_start, 0)
        self.assertLessEqual(chunks[-1].char_end, len(text))
        for previous, current in zip(chunks, chunks[1:]):
            self.assertLess(current.char_start, previous.char_end)

    def test_single_character_chunk_size_terminates(self):
        chunks = split_text_into_chunks("abcdef", chunk_size=1, overlap=99)

        self.assertEqual([chunk.text for chunk in chunks], ["a", "b", "c", "d", "e", "f"])


if __name__ == "__main__":
    unittest.main()
