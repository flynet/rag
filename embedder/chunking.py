# chunking.py
"""Structure-aware text chunking for the indexer.

The splitter prefers paragraph breaks, then sentence ends, then whitespace,
and only falls back to a hard character cut when none of those exist in the
window. Markdown ATX headings (outside fenced code blocks) are tracked so every
chunk carries a section path and chunks do not cross section boundaries when a
heading is near the window edge.

Character offsets refer to the normalized text (CRLF and CR converted to LF).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

# Bump when the chunking algorithm changes: the value is stored in the file
# signature, so existing documents are re-indexed after an upgrade.
CHUNKER_VERSION = 2

_HEADING_LINE_RE = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n")
_SENTENCE_END_RE = re.compile(r"""[.!?…](?:["'”’)\]]*)\s""")
_WHITESPACE_RE = re.compile(r"\s")


@dataclass(frozen=True)
class TextChunk:
    """A chunk plus the metadata needed for citations and filtering."""

    text: str
    section: str
    char_start: int
    char_end: int

    def embedding_text(self, title: str) -> str:
        """Text sent to the embedding model (document title prepended)."""
        if not title:
            return self.text
        # Do not duplicate a title that already starts the chunk.
        head = self.text.lstrip("# \t")
        if self.text.startswith(title) or head.startswith(title):
            return self.text
        return f"{title}\n\n{self.text}"


Heading = Tuple[int, int, str]  # (char offset, level, title)


def _headings(text: str) -> List[Heading]:
    """Parse ATX headings, ignoring lines inside fenced code blocks."""
    headings: List[Heading] = []
    in_fence = False
    offset = 0

    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
        elif not in_fence:
            match = _HEADING_LINE_RE.match(line.rstrip("\r\n"))
            if match:
                headings.append(
                    (offset, len(match.group(1)), match.group(2).strip())
                )
        offset += len(line)

    return headings


def document_title(text: str, path: Optional[Path] = None) -> str:
    """Best-effort document title: first heading, first line, or file stem."""
    headings = _headings(text)
    if headings:
        return headings[0][2]

    for line in text.splitlines():
        value = line.strip()
        if value:
            return re.sub(r"^#{1,6}\s+", "", value).strip()[:120]

    return path.stem if path is not None else ""


def _section_at(headings: List[Heading], position: int) -> str:
    """Section path at a character offset, e.g. "Guide > Installation"."""
    stack: List[Tuple[int, str]] = []
    for heading_start, level, title in headings:
        if heading_start > position:
            break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return " > ".join(title for _, title in stack)


def _trim_end(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1].isspace():
        end -= 1
    return end


def _first_heading_start(
    headings: Sequence[Heading], start: int, limit: int
) -> Optional[int]:
    for heading_start, _level, _title in headings:
        if heading_start <= start:
            continue
        if heading_start > limit:
            break
        return heading_start
    return None


def _last_paragraph_break(text: str, min_end: int, limit: int) -> Optional[int]:
    index = text.rfind("\n\n", min_end, limit)
    if index != -1:
        return index + 2

    matches = list(_BLANK_LINE_RE.finditer(text, min_end, limit))
    if matches:
        return matches[-1].end()
    return None


def _last_sentence_end(text: str, min_end: int, limit: int) -> Optional[int]:
    found: Optional[int] = None
    for match in _SENTENCE_END_RE.finditer(text, min_end, limit):
        found = match.end()
    return found


def _find_chunk_end(
    text: str, start: int, limit: int, headings: Sequence[Heading]
) -> Tuple[int, Optional[int]]:
    """Return ``(end, next_heading_start)`` for the chunk starting at `start`.

    A heading is the strongest boundary: chunks never include a heading that
    starts later in the document, so section metadata stays correct. Otherwise
    prefer a paragraph break, then a sentence end, then whitespace.
    """
    heading_start = _first_heading_start(headings, start, limit)
    if heading_start is not None:
        return heading_start, heading_start

    if limit >= len(text):
        return len(text), None

    min_end = start + max(1, (limit - start) // 2)

    paragraph_end = _last_paragraph_break(text, min_end, limit)
    if paragraph_end is not None:
        return paragraph_end, None

    sentence_end = _last_sentence_end(text, min_end, limit)
    if sentence_end is not None:
        return sentence_end, None

    whitespace = max(
        text.rfind(" ", min_end, limit),
        text.rfind("\n", min_end, limit),
        text.rfind("\t", min_end, limit),
    )
    if whitespace > start:
        return whitespace, None

    return limit, None


def _next_start(text: str, start: int, end: int, overlap: int) -> int:
    """Start position of the next chunk, with overlap aligned to a word start."""
    if overlap <= 0:
        return end

    candidate = max(start + 1, end - overlap)
    if candidate >= end:
        return end

    # Do not start inside a word: move forward to the next whitespace.
    if not text[candidate - 1].isspace() and not text[candidate].isspace():
        whitespace = _WHITESPACE_RE.search(text, candidate, end)
        candidate = whitespace.end() if whitespace else end

    while candidate < end and text[candidate].isspace():
        candidate += 1

    return candidate if candidate > start else start + 1


def split_text_into_chunks(
    text: str,
    chunk_size: int = 1024,
    overlap: int = 128,
    path: Optional[Path] = None,
    title: Optional[str] = None,
) -> List[TextChunk]:
    """Split text into chunks of at most `chunk_size` characters.

    Chunks prefer paragraph boundaries, then sentence ends, then whitespace.
    Consecutive chunks within a section overlap by up to `overlap` characters.
    """
    if not text or not text.strip():
        return []

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    chunk_size = max(1, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size - 1))
    headings = _headings(normalized)
    if title is None:
        title = document_title(normalized, path)

    chunks: List[TextChunk] = []
    length = len(normalized)
    position = 0

    while position < length:
        start = position
        while start < length and normalized[start].isspace():
            start += 1
        if start >= length:
            break

        limit = min(length, start + chunk_size)
        end, next_heading = _find_chunk_end(normalized, start, limit, headings)
        end = _trim_end(normalized, start, end)
        if end <= start:
            end = _trim_end(normalized, start, min(length, start + chunk_size))
        if end <= start:
            end = start + 1

        chunk_text = normalized[start:end]
        if chunk_text:
            section = _section_at(headings, start) or title
            chunks.append(
                TextChunk(
                    text=chunk_text,
                    section=section,
                    char_start=start,
                    char_end=end,
                )
            )

        if end >= length:
            break
        if not normalized[end:].strip():
            break

        if next_heading is not None and next_heading > end:
            position = next_heading
            continue

        next_position = _next_start(normalized, start, end, overlap)
        position = next_position if next_position > start else start + 1

    return chunks
