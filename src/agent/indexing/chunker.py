"""Code-aware chunking.

Splits a file into logical blocks (function/class defs for code, headings for
markdown) and greedily packs those blocks into token-bounded chunks with a
small line overlap between consecutive chunks. The atomic unit is always a
whole source line — a chunk boundary never falls inside a line.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import tiktoken

_ENCODING = tiktoken.get_encoding("cl100k_base")

_CODE_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
}
_DOC_EXTENSIONS = {".md": "markdown", ".rst": "rst"}

_CODE_BOUNDARY_RE = re.compile(
    r"^\s*(def |class |async def |function |export (default )?function|export class|export const .*=.*\(.*\)\s*=>)"
)
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s")

DEFAULT_MAX_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 50

NumberedLine = tuple[int, str]  # (1-indexed line number, line text)


@dataclass(frozen=True)
class Chunk:
    file_path: str
    start_line: int  # 1-indexed, inclusive
    end_line: int  # 1-indexed, inclusive
    language: str
    chunk_text: str


def detect_language(file_path: str) -> str:
    for ext, lang in {**_CODE_EXTENSIONS, **_DOC_EXTENSIONS}.items():
        if file_path.endswith(ext):
            return lang
    return "text"


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _is_boundary(line: str, language: str) -> bool:
    if language == "markdown":
        return bool(_MARKDOWN_HEADING_RE.match(line))
    if language == "rst":
        return line.startswith(("=", "-", "~")) and len(line.strip()) > 0
    if language in ("python", "javascript", "typescript"):
        return bool(_CODE_BOUNDARY_RE.match(line))
    return False


def _split_into_blocks(lines: list[str], language: str) -> list[tuple[int, int]]:
    """Return 0-indexed half-open [start, end) ranges for logical blocks."""
    boundaries = [i for i, line in enumerate(lines) if i > 0 and _is_boundary(line, language)]
    boundaries = [0] + boundaries + [len(lines)]
    return [(s, e) for s, e in zip(boundaries, boundaries[1:]) if e > s]


def _tail_overlap(numbered: list[NumberedLine], overlap_tokens: int) -> list[NumberedLine]:
    """Trailing lines of `numbered` totalling at most `overlap_tokens`."""
    tail: list[NumberedLine] = []
    tokens = 0
    for item in reversed(numbered):
        t = count_tokens(item[1]) + 1
        if tail and tokens + t > overlap_tokens:
            break
        tail.append(item)
        tokens += t
    tail.reverse()
    return tail


def _pack_lines(numbered: list[NumberedLine], max_tokens: int, overlap_tokens: int) -> list[list[NumberedLine]]:
    """Greedily pack individual lines into token-bounded groups.

    Used when a single logical block is itself larger than `max_tokens`.
    """
    groups: list[list[NumberedLine]] = []
    current: list[NumberedLine] = []
    current_tokens = 0
    for line_no, text in numbered:
        line_tokens = count_tokens(text) + 1
        if current and current_tokens + line_tokens > max_tokens:
            groups.append(current)
            current = _tail_overlap(current, overlap_tokens)
            current_tokens = sum(count_tokens(t) + 1 for _, t in current)
        current.append((line_no, text))
        current_tokens += line_tokens
    if current:
        groups.append(current)
    return groups


def chunk_file(
    file_path: str,
    text: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk a file's contents into token-bounded, line-aligned chunks."""
    if not text.strip():
        return []

    language = detect_language(file_path)
    lines = text.splitlines()
    numbered: list[NumberedLine] = list(enumerate(lines, start=1))
    blocks = _split_into_blocks(lines, language)

    chunks: list[Chunk] = []
    pending: list[NumberedLine] = []
    pending_tokens = 0

    def flush_pending() -> None:
        if not pending:
            return
        chunks.append(
            Chunk(
                file_path=file_path,
                start_line=pending[0][0],
                end_line=pending[-1][0],
                language=language,
                chunk_text="\n".join(t for _, t in pending),
            )
        )

    for block_start, block_end in blocks:
        block_numbered = numbered[block_start:block_end]
        block_text = "\n".join(t for _, t in block_numbered)
        block_tokens = count_tokens(block_text)

        if block_tokens > max_tokens:
            flush_pending()
            pending, pending_tokens = [], 0
            for group in _pack_lines(block_numbered, max_tokens, overlap_tokens):
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        start_line=group[0][0],
                        end_line=group[-1][0],
                        language=language,
                        chunk_text="\n".join(t for _, t in group),
                    )
                )
            continue

        if pending and pending_tokens + block_tokens > max_tokens:
            flush_pending()
            pending = _tail_overlap(pending, overlap_tokens)
            pending_tokens = sum(count_tokens(t) + 1 for _, t in pending)

        pending.extend(block_numbered)
        pending_tokens += block_tokens

    flush_pending()
    return chunks
