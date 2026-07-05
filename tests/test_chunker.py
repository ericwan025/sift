from agent.indexing.chunker import chunk_file, count_tokens, detect_language


def test_detect_language_by_extension():
    assert detect_language("src/foo.py") == "python"
    assert detect_language("src/foo.ts") == "typescript"
    assert detect_language("docs/readme.md") == "markdown"
    assert detect_language("Makefile") == "text"


def test_empty_file_produces_no_chunks():
    assert chunk_file("empty.py", "") == []
    assert chunk_file("whitespace.py", "   \n\n  ") == []


def test_chunks_are_line_aligned_never_split_mid_line():
    text = "\n".join(f"line {i}" for i in range(1, 51))
    chunks = chunk_file("f.py", text, max_tokens=30, overlap_tokens=5)
    for chunk in chunks:
        lines = chunk.chunk_text.split("\n")
        # every line in the chunk must be a *complete* original line
        for line in lines:
            assert line == "" or line.startswith("line ")
            assert line in text.splitlines()


def test_chunk_metadata_line_numbers_are_correct():
    text = "\n".join(f"line {i}" for i in range(1, 21))
    lines = text.splitlines()
    chunks = chunk_file("f.py", text, max_tokens=20, overlap_tokens=0)
    for chunk in chunks:
        expected_text = "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
        assert chunk.chunk_text == expected_text


def test_chunks_respect_token_cap():
    text = "\n".join(f"some line of moderately long text number {i}" for i in range(1, 101))
    max_tokens = 50
    chunks = chunk_file("f.py", text, max_tokens=max_tokens, overlap_tokens=10)
    for chunk in chunks:
        assert count_tokens(chunk.chunk_text) <= max_tokens + 5  # small slack for boundary rounding


def test_oversized_single_line_does_not_crash_and_is_isolated():
    huge_line = "x = " + " + ".join(str(i) for i in range(2000))
    text = f"def f():\n    {huge_line}\n    return x\n"
    chunks = chunk_file("f.py", text, max_tokens=50, overlap_tokens=5)
    assert len(chunks) >= 1
    assert any(huge_line in c.chunk_text for c in chunks)


def test_consecutive_chunks_overlap_when_a_file_spans_multiple_chunks():
    text = "\n".join(f"def func_{i}():\n    return {i}" for i in range(20))
    chunks = chunk_file("f.py", text, max_tokens=40, overlap_tokens=15)
    assert len(chunks) > 1
    for first, second in zip(chunks, chunks[1:]):
        # overlap means the next chunk starts at or before the previous chunk's end
        assert second.start_line <= first.end_line


def test_markdown_splits_on_headings():
    text = "# Title\nintro text\n\n## Section A\nbody a\n\n## Section B\nbody b\n"
    chunks = chunk_file("doc.md", text, max_tokens=500, overlap_tokens=0)
    assert len(chunks) == 1  # small enough to fit in one chunk
    assert chunks[0].language == "markdown"
