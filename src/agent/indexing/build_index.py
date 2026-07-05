"""Walks a local repo checkout, chunks every source file, embeds the chunks,
and writes a FAISS index + parallel metadata store to disk.

Invoked via `sift index <repo_path>`.
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import faiss
import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential

from agent.config import Settings, get_settings
from agent.indexing.chunker import Chunk, chunk_file

INCLUDED_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".rst", ".yaml", ".yml", ".toml", ".cfg", ".ini"}
EXCLUDED_DIRS = {".git", "node_modules", "venv", ".venv", "__pycache__", "dist", "build", ".index", ".cache"}

EMBEDDING_BATCH_SIZE = 100


def _load_gitignore_patterns(repo_path: Path) -> list[str]:
    gitignore = repo_path / ".gitignore"
    if not gitignore.exists():
        return []
    patterns = []
    for line in gitignore.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(line)
    return patterns


def _is_ignored(relative_path: Path, patterns: list[str]) -> bool:
    posix = relative_path.as_posix()
    for pattern in patterns:
        pattern = pattern.rstrip("/")
        if (
            fnmatch.fnmatch(posix, pattern)
            or fnmatch.fnmatch(posix, f"*/{pattern}")
            or fnmatch.fnmatch(relative_path.name, pattern)
        ):
            return True
    return False


def discover_files(repo_path: Path) -> list[Path]:
    """Walk repo_path, returning the source files worth indexing."""
    patterns = _load_gitignore_patterns(repo_path)
    files = []
    for path in repo_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in INCLUDED_EXTENSIONS:
            continue
        relative = path.relative_to(repo_path)
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        if _is_ignored(relative, patterns):
            continue
        files.append(path)
    return sorted(files)


def chunk_repo(repo_path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in discover_files(repo_path):
        relative = path.relative_to(repo_path).as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        chunks.extend(chunk_file(relative, text))
    return chunks


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=30))
def _embed_batch(embeddings_client, texts: list[str]) -> list[list[float]]:
    return embeddings_client.embed_documents(texts)


def embed_chunks(chunks: list[Chunk], settings: Settings) -> np.ndarray:
    from langchain_openai import OpenAIEmbeddings

    client = OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = [c.chunk_text for c in chunks[i : i + EMBEDDING_BATCH_SIZE]]
        vectors.extend(_embed_batch(client, batch))
    return np.array(vectors, dtype="float32")


def build_index(repo_path: str | Path, settings: Settings | None = None) -> dict:
    """Build and persist the FAISS index for `repo_path`. Returns a summary dict."""
    settings = settings or get_settings()
    repo_path = Path(repo_path).resolve()

    files = discover_files(repo_path)
    chunks = chunk_repo(repo_path)
    if not chunks:
        raise ValueError(f"No indexable files found under {repo_path}")

    vectors = embed_chunks(chunks, settings)
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)

    settings.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(settings.faiss_index_path))

    metadata = [
        {
            "file_path": c.file_path,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "language": c.language,
            "chunk_text": c.chunk_text,
        }
        for c in chunks
    ]
    settings.metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "files_indexed": len(files),
        "chunks_created": len(chunks),
        "index_path": str(settings.faiss_index_path),
        "index_size_bytes": settings.faiss_index_path.stat().st_size,
    }
