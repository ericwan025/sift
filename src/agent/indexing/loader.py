"""Loads a previously-built FAISS index + metadata store and exposes the
similarity-search interface the doc_lookup tool retrieves from.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import faiss
import numpy as np

from agent.config import Settings, get_settings


@dataclass(frozen=True)
class RetrievedChunk:
    file_path: str
    start_line: int
    end_line: int
    language: str
    chunk_text: str
    score: float  # L2 distance; lower means more similar


class IndexNotBuiltError(RuntimeError):
    """Raised when `sift ask`/`doc_lookup` is used before `sift index` has run."""


class FaissRetriever:
    """Wraps a FAISS index + sidecar metadata; embeds queries on demand.

    `search_vector` takes a pre-computed embedding directly, which keeps the
    embedding client out of the hot path for tests — they can exercise FAISS
    lookup + metadata mapping without any network calls.
    """

    def __init__(self, settings: Settings | None = None):
        self._settings = settings or get_settings()
        if not self._settings.faiss_index_path.exists() or not self._settings.metadata_path.exists():
            raise IndexNotBuiltError(
                f"No index found at {self._settings.faiss_index_path}. Run `sift index <repo_path>` first."
            )
        self._index = faiss.read_index(str(self._settings.faiss_index_path))
        self._metadata = json.loads(self._settings.metadata_path.read_text(encoding="utf-8"))
        self._embeddings = None  # lazily constructed to avoid network calls in offline paths

    def _embeddings_client(self):
        if self._embeddings is None:
            from langchain_openai import OpenAIEmbeddings

            self._embeddings = OpenAIEmbeddings(
                model=self._settings.embedding_model, api_key=self._settings.openai_api_key
            )
        return self._embeddings

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self._settings.retrieval_top_k
        query_vector = np.array([self._embeddings_client().embed_query(query)], dtype="float32")
        return self.search_vector(query_vector, top_k)

    def search_vector(self, query_vector: np.ndarray, top_k: int) -> list[RetrievedChunk]:
        distances, indices = self._index.search(query_vector, top_k)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            meta = self._metadata[idx]
            results.append(
                RetrievedChunk(
                    file_path=meta["file_path"],
                    start_line=meta["start_line"],
                    end_line=meta["end_line"],
                    language=meta["language"],
                    chunk_text=meta["chunk_text"],
                    score=float(score),
                )
            )
        return results
