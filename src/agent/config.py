"""Typed settings loaded from the environment / .env file.

Nothing in this codebase should hardcode a model name, path, or secret —
everything flows through a `Settings` instance built here.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    github_token: str = Field(default="", alias="GITHUB_TOKEN")

    agent_model: str = Field(default="gpt-4o-mini", alias="AGENT_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    faiss_index_path: Path = Field(default=Path("./.index/repo.faiss"), alias="FAISS_INDEX_PATH")
    retrieval_top_k: int = Field(default=6, alias="RETRIEVAL_TOP_K")
    cluster_distance_threshold: float = Field(default=0.35, alias="CLUSTER_DISTANCE_THRESHOLD")
    # The offline hashing embedding produces a different cosine-distance
    # distribution than text-embedding-3-small (sparser, higher-magnitude),
    # so it needs its own threshold rather than sharing the tuned-for-real-
    # embeddings default above. See agent.tools.issue_clustering.
    offline_cluster_distance_threshold: float = Field(default=0.8, alias="OFFLINE_CLUSTER_DISTANCE_THRESHOLD")

    pr_cache_path: Path = Field(default=Path("./.cache/pr_cache.json"), alias="PR_CACHE_PATH")
    max_agent_iterations: int = Field(default=6, alias="MAX_AGENT_ITERATIONS")

    @property
    def metadata_path(self) -> Path:
        """Sidecar metadata store next to the FAISS index."""
        return self.faiss_index_path.with_suffix(".meta.json")


@lru_cache
def get_settings() -> Settings:
    return Settings()
