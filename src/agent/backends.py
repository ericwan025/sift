"""Picks the real OpenAI-backed client or the offline fallback for embeddings
and chat, based on whether `OPENAI_API_KEY` is configured.

Every tool takes its embeddings/chat client as an optional constructor
argument (for test injection); when the caller doesn't supply one, it comes
from here. This is the single place that decides "online vs offline" so the
rest of the codebase doesn't need `if settings.openai_api_key` scattered
through it.
"""
from __future__ import annotations

from agent.config import Settings
from agent.offline import OfflineEmbeddings


def has_openai_key(settings: Settings) -> bool:
    return bool(settings.openai_api_key)


def get_embeddings_client(settings: Settings):
    """Real `text-embedding-3-small` client, or the offline hashing fallback."""
    if has_openai_key(settings):
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
    return OfflineEmbeddings()


def get_chat_model(settings: Settings):
    """Real `ChatOpenAI` client, or `None` if no key is configured.

    There is no offline stand-in for a general-purpose structured-output
    chat model -- callers that get `None` back should fall back to the
    rule-based helpers in `agent.offline` instead.
    """
    if not has_openai_key(settings):
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=settings.agent_model, api_key=settings.openai_api_key, temperature=0)
