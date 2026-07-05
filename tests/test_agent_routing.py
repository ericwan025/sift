"""Routing tests: for a set of representative queries, `run_agent()` should
select the expected tool. Runs fully offline via the rule-based fallback
router (`agent.offline.route_offline`) -- see the README's Evaluation
section for why the full LangGraph tool-calling loop needs a live
OPENAI_API_KEY and isn't exercised here.
"""
from __future__ import annotations

import pytest

from agent.agent import run_agent
from agent.config import Settings
from agent.indexing.build_index import build_index


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        OPENAI_API_KEY="",
        GITHUB_TOKEN="",
        FAISS_INDEX_PATH=tmp_path / "repo.faiss",
    )


@pytest.mark.parametrize(
    "query,expected_tool",
    [
        ("What is the status of open pull requests right now?", "pr_triage"),
        ("Summarize the pull request backlog by priority", "pr_triage"),
        ("Can you group the open issues into themes?", "issue_clustering"),
        ("Are there duplicate issues in the backlog?", "issue_clustering"),
    ],
)
def test_routes_to_expected_tool_without_a_repo(query, expected_tool, settings):
    result = run_agent(query, settings=settings)
    assert result["tool_calls"] == [expected_tool]


@pytest.mark.parametrize(
    "query",
    [
        "What's your favorite programming language?",
        "What is 2+2?",
    ],
)
def test_no_tool_selected_falls_back_conversationally(query, settings):
    result = run_agent(query, settings=settings)
    assert result["tool_calls"] == []
    assert "not grounded" in result["answer"].lower()


def test_doc_lookup_routing_with_a_real_tiny_index(tmp_path, settings):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "auth.py").write_text("def refresh_session():\n    return retry_with_backoff()\n")
    build_index(repo, settings)

    result = run_agent("How does the retry/backoff logic work in this codebase?", settings=settings)
    assert result["tool_calls"] == ["doc_lookup"]


def test_baseline_prompt_is_strictly_weaker_at_routing_than_tuned(settings):
    query = "How does the retry/backoff logic work in this codebase?"
    baseline = run_agent(query, settings=settings, prompt_version="baseline")
    tuned = run_agent(query, settings=settings, prompt_version="tuned")
    assert baseline["tool_calls"] == []
    assert tuned["tool_calls"] == ["doc_lookup"]
