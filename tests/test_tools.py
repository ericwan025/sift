"""Tool tests -- everything is mocked (GitHub API, FAISS retriever, LLM),
so this file runs fully offline per the build spec.
"""
from __future__ import annotations

import numpy as np
import pytest

from agent.config import Settings
from agent.indexing.loader import RetrievedChunk
from agent.schemas import PRCategory, PRPriority
from agent.tools.doc_lookup import doc_lookup
from agent.tools.issue_clustering import cluster_vectors, issue_clustering
from agent.tools.pr_triage import pr_triage
from eval.fixtures import FakeGitHubClient, FIXTURE_REPO


@pytest.fixture
def settings(tmp_path) -> Settings:
    # Give every test its own PR cache file -- pr_triage() caches classifications
    # on disk keyed by PR number, and the default path is shared/persistent.
    return Settings(_env_file=None, OPENAI_API_KEY="", GITHUB_TOKEN="", PR_CACHE_PATH=tmp_path / "pr_cache.json")


class FakeRetriever:
    def __init__(self, chunks: list[RetrievedChunk]):
        self._chunks = chunks

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        return self._chunks


class FakeStructuredLLM:
    """Stand-in for `ChatOpenAI(...).with_structured_output(Schema).invoke(prompt)`."""

    def __init__(self, response):
        self._response = response

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        return self._response


# --- doc_lookup --------------------------------------------------------


def test_doc_lookup_returns_grounded_answer_with_citations(settings):
    chunks = [RetrievedChunk("src/loader.py", 10, 20, "python", "def load(): ...", score=0.1)]
    retriever = FakeRetriever(chunks)

    from agent.tools.doc_lookup import _LLMAnswer

    llm = FakeStructuredLLM(_LLMAnswer(answer="It's loaded in load().", grounded=True))

    result = doc_lookup("Where is the index loaded?", retriever=retriever, settings=settings, llm=llm)

    assert result.grounded is True
    assert result.answer == "It's loaded in load()."
    assert len(result.sources) == 1
    assert result.sources[0].file_path == "src/loader.py"
    assert (result.sources[0].start_line, result.sources[0].end_line) == (10, 20)


def test_doc_lookup_reports_not_grounded_when_llm_says_no_answer(settings):
    chunks = [RetrievedChunk("src/unrelated.py", 1, 5, "python", "x = 1", score=0.9)]
    retriever = FakeRetriever(chunks)

    from agent.tools.doc_lookup import NOT_FOUND_SENTENCE, _LLMAnswer

    llm = FakeStructuredLLM(_LLMAnswer(answer=NOT_FOUND_SENTENCE, grounded=False))

    result = doc_lookup("How does billing work?", retriever=retriever, settings=settings, llm=llm)

    assert result.grounded is False
    assert result.sources == []


def test_doc_lookup_handles_missing_index_cleanly(settings, tmp_path):
    # No retriever injected and no index built at settings.faiss_index_path,
    # so doc_lookup() must catch IndexNotBuiltError itself rather than raising.
    settings.faiss_index_path = tmp_path / "missing.faiss"
    result = doc_lookup("anything", settings=settings)
    assert result.grounded is False
    assert "index" in result.answer.lower()


def test_doc_lookup_offline_fallback_without_llm(settings):
    chunks = [RetrievedChunk("src/loader.py", 1, 3, "python", "def load(): pass", score=0.05)]
    retriever = FakeRetriever(chunks)

    result = doc_lookup("Where is load defined?", retriever=retriever, settings=settings)

    assert result.grounded is True
    assert result.answer.strip() == "def load(): pass"
    assert result.sources[0].file_path == "src/loader.py"


# --- pr_triage -----------------------------------------------------------


def test_pr_triage_ranks_by_priority(settings):
    result = pr_triage(FIXTURE_REPO, client=FakeGitHubClient(), settings=settings)
    priorities = [p.priority for p in result.triaged]
    order = {PRPriority.HIGH: 0, PRPriority.MEDIUM: 1, PRPriority.LOW: 2}
    assert priorities == sorted(priorities, key=lambda p: order[p])


def test_pr_triage_classifies_known_fixture_pr(settings):
    result = pr_triage(FIXTURE_REPO, client=FakeGitHubClient(), settings=settings)
    hotfix = next(p for p in result.triaged if p.number == 108)
    assert hotfix.category == PRCategory.BUG_FIX
    assert hotfix.priority == PRPriority.HIGH


def test_pr_triage_suggests_reviewers_from_codeowners(settings):
    result = pr_triage(FIXTURE_REPO, client=FakeGitHubClient(), settings=settings)
    auth_pr = next(p for p in result.triaged if p.number == 101)
    assert "auth-team" in auth_pr.suggested_reviewers


def test_pr_triage_uses_injected_llm_when_provided(settings):
    from agent.tools.pr_triage import _LLMClassification

    llm_response = _LLMClassification(
        category=PRCategory.FEATURE, priority=PRPriority.HIGH, summary="s", rationale="r"
    )
    result = pr_triage(
        FIXTURE_REPO, client=FakeGitHubClient(), settings=settings, llm=FakeStructuredLLM(llm_response)
    )
    assert all(p.category == PRCategory.FEATURE for p in result.triaged)


# --- issue_clustering ------------------------------------------------------


def test_cluster_vectors_groups_similar_points():
    vectors = np.array([[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]])
    assignments = cluster_vectors(vectors, distance_threshold=0.1)
    assert assignments[0] == assignments[1]
    assert assignments[2] != assignments[0]


def test_issue_clustering_groups_known_duplicates(settings):
    result = issue_clustering(FIXTURE_REPO, client=FakeGitHubClient(), settings=settings)
    login_cluster = next(c for c in result.clusters if 201 in c.issue_numbers)
    assert 202 in login_cluster.issue_numbers
    assert 203 in login_cluster.issue_numbers


def test_issue_clustering_sorted_by_size_descending(settings):
    result = issue_clustering(FIXTURE_REPO, client=FakeGitHubClient(), settings=settings)
    sizes = [c.size for c in result.clusters]
    assert sizes == sorted(sizes, reverse=True)


def test_issue_clustering_empty_when_no_issues(settings):
    result = issue_clustering(FIXTURE_REPO, client=FakeGitHubClient(issues=[]), settings=settings)
    assert result.clusters == []
