"""Issue clustering: embeds open issues and groups them with agglomerative
clustering (no preset cluster count -- a cosine-distance threshold decides
where to cut), then asks the LLM for a short theme label per cluster. This
surfaces duplicate/related issues without requiring a fixed number of groups.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from langchain_core.tools import tool
from pydantic import BaseModel
from sklearn.cluster import AgglomerativeClustering

from agent.backends import get_chat_model, get_embeddings_client
from agent.config import Settings, get_settings
from agent.github_client import GitHubClient, IssueData
from agent.offline import OfflineEmbeddings, heuristic_label_cluster
from agent.schemas import IssueCluster, IssueClusteringResult

MAX_BODY_CHARS = 500


class _LLMLabel(BaseModel):
    label: str


def _issue_text(issue: IssueData) -> str:
    return f"{issue.title}\n{issue.body[:MAX_BODY_CHARS]}"


def cluster_vectors(vectors: np.ndarray, distance_threshold: float) -> np.ndarray:
    """Assign a cluster id to each row of `vectors`. Public for direct testing."""
    if len(vectors) == 1:
        return np.array([0])
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold, metric="cosine", linkage="average"
    )
    return clustering.fit_predict(vectors)


def _label_cluster(llm, titles: list[str]) -> str:
    structured_llm = llm.with_structured_output(_LLMLabel)
    prompt = (
        "These issue titles were grouped together because they're semantically "
        "similar. Give a short (2-5 word) theme label describing what they have "
        "in common.\n\n" + "\n".join(f"- {t}" for t in titles)
    )
    return structured_llm.invoke(prompt).label


def issue_clustering(
    repo: str,
    labels: list[str] | None = None,
    limit: int | None = None,
    client: GitHubClient | None = None,
    settings: Settings | None = None,
    llm=None,
    embeddings=None,
) -> IssueClusteringResult:
    """Fetch open issues for `repo` and cluster them by semantic similarity.

    `embeddings`/`llm` default to the real OpenAI-backed clients when
    `OPENAI_API_KEY` is set; with no key configured, embeddings fall back to
    a hashing vectorizer and cluster labels fall back to
    `heuristic_label_cluster` -- see `agent.offline`.
    """
    settings = settings or get_settings()
    client = client or GitHubClient(settings)

    issues = client.fetch_open_issues(repo, labels=labels, limit=limit)
    if not issues:
        return IssueClusteringResult(repo=repo, clusters=[])

    embeddings = embeddings or get_embeddings_client(settings)
    vectors = np.array(embeddings.embed_documents([_issue_text(i) for i in issues]))

    # The offline hashing embedding has a different cosine-distance profile
    # than a real learned embedding, so it needs its own threshold.
    threshold = (
        settings.offline_cluster_distance_threshold
        if isinstance(embeddings, OfflineEmbeddings)
        else settings.cluster_distance_threshold
    )
    assignments = cluster_vectors(vectors, threshold)

    groups: dict[int, list[IssueData]] = defaultdict(list)
    for issue, cluster_id in zip(issues, assignments):
        groups[int(cluster_id)].append(issue)

    llm = llm or get_chat_model(settings)
    clusters = []
    for members in groups.values():
        titles = [m.title for m in members]
        label = _label_cluster(llm, titles) if llm is not None else heuristic_label_cluster(titles)
        representative = min(members, key=lambda m: m.number)
        clusters.append(
            IssueCluster(
                label=label,
                issue_numbers=sorted(m.number for m in members),
                size=len(members),
                representative_issue=representative.number,
            )
        )

    result = IssueClusteringResult(repo=repo, clusters=clusters)
    result.clusters = result.sorted_by_size()
    return result


@tool
def issue_clustering_tool(repo: str, limit: int = 50) -> str:
    """Group open GitHub issues for a repo (format: owner/repo) by semantic
    similarity to surface duplicates and related issues. Use for questions
    about grouping issues, finding duplicates, or themes in the backlog."""
    try:
        return issue_clustering(repo, limit=limit).model_dump_json()
    except Exception as exc:  # noqa: BLE001 -- surfaced to the agent as a clean tool error
        return json.dumps({"error": f"issue_clustering failed: {exc}", "repo": repo})
