"""Per-task-type scoring logic for the eval harness.

Every scorer takes the task dict (from `tasks.yaml`) and the actual result,
and returns a `ScoreResult`. doc_lookup/routing/multi_step tasks score
against `run_agent()`'s uniform `{tool_calls, tool_results}` shape (see
`agent.agent.run_agent`); pr_triage/issue_clustering tasks score against the
tool's own pydantic result directly.
"""
from __future__ import annotations

from dataclasses import dataclass

from agent.schemas import IssueClusteringResult, PRTriageResult


@dataclass
class ScoreResult:
    passed: bool
    reason: str


def score_doc_lookup(task: dict, actual: dict) -> ScoreResult:
    expected = task["expected"]
    tool_calls = actual.get("tool_calls", [])

    must_route_to = expected.get("must_route_to")
    if must_route_to and must_route_to not in tool_calls:
        return ScoreResult(False, f"expected routing to '{must_route_to}', got {tool_calls}")

    doc_result = actual.get("tool_results", {}).get("doc_lookup", {})

    must_be_grounded = expected.get("must_be_grounded")
    if must_be_grounded is not None and doc_result.get("grounded", False) != must_be_grounded:
        return ScoreResult(False, f"expected grounded={must_be_grounded}, got {doc_result.get('grounded')}")

    must_cite_file = expected.get("must_cite_file")
    if must_cite_file:
        sources = doc_result.get("sources", [])
        if not any(must_cite_file in s.get("file_path", "") for s in sources):
            return ScoreResult(False, f"expected a citation containing '{must_cite_file}', got {sources}")

    return ScoreResult(True, "ok")


def score_routing(task: dict, actual: dict) -> ScoreResult:
    expected = task["expected"]
    tool_calls = actual.get("tool_calls", [])
    must_route_to = expected.get("must_route_to")

    if must_route_to is None:
        if tool_calls:
            return ScoreResult(False, f"expected no tool call (conversational fallback), got {tool_calls}")
        return ScoreResult(True, "ok")

    if must_route_to not in tool_calls:
        return ScoreResult(False, f"expected routing to '{must_route_to}', got {tool_calls}")
    return ScoreResult(True, "ok")


def score_multi_step(task: dict, actual: dict) -> ScoreResult:
    expected = task["expected"]
    must_call_tools = expected.get("must_call_tools", [])
    tool_calls = actual.get("tool_calls", [])
    missing = [t for t in must_call_tools if t not in tool_calls]
    if missing:
        return ScoreResult(False, f"missing expected tool call(s) {missing}, got {tool_calls}")
    return ScoreResult(True, "ok")


def score_pr_triage(task: dict, actual: PRTriageResult) -> ScoreResult:
    expected = task["expected"]
    pr_number = expected["pr_number"]
    match = next((p for p in actual.triaged if p.number == pr_number), None)
    if match is None:
        return ScoreResult(False, f"PR #{pr_number} not present in triage result")

    if "category" in expected and match.category.value != expected["category"]:
        return ScoreResult(False, f"expected category='{expected['category']}', got '{match.category.value}'")

    if "priority" in expected and match.priority.value != expected["priority"]:
        return ScoreResult(False, f"expected priority='{expected['priority']}', got '{match.priority.value}'")

    return ScoreResult(True, "ok")


def score_issue_clustering(task: dict, actual: IssueClusteringResult) -> ScoreResult:
    expected = task["expected"]

    if "singleton" in expected:
        number = expected["singleton"]
        cluster = next((c for c in actual.clusters if number in c.issue_numbers), None)
        if cluster is None:
            return ScoreResult(False, f"issue #{number} not present in any cluster")
        if cluster.size != 1:
            return ScoreResult(False, f"expected issue #{number} to be a singleton, got cluster size {cluster.size}")
        return ScoreResult(True, "ok")

    issue_numbers = expected["same_cluster"]

    owning_clusters = []
    for number in issue_numbers:
        cluster = next((c for c in actual.clusters if number in c.issue_numbers), None)
        if cluster is None:
            return ScoreResult(False, f"issue #{number} not present in any cluster")
        owning_clusters.append(id(cluster))

    if len(set(owning_clusters)) != 1:
        return ScoreResult(False, f"issues {issue_numbers} were split across different clusters")

    return ScoreResult(True, "ok")


SCORERS = {
    "doc_lookup": score_doc_lookup,
    "routing": score_routing,
    "multi_step": score_multi_step,
    "pr_triage": score_pr_triage,
    "issue_clustering": score_issue_clustering,
}
