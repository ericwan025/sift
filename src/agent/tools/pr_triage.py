"""PR triage: fetches open PRs, classifies each with the LLM (category,
priority, suggested reviewers, summary), and returns a priority-ranked list.
Classifications are cached on disk keyed by PR number + updated_at so a
re-run only pays the LLM/API cost for PRs that actually changed.
"""
from __future__ import annotations

import fnmatch
import json
from pathlib import Path

from langchain_core.tools import tool
from pydantic import BaseModel

from agent.backends import get_chat_model
from agent.config import Settings, get_settings
from agent.github_client import GitHubClient, PullRequestData
from agent.offline import heuristic_classify_pr
from agent.schemas import PRCategory, PRClassification, PRPriority, PRTriageResult

_HOTFIX_KEYWORDS = ("hotfix", "security", "critical", "urgent", "cve")
_LARGE_DIFF_LINES = 500
_SMALL_DIFF_LINES = 20


class _LLMClassification(BaseModel):
    category: PRCategory
    priority: PRPriority
    summary: str
    rationale: str


def _load_cache(cache_path: Path) -> dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache_path: Path, cache: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _cache_key(pr: PullRequestData) -> str:
    return f"{pr.number}:{pr.updated_at}"


def _heuristic_priority_hint(pr: PullRequestData) -> str:
    total_changes = pr.additions + pr.deletions
    text = f"{pr.title} {pr.body}".lower()
    if any(keyword in text for keyword in _HOTFIX_KEYWORDS):
        return "keyword match suggests high priority (hotfix/security/critical)"
    if total_changes > _LARGE_DIFF_LINES:
        return f"large diff (>{_LARGE_DIFF_LINES} lines changed) suggests higher priority/risk"
    if total_changes < _SMALL_DIFF_LINES:
        return f"small diff (<{_SMALL_DIFF_LINES} lines changed) suggests lower priority"
    return "no strong heuristic signal"


def _parse_codeowners(codeowners_text: str) -> list[tuple[str, list[str]]]:
    rules = []
    for line in codeowners_text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        rules.append((parts[0], parts[1:]))
    return rules


def _suggest_reviewers(pr: PullRequestData, codeowners_text: str | None) -> list[str]:
    if not codeowners_text:
        return []
    reviewers: set[str] = set()
    for pattern, owners in _parse_codeowners(codeowners_text):
        glob_pattern = pattern.lstrip("/")
        if glob_pattern.endswith("/"):
            glob_pattern += "**"
        for changed_file in pr.changed_files:
            if fnmatch.fnmatch(changed_file, glob_pattern) or changed_file.startswith(glob_pattern.rstrip("*")):
                reviewers.update(owner.lstrip("@") for owner in owners)
    return sorted(reviewers)


def _classify_pr(llm, pr: PullRequestData) -> _LLMClassification:
    structured_llm = llm.with_structured_output(_LLMClassification)
    prompt = (
        "Classify this pull request.\n\n"
        f"Title: {pr.title}\n"
        f"Body: {pr.body[:2000]}\n"
        f"Files changed ({len(pr.changed_files)}): {', '.join(pr.changed_files[:30])}\n"
        f"Additions: {pr.additions}, Deletions: {pr.deletions}\n"
        f"Heuristic signal: {_heuristic_priority_hint(pr)}\n\n"
        "category must be one of: bug-fix, feature, refactor, docs, test, chore, dependency.\n"
        "priority must be one of: high, medium, low -- weigh the heuristic signal alongside diff size and keywords.\n"
        "Provide a one-line summary and a short rationale explaining the category and priority choice."
    )
    return structured_llm.invoke(prompt)


def pr_triage(
    repo: str,
    open_only: bool = True,
    limit: int | None = None,
    client: GitHubClient | None = None,
    settings: Settings | None = None,
    llm=None,
) -> PRTriageResult:
    """Fetch open PRs for `repo` and classify each by category and priority.

    `llm` defaults to the real chat model when `OPENAI_API_KEY` is set; with
    no key configured, falls back to `heuristic_classify_pr` (keyword +
    diff-size rules only, no LLM call) -- see `agent.offline`.
    """
    settings = settings or get_settings()
    client = client or GitHubClient(settings)

    prs = client.fetch_open_pull_requests(repo, limit=limit)
    codeowners_text = client.fetch_codeowners(repo)

    cache = _load_cache(settings.pr_cache_path)
    llm = llm or get_chat_model(settings)

    triaged: list[PRClassification] = []
    for pr in prs:
        key = _cache_key(pr)
        if key in cache:
            classification = _LLMClassification(**cache[key])
        elif llm is not None:
            classification = _classify_pr(llm, pr)
            cache[key] = classification.model_dump()
        else:
            category, priority = heuristic_classify_pr(pr.title, pr.body, pr.additions, pr.deletions)
            classification = _LLMClassification(
                category=category,
                priority=priority,
                summary=pr.title,
                rationale=f"Offline heuristic classification (no OPENAI_API_KEY configured): {_heuristic_priority_hint(pr)}",
            )
            cache[key] = classification.model_dump()

        triaged.append(
            PRClassification(
                number=pr.number,
                title=pr.title,
                url=pr.url,
                category=classification.category,
                priority=classification.priority,
                suggested_reviewers=_suggest_reviewers(pr, codeowners_text),
                summary=classification.summary,
                rationale=classification.rationale,
            )
        )

    _save_cache(settings.pr_cache_path, cache)

    result = PRTriageResult(repo=repo, triaged=triaged)
    result.triaged = result.ranked()
    return result


@tool
def pr_triage_tool(repo: str, limit: int = 20) -> str:
    """Triage open pull requests for a GitHub repo (format: owner/repo).
    Classifies each PR by category and priority, suggests reviewers from
    CODEOWNERS, and ranks the list by priority. Use for any question about
    pull requests, code review backlog, or what to review next."""
    try:
        return pr_triage(repo, limit=limit).model_dump_json()
    except Exception as exc:  # noqa: BLE001 -- surfaced to the agent as a clean tool error
        return json.dumps({"error": f"pr_triage failed: {exc}", "repo": repo})
