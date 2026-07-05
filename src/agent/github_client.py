"""Thin wrapper around PyGithub: auth, rate-limit awareness, and the narrow
read-only surface the tools actually need (open PRs, open issues, CODEOWNERS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from github import Github, Auth
from github.GithubException import RateLimitExceededException

from agent.config import Settings, get_settings


@dataclass
class PullRequestData:
    number: int
    title: str
    body: str
    author: str
    additions: int
    deletions: int
    changed_files: list[str]
    updated_at: str
    url: str


@dataclass
class IssueData:
    number: int
    title: str
    body: str
    labels: list[str] = field(default_factory=list)
    url: str = ""


class GitHubClient:
    """Read-only accessor for the repo data the agent's tools consume."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        auth = Auth.Token(self._settings.github_token) if self._settings.github_token else None
        self._client = Github(auth=auth)

    def fetch_open_pull_requests(self, repo_name: str, limit: int | None = None) -> list[PullRequestData]:
        repo = self._client.get_repo(repo_name)
        prs: list[PullRequestData] = []
        try:
            for pr in repo.get_pulls(state="open"):
                prs.append(
                    PullRequestData(
                        number=pr.number,
                        title=pr.title,
                        body=pr.body or "",
                        author=pr.user.login if pr.user else "unknown",
                        additions=pr.additions,
                        deletions=pr.deletions,
                        changed_files=[f.filename for f in pr.get_files()],
                        updated_at=pr.updated_at.isoformat() if pr.updated_at else "",
                        url=pr.html_url,
                    )
                )
                if limit is not None and len(prs) >= limit:
                    break
        except RateLimitExceededException as exc:
            raise RuntimeError("GitHub API rate limit exceeded while fetching pull requests") from exc
        return prs

    def fetch_open_issues(
        self, repo_name: str, labels: list[str] | None = None, limit: int | None = None
    ) -> list[IssueData]:
        repo = self._client.get_repo(repo_name)
        issues: list[IssueData] = []
        try:
            kwargs = {"state": "open"}
            if labels:
                kwargs["labels"] = labels
            for issue in repo.get_issues(**kwargs):
                if issue.pull_request is not None:
                    continue  # PRs show up in the issues endpoint; skip them
                issues.append(
                    IssueData(
                        number=issue.number,
                        title=issue.title,
                        body=issue.body or "",
                        labels=[label.name for label in issue.labels],
                        url=issue.html_url,
                    )
                )
                if limit is not None and len(issues) >= limit:
                    break
        except RateLimitExceededException as exc:
            raise RuntimeError("GitHub API rate limit exceeded while fetching issues") from exc
        return issues

    def fetch_codeowners(self, repo_name: str) -> str | None:
        repo = self._client.get_repo(repo_name)
        for path in (".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS"):
            try:
                content = repo.get_contents(path)
                return content.decoded_content.decode("utf-8")
            except Exception:
                continue
        return None
