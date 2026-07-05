"""Fixture GitHub data + a fake client, so pr_triage/issue_clustering eval
tasks (and tests) score against known data without hitting the real GitHub
API. Reused by `tests/test_tools.py` too.
"""
from __future__ import annotations

from agent.github_client import IssueData, PullRequestData

FIXTURE_REPO = "acme/fixture-repo"

FIXTURE_CODEOWNERS = """\
/src/auth/ @auth-team
/src/billing/ @billing-team
*.md @docs-team
"""

FIXTURE_PRS = [
    PullRequestData(
        number=101,
        title="Fix null pointer crash in session refresh",
        body="Fixes a crash when the session token expires mid-request.",
        author="alice",
        additions=12,
        deletions=4,
        changed_files=["src/auth/session.py"],
        updated_at="2026-06-01T00:00:00",
        url="https://example.com/pr/101",
    ),
    PullRequestData(
        number=102,
        title="Add dark mode toggle to settings page",
        body="Implements a new dark mode preference and persists it per-user.",
        author="bob",
        additions=340,
        deletions=20,
        changed_files=["src/ui/settings.py", "src/ui/theme.py"],
        updated_at="2026-06-02T00:00:00",
        url="https://example.com/pr/102",
    ),
    PullRequestData(
        number=103,
        title="Refactor billing invoice generation for clarity",
        body="No behavior change, splits a 300-line function into smaller pieces.",
        author="carol",
        additions=180,
        deletions=170,
        changed_files=["src/billing/invoices.py"],
        updated_at="2026-06-03T00:00:00",
        url="https://example.com/pr/103",
    ),
    PullRequestData(
        number=104,
        title="Update README with new setup instructions",
        body="Docs only change.",
        author="dave",
        additions=40,
        deletions=5,
        changed_files=["README.md"],
        updated_at="2026-06-04T00:00:00",
        url="https://example.com/pr/104",
    ),
    PullRequestData(
        number=105,
        title="Add tests for the retry/backoff helper",
        body="Covers the exponential backoff edge cases.",
        author="erin",
        additions=95,
        deletions=0,
        changed_files=["tests/test_retry.py"],
        updated_at="2026-06-05T00:00:00",
        url="https://example.com/pr/105",
    ),
    PullRequestData(
        number=106,
        title="chore: tidy up lint warnings across src/",
        body="Pure formatting/lint fixes, no logic changes.",
        author="frank",
        additions=60,
        deletions=60,
        changed_files=["src/utils/format.py"],
        updated_at="2026-06-06T00:00:00",
        url="https://example.com/pr/106",
    ),
    PullRequestData(
        number=107,
        title="Bump requests dependency to 2.32.0",
        body="Routine dependency bump picked up by dependabot.",
        author="dependabot",
        additions=2,
        deletions=2,
        changed_files=["requirements.txt"],
        updated_at="2026-06-07T00:00:00",
        url="https://example.com/pr/107",
    ),
    PullRequestData(
        number=108,
        title="Hotfix: patch critical auth bypass vulnerability",
        body="Security hotfix, patches a CVE allowing auth bypass. Merge ASAP.",
        author="alice",
        additions=8,
        deletions=2,
        changed_files=["src/auth/session.py"],
        updated_at="2026-06-08T00:00:00",
        url="https://example.com/pr/108",
    ),
    PullRequestData(
        number=109,
        title="Fix typo in billing error message",
        body="One-line fix.",
        author="carol",
        additions=1,
        deletions=1,
        changed_files=["src/billing/errors.py"],
        updated_at="2026-06-09T00:00:00",
        url="https://example.com/pr/109",
    ),
    PullRequestData(
        number=110,
        title="Add support for CSV export of billing reports",
        body="New feature: users can export their billing history as CSV.",
        author="bob",
        additions=610,
        deletions=15,
        changed_files=["src/billing/export.py", "src/billing/reports.py"],
        updated_at="2026-06-10T00:00:00",
        url="https://example.com/pr/110",
    ),
]

FIXTURE_ISSUES = [
    IssueData(
        number=201,
        title="Login fails with expired session token",
        body="Users are logged out unexpectedly when their session token expires mid-request, login broken.",
        labels=["bug"],
        url="https://example.com/issue/201",
    ),
    IssueData(
        number=202,
        title="Session token expiry causes login crash",
        body="Similar to other reports, the login session crashes when the token expires.",
        labels=["bug"],
        url="https://example.com/issue/202",
    ),
    IssueData(
        number=203,
        title="Login page throws error on session token refresh",
        body="Login session token refresh throws an unhandled error for some users.",
        labels=["bug"],
        url="https://example.com/issue/203",
    ),
    IssueData(
        number=204,
        title="Docs: setup instructions are outdated",
        body="The setup docs reference an old install process that no longer works.",
        labels=["docs"],
        url="https://example.com/issue/204",
    ),
    IssueData(
        number=205,
        title="Documentation missing for CLI setup flags",
        body="The setup docs don't mention the new CLI flags added last release.",
        labels=["docs"],
        url="https://example.com/issue/205",
    ),
    IssueData(
        number=206,
        title="Setup docs need an example for the retry/backoff config",
        body="Would like the setup docs to include an example of retry/backoff configuration.",
        labels=["docs"],
        url="https://example.com/issue/206",
    ),
    IssueData(
        number=207,
        title="Dashboard is slow to load with many billing reports",
        body="Dashboard performance degrades significantly with large billing reports datasets.",
        labels=["performance"],
        url="https://example.com/issue/207",
    ),
    IssueData(
        number=208,
        title="Billing dashboard performance regression after last release",
        body="Noticed a performance regression loading the billing dashboard after the last release.",
        labels=["performance"],
        url="https://example.com/issue/208",
    ),
    IssueData(
        number=209,
        title="Add keyboard shortcut for quick search",
        body="It would be nice to have a keyboard shortcut to open quick search.",
        labels=["feature"],
        url="https://example.com/issue/209",
    ),
]


class FakeGitHubClient:
    """Drop-in stand-in for `GitHubClient`, backed by the fixtures above."""

    def __init__(self, prs=None, issues=None, codeowners: str | None = FIXTURE_CODEOWNERS) -> None:
        self._prs = prs if prs is not None else FIXTURE_PRS
        self._issues = issues if issues is not None else FIXTURE_ISSUES
        self._codeowners = codeowners

    def fetch_open_pull_requests(self, repo_name: str, limit: int | None = None) -> list[PullRequestData]:
        return self._prs[:limit] if limit is not None else list(self._prs)

    def fetch_open_issues(
        self, repo_name: str, labels: list[str] | None = None, limit: int | None = None
    ) -> list[IssueData]:
        issues = self._issues
        if labels:
            issues = [i for i in issues if set(i.labels) & set(labels)]
        return issues[:limit] if limit is not None else list(issues)

    def fetch_codeowners(self, repo_name: str) -> str | None:
        return self._codeowners
