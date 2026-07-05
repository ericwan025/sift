"""Pydantic models for tool inputs/outputs. Every tool returns one of these —
never a raw dict or string — so the CLI and eval harness can rely on a typed
contract.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class SourceRef(BaseModel):
    """A citation pointing at the exact lines an answer was grounded in."""

    file_path: str
    start_line: int
    end_line: int

    def __str__(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


class DocLookupResult(BaseModel):
    answer: str
    sources: list[SourceRef] = Field(default_factory=list)
    grounded: bool


class PRCategory(str, Enum):
    BUG_FIX = "bug-fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    DOCS = "docs"
    TEST = "test"
    CHORE = "chore"
    DEPENDENCY = "dependency"


class PRPriority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class PRClassification(BaseModel):
    number: int
    title: str
    url: str
    category: PRCategory
    priority: PRPriority
    suggested_reviewers: list[str] = Field(default_factory=list)
    summary: str
    rationale: str


class PRTriageResult(BaseModel):
    repo: str
    triaged: list[PRClassification] = Field(default_factory=list)

    def ranked(self) -> list[PRClassification]:
        order = {PRPriority.HIGH: 0, PRPriority.MEDIUM: 1, PRPriority.LOW: 2}
        return sorted(self.triaged, key=lambda p: order[p.priority])


class IssueCluster(BaseModel):
    label: str
    issue_numbers: list[int]
    size: int
    representative_issue: int


class IssueClusteringResult(BaseModel):
    repo: str
    clusters: list[IssueCluster] = Field(default_factory=list)

    def sorted_by_size(self) -> list[IssueCluster]:
        return sorted(self.clusters, key=lambda c: c.size, reverse=True)
