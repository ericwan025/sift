"""Runs `eval/tasks.yaml` through the agent/tools and reports pass rates.

`run_evaluation(tasks_path, prompt_version)` is the entry point the CLI's
`sift eval` command and this module's `__main__` block both call. Tasks
flagged `requires_live_llm: true` are reported as SKIPPED rather than
scored when no `OPENAI_API_KEY` is configured, since the offline fallbacks
(hashing embeddings, keyword routing) can't meaningfully attempt them --
see `agent.offline` and the README's Evaluation section.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from agent.backends import has_openai_key  # noqa: E402
from agent.config import get_settings  # noqa: E402
from eval.fixtures import FIXTURE_REPO, FakeGitHubClient  # noqa: E402
from eval.scorers import SCORERS, ScoreResult  # noqa: E402

DEFAULT_TASKS_PATH = REPO_ROOT / "eval" / "tasks.yaml"


@dataclass
class TaskResult:
    task_id: str
    task_type: str
    passed: bool
    skipped: bool
    reason: str


@dataclass
class EvalReport:
    prompt_version: str
    results: list[TaskResult] = field(default_factory=list)

    @property
    def scored(self) -> list[TaskResult]:
        return [r for r in self.results if not r.skipped]

    @property
    def pass_rate(self) -> float:
        scored = self.scored
        return sum(r.passed for r in scored) / len(scored) if scored else 0.0

    def pass_rate_by_type(self) -> dict[str, float]:
        rates = {}
        for task_type in sorted({r.task_type for r in self.results}):
            scored = [r for r in self.scored if r.task_type == task_type]
            rates[task_type] = sum(r.passed for r in scored) / len(scored) if scored else float("nan")
        return rates

    @property
    def failures(self) -> list[TaskResult]:
        return [r for r in self.scored if not r.passed]


def load_tasks(tasks_path: str | Path = DEFAULT_TASKS_PATH) -> list[dict]:
    with open(tasks_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_task(task: dict, prompt_version: str, settings) -> TaskResult:
    from agent.agent import run_agent
    from agent.tools.issue_clustering import issue_clustering
    from agent.tools.pr_triage import pr_triage

    task_id, task_type = task["id"], task["type"]
    scorer = SCORERS[task_type]

    if task.get("requires_live_llm") and not has_openai_key(settings):
        return TaskResult(task_id, task_type, passed=False, skipped=True, reason="requires OPENAI_API_KEY")

    if task_type in ("doc_lookup", "routing", "multi_step"):
        actual = run_agent(task["query"], settings=settings, prompt_version=prompt_version)
        score: ScoreResult = scorer(task, actual)
    elif task_type == "pr_triage":
        result = pr_triage(FIXTURE_REPO, client=FakeGitHubClient(), settings=settings)
        score = scorer(task, result)
    elif task_type == "issue_clustering":
        result = issue_clustering(FIXTURE_REPO, client=FakeGitHubClient(), settings=settings)
        score = scorer(task, result)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

    return TaskResult(task_id, task_type, passed=score.passed, skipped=False, reason=score.reason)


def run_evaluation(tasks_path: str | Path = DEFAULT_TASKS_PATH, prompt_version: str = "tuned") -> EvalReport:
    settings = get_settings()
    tasks = load_tasks(tasks_path)
    report = EvalReport(prompt_version=prompt_version)
    for task in tasks:
        report.results.append(_run_task(task, prompt_version, settings))
    return report


def print_report(report: EvalReport, console=None) -> None:
    if console is None:
        from rich.console import Console

        console = Console()

    from rich.table import Table

    online = has_openai_key(get_settings())
    console.print(
        f"[bold]Sift eval report[/bold] -- prompt_version={report.prompt_version}, "
        f"mode={'live' if online else 'offline'}"
    )

    table = Table(title="Pass rate by task type")
    table.add_column("Type")
    table.add_column("Pass rate")
    for task_type, rate in report.pass_rate_by_type().items():
        table.add_row(task_type, "n/a (all skipped)" if math.isnan(rate) else f"{rate:.0%}")
    console.print(table)

    skipped = [r for r in report.results if r.skipped]
    console.print(
        f"[bold]Overall:[/bold] {report.pass_rate:.0%} "
        f"({sum(r.passed for r in report.scored)}/{len(report.scored)} scored, {len(skipped)} skipped)"
    )

    if report.failures:
        console.print("\n[bold red]Failures:[/bold red]")
        for r in report.failures:
            console.print(f"  [{r.task_id}] {r.task_type}: {r.reason}")

    if skipped:
        console.print(f"\n[dim]Skipped (requires OPENAI_API_KEY): {', '.join(r.task_id for r in skipped)}[/dim]")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Sift eval suite.")
    parser.add_argument("--baseline", action="store_true", help="Run against the weak baseline prompt/config.")
    parser.add_argument("--current", action="store_true", help="Run against the tuned current prompt/config.")
    args = parser.parse_args()

    version = "baseline" if args.baseline and not args.current else "tuned"
    print_report(run_evaluation(prompt_version=version))
