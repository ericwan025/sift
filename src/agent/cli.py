"""Typer CLI entrypoint. See README for full usage examples."""
from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from agent.backends import has_openai_key
from agent.config import get_settings

app = typer.Typer(help="Sift: an AI assistant over a GitHub repository.")
console = Console()


@app.command()
def index(repo_path: str = typer.Argument(..., help="Path to a local repo checkout to index.")) -> None:
    """Build the FAISS index for a local repo checkout."""
    from agent.indexing.build_index import build_index as build_index_impl

    settings = get_settings()
    if not has_openai_key(settings):
        console.print(
            "[yellow]No OPENAI_API_KEY set -- indexing with the offline hashing-embedding "
            "fallback (lower retrieval quality than text-embedding-3-small).[/yellow]"
        )
    with console.status(f"Indexing {repo_path}..."):
        summary = build_index_impl(repo_path, settings)
    console.print(f"[green]Files indexed:[/green] {summary['files_indexed']}")
    console.print(f"[green]Chunks created:[/green] {summary['chunks_created']}")
    console.print(f"[green]Index written to:[/green] {summary['index_path']} ({summary['index_size_bytes']} bytes)")


@app.command()
def ask(
    question: str = typer.Argument(..., help="A question for the agent."),
    repo: str | None = typer.Option(None, help="owner/repo, needed for PR/issue questions."),
) -> None:
    """One-shot query through the agent."""
    from agent.agent import run_agent

    settings = get_settings()
    result = run_agent(question, settings=settings, repo=repo)
    console.print(result["answer"])
    if result["tool_calls"]:
        console.print(f"[dim]tool calls: {', '.join(result['tool_calls'])}[/dim]")


@app.command()
def chat(repo: str | None = typer.Option(None, help="owner/repo, needed for PR/issue questions.")) -> None:
    """Interactive REPL loop."""
    from agent.agent import run_agent

    settings = get_settings()
    console.print("[bold]Sift chat[/bold] -- type 'exit' or Ctrl-D to quit.")
    while True:
        try:
            question = console.input("[bold cyan]> [/bold cyan]")
        except EOFError:
            break
        if question.strip().lower() in {"exit", "quit"}:
            break
        if not question.strip():
            continue
        result = run_agent(question, settings=settings, repo=repo)
        console.print(result["answer"])
        if result["tool_calls"]:
            console.print(f"[dim]tool calls: {', '.join(result['tool_calls'])}[/dim]")


@app.command()
def triage(
    repo: str = typer.Argument(..., help="owner/repo"),
    limit: int = typer.Option(20, help="Max number of PRs to fetch."),
) -> None:
    """Direct PR triage (bypasses agent routing)."""
    from agent.tools.pr_triage import pr_triage as pr_triage_impl

    settings = get_settings()
    result = pr_triage_impl(repo, limit=limit, settings=settings)

    table = Table(title=f"PR triage: {repo}")
    table.add_column("#")
    table.add_column("Priority")
    table.add_column("Category")
    table.add_column("Title")
    table.add_column("Reviewers")
    table.add_column("Summary")
    for pr in result.triaged:
        table.add_row(
            str(pr.number),
            pr.priority.value,
            pr.category.value,
            pr.title,
            ", ".join(pr.suggested_reviewers) or "-",
            pr.summary,
        )
    console.print(table)


@app.command()
def cluster(
    repo: str = typer.Argument(..., help="owner/repo"),
    limit: int = typer.Option(50, help="Max number of issues to fetch."),
) -> None:
    """Direct issue clustering (bypasses agent routing)."""
    from agent.tools.issue_clustering import issue_clustering as issue_clustering_impl

    settings = get_settings()
    result = issue_clustering_impl(repo, limit=limit, settings=settings)

    table = Table(title=f"Issue clusters: {repo}")
    table.add_column("Label")
    table.add_column("Size")
    table.add_column("Representative")
    table.add_column("Issue #s")
    for c in result.clusters:
        table.add_row(c.label, str(c.size), f"#{c.representative_issue}", ", ".join(f"#{n}" for n in c.issue_numbers))
    console.print(table)


@app.command(name="eval")
def eval_command(
    baseline: bool = typer.Option(False, "--baseline", help="Run against the weak baseline prompt/config."),
    current: bool = typer.Option(False, "--current", help="Run against the tuned current prompt/config."),
    tasks_path: str = typer.Option(
        str(Path(__file__).resolve().parents[2] / "eval" / "tasks.yaml"), help="Path to tasks.yaml."
    ),
) -> None:
    """Run the evaluation suite and print the report."""
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from eval.run_eval import print_report, run_evaluation

    prompt_version = "baseline" if baseline and not current else "tuned"
    report = run_evaluation(tasks_path=tasks_path, prompt_version=prompt_version)
    print_report(report, console=console)


if __name__ == "__main__":
    app()
