"""Builds the LangChain tool-calling agent and registers the three tools.

The agent is a LangGraph `create_agent` graph -- the langchain 1.x
replacement for the older `create_tool_calling_agent` / `AgentExecutor`
pair, which this codebase originally targeted but which has been removed
from the `langchain` package. The graph loop is equivalent: the model
proposes a tool call, the graph runs it, feeds the result back to the
model, and repeats until the model answers without calling a tool or the
`recursion_limit` is hit.
"""
from __future__ import annotations

from langchain.agents import create_agent

from agent.backends import has_openai_key
from agent.config import Settings, get_settings
from agent.offline import route_offline
from agent.tools.doc_lookup import doc_lookup, doc_lookup_tool
from agent.tools.issue_clustering import issue_clustering, issue_clustering_tool
from agent.tools.pr_triage import pr_triage, pr_triage_tool

# Weak first-pass prompt: terse tool descriptions only, no routing guidance.
# Kept so `sift eval --baseline` can reproduce the "before" reliability number.
BASELINE_SYSTEM_PROMPT = (
    "You are an assistant with access to tools for a GitHub repository. Use the tools if helpful."
)

# Tuned prompt: explicit per-tool routing rules, multi-step guidance, and an
# instruction for the no-tool-fits case so the agent never silently answers
# ungrounded when it should have said so.
TUNED_SYSTEM_PROMPT = """You are Sift, an assistant that helps developers work with a GitHub repository.
You have three tools:

- doc_lookup: answers "how/where/why does X work" questions about the indexed codebase, grounded in \
retrieved source. Prefer this whenever the question is about how the repo's code or docs work.
- pr_triage: classifies and ranks open pull requests. Use for anything about pull requests, code review \
backlog, or what to review next.
- issue_clustering: groups open issues by theme to surface duplicates/related issues. Use for anything \
about grouping, deduplicating, or finding themes in issues.

Rules:
1. Pick the single tool that best matches the question. If a question genuinely needs more than one \
capability (e.g. "summarize open PRs and also explain how the retry logic works"), call each relevant \
tool in turn.
2. If no tool fits the question, answer conversationally but explicitly say the answer is not grounded \
in the repository.
3. Never fabricate file paths, PR numbers, or issue numbers that a tool didn't return.
4. If a tool returns an error, report it plainly instead of guessing at an answer."""

TOOLS = [doc_lookup_tool, pr_triage_tool, issue_clustering_tool]


def build_agent(settings: Settings | None = None, prompt_version: str = "tuned"):
    """Build the tool-calling agent graph.

    `prompt_version` selects between the weak baseline prompt and the tuned
    prompt -- this is what lets the eval harness produce an honest
    before/after reliability number from the *same* tools and code.
    """
    settings = settings or get_settings()
    system_prompt = TUNED_SYSTEM_PROMPT if prompt_version == "tuned" else BASELINE_SYSTEM_PROMPT

    from langchain_openai import ChatOpenAI

    model = ChatOpenAI(model=settings.agent_model, api_key=settings.openai_api_key, temperature=0)
    return create_agent(model=model, tools=TOOLS, system_prompt=system_prompt)


def run_agent(
    question: str,
    settings: Settings | None = None,
    prompt_version: str = "tuned",
    repo: str | None = None,
) -> dict:
    """One-shot query through the agent.

    Returns the final answer text plus the ordered list of tool names
    invoked, which the eval harness uses to score routing correctness. With
    no `OPENAI_API_KEY` configured, falls back to `_run_offline` since the
    LangGraph tool-calling loop has no offline stand-in.
    """
    settings = settings or get_settings()
    if not has_openai_key(settings):
        return _run_offline(question, settings, repo=repo, prompt_version=prompt_version)

    graph = build_agent(settings, prompt_version=prompt_version)
    result = graph.invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": settings.max_agent_iterations * 2},
    )

    messages = result["messages"]
    tool_calls = [m.name for m in messages if getattr(m, "type", None) == "tool"]
    return {"answer": messages[-1].content, "tool_calls": tool_calls, "messages": messages}


def _run_offline(question: str, settings: Settings, repo: str | None, prompt_version: str = "tuned") -> dict:
    """Rule-based single-tool dispatch used when no `OPENAI_API_KEY` is set.

    There's no offline stand-in for the LangGraph tool-calling loop itself,
    so this routes to at most one tool by keyword match (`agent.offline.route_offline`)
    rather than reasoning about the question. Multi-step questions that
    genuinely need more than one tool are out of scope for this fallback.
    """
    tool_name = route_offline(question, prompt_version=prompt_version)
    if tool_name is None:
        return {
            "answer": (
                "This question doesn't clearly match doc_lookup, pr_triage, or "
                "issue_clustering, and there's no general-purpose reasoning available "
                "without OPENAI_API_KEY configured. Not grounded in the repository."
            ),
            "tool_calls": [],
        }

    if tool_name == "doc_lookup":
        result = doc_lookup(question, settings=settings)
        return {"answer": result.answer, "tool_calls": ["doc_lookup"], "result": result}

    if repo is None:
        return {
            "answer": f"This looks like a {tool_name} question, but no repo (owner/repo) was specified.",
            "tool_calls": [tool_name],
        }

    if tool_name == "pr_triage":
        result = pr_triage(repo, settings=settings)
        return {
            "answer": f"Triaged {len(result.triaged)} open PR(s) for {repo}.",
            "tool_calls": ["pr_triage"],
            "result": result,
        }

    result = issue_clustering(repo, settings=settings)
    return {
        "answer": f"Found {len(result.clusters)} issue cluster(s) for {repo}.",
        "tool_calls": ["issue_clustering"],
        "result": result,
    }
