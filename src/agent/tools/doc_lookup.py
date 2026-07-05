"""RAG doc lookup: retrieves top-k chunks from the FAISS index and asks the
LLM to answer strictly from that context, citing sources. Returns
`grounded=False` instead of hallucinating when the context doesn't support
an answer -- this is the anti-hallucination guarantee the tool exists for.
"""
from __future__ import annotations

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from agent.config import Settings, get_settings
from agent.indexing.loader import FaissRetriever, IndexNotBuiltError, RetrievedChunk
from agent.schemas import DocLookupResult, SourceRef

NOT_FOUND_SENTENCE = "I don't have enough information in the codebase to answer that."

SYSTEM_PROMPT = f"""You are a documentation lookup assistant for a codebase.
Answer the user's question using ONLY the provided context chunks below.
Do not use outside knowledge and do not guess.
If the context does not contain enough information to answer, respond with
exactly: "{NOT_FOUND_SENTENCE}"

Set grounded to false only if you used that exact not-found sentence."""


class _LLMAnswer(BaseModel):
    answer: str
    grounded: bool


def _build_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(f"### {c.file_path}:{c.start_line}-{c.end_line}\n{c.chunk_text}" for c in chunks)


def doc_lookup(
    question: str,
    retriever: FaissRetriever | None = None,
    settings: Settings | None = None,
    top_k: int | None = None,
) -> DocLookupResult:
    """Answer a natural-language question about the codebase, grounded in the FAISS index."""
    settings = settings or get_settings()
    try:
        retriever = retriever or FaissRetriever(settings)
    except IndexNotBuiltError as exc:
        return DocLookupResult(answer=str(exc), sources=[], grounded=False)

    chunks = retriever.search(question, top_k=top_k)
    if not chunks:
        return DocLookupResult(answer=NOT_FOUND_SENTENCE, sources=[], grounded=False)

    llm = ChatOpenAI(model=settings.agent_model, api_key=settings.openai_api_key, temperature=0)
    structured_llm = llm.with_structured_output(_LLMAnswer)
    result: _LLMAnswer = structured_llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"Context:\n{_build_context(chunks)}\n\nQuestion: {question}"),
        ]
    )

    sources = (
        [SourceRef(file_path=c.file_path, start_line=c.start_line, end_line=c.end_line) for c in chunks]
        if result.grounded
        else []
    )
    return DocLookupResult(answer=result.answer, sources=sources, grounded=result.grounded)


@tool
def doc_lookup_tool(question: str) -> str:
    """Answer a question about how, where, or why something works in the indexed
    codebase. Use for any "how does X work" / "where is Y" style question.
    Returns a grounded answer with file:line citations, or reports that no
    answer was found in the codebase rather than guessing."""
    return doc_lookup(question).model_dump_json()
