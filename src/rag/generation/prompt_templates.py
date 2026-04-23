"""Prompt templates for the RAG generator."""

from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question STRICTLY using the provided "
    "context snippets. If the context does not contain the answer, reply exactly: "
    "'I don't have enough information in the provided context.'"
)


def format_context(snippets: list[str]) -> str:
    return "\n\n".join(f"[Source {i + 1}]\n{s}" for i, s in enumerate(snippets))


def build_messages(question: str, contexts: list[str]) -> list[dict]:
    ctx = format_context(contexts)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer concisely.",
        },
    ]
