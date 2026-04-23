"""Cross-encoder reranker. Uses an injected score function so it's testable."""

from __future__ import annotations

from collections.abc import Callable

from src.rag.types import RetrievalResult

ScoreFn = Callable[[str, str], float]


def rerank(
    query: str,
    candidates: list[RetrievalResult],
    *,
    score_fn: ScoreFn,
    top_k: int | None = None,
) -> list[RetrievalResult]:
    rescored = [
        RetrievalResult(chunk=c.chunk, score=score_fn(query, c.chunk.text)) for c in candidates
    ]
    rescored.sort(key=lambda r: r.score, reverse=True)
    return rescored[:top_k] if top_k else rescored


def keyword_overlap_score(query: str, doc: str) -> float:
    """Simple deterministic reranker for tests: lowercase token overlap ratio."""
    q = set(query.lower().split())
    d = set(doc.lower().split())
    return len(q & d) / max(1, len(q))
