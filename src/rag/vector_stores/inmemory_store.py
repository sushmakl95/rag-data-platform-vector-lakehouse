"""In-memory vector store — used by tests and single-process demos."""

from __future__ import annotations

import math

from src.rag.types import Chunk, EmbeddedChunk, RetrievalResult


def cosine_sim(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=False))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(x * x for x in b))
    return num / (da * db) if da and db else 0.0


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._items: dict[str, tuple[Chunk, list[float]]] = {}

    def add(self, items: list[EmbeddedChunk]) -> None:
        for item in items:
            self._items[item.chunk.id] = (item.chunk, item.embedding)

    def delete(self, ids: list[str]) -> None:
        for i in ids:
            self._items.pop(i, None)

    def count(self) -> int:
        return len(self._items)

    def query(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]:
        def matches(meta: dict, flt: dict) -> bool:
            return all(meta.get(k) == v for k, v in flt.items())

        scored: list[RetrievalResult] = []
        for _, (chunk, vec) in self._items.items():
            if filters and not matches(chunk.metadata, filters):
                continue
            scored.append(RetrievalResult(chunk=chunk, score=cosine_sim(embedding, vec)))
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]
