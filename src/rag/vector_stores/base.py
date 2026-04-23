"""VectorStore protocol — uniform contract for all backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.rag.types import EmbeddedChunk, RetrievalResult


@runtime_checkable
class VectorStore(Protocol):
    def add(self, items: list[EmbeddedChunk]) -> None: ...

    def query(
        self,
        embedding: list[float],
        *,
        top_k: int = 5,
        filters: dict | None = None,
    ) -> list[RetrievalResult]: ...

    def delete(self, ids: list[str]) -> None: ...

    def count(self) -> int: ...
