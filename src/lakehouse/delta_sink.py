"""Delta Lake sink for `doc_chunks`. Lazy-imports delta-spark; pure-functional core
so the projection logic is testable without Spark.
"""

from __future__ import annotations

from typing import Any

from src.rag.types import EmbeddedChunk


def to_record(item: EmbeddedChunk) -> dict[str, Any]:
    return {
        "chunk_id": item.chunk.id,
        "source": item.chunk.source,
        "text": item.chunk.text,
        "metadata": item.chunk.metadata,
        "embedding": item.embedding,
        "embedding_dim": len(item.embedding),
    }


def to_records(items: list[EmbeddedChunk]) -> list[dict[str, Any]]:
    return [to_record(i) for i in items]


def partition_cols() -> list[str]:
    return ["source", "ingested_date"]


def merge_sql(target: str = "doc_chunks") -> str:
    return (
        f"MERGE INTO {target} t "
        "USING _stage s ON t.chunk_id = s.chunk_id "
        "WHEN MATCHED THEN UPDATE SET "
        "  text = s.text, metadata = s.metadata, embedding = s.embedding "
        "WHEN NOT MATCHED THEN INSERT *"
    )
