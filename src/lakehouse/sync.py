"""Delta → vector-store sync. Reads the CDF from doc_chunks and upserts
into whatever `VectorStore` is injected. Kept pure + testable.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.types import Chunk, EmbeddedChunk
from src.rag.vector_stores.base import VectorStore


@dataclass(frozen=True)
class CdcRow:
    chunk_id: str
    source: str
    text: str
    metadata: dict
    embedding: list[float]
    _change_type: str  # 'insert' | 'update_postimage' | 'delete'


def partition_cdc(rows: list[CdcRow]) -> tuple[list[CdcRow], list[str]]:
    upserts = [r for r in rows if r._change_type in ("insert", "update_postimage")]
    deletes = [r.chunk_id for r in rows if r._change_type == "delete"]
    return upserts, deletes


def cdc_to_embedded_chunks(rows: list[CdcRow]) -> list[EmbeddedChunk]:
    return [
        EmbeddedChunk(
            chunk=Chunk(id=r.chunk_id, text=r.text, source=r.source, metadata=r.metadata),
            embedding=r.embedding,
        )
        for r in rows
    ]


def apply_cdc(store: VectorStore, rows: list[CdcRow]) -> dict[str, int]:
    """Apply a batch of CDC events. Order inside the batch is honoured:
    upserts land first, then deletes. If the same `chunk_id` appears as both
    an upsert and a delete, the delete wins (last-write semantics).
    """
    upserts, deletes = partition_cdc(rows)
    if upserts:
        store.add(cdc_to_embedded_chunks(upserts))
    if deletes:
        store.delete(deletes)
    return {"upserts": len(upserts), "deletes": len(deletes)}
