"""Qdrant HTTP-based store. Lazy-imports qdrant_client to keep CI light."""

from __future__ import annotations

from dataclasses import dataclass

from src.rag.types import Chunk, EmbeddedChunk, RetrievalResult


@dataclass
class QdrantConfig:
    collection: str = "doc_chunks"
    dim: int = 384
    distance: str = "Cosine"  # Cosine | Dot | Euclid


def build_collection_payload(cfg: QdrantConfig) -> dict:
    return {
        "vectors": {"size": cfg.dim, "distance": cfg.distance},
        "hnsw_config": {"m": 16, "ef_construct": 100},
        "optimizers_config": {"default_segment_number": 2},
    }


def build_point(item: EmbeddedChunk) -> dict:
    return {
        "id": item.chunk.id,
        "vector": item.embedding,
        "payload": {
            "text": item.chunk.text,
            "source": item.chunk.source,
            "metadata": item.chunk.metadata,
        },
    }


def build_query_filter(filters: dict) -> dict:
    return {"must": [{"key": f"metadata.{k}", "match": {"value": v}} for k, v in filters.items()]}


class QdrantStore:  # pragma: no cover - exercised in integration tests
    def __init__(
        self, url: str = "http://localhost:6333", config: QdrantConfig | None = None
    ) -> None:
        try:
            from qdrant_client import QdrantClient
        except ImportError as e:
            raise RuntimeError("qdrant-client not installed") from e
        self._client = QdrantClient(url=url)
        self.config = config or QdrantConfig()

    def add(self, items: list[EmbeddedChunk]) -> None:
        self._client.upsert(
            collection_name=self.config.collection,
            points=[build_point(i) for i in items],
        )

    def query(
        self, embedding: list[float], *, top_k: int = 5, filters: dict | None = None
    ) -> list[RetrievalResult]:
        flt = build_query_filter(filters) if filters else None
        hits = self._client.search(
            collection_name=self.config.collection,
            query_vector=embedding,
            limit=top_k,
            query_filter=flt,
        )
        return [
            RetrievalResult(
                chunk=Chunk(
                    id=str(h.id),
                    text=h.payload["text"],
                    source=h.payload["source"],
                    metadata=h.payload.get("metadata", {}),
                ),
                score=float(h.score),
            )
            for h in hits
        ]

    def delete(self, ids: list[str]) -> None:
        self._client.delete(collection_name=self.config.collection, points_selector=ids)

    def count(self) -> int:
        return int(self._client.count(collection_name=self.config.collection).count)
