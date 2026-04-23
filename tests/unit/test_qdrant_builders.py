from __future__ import annotations

from src.rag.types import Chunk, EmbeddedChunk
from src.rag.vector_stores.qdrant_store import (
    QdrantConfig,
    build_collection_payload,
    build_point,
    build_query_filter,
)


def test_collection_payload_shape():
    p = build_collection_payload(QdrantConfig(dim=768, distance="Cosine"))
    assert p["vectors"] == {"size": 768, "distance": "Cosine"}
    assert "hnsw_config" in p


def test_build_point_preserves_payload():
    ec = EmbeddedChunk(
        chunk=Chunk(id="x", text="hi", source="s", metadata={"k": "v"}), embedding=[0.1, 0.2]
    )
    point = build_point(ec)
    assert point["id"] == "x"
    assert point["vector"] == [0.1, 0.2]
    assert point["payload"]["metadata"] == {"k": "v"}


def test_build_query_filter():
    flt = build_query_filter({"tenant": "acme", "env": "prod"})
    keys = {m["key"] for m in flt["must"]}
    assert keys == {"metadata.tenant", "metadata.env"}
