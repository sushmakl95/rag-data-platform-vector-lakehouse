from __future__ import annotations

from src.rag.chunking.semantic import semantic_chunks
from src.rag.embeddings.hash_provider import HashEmbeddingProvider


def test_semantic_produces_chunks_from_multi_sentence_input():
    e = HashEmbeddingProvider(dim=64)
    text = (
        "The lakehouse stores data in Delta. "
        "It supports time travel. "
        "Marketing runs quarterly campaigns. "
        "Email open rates rose 12 percent."
    )
    chunks = semantic_chunks(text, source="t", embed_fn=e.embed, breakpoint_percentile=0.5)
    assert len(chunks) >= 1
    assert all(c.metadata["strategy"] == "semantic" for c in chunks)


def test_semantic_collapses_short_text_to_single_chunk():
    e = HashEmbeddingProvider(dim=64)
    chunks = semantic_chunks("Hi.", source="t", embed_fn=e.embed)
    assert len(chunks) == 1
