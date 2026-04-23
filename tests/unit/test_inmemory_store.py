from __future__ import annotations

from src.rag.types import Chunk, EmbeddedChunk


def _embedded(id_: str, text: str, vec: list[float], meta: dict | None = None) -> EmbeddedChunk:
    return EmbeddedChunk(
        chunk=Chunk(id=id_, text=text, source="t", metadata=meta or {}), embedding=vec
    )


def test_add_and_count(store):
    store.add([_embedded("1", "a", [1.0, 0.0]), _embedded("2", "b", [0.0, 1.0])])
    assert store.count() == 2


def test_query_returns_top_k_by_cosine(store):
    store.add(
        [
            _embedded("1", "aa", [1.0, 0.0]),
            _embedded("2", "bb", [0.9, 0.1]),
            _embedded("3", "cc", [0.0, 1.0]),
        ]
    )
    results = store.query([1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk.id == "1"
    assert results[1].chunk.id == "2"


def test_query_with_metadata_filter(store):
    store.add(
        [
            _embedded("1", "aa", [1.0, 0.0], {"tenant": "acme"}),
            _embedded("2", "bb", [1.0, 0.0], {"tenant": "globex"}),
        ]
    )
    out = store.query([1.0, 0.0], top_k=5, filters={"tenant": "acme"})
    assert len(out) == 1
    assert out[0].chunk.id == "1"


def test_delete(store):
    store.add([_embedded("1", "a", [1.0])])
    store.delete(["1"])
    assert store.count() == 0
