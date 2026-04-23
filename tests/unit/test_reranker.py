from __future__ import annotations

from src.rag.retrieval.reranker import keyword_overlap_score, rerank
from src.rag.types import Chunk, RetrievalResult


def test_keyword_overlap_score_identity():
    assert keyword_overlap_score("hello world", "hello world") == 1.0


def test_keyword_overlap_score_partial():
    assert 0.0 < keyword_overlap_score("hello data", "hello world") < 1.0


def test_rerank_reorders_by_score_fn():
    candidates = [
        RetrievalResult(Chunk(id="1", text="marketing email campaigns", source="t"), score=0.9),
        RetrievalResult(Chunk(id="2", text="delta lakehouse architecture", source="t"), score=0.2),
    ]
    out = rerank("delta lakehouse", candidates, score_fn=keyword_overlap_score)
    assert out[0].chunk.id == "2"
