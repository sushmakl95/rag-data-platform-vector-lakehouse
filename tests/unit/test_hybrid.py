from __future__ import annotations

from src.rag.retrieval.hybrid import BM25Index, reciprocal_rank_fusion
from src.rag.types import Chunk, RetrievalResult


def _r(id_: str, text: str, score: float) -> RetrievalResult:
    return RetrievalResult(chunk=Chunk(id=id_, text=text, source="t"), score=score)


def test_rrf_merges_and_ranks():
    list_a = [_r("a", "aa", 0.9), _r("b", "bb", 0.8), _r("c", "cc", 0.7)]
    list_b = [_r("b", "bb", 0.95), _r("d", "dd", 0.9), _r("a", "aa", 0.7)]
    fused = reciprocal_rank_fusion([list_a, list_b], top_n=4)
    ids = [r.chunk.id for r in fused]
    # b appears first in both → should rank top
    assert ids[0] in {"a", "b"}
    assert "d" in ids


def test_rrf_handles_empty():
    assert reciprocal_rank_fusion([]) == []


def test_bm25_query_orders_by_lexical_overlap():
    corpus = [
        Chunk(id="1", text="lakehouse architecture delta", source="t"),
        Chunk(id="2", text="marketing email campaigns", source="t"),
        Chunk(id="3", text="delta lake schema evolution", source="t"),
    ]
    bm = BM25Index(corpus)
    results = bm.query("delta lakehouse", top_k=2)
    top_ids = [r.chunk.id for r in results]
    assert "1" in top_ids or "3" in top_ids
    assert "2" not in top_ids
