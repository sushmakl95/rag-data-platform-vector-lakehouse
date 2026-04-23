from __future__ import annotations

from src.rag.retrieval.hybrid import BM25Index
from src.rag.retrieval.reranker import keyword_overlap_score
from src.rag.retrieval.retriever import HybridRetriever, RetrieverConfig
from src.rag.types import Chunk, EmbeddedChunk


def _prime(store, embedder, docs):
    chunks = [Chunk(id=str(i), text=t, source="t") for i, t in enumerate(docs)]
    vecs = embedder.embed([c.text for c in chunks])
    store.add([EmbeddedChunk(chunk=c, embedding=v) for c, v in zip(chunks, vecs, strict=True)])
    return chunks


def test_hybrid_retriever_returns_top_k(embedder, store):
    chunks = _prime(
        store, embedder, ["delta lake architecture", "marketing email campaigns", "vector stores"]
    )
    r = HybridRetriever(
        embedder=embedder,
        vector_store=store,
        bm25=BM25Index(chunks),
        rerank_score_fn=keyword_overlap_score,
        config=RetrieverConfig(top_k_vector=5, top_k_bm25=5, top_k_rerank=2, use_rerank=True),
    )
    hits = r.retrieve("delta architecture")
    assert len(hits) == 2
    assert any("delta" in h.chunk.text for h in hits)


def test_retriever_respects_use_rerank_false(embedder, store):
    chunks = _prime(store, embedder, ["a b c", "x y z"])
    r = HybridRetriever(
        embedder=embedder,
        vector_store=store,
        bm25=BM25Index(chunks),
        config=RetrieverConfig(top_k_vector=5, top_k_bm25=5, top_k_rerank=1, use_rerank=False),
    )
    hits = r.retrieve("a b")
    assert len(hits) == 1
