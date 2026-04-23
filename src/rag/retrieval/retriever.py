"""High-level retriever orchestrating vector + BM25 + rerank."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.rag.embeddings.base import EmbeddingProvider
from src.rag.retrieval.hybrid import BM25Index, reciprocal_rank_fusion
from src.rag.retrieval.reranker import rerank
from src.rag.types import Chunk, RetrievalResult
from src.rag.vector_stores.base import VectorStore


@dataclass
class RetrieverConfig:
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_rerank: int = 5
    use_rerank: bool = True


class HybridRetriever:
    def __init__(
        self,
        *,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        bm25: BM25Index | None = None,
        rerank_score_fn: Callable[[str, str], float] | None = None,
        config: RetrieverConfig | None = None,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.bm25 = bm25
        self.rerank_score_fn = rerank_score_fn
        self.config = config or RetrieverConfig()

    def retrieve(self, query: str, *, filters: dict | None = None) -> list[RetrievalResult]:
        q_vec = self.embedder.embed_query(query)
        vec_hits = self.vector_store.query(q_vec, top_k=self.config.top_k_vector, filters=filters)

        result_lists: list[list[RetrievalResult]] = [vec_hits]
        if self.bm25 is not None:
            result_lists.append(self.bm25.query(query, top_k=self.config.top_k_bm25))

        fused = reciprocal_rank_fusion(result_lists, top_n=self.config.top_k_rerank * 4)

        if self.config.use_rerank and self.rerank_score_fn is not None:
            return rerank(
                query, fused, score_fn=self.rerank_score_fn, top_k=self.config.top_k_rerank
            )
        return fused[: self.config.top_k_rerank]

    def chunks(self, results: list[RetrievalResult]) -> list[Chunk]:
        return [r.chunk for r in results]
