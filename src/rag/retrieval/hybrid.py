"""Hybrid retrieval: BM25 lexical + vector semantic, fused via Reciprocal Rank Fusion."""

from __future__ import annotations

from src.rag.types import Chunk, RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    *,
    k: int = 60,
    top_n: int = 10,
) -> list[RetrievalResult]:
    """Standard RRF: score(d) = sum_i 1 / (k + rank_i(d))."""
    if not result_lists:
        return []
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for rlist in result_lists:
        for rank, r in enumerate(rlist):
            cid = r.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            chunks.setdefault(cid, r.chunk)
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [RetrievalResult(chunk=chunks[cid], score=s) for cid, s in ordered]


class BM25Index:
    """Thin wrapper around rank_bm25.BM25Okapi, lazy-imported."""

    def __init__(self, corpus: list[Chunk]) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("rank_bm25 not installed") from e
        self._chunks = corpus
        tokenised = [c.text.lower().split() for c in corpus]
        self._bm25 = BM25Okapi(tokenised)

    def query(self, text: str, *, top_k: int = 10) -> list[RetrievalResult]:
        tokens = text.lower().split()
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._chunks, scores, strict=False), key=lambda t: t[1], reverse=True)[
            :top_k
        ]
        return [RetrievalResult(chunk=c, score=float(s)) for c, s in ranked]
