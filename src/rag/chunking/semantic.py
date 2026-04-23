"""Semantic chunking — split on embedding-similarity breakpoints between sentences.

Takes any embedding provider via dependency-injection so tests can use the
hash provider (no model download, no network).
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable

from src.rag.types import Chunk

EmbedFn = Callable[[list[str]], list[list[float]]]


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b, strict=False))
    da = sum(x * x for x in a) ** 0.5
    db = sum(x * x for x in b) ** 0.5
    return num / (da * db) if da and db else 0.0


def semantic_chunks(
    text: str,
    *,
    source: str,
    embed_fn: EmbedFn,
    breakpoint_percentile: float = 0.75,
    min_sentences_per_chunk: int = 2,
) -> list[Chunk]:
    sents = _split_sentences(text)
    if len(sents) <= min_sentences_per_chunk:
        cid = hashlib.md5(f"{source}:0:{text}".encode()).hexdigest()[:16]
        return [Chunk(id=cid, text=text, source=source, metadata={"strategy": "semantic"})]

    embeddings = embed_fn(sents)
    distances = [
        1.0 - _cosine(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)
    ]
    sorted_dists = sorted(distances)
    cut_idx = int(breakpoint_percentile * len(sorted_dists))
    cut_idx = min(cut_idx, len(sorted_dists) - 1)
    threshold = sorted_dists[cut_idx]

    segments: list[list[str]] = []
    current: list[str] = [sents[0]]
    for i, d in enumerate(distances):
        if d >= threshold and len(current) >= min_sentences_per_chunk:
            segments.append(current)
            current = [sents[i + 1]]
        else:
            current.append(sents[i + 1])
    if current:
        segments.append(current)

    out: list[Chunk] = []
    for idx, seg in enumerate(segments):
        body = " ".join(seg)
        cid = hashlib.md5(f"{source}:{idx}:{body}".encode()).hexdigest()[:16]
        out.append(
            Chunk(
                id=cid,
                text=body,
                source=source,
                metadata={"strategy": "semantic", "index": idx, "sentences": len(seg)},
            )
        )
    return out
