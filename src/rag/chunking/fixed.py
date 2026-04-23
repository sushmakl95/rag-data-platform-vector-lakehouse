"""Fixed-size character chunking with optional overlap."""

from __future__ import annotations

import hashlib

from src.rag.types import Chunk


def fixed_chunks(
    text: str,
    *,
    source: str,
    size: int = 1000,
    overlap: int = 100,
) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be positive")
    if overlap < 0 or overlap >= size:
        raise ValueError("overlap must be in [0, size)")

    chunks: list[Chunk] = []
    stride = size - overlap
    i = 0
    idx = 0
    while i < len(text):
        segment = text[i : i + size]
        cid = hashlib.md5(f"{source}:{i}:{segment}".encode()).hexdigest()[:16]
        chunks.append(
            Chunk(
                id=cid,
                text=segment,
                source=source,
                metadata={"strategy": "fixed", "start": i, "end": i + len(segment), "index": idx},
            )
        )
        idx += 1
        if i + size >= len(text):
            break
        i += stride
    return chunks
