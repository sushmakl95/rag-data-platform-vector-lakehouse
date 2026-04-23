"""Deterministic hash-based embeddings for hermetic CI.

Never reach for this in production — the vectors are not semantically meaningful,
they're designed to be stable for tests that verify wiring, similarity ordering
with known inputs, and vector-store contracts.
"""

from __future__ import annotations

import hashlib
import math


class HashEmbeddingProvider:
    """Deterministic embeddings from SHA256 bytes, L2-normalised."""

    def __init__(self, dim: int = 384) -> None:
        if dim <= 0 or dim % 8 != 0:
            raise ValueError("dim must be a positive multiple of 8")
        self.dim = dim

    def _raw(self, text: str) -> list[float]:
        bytes_per_dim = max(1, 32 // (self.dim // 8))
        out: list[float] = []
        i = 0
        while len(out) < self.dim:
            h = hashlib.sha256(f"{i}:{text}".encode()).digest()
            for offset in range(0, len(h), bytes_per_dim):
                if len(out) >= self.dim:
                    break
                chunk = h[offset : offset + bytes_per_dim]
                # Map to [-1, 1]
                val = int.from_bytes(chunk, "big", signed=False)
                scaled = (val / (1 << (8 * len(chunk)))) * 2.0 - 1.0
                out.append(scaled)
            i += 1
        return out

    def _normalise(self, v: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in v))
        if norm == 0:
            return v
        return [x / norm for x in v]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._normalise(self._raw(t)) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._normalise(self._raw(text))
