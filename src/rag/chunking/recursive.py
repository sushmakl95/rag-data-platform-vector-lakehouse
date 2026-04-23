"""Recursive character splitter — LangChain-style, zero deps.

Tries separators in order (most-preferred first) and recurses on oversized segments.
"""

from __future__ import annotations

import hashlib

from src.rag.types import Chunk

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _split(text: str, separators: list[str], size: int) -> list[str]:
    if not text:
        return []
    if len(text) <= size:
        return [text]
    sep = next((s for s in separators if s and s in text), "")
    if not sep:
        return [text[i : i + size] for i in range(0, len(text), size)]
    parts = text.split(sep)
    out: list[str] = []
    buf = ""
    for p in parts:
        candidate = (buf + sep + p).strip(sep) if buf else p
        if len(candidate) <= size:
            buf = candidate
        else:
            if buf:
                out.append(buf)
            if len(p) > size:
                rest = separators[separators.index(sep) + 1 :]
                out.extend(_split(p, rest, size))
                buf = ""
            else:
                buf = p
    if buf:
        out.append(buf)
    return out


def recursive_chunks(
    text: str,
    *,
    source: str,
    size: int = 1000,
    separators: list[str] | None = None,
) -> list[Chunk]:
    if size <= 0:
        raise ValueError("size must be positive")
    seps = separators or DEFAULT_SEPARATORS
    pieces = _split(text, seps, size)
    out: list[Chunk] = []
    for idx, p in enumerate(pieces):
        cid = hashlib.md5(f"{source}:{idx}:{p}".encode()).hexdigest()[:16]
        out.append(
            Chunk(
                id=cid,
                text=p,
                source=source,
                metadata={"strategy": "recursive", "index": idx, "length": len(p)},
            )
        )
    return out
