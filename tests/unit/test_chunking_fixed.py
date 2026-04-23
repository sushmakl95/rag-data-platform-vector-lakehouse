from __future__ import annotations

import pytest

from src.rag.chunking.fixed import fixed_chunks


def test_fixed_respects_size_and_overlap():
    text = "abcdefghij" * 10  # 100 chars
    chunks = fixed_chunks(text, source="t", size=30, overlap=5)
    assert all(len(c.text) <= 30 for c in chunks)
    assert chunks[0].metadata["start"] == 0
    assert chunks[1].metadata["start"] == 25  # 30 - 5
    assert chunks[-1].metadata["end"] <= len(text)


def test_fixed_single_chunk_when_text_fits():
    chunks = fixed_chunks("short", source="t", size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].text == "short"


def test_fixed_chunk_ids_are_unique():
    chunks = fixed_chunks("a" * 1000, source="t", size=50, overlap=0)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_fixed_rejects_bad_overlap():
    with pytest.raises(ValueError):
        fixed_chunks("x", source="t", size=10, overlap=10)


def test_fixed_rejects_non_positive_size():
    with pytest.raises(ValueError):
        fixed_chunks("x", source="t", size=0)


def test_fixed_metadata_records_strategy():
    chunks = fixed_chunks("xyz" * 40, source="t", size=30, overlap=5)
    assert all(c.metadata["strategy"] == "fixed" for c in chunks)
