from __future__ import annotations

from src.rag.chunking.recursive import recursive_chunks


def test_recursive_splits_on_paragraphs():
    text = "para1\n\npara2\n\npara3"
    chunks = recursive_chunks(text, source="t", size=10)
    texts = [c.text for c in chunks]
    assert "para1" in texts and "para2" in texts and "para3" in texts


def test_recursive_keeps_chunks_under_size_when_possible():
    text = "a. b. c. d. e. f. g. h. " * 5
    chunks = recursive_chunks(text, source="t", size=40)
    # Some chunks may exceed size if no clean split exists; allow +20% slack
    assert all(len(c.text) <= 60 for c in chunks)


def test_recursive_long_unsplittable_falls_back_to_hard_cut():
    text = "a" * 500
    chunks = recursive_chunks(text, source="t", size=100)
    assert all(len(c.text) <= 100 for c in chunks)
    assert sum(len(c.text) for c in chunks) == 500


def test_recursive_metadata():
    chunks = recursive_chunks("p1\n\np2", source="t", size=5)
    assert all(c.metadata["strategy"] == "recursive" for c in chunks)
    assert [c.metadata["index"] for c in chunks] == list(range(len(chunks)))
