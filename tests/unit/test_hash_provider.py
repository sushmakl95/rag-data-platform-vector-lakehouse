from __future__ import annotations

import math

import pytest

from src.rag.embeddings.hash_provider import HashEmbeddingProvider


def test_embeddings_are_deterministic():
    e = HashEmbeddingProvider(dim=128)
    a1 = e.embed_query("hello world")
    a2 = e.embed_query("hello world")
    assert a1 == a2


def test_embeddings_differ_for_different_inputs():
    e = HashEmbeddingProvider(dim=128)
    assert e.embed_query("alpha") != e.embed_query("beta")


def test_embeddings_are_unit_normalised():
    e = HashEmbeddingProvider(dim=128)
    v = e.embed_query("hello")
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


def test_dim_must_be_multiple_of_8():
    with pytest.raises(ValueError):
        HashEmbeddingProvider(dim=100)


def test_batched_and_single_match():
    e = HashEmbeddingProvider(dim=64)
    b = e.embed(["a", "b"])
    assert b[0] == e.embed_query("a")
    assert b[1] == e.embed_query("b")
