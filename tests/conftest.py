"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def embedder():
    from src.rag.embeddings.hash_provider import HashEmbeddingProvider

    return HashEmbeddingProvider(dim=384)


@pytest.fixture
def store():
    from src.rag.vector_stores.inmemory_store import InMemoryVectorStore

    return InMemoryVectorStore()
