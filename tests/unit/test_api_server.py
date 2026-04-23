from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.rag.api.server import STORE, app


@pytest.fixture(autouse=True)
def _clear_store():
    STORE._items.clear()
    yield
    STORE._items.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_healthz_reports_zero_before_ingest(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["stored_chunks"] == 0


def test_ask_without_ingest_returns_409(client):
    r = client.post("/ask", json={"question": "anything"})
    assert r.status_code == 409


def test_ingest_and_ask_roundtrip(client):
    r = client.post(
        "/ingest",
        json={
            "source": "doc.md",
            "text": "Delta Lake is a table format. It supports time travel. Lakehouse stores use it.",
            "chunk_size": 80,
        },
    )
    assert r.status_code == 200
    assert r.json()["chunks"] >= 1

    r = client.post("/ask", json={"question": "What is Delta Lake?", "top_k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"]
    assert len(body["contexts"]) >= 1


def test_ingest_rejects_empty_text(client):
    r = client.post("/ingest", json={"source": "empty", "text": "   "})
    assert r.status_code == 400
