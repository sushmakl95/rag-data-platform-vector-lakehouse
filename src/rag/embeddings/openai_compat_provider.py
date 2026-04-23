"""OpenAI-compatible HTTP embedding provider.

Works with OpenAI, Azure OpenAI, Together AI, Groq, Ollama
(via `/v1/embeddings`), or any vendor speaking the OpenAI wire format.
API key lives in env var `RAG_EMBED_API_KEY`; base URL in `RAG_EMBED_BASE_URL`.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


class OpenAICompatEmbeddingProvider:
    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.dim = dim
        self.base_url = (
            base_url or os.environ.get("RAG_EMBED_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = api_key or os.environ.get("RAG_EMBED_API_KEY", "")
        self.timeout = timeout

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(f"{self.base_url}/embeddings", headers=headers, json=payload)
            r.raise_for_status()
            return r.json()

    def embed(self, texts: list[str]) -> list[list[float]]:  # pragma: no cover
        resp = self._post({"model": self.model, "input": texts})
        return [d["embedding"] for d in resp["data"]]

    def embed_query(self, text: str) -> list[float]:  # pragma: no cover
        return self.embed([text])[0]
