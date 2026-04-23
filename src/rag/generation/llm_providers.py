"""Provider-agnostic LLM interface.

For hermetic CI we use the `EchoLLM` provider that returns a deterministic
summary. Real providers (Ollama local, and any OpenAI-wire-compatible cloud)
are available under env-var-driven instantiation.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, messages: list[dict]) -> str: ...


class EchoLLM:
    """Deterministic local provider: returns the concatenated user message, truncated."""

    name = "echo"

    def __init__(self, max_chars: int = 400) -> None:
        self.max_chars = max_chars

    def complete(self, messages: list[dict]) -> str:
        user = next((m["content"] for m in messages if m["role"] == "user"), "")
        return user[: self.max_chars]


class OllamaLLM:  # pragma: no cover - integration only
    name = "ollama"

    def __init__(self, model: str = "llama3.2", base_url: str | None = None) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_URL", "http://localhost:11434")).rstrip(
            "/"
        )

    def complete(self, messages: list[dict]) -> str:
        import httpx

        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                f"{self.base_url}/api/chat",
                json={"model": self.model, "messages": messages, "stream": False},
            )
            r.raise_for_status()
            return r.json()["message"]["content"]


class OpenAICompatLLM:  # pragma: no cover - integration only
    name = "openai-compat"

    def __init__(self, model: str = "gpt-4o-mini", base_url: str | None = None) -> None:
        self.model = model
        self.base_url = (
            base_url or os.environ.get("RAG_LLM_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = os.environ.get("RAG_LLM_API_KEY", "")

    def complete(self, messages: list[dict]) -> str:
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(timeout=60.0) as client:
            r = client.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json={"model": self.model, "messages": messages},
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
