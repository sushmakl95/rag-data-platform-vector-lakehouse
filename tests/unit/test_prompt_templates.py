from __future__ import annotations

from src.rag.generation.prompt_templates import build_messages, format_context


def test_format_context_numbers_sources():
    out = format_context(["alpha", "beta"])
    assert "[Source 1]" in out and "[Source 2]" in out
    assert "alpha" in out and "beta" in out


def test_build_messages_has_system_and_user():
    msgs = build_messages("why?", ["because"])
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user"]
    assert "because" in msgs[1]["content"]
    assert "why?" in msgs[1]["content"]
