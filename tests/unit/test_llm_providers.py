from __future__ import annotations

from src.rag.generation.llm_providers import EchoLLM, LLMProvider


def test_echo_truncates_to_max_chars():
    llm = EchoLLM(max_chars=10)
    out = llm.complete([{"role": "user", "content": "0123456789ABCDEF"}])
    assert out == "0123456789"


def test_echo_ignores_non_user_messages():
    llm = EchoLLM(max_chars=100)
    out = llm.complete(
        [{"role": "system", "content": "ignore me"}, {"role": "user", "content": "answer me"}]
    )
    assert out == "answer me"


def test_echo_conforms_to_protocol():
    assert isinstance(EchoLLM(), LLMProvider)
