from __future__ import annotations

from pathlib import Path

from src.rag.loaders.html import load_html
from src.rag.loaders.markdown import load_markdown


def test_markdown_returns_title_and_text(tmp_path: Path):
    p = tmp_path / "a.md"
    p.write_text("# Hello\n\nBody goes here.")
    doc = load_markdown(p)
    assert doc["title"] == "Hello"
    assert "Body goes here" in doc["text"]


def test_markdown_title_fallback(tmp_path: Path):
    p = tmp_path / "untitled.md"
    p.write_text("just body, no heading")
    assert load_markdown(p)["title"] == "untitled"


def test_html_strips_tags_and_scripts(tmp_path: Path):
    p = tmp_path / "a.html"
    p.write_text("<html><script>alert(1)</script><body>Hello <b>World</b>!</body></html>")
    out = load_html(p)
    assert "Hello" in out["text"]
    assert "World" in out["text"]
    assert "alert" not in out["text"]
    assert "<" not in out["text"]
