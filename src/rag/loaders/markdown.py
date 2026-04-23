"""Markdown loader — strips code fences' language hints, preserves headings."""

from __future__ import annotations

import re
from pathlib import Path


def load_markdown(path: Path | str) -> dict:
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    # Normalise: drop language hints on code fences, collapse 3+ blank lines
    text = re.sub(r"```[a-zA-Z0-9_-]+\n", "```\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    title = _first_title(text) or p.stem
    return {"source": str(p), "title": title, "text": text}


def _first_title(text: str) -> str | None:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return None
