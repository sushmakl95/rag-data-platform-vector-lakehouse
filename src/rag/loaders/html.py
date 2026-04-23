"""Minimal HTML→text loader (regex-only; avoids pulling BeautifulSoup in CI)."""

from __future__ import annotations

import html
import re
from pathlib import Path

_SCRIPT_STYLE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")


def load_html(path: Path | str) -> dict:
    p = Path(path)
    raw = p.read_text(encoding="utf-8", errors="ignore")
    raw = _SCRIPT_STYLE.sub(" ", raw)
    text = _TAG.sub(" ", raw)
    text = html.unescape(text)
    text = _WHITESPACE.sub(" ", text).strip()
    return {"source": str(p), "title": p.stem, "text": text}
