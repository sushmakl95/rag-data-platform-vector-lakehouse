"""PDF loader — lazy-imports pypdf so CI doesn't need the package."""

from __future__ import annotations

from pathlib import Path


def load_pdf(path: Path | str) -> dict:  # pragma: no cover - needs pypdf
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("pypdf not installed — `pip install pypdf`") from e
    p = Path(path)
    reader = PdfReader(str(p))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)
    return {"source": str(p), "title": p.stem, "text": text}
