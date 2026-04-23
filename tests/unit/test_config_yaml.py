from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def test_rag_config_has_required_sections():
    cfg = yaml.safe_load((REPO / "configs" / "rag_config.yaml").read_text())
    assert {"embedder", "chunker", "vector_store", "retriever", "llm", "eval"}.issubset(cfg.keys())


def test_golden_set_yaml_parseable_and_nonempty():
    g = yaml.safe_load((REPO / "configs" / "golden_set.yaml").read_text())
    assert len(g["cases"]) >= 3
