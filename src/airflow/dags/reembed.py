"""Airflow DAG: re-embed all chunks when the embedding model is upgraded."""

from __future__ import annotations

DAG_ID = "rag_reembed"
SCHEDULE = None  # manual trigger


def build_commands(new_provider: str, dim: int) -> list[str]:
    return [
        f"python -m scripts.reembed_all --provider {new_provider} --dim {dim}",
        "python -m scripts.swap_vector_collection --alias rag --to rag_v2",
        "python -m scripts.eval_snapshot --tag post-reembed",
    ]
