"""Airflow DAG: document ingestion into doc_chunks (Delta) + vector sync.

Uses `airflow` lazily so the module imports cleanly when airflow isn't present.
Structure mirrors the Astronomer-convention layout from the sibling Airflow repo.
"""

from __future__ import annotations

DAG_ID = "rag_ingest_docs"
SCHEDULE = "@hourly"
DEFAULT_ARGS = {"owner": "data-platform", "retries": 2}

TASKS_ORDER = [
    "discover_new_docs",
    "load_raw",
    "chunk",
    "embed",
    "write_delta",
    "sync_to_pgvector",
    "run_eval_snapshot",
]


def build_commands(docs_prefix: str) -> list[str]:
    """Return the shell-equivalent pipeline for offline review."""
    return [
        f"python -m scripts.discover --prefix {docs_prefix}",
        "python -m scripts.load",
        "python -m scripts.chunk  --strategy recursive --size 800",
        "python -m scripts.embed --provider fastembed",
        "python -m scripts.write_delta --target doc_chunks",
        "python -m scripts.sync_pgvector --table doc_chunks",
        "python -m scripts.eval_snapshot",
    ]
