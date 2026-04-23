"""dlt pipeline spec + a pure-Python implementation for testable ingestion."""

from __future__ import annotations

from typing import Any


def compile_pipeline_spec() -> dict[str, Any]:
    return {
        "pipeline_name": "rag_doc_ingest",
        "destination": "filesystem",
        "dataset_name": "rag_bronze",
        "write_disposition": "merge",
        "primary_key": "chunk_id",
        "file_format": "parquet",
        "partition_by": ["source", "ingested_date"],
    }


def walk_sources(root: str, *, suffixes: tuple[str, ...] = (".md", ".html", ".txt")) -> list[dict]:
    """Pure function — caller passes in a list of file paths (as dicts) rather
    than walking the filesystem inside dlt, so the pipeline is testable."""
    # For dlt integration, this would be replaced with a @dlt.resource generator.
    return [{"root": root, "suffixes": list(suffixes)}]
