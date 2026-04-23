from __future__ import annotations

import pytest

from src.rag.vector_stores.pgvector_store import (
    PgVectorConfig,
    build_create_sql,
    build_delete_sql,
    build_index_sql,
    build_query_sql,
    build_upsert_sql,
)


def test_create_includes_vector_column():
    cfg = PgVectorConfig(dim=768)
    sql = build_create_sql(cfg)
    assert "VECTOR(768)" in sql
    assert "id TEXT PRIMARY KEY" in sql


def test_index_uses_cosine_ops_by_default():
    sql = build_index_sql(PgVectorConfig())
    assert "vector_cosine_ops" in sql
    assert "USING hnsw" in sql


def test_index_switches_for_l2():
    sql = build_index_sql(PgVectorConfig(distance="l2"))
    assert "vector_l2_ops" in sql


def test_upsert_uses_on_conflict():
    sql = build_upsert_sql(PgVectorConfig())
    assert "ON CONFLICT (id) DO UPDATE" in sql


@pytest.mark.parametrize(
    "has_filter,should_contain", [(True, "metadata @> CAST"), (False, "FROM doc_chunks")]
)
def test_query_sql_includes_filter_branch(has_filter, should_contain):
    sql = build_query_sql(PgVectorConfig(), has_filter=has_filter)
    assert should_contain in sql
    assert "ORDER BY embedding" in sql
    assert "LIMIT :top_k" in sql


def test_delete_by_ids():
    assert build_delete_sql(PgVectorConfig()) == "DELETE FROM doc_chunks WHERE id = ANY(:ids);"


@pytest.mark.parametrize("dist", ["cosine", "l2", "ip"])
def test_query_uses_correct_operator(dist):
    sql = build_query_sql(PgVectorConfig(distance=dist), has_filter=False)
    expected_op = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}[dist]
    assert expected_op in sql
