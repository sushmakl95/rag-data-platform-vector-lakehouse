"""pgvector store — SQL-first implementation.

Queries are constructed as SQLAlchemy `text()` strings so they're unit-testable
without a live Postgres connection. Integration against a real pgvector database
is done in the optional `tests/integration/` suite.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine

from src.rag.types import Chunk, EmbeddedChunk, RetrievalResult


@dataclass
class PgVectorConfig:
    table: str = "doc_chunks"
    dim: int = 384
    distance: str = "cosine"  # cosine | l2 | ip


_DIST_OP = {"cosine": "<=>", "l2": "<->", "ip": "<#>"}


def build_create_sql(cfg: PgVectorConfig) -> str:
    return (
        f"CREATE TABLE IF NOT EXISTS {cfg.table} ("
        "  id TEXT PRIMARY KEY,"
        "  text TEXT NOT NULL,"
        "  source TEXT NOT NULL,"
        "  metadata JSONB,"
        f"  embedding VECTOR({cfg.dim}) NOT NULL"
        ");"
    )


def build_index_sql(cfg: PgVectorConfig) -> str:
    ops = {"cosine": "vector_cosine_ops", "l2": "vector_l2_ops", "ip": "vector_ip_ops"}[
        cfg.distance
    ]
    return (
        f"CREATE INDEX IF NOT EXISTS {cfg.table}_embedding_idx "
        f"ON {cfg.table} USING hnsw (embedding {ops}) "
        "WITH (m = 16, ef_construction = 64);"
    )


def build_upsert_sql(cfg: PgVectorConfig) -> str:
    return (
        f"INSERT INTO {cfg.table} (id, text, source, metadata, embedding) "
        "VALUES (:id, :text, :source, CAST(:metadata AS JSONB), :embedding) "
        "ON CONFLICT (id) DO UPDATE SET "
        "  text = EXCLUDED.text, source = EXCLUDED.source, "
        "  metadata = EXCLUDED.metadata, embedding = EXCLUDED.embedding;"
    )


def build_query_sql(cfg: PgVectorConfig, *, has_filter: bool) -> str:
    op = _DIST_OP[cfg.distance]
    where = "WHERE metadata @> CAST(:filter AS JSONB) " if has_filter else ""
    return (
        f"SELECT id, text, source, metadata, (embedding {op} :embedding) AS distance "
        f"FROM {cfg.table} {where}"
        f"ORDER BY embedding {op} :embedding ASC "
        "LIMIT :top_k;"
    )


def build_delete_sql(cfg: PgVectorConfig) -> str:
    return f"DELETE FROM {cfg.table} WHERE id = ANY(:ids);"


class PgVectorStore:  # pragma: no cover - exercised in integration tests
    def __init__(self, engine: Engine, config: PgVectorConfig | None = None) -> None:
        self.engine = engine
        self.config = config or PgVectorConfig()

    def bootstrap(self) -> None:
        with self.engine.begin() as conn:
            conn.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.execute(sql_text(build_create_sql(self.config)))
            conn.execute(sql_text(build_index_sql(self.config)))

    def add(self, items: list[EmbeddedChunk]) -> None:
        upsert = sql_text(build_upsert_sql(self.config))
        with self.engine.begin() as conn:
            for item in items:
                conn.execute(
                    upsert,
                    {
                        "id": item.chunk.id,
                        "text": item.chunk.text,
                        "source": item.chunk.source,
                        "metadata": item.chunk.metadata,
                        "embedding": item.embedding,
                    },
                )

    def query(
        self, embedding: list[float], *, top_k: int = 5, filters: dict | None = None
    ) -> list[RetrievalResult]:
        stmt = sql_text(build_query_sql(self.config, has_filter=bool(filters)))
        params = {"embedding": embedding, "top_k": top_k}
        if filters:
            params["filter"] = filters
        with self.engine.begin() as conn:
            rows = conn.execute(stmt, params).fetchall()
        return [
            RetrievalResult(
                chunk=Chunk(
                    id=row.id, text=row.text, source=row.source, metadata=row.metadata or {}
                ),
                score=(
                    1.0 - float(row.distance)
                    if self.config.distance == "cosine"
                    else -float(row.distance)
                ),
            )
            for row in rows
        ]

    def delete(self, ids: list[str]) -> None:
        with self.engine.begin() as conn:
            conn.execute(sql_text(build_delete_sql(self.config)), {"ids": ids})

    def count(self) -> int:
        with self.engine.begin() as conn:
            row = conn.execute(
                sql_text(f"SELECT COUNT(*) AS c FROM {self.config.table}")
            ).fetchone()
            return int(row.c)
