# RAG Data Platform Overview

The platform treats a corpus as a durable, versioned table in the lakehouse. Document chunks live in a Delta or Iceberg table (`doc_chunks`), partitioned by `source` and `ingested_date`. Vector stores are projections of the chunks table and can be rebuilt from the source of truth at any time.

## Supported Vector Stores

- **pgvector** — primary. SQL-native, HNSW index, filter on metadata JSONB.
- **Qdrant** — production-grade, HTTP/gRPC, payload filters.
- **LanceDB** — embedded, columnar, great for notebooks.
- **InMemory** — used by tests and single-process demos.

## Retrieval

Queries go through hybrid retrieval: a vector top-K, a BM25 lexical top-K, results fused via **Reciprocal Rank Fusion**, and the top candidates reranked by a cross-encoder. Metadata filters are pushed down into the vector store.

## Why lakehouse-backed

Treating chunks as data engineering artifacts (rather than opaque embeddings in a specialty DB) gives you: schema evolution, CDC, time travel, dbt-style modelling, and painless backfills. When the embedding model is upgraded, you re-project the same source-of-truth chunks into a new vector-store collection.
