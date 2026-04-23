# Operations Runbook

## Re-embedding the entire corpus

When the embedding model is upgraded, the `reembed` DAG runs:

1. Load all chunks from `doc_chunks`.
2. Compute new embeddings with the new model.
3. Write into a new vector-store collection (e.g. `rag_v2`).
4. Swap the collection alias from `rag` → `rag_v2` atomically.
5. Run golden-set eval against the new collection and compare metrics.

If metrics regress by more than 5% on any threshold, the alias is automatically rolled back.

## Monitoring

- **Ingestion freshness**: alert if no new chunks in 1h for tenant-critical sources.
- **Query latency**: P99 < 400ms for `/ask`.
- **Eval regression**: weekly golden-set run; PagerDuty on >5% metric drop.
