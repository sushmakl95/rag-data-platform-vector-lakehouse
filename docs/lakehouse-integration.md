# Lakehouse Integration

## `doc_chunks` schema

| Column | Type | Notes |
|---|---|---|
| chunk_id | STRING | primary key, derived from source + index + content hash |
| source | STRING | partition column |
| ingested_date | DATE | partition column |
| text | STRING | chunk body |
| metadata | MAP<STRING, STRING> | loader-provided metadata (title, page, section) |
| embedding | ARRAY<FLOAT> | current-generation embedding |
| embedding_dim | INT | sanity check — filter mismatched rows |

## CDF-driven vector sync

```sql
ALTER TABLE doc_chunks SET TBLPROPERTIES (delta.enableChangeDataFeed = true);
```

The sync job reads `table_changes('doc_chunks', <last_version>)` and applies the
diff to the vector store via `apply_cdc()` in `src/lakehouse/sync.py`.

## Backfilling after an embedding-model upgrade

1. Freeze writes to `doc_chunks`.
2. Read latest snapshot as a batch.
3. Compute new embeddings with the new model.
4. Write to a **new** vector store collection (e.g. `rag_v2`).
5. Run eval; if acceptable, swap the collection alias.
6. Old collection can be dropped after 7 days.

## dbt exposure

Mart models in a downstream dbt project reference `doc_chunks` as a source:

```yaml
sources:
  - name: rag
    schema: rag_bronze
    tables:
      - name: doc_chunks
        freshness:
          warn_after: {count: 4, period: hour}
          error_after: {count: 12, period: hour}
        loaded_at_field: ingested_at
```
