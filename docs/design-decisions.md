# ADRs

## ADR-001 — Chunks are a lakehouse table, not a black-box embedding DB
**Decision:** Source of truth is `doc_chunks` in Delta/Iceberg. Vector stores are rebuildable projections.
**Why:** Survives vector-DB migrations, embedding-model upgrades, schema changes. You can `SELECT * FROM doc_chunks WHERE source = '...'` for debugging. Standard dbt/ETL tooling works.

## ADR-002 — Hybrid retrieval baseline (BM25 + vector + RRF + rerank)
**Decision:** Never ship vector-only search.
**Why:** Every 2025-2026 paper confirms hybrid retrieval beats vector-only by 10-30% on most benchmarks. BM25 is free; reranker is cheap.

## ADR-003 — Deterministic hash embeddings in CI
**Decision:** `HashEmbeddingProvider` is the default in tests. Real providers (fastembed, OpenAI-compat, HF) only instantiate outside CI.
**Why:** Eliminates network calls, model downloads, and flakiness. Contract tests (add/query/rerank) don't care about semantic quality, only wiring correctness.

## ADR-004 — Provider-agnostic LLM + embedding interfaces
**Decision:** `EmbeddingProvider` and `LLMProvider` are Protocols. Swap Ollama → OpenAI → any compatible vendor with a config change.
**Why:** Avoids vendor lock-in; reflects how real production RAG platforms evolve (start with a managed cloud endpoint, move to self-hosted Ollama for cost).

## ADR-005 — Eval metrics are hand-rolled, not LLM-as-judge (in CI)
**Decision:** `metrics.py` implements token-overlap RAGAS-style metrics; LLM-judge variants are a swap-in.
**Why:** Token overlap is reproducible, free, zero-latency. Good enough to catch regressions. In staging, run LLM-judge on the same golden set for absolute quality.

## ADR-006 — FastAPI wires the full pipeline with in-memory defaults
**Decision:** `server.py` works out-of-the-box with no Postgres, no Ollama, no API keys.
**Why:** First-run friction kills adoption. Zero-config demo → swap components via env vars in prod.
