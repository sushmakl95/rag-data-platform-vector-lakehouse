#!/usr/bin/env bash
set -euo pipefail

python - <<'PY'
from pathlib import Path
from src.rag.eval.golden_set import evaluate_case, load_golden
from src.rag.eval.metrics import aggregate
from fastapi.testclient import TestClient
from src.rag.api.server import app, STORE

client = TestClient(app)

# Seed sample docs
for p in Path("data/samples").glob("*.md"):
    client.post("/ingest", json={"source": str(p), "text": p.read_text()})

cases = load_golden("configs/golden_set.yaml")
results = []
for c in cases:
    r = client.post("/ask", json={"question": c.question, "top_k": 3})
    body = r.json()
    results.append(
        evaluate_case(
            c,
            answer=body["answer"],
            retrieved_texts=[x["text"] for x in body["contexts"]],
        )
    )

agg = aggregate(results)
print(agg)
PY
