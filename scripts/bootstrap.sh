#!/usr/bin/env bash
set -euo pipefail

docker compose up -d postgres qdrant
echo "Waiting for postgres..."
until docker compose exec -T postgres pg_isready -U rag >/dev/null 2>&1; do sleep 1; done

docker compose exec -T postgres psql -U rag -d rag -c "CREATE EXTENSION IF NOT EXISTS vector;"
echo "[bootstrap] ready"
