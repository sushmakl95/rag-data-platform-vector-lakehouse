#!/usr/bin/env bash
set -euo pipefail

for f in data/samples/*.md; do
  echo "[seed] $f"
  text=$(cat "$f")
  curl -s -X POST http://localhost:8000/ingest \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg s "$f" --arg t "$text" '{source:$s, text:$t}')"
  echo
done
