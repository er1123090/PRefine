#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RELEASE_ROOT"
exec python src/baselines/rag/RAG_add.py \
  --input_path "$RELEASE_ROOT/data/1229_dev_6.json" \
  --db_path "$RELEASE_ROOT/runs/rag/chroma_db" \
  "$@"
