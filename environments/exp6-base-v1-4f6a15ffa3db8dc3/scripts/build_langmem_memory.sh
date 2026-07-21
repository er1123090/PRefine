#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RELEASE_ROOT"
exec python src/baselines/langmem/step1_build_memory.py \
  --input_path "$RELEASE_ROOT/data/1229_dev_6.json" \
  --output_path "$RELEASE_ROOT/runs/langmem/langmem_1229_dev_6.jsonl" \
  --manifest_path "$RELEASE_ROOT/runs/langmem/langmem_1229_dev_6.manifest.json" \
  "$@"
