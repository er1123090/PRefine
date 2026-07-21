#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RELEASE_ROOT"
exec python src/self_refine/self_refine_preference_to_api.py \
  --input "$RELEASE_ROOT/data/1229_dev_6.json" \
  --schema "$RELEASE_ROOT/schema_all.json" \
  "$@"
