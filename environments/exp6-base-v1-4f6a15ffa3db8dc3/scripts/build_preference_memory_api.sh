#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RELEASE_ROOT"
exec python src/preference_memory/Preference_Memory_step1_LATENTPREF.py \
  --input "$RELEASE_ROOT/data/1229_dev_6.json" \
  "$@"
