#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RELEASE_ROOT"
exec python src/evaluation/evaluation_multiturn-f1-parse-aggregate.py \
  --pref_list_path "$RELEASE_ROOT/pref_list.json" \
  "$@"
