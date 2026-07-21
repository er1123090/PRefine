#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RELEASE_ROOT"
exec python src/baselines/mem0/step2_evaluate_multiturn.py \
  --input_path "$RELEASE_ROOT/data/1229_dev_6.json" \
  --multiturn_path "$RELEASE_ROOT/query_multiturn-domain.json" \
  --pref_list_path "$RELEASE_ROOT/pref_list.json" \
  --pref_group_path "$RELEASE_ROOT/pref_group.json" \
  --tools_schema_path "$RELEASE_ROOT/schema_all.json" \
  "$@"
