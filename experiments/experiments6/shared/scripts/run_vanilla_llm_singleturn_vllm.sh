#!/usr/bin/env bash
set -euo pipefail
RELEASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RELEASE_ROOT"
exec python src/baselines/vanilla_llm/vanillaLLM_inference-vllm-single.py \
  --input_path "$RELEASE_ROOT/data/1229_dev_6.json" \
  --query_path "$RELEASE_ROOT/query_singleturn.json" \
  --pref_list_path "$RELEASE_ROOT/pref_list.json" \
  --pref_group_path "$RELEASE_ROOT/pref_group.json" \
  --tools_schema_path "$RELEASE_ROOT/schema_easy.json" \
  "$@"
