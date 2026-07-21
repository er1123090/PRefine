#!/bin/bash

set -euo pipefail

BASE_DIR="/data/minseo/experiments4"
REPO_DIR="${BASE_DIR}/e-mem"
PYTHON_BIN="${REPO_DIR}/.venv-emem/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

MANIFEST_PATH="${REPO_DIR}/output/emem_1229_dev_6.manifest.jsonl"
OUTPUT_DIR="${REPO_DIR}/inference_multi/gpt-5"
MODEL_NAME="gpt-5"
CONTEXT_TYPE="memory_only"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

for PREF_TYPE in easy medium hard; do
  "$PYTHON_BIN" "${REPO_DIR}/step2_evaluate_multiturn.py" \
    --manifest_path "$MANIFEST_PATH" \
    --input_path "${BASE_DIR}/data/1229_dev_6.json" \
    --multiturn_path "${BASE_DIR}/query_multiturn-domain.json" \
    --pref_list_path "${BASE_DIR}/pref_list.json" \
    --pref_group_path "${BASE_DIR}/pref_group.json" \
    --tools_schema_path "${BASE_DIR}/schema_all.json" \
    --pref_type "$PREF_TYPE" \
    --context_type "$CONTEXT_TYPE" \
    --model_name "$MODEL_NAME" \
    --output_path "${OUTPUT_DIR}/${DATE_TAG}_${PREF_TYPE}.json" \
    --log_path "${OUTPUT_DIR}/${DATE_TAG}_${PREF_TYPE}.jsonl" \
    --run_log_path "${OUTPUT_DIR}/${DATE_TAG}_${PREF_TYPE}.run.log"
done
