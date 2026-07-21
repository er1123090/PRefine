#!/bin/bash

set -euo pipefail

BASE_DIR="/data/minseo/experiments4"
REPO_DIR="${BASE_DIR}/e-mem"
PYTHON_BIN="${REPO_DIR}/.venv-emem/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python"
fi

DATE_TAG="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${REPO_DIR}/output"
INDEX_ROOT="${REPO_DIR}/indexes/emem_1229_dev_6"

mkdir -p "$OUTPUT_DIR" "$INDEX_ROOT"

"$PYTHON_BIN" "${REPO_DIR}/step1_build_memory.py" \
  --input_path "${BASE_DIR}/data/1229_dev_6.json" \
  --index_root "$INDEX_ROOT" \
  --manifest_path "${OUTPUT_DIR}/emem_1229_dev_6.manifest.jsonl" \
  --llm_model "gpt-4o-mini" \
  --embedding_model "text-embedding-3-small" \
  --run_log_path "${OUTPUT_DIR}/emem_1229_dev_6_${DATE_TAG}.run.log"
