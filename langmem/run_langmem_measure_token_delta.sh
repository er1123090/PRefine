#!/bin/bash

set -euo pipefail

PYTHON_SCRIPT="/data/minseo/experiments4/langmem/measure_session_memory_token_delta.py"
BASE_DIR="/data/minseo/experiments4"
OUTPUT_DIR="${BASE_DIR}/langmem/token_growth/gpt-4o-mini"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"

mkdir -p "$OUTPUT_DIR"

python "$PYTHON_SCRIPT" \
  --input_path "${BASE_DIR}/data/1229_dev_6.json" \
  --output_csv "${OUTPUT_DIR}/session_memory_token_deltas_${DATE_TAG}.csv" \
  --summary_json "${OUTPUT_DIR}/session_memory_token_deltas_${DATE_TAG}.summary.json" \
  --run_log_path "${OUTPUT_DIR}/session_memory_token_deltas_${DATE_TAG}.run.log" \
  --memory_model "gpt-4o-mini"
