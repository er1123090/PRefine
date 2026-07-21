#!/bin/bash

set -euo pipefail

PYTHON_SCRIPT="/data/minseo/experiments5/methods/langmem/step1_build_memory.py"
BASE_DIR="/data/minseo/experiments5"
INPUT_PATH="${INPUT_PATH:-${BASE_DIR}/data/1229_dev_6.json}"
MEMORY_MODEL="${MEMORY_MODEL:-gpt-4o-mini}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-text-embedding-3-small}"
CONCURRENCY="${CONCURRENCY:-5}"
DATE_TAG="$(date +%Y%m%d_%H%M%S)"

#
# LangMem build presets:
# - auto: keep the existing auto backend routing
# - structured_output: force structured-output extraction
# - langmem_custom_semantic: LangMem backend + custom prompt + structured semantic memories
# - langmem_default_string: LangMem backend + default LangMem prompt + plain string memories
#
LANGMEM_BUILD_MODE="${LANGMEM_BUILD_MODE:-langmem_default_string}"
case "${LANGMEM_BUILD_MODE}" in
    auto)
        DEFAULT_MEMORY_BACKEND="auto"
        DEFAULT_LANGMEM_MEMORY_SCHEMA="semantic"
        DEFAULT_LANGMEM_PROMPT_STYLE="custom"
        DEFAULT_OUTPUT_SUBDIR="semantic-custom"
        ;;
    structured_output)
        DEFAULT_MEMORY_BACKEND="structured_output"
        DEFAULT_LANGMEM_MEMORY_SCHEMA="semantic"
        DEFAULT_LANGMEM_PROMPT_STYLE="custom"
        DEFAULT_OUTPUT_SUBDIR="semantic-custom"
        ;;
    langmem_custom_semantic)
        DEFAULT_MEMORY_BACKEND="langmem"
        DEFAULT_LANGMEM_MEMORY_SCHEMA="semantic"
        DEFAULT_LANGMEM_PROMPT_STYLE="custom"
        DEFAULT_OUTPUT_SUBDIR="semantic-custom"
        ;;
    langmem_default_string)
        DEFAULT_MEMORY_BACKEND="langmem"
        DEFAULT_LANGMEM_MEMORY_SCHEMA="string"
        DEFAULT_LANGMEM_PROMPT_STYLE="default"
        DEFAULT_OUTPUT_SUBDIR="string-default"
        ;;
    *)
        echo "[ERROR] Unsupported LANGMEM_BUILD_MODE='${LANGMEM_BUILD_MODE}'."
        echo "        Choose one of: auto, structured_output, langmem_custom_semantic, langmem_default_string"
        exit 1
        ;;
esac

MEMORY_BACKEND="${MEMORY_BACKEND:-${DEFAULT_MEMORY_BACKEND}}"
LANGMEM_MEMORY_SCHEMA="${LANGMEM_MEMORY_SCHEMA:-${DEFAULT_LANGMEM_MEMORY_SCHEMA}}"
LANGMEM_PROMPT_STYLE="${LANGMEM_PROMPT_STYLE:-${DEFAULT_LANGMEM_PROMPT_STYLE}}"
OUTPUT_SUBDIR="${OUTPUT_SUBDIR:-${DEFAULT_OUTPUT_SUBDIR}}"
OUTPUT_DIR="${OUTPUT_DIR:-${BASE_DIR}/methods/langmem/memory_snapshots/${OUTPUT_SUBDIR}/${MEMORY_MODEL}}"
RUN_LOG_PATH="${RUN_LOG_PATH:-${OUTPUT_DIR}/langmem_1229_dev_6_${DATE_TAG}.run.log}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/langmem_1229_dev_6.jsonl}"
MANIFEST_PATH="${MANIFEST_PATH:-${OUTPUT_DIR}/langmem_1229_dev_6.manifest.json}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "[ERROR] OPENAI_API_KEY is not set."
    echo "        Export OPENAI_API_KEY and rerun this script."
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "========================================================"
echo "LangMem API Memory Build Started at $(date)"
echo "Memory model: ${MEMORY_MODEL}"
echo "Embedding model: ${EMBEDDING_MODEL}"
echo "LangMem build mode: ${LANGMEM_BUILD_MODE}"
echo "Memory backend: ${MEMORY_BACKEND}"
echo "LangMem memory schema: ${LANGMEM_MEMORY_SCHEMA}"
echo "LangMem prompt style: ${LANGMEM_PROMPT_STYLE}"
echo "Output dir: ${OUTPUT_DIR}"
echo "========================================================"

python "${PYTHON_SCRIPT}" \
  --input_path "${INPUT_PATH}" \
  --output_path "${OUTPUT_PATH}" \
  --manifest_path "${MANIFEST_PATH}" \
  --run_log_path "${RUN_LOG_PATH}" \
  --memory_model "${MEMORY_MODEL}" \
  --embedding_model "${EMBEDDING_MODEL}" \
  --concurrency "${CONCURRENCY}" \
  --memory_backend "${MEMORY_BACKEND}" \
  --langmem_memory_schema "${LANGMEM_MEMORY_SCHEMA}" \
  --langmem_prompt_style "${LANGMEM_PROMPT_STYLE}"
