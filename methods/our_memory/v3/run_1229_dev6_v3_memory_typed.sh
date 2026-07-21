#!/usr/bin/env bash

# Build ours_memory_v3 typed-slot memories for the 1229_dev_6 split.
# v3 memory is deterministic evidence extraction from historical preference
# slots, but outputs are laid out by memory source model for combo evaluation.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
RUN_ID="${RUN_ID:-1229_dev6_ours_memory_v3_typed_20260523}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/1229_dev_6.json}"
CONCURRENCY="${CONCURRENCY:-64}"
EXPECTED_MEMORY_ROWS="${EXPECTED_MEMORY_ROWS:-265}"
FORCE_RERUN="${FORCE_RERUN:-0}"

RUN_ROOT="$ROOT_DIR/outputs/our_memory/$RUN_ID"
MEMORY_ROOT="$RUN_ROOT/memories"
LOG_ROOT="$ROOT_DIR/logs/our_memory/$RUN_ID"
RUN_LOG="$LOG_ROOT/memory.typed.run.log"
STEP1_SCRIPT="$SCRIPT_DIR/step1_extract.py"
PREF_LIST="$ROOT_DIR/config/pref_list.json"

MEMORY_SOURCE_MODELS=(
  "google/gemma-3-12b-it"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

if [[ -n "${ONLY_MEMORY_MODELS:-}" ]]; then
  read -r -a MEMORY_SOURCE_MODELS <<< "$ONLY_MEMORY_MODELS"
fi

log() {
  mkdir -p "$LOG_ROOT"
  printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$RUN_LOG"
}

model_safe() {
  printf '%s' "${1//\//_}"
}

memory_complete() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  local rows
  rows="$(wc -l < "$path" | tr -d ' ')"
  [[ "$rows" -ge "$EXPECTED_MEMORY_ROWS" ]]
}

run_memory() {
  local model="$1"
  local safe out_dir memory_file verifier_file refinement_file
  safe="$(model_safe "$model")"
  out_dir="$MEMORY_ROOT/$safe"
  memory_file="$out_dir/_memory1.jsonl"
  verifier_file="$out_dir/_verifier_logs1.jsonl"
  refinement_file="$out_dir/_refinement_logs1.jsonl"

  mkdir -p "$out_dir"
  if [[ "$FORCE_RERUN" != "1" ]] && memory_complete "$memory_file"; then
    log "v3 typed memory exists, skipping: model=$model file=$memory_file"
    return 0
  fi

  log "Running ours_memory_v3 typed memory: model_label=$model"
  python "$STEP1_SCRIPT" \
    --input "$DATA_PATH" \
    --output "$memory_file" \
    --verifier_output "$verifier_file" \
    --refinement_output "$refinement_file" \
    --provider openai \
    --model "$model" \
    --api_base "http://127.0.0.1:9/v1" \
    --api_key EMPTY \
    --concurrency "$CONCURRENCY" \
    --pref_list_path "$PREF_LIST" \
    --memory_mode typed_slots_v3 2>&1 | tee -a "$RUN_LOG"

  if [[ -f "$memory_file" ]]; then
    ln -sf "_memory1.jsonl" "$out_dir/memory_result.jsonl"
  fi
}

main() {
  cd "$ROOT_DIR" || exit 1
  mkdir -p "$MEMORY_ROOT" "$LOG_ROOT"
  log "Run ID: $RUN_ID"
  log "Data path: $DATA_PATH"
  log "Memory source labels: ${MEMORY_SOURCE_MODELS[*]}"

  local model
  for model in "${MEMORY_SOURCE_MODELS[@]}"; do
    run_memory "$model" || exit 1
  done

  log "ours_memory_v3 typed memory run finished"
  log "Outputs: $RUN_ROOT"
}

main "$@"
