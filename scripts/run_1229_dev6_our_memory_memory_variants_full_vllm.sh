#!/usr/bin/env bash

# End-to-end runner for the requested Our Memory ablation:
#   1. Build four memory variants.
#   2. Run memory_api inference over easy/medium/hard for singleturn/multiturn.
#
# Both steps use the open-source model set excluding meta-llama.

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/data/minseo/experiments5}"
RUN_ID="${RUN_ID:-1229_dev6_memory_variants_20260517}"
PORT="${PORT:-8004}"
WAIT_FOR_EXISTING_1229_RUN="${WAIT_FOR_EXISTING_1229_RUN:-1}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-300}"
RUN_USER="${RUN_USER:-$(id -un)}"

MEMORY_VARIANT_SCRIPT="$ROOT_DIR/scripts/run_1229_dev6_our_memory_memory_variants_vllm.sh"
INFERENCE_SCRIPT="$ROOT_DIR/scripts/run_1229_dev6_our_memory_memory_variant_inference_vllm.sh"
LOG_ROOT="$ROOT_DIR/logs/our_memory/$RUN_ID"
RUN_LOG="$LOG_ROOT/full.run.log"

NON_LLAMA_MODELS=(
  "Qwen/Qwen3-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "google/gemma-3-12b-it"
  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
)

MEMORY_MODES=(
  "generation_only"
  "blind_refine_1"
  "blind_refine_2"
  "blind_refine_3"
)

join_by_space() {
  local IFS=" "
  printf '%s' "$*"
}

log() {
  mkdir -p "$LOG_ROOT"
  printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$RUN_LOG"
}

wait_for_existing_main_run() {
  [[ "$WAIT_FOR_EXISTING_1229_RUN" == "1" ]] || return 0

  while pgrep -a -u "$RUN_USER" -f "$ROOT_DIR/scripts/run_1229_dev6_our_memory_vllm.sh" >/dev/null 2>&1; do
    log "Waiting for existing 1229_dev6 main run to finish."
    pgrep -a -u "$RUN_USER" -f "$ROOT_DIR/scripts/run_1229_dev6_our_memory_vllm.sh" | tee -a "$RUN_LOG"
    sleep "$WAIT_INTERVAL_SECONDS"
  done

  while pgrep -a -u "$RUN_USER" -f 'vllm serve' >/dev/null 2>&1; do
    log "Waiting for existing vLLM server to stop."
    pgrep -a -u "$RUN_USER" -f 'vllm serve' | tee -a "$RUN_LOG"
    sleep "$WAIT_INTERVAL_SECONDS"
  done
}

main() {
  cd "$ROOT_DIR" || exit 1
  mkdir -p "$LOG_ROOT"

  local model_list mode_list
  model_list="$(join_by_space "${NON_LLAMA_MODELS[@]}")"
  mode_list="$(join_by_space "${MEMORY_MODES[@]}")"

  log "Full memory-variant run queued."
  log "Run ID: $RUN_ID"
  log "Models excluding meta-llama: $model_list"
  log "Memory modes: $mode_list"

  wait_for_existing_main_run

  log "Starting memory variant generation."
  RUN_ID="$RUN_ID" \
  PORT="$PORT" \
  ONLY_MEMORY_MODELS="$model_list" \
  ONLY_MEMORY_MODES="$mode_list" \
  bash "$MEMORY_VARIANT_SCRIPT"

  log "Starting memory variant inference."
  RUN_ID="$RUN_ID" \
  PORT="$PORT" \
  ONLY_MEMORY_MODELS="$model_list" \
  ONLY_MODELS="$model_list" \
  ONLY_MEMORY_MODES="$mode_list" \
  WAIT_FOR_MEMORIES=1 \
  bash "$INFERENCE_SCRIPT"

  log "Full memory-variant run complete."
}

main "$@"
