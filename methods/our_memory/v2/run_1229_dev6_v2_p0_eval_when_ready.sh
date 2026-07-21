#!/usr/bin/env bash

# Wait for ours_memory_v2 P0 inference outputs, then run evaluation.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
RUN_ID="${RUN_ID:-1229_dev6_ours_memory_v2_p0_20260520}"
EXPECTED_RESULT_FILES="${EXPECTED_RESULT_FILES:-96}"
WAIT_INTERVAL_SECONDS="${WAIT_INTERVAL_SECONDS:-120}"
RESULT_WAIT_TIMEOUT_SECONDS="${RESULT_WAIT_TIMEOUT_SECONDS:-0}"

INFERENCE_ROOT="$ROOT_DIR/outputs/our_memory/$RUN_ID/inference"
LOG_ROOT="$ROOT_DIR/logs/our_memory/$RUN_ID"
RUN_LOG="$LOG_ROOT/eval.wait.log"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_1229_dev6_v2_p0.py"

log() {
  mkdir -p "$LOG_ROOT"
  printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$RUN_LOG"
}

result_count() {
  find "$INFERENCE_ROOT" -type f -name result.json 2>/dev/null | wc -l | tr -d ' '
}

main() {
  cd "$ROOT_DIR" || exit 1
  mkdir -p "$LOG_ROOT"

  local started count elapsed
  started="$(date +%s)"
  log "Waiting for v2 inference result files. expected=$EXPECTED_RESULT_FILES root=$INFERENCE_ROOT"
  while true; do
    count="$(result_count)"
    if [[ "$count" -ge "$EXPECTED_RESULT_FILES" ]]; then
      log "All v2 inference outputs are present. count=$count"
      break
    fi
    elapsed=$(( $(date +%s) - started ))
    if [[ "$RESULT_WAIT_TIMEOUT_SECONDS" -gt 0 && "$elapsed" -ge "$RESULT_WAIT_TIMEOUT_SECONDS" ]]; then
      log "Timed out waiting for v2 inference outputs. count=$count expected=$EXPECTED_RESULT_FILES elapsed=${elapsed}s"
      exit 1
    fi
    log "Waiting for v2 inference outputs. count=$count expected=$EXPECTED_RESULT_FILES elapsed=${elapsed}s"
    sleep "$WAIT_INTERVAL_SECONDS"
  done

  log "Starting v2 evaluation"
  python "$EVAL_SCRIPT" 2>&1 | tee -a "$RUN_LOG"
  log "v2 evaluation finished"
}

main "$@"
