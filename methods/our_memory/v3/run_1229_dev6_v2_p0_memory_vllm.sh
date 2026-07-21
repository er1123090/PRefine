#!/usr/bin/env bash

# Build ours_memory_v2 P0 memories for the 1229_dev_6 split.
# This script stays inside methods/our_memory/v2 and only calls v2 code.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
RUN_ID="${RUN_ID:-1229_dev6_ours_memory_v2_p0_20260520}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/1229_dev_6.json}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-8004}"
VLLM_URL="${VLLM_URL:-http://localhost:$PORT/v1}"
CONCURRENCY="${CONCURRENCY:-50}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
SERVER_WAIT_RETRIES="${SERVER_WAIT_RETRIES:-120}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---disable-custom-all-reduce}"
EXPECTED_MEMORY_ROWS="${EXPECTED_MEMORY_ROWS:-265}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_USER="${RUN_USER:-$(id -un)}"
ALLOW_CONCURRENT_VLLM="${ALLOW_CONCURRENT_VLLM:-0}"

RUN_ROOT="$ROOT_DIR/outputs/our_memory/$RUN_ID"
MEMORY_ROOT="$RUN_ROOT/memories"
LOG_ROOT="$ROOT_DIR/logs/our_memory/$RUN_ID"
RUN_LOG="$LOG_ROOT/memory.run.log"
STEP1_SCRIPT="$SCRIPT_DIR/step1_extract.py"

MEMORY_SOURCE_MODELS=(
  "Qwen/Qwen3-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "google/gemma-3-12b-it"
  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
)

if [[ -n "${ONLY_MEMORY_MODELS:-}" ]]; then
  read -r -a MEMORY_SOURCE_MODELS <<< "$ONLY_MEMORY_MODELS"
fi

SERVER_PID=""

log() {
  mkdir -p "$LOG_ROOT"
  printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$RUN_LOG"
}

model_safe() {
  printf '%s' "${1//\//_}"
}

parser_flag_for_model() {
  local model="$1"
  if [[ "$model" == *"Llama-3"* ]]; then
    printf '%s' "--tool-call-parser llama3_json"
  elif [[ "$model" == *"Mistral"* ]]; then
    printf '%s' "--tool-call-parser mistral"
  else
    printf '%s' "--tool-call-parser hermes"
  fi
}

pid_tree() {
  local pid="$1"
  ps -p "$pid" >/dev/null 2>&1 || return 0
  printf '%s\n' "$pid"
  local child
  while read -r child; do
    [[ -n "$child" ]] && pid_tree "$child"
  done < <(pgrep -P "$pid" 2>/dev/null || true)
}

server_tree_alive() {
  [[ -n "${SERVER_PID:-}" ]] && pid_tree "$SERVER_PID" | grep -q .
}

signal_server_tree() {
  local signal="$1"
  [[ -n "${SERVER_PID:-}" ]] || return 0
  mapfile -t pids < <(pid_tree "$SERVER_PID" | tac)
  [[ "${#pids[@]}" -gt 0 ]] || return 0
  kill "-$signal" "${pids[@]}" >/dev/null 2>&1 || true
}

cleanup_server() {
  if [[ -n "${SERVER_PID:-}" ]] && ps -p "$SERVER_PID" >/dev/null 2>&1; then
    log "Stopping vLLM server PID=$SERVER_PID"
    signal_server_tree TERM
    for _ in $(seq 1 30); do
      server_tree_alive || break
      sleep 1
    done
    if server_tree_alive; then
      log "Force stopping vLLM server tree PID=$SERVER_PID"
      signal_server_tree KILL
      sleep 2
    fi
    wait "$SERVER_PID" >/dev/null 2>&1 || true
  fi
  SERVER_PID=""
}

trap cleanup_server EXIT INT TERM

wait_for_server() {
  local retries="$SERVER_WAIT_RETRIES"
  local count=0
  log "Waiting for vLLM server at $VLLM_URL"
  while ! curl -fsS --connect-timeout 2 -m 5 "$VLLM_URL/models" >/dev/null 2>&1; do
    sleep 5
    count=$((count + 1))
    if [[ -n "${SERVER_PID:-}" ]] && ! ps -p "$SERVER_PID" >/dev/null 2>&1; then
      log "vLLM server died before readiness. See $LOG_ROOT/vllm_server.log"
      return 1
    fi
    if [[ "$count" -ge "$retries" ]]; then
      log "Timed out waiting for vLLM server"
      return 1
    fi
  done
  log "vLLM server is ready"
}

start_server() {
  local model="$1"
  local parser_flag
  parser_flag="$(parser_flag_for_model "$model")"

  cleanup_server
  mkdir -p "$LOG_ROOT"
  log "Starting vLLM server for $model on GPUs $GPU_IDS"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU_IDS" nohup vllm serve "$model" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --enable-auto-tool-choice \
    $parser_flag \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    $VLLM_EXTRA_ARGS > "$LOG_ROOT/vllm_server.log" 2>&1 &
  SERVER_PID=$!
  log "vLLM server PID=$SERVER_PID"
  wait_for_server
}

memory_complete() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  local rows
  rows="$(wc -l < "$path" | tr -d ' ')"
  [[ "$rows" -ge "$EXPECTED_MEMORY_ROWS" ]]
}

guard_no_other_vllm() {
  [[ "$ALLOW_CONCURRENT_VLLM" == "1" ]] && return 0
  if ps -u "$RUN_USER" -o cmd= | rg -q '[v]llm serve'; then
    log "Another vLLM server is already running. Refusing to start a second server."
    ps -u "$RUN_USER" -o pid,cmd | rg '[v]llm serve' | tee -a "$RUN_LOG"
    return 1
  fi
}

run_memory() {
  local model="$1"
  local safe
  safe="$(model_safe "$model")"
  local out_dir="$MEMORY_ROOT/$safe"
  local memory_file="$out_dir/_memory1.jsonl"
  local verifier_file="$out_dir/_verifier_logs1.jsonl"
  local refinement_file="$out_dir/_refinement_logs1.jsonl"

  mkdir -p "$out_dir"
  if [[ "$FORCE_RERUN" != "1" ]] && memory_complete "$memory_file"; then
    log "v2 memory exists, skipping: model=$model file=$memory_file"
    return 0
  fi

  log "Running ours_memory_v2 P0 memory: model=$model"
  python "$STEP1_SCRIPT" \
    --input "$DATA_PATH" \
    --output "$memory_file" \
    --verifier_output "$verifier_file" \
    --refinement_output "$refinement_file" \
    --provider openai \
    --model "$model" \
    --api_base "$VLLM_URL" \
    --api_key EMPTY \
    --concurrency "$CONCURRENCY" \
    --max_retries "$MAX_RETRIES" \
    --memory_mode verified_refine_v2 2>&1 | tee -a "$RUN_LOG"

  if [[ -f "$memory_file" ]]; then
    ln -sf "_memory1.jsonl" "$out_dir/memory_result.jsonl"
  fi
}

main() {
  cd "$ROOT_DIR" || exit 1
  mkdir -p "$MEMORY_ROOT" "$LOG_ROOT"

  guard_no_other_vllm || exit 1

  log "Run ID: $RUN_ID"
  log "Data path: $DATA_PATH"
  log "Memory source models: ${MEMORY_SOURCE_MODELS[*]}"
  log "Concurrency: $CONCURRENCY"
  log "Max retries: $MAX_RETRIES"
  [[ -z "$VLLM_EXTRA_ARGS" ]] || log "vLLM extra args: $VLLM_EXTRA_ARGS"

  local model
  for model in "${MEMORY_SOURCE_MODELS[@]}"; do
    if ! start_server "$model"; then
      log "Skipping memory after server startup failure: $model"
      cleanup_server
      continue
    fi
    run_memory "$model"
    cleanup_server
    sleep 10
  done

  log "ours_memory_v2 P0 memory run finished"
  log "Outputs: $RUN_ROOT"
}

main "$@"
