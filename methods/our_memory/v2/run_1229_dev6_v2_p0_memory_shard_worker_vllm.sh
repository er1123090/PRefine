#!/usr/bin/env bash

# Run one ours_memory_v2 P0 memory shard on one GPU and one vLLM port.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
RUN_ID="${RUN_ID:-1229_dev6_ours_memory_v2_p0_20260520}"
DATA_PATH="${DATA_PATH:?DATA_PATH is required}"
MEMORY_MODEL="${MEMORY_MODEL:?MEMORY_MODEL is required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"
GPU_IDS="${GPU_IDS:?GPU_IDS is required}"
PORT="${PORT:?PORT is required}"
TP_SIZE="${TP_SIZE:-1}"
VLLM_URL="${VLLM_URL:-http://localhost:$PORT/v1}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_RETRIES="${MAX_RETRIES:-3}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
SERVER_WAIT_RETRIES="${SERVER_WAIT_RETRIES:-120}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:---disable-custom-all-reduce}"
SHARD_ID="${SHARD_ID:-0}"

LOG_ROOT="$ROOT_DIR/logs/our_memory/$RUN_ID/memory_shards"
LOG_FILE="$LOG_ROOT/shard_${SHARD_ID}.log"
STEP1_SCRIPT="$SCRIPT_DIR/step1_extract.py"
MEMORY_FILE="$OUTPUT_DIR/_memory1.jsonl"
VERIFIER_FILE="$OUTPUT_DIR/_verifier_logs1.jsonl"
REFINEMENT_FILE="$OUTPUT_DIR/_refinement_logs1.jsonl"
SERVER_PID=""

log() {
  mkdir -p "$LOG_ROOT"
  printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$LOG_FILE"
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

wait_for_server() {
  local count=0
  log "Waiting for vLLM server at $VLLM_URL"
  while ! curl -fsS --connect-timeout 2 -m 5 "$VLLM_URL/models" >/dev/null 2>&1; do
    sleep 5
    count=$((count + 1))
    if [[ -n "${SERVER_PID:-}" ]] && ! ps -p "$SERVER_PID" >/dev/null 2>&1; then
      log "vLLM server died before readiness"
      return 1
    fi
    if [[ "$count" -ge "$SERVER_WAIT_RETRIES" ]]; then
      log "Timed out waiting for vLLM server"
      return 1
    fi
  done
  log "vLLM server is ready"
}

start_server() {
  local parser_flag
  parser_flag="$(parser_flag_for_model "$MEMORY_MODEL")"
  mkdir -p "$LOG_ROOT"
  log "Starting vLLM server model=$MEMORY_MODEL gpu=$GPU_IDS port=$PORT"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$GPU_IDS" nohup vllm serve "$MEMORY_MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --tensor-parallel-size "$TP_SIZE" \
    --enable-auto-tool-choice \
    $parser_flag \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    $VLLM_EXTRA_ARGS > "$LOG_ROOT/vllm_shard_${SHARD_ID}.log" 2>&1 &
  SERVER_PID=$!
  log "vLLM server PID=$SERVER_PID"
  wait_for_server
}

main() {
  cd "$ROOT_DIR" || exit 1
  mkdir -p "$OUTPUT_DIR" "$LOG_ROOT"

  start_server || exit 1
  log "Running shard memory extraction model=$MEMORY_MODEL shard=$SHARD_ID data=$DATA_PATH"
  python "$STEP1_SCRIPT" \
    --input "$DATA_PATH" \
    --output "$MEMORY_FILE" \
    --verifier_output "$VERIFIER_FILE" \
    --refinement_output "$REFINEMENT_FILE" \
    --provider openai \
    --model "$MEMORY_MODEL" \
    --api_base "$VLLM_URL" \
    --api_key EMPTY \
    --concurrency "$CONCURRENCY" \
    --max_retries "$MAX_RETRIES" \
    --memory_mode verified_refine_v2 2>&1 | tee -a "$LOG_FILE"
  log "Shard memory extraction finished model=$MEMORY_MODEL shard=$SHARD_ID"
}

main "$@"
