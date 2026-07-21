#!/usr/bin/env bash

# Run accumulated-preference-history inference locally while freesia serves
# one vLLM inference model at a time.

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/data/minseo/experiments5}"
E4_ROOT="${E4_ROOT:-/data/minseo/experiments4}"
RUN_ID="${RUN_ID:-1229_dev6_accum_history_e4_ours_memory_inference_20260602}"
SOURCE_MEMORY_RUN_ID="${SOURCE_MEMORY_RUN_ID:-1229_dev6_accum_history_memory_20260602}"
MEMORY_MODE="${MEMORY_MODE:-accumulated_preference_history}"

REMOTE_HOST="${REMOTE_HOST:-166.104.110.47}"
SSH_PORT="${SSH_PORT:-14233}"
SSH_USER="${SSH_USER:-minseo}"
REMOTE_GPU_IDS="${REMOTE_GPU_IDS:-0,1}"
REMOTE_TP_SIZE="${REMOTE_TP_SIZE:-2}"
REMOTE_PORT="${REMOTE_PORT:-8004}"
LOCAL_PORT="${LOCAL_PORT:-18007}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:${LOCAL_PORT}/v1}"
REMOTE_VLLM_BIN="${REMOTE_VLLM_BIN:-/data/minseo/.venvs/vllm/bin/vllm}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/data/minseo/.cache/huggingface}"
REMOTE_MAX_MODEL_LEN="${REMOTE_MAX_MODEL_LEN:-16384}"
REMOTE_GPU_MEMORY_UTILIZATION="${REMOTE_GPU_MEMORY_UTILIZATION:-0.82}"
REMOTE_MAX_NUM_SEQS="${REMOTE_MAX_NUM_SEQS:-64}"
SERVER_WAIT_RETRIES="${SERVER_WAIT_RETRIES:-720}"
REMOTE_CTL_TIMEOUT="${REMOTE_CTL_TIMEOUT:-180}"

CONCURRENCY_SINGLE="${CONCURRENCY_SINGLE:-32}"
CONCURRENCY_MULTI="${CONCURRENCY_MULTI:-48}"
EXPECTED_MEMORY_ROWS="${EXPECTED_MEMORY_ROWS:-265}"

LOG_ROOT="$ROOT_DIR/logs/our_memory/$RUN_ID"
ORCH_LOG="$LOG_ROOT/freesia23_orchestrator.log"
TUNNEL_LOG="$LOG_ROOT/freesia23_tunnel.log"
REMOTE_LOG_DIR="/data/minseo/.logs/experiments5_accum_history_freesia23_$RUN_ID"
REMOTE_LOG_FILE="$REMOTE_LOG_DIR/vllm.log"
REMOTE_PID_FILE="$REMOTE_LOG_DIR/vllm.pid"

REMOTE_CTL="$ROOT_DIR/scripts/remote_pid_vllm_ctl.py"
TUNNEL_SCRIPT="$ROOT_DIR/scripts/ssh_vllm_tunnel.py"
LOCAL_RUNNER="$ROOT_DIR/scripts/run_gen_only_accum_e4_ours_memory_inference_vllm.sh"

INFERENCE_MODELS=(
  "google/codegemma-7b-it"
  "google/gemma-3-12b-it"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

MEMORY_SOURCE_MODELS=(
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
  "google/gemma-3-12b-it"
)

if [[ -n "${ONLY_MODELS:-}" ]]; then
  read -r -a INFERENCE_MODELS <<< "$ONLY_MODELS"
fi

if [[ -n "${ONLY_MEMORY_MODELS:-}" ]]; then
  read -r -a MEMORY_SOURCE_MODELS <<< "$ONLY_MEMORY_MODELS"
fi

mkdir -p "$LOG_ROOT"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$ORCH_LOG"
}

load_hf_token() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    printf '%s' "$HF_TOKEN"
  elif [[ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]]; then
    printf '%s' "$HUGGING_FACE_HUB_TOKEN"
  elif [[ -s "$HOME/.cache/huggingface/token" ]]; then
    tr -d '\n' < "$HOME/.cache/huggingface/token"
  fi
}

emit_secrets() {
  python - <<'PY'
import json
import os

password = os.environ.get("VLLM_SSH_PASSWORD", "")
if not password:
    raise SystemExit("VLLM_SSH_PASSWORD is required")
print(json.dumps({
    "password": password,
    "hf_token": os.environ.get("FREESIA_HF_TOKEN", ""),
}))
PY
}

remote_ctl() {
  emit_secrets | python "$REMOTE_CTL" "$@" \
    --host "$REMOTE_HOST" \
    --ssh-port "$SSH_PORT" \
    --user "$SSH_USER" \
    --gpu "$REMOTE_GPU_IDS" \
    --remote-port "$REMOTE_PORT" \
    --tp-size "$REMOTE_TP_SIZE" \
    --max-model-len "${REMOTE_EFFECTIVE_MAX_MODEL_LEN:-$REMOTE_MAX_MODEL_LEN}" \
    --gpu-memory-utilization "$REMOTE_GPU_MEMORY_UTILIZATION" \
    --max-num-seqs "$REMOTE_MAX_NUM_SEQS" \
    --vllm-bin "$REMOTE_VLLM_BIN" \
    --hf-home "$REMOTE_HF_HOME" \
    --log-dir "$REMOTE_LOG_DIR" \
    --log-file "$REMOTE_LOG_FILE" \
    --pid-file "$REMOTE_PID_FILE" \
    --timeout "$REMOTE_CTL_TIMEOUT"
}

max_model_len_for_model() {
  case "$1" in
    google/codegemma-7b-it)
      printf '8192\n'
      ;;
    *)
      printf '%s\n' "$REMOTE_MAX_MODEL_LEN"
      ;;
  esac
}

local_port_busy() {
  python - "$LOCAL_PORT" <<'PY'
import socket
import sys

port = int(sys.argv[1])
s = socket.socket()
try:
    s.bind(("127.0.0.1", port))
except OSError:
    raise SystemExit(0)
finally:
    s.close()
raise SystemExit(1)
PY
}

start_tunnel() {
  if curl -fsS "$VLLM_URL/models" >/dev/null 2>&1; then
    log "Tunnel/API already reachable at $VLLM_URL"
    return 0
  fi
  if local_port_busy; then
    log "Local port $LOCAL_PORT is busy but $VLLM_URL is not reachable"
    return 1
  fi

  log "Starting tunnel $VLLM_URL -> $REMOTE_HOST:127.0.0.1:$REMOTE_PORT"
  (
    export VLLM_SSH_PASSWORD
    export REMOTE_HOST SSH_PORT SSH_USER LOCAL_PORT REMOTE_VLLM_PORT="$REMOTE_PORT"
    exec python "$TUNNEL_SCRIPT"
  ) >> "$TUNNEL_LOG" 2>&1 &
  TUNNEL_PID=$!
  echo "$TUNNEL_PID" > "$LOG_ROOT/freesia23_tunnel.pid"
  sleep 3
  if ! ps -p "$TUNNEL_PID" >/dev/null 2>&1; then
    log "Tunnel failed to stay alive. See $TUNNEL_LOG"
    return 1
  fi
  log "Tunnel PID=$TUNNEL_PID"
}

stop_tunnel() {
  local pid
  pid="$(cat "$LOG_ROOT/freesia23_tunnel.pid" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && ps -p "$pid" >/dev/null 2>&1; then
    log "Stopping tunnel PID=$pid"
    kill "$pid" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  log "Stopping freesia vLLM lane"
  remote_ctl stop >> "$ORCH_LOG" 2>&1 || true
  stop_tunnel || true
}
trap cleanup EXIT INT TERM

start_model() {
  local model="$1"
  local attempt max_model_len
  max_model_len="$(max_model_len_for_model "$model")"
  log "Starting freesia vLLM model=$model GPUs=$REMOTE_GPU_IDS TP=$REMOTE_TP_SIZE max_model_len=$max_model_len"
  for attempt in 1 2; do
    if REMOTE_EFFECTIVE_MAX_MODEL_LEN="$max_model_len" remote_ctl start --model "$model" >> "$ORCH_LOG" 2>&1; then
      if wait_for_local_server "$model"; then
        return 0
      fi
      log "Readiness check failed after start for model=$model"
    else
      log "Start attempt $attempt failed for model=$model"
      if wait_for_local_server "$model"; then
        log "Model became reachable after start command failure: $model"
        return 0
      fi
    fi
    remote_ctl status >> "$ORCH_LOG" 2>&1 || true
    sleep 20
  done
  return 1
}

wait_for_local_server() {
  local model="$1"
  local count=0
  log "Waiting for vLLM readiness model=$model at $VLLM_URL"
  while ! curl -fsS "$VLLM_URL/models" >/dev/null 2>&1; do
    sleep 5
    count=$((count + 1))
    if [[ "$count" -ge "$SERVER_WAIT_RETRIES" ]]; then
      log "Timed out waiting for vLLM readiness model=$model at $VLLM_URL"
      return 1
    fi
  done
  log "vLLM ready model=$model at $VLLM_URL"
  return 0
}

run_local_inference_for_model() {
  local model="$1"
  (
    cd "$ROOT_DIR" || exit 1
    RUN_ID="$RUN_ID" \
    SOURCE_MEMORY_RUN_ID="$SOURCE_MEMORY_RUN_ID" \
    MEMORY_MODE="$MEMORY_MODE" \
    ROOT_DIR="$ROOT_DIR" \
    E4_ROOT="$E4_ROOT" \
    EXTERNAL_VLLM=1 \
    VLLM_URL="$VLLM_URL" \
    SERVER_WAIT_RETRIES="$SERVER_WAIT_RETRIES" \
    CONCURRENCY_SINGLE="$CONCURRENCY_SINGLE" \
    CONCURRENCY_MULTI="$CONCURRENCY_MULTI" \
    VLLM_MAX_TOKENS="${VLLM_MAX_TOKENS:-}" \
    EXPECTED_MEMORY_ROWS="$EXPECTED_MEMORY_ROWS" \
    ONLY_MODELS="$model" \
    ONLY_MEMORY_MODELS="${MEMORY_SOURCE_MODELS[*]}" \
    bash "$LOCAL_RUNNER"
  )
}

main() {
  if [[ -z "${VLLM_SSH_PASSWORD:-}" ]]; then
    log "VLLM_SSH_PASSWORD is required"
    exit 1
  fi

  export FREESIA_HF_TOKEN
  FREESIA_HF_TOKEN="$(load_hf_token)"
  if [[ -z "$FREESIA_HF_TOKEN" ]]; then
    log "HF token not found; gated Gemma/CodeGemma models may fail"
  fi

  log "Run ID: $RUN_ID"
  log "Source memory run ID: $SOURCE_MEMORY_RUN_ID"
  log "Memory mode: $MEMORY_MODE"
  log "Using freesia $REMOTE_HOST GPUs=$REMOTE_GPU_IDS port=$REMOTE_PORT local=$VLLM_URL"
  log "Inference models: ${INFERENCE_MODELS[*]}"
  log "Memory source models: ${MEMORY_SOURCE_MODELS[*]}"

  start_tunnel

  local model
  for model in "${INFERENCE_MODELS[@]}"; do
    if start_model "$model"; then
      log "Running accumulated-history inference for $model"
      run_local_inference_for_model "$model"
    else
      log "Failed to start inference model=$model; continuing"
    fi
  done

  log "Done. Outputs: $ROOT_DIR/outputs/our_memory/$RUN_ID/inference/$MEMORY_MODE"
}

main "$@"
