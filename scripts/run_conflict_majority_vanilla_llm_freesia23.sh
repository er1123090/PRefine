#!/usr/bin/env bash

# Run conflict-majority vanilla_llm inference from ivy while freesia serves
# one vLLM model at a time on GPUs 2,3.

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/data/minseo/experiments5}"
RUN_ID="${RUN_ID:-conflict_majority_vanilla_llm_freesia23_$(date +%Y%m%d_%H%M%S)}"
METHOD_DIR="${METHOD_DIR:-vanilla_llm}"
METHOD_NAME="${METHOD_NAME:-vanilla_llm}"

REMOTE_HOST="${REMOTE_HOST:-166.104.110.47}"
SSH_PORT="${SSH_PORT:-14233}"
SSH_USER="${SSH_USER:-minseo}"
REMOTE_GPU_IDS="${REMOTE_GPU_IDS:-2,3}"
REMOTE_TP_SIZE="${REMOTE_TP_SIZE:-2}"
REMOTE_PORT="${REMOTE_PORT:-8004}"
LOCAL_PORT="${LOCAL_PORT:-18006}"
VLLM_URL="${VLLM_URL:-http://127.0.0.1:${LOCAL_PORT}/v1}"
REMOTE_VLLM_BIN="${REMOTE_VLLM_BIN:-/data/minseo/.venvs/vllm/bin/vllm}"
REMOTE_HF_HOME="${REMOTE_HF_HOME:-/data/minseo/.cache/huggingface}"
REMOTE_MAX_MODEL_LEN="${REMOTE_MAX_MODEL_LEN:-16384}"
REMOTE_GPU_MEMORY_UTILIZATION="${REMOTE_GPU_MEMORY_UTILIZATION:-0.82}"
REMOTE_MAX_NUM_SEQS="${REMOTE_MAX_NUM_SEQS:-64}"
SERVER_WAIT_RETRIES="${SERVER_WAIT_RETRIES:-720}"

CONCURRENCY="${CONCURRENCY:-50}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAX_QUERIES="${MAX_QUERIES:-}"

LOG_ROOT="$ROOT_DIR/logs/$METHOD_DIR/$RUN_ID"
ORCH_LOG="$LOG_ROOT/freesia23_orchestrator.log"
TUNNEL_LOG="$LOG_ROOT/freesia23_tunnel.log"
REMOTE_LOG_DIR="/data/minseo/.logs/experiments5_conflict_vanilla_llm_freesia23_$RUN_ID"
REMOTE_LOG_FILE="$REMOTE_LOG_DIR/vllm.log"
REMOTE_PID_FILE="$REMOTE_LOG_DIR/vllm.pid"

REMOTE_CTL="$ROOT_DIR/scripts/remote_pid_vllm_ctl.py"
TUNNEL_SCRIPT="$ROOT_DIR/scripts/ssh_vllm_tunnel.py"
VANILLA_RUNNER="${VANILLA_RUNNER:-$ROOT_DIR/scripts/run_conflict_majority_vanilla_llm_vllm.sh}"

INFERENCE_MODELS=(
  "google/codegemma-7b-it"
  "google/gemma-3-12b-it"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

if [[ -n "${ONLY_MODELS:-}" ]]; then
  read -r -a INFERENCE_MODELS <<< "$ONLY_MODELS"
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
    --timeout 60
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

run_vanilla_stage() {
  local stage="$1"
  shift
  (
    cd "$ROOT_DIR" || exit 1
    RUN_ID="$RUN_ID" \
    METHOD_DIR="$METHOD_DIR" \
    METHOD_NAME="$METHOD_NAME" \
    RUN_STAGES="$stage" \
    EXTERNAL_VLLM=1 \
    VLLM_URL="$VLLM_URL" \
    SERVER_WAIT_RETRIES="$SERVER_WAIT_RETRIES" \
    CONCURRENCY="$CONCURRENCY" \
    FORCE_RERUN="$FORCE_RERUN" \
    MAX_QUERIES="$MAX_QUERIES" \
    INFER_SCRIPT="${INFER_SCRIPT:-}" \
    EVAL_SCRIPT="${EVAL_SCRIPT:-}" \
    "$@" \
    bash "$VANILLA_RUNNER"
  )
}

start_model() {
  local model="$1"
  local attempt max_model_len
  max_model_len="$(max_model_len_for_model "$model")"
  log "Starting freesia vLLM model=$model GPUs=$REMOTE_GPU_IDS TP=$REMOTE_TP_SIZE max_model_len=$max_model_len"
  for attempt in 1 2; do
    if REMOTE_EFFECTIVE_MAX_MODEL_LEN="$max_model_len" remote_ctl start --model "$model" >> "$ORCH_LOG" 2>&1; then
      return 0
    fi
    log "Start attempt $attempt failed for model=$model"
    sleep 20
  done
  return 1
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
  log "Using freesia $REMOTE_HOST GPUs=$REMOTE_GPU_IDS port=$REMOTE_PORT local=$VLLM_URL"
  log "Inference models: ${INFERENCE_MODELS[*]}"

  start_tunnel

  log "Building conflict task bundles"
  run_vanilla_stage tasks env

  local model
  for model in "${INFERENCE_MODELS[@]}"; do
    if start_model "$model"; then
      log "Running $METHOD_NAME inference stage for $model"
      run_vanilla_stage inference env ONLY_MODELS="$model"
    else
      log "Failed to start inference model=$model; continuing"
    fi
  done

  log "Building $METHOD_NAME evaluation summary"
  run_vanilla_stage eval env
  log "Done. Outputs: $ROOT_DIR/outputs/$METHOD_DIR/$RUN_ID"
  log "Done. Results: $ROOT_DIR/results/$METHOD_DIR/$RUN_ID"
}

main "$@"
