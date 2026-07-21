#!/usr/bin/env bash

# Finish DeepSeek-R1-Distill-Qwen-7B ablation inference with one vLLM server per
# GPU. This keeps experiments4 read-only: each shard still delegates to the
# experiments5 base runner, which imports/runs the experiments4 inference code.

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/data/minseo/experiments5}"
RUN_ID="${RUN_ID:-1229_dev6_ablation_e4_ours_memory_inference_20260528}"
REFINE_SOURCE_RUN_ID="${REFINE_SOURCE_RUN_ID:-1229_dev6_memory_variants_20260517}"
BASE_RUNNER="$ROOT_DIR/scripts/run_gen_only_accum_e4_ours_memory_inference_vllm.sh"

MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
GPU_LIST=(${GPU_LIST:-0 1 2 3})
PORT_LIST=(${PORT_LIST:-8010 8011 8012 8013})
TP_SIZE="${TP_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SERVER_WAIT_RETRIES="${SERVER_WAIT_RETRIES:-180}"
CONCURRENCY_SINGLE="${CONCURRENCY_SINGLE:-20}"
CONCURRENCY_MULTI="${CONCURRENCY_MULTI:-40}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

LOG_ROOT="$ROOT_DIR/logs/our_memory/$RUN_ID"
RUN_LOG="$LOG_ROOT/qwen7b_parallel_shards.run.log"

SERVER_PIDS=()
WORKER_PIDS=()

TASKS=(
  "blind_refine_2|deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "blind_refine_2|deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
  "blind_refine_2|google/gemma-3-12b-it"
  "blind_refine_3|deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "blind_refine_3|deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
  "blind_refine_3|google/gemma-3-12b-it"
)

log() {
  mkdir -p "$LOG_ROOT"
  printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$RUN_LOG"
}

model_safe() {
  printf '%s' "${1//\//_}"
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

kill_tree() {
  local signal="$1"
  local pid="$2"
  mapfile -t pids < <(pid_tree "$pid" | tac)
  [[ "${#pids[@]}" -gt 0 ]] || return 0
  kill "-$signal" "${pids[@]}" >/dev/null 2>&1 || true
}

cleanup() {
  local pid
  for pid in "${WORKER_PIDS[@]:-}"; do
    kill_tree TERM "$pid"
  done
  for pid in "${SERVER_PIDS[@]:-}"; do
    if ps -p "$pid" >/dev/null 2>&1; then
      log "Stopping vLLM server PID=$pid"
      kill_tree TERM "$pid"
    fi
  done
  sleep 2
  for pid in "${SERVER_PIDS[@]:-}"; do
    if ps -p "$pid" >/dev/null 2>&1; then
      log "Force stopping vLLM server PID=$pid"
      kill_tree KILL "$pid"
    fi
  done
}

trap cleanup EXIT INT TERM

wait_for_server() {
  local port="$1"
  local pid="$2"
  local url="http://localhost:$port/v1"
  local count=0
  log "Waiting for vLLM server at $url"
  while ! curl -fsS "$url/models" >/dev/null 2>&1; do
    sleep 5
    count=$((count + 1))
    if ! ps -p "$pid" >/dev/null 2>&1; then
      log "vLLM server died before readiness: port=$port pid=$pid"
      return 1
    fi
    if [[ "$count" -ge "$SERVER_WAIT_RETRIES" ]]; then
      log "Timed out waiting for vLLM server: port=$port pid=$pid"
      return 1
    fi
  done
  log "vLLM server is ready at $url"
}

start_server() {
  local gpu="$1"
  local port="$2"
  local log_file="$LOG_ROOT/vllm_server.$(model_safe "$MODEL").gpu${gpu}.port${port}.log"

  mkdir -p "$LOG_ROOT"
  log "Starting vLLM server model=$MODEL gpu=$gpu port=$port TP=$TP_SIZE"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" nohup vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$port" \
    --tensor-parallel-size "$TP_SIZE" \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --trust-remote-code \
    $VLLM_EXTRA_ARGS > "$log_file" 2>&1 &
  local pid=$!
  SERVER_PIDS+=("$pid")
  wait_for_server "$port" "$pid"
}

run_task() {
  local port="$1"
  local task="$2"
  local mode="${task%%|*}"
  local memory_model="${task#*|}"

  log "Shard task start port=$port mode=$mode memory=$memory_model"
  RUN_ID="$RUN_ID" \
    SOURCE_MEMORY_RUN_ID="$REFINE_SOURCE_RUN_ID" \
    MEMORY_MODE="$mode" \
    EXTERNAL_VLLM=1 \
    VLLM_URL="http://localhost:$port/v1" \
    ONLY_MODELS="$MODEL" \
    ONLY_MEMORY_MODELS="$memory_model" \
    CONCURRENCY_SINGLE="$CONCURRENCY_SINGLE" \
    CONCURRENCY_MULTI="$CONCURRENCY_MULTI" \
    "$BASE_RUNNER"
  log "Shard task done port=$port mode=$mode memory=$memory_model"
}

worker() {
  local worker_index="$1"
  local port="$2"
  local task_index="$worker_index"

  while [[ "$task_index" -lt "${#TASKS[@]}" ]]; do
    run_task "$port" "${TASKS[$task_index]}"
    task_index=$((task_index + ${#PORT_LIST[@]}))
  done
}

main() {
  if [[ "${#GPU_LIST[@]}" -ne "${#PORT_LIST[@]}" ]]; then
    log "GPU_LIST and PORT_LIST must have the same length"
    exit 1
  fi
  if [[ ! -x "$BASE_RUNNER" ]]; then
    log "Base runner is missing or not executable: $BASE_RUNNER"
    exit 1
  fi

  log "Run ID: $RUN_ID"
  log "Parallel Qwen model: $MODEL"
  log "GPUs: ${GPU_LIST[*]} ports: ${PORT_LIST[*]}"
  log "Tasks: ${TASKS[*]}"

  local index
  for index in "${!GPU_LIST[@]}"; do
    start_server "${GPU_LIST[$index]}" "${PORT_LIST[$index]}"
  done

  for index in "${!PORT_LIST[@]}"; do
    worker "$index" "${PORT_LIST[$index]}" &
    WORKER_PIDS+=("$!")
  done

  local failed=0
  local pid
  for pid in "${WORKER_PIDS[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  WORKER_PIDS=()

  if [[ "$failed" -ne 0 ]]; then
    log "One or more shard workers failed"
    exit 1
  fi
  log "Parallel Qwen shard inference finished"
}

main "$@"
