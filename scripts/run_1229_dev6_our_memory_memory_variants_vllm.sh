#!/usr/bin/env bash

# Build Our Memory step-1 memory variants for the 1229_dev_6 split using
# open-source vLLM models only.
#
# Variants:
#   - generation_only: legacy overwrite, one generation per session, no verifier, no refinement
#   - generation_only_accum: append one generated preference per session, no verifier, no refinement
#   - blind_refine_1: one generation + one unconditional refinement, no verifier
#   - blind_refine_2: one generation + two unconditional refinements, no verifier
#   - blind_refine_3: one generation + three unconditional refinements, no verifier

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/data/minseo/experiments5}"
RUN_ID="${RUN_ID:-1229_dev6_memory_variants_20260517}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/1229_dev_6.json}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-8004}"
VLLM_URL="${VLLM_URL:-http://localhost:$PORT/v1}"
CONCURRENCY="${CONCURRENCY:-50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SERVER_WAIT_RETRIES="${SERVER_WAIT_RETRIES:-120}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
EXPECTED_MEMORY_ROWS="${EXPECTED_MEMORY_ROWS:-265}"
ALLOW_CONCURRENT_VLLM="${ALLOW_CONCURRENT_VLLM:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RUN_USER="${RUN_USER:-$(id -un)}"
TRUE_BLIND="${TRUE_BLIND:-0}"

RUN_ROOT="$ROOT_DIR/outputs/our_memory/$RUN_ID"
MEMORY_ROOT="$RUN_ROOT/memories"
LOG_ROOT="$ROOT_DIR/logs/our_memory/$RUN_ID"
RUN_LOG="$LOG_ROOT/run.log"

STEP1_SCRIPT="$ROOT_DIR/methods/our_memory/step1_extract.py"

ALL_OPEN_SOURCE_MODELS=(
  "Qwen/Qwen3-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "google/gemma-3-12b-it"
  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
  "meta-llama/Llama-3.1-8B-Instruct"
)

MEMORY_SOURCE_MODELS=("${ALL_OPEN_SOURCE_MODELS[@]}")
MEMORY_MODES=(
  "generation_only"
  "generation_only_accum"
  "blind_refine_1"
  "blind_refine_2"
  "blind_refine_3"
)

if [[ -n "${ONLY_MEMORY_MODELS:-}" ]]; then
  # Space-separated Hugging Face model ids.
  read -r -a MEMORY_SOURCE_MODELS <<< "$ONLY_MEMORY_MODELS"
fi

if [[ -n "${ONLY_MEMORY_MODES:-}" ]]; then
  # Space-separated mode names from MEMORY_MODES above.
  read -r -a MEMORY_MODES <<< "$ONLY_MEMORY_MODES"
fi

STEP1_EXTRA_ARGS=()
if [[ "$TRUE_BLIND" == "1" ]]; then
  STEP1_EXTRA_ARGS+=(--true_blind)
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
  while ! curl -fsS "$VLLM_URL/models" >/dev/null 2>&1; do
    sleep 5
    count=$((count + 1))
    if [[ -n "${SERVER_PID:-}" ]] && ! ps -p "$SERVER_PID" >/dev/null 2>&1; then
      log "vLLM server died before readiness. See $LOG_ROOT/vllm_server_${PORT}.log"
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
    $VLLM_EXTRA_ARGS > "$LOG_ROOT/vllm_server_${PORT}.log" 2>&1 &
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

run_memory_variant() {
  local model="$1"
  local mode="$2"
  local safe
  safe="$(model_safe "$model")"
  local out_dir="$MEMORY_ROOT/$mode/$safe"
  local memory_file="$out_dir/_memory1.jsonl"
  local verifier_file="$out_dir/_verifier_logs1.jsonl"
  local refinement_file="$out_dir/_refinement_logs1.jsonl"

  mkdir -p "$out_dir"
  if [[ "$FORCE_RERUN" != "1" ]] && memory_complete "$memory_file"; then
    log "Memory variant exists, skipping: mode=$mode model=$model file=$memory_file"
  else
    log "Running memory variant: mode=$mode model=$model"
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
      --memory_mode "$mode" \
      "${STEP1_EXTRA_ARGS[@]}" 2>&1 | tee -a "$RUN_LOG"
  fi

  if [[ -f "$memory_file" ]]; then
    ln -sf "_memory1.jsonl" "$out_dir/memory_result.jsonl"
  fi
}

guard_no_other_vllm() {
  [[ "$ALLOW_CONCURRENT_VLLM" == "1" ]] && return 0
  if pgrep -a -u "$RUN_USER" -f 'vllm serve' >/dev/null 2>&1; then
    log "Another vLLM server is already running. Refusing to start a second server."
    log "Set ALLOW_CONCURRENT_VLLM=1 only if you intentionally want concurrent vLLM servers."
    pgrep -a -u "$RUN_USER" -f 'vllm serve' | tee -a "$RUN_LOG"
    return 1
  fi
}

main() {
  cd "$ROOT_DIR" || exit 1
  mkdir -p "$MEMORY_ROOT" "$LOG_ROOT"

  guard_no_other_vllm || exit 1

  log "Run ID: $RUN_ID"
  log "Data path: $DATA_PATH"
  log "Memory modes: ${MEMORY_MODES[*]}"
  log "Memory source models: ${MEMORY_SOURCE_MODELS[*]}"
  log "True blind memory input: $TRUE_BLIND"
  log "Concurrency: $CONCURRENCY"
  [[ -z "$VLLM_EXTRA_ARGS" ]] || log "vLLM extra args: $VLLM_EXTRA_ARGS"

  local model
  for model in "${MEMORY_SOURCE_MODELS[@]}"; do
    if ! start_server "$model"; then
      log "Skipping variants after server startup failure: $model"
      cleanup_server
      continue
    fi

    local mode
    for mode in "${MEMORY_MODES[@]}"; do
      run_memory_variant "$model" "$mode"
    done

    cleanup_server
    sleep 10
  done

  log "Memory variant run finished"
  log "Outputs: $RUN_ROOT"
}

main "$@"
