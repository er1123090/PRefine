#!/usr/bin/env bash

# Reproduce the experiments4 1229_dev_6 Our Memory vLLM flow using
# experiments5 entrypoints and experiments5-local outputs.

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/data/minseo/experiments5}"
RUN_ID="${RUN_ID:-1229_dev6_autoresearch_20260515}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/1229_dev_6.json}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-8003}"
VLLM_URL="${VLLM_URL:-http://localhost:$PORT/v1}"
CONCURRENCY="${CONCURRENCY:-50}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SERVER_WAIT_RETRIES="${SERVER_WAIT_RETRIES:-120}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

RUN_ROOT="$ROOT_DIR/outputs/our_memory/$RUN_ID"
MEMORY_ROOT="$RUN_ROOT/memories"
INFERENCE_ROOT="$RUN_ROOT/inference"
LOG_ROOT="$ROOT_DIR/logs/our_memory/$RUN_ID"
RESULT_ROOT="$ROOT_DIR/results/our_memory/$RUN_ID"
RUN_LOG="$LOG_ROOT/run.log"

STEP1_SCRIPT="$ROOT_DIR/methods/our_memory/step1_extract.py"
SINGLE_SCRIPT="$ROOT_DIR/methods/our_memory/inference_vllm_singleturn.py"
MULTI_SCRIPT="$ROOT_DIR/methods/our_memory/inference_vllm_multiturn.py"
EVAL_SINGLE="$ROOT_DIR/evaluation/eval_singleturn.py"
EVAL_MULTI="$ROOT_DIR/evaluation/eval_multiturn.py"

QUERY_SINGLE="$ROOT_DIR/config/query_singleturn.json"
QUERY_MULTI="$ROOT_DIR/config/query_multiturn-domain.json"
PREF_LIST="$ROOT_DIR/config/pref_list.json"
PREF_GROUP="$ROOT_DIR/config/pref_group.json"
SCHEMA_SINGLE="$ROOT_DIR/config/schema_easy.json"
SCHEMA_MULTI="$ROOT_DIR/config/schema_all.json"

ALL_OPEN_SOURCE_MODELS=(
  "Qwen/Qwen3-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "google/gemma-3-12b-it"
  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
  "meta-llama/Llama-3.1-8B-Instruct"
)

OPEN_SOURCE_MODELS=("${ALL_OPEN_SOURCE_MODELS[@]}")
MEMORY_SOURCE_MODELS=("${ALL_OPEN_SOURCE_MODELS[@]}")

if [[ -n "${ONLY_MODELS:-}" ]]; then
  # Space-separated inference model ids.
  read -r -a OPEN_SOURCE_MODELS <<< "$ONLY_MODELS"
fi

if [[ -n "${ONLY_MEMORY_MODELS:-}" ]]; then
  # Space-separated memory-source model ids.
  read -r -a MEMORY_SOURCE_MODELS <<< "$ONLY_MEMORY_MODELS"
fi

API_MODEL_DIRS=(
  "gpt-4o-mini"
)

PREF_TYPES=("easy" "medium" "hard")
TURNS=("singleturn" "multiturn")
CONTEXT_TYPE="memory_api"
PROMPT_NAME="implicit_zs"
EXPECTED_MEMORY_ROWS="${EXPECTED_MEMORY_ROWS:-265}"

if [[ -n "${ONLY_PREFS:-}" ]]; then
  # Space-separated preference difficulty names: easy medium hard.
  read -r -a PREF_TYPES <<< "$ONLY_PREFS"
fi

if [[ -n "${ONLY_TURNS:-}" ]]; then
  # Space-separated turn names: singleturn multiturn.
  read -r -a TURNS <<< "$ONLY_TURNS"
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

output_complete() {
  local turn="$1"
  local inference_model="$2"
  local pref="$3"
  local memory_model="$4"
  local inference_safe
  local memory_safe
  inference_safe="$(model_safe "$inference_model")"
  memory_safe="$(model_safe "$memory_model")"
  [[ -s "$INFERENCE_ROOT/$turn/$inference_safe/$CONTEXT_TYPE/$pref/$memory_safe/$PROMPT_NAME/result.json" ]]
}

eval_complete() {
  local turn="$1"
  local inference_model="$2"
  local pref="$3"
  local memory_model="$4"
  local inference_safe
  local memory_safe
  inference_safe="$(model_safe "$inference_model")"
  memory_safe="$(model_safe "$memory_model")"
  [[ -s "$RESULT_ROOT/eval/$turn/$inference_safe/$CONTEXT_TYPE/$pref/$memory_safe/$PROMPT_NAME/eval.txt" ]]
}

inference_model_complete() {
  local inference_model="$1"
  local memory_model
  local memory_safe

  for memory_model in "${MEMORY_SOURCE_MODELS[@]}"; do
    memory_safe="$(model_safe "$memory_model")"
    memory_complete "$MEMORY_ROOT/$memory_safe/_memory1.jsonl" || return 1

    local pref
    for pref in "${PREF_TYPES[@]}"; do
      local turn
      for turn in "${TURNS[@]}"; do
        output_complete "$turn" "$inference_model" "$pref" "$memory_model" || return 1
        eval_complete "$turn" "$inference_model" "$pref" "$memory_model" || return 1
      done
    done
  done
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
  if memory_complete "$memory_file"; then
    log "Memory exists, skipping: $memory_file"
  else
    log "Running memory extraction for $model"
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
      --max_retries 3 2>&1 | tee -a "$RUN_LOG"
  fi

  if [[ -f "$memory_file" ]]; then
    ln -sf "_memory1.jsonl" "$out_dir/memory_result.jsonl"
  fi
}

run_inference_one() {
  local turn="$1"
  local inference_model="$2"
  local pref="$3"
  local memory_model="$4"
  local inference_safe
  local memory_safe
  inference_safe="$(model_safe "$inference_model")"
  memory_safe="$(model_safe "$memory_model")"
  local memory_file="$MEMORY_ROOT/$memory_safe/_memory1.jsonl"
  local out_dir="$INFERENCE_ROOT/$turn/$inference_safe/$CONTEXT_TYPE/$pref/$memory_safe/$PROMPT_NAME"
  local log_dir="$LOG_ROOT/inference/$turn/$inference_safe/$CONTEXT_TYPE/$pref/$memory_safe/$PROMPT_NAME"
  local output_file="$out_dir/result.json"
  local log_file="$log_dir/result.jsonl"

  if [[ ! -f "$memory_file" ]]; then
    log "Missing memory, skipping $turn inference_model=$inference_model memory_model=$memory_model pref=$pref: $memory_file"
    return 1
  fi
  if [[ -s "$output_file" ]]; then
    log "Output exists, skipping: $output_file"
    return 0
  fi

  mkdir -p "$out_dir" "$log_dir"
  log "Running $turn inference for model=$inference_model memory=$memory_model context=$CONTEXT_TYPE pref=$pref"
  if [[ "$turn" == "singleturn" ]]; then
    python "$SINGLE_SCRIPT" \
      --input_path "$DATA_PATH" \
      --memory_path "$memory_file" \
      --output_path "$output_file" \
      --log_path "$log_file" \
      --query_path "$QUERY_SINGLE" \
      --pref_list_path "$PREF_LIST" \
      --pref_group_path "$PREF_GROUP" \
      --tools_schema_path "$SCHEMA_SINGLE" \
      --context_type "$CONTEXT_TYPE" \
      --pref_type "$pref" \
      --model_name "$inference_model" \
      --vllm_url "$VLLM_URL" \
      --concurrency "$CONCURRENCY" 2>&1 | tee -a "$RUN_LOG"
  else
    python "$MULTI_SCRIPT" \
      --input_path "$DATA_PATH" \
      --memory_path "$memory_file" \
      --output_path "$output_file" \
      --log_path "$log_file" \
      --multiturn_path "$QUERY_MULTI" \
      --pref_list_path "$PREF_LIST" \
      --pref_group_path "$PREF_GROUP" \
      --tools_schema_path "$SCHEMA_MULTI" \
      --context_type "$CONTEXT_TYPE" \
      --pref_type "$pref" \
      --model_name "$inference_model" \
      --vllm_url "$VLLM_URL" \
      --concurrency "$CONCURRENCY" 2>&1 | tee -a "$RUN_LOG"
  fi
}

run_eval_one() {
  local turn="$1"
  local inference_model="$2"
  local pref="$3"
  local memory_model="$4"
  local inference_safe
  local memory_safe
  inference_safe="$(model_safe "$inference_model")"
  memory_safe="$(model_safe "$memory_model")"
  local output_file="$INFERENCE_ROOT/$turn/$inference_safe/$CONTEXT_TYPE/$pref/$memory_safe/$PROMPT_NAME/result.json"
  local eval_dir="$RESULT_ROOT/eval/$turn/$inference_safe/$CONTEXT_TYPE/$pref/$memory_safe/$PROMPT_NAME"
  local eval_txt="$eval_dir/eval.txt"
  local eval_csv="$eval_dir/eval.csv"

  if [[ ! -s "$output_file" ]]; then
    log "Missing output, skipping eval: $output_file"
    return 1
  fi

  mkdir -p "$eval_dir"
  log "Evaluating $turn output: $output_file"
  if [[ "$turn" == "singleturn" ]]; then
    python "$EVAL_SINGLE" \
      --input_path "$output_file" \
      --pref_list_path "$PREF_LIST" > "$eval_txt" 2>&1
  else
    python "$EVAL_MULTI" \
      --input_path "$output_file" \
      --pref_list_path "$PREF_LIST" \
      --csv_output "$eval_csv" > "$eval_txt" 2>&1
  fi
}

create_api_dirs_only() {
  for safe in "${API_MODEL_DIRS[@]}"; do
    mkdir -p "$MEMORY_ROOT/$safe"
    mkdir -p "$INFERENCE_ROOT/singleturn/$safe"
    mkdir -p "$INFERENCE_ROOT/multiturn/$safe"
  done
  log "Created API model directories only: ${API_MODEL_DIRS[*]}"
}

main() {
  cd "$ROOT_DIR" || exit 1
  mkdir -p "$MEMORY_ROOT" "$INFERENCE_ROOT" "$LOG_ROOT" "$RESULT_ROOT"
  create_api_dirs_only

  log "Run ID: $RUN_ID"
  log "Data path: $DATA_PATH"
  log "Concurrency: $CONCURRENCY"
  log "Inference models: ${OPEN_SOURCE_MODELS[*]}"
  log "Memory source models: ${MEMORY_SOURCE_MODELS[*]}"
  log "Context/prefs: $CONTEXT_TYPE / ${PREF_TYPES[*]}"
  log "Turns: ${TURNS[*]}"
  log "Server wait retries: $SERVER_WAIT_RETRIES"
  [[ -z "$VLLM_EXTRA_ARGS" ]] || log "vLLM extra args: $VLLM_EXTRA_ARGS"

  for memory_model in "${MEMORY_SOURCE_MODELS[@]}"; do
    memory_safe="$(model_safe "$memory_model")"
    if memory_complete "$MEMORY_ROOT/$memory_safe/_memory1.jsonl"; then
      log "Memory already complete, skipping memory server startup: $memory_model"
      continue
    fi

    if ! start_server "$memory_model"; then
      log "Skipping memory extraction after server startup failure: $memory_model"
      cleanup_server
      continue
    fi

    run_memory "$memory_model"
    cleanup_server
    sleep 10
  done

  for model in "${OPEN_SOURCE_MODELS[@]}"; do
    if inference_model_complete "$model"; then
      log "Inference model already complete across memory sources, skipping server startup: $model"
      continue
    fi

    if ! start_server "$model"; then
      log "Skipping model after server startup failure: $model"
      cleanup_server
      continue
    fi

    for memory_model in "${MEMORY_SOURCE_MODELS[@]}"; do
      memory_safe="$(model_safe "$memory_model")"
      if ! memory_complete "$MEMORY_ROOT/$memory_safe/_memory1.jsonl"; then
        log "Missing complete memory for memory source, skipping inference_model=$model memory_model=$memory_model"
        continue
      fi

      for pref in "${PREF_TYPES[@]}"; do
        for turn in "${TURNS[@]}"; do
          run_inference_one "$turn" "$model" "$pref" "$memory_model"
          run_eval_one "$turn" "$model" "$pref" "$memory_model"
        done
      done
    done

    cleanup_server
    sleep 10
  done

  log "Run finished"
  log "Outputs: $RUN_ROOT"
  log "Evaluations: $RESULT_ROOT"
}

main "$@"
