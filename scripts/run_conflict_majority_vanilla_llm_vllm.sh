#!/usr/bin/env bash

# End-to-end conflict-majority vanilla_llm inference:
#   1. Build strict-majority task bundles for both conflict files.
#   2. Run vanilla_llm inference over those task bundles.
#   3. Build evaluator summaries under experiments5/results.

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/data/minseo/experiments5}"
RUN_ID="${RUN_ID:-conflict_majority_vanilla_llm_$(date +%Y%m%d_%H%M%S)}"
METHOD_DIR="${METHOD_DIR:-vanilla_llm}"
METHOD_NAME="${METHOD_NAME:-vanilla_llm}"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-8004}"
VLLM_URL_WAS_SET="${VLLM_URL+x}"
VLLM_URL="${VLLM_URL:-http://localhost:$PORT/v1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
SERVER_WAIT_RETRIES="${SERVER_WAIT_RETRIES:-180}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"
EXTERNAL_VLLM="${EXTERNAL_VLLM:-0}"
ALLOW_CONCURRENT_VLLM="${ALLOW_CONCURRENT_VLLM:-0}"
RUN_USER="${RUN_USER:-$(id -un)}"

CONCURRENCY="${CONCURRENCY:-50}"
CONTEXT_TYPE="${CONTEXT_TYPE:-diag-apilist}"
PROMPT_NAME="${PROMPT_NAME:-imp-zs}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAX_QUERIES="${MAX_QUERIES:-}"
RUN_STAGES="${RUN_STAGES:-tasks,inference,eval}"

RUN_ROOT="$ROOT_DIR/outputs/$METHOD_DIR/$RUN_ID"
RESULT_ROOT="$ROOT_DIR/results/$METHOD_DIR/$RUN_ID"
LOG_ROOT="$ROOT_DIR/logs/$METHOD_DIR/$RUN_ID"
TASK_ROOT="$RUN_ROOT/conflict_tasks"
INFERENCE_ROOT="$RUN_ROOT/inference"
RUN_LOG="$LOG_ROOT/run.log"

TASK_BUILDER="$ROOT_DIR/scripts/build_conflict_majority_tasks.py"
INFER_SCRIPT="${INFER_SCRIPT:-$ROOT_DIR/scripts/run_conflict_majority_vanilla_llm_inference.py}"
EVAL_SCRIPT="${EVAL_SCRIPT:-$ROOT_DIR/scripts/build_conflict_majority_vanilla_llm_results.py}"

PREF_GROUP="$ROOT_DIR/config/pref_group.json"
QUERY_SINGLE="$ROOT_DIR/config/query_singleturn.json"
QUERY_MULTI="$ROOT_DIR/config/query_multiturn-domain.json"
SCHEMA_ALL="$ROOT_DIR/config/schema_all.json"

CONFLICT_FILES=(
  "$ROOT_DIR/data/e_dev_conflict_ordered_ratio.json"
  "$ROOT_DIR/data/e_dev_conflict_random_ratio.json"
)

INFERENCE_MODELS=(
  "google/codegemma-7b-it"
  "google/gemma-3-12b-it"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

TURNS=("singleturn" "multiturn")
DIFFICULTIES=("easy" "medium" "hard")

if [[ -n "${ONLY_CONFLICT_FILES:-}" ]]; then
  read -r -a CONFLICT_FILES <<< "$ONLY_CONFLICT_FILES"
fi
if [[ -n "${ONLY_MODELS:-}" ]]; then
  read -r -a INFERENCE_MODELS <<< "$ONLY_MODELS"
fi
if [[ -n "${ONLY_TURNS:-}" ]]; then
  read -r -a TURNS <<< "$ONLY_TURNS"
fi
if [[ -n "${ONLY_PREFS:-}" ]]; then
  read -r -a DIFFICULTIES <<< "$ONLY_PREFS"
fi

SERVER_PID=""
SERVER_LOG_FILE=""

log() {
  mkdir -p "$LOG_ROOT"
  printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$RUN_LOG"
}

model_safe() {
  printf '%s' "${1//\//_}"
}

conflict_slug() {
  local path base slug
  path="$1"
  base="$(basename "$path" .json)"
  slug="${base#e_dev_conflict_}"
  slug="${slug%_ratio}"
  printf '%s' "$slug"
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
  local expected_model="${1:-}"
  local count=0
  local body=""
  log "Waiting for vLLM server at $VLLM_URL"
  while true; do
    if body="$(curl -fsS "$VLLM_URL/models" 2>/dev/null)"; then
      if [[ -z "$expected_model" || "$EXTERNAL_VLLM" != "1" || "$body" == *"$expected_model"* ]]; then
        break
      fi
      if (( count % 12 == 0 )); then
        log "vLLM endpoint reachable but not serving expected model=$expected_model"
      fi
    fi
    sleep 5
    count=$((count + 1))
    if (( count % 12 == 0 )); then
      log "Still waiting for vLLM server at $VLLM_URL (${count}/${SERVER_WAIT_RETRIES})"
    fi
    if [[ -n "${SERVER_PID:-}" ]] && ! ps -p "$SERVER_PID" >/dev/null 2>&1; then
      log "vLLM server died before readiness. See ${SERVER_LOG_FILE:-$LOG_ROOT/vllm_server.log}"
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
  local model parser_flag
  model="$1"

  if [[ "$EXTERNAL_VLLM" == "1" ]]; then
    SERVER_PID=""
    SERVER_LOG_FILE=""
    log "Using external vLLM endpoint for model=$model url=$VLLM_URL"
    wait_for_server "$model"
    return $?
  fi

  parser_flag="$(parser_flag_for_model "$model")"

  cleanup_server
  mkdir -p "$LOG_ROOT"
  SERVER_LOG_FILE="$LOG_ROOT/vllm_server.$(model_safe "$model").log"
  log "Starting vLLM server model=$model gpu=$GPU_IDS port=$PORT TP=$TP_SIZE"
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
    $VLLM_EXTRA_ARGS > "$SERVER_LOG_FILE" 2>&1 &
  SERVER_PID=$!
  log "vLLM server PID=$SERVER_PID"
  wait_for_server "$model"
}

stage_enabled() {
  local stage="$1"
  local normalized
  normalized=" ${RUN_STAGES//,/ } "
  [[ "$normalized" == *" all "* || "$normalized" == *" $stage "* ]]
}

guard_no_other_vllm() {
  [[ "$EXTERNAL_VLLM" == "1" ]] && return 0
  [[ "$ALLOW_CONCURRENT_VLLM" == "1" ]] && return 0
  local matches
  matches="$(
    ps -u "$RUN_USER" -o pid=,args= | python -c '
import sys
for line in sys.stdin:
    stripped = line.strip()
    if not stripped:
        continue
    parts = stripped.split(None, 1)
    if len(parts) != 2:
        continue
    pid, args = parts
    argv = args.split()
    if len(argv) >= 2 and (argv[0].endswith("/vllm") or argv[0] == "vllm") and argv[1] == "serve":
        print(f"{pid} {args}")
    elif len(argv) >= 3 and argv[1].endswith("/vllm") and argv[2] == "serve":
        print(f"{pid} {args}")
'
  )"
  if [[ -n "$matches" ]]; then
    log "Another vLLM server is already running. Refusing to start a second server."
    printf '%s\n' "$matches" | tee -a "$RUN_LOG"
    return 1
  fi
}

inference_output_for() {
  local conflict turn inference_model pref inference_safe
  conflict="$1"
  turn="$2"
  inference_model="$3"
  pref="$4"
  inference_safe="$(model_safe "$inference_model")"
  printf '%s/%s/%s/%s/%s/%s/%s/result.json' \
    "$INFERENCE_ROOT" "$conflict" "$turn" "$inference_safe" \
    "$CONTEXT_TYPE" "$pref" "$PROMPT_NAME"
}

max_queries_args() {
  if [[ -n "$MAX_QUERIES" ]]; then
    printf '%s\n' "--max_queries" "$MAX_QUERIES"
  fi
}

build_tasks() {
  local conflict_path slug out_dir
  for conflict_path in "${CONFLICT_FILES[@]}"; do
    slug="$(conflict_slug "$conflict_path")"
    out_dir="$TASK_ROOT/$slug"
    if [[ "$FORCE_RERUN" != "1" && -s "$out_dir/summary.json" ]]; then
      log "Task bundles exist, skipping: $out_dir"
      continue
    fi
    log "Building conflict-majority tasks conflict=$slug input=$conflict_path"
    python "$TASK_BUILDER" \
      --input_path "$conflict_path" \
      --out_dir "$out_dir" \
      --pref_group_path "$PREF_GROUP" \
      --singleturn_query_path "$QUERY_SINGLE" \
      --multiturn_query_path "$QUERY_MULTI" 2>&1 | tee -a "$RUN_LOG"
  done
}

run_inference_one() {
  local conflict_path conflict turn inference_model pref tasks_path output_file log_dir log_file
  conflict_path="$1"
  turn="$2"
  inference_model="$3"
  pref="$4"
  conflict="$(conflict_slug "$conflict_path")"
  tasks_path="$TASK_ROOT/$conflict/$turn/$pref.json"
  output_file="$(inference_output_for "$conflict" "$turn" "$inference_model" "$pref")"
  log_dir="$(dirname "$output_file")"
  log_file="$log_dir/result.jsonl"

  if [[ ! -s "$tasks_path" ]]; then
    log "Missing tasks, skipping inference: $tasks_path"
    return 1
  fi
  if [[ "$FORCE_RERUN" != "1" && -s "$output_file" ]]; then
    log "Inference output exists, skipping: $output_file"
    return 0
  fi

  mkdir -p "$log_dir"
  log "Running $METHOD_NAME inference conflict=$conflict turn=$turn pref=$pref inf=$inference_model"
  PYTHONDONTWRITEBYTECODE=1 python "$INFER_SCRIPT" \
    --tasks_path "$tasks_path" \
    --output_path "$output_file" \
    --log_path "$log_file" \
    --tools_schema_path "$SCHEMA_ALL" \
    --context_type "$CONTEXT_TYPE" \
    --prompt_type "$PROMPT_NAME" \
    --model_name "$inference_model" \
    --base_url "$VLLM_URL" \
    --api_key EMPTY \
    --turn_type "$turn" \
    --concurrency "$CONCURRENCY" \
    $(max_queries_args) 2>&1 | tee -a "$RUN_LOG"
}

run_inference_matrix() {
  local model conflict_path pref turn
  for model in "${INFERENCE_MODELS[@]}"; do
    if ! start_server "$model"; then
      log "Skipping inference model after server startup failure: $model"
      cleanup_server
      continue
    fi
    for conflict_path in "${CONFLICT_FILES[@]}"; do
      for turn in "${TURNS[@]}"; do
        for pref in "${DIFFICULTIES[@]}"; do
          run_inference_one "$conflict_path" "$turn" "$model" "$pref" || true
        done
      done
    done
    cleanup_server
    sleep 10
  done
}

write_manifest() {
  mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$LOG_ROOT"
  local conflict_files_json inference_models_json turns_json difficulties_json
  conflict_files_json="$(printf '%s\n' "${CONFLICT_FILES[@]}" | python -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"
  inference_models_json="$(printf '%s\n' "${INFERENCE_MODELS[@]}" | python -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"
  turns_json="$(printf '%s\n' "${TURNS[@]}" | python -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"
  difficulties_json="$(printf '%s\n' "${DIFFICULTIES[@]}" | python -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"

  CONFLICT_FILES_JSON="$conflict_files_json" \
  INFERENCE_MODELS_JSON="$inference_models_json" \
  TURNS_JSON="$turns_json" \
  DIFFICULTIES_JSON="$difficulties_json" \
  python - <<PY
import json
import os
from pathlib import Path
manifest = {
  "run_id": "$RUN_ID",
  "root_dir": "$ROOT_DIR",
  "run_root": "$RUN_ROOT",
  "result_root": "$RESULT_ROOT",
  "log_root": "$LOG_ROOT",
  "method": "$METHOD_NAME",
  "context_type": "$CONTEXT_TYPE",
  "prompt_name": "$PROMPT_NAME",
  "max_queries": "$MAX_QUERIES",
  "conflict_files": json.loads(os.environ["CONFLICT_FILES_JSON"]),
  "inference_models": json.loads(os.environ["INFERENCE_MODELS_JSON"]),
  "turns": json.loads(os.environ["TURNS_JSON"]),
  "difficulties": json.loads(os.environ["DIFFICULTIES_JSON"]),
}
Path("$RUN_ROOT/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY
}

evaluate_results() {
  log "Building $METHOD_NAME evaluation summary"
  python "$EVAL_SCRIPT" --run_id "$RUN_ID" 2>&1 | tee -a "$RUN_LOG"
}

main() {
  cd "$ROOT_DIR" || exit 1
  mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$LOG_ROOT"

  if [[ "$EXTERNAL_VLLM" == "1" && -z "$VLLM_URL_WAS_SET" ]]; then
    log "EXTERNAL_VLLM=1 requires VLLM_URL, e.g. http://127.0.0.1:18004/v1"
    exit 1
  fi

  guard_no_other_vllm || exit 1
  write_manifest

  log "Run ID: $RUN_ID"
  log "Conflict files: ${CONFLICT_FILES[*]}"
  log "Inference models: ${INFERENCE_MODELS[*]}"
  log "Turns/difficulties: ${TURNS[*]} / ${DIFFICULTIES[*]}"
  log "Context: $CONTEXT_TYPE | max_queries=${MAX_QUERIES:-full} | external_vllm=$EXTERNAL_VLLM | stages=$RUN_STAGES"

  stage_enabled tasks && build_tasks
  stage_enabled inference && run_inference_matrix
  stage_enabled eval && evaluate_results

  log "Conflict-majority $METHOD_NAME finished"
  log "Outputs: $RUN_ROOT"
  log "Results: $RESULT_ROOT"
}

main "$@"
