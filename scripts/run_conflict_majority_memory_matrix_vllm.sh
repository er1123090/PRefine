#!/usr/bin/env bash

# End-to-end conflict-majority Our Memory matrix:
#   1. Build strict-majority task bundles for both conflict files.
#   2. Build memories with experiments4 step1 code and experiments5 step1 code.
#   3. Run conflict-majority inference with experiments4-compatible and
#      experiments5-native wrappers.
#   4. Build evaluator summaries under experiments5/results.
#
# experiments4 is treated as read-only.  PYTHONDONTWRITEBYTECODE is set for
# exp4 invocations to avoid writing __pycache__ files there.

set -u
set -o pipefail

ROOT_DIR="${ROOT_DIR:-/data/minseo/experiments5}"
E4_ROOT="${E4_ROOT:-/data/minseo/experiments4}"
RUN_ID="${RUN_ID:-conflict_majority_memory_matrix_$(date +%Y%m%d_%H%M%S)}"

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

MEMORY_CONCURRENCY="${MEMORY_CONCURRENCY:-31}"
INFERENCE_CONCURRENCY="${INFERENCE_CONCURRENCY:-50}"
EXP5_MEMORY_MODE="${EXP5_MEMORY_MODE:-verified_refine}"
EXP5_MAX_RETRIES="${EXP5_MAX_RETRIES:-3}"
E4_MAX_RETRIES="${E4_MAX_RETRIES:-3}"
CONTEXT_TYPE="${CONTEXT_TYPE:-memory_api}"
PROMPT_NAME="${PROMPT_NAME:-implicit_zs}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAX_QUERIES="${MAX_QUERIES:-}"
EXPECTED_MEMORY_ROWS="${EXPECTED_MEMORY_ROWS:-31}"
RUN_STAGES="${RUN_STAGES:-tasks,memory,inference,eval}"

RUN_ROOT="$ROOT_DIR/outputs/our_memory/$RUN_ID"
RESULT_ROOT="$ROOT_DIR/results/our_memory/$RUN_ID"
LOG_ROOT="$ROOT_DIR/logs/our_memory/$RUN_ID"
TASK_ROOT="$RUN_ROOT/conflict_tasks"
MEMORY_ROOT="$RUN_ROOT/memories"
INFERENCE_ROOT="$RUN_ROOT/inference"
RUN_LOG="$LOG_ROOT/run.log"

TASK_BUILDER="$ROOT_DIR/scripts/build_conflict_majority_tasks.py"
EXP5_STEP1="$ROOT_DIR/methods/our_memory/step1_extract.py"
E4_STEP1="${E4_STEP1:-$E4_ROOT/ours_memory/Preference_Memory_step1_LATENTPREF_ablation.py}"
EXP5_INFER="$ROOT_DIR/scripts/run_conflict_majority_exp5_memory_inference.py"
E4_INFER="$ROOT_DIR/scripts/run_conflict_majority_exp4_memory_inference.py"
EVAL_SCRIPT="$ROOT_DIR/scripts/build_conflict_majority_memory_matrix_results.py"

PREF_GROUP="$ROOT_DIR/config/pref_group.json"
QUERY_SINGLE="$ROOT_DIR/config/query_singleturn.json"
QUERY_MULTI="$ROOT_DIR/config/query_multiturn-domain.json"
SCHEMA_ALL="$ROOT_DIR/config/schema_all.json"

CONFLICT_FILES=(
  "$ROOT_DIR/data/e_dev_conflict_ordered_ratio.json"
  "$ROOT_DIR/data/e_dev_conflict_random_ratio.json"
)

MEMORY_SOURCE_MODELS=(
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
  "google/gemma-3-12b-it"
)

INFERENCE_MODELS=(
  "google/codegemma-7b-it"
  "google/gemma-3-12b-it"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
)

PIPELINES=("e4" "exp5")
TURNS=("singleturn" "multiturn")
DIFFICULTIES=("easy" "medium" "hard")

if [[ -n "${ONLY_CONFLICT_FILES:-}" ]]; then
  read -r -a CONFLICT_FILES <<< "$ONLY_CONFLICT_FILES"
fi
if [[ -n "${ONLY_MEMORY_MODELS:-}" ]]; then
  read -r -a MEMORY_SOURCE_MODELS <<< "$ONLY_MEMORY_MODELS"
fi
if [[ -n "${ONLY_MODELS:-}" ]]; then
  read -r -a INFERENCE_MODELS <<< "$ONLY_MODELS"
fi
if [[ -n "${ONLY_PIPELINES:-}" ]]; then
  read -r -a PIPELINES <<< "$ONLY_PIPELINES"
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

memory_file_for() {
  local pipeline conflict model safe
  pipeline="$1"
  conflict="$2"
  model="$3"
  safe="$(model_safe "$model")"
  printf '%s/%s/%s/%s/_memory1.jsonl' "$MEMORY_ROOT" "$pipeline" "$conflict" "$safe"
}

memory_complete() {
  local path rows
  path="$1"
  [[ -f "$path" ]] || return 1
  rows="$(wc -l < "$path" | tr -d ' ')"
  [[ "$rows" -ge "$EXPECTED_MEMORY_ROWS" ]]
}

inference_output_for() {
  local pipeline conflict turn inference_model pref memory_model inference_safe memory_safe
  pipeline="$1"
  conflict="$2"
  turn="$3"
  inference_model="$4"
  pref="$5"
  memory_model="$6"
  inference_safe="$(model_safe "$inference_model")"
  memory_safe="$(model_safe "$memory_model")"
  printf '%s/%s/%s/%s/%s/%s/%s/%s/%s/result.json' \
    "$INFERENCE_ROOT" "$pipeline" "$conflict" "$turn" "$inference_safe" \
    "$CONTEXT_TYPE" "$pref" "$memory_safe" "$PROMPT_NAME"
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

run_memory_one() {
  local pipeline conflict_path conflict model memory_file out_dir verifier_file refinement_file
  pipeline="$1"
  conflict_path="$2"
  model="$3"
  conflict="$(conflict_slug "$conflict_path")"
  memory_file="$(memory_file_for "$pipeline" "$conflict" "$model")"
  out_dir="$(dirname "$memory_file")"
  verifier_file="$out_dir/_verifier_logs1.jsonl"
  refinement_file="$out_dir/_refinement_logs1.jsonl"
  mkdir -p "$out_dir"

  if [[ "$FORCE_RERUN" != "1" ]] && memory_complete "$memory_file"; then
    log "Memory exists, skipping pipeline=$pipeline conflict=$conflict model=$model file=$memory_file"
    return 0
  fi

  log "Building memory pipeline=$pipeline conflict=$conflict model=$model"
  if [[ "$pipeline" == "e4" ]]; then
    (
      cd "$E4_ROOT/ours_memory" || exit 1
      PYTHONDONTWRITEBYTECODE=1 python "$E4_STEP1" \
        --input "$conflict_path" \
        --output "$memory_file" \
        --verifier_output "$verifier_file" \
        --refinement_output "$refinement_file" \
        --provider openai \
        --model "$model" \
        --api_base "$VLLM_URL" \
        --api_key EMPTY \
        --concurrency "$MEMORY_CONCURRENCY" \
        --max_retries "$E4_MAX_RETRIES"
    ) 2>&1 | tee -a "$RUN_LOG"
  elif [[ "$pipeline" == "exp5" ]]; then
    PYTHONDONTWRITEBYTECODE=1 python "$EXP5_STEP1" \
      --input "$conflict_path" \
      --output "$memory_file" \
      --verifier_output "$verifier_file" \
      --refinement_output "$refinement_file" \
      --provider openai \
      --model "$model" \
      --api_base "$VLLM_URL" \
      --api_key EMPTY \
      --concurrency "$MEMORY_CONCURRENCY" \
      --memory_mode "$EXP5_MEMORY_MODE" \
      --max_retries "$EXP5_MAX_RETRIES" 2>&1 | tee -a "$RUN_LOG"
  else
    log "Unknown pipeline=$pipeline"
    return 1
  fi

  if memory_complete "$memory_file"; then
    ln -sf "_memory1.jsonl" "$out_dir/memory_result.jsonl"
    log "Memory complete: $memory_file"
  else
    log "Memory incomplete after run: $memory_file"
    return 1
  fi
}

build_memories() {
  local model pipeline conflict_path
  for model in "${MEMORY_SOURCE_MODELS[@]}"; do
    if ! start_server "$model"; then
      log "Skipping memory model after server startup failure: $model"
      cleanup_server
      continue
    fi
    for pipeline in "${PIPELINES[@]}"; do
      for conflict_path in "${CONFLICT_FILES[@]}"; do
        run_memory_one "$pipeline" "$conflict_path" "$model" || true
      done
    done
    cleanup_server
    sleep 10
  done
}

run_inference_one() {
  local pipeline conflict_path conflict turn inference_model pref memory_model tasks_path memory_file output_file log_dir log_file script
  pipeline="$1"
  conflict_path="$2"
  turn="$3"
  inference_model="$4"
  pref="$5"
  memory_model="$6"
  conflict="$(conflict_slug "$conflict_path")"
  tasks_path="$TASK_ROOT/$conflict/$turn/$pref.json"
  memory_file="$(memory_file_for "$pipeline" "$conflict" "$memory_model")"
  output_file="$(inference_output_for "$pipeline" "$conflict" "$turn" "$inference_model" "$pref" "$memory_model")"
  log_dir="$(dirname "$output_file")"
  log_file="$log_dir/result.jsonl"

  if ! memory_complete "$memory_file"; then
    log "Missing memory, skipping inference pipeline=$pipeline conflict=$conflict turn=$turn pref=$pref inf=$inference_model mem=$memory_model"
    return 1
  fi
  if [[ ! -s "$tasks_path" ]]; then
    log "Missing tasks, skipping inference: $tasks_path"
    return 1
  fi
  if [[ "$FORCE_RERUN" != "1" && -s "$output_file" ]]; then
    log "Inference output exists, skipping: $output_file"
    return 0
  fi

  mkdir -p "$log_dir"
  if [[ "$pipeline" == "e4" ]]; then
    script="$E4_INFER"
  else
    script="$EXP5_INFER"
  fi

  log "Running inference pipeline=$pipeline conflict=$conflict turn=$turn pref=$pref inf=$inference_model mem=$memory_model"
  PYTHONDONTWRITEBYTECODE=1 python "$script" \
    --tasks_path "$tasks_path" \
    --memory_path "$memory_file" \
    --output_path "$output_file" \
    --log_path "$log_file" \
    --tools_schema_path "$SCHEMA_ALL" \
    --context_type "$CONTEXT_TYPE" \
    --model_name "$inference_model" \
    --base_url "$VLLM_URL" \
    --api_key EMPTY \
    --concurrency "$INFERENCE_CONCURRENCY" \
    $(max_queries_args) 2>&1 | tee -a "$RUN_LOG"
}

run_inference_matrix() {
  local model pipeline conflict_path memory_model pref turn
  for model in "${INFERENCE_MODELS[@]}"; do
    if ! start_server "$model"; then
      log "Skipping inference model after server startup failure: $model"
      cleanup_server
      continue
    fi
    for pipeline in "${PIPELINES[@]}"; do
      for conflict_path in "${CONFLICT_FILES[@]}"; do
        for memory_model in "${MEMORY_SOURCE_MODELS[@]}"; do
          for turn in "${TURNS[@]}"; do
            for pref in "${DIFFICULTIES[@]}"; do
              run_inference_one "$pipeline" "$conflict_path" "$turn" "$model" "$pref" "$memory_model" || true
            done
          done
        done
      done
    done
    cleanup_server
    sleep 10
  done
}

write_manifest() {
  mkdir -p "$RUN_ROOT" "$RESULT_ROOT" "$LOG_ROOT"
  local conflict_files_json pipelines_json memory_models_json inference_models_json turns_json difficulties_json
  conflict_files_json="$(printf '%s\n' "${CONFLICT_FILES[@]}" | python -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"
  pipelines_json="$(printf '%s\n' "${PIPELINES[@]}" | python -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"
  memory_models_json="$(printf '%s\n' "${MEMORY_SOURCE_MODELS[@]}" | python -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"
  inference_models_json="$(printf '%s\n' "${INFERENCE_MODELS[@]}" | python -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"
  turns_json="$(printf '%s\n' "${TURNS[@]}" | python -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"
  difficulties_json="$(printf '%s\n' "${DIFFICULTIES[@]}" | python -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')"

  CONFLICT_FILES_JSON="$conflict_files_json" \
  PIPELINES_JSON="$pipelines_json" \
  MEMORY_MODELS_JSON="$memory_models_json" \
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
  "context_type": "$CONTEXT_TYPE",
  "prompt_name": "$PROMPT_NAME",
  "exp5_memory_mode": "$EXP5_MEMORY_MODE",
  "exp5_max_retries": int("$EXP5_MAX_RETRIES"),
  "e4_step1": "$E4_STEP1",
  "e4_max_retries": int("$E4_MAX_RETRIES"),
  "max_queries": "$MAX_QUERIES",
  "conflict_files": json.loads(os.environ["CONFLICT_FILES_JSON"]),
  "pipelines": json.loads(os.environ["PIPELINES_JSON"]),
  "memory_source_models": json.loads(os.environ["MEMORY_MODELS_JSON"]),
  "inference_models": json.loads(os.environ["INFERENCE_MODELS_JSON"]),
  "turns": json.loads(os.environ["TURNS_JSON"]),
  "difficulties": json.loads(os.environ["DIFFICULTIES_JSON"]),
}
Path("$RUN_ROOT/manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
PY
}

evaluate_results() {
  log "Building evaluation summary"
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
  log "Pipelines: ${PIPELINES[*]}"
  log "Memory source models: ${MEMORY_SOURCE_MODELS[*]}"
  log "Inference models: ${INFERENCE_MODELS[*]}"
  log "Turns/difficulties: ${TURNS[*]} / ${DIFFICULTIES[*]}"
  log "Context: $CONTEXT_TYPE | max_queries=${MAX_QUERIES:-full} | external_vllm=$EXTERNAL_VLLM | stages=$RUN_STAGES"

  stage_enabled tasks && build_tasks
  stage_enabled memory && build_memories
  stage_enabled inference && run_inference_matrix
  stage_enabled eval && evaluate_results

  log "Conflict-majority memory matrix finished"
  log "Outputs: $RUN_ROOT"
  log "Results: $RESULT_ROOT"
}

main "$@"
