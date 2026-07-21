#!/usr/bin/env bash

# Run vanilla_llm seed-variance inference through existing vLLM endpoints.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

PARENT_RUN_ID="${PARENT_RUN_ID:-vanilla_llm_seed_variance_20260524}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/1229_dev_6.json}"
RUN_ROOT="$ROOT_DIR/outputs/vanilla_llm/$PARENT_RUN_ID"
RESULT_ROOT="$ROOT_DIR/results/vanilla_llm/$PARENT_RUN_ID"
LOG_ROOT="$ROOT_DIR/logs/vanilla_llm/$PARENT_RUN_ID"
RUN_LOG="$LOG_ROOT/run_vanilla_llm_seed_variance.log"

SINGLE_SCRIPT="$SCRIPT_DIR/inference_vllm_singleturn.py"
MULTI_SCRIPT="$SCRIPT_DIR/inference_vllm_multiturn.py"
QUERY_SINGLE="$ROOT_DIR/config/query_singleturn.json"
QUERY_MULTI="$ROOT_DIR/config/query_multiturn-domain.json"
PREF_LIST="$ROOT_DIR/config/pref_list.json"
PREF_GROUP="$ROOT_DIR/config/pref_group.json"
SCHEMA_SINGLE="$ROOT_DIR/config/schema_easy.json"
SCHEMA_MULTI="$ROOT_DIR/config/schema_all.json"

CONTEXT_TYPE="${CONTEXT_TYPE:-diag-apilist}"
PROMPT_NAME="${PROMPT_NAME:-imp-zs}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_QUERIES="${MAX_QUERIES:-}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"

INFERENCE_MODELS=(
  "google/gemma-3-12b-it"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "google/codegemma-7b-it"
)
VLLM_URLS_ARR=(
  "http://localhost:8100/v1"
  "http://localhost:8101/v1"
  "http://localhost:8102/v1"
  "http://localhost:8103/v1"
)
PREF_TYPES=("easy" "medium" "hard")
TURNS=("singleturn" "multiturn")
REPEATS=("seed_001" "seed_002" "seed_003")

if [[ -n "${ONLY_MODELS:-}" ]]; then read -r -a INFERENCE_MODELS <<< "$ONLY_MODELS"; fi
if [[ -n "${VLLM_URLS:-}" ]]; then read -r -a VLLM_URLS_ARR <<< "$VLLM_URLS"; fi
if [[ -n "${ONLY_PREFS:-}" ]]; then read -r -a PREF_TYPES <<< "$ONLY_PREFS"; fi
if [[ -n "${ONLY_TURNS:-}" ]]; then read -r -a TURNS <<< "$ONLY_TURNS"; fi
if [[ -n "${ONLY_REPEATS:-}" ]]; then read -r -a REPEATS <<< "$ONLY_REPEATS"; fi
if [[ "$SMOKE_ONLY" == "1" && -z "$MAX_QUERIES" ]]; then MAX_QUERIES=2; fi

log() {
  mkdir -p "$LOG_ROOT"
  printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$RUN_LOG"
}

model_safe() {
  printf '%s' "${1//\//_}"
}

output_file_for() {
  local repeat="$1" turn="$2" model="$3" pref="$4"
  printf '%s/repeats/%s/inference/%s/%s/%s/%s/%s/result.json' \
    "$RUN_ROOT" "$repeat" "$turn" "$(model_safe "$model")" "$CONTEXT_TYPE" "$pref" "$PROMPT_NAME"
}

log_file_for() {
  local repeat="$1" turn="$2" model="$3" pref="$4"
  printf '%s/repeats/%s/inference/%s/%s/%s/%s/%s/result.jsonl' \
    "$LOG_ROOT" "$repeat" "$turn" "$(model_safe "$model")" "$CONTEXT_TYPE" "$pref" "$PROMPT_NAME"
}

write_metadata() {
  mkdir -p "$RUN_ROOT" "$RESULT_ROOT"
  python - "$RUN_ROOT/metadata.json" "$PARENT_RUN_ID" "$CONTEXT_TYPE" "$PROMPT_NAME" \
    "${INFERENCE_MODELS[@]}" -- "${REPEATS[@]}" -- "${TURNS[@]}" -- "${PREF_TYPES[@]}" -- "${VLLM_URLS_ARR[@]}" <<'PY'
import json
import sys
from datetime import datetime

path, run_id, context, prompt = sys.argv[1:5]
seps = [i for i, arg in enumerate(sys.argv) if arg == "--"]
models = sys.argv[5:seps[0]]
repeats = sys.argv[seps[0] + 1:seps[1]]
turns = sys.argv[seps[1] + 1:seps[2]]
prefs = sys.argv[seps[2] + 1:seps[3]]
urls = sys.argv[seps[3] + 1:]
payload = {
    "created_at": datetime.now().isoformat(),
    "parent_run_id": run_id,
    "method": "vanilla_llm",
    "input_style": context,
    "prompt_type": prompt,
    "inference_models": models,
    "repeats": repeats,
    "turns": turns,
    "difficulties": prefs,
    "vllm_urls": urls,
    "seed_policy": "deterministic_rerun",
    "expected_result_files": len(models) * len(repeats) * len(turns) * len(prefs),
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
PY
}

check_endpoints() {
  local idx model url
  for idx in "${!INFERENCE_MODELS[@]}"; do
    model="${INFERENCE_MODELS[$idx]}"
    url="${VLLM_URLS_ARR[$idx]:-}"
    if [[ -z "$url" ]]; then
      log "Missing vLLM URL for model=$model"
      return 1
    fi
    if ! curl -fsS --connect-timeout 2 -m 5 "$url/models" >/dev/null; then
      log "Endpoint not ready for model=$model url=$url"
      return 1
    fi
    log "Endpoint ready model=$model url=$url"
  done
}

run_one() {
  local repeat="$1" turn="$2" model="$3" url="$4" pref="$5"
  local output_file log_file extra_args=()
  output_file="$(output_file_for "$repeat" "$turn" "$model" "$pref")"
  log_file="$(log_file_for "$repeat" "$turn" "$model" "$pref")"
  if [[ -s "$output_file" ]]; then
    log "Output exists, skipping: $output_file"
    return 0
  fi
  mkdir -p "$(dirname "$output_file")" "$(dirname "$log_file")"
  if [[ -n "$MAX_QUERIES" ]]; then
    extra_args+=(--max_queries "$MAX_QUERIES")
  fi
  log "Running repeat=$repeat turn=$turn model=$model pref=$pref"
  if [[ "$turn" == "singleturn" ]]; then
    python "$SINGLE_SCRIPT" \
      --input_path "$DATA_PATH" \
      --output_path "$output_file" \
      --log_path "$log_file" \
      --query_path "$QUERY_SINGLE" \
      --pref_list_path "$PREF_LIST" \
      --pref_group_path "$PREF_GROUP" \
      --tools_schema_path "$SCHEMA_SINGLE" \
      --context_type "$CONTEXT_TYPE" \
      --pref_type "$pref" \
      --prompt_type "$PROMPT_NAME" \
      --model_name "$model" \
      --vllm_url "$url" \
      --concurrency "$CONCURRENCY" \
      "${extra_args[@]}" 2>&1 | tee -a "$RUN_LOG"
  else
    python "$MULTI_SCRIPT" \
      --input_path "$DATA_PATH" \
      --output_path "$output_file" \
      --log_path "$log_file" \
      --multiturn_path "$QUERY_MULTI" \
      --pref_list_path "$PREF_LIST" \
      --pref_group_path "$PREF_GROUP" \
      --tools_schema_path "$SCHEMA_MULTI" \
      --context_type "$CONTEXT_TYPE" \
      --pref_type "$pref" \
      --prompt_type "$PROMPT_NAME" \
      --model_name "$model" \
      --vllm_url "$url" \
      --concurrency "$CONCURRENCY" \
      "${extra_args[@]}" 2>&1 | tee -a "$RUN_LOG"
  fi
}

run_model_jobs() {
  local model="$1" url="$2" repeat turn pref
  for repeat in "${REPEATS[@]}"; do
    for pref in "${PREF_TYPES[@]}"; do
      for turn in "${TURNS[@]}"; do
        run_one "$repeat" "$turn" "$model" "$url" "$pref" || return 1
      done
    done
  done
}

main() {
  cd "$ROOT_DIR" || exit 1
  export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
  mkdir -p "$LOG_ROOT" "$RUN_ROOT" "$RESULT_ROOT"
  log "PARENT_RUN_ID=$PARENT_RUN_ID"
  log "Models: ${INFERENCE_MODELS[*]}"
  log "Repeats: ${REPEATS[*]} Turns: ${TURNS[*]} Prefs: ${PREF_TYPES[*]}"
  write_metadata | tee -a "$RUN_LOG"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1 complete"
    exit 0
  fi
  check_endpoints || exit 1

  local idx model url worker_pids=()
  for idx in "${!INFERENCE_MODELS[@]}"; do
    model="${INFERENCE_MODELS[$idx]}"
    url="${VLLM_URLS_ARR[$idx]}"
    run_model_jobs "$model" "$url" &
    worker_pids+=("$!")
  done

  local failed=0 pid
  for pid in "${worker_pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  [[ "$failed" == "0" ]] || exit 1
  log "Vanilla inference finished. Outputs: $RUN_ROOT"
}

main "$@"
