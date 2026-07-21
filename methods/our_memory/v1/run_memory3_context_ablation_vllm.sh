#!/usr/bin/env bash

# Run MEMORY3 context ablation on already-running vLLM endpoints.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

PARENT_RUN_ID="${PARENT_RUN_ID:-memory3_0311_context_ablation_20260524}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/1229_dev_6.json}"
SOURCE_MEMORY_ROOT="${SOURCE_MEMORY_ROOT:-/data/minseo/experiments4/ours_memory/inference/1231_MEMORY3_0311}"
FALLBACK_MEMORY_ROOT="${FALLBACK_MEMORY_ROOT:-/data/minseo/experiments4/ours_memory/inference/1231_MEMORY3}"
V1_MEMORY_FILENAME="${V1_MEMORY_FILENAME:-_memory1.jsonl}"

RUN_ROOT="$ROOT_DIR/outputs/our_memory/$PARENT_RUN_ID"
RESULT_ROOT="$ROOT_DIR/results/our_memory/$PARENT_RUN_ID"
LOG_ROOT="$ROOT_DIR/logs/our_memory/$PARENT_RUN_ID"
RUN_LOG="$LOG_ROOT/run_memory3_context_ablation.log"

INFERENCE_SCRIPT="$SCRIPT_DIR/inference_vllm_context_ablation.py"
EVAL_SCRIPT="$SCRIPT_DIR/evaluate_memory3_context_ablation.py"
QUERY_SINGLE="$ROOT_DIR/config/query_singleturn.json"
QUERY_MULTI="$ROOT_DIR/config/query_multiturn-domain.json"
PREF_LIST="$ROOT_DIR/config/pref_list.json"
PREF_GROUP="$ROOT_DIR/config/pref_group.json"
SCHEMA_SINGLE="$ROOT_DIR/config/schema_easy.json"
SCHEMA_MULTI="$ROOT_DIR/config/schema_all.json"

PROMPT_NAME="${PROMPT_NAME:-implicit_zs}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_QUERIES="${MAX_QUERIES:-}"
DRY_RUN="${DRY_RUN:-0}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
RUN_EVALUATION="${RUN_EVALUATION:-1}"
ALLOW_MISSING_EVAL="${ALLOW_MISSING_EVAL:-0}"

INFERENCE_MODELS=(
  "google/gemma-3-12b-it"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
  "google/codegemma-7b-it"
)
VLLM_URL_LIST=(
  "http://localhost:8100/v1"
  "http://localhost:8101/v1"
  "http://localhost:8102/v1"
  "http://localhost:8103/v1"
)
CONTEXTS=("memory_only" "full_dialogue_only" "api_only" "memory_api" "full_dialogue_api")
PREF_TYPES=("easy" "medium" "hard")
TURNS=("singleturn" "multiturn")
REPEATS=("seed_001")

if [[ -n "${ONLY_MODELS:-}" ]]; then read -r -a INFERENCE_MODELS <<< "$ONLY_MODELS"; fi
if [[ -n "${VLLM_URLS:-}" ]]; then read -r -a VLLM_URL_LIST <<< "$VLLM_URLS"; fi
if [[ -n "${ONLY_CONTEXTS:-}" ]]; then read -r -a CONTEXTS <<< "$ONLY_CONTEXTS"; fi
if [[ -n "${ONLY_PREFS:-}" ]]; then read -r -a PREF_TYPES <<< "$ONLY_PREFS"; fi
if [[ -n "${ONLY_TURNS:-}" ]]; then read -r -a TURNS <<< "$ONLY_TURNS"; fi
if [[ -n "${ONLY_REPEATS:-}" ]]; then read -r -a REPEATS <<< "$ONLY_REPEATS"; fi
if [[ "$SMOKE_ONLY" == "1" && -z "$MAX_QUERIES" ]]; then MAX_QUERIES=3; fi

log() {
  mkdir -p "$LOG_ROOT"
  printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$RUN_LOG"
}

model_safe() {
  local safe="$1"
  safe="${safe//\//_}"
  safe="${safe//[/__}"
  safe="${safe//]/}"
  safe="${safe//:/_}"
  printf '%s' "$safe"
}

discover_memory_sources() {
  local root="$1"
  [[ -d "$root" ]] || return 0
  local dir base
  for dir in "$root"/*; do
    [[ -d "$dir" ]] || continue
    base="$(basename "$dir")"
    [[ "$base" == _deprecated* ]] && continue
    [[ -s "$dir/$V1_MEMORY_FILENAME" ]] || continue
    printf '%s\n' "$base"
  done | sort
}

resolve_memory_sources() {
  if [[ -n "${ONLY_MEMORY_MODELS:-}" ]]; then
    read -r -a MEMORY_MODELS <<< "$ONLY_MEMORY_MODELS"
    EFFECTIVE_MEMORY_ROOT="$SOURCE_MEMORY_ROOT"
    return 0
  fi

  mapfile -t MEMORY_MODELS < <(discover_memory_sources "$SOURCE_MEMORY_ROOT")
  EFFECTIVE_MEMORY_ROOT="$SOURCE_MEMORY_ROOT"
  if [[ "${#MEMORY_MODELS[@]}" -eq 0 ]]; then
    mapfile -t MEMORY_MODELS < <(discover_memory_sources "$FALLBACK_MEMORY_ROOT")
    EFFECTIVE_MEMORY_ROOT="$FALLBACK_MEMORY_ROOT"
    log "No non-deprecated memory files found under SOURCE_MEMORY_ROOT=$SOURCE_MEMORY_ROOT; using FALLBACK_MEMORY_ROOT=$FALLBACK_MEMORY_ROOT"
  fi
  if [[ "${#MEMORY_MODELS[@]}" -eq 0 ]]; then
    log "No usable non-deprecated memory sources found."
    return 1
  fi
}

context_memory_sources_csv() {
  local context="$1"
  if [[ "$context" == "full_dialogue_only" ]]; then
    printf '%s\n' "none"
  else
    printf '%s\n' "${MEMORY_MODELS[@]}"
  fi
}

memory_file_for() {
  local memory_safe="$1"
  if [[ "$memory_safe" == "none" ]]; then
    printf ''
  else
    printf '%s/%s/%s' "$EFFECTIVE_MEMORY_ROOT" "$memory_safe" "$V1_MEMORY_FILENAME"
  fi
}

output_file_for() {
  local repeat="$1" turn="$2" inference_model="$3" context="$4" pref="$5" memory_safe="$6"
  printf '%s/repeats/%s/inference/%s/%s/%s/%s/%s/%s/result.json' \
    "$RUN_ROOT" "$repeat" "$turn" "$(model_safe "$inference_model")" \
    "$context" "$pref" "$memory_safe" "$PROMPT_NAME"
}

log_file_for() {
  local repeat="$1" turn="$2" inference_model="$3" context="$4" pref="$5" memory_safe="$6"
  printf '%s/repeats/%s/inference/%s/%s/%s/%s/%s/%s/result.jsonl' \
    "$LOG_ROOT" "$repeat" "$turn" "$(model_safe "$inference_model")" \
    "$context" "$pref" "$memory_safe" "$PROMPT_NAME"
}

validate_vllm_urls() {
  local idx model url
  for idx in "${!INFERENCE_MODELS[@]}"; do
    model="${INFERENCE_MODELS[$idx]}"
    url="${VLLM_URL_LIST[$idx]:-}"
    if [[ -z "$url" ]]; then
      log "Missing VLLM URL for model=$model index=$idx"
      return 1
    fi
    if ! curl -fsS --connect-timeout 2 -m 5 "$url/models" >/dev/null 2>&1; then
      log "vLLM endpoint is not ready: model=$model url=$url"
      return 1
    fi
    log "vLLM endpoint ready: model=$model url=$url"
  done
}

write_metadata() {
  mkdir -p "$RUN_ROOT" "$RESULT_ROOT"
  python - "$RUN_ROOT/metadata.json" "$SOURCE_MEMORY_ROOT" "$FALLBACK_MEMORY_ROOT" "$EFFECTIVE_MEMORY_ROOT" "$V1_MEMORY_FILENAME" \
    "${INFERENCE_MODELS[@]}" --urls "${VLLM_URL_LIST[@]}" --mem "${MEMORY_MODELS[@]}" --ctx "${CONTEXTS[@]}" --turns "${TURNS[@]}" --prefs "${PREF_TYPES[@]}" --repeats "${REPEATS[@]}" <<'PY'
import json
import sys

path = sys.argv[1]
source_root = sys.argv[2]
fallback_root = sys.argv[3]
effective_root = sys.argv[4]
memory_filename = sys.argv[5]
tokens = sys.argv[6:]

def split_after(marker, items):
    idx = items.index(marker)
    next_markers = [i for i, x in enumerate(items[idx + 1:], start=idx + 1) if x.startswith("--")]
    end = next_markers[0] if next_markers else len(items)
    return items[idx + 1:end]

urls = split_after("--urls", tokens)
memories = split_after("--mem", tokens)
contexts = split_after("--ctx", tokens)
turns = split_after("--turns", tokens)
prefs = split_after("--prefs", tokens)
repeats = split_after("--repeats", tokens)
models = tokens[:tokens.index("--urls")]
context_memory_sources = {
    context: (["none"] if context == "full_dialogue_only" else memories)
    for context in contexts
}
expected = 0
for context in contexts:
    expected += len(context_memory_sources[context]) * len(models) * len(turns) * len(prefs) * len(repeats)

metadata = {
    "memory_method": "ours_memory_v1",
    "memory_variant": "verified_refine",
    "input_ablation": True,
    "source_memory_root_requested": source_root,
    "fallback_memory_root": fallback_root,
    "source_memory_root_effective": effective_root,
    "v1_memory_filename": memory_filename,
    "inference_models": models,
    "vllm_urls": urls,
    "memory_sources": memories,
    "contexts": contexts,
    "context_memory_sources": context_memory_sources,
    "turns": turns,
    "difficulties": prefs,
    "repeats": repeats,
    "expected_result_files": expected,
    "seed_policy": "single deterministic vLLM pass unless ONLY_REPEATS overrides",
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print(json.dumps({"metadata": path, "expected_result_files": expected}, indent=2))
PY
}

run_inference_one() {
  local repeat="$1" turn="$2" inference_model="$3" vllm_url="$4" context="$5" pref="$6" memory_safe="$7"
  local memory_file output_file log_file schema_file
  memory_file="$(memory_file_for "$memory_safe")"
  output_file="$(output_file_for "$repeat" "$turn" "$inference_model" "$context" "$pref" "$memory_safe")"
  log_file="$(log_file_for "$repeat" "$turn" "$inference_model" "$context" "$pref" "$memory_safe")"
  schema_file="$SCHEMA_SINGLE"
  [[ "$turn" == "multiturn" ]] && schema_file="$SCHEMA_MULTI"

  if [[ -s "$output_file" ]]; then
    log "Output exists, skipping: $output_file"
    return 0
  fi

  mkdir -p "$(dirname "$output_file")" "$(dirname "$log_file")"
  log "Running repeat=$repeat model=$inference_model turn=$turn context=$context memory=$memory_safe pref=$pref"

  local extra_args=()
  if [[ -n "$MAX_QUERIES" ]]; then
    extra_args+=(--max_queries "$MAX_QUERIES")
  fi
  if [[ -n "$memory_file" ]]; then
    extra_args+=(--memory_path "$memory_file")
  fi

  python "$INFERENCE_SCRIPT" \
    --input_path "$DATA_PATH" \
    --output_path "$output_file" \
    --log_path "$log_file" \
    --query_path "$QUERY_SINGLE" \
    --multiturn_path "$QUERY_MULTI" \
    --pref_list_path "$PREF_LIST" \
    --pref_group_path "$PREF_GROUP" \
    --tools_schema_path "$schema_file" \
    --context_type "$context" \
    --pref_type "$pref" \
    --turn "$turn" \
    --model_name "$inference_model" \
    --vllm_url "$vllm_url" \
    --memory_safe_name "$memory_safe" \
    --concurrency "$CONCURRENCY" \
    "${extra_args[@]}" 2>&1 | tee -a "$RUN_LOG"
}

run_model_jobs() {
  local inference_model="$1" vllm_url="$2"
  local repeat context memory_safe pref turn
  for repeat in "${REPEATS[@]}"; do
    for context in "${CONTEXTS[@]}"; do
      while read -r memory_safe; do
        [[ -n "$memory_safe" ]] || continue
        for pref in "${PREF_TYPES[@]}"; do
          for turn in "${TURNS[@]}"; do
            run_inference_one "$repeat" "$turn" "$inference_model" "$vllm_url" "$context" "$pref" "$memory_safe" || return 1
          done
        done
      done < <(context_memory_sources_csv "$context")
    done
  done
}

run_evaluation() {
  local eval_args=(--parent-run-id "$PARENT_RUN_ID")
  if [[ "$ALLOW_MISSING_EVAL" == "1" ]]; then
    eval_args+=(--allow-missing)
  fi
  python "$EVAL_SCRIPT" "${eval_args[@]}" 2>&1 | tee -a "$RUN_LOG"
}

main() {
  cd "$ROOT_DIR" || exit 1
  export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"
  mkdir -p "$LOG_ROOT" "$RUN_ROOT" "$RESULT_ROOT"

  resolve_memory_sources || exit 1
  log "PARENT_RUN_ID=$PARENT_RUN_ID"
  log "Data: $DATA_PATH"
  log "Requested memory root: $SOURCE_MEMORY_ROOT"
  log "Effective memory root: $EFFECTIVE_MEMORY_ROOT"
  log "Inference models: ${INFERENCE_MODELS[*]}"
  log "vLLM URLs: ${VLLM_URL_LIST[*]}"
  log "Memory sources: ${MEMORY_MODELS[*]}"
  log "Contexts: ${CONTEXTS[*]}"
  log "Turns: ${TURNS[*]} Prefs: ${PREF_TYPES[*]} Repeats: ${REPEATS[*]}"
  [[ -n "$MAX_QUERIES" ]] && log "MAX_QUERIES=$MAX_QUERIES"

  write_metadata | tee -a "$RUN_LOG"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN=1 complete. Metadata: $RUN_ROOT/metadata.json"
    exit 0
  fi

  validate_vllm_urls || exit 1

  local worker_pids=()
  local idx model url
  for idx in "${!INFERENCE_MODELS[@]}"; do
    model="${INFERENCE_MODELS[$idx]}"
    url="${VLLM_URL_LIST[$idx]:-}"
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

  log "Inference run finished. Outputs: $RUN_ROOT"
  if [[ "$RUN_EVALUATION" == "1" ]]; then
    run_evaluation
    log "Evaluation finished. Results: $RESULT_ROOT"
  fi
}

main "$@"
