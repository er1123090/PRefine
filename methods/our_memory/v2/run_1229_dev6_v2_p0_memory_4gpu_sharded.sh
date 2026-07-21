#!/usr/bin/env bash

# Build ours_memory_v2 P0 memories using all 4 GPUs via data sharding.

set -u
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
RUN_ID="${RUN_ID:-1229_dev6_ours_memory_v2_p0_20260520}"
DATA_PATH="${DATA_PATH:-$ROOT_DIR/data/1229_dev_6.json}"
GPU_LIST=(${GPU_LIST:-0 1 2 3})
PORT_LIST=(${PORT_LIST:-8004 8005 8006 8007})
CONCURRENCY_PER_GPU="${CONCURRENCY_PER_GPU:-8}"
EXPECTED_MEMORY_ROWS="${EXPECTED_MEMORY_ROWS:-265}"
FORCE_RERUN="${FORCE_RERUN:-0}"

RUN_ROOT="$ROOT_DIR/outputs/our_memory/$RUN_ID"
MEMORY_ROOT="$RUN_ROOT/memories"
LOG_ROOT="$ROOT_DIR/logs/our_memory/$RUN_ID"
RUN_LOG="$LOG_ROOT/memory.4gpu_sharded.run.log"
WORKER_SCRIPT="$SCRIPT_DIR/run_1229_dev6_v2_p0_memory_shard_worker_vllm.sh"

MEMORY_SOURCE_MODELS=(
  "Qwen/Qwen3-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
  "google/gemma-3-12b-it"
  "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
)

if [[ -n "${ONLY_MEMORY_MODELS:-}" ]]; then
  read -r -a MEMORY_SOURCE_MODELS <<< "$ONLY_MEMORY_MODELS"
fi

log() {
  mkdir -p "$LOG_ROOT"
  printf '[%s] %s\n' "$(date '+%Y-%m-%dT%H:%M:%S%z')" "$*" | tee -a "$RUN_LOG"
}

model_safe() {
  printf '%s' "${1//\//_}"
}

memory_complete() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  local rows
  rows="$(wc -l < "$path" | tr -d ' ')"
  [[ "$rows" -ge "$EXPECTED_MEMORY_ROWS" ]]
}

split_dataset() {
  local shard_root="$1"
  python - "$DATA_PATH" "$shard_root" "${#GPU_LIST[@]}" <<'PY'
import json
import sys
from pathlib import Path

data_path = Path(sys.argv[1])
shard_root = Path(sys.argv[2])
n = int(sys.argv[3])
data = json.loads(data_path.read_text(encoding="utf-8"))
if not isinstance(data, list):
    data = [data]
shard_root.mkdir(parents=True, exist_ok=True)
for idx in range(n):
    shard = data[idx::n]
    (shard_root / f"input_shard_{idx}.json").write_text(
        json.dumps(shard, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
PY
}

merge_shards() {
  local model="$1"
  local out_dir="$2"
  local shard_root="$3"
  local memory_file="$out_dir/_memory1.jsonl"
  local verifier_file="$out_dir/_verifier_logs1.jsonl"
  local refinement_file="$out_dir/_refinement_logs1.jsonl"
  local tmp_memory="$memory_file.tmp"
  local tmp_verifier="$verifier_file.tmp"
  local tmp_refinement="$refinement_file.tmp"

  : > "$tmp_memory"
  : > "$tmp_verifier"
  : > "$tmp_refinement"

  local idx shard_dir shard_memory
  for idx in "${!GPU_LIST[@]}"; do
    shard_dir="$shard_root/shard_$idx"
    shard_memory="$shard_dir/_memory1.jsonl"
    if [[ ! -f "$shard_memory" ]]; then
      log "Missing shard output for model=$model shard=$idx file=$shard_memory"
      return 1
    fi
    cat "$shard_memory" >> "$tmp_memory"
    [[ -f "$shard_dir/_verifier_logs1.jsonl" ]] && cat "$shard_dir/_verifier_logs1.jsonl" >> "$tmp_verifier"
    [[ -f "$shard_dir/_refinement_logs1.jsonl" ]] && cat "$shard_dir/_refinement_logs1.jsonl" >> "$tmp_refinement"
  done

  local rows
  rows="$(wc -l < "$tmp_memory" | tr -d ' ')"
  if [[ "$rows" -lt "$EXPECTED_MEMORY_ROWS" ]]; then
    log "Merged memory is incomplete for model=$model rows=$rows expected=$EXPECTED_MEMORY_ROWS"
    return 1
  fi

  mv "$tmp_memory" "$memory_file"
  mv "$tmp_verifier" "$verifier_file"
  mv "$tmp_refinement" "$refinement_file"
  ln -sf "_memory1.jsonl" "$out_dir/memory_result.jsonl"
  log "Merged sharded memory model=$model rows=$rows file=$memory_file"
}

run_model_sharded() {
  local model="$1"
  local safe out_dir memory_file shard_root ts
  safe="$(model_safe "$model")"
  out_dir="$MEMORY_ROOT/$safe"
  memory_file="$out_dir/_memory1.jsonl"
  mkdir -p "$out_dir"

  if [[ "$FORCE_RERUN" != "1" ]] && memory_complete "$memory_file"; then
    log "Memory already complete, skipping model=$model file=$memory_file"
    return 0
  fi

  ts="$(date '+%Y%m%d_%H%M%S')"
  shard_root="$out_dir/shards/$ts"
  mkdir -p "$shard_root"
  split_dataset "$shard_root"

  log "Starting 4GPU sharded memory model=$model gpus=${GPU_LIST[*]} ports=${PORT_LIST[*]}"
  local pids=()
  local idx gpu port shard_input shard_out
  for idx in "${!GPU_LIST[@]}"; do
    gpu="${GPU_LIST[$idx]}"
    port="${PORT_LIST[$idx]}"
    shard_input="$shard_root/input_shard_$idx.json"
    shard_out="$shard_root/shard_$idx"
    mkdir -p "$shard_out"
    (
      RUN_ID="$RUN_ID" \
      DATA_PATH="$shard_input" \
      MEMORY_MODEL="$model" \
      OUTPUT_DIR="$shard_out" \
      GPU_IDS="$gpu" \
      PORT="$port" \
      TP_SIZE=1 \
      CONCURRENCY="$CONCURRENCY_PER_GPU" \
      SHARD_ID="${safe}_$idx" \
      bash "$WORKER_SCRIPT"
    ) >> "$LOG_ROOT/memory_shards/${safe}_shard_${idx}.launcher.log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    log "One or more shards failed for model=$model"
    return 1
  fi

  merge_shards "$model" "$out_dir" "$shard_root"
}

main() {
  cd "$ROOT_DIR" || exit 1
  mkdir -p "$MEMORY_ROOT" "$LOG_ROOT/memory_shards"
  log "Run ID: $RUN_ID"
  log "Data path: $DATA_PATH"
  log "Memory source models: ${MEMORY_SOURCE_MODELS[*]}"
  log "4GPU sharding: gpus=${GPU_LIST[*]} ports=${PORT_LIST[*]} concurrency_per_gpu=$CONCURRENCY_PER_GPU"

  local model
  for model in "${MEMORY_SOURCE_MODELS[@]}"; do
    run_model_sharded "$model" || exit 1
  done

  log "4GPU sharded memory run finished"
}

main "$@"
