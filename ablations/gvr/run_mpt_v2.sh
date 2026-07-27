#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_PYTHON="/data/minseo/.venvs/experiment8/bin/python"
if [[ ! -x "${DEFAULT_PYTHON}" ]]; then
  DEFAULT_PYTHON="/data/minseo/.venvs/vllm/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-${DEFAULT_PYTHON}}"
MODEL="${MODEL:-gpt-4o-mini}"
PROVIDER="${PROVIDER:-openai}"
API_BASE="${API_BASE:-}"
API_KEY="${API_KEY:-}"
INPUT_PATH="${INPUT_PATH:-${ROOT_DIR}/data/MPT_v2_mix600.json}"
CONCURRENCY="${CONCURRENCY:-20}"
MAX_RETRIES="${MAX_RETRIES:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/outputs/ablations/gvr}"
DRY_RUN=0

if [[ "${PROVIDER}" == "openrouter" && -z "${API_BASE}" ]]; then
  API_BASE="https://openrouter.ai/api/v1"
fi

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=1
fi

conditions=(
  "g_only:generation_only"
  "gv_no_refine:generate_verify"
  "gr_no_verifier:blind_refine_1"
  "gvr:verified_refine"
)

for condition in "${conditions[@]}"; do
  label="${condition%%:*}"
  mode="${condition#*:}"
  output_dir="${OUTPUT_ROOT}/${label}/${MODEL//\//__}"

  cmd=(
    "${PYTHON_BIN}" "${ROOT_DIR}/ablations/gvr/build_memory.py"
    --input "${INPUT_PATH}"
    --output "${output_dir}/memory.jsonl"
    --verifier_output "${output_dir}/verifier.jsonl"
    --refinement_output "${output_dir}/refinement.jsonl"
    --provider "${PROVIDER}"
    --model "${MODEL}"
    --concurrency "${CONCURRENCY}"
    --max_retries "${MAX_RETRIES}"
    --memory_mode "${mode}"
  )
  if [[ -n "${API_BASE}" ]]; then
    cmd+=(--api_base "${API_BASE}")
  fi
  if [[ -n "${API_KEY}" ]]; then
    cmd+=(--api_key "${API_KEY}")
  fi
  if [[ "${TRUE_BLIND:-0}" == "1" ]]; then
    cmd+=(--true_blind)
  fi

  printf '[%s] ' "${label}"
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "${DRY_RUN}" -eq 0 ]]; then
    mkdir -p "${output_dir}"
    "${cmd[@]}"
  fi
done
