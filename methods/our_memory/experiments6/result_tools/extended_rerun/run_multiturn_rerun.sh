#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_RUNNER="${SCRIPT_DIR}/run_rerun_job.py"

TARGET_MEMORY_MODELS="${TARGET_MEMORY_MODELS:-gpt-4o-mini|google_gemma-3-12b-it}"
TARGET_ACTION_MODELS="${TARGET_ACTION_MODELS:-gpt-5|gemini-3-flash-preview}"
CONTEXT_TYPE="${CONTEXT_TYPE:-memory_api}"
PREF_TYPE="${PREF_TYPE:-hard}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
CONCURRENCY="${CONCURRENCY:-10}"
DRY_RUN="${DRY_RUN:-0}"

IFS='|' read -r -a MEMORY_MODELS <<< "${TARGET_MEMORY_MODELS}"
IFS='|' read -r -a ACTION_MODELS <<< "${TARGET_ACTION_MODELS}"

echo "========================================================"
echo "Multi-turn rerun jobs"
echo "  memory models : ${TARGET_MEMORY_MODELS}"
echo "  action models : ${TARGET_ACTION_MODELS}"
echo "  context/pref  : ${CONTEXT_TYPE} / ${PREF_TYPE}"
echo "========================================================"

for memory_model in "${MEMORY_MODELS[@]}"; do
    [ -n "${memory_model}" ] || continue
    for action_model in "${ACTION_MODELS[@]}"; do
        [ -n "${action_model}" ] || continue

        echo "--------------------------------------------------------"
        echo "[RUN] multiturn memory=${memory_model} action=${action_model}"

        cmd=(
            python "${JOB_RUNNER}"
            --turn-type multiturn
            --memory-model "${memory_model}"
            --action-model "${action_model}"
            --context-type "${CONTEXT_TYPE}"
            --pref-type "${PREF_TYPE}"
            --reasoning-effort "${REASONING_EFFORT}"
            --concurrency "${CONCURRENCY}"
        )

        if [ "${DRY_RUN}" = "1" ]; then
            cmd+=(--dry-run)
        fi

        "${cmd[@]}"
    done
done

echo "========================================================"
echo "Multi-turn rerun jobs finished"
echo "========================================================"
