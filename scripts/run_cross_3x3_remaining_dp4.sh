#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
RUN_TAG="${RUN_TAG:-cross_3x3_remaining_dp4_$(date '+%Y%m%d_%H%M%S')}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/outputs/runtime/cross_memory_3x3}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/${RUN_TAG}.queue.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${RUN_TAG}.queue.pid}"

mkdir -p "${LOG_DIR}"
touch "${RUN_LOG}"
printf '%s\n' "$$" >"${PID_FILE}"
trap 'rm -f "${PID_FILE}"' EXIT

status() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${RUN_LOG}"
}

status "Starting remaining cross-memory 3x3 queue"
status "GPUs=${CUDA_DEVICES}; completed qwen3_8b -> qwen3_8b will be skipped"

profiles=(qwen3-8b gemma4-12b gpt-oss-20b)
for profile in "${profiles[@]}"; do
    status "Starting target profile: ${profile}"
    CUDA_DEVICES="${CUDA_DEVICES}" \
        "${SCRIPT_DIR}/run_cross_target_memory_dp4.sh" "${profile}" \
        2>&1 | tee -a "${RUN_LOG}"
    status "Target profile complete: ${profile}"
done

status "Remaining cross-memory 3x3 queue complete: 8/8 combinations"
