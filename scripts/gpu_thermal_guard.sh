#!/usr/bin/env bash
set -euo pipefail

GPU_IDS="${GPU_IDS:-0,1,2,3}"
TARGET_PIDS="${TARGET_PIDS:?TARGET_PIDS must contain comma-separated GPU process IDs}"
PAUSE_TEMP_C="${PAUSE_TEMP_C:-85}"
RESUME_TEMP_C="${RESUME_TEMP_C:-78}"
POLL_SECONDS="${POLL_SECONDS:-5}"
LOG_PATH="${LOG_PATH:-/tmp/gpu_thermal_guard.log}"

IFS=',' read -r -a target_pids <<<"${TARGET_PIDS}"
paused=0

log() {
    local message="$1"
    printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "${message}" | tee -a "${LOG_PATH}"
}

signal_live_targets() {
    local signal="$1"
    local pid
    for pid in "${target_pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            kill "-${signal}" "${pid}"
        fi
    done
}

any_target_alive() {
    local pid
    for pid in "${target_pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            return 0
        fi
    done
    return 1
}

resume_on_exit() {
    if (( paused == 1 )); then
        signal_live_targets CONT
        log "Guard exiting; resumed paused GPU processes"
    fi
}
trap resume_on_exit EXIT
trap 'exit 0' INT TERM

mkdir -p "$(dirname "${LOG_PATH}")"
log "Thermal guard started: GPUs=${GPU_IDS}, pause>=${PAUSE_TEMP_C}C, resume<=${RESUME_TEMP_C}C, PIDs=${TARGET_PIDS}"

while any_target_alive; do
    mapfile -t temperatures < <(
        nvidia-smi \
            --id="${GPU_IDS}" \
            --query-gpu=temperature.gpu \
            --format=csv,noheader,nounits
    )

    max_temp=0
    for temperature in "${temperatures[@]}"; do
        temperature="${temperature//[[:space:]]/}"
        if [[ "${temperature}" =~ ^[0-9]+$ ]] && (( temperature > max_temp )); then
            max_temp="${temperature}"
        fi
    done

    if (( paused == 0 && max_temp >= PAUSE_TEMP_C )); then
        signal_live_targets STOP
        paused=1
        log "Paused GPU processes at ${max_temp}C"
    elif (( paused == 1 && max_temp <= RESUME_TEMP_C )); then
        signal_live_targets CONT
        paused=0
        log "Resumed GPU processes at ${max_temp}C"
    fi

    sleep "${POLL_SECONDS}"
done

log "All target GPU processes exited; thermal guard finished"
