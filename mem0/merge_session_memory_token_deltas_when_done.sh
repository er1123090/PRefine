#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <run_dir>" >&2
  exit 1
fi

RUN_DIR="$1"
PID_FILE="${RUN_DIR}/shard_pids.txt"
MERGE_LOG="${RUN_DIR}/merge_watcher.log"
MERGE_WATCHER_PID_FILE="${RUN_DIR}/merge_watcher.pid"
BASE_DIR="/data/minseo/experiments4/mem0"

exec >> "${MERGE_LOG}" 2>&1
printf '%s\n' "$$" > "${MERGE_WATCHER_PID_FILE}"

echo "Merge watcher started at: $(date --iso-8601=seconds)"
echo "Run directory: ${RUN_DIR}"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "Missing shard PID file: ${PID_FILE}" >&2
  exit 1
fi

mapfile -t SHARD_PIDS < "${PID_FILE}"

while true; do
  alive=0
  for pid in "${SHARD_PIDS[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      alive=1
      break
    fi
  done

  if [[ "${alive}" -eq 0 ]]; then
    break
  fi
  sleep 60
done

python "${BASE_DIR}/merge_session_memory_token_deltas.py" \
  --input_dir "${RUN_DIR}" \
  --csv_glob "shard*.csv" \
  --summary_glob "shard*.summary.json" \
  --output_csv "${RUN_DIR}/merged.csv" \
  --output_summary "${RUN_DIR}/merged.summary.json"

echo "Merge watcher finished at: $(date --iso-8601=seconds)"
