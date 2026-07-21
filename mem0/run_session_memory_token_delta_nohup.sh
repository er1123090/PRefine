#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/data/minseo/experiments4/mem0"
DATASET_PATH="/data/minseo/experiments4/data/1229_dev_6.json"
OUTPUT_ROOT="${BASE_DIR}/output"
LATEST_RUN_FILE="${OUTPUT_ROOT}/latest_session_memory_token_deltas_1229_dev_6.txt"

if [[ -z "${MEM0_API_KEY:-}" ]]; then
  echo "MEM0_API_KEY is not set." >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}"

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="session_memory_token_deltas_1229_dev_6_${RUN_STAMP}"
RUN_DIR="${OUTPUT_ROOT}/${RUN_NAME}"
mkdir -p "${RUN_DIR}"

printf '%s\n' "${RUN_DIR}" > "${LATEST_RUN_FILE}"

exec >> "${RUN_DIR}/launcher.log" 2>&1

echo "Run directory: ${RUN_DIR}"
echo "Started at: $(date --iso-8601=seconds)"

python -c '
import json
from datetime import datetime

config = {
    "dataset_path": "/data/minseo/experiments4/data/1229_dev_6.json",
    "output_root": "/data/minseo/experiments4/mem0/0323_memory-tokens_265",
    "started_at": datetime.now().astimezone().isoformat(),
    "shards": [
        {"id": 0, "start_example": 0, "end_example": 45},
        {"id": 1, "start_example": 45, "end_example": 88},
        {"id": 2, "start_example": 88, "end_example": 131},
        {"id": 3, "start_example": 131, "end_example": 176},
        {"id": 4, "start_example": 176, "end_example": 220},
        {"id": 5, "start_example": 220, "end_example": 265},
    ],
}
print(json.dumps(config, indent=2))
' > "${RUN_DIR}/run_config.json"

declare -a SHARD_IDS=(0 1 2 3 4 5)
declare -a SHARD_STARTS=(0 45 88 131 176 220)
declare -a SHARD_ENDS=(45 88 131 176 220 265)
declare -a PID_FILES=()

for i in "${!SHARD_IDS[@]}"; do
  shard_id="${SHARD_IDS[$i]}"
  start_example="${SHARD_STARTS[$i]}"
  end_example="${SHARD_ENDS[$i]}"
  shard_log="${RUN_DIR}/shard${shard_id}.log"
  shard_csv="${RUN_DIR}/shard${shard_id}.csv"
  shard_summary="${RUN_DIR}/shard${shard_id}.summary.json"
  shard_pid_file="${RUN_DIR}/shard${shard_id}.pid"
  shard_wrapper="${RUN_DIR}/launch_shard${shard_id}.sh"
  app_id="experiments4-session-token-delta-${RUN_STAMP}-shard${shard_id}"

  echo "Launching shard ${shard_id}: examples ${start_example}-${end_example}"
  cat > "${shard_wrapper}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "\$\$" > "${shard_pid_file}"
exec python -u "${BASE_DIR}/measure_session_memory_token_delta.py" \
  --input_path "${DATASET_PATH}" \
  --start_example "${start_example}" \
  --end_example "${end_example}" \
  --skip_delete_all \
  --continue_on_error \
  --app_id "${app_id}" \
  --output_csv "${shard_csv}" \
  --summary_json "${shard_summary}" \
  > "${shard_log}" 2>&1 < /dev/null
EOF
  chmod +x "${shard_wrapper}"
  setsid -f bash "${shard_wrapper}"
  PID_FILES+=("${shard_pid_file}")
done

for pid_file in "${PID_FILES[@]}"; do
  for _ in $(seq 1 50); do
    [[ -s "${pid_file}" ]] && break
    sleep 0.1
  done
done

> "${RUN_DIR}/shard_pids.txt"
for pid_file in "${PID_FILES[@]}"; do
  [[ -s "${pid_file}" ]] || continue
  cat "${pid_file}" >> "${RUN_DIR}/shard_pids.txt"
done

setsid -f bash "${BASE_DIR}/merge_session_memory_token_deltas_when_done.sh" "${RUN_DIR}"
for _ in $(seq 1 50); do
  [[ -s "${RUN_DIR}/merge_watcher.pid" ]] && break
  sleep 0.1
done
MERGE_WATCHER_PID="$(cat "${RUN_DIR}/merge_watcher.pid" 2>/dev/null || true)"

echo "Shard PIDs: $(tr '\n' ' ' < "${RUN_DIR}/shard_pids.txt")"
echo "Merge watcher PID: ${MERGE_WATCHER_PID}"
echo "Launcher finished at: $(date --iso-8601=seconds)"
