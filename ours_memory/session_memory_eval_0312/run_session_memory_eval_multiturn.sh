#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$SCRIPT_DIR"

PYTHON_SCRIPT="/data/minseo/experiments4/ours_memory/Preference_Memory_step2_ACTION_multiturn_VLLM.py"
MULTITURN_PATH="/data/minseo/experiments4/query_multiturn-domain.json"
PREF_LIST_PATH="/data/minseo/experiments4/pref_list.json"
PREF_GROUP_PATH="/data/minseo/experiments4/pref_group.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments4/schema_all.json"

PREP_MANIFEST="${PREP_MANIFEST:-$PACKAGE_ROOT/outputs/prepared/prep_manifest.jsonl}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PACKAGE_ROOT/outputs/runs/multiturn}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

ACTION_MODELS_CSV="${ACTION_MODELS_CSV:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B,deepseek-ai/DeepSeek-R1-Distill-Qwen-7B,google/gemma-3-12b-it,google/codegemma-7b-it}"
PREF_TYPES_CSV="${PREF_TYPES_CSV:-easy,medium,hard}"
PROMPT_NAME="${PROMPT_NAME:-implicit_zs}"
CONTEXT_TYPE="memory_api"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
PORT="${PORT:-8003}"
BASE_URL_WAS_SET=0
if [[ -n "${BASE_URL:-}" ]]; then
    BASE_URL_WAS_SET=1
fi
BASE_URL="${BASE_URL:-http://localhost:${PORT}/v1}"
CONCURRENCY="${CONCURRENCY:-50}"

LIMIT_MEMORY_MODEL="${LIMIT_MEMORY_MODEL:-}"
LIMIT_COHORT="${LIMIT_COHORT:-all}"
LIMIT_SESSION_INDEX="${LIMIT_SESSION_INDEX:-}"
LIMIT_PREF_TYPE="${LIMIT_PREF_TYPE:-}"
LIMIT_ACTION_MODEL="${LIMIT_ACTION_MODEL:-}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

DRY_RUN="${DRY_RUN:-0}"
SERVER_PID=""

print_help() {
    cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --prep-manifest PATH
  --output-root PATH
  --run-tag TAG
  --action-models CSV
  --pref-types CSV
  --prompt-name NAME
  --gpu-ids IDS
  --tp-size N
  --port PORT
  --base-url URL
  --concurrency N
  --multiturn-path PATH
  --tools-schema-path PATH
  --limit-memory-model NAME
  --limit-cohort NAME
  --limit-session-index N
  --limit-pref-type NAME
  --limit-action-model NAME
  --overwrite
  --dry-run
  --help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prep-manifest)
            PREP_MANIFEST="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --run-tag)
            RUN_TAG="$2"
            shift 2
            ;;
        --action-models)
            ACTION_MODELS_CSV="$2"
            shift 2
            ;;
        --pref-types)
            PREF_TYPES_CSV="$2"
            shift 2
            ;;
        --prompt-name)
            PROMPT_NAME="$2"
            shift 2
            ;;
        --gpu-ids)
            GPU_IDS="$2"
            shift 2
            ;;
        --tp-size)
            TP_SIZE="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --base-url)
            BASE_URL="$2"
            BASE_URL_WAS_SET=1
            shift 2
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --multiturn-path)
            MULTITURN_PATH="$2"
            shift 2
            ;;
        --tools-schema-path)
            TOOLS_SCHEMA_PATH="$2"
            shift 2
            ;;
        --limit-memory-model)
            LIMIT_MEMORY_MODEL="$2"
            shift 2
            ;;
        --limit-cohort)
            LIMIT_COHORT="$2"
            shift 2
            ;;
        --limit-session-index)
            LIMIT_SESSION_INDEX="$2"
            shift 2
            ;;
        --limit-pref-type)
            LIMIT_PREF_TYPE="$2"
            shift 2
            ;;
        --limit-action-model)
            LIMIT_ACTION_MODEL="$2"
            shift 2
            ;;
        --overwrite)
            SKIP_EXISTING=0
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        --help)
            print_help
            exit 0
            ;;
        *)
            echo "[ERROR] Unknown argument: $1" >&2
            print_help
            exit 1
            ;;
    esac
done

IFS=',' read -r -a ACTION_MODELS <<< "$ACTION_MODELS_CSV"
IFS=',' read -r -a PREF_TYPES <<< "$PREF_TYPES_CSV"

if [[ "$BASE_URL_WAS_SET" == "0" ]]; then
    BASE_URL="http://localhost:${PORT}/v1"
fi

RUN_ROOT="$OUTPUT_ROOT/$RUN_TAG"
RUN_MANIFEST="$RUN_ROOT/run_manifest.jsonl"
VLLM_SERVER_LOG="$RUN_ROOT/vllm_server.log"
mkdir -p "$RUN_ROOT"
touch "$VLLM_SERVER_LOG"

declare -A RECORDED_OUTPUTS=()
if [[ -f "$RUN_MANIFEST" ]]; then
    while IFS= read -r recorded_output; do
        if [[ -n "$recorded_output" ]]; then
            RECORDED_OUTPUTS["$recorded_output"]=1
        fi
    done < <(
        python - "$RUN_MANIFEST" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        output_path = row.get("output_path") or row.get("json_path")
        if output_path:
            print(output_path)
PY
    )
fi

cleanup() {
    if [[ -n "$SERVER_PID" ]]; then
        echo ""
        echo "[WARN] Killing vLLM server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

trap cleanup SIGINT SIGTERM ERR

wait_for_server() {
    local log_path="$1"
    local retries=120
    local count=0

    echo "Waiting for vLLM server at $BASE_URL..."
    while ! curl -s "$BASE_URL/models" > /dev/null; do
        sleep 5
        echo -n "."
        count=$((count + 1))

        if ! ps -p "$SERVER_PID" > /dev/null; then
            echo ""
            echo "[ERROR] vLLM server died unexpectedly."
            cat "$log_path"
            cleanup
            exit 1
        fi

        if [[ $count -ge $retries ]]; then
            echo ""
            echo "[ERROR] Timeout waiting for vLLM server."
            cleanup
            exit 1
        fi
    done
    echo ""
    echo ">> Server is READY!"
}

append_manifest_row() {
    local run_manifest="$1"
    local run_tag="$2"
    local action_model="$3"
    local action_model_safe="$4"
    local memory_model="$5"
    local cohort="$6"
    local session_index="$7"
    local pref_type="$8"
    local eligible_count="$9"
    local input_path="${10}"
    local memory_path="${11}"
    local output_path="${12}"
    local log_path="${13}"
    local prompt_name="${14}"
    local context_type="${15}"

    python - "$run_manifest" "$run_tag" "$action_model" "$action_model_safe" "$memory_model" "$cohort" "$session_index" "$pref_type" "$eligible_count" "$input_path" "$memory_path" "$output_path" "$log_path" "$prompt_name" "$context_type" <<'PY'
import json
import sys

(
    manifest_path,
    run_tag,
    action_model,
    action_model_safe,
    memory_model,
    cohort,
    session_index,
    pref_type,
    eligible_count,
    input_path,
    memory_path,
    output_path,
    log_path,
    prompt_name,
    context_type,
) = sys.argv[1:]

row = {
    "run_tag": run_tag,
    "action_model": action_model,
    "action_model_safe": action_model_safe,
    "memory_model": memory_model,
    "cohort": cohort,
    "session_index": int(session_index),
    "pref_type": pref_type,
    "prompt_name": prompt_name,
    "context_type": context_type,
    "eligible_count": int(eligible_count),
    "input_path": input_path,
    "memory_path": memory_path,
    "output_path": output_path,
    "json_path": output_path,
    "log_path": log_path,
    "status": "ok",
}

with open(manifest_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(row, ensure_ascii=False) + "\n")
PY
}

output_is_valid_json() {
    local output_path="$1"

    python - "$output_path" <<'PY'
import json
import sys

path = sys.argv[1]

try:
    with open(path, "r", encoding="utf-8") as f:
        json.load(f)
except Exception:
    raise SystemExit(1)

raise SystemExit(0)
PY
}

ensure_manifest_row_for_existing_output() {
    local run_manifest="$1"
    local run_tag="$2"
    local action_model="$3"
    local action_model_safe="$4"
    local memory_model="$5"
    local cohort="$6"
    local session_index="$7"
    local pref_type="$8"
    local eligible_count="$9"
    local input_path="${10}"
    local memory_path="${11}"
    local output_path="${12}"
    local log_path="${13}"
    local prompt_name="${14}"
    local context_type="${15}"

    if [[ -n "${RECORDED_OUTPUTS[$output_path]+x}" ]]; then
        return 0
    fi

    append_manifest_row \
        "$run_manifest" \
        "$run_tag" \
        "$action_model" \
        "$action_model_safe" \
        "$memory_model" \
        "$cohort" \
        "$session_index" \
        "$pref_type" \
        "$eligible_count" \
        "$input_path" \
        "$memory_path" \
        "$output_path" \
        "$log_path" \
        "$prompt_name" \
        "$context_type"

    RECORDED_OUTPUTS["$output_path"]=1
}

iter_manifest_jobs() {
    python - "$PREP_MANIFEST" "$LIMIT_MEMORY_MODEL" "$LIMIT_COHORT" "$LIMIT_SESSION_INDEX" <<'PY'
import json
import sys

manifest_path, limit_memory_model, limit_cohort, limit_session_index = sys.argv[1:]
limit_session_index = int(limit_session_index) if limit_session_index else None

with open(manifest_path, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)
        if limit_memory_model and row["memory_model"] != limit_memory_model:
            continue
        if limit_cohort and row["cohort"] != limit_cohort:
            continue
        if limit_session_index is not None and int(row["session_index"]) != limit_session_index:
            continue
        print(
            "\t".join(
                [
                    row["memory_model"],
                    row["cohort"],
                    str(row["session_index"]),
                    str(row["eligible_count"]),
                    row["dataset_path"],
                    row["memory_path"],
                ]
            )
        )
PY
}

if [[ ! -f "$PREP_MANIFEST" ]]; then
    echo "[ERROR] Prep manifest not found: $PREP_MANIFEST" >&2
    exit 1
fi

echo "========================================================"
echo "Starting standalone session-memory multiturn runs"
echo "========================================================"
echo "Prep manifest   : $PREP_MANIFEST"
echo "Run root        : $RUN_ROOT"
echo "Run tag         : $RUN_TAG"
echo "Multiturn path  : $MULTITURN_PATH"
echo "Tools schema    : $TOOLS_SCHEMA_PATH"
echo "vLLM log        : $VLLM_SERVER_LOG"

mapfile -t PREP_JOBS < <(iter_manifest_jobs)
if [[ ${#PREP_JOBS[@]} -eq 0 ]]; then
    echo "[ERROR] No prep jobs matched the current filters." >&2
    exit 1
fi

for ACTION_MODEL in "${ACTION_MODELS[@]}"; do
    if [[ -n "$LIMIT_ACTION_MODEL" && "$ACTION_MODEL" != "$LIMIT_ACTION_MODEL" ]]; then
        continue
    fi

    ACTION_MODEL_SAFE="${ACTION_MODEL//\//_}"
    SERVER_LOG_PATH="$VLLM_SERVER_LOG"
    ACTION_MODEL_NEEDS_SERVER=0

    for JOB in "${PREP_JOBS[@]}"; do
        IFS=$'\t' read -r MEMORY_MODEL COHORT SESSION_INDEX ELIGIBLE_COUNT INPUT_PATH MEMORY_PATH <<< "$JOB"
        for PREF_TYPE in "${PREF_TYPES[@]}"; do
            if [[ -n "$LIMIT_PREF_TYPE" && "$PREF_TYPE" != "$LIMIT_PREF_TYPE" ]]; then
                continue
            fi

            CURRENT_OUT_DIR="$RUN_ROOT/$ACTION_MODEL_SAFE/$MEMORY_MODEL/$COHORT/session_${SESSION_INDEX}/$CONTEXT_TYPE/$PREF_TYPE/$PROMPT_NAME"
            OUTPUT_FILE="$CURRENT_OUT_DIR/result.json"

            if [[ "$SKIP_EXISTING" == "1" && -f "$OUTPUT_FILE" ]] && output_is_valid_json "$OUTPUT_FILE"; then
                continue
            fi

            ACTION_MODEL_NEEDS_SERVER=1
            break 2
        done
    done

    if [[ "$ACTION_MODEL_NEEDS_SERVER" != "1" ]]; then
        echo "####################################################################"
        echo "[SKIP] No pending jobs for: $ACTION_MODEL"
        echo "####################################################################"
        continue
    fi

    SERVER_STARTED=0

    echo "####################################################################"
    echo "[STEP 1] Starting vLLM server for: $ACTION_MODEL"
    echo "####################################################################"

    PARSER_FLAG="--tool-call-parser hermes"
    if [[ "$ACTION_MODEL" == *"Llama-3"* ]]; then
        PARSER_FLAG="--tool-call-parser llama3_json"
    elif [[ "$ACTION_MODEL" == *"Mistral"* ]]; then
        PARSER_FLAG="--tool-call-parser mistral"
    fi

    if [[ "$DRY_RUN" != "1" ]]; then
        {
            echo ""
            echo "===================================================================="
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting vLLM server for: $ACTION_MODEL"
            echo "===================================================================="
        } >> "$SERVER_LOG_PATH"

        CUDA_VISIBLE_DEVICES="$GPU_IDS" nohup vllm serve "$ACTION_MODEL" \
            --host 0.0.0.0 \
            --port "$PORT" \
            --tensor-parallel-size "$TP_SIZE" \
            --enable-auto-tool-choice \
            $PARSER_FLAG \
            --max-model-len 8192 \
            --gpu-memory-utilization 0.9 \
            --trust-remote-code >> "$SERVER_LOG_PATH" 2>&1 &

        SERVER_PID=$!
        SERVER_STARTED=1
        echo ">> vLLM Server PID: $SERVER_PID"
        wait_for_server "$SERVER_LOG_PATH"
    fi

    for JOB in "${PREP_JOBS[@]}"; do
        IFS=$'\t' read -r MEMORY_MODEL COHORT SESSION_INDEX ELIGIBLE_COUNT INPUT_PATH MEMORY_PATH <<< "$JOB"
        for PREF_TYPE in "${PREF_TYPES[@]}"; do
            if [[ -n "$LIMIT_PREF_TYPE" && "$PREF_TYPE" != "$LIMIT_PREF_TYPE" ]]; then
                continue
            fi

            CURRENT_OUT_DIR="$RUN_ROOT/$ACTION_MODEL_SAFE/$MEMORY_MODEL/$COHORT/session_${SESSION_INDEX}/$CONTEXT_TYPE/$PREF_TYPE/$PROMPT_NAME"
            CURRENT_LOG_DIR="$CURRENT_OUT_DIR"
            mkdir -p "$CURRENT_OUT_DIR" "$CURRENT_LOG_DIR"

            OUTPUT_FILE="$CURRENT_OUT_DIR/result.json"
            LOG_FILE="$CURRENT_LOG_DIR/result.log"

            if [[ "$SKIP_EXISTING" == "1" && -f "$OUTPUT_FILE" ]]; then
                if output_is_valid_json "$OUTPUT_FILE"; then
                    echo " >> [SKIP EXISTING]"
                    echo "    - Action Model  : $ACTION_MODEL"
                    echo "    - Memory Model  : $MEMORY_MODEL"
                    echo "    - Cohort        : $COHORT"
                    echo "    - Session Index : $SESSION_INDEX"
                    echo "    - Pref Type     : $PREF_TYPE"
                    echo "    - Output        : $OUTPUT_FILE"

                    ensure_manifest_row_for_existing_output \
                        "$RUN_MANIFEST" \
                        "$RUN_TAG" \
                        "$ACTION_MODEL" \
                        "$ACTION_MODEL_SAFE" \
                        "$MEMORY_MODEL" \
                        "$COHORT" \
                        "$SESSION_INDEX" \
                        "$PREF_TYPE" \
                        "$ELIGIBLE_COUNT" \
                        "$INPUT_PATH" \
                        "$MEMORY_PATH" \
                        "$OUTPUT_FILE" \
                        "$LOG_FILE" \
                        "$PROMPT_NAME" \
                        "$CONTEXT_TYPE"
                    continue
                fi

                echo " >> [RETRY INVALID OUTPUT]"
                echo "    - Action Model  : $ACTION_MODEL"
                echo "    - Memory Model  : $MEMORY_MODEL"
                echo "    - Cohort        : $COHORT"
                echo "    - Session Index : $SESSION_INDEX"
                echo "    - Pref Type     : $PREF_TYPE"
                echo "    - Output        : $OUTPUT_FILE"
            fi

            echo " >> [RUNNING]"
            echo "    - Action Model  : $ACTION_MODEL"
            echo "    - Memory Model  : $MEMORY_MODEL"
            echo "    - Cohort        : $COHORT"
            echo "    - Session Index : $SESSION_INDEX"
            echo "    - Pref Type     : $PREF_TYPE"
            echo "    - Eligible      : $ELIGIBLE_COUNT"

            if [[ "$DRY_RUN" == "1" ]]; then
                echo "    - DRY_RUN       : python $PYTHON_SCRIPT --input_path $INPUT_PATH --memory_path $MEMORY_PATH --output_path $OUTPUT_FILE ..."
                continue
            fi

            python "$PYTHON_SCRIPT" \
                --input_path "$INPUT_PATH" \
                --memory_path "$MEMORY_PATH" \
                --output_path "$OUTPUT_FILE" \
                --log_path "$LOG_FILE" \
                --multiturn_path "$MULTITURN_PATH" \
                --pref_list_path "$PREF_LIST_PATH" \
                --pref_group_path "$PREF_GROUP_PATH" \
                --tools_schema_path "$TOOLS_SCHEMA_PATH" \
                --context_type "$CONTEXT_TYPE" \
                --pref_type "$PREF_TYPE" \
                --model_name "$ACTION_MODEL" \
                --base_url "$BASE_URL" \
                --api_key "EMPTY" \
                --concurrency "$CONCURRENCY"

            append_manifest_row \
                "$RUN_MANIFEST" \
                "$RUN_TAG" \
                "$ACTION_MODEL" \
                "$ACTION_MODEL_SAFE" \
                "$MEMORY_MODEL" \
                "$COHORT" \
                "$SESSION_INDEX" \
                "$PREF_TYPE" \
                "$ELIGIBLE_COUNT" \
                "$INPUT_PATH" \
                "$MEMORY_PATH" \
                "$OUTPUT_FILE" \
                "$LOG_FILE" \
                "$PROMPT_NAME" \
                "$CONTEXT_TYPE"

            RECORDED_OUTPUTS["$OUTPUT_FILE"]=1
        done
    done

    if [[ "$DRY_RUN" != "1" && "$SERVER_STARTED" == "1" ]]; then
        echo "[STEP 2] Stopping vLLM server..."
        cleanup
        echo ">> Server stopped. Cooling down..."
        sleep 10
    fi
done

echo "========================================================"
echo "Finished standalone session-memory multiturn runs"
echo "========================================================"
if [[ "$DRY_RUN" != "1" ]]; then
    echo "Run manifest: $RUN_MANIFEST"
fi
