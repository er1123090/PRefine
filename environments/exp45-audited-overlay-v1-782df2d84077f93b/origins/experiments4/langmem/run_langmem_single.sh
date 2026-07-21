#!/bin/bash

set -euo pipefail

PYTHON_SCRIPT="/data/minseo/experiments4/langmem/langmem_inference_single.py"
BASE_DIR="/data/minseo/experiments4"
INPUT_PATH="${BASE_DIR}/data/1229_dev_6.json"
QUERY_PATH="${BASE_DIR}/query_singleturn.json"
PREF_LIST_PATH="${BASE_DIR}/pref_list.json"
PREF_GROUP_PATH="${BASE_DIR}/pref_group.json"
TOOLS_SCHEMA_PATH="${BASE_DIR}/schema_easy.json"
MEMORY_ROOT="${MEMORY_ROOT:-${BASE_DIR}/langmem/memory_snapshots/semantic-custom}"
OUTPUT_ROOT="${BASE_DIR}/langmem/inference_single"
PROMPT_NAME="implicit_zs"
CANONICAL_SNAPSHOT_NAME="langmem_1229_dev_6.jsonl"
CANONICAL_MANIFEST_NAME="langmem_1229_dev_6.manifest.json"

GPU_IDS="${GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a RAW_GPU_ID_LIST <<< "${GPU_IDS}"
GPU_ID_LIST=()
for gpu_id in "${RAW_GPU_ID_LIST[@]}"; do
    gpu_id="${gpu_id// /}"
    [ -n "${gpu_id}" ] || continue
    GPU_ID_LIST+=("${gpu_id}")
done
GPU_COUNT="${#GPU_ID_LIST[@]}"
TP_SIZE="${TP_SIZE:-${GPU_COUNT}}"
PORT="${PORT:-8003}"
VLLM_BASE_URL="http://localhost:${PORT}/v1"
CONCURRENCY="${CONCURRENCY:-30}"
MAX_QUERIES="${MAX_QUERIES:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

if [ "${GPU_COUNT}" -eq 0 ]; then
    echo "[ERROR] GPU_IDS is empty. Set GPU_IDS to at least one visible GPU id."
    exit 1
fi

if [ "${TP_SIZE}" -gt "${GPU_COUNT}" ]; then
    echo "[ERROR] TP_SIZE=${TP_SIZE} is larger than the number of visible GPUs (${GPU_COUNT}) from GPU_IDS='${GPU_IDS}'."
    echo "        Set TP_SIZE<=${GPU_COUNT}, or adjust GPU_IDS."
    exit 1
fi

# Remote API configuration:
# - GPT family uses OPENAI_API_KEY and optionally OPENAI_CHAT_BASE_URL / OPENAI_BASE_URL.
# - Gemini family uses GOOGLE_API_KEY (or GEMINI_CHAT_API_KEY / GEMINI_API_KEY)
#   directly, or GEMINI_CHAT_BASE_URL / GEMINI_BASE_URL for an OpenAI-compatible proxy.
OPENAI_CHAT_BASE_URL="${OPENAI_CHAT_BASE_URL:-${OPENAI_BASE_URL:-}}"
GEMINI_CHAT_BASE_URL="${GEMINI_CHAT_BASE_URL:-${GEMINI_BASE_URL:-}}"
GEMINI_CHAT_API_KEY="${GEMINI_CHAT_API_KEY:-${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}}"
DEFAULT_EMBEDDING_MODEL="${DEFAULT_EMBEDDING_MODEL:-text-embedding-3-small}"
DEFAULT_EMBEDDING_BASE_URL="${DEFAULT_EMBEDDING_BASE_URL:-}"
DEFAULT_EMBEDDING_API_KEY="${DEFAULT_EMBEDDING_API_KEY:-${OPENAI_API_KEY:-}}"
REASONING_EFFORT="${REASONING_EFFORT:-minimal}"  # minimal|low|medium|high
VLLM_PREFER_LOCAL_SNAPSHOT="${VLLM_PREFER_LOCAL_SNAPSHOT:-1}"

MODELS=(
    #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    # "google/gemma-3-12b-it"
    # "google/codegemma-7b-it"
    # "gpt-4o-mini"
    # "gpt-5-mini"
    "gpt-5"
    #"gemini-3-flash-preview"
)

# Leave empty to use every discovered memory snapshot.
MEMORY_MODELS=(
    # "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "google/gemma-3-12b-it"
    "gpt-4o-mini"
    #"Qwen/Qwen3-8B"
)

CONTEXT_TYPES=("memory_api")
PREF_TYPES=("easy" "medium" "hard")

if [ -n "${TARGET_MODELS:-}" ]; then
    IFS='|' read -r -a MODELS <<< "${TARGET_MODELS}"
fi

if [ -n "${TARGET_CONTEXT_TYPES:-}" ]; then
    IFS='|' read -r -a CONTEXT_TYPES <<< "${TARGET_CONTEXT_TYPES}"
fi

if [ -n "${TARGET_PREF_TYPES:-}" ]; then
    IFS='|' read -r -a PREF_TYPES <<< "${TARGET_PREF_TYPES}"
fi

if [ -n "${TARGET_MEMORY_MODELS:-}" ]; then
    IFS='|' read -r -a MEMORY_MODELS <<< "${TARGET_MEMORY_MODELS}"
fi

SERVER_PID=""
SERVER_LOG_PATH=""
TEMP_JOB_SPEC_FILES=()

cleanup_temp_job_specs() {
    local temp_path=""
    for temp_path in "${TEMP_JOB_SPEC_FILES[@]}"; do
        [ -n "${temp_path}" ] || continue
        rm -f "${temp_path}" 2>/dev/null || true
    done
    TEMP_JOB_SPEC_FILES=()
}

cleanup() {
    local exit_code="${1:-1}"
    if [ -n "${SERVER_PID:-}" ]; then
        echo ""
        echo "[WARN] Killing vLLM Server (PID: ${SERVER_PID})..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
    fi
    cleanup_temp_job_specs
    exit "${exit_code}"
}

stop_server() {
    if [ -n "${SERVER_PID:-}" ]; then
        echo "[CLEANUP] Stopping vLLM Server..."
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
        SERVER_PID=""
        echo ">> Server Stopped. Cooling down..."
        sleep 10
    fi
}

trap 'cleanup 1' SIGINT SIGTERM ERR

wait_for_server() {
    echo "Waiting for vLLM server at ${VLLM_BASE_URL}..."
    echo "[INFO] First load can take 1-2 minutes while weights, torch.compile, and CUDA graphs initialize."
    local max_retries=120
    local count=0

    while ! curl -fsS --max-time 5 "${VLLM_BASE_URL}/models" > /dev/null 2>&1; do
        sleep 5
        echo -n "."
        count=$((count + 1))

        if [ $((count % 6)) -eq 0 ]; then
            echo ""
            echo "[INFO] vLLM is still starting (${count} checks, about $((count * 5))s elapsed)."
            if [ -n "${SERVER_LOG_PATH}" ] && [ -f "${SERVER_LOG_PATH}" ]; then
                echo "[INFO] Recent server log:"
                tail -n 12 "${SERVER_LOG_PATH}" || true
            fi
        fi

        if ! ps -p "${SERVER_PID}" > /dev/null; then
            echo ""
            echo "[ERROR] vLLM Server process died unexpectedly."
            if [ -n "${SERVER_LOG_PATH}" ] && [ -f "${SERVER_LOG_PATH}" ]; then
                cat "${SERVER_LOG_PATH}"
            fi
            cleanup 1
        fi

        if [ "${count}" -ge "${max_retries}" ]; then
            echo ""
            echo "[ERROR] Timeout waiting for vLLM server."
            if [ -n "${SERVER_LOG_PATH}" ] && [ -f "${SERVER_LOG_PATH}" ]; then
                echo "[INFO] Final server log tail before timeout:"
                tail -n 40 "${SERVER_LOG_PATH}" || true
            fi
            cleanup 1
        fi
    done

    echo ""
    echo ">> Server is READY!"
}

tool_parser_for_model() {
    local model="$1"
    if [[ "${model}" == *"Llama-3"* ]]; then
        echo "llama3_json"
    elif [[ "${model}" == *"Mistral"* ]]; then
        echo "mistral"
    else
        echo "hermes"
    fi
}

is_openai_api_model() {
    local lowered="${1,,}"
    [[ "${lowered}" == gpt-* ]] || [[ "${lowered}" == chatgpt-* ]] || [[ "${lowered}" == o1* ]] || [[ "${lowered}" == o3* ]] || [[ "${lowered}" == o4* ]]
}

is_gemini_api_model() {
    local lowered="${1,,}"
    [[ "${lowered}" == gemini* ]]
}

is_remote_api_model() {
    is_openai_api_model "$1" || is_gemini_api_model "$1"
}

prefer_local_snapshot_enabled() {
    case "${VLLM_PREFER_LOCAL_SNAPSHOT,,}" in
        1|true|yes|y|on)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

resolve_hf_snapshot_path() {
    local model="$1"
    python - "${model}" <<'PY'
import os
import sys
from pathlib import Path

model = sys.argv[1]
repo_dir_name = "models--" + model.replace("/", "--")
cache_roots = []

for env_name in ("HUGGINGFACE_HUB_CACHE", "HF_HOME"):
    value = os.environ.get(env_name)
    if not value:
        continue
    root = Path(value).expanduser()
    if env_name == "HF_HOME":
        root = root / "hub"
    cache_roots.append(root)

cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")

seen = set()
for cache_root in cache_roots:
    normalized = str(cache_root)
    if normalized in seen:
        continue
    seen.add(normalized)

    repo_dir = cache_root / repo_dir_name
    ref_path = repo_dir / "refs" / "main"
    if not ref_path.is_file():
        continue

    sha = ref_path.read_text(encoding="utf-8").strip()
    if not sha:
        continue

    snapshot_dir = repo_dir / "snapshots" / sha
    if snapshot_dir.is_dir():
        print(snapshot_dir)
        break
PY
}

resolve_vllm_model_target() {
    local model="$1"
    local snapshot_path=""

    if is_remote_api_model "${model}" || ! prefer_local_snapshot_enabled; then
        printf '%s\n' "${model}"
        return 0
    fi

    snapshot_path="$(resolve_hf_snapshot_path "${model}")"
    if [ -n "${snapshot_path}" ]; then
        printf '%s\n' "${snapshot_path}"
    else
        printf '%s\n' "${model}"
    fi
}

resolve_model_base_url() {
    local model="$1"
    if is_gemini_api_model "${model}"; then
        printf '%s\n' "${GEMINI_CHAT_BASE_URL}"
    elif is_openai_api_model "${model}"; then
        printf '%s\n' "${OPENAI_CHAT_BASE_URL}"
    else
        printf '%s\n' "${VLLM_BASE_URL}"
    fi
}

resolve_model_api_key() {
    local model="$1"
    if is_gemini_api_model "${model}"; then
        printf '%s\n' "${GEMINI_CHAT_API_KEY}"
    elif is_openai_api_model "${model}"; then
        printf '%s\n' "${OPENAI_API_KEY:-}"
    else
        printf '%s\n' "EMPTY"
    fi
}

require_model_config() {
    local model="$1"
    if is_gemini_api_model "${model}" && [ -z "${GEMINI_CHAT_BASE_URL}" ] && [ -z "${GEMINI_CHAT_API_KEY}" ]; then
        echo "[ERROR] Gemini model '${model}' requires GOOGLE_API_KEY (or GEMINI_CHAT_API_KEY / GEMINI_API_KEY) when no GEMINI_CHAT_BASE_URL is set."
        exit 1
    fi
    if is_openai_api_model "${model}" && [ -z "${OPENAI_CHAT_BASE_URL}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "[ERROR] OpenAI model '${model}' requires OPENAI_API_KEY when no OPENAI_CHAT_BASE_URL is set."
        exit 1
    fi
}

is_openai_embedding_model() {
    local model="$1"
    case "${model}" in
        text-embedding-3-small|text-embedding-3-large|text-embedding-ada-002)
            return 0
            ;;
        *)
            return 1
            ;;
    esac
}

is_local_embedding_backend() {
    local embedding_model="$1"
    local embedding_base_url="$2"
    if [ -n "${embedding_base_url}" ]; then
        return 1
    fi
    if is_openai_embedding_model "${embedding_model}"; then
        return 1
    fi
    return 0
}

resolve_embedding_api_key() {
    local embedding_model="$1"
    local embedding_base_url="$2"
    if is_local_embedding_backend "${embedding_model}" "${embedding_base_url}"; then
        printf '%s\n' ""
    else
        printf '%s\n' "${DEFAULT_EMBEDDING_API_KEY}"
    fi
}

require_embedding_config() {
    local memory_folder="$1"
    local embedding_model="$2"
    local embedding_base_url="$3"
    local embedding_api_key="$4"

    if is_local_embedding_backend "${embedding_model}" "${embedding_base_url}"; then
        return
    fi

    if [ -z "${embedding_base_url}" ] && [ -z "${embedding_api_key}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "[ERROR] Memory folder '${memory_folder}' needs remote embeddings (${embedding_model}), but no embedding API key is available."
        exit 1
    fi
}

append_job_spec() {
    local job_specs_path="$1"
    local model="$2"
    local chat_base_url="$3"
    local chat_api_key="$4"
    local context="$5"
    local output_path_template="$6"
    local log_path_template="$7"
    local run_log_path_template="$8"
    local vllm_model_target="$9"
    local parser_flag="${10}"
    local server_log_path="${11}"
    shift 11

    python - "${job_specs_path}" "${model}" "${chat_base_url}" "${chat_api_key}" "${context}" "${output_path_template}" "${log_path_template}" "${run_log_path_template}" "${vllm_model_target}" "${parser_flag}" "${server_log_path}" "${PORT}" "${GPU_IDS}" "${TP_SIZE}" "${MAX_MODEL_LEN}" "${GPU_MEMORY_UTILIZATION}" "$@" <<'PY'
import json
import sys

(
    job_specs_path,
    model,
    chat_base_url,
    chat_api_key,
    context,
    output_path_template,
    log_path_template,
    run_log_path_template,
    vllm_model_target,
    parser_flag,
    server_log_path,
    port,
    gpu_ids,
    tp_size,
    max_model_len,
    gpu_memory_utilization,
    *pref_types,
) = sys.argv[1:]

record = {
    "model_name": model,
    "base_url": chat_base_url,
    "api_key": chat_api_key,
    "context_type": context,
    "pref_types": pref_types,
    "output_path_template": output_path_template,
    "log_path_template": log_path_template,
    "run_log_path_template": run_log_path_template,
}

if vllm_model_target:
    record.update(
        {
            "vllm_model_target": vllm_model_target,
            "vllm_tool_parser": parser_flag,
            "vllm_server_log_path": server_log_path,
            "vllm_port": int(port),
            "vllm_gpu_ids": gpu_ids,
            "vllm_tp_size": int(tp_size),
            "vllm_max_model_len": int(max_model_len),
            "vllm_gpu_memory_utilization": float(gpu_memory_utilization),
        }
    )

with open(job_specs_path, "a", encoding="utf-8") as f:
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
PY
}

load_snapshot_embedding_config() {
    local manifest_path="$1"
    python - "${manifest_path}" "${DEFAULT_EMBEDDING_MODEL}" "${DEFAULT_EMBEDDING_BASE_URL}" <<'PY'
import json
import sys

manifest_path, default_model, default_base_url = sys.argv[1:]
embedding_model = default_model
embedding_base_url = default_base_url

try:
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    embedding_model = manifest.get("embedding_model") or default_model
    if "embedding_base_url" in manifest:
        raw_base_url = manifest.get("embedding_base_url")
        embedding_base_url = "" if raw_base_url is None else str(raw_base_url)
except FileNotFoundError:
    pass

print(embedding_model)
print(embedding_base_url)
PY
}

discover_memory_folders() {
    local discovered=()
    local folder=""
    while IFS= read -r folder; do
        discovered+=("${folder}")
    done < <(
        find "${MEMORY_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort
    )

    local filtered=()
    local memory_folder=""
    for memory_folder in "${discovered[@]}"; do
        if [ -f "${MEMORY_ROOT}/${memory_folder}/${CANONICAL_SNAPSHOT_NAME}" ]; then
            filtered+=("${memory_folder}")
        fi
    done

    if [ "${#filtered[@]}" -gt 0 ]; then
        printf '%s\n' "${filtered[@]}"
    fi
}

resolve_memory_folder_from_model() {
    local memory_model="$1"
    local candidate_folder="${memory_model//\//_}"
    local candidate_snapshot="${MEMORY_ROOT}/${candidate_folder}/${CANONICAL_SNAPSHOT_NAME}"

    if [ -f "${candidate_snapshot}" ]; then
        printf '%s\n' "${candidate_folder}"
        return 0
    fi

    python - "${MEMORY_ROOT}" "${CANONICAL_MANIFEST_NAME}" "${CANONICAL_SNAPSHOT_NAME}" "${memory_model}" <<'PY'
import json
import sys
from pathlib import Path

memory_root = Path(sys.argv[1])
manifest_name = sys.argv[2]
snapshot_name = sys.argv[3]
target_model = sys.argv[4]

for manifest_path in sorted(memory_root.glob(f"*/{manifest_name}")):
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        continue
    if manifest.get("memory_model") != target_model:
        continue
    snapshot_path = manifest_path.parent / snapshot_name
    if snapshot_path.is_file():
        print(manifest_path.parent.name)
        break
PY
}

resolve_memory_folders_from_models() {
    local resolved=()
    local memory_model=""
    local memory_folder=""

    for memory_model in "${MEMORY_MODELS[@]}"; do
        [ -n "${memory_model}" ] || continue
        memory_folder="$(resolve_memory_folder_from_model "${memory_model}")"
        if [ -z "${memory_folder}" ]; then
            echo "[ERROR] Could not resolve memory model '${memory_model}' under ${MEMORY_ROOT}." >&2
            return 1
        fi
        resolved+=("${memory_folder}")
    done

    if [ "${#resolved[@]}" -gt 0 ]; then
        printf '%s\n' "${resolved[@]}"
    fi
}

MEMORY_FOLDERS=()
if [ "${#MEMORY_MODELS[@]}" -gt 0 ]; then
    if resolved_memory_folders="$(resolve_memory_folders_from_models)"; then
        if [ -n "${resolved_memory_folders}" ]; then
            readarray -t MEMORY_FOLDERS <<< "${resolved_memory_folders}"
        fi
    else
        exit 1
    fi
else
    discovered_memory_folders="$(discover_memory_folders)"
    if [ -n "${discovered_memory_folders}" ]; then
        readarray -t MEMORY_FOLDERS <<< "${discovered_memory_folders}"
    fi
fi

if [ -n "${TARGET_MEMORY_FOLDERS:-}" ]; then
    IFS='|' read -r -a MEMORY_FOLDERS <<< "${TARGET_MEMORY_FOLDERS}"
fi

if [ "${#MEMORY_FOLDERS[@]}" -eq 0 ]; then
    echo "[ERROR] No memory folders found under ${MEMORY_ROOT}."
    exit 1
fi

for memory_folder in "${MEMORY_FOLDERS[@]}"; do
    snapshot_path="${MEMORY_ROOT}/${memory_folder}/${CANONICAL_SNAPSHOT_NAME}"
    if [ ! -f "${snapshot_path}" ]; then
        echo "[ERROR] Missing canonical snapshot: ${snapshot_path}"
        exit 1
    fi
done

mkdir -p "${OUTPUT_ROOT}"

echo "========================================================"
echo "LangMem Singleturn Inference Started at $(date)"
echo "GPU: ${GPU_IDS} | TP Size: ${TP_SIZE}"
echo "Models: ${MODELS[*]}"
if [ "${#MEMORY_MODELS[@]}" -gt 0 ]; then
    echo "Memory models: ${MEMORY_MODELS[*]}"
fi
echo "Memory folders: ${MEMORY_FOLDERS[*]}"
echo "========================================================"

for memory_folder in "${MEMORY_FOLDERS[@]}"; do
    memory_path="${MEMORY_ROOT}/${memory_folder}/${CANONICAL_SNAPSHOT_NAME}"
    manifest_path="${MEMORY_ROOT}/${memory_folder}/${CANONICAL_MANIFEST_NAME}"
    readarray -t embedding_config < <(load_snapshot_embedding_config "${manifest_path}")
    embedding_model="${embedding_config[0]:-${DEFAULT_EMBEDDING_MODEL}}"
    embedding_base_url="${embedding_config[1]-${DEFAULT_EMBEDDING_BASE_URL}}"
    embedding_api_key="$(resolve_embedding_api_key "${embedding_model}" "${embedding_base_url}")"
    require_embedding_config "${memory_folder}" "${embedding_model}" "${embedding_base_url}" "${embedding_api_key}"

    echo "####################################################################"
    echo "[STEP 1] Preparing Memory Snapshot: ${memory_folder}"
    echo "####################################################################"
    echo "Memory Path     : ${memory_path}"
    echo "Embedding Model : ${embedding_model}"
    echo "Embedding Base  : ${embedding_base_url:-<default/local>}"

    job_specs_path="$(mktemp "${OUTPUT_ROOT}/.${memory_folder}.single.batch.XXXXXX.jsonl")"
    TEMP_JOB_SPEC_FILES+=("${job_specs_path}")
    job_count=0

    for model in "${MODELS[@]}"; do
        model_safe_name="${model//\//_}"
        chat_base_url="$(resolve_model_base_url "${model}")"
        chat_api_key="$(resolve_model_api_key "${model}")"
        require_model_config "${model}"

        vllm_model_target=""
        parser_flag=""
        server_log_path=""

        if is_remote_api_model "${model}"; then
            echo "####################################################################"
            echo "[STEP 2] Queueing Remote API Runs for: ${model}"
            echo "####################################################################"
            if [ -n "${chat_base_url}" ]; then
                echo "Base URL: ${chat_base_url}"
            else
                echo "Base URL: <provider default>"
            fi
        else
            vllm_model_target="$(resolve_vllm_model_target "${model}")"
            parser_flag="$(tool_parser_for_model "${model}")"
            server_log_dir="${OUTPUT_ROOT}/_server_logs/${model_safe_name}"
            date_tag="$(date +%Y%m%d_%H%M%S)"
            server_log_path="${server_log_dir}/${date_tag}.vllm_server.log"

            mkdir -p "${server_log_dir}"

            echo "####################################################################"
            echo "[STEP 2] Queueing Local vLLM Runs for: ${model}"
            echo "####################################################################"
            echo "Tool parser: ${parser_flag}"
            if [ "${vllm_model_target}" != "${model}" ]; then
                echo "Model Source: local snapshot ${vllm_model_target}"
            fi
        fi

        echo "[STEP 3] Queueing Singleturn Experiments for Current Memory..."

        for context in "${CONTEXT_TYPES[@]}"; do
            run_tag="$(date +%Y%m%d_%H%M%S)"
            output_path_template="${OUTPUT_ROOT}/${memory_folder}/${context}/{pref_type}/${model_safe_name}/${PROMPT_NAME}/${run_tag}.json"
            log_path_template="${OUTPUT_ROOT}/${memory_folder}/${context}/{pref_type}/${model_safe_name}/${PROMPT_NAME}/${run_tag}.jsonl"
            run_log_path_template="${OUTPUT_ROOT}/${memory_folder}/${context}/{pref_type}/${model_safe_name}/${PROMPT_NAME}/${run_tag}.run.log"

            append_job_spec \
                "${job_specs_path}" \
                "${model}" \
                "${chat_base_url}" \
                "${chat_api_key}" \
                "${context}" \
                "${output_path_template}" \
                "${log_path_template}" \
                "${run_log_path_template}" \
                "${vllm_model_target}" \
                "${parser_flag}" \
                "${server_log_path}" \
                "${PREF_TYPES[@]}"

            job_count=$((job_count + 1))

            echo " >> [QUEUED]"
            echo "    - Model           : ${model}"
            echo "    - Memory Source   : ${memory_folder}"
            echo "    - Context/Prefs   : ${context} / ${PREF_TYPES[*]}"
            echo "    - Memory Path     : ${memory_path}"
            echo "    - Embedding Model : ${embedding_model}"
            echo "    - Snapshot Load   : reused across queued models"
        done
    done

    if [ "${job_count}" -eq 0 ]; then
        echo "[SKIP] No jobs were queued for memory: ${memory_folder}"
        rm -f "${job_specs_path}"
        continue
    fi

    cmd=(
        python "${PYTHON_SCRIPT}"
        --memory_path "${memory_path}"
        --input_path "${INPUT_PATH}"
        --query_path "${QUERY_PATH}"
        --pref_list_path "${PREF_LIST_PATH}"
        --pref_group_path "${PREF_GROUP_PATH}"
        --tools_schema_path "${TOOLS_SCHEMA_PATH}"
        --job_specs_path "${job_specs_path}"
        --concurrency "${CONCURRENCY}"
        --embedding_model "${embedding_model}"
        --embedding_base_url "${embedding_base_url}"
    )

    if [ -n "${MAX_QUERIES}" ]; then
        cmd+=(--max_queries "${MAX_QUERIES}")
    fi
    if [ -n "${embedding_api_key}" ]; then
        cmd+=(--embedding_api_key "${embedding_api_key}")
    fi
    if [ -n "${REASONING_EFFORT}" ]; then
        cmd+=(--reasoning_effort "${REASONING_EFFORT}")
    fi

    "${cmd[@]}"
    rm -f "${job_specs_path}"
done

echo "========================================================"
echo "All LangMem Singleturn Jobs Finished at $(date)"
echo "========================================================"
cleanup_temp_job_specs
