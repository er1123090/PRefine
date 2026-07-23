#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/data/minseo/experiments4"
OURS_ROOT="$ROOT/ours_memory"
INPUT="$ROOT/data/mix600.json"
RUN_ROOT="$OURS_ROOT/inference/cross_model_mix600_high_reasoning_20260721"
STEP1="$OURS_ROOT/cross_model_high_reasoning_step1.py"
ACTION="$OURS_ROOT/cross_model_high_reasoning_action.py"
PORT="${PORT:-8002}"
API_BASE="http://127.0.0.1:$PORT/v1"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
TP_SIZE="${TP_SIZE:-4}"
STEP1_CONCURRENCY="${STEP1_CONCURRENCY:-3}"
ACTION_CONCURRENCY="${ACTION_CONCURRENCY:-8}"
SERVER_PID=""
VLLM_BIN_DEFAULT="/data/minseo/.venvs/astraglus-vllm/bin/vllm"
VLLM_BIN_GEMMA4="/data/minseo/.venvs/astraglus-vllm-gemma4/bin/vllm"

mkdir -p "$RUN_ROOT/run_logs"

timestamp() {
    date -Is
}

log() {
    echo "[$(timestamp)] $*"
}

stop_server() {
    if [[ -n "$SERVER_PID" ]]; then
        log "Stopping vLLM process group $SERVER_PID"
        kill -- "-$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
        sleep 5
    fi
}

cleanup() {
    stop_server
}

trap cleanup EXIT INT TERM

wait_for_server() {
    local waited=0
    while ! curl --fail --silent "$API_BASE/models" >/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "vLLM exited before readiness"
            return 1
        fi
        if (( waited >= 1800 )); then
            log "Timed out waiting for vLLM after ${waited}s"
            return 1
        fi
        sleep 5
        waited=$((waited + 5))
    done
    log "vLLM ready after ${waited}s"
}

start_server() {
    local model_key="$1"
    local model_id="$2"
    local reasoning_parser="$3"
    local vllm_bin="${4:-$VLLM_BIN_DEFAULT}"
    local model_impl="${5:-auto}"
    local server_log="$RUN_ROOT/run_logs/server_${model_key}.log"
    local extra_args=()
    if [[ "$model_key" == gemma4* ]]; then
        extra_args+=(--language-model-only)
    fi

    stop_server
    log "Starting $model_id with highest reasoning parser=$reasoning_parser vllm=$vllm_bin"
    CUDA_VISIBLE_DEVICES="$GPU_IDS" \
    PYTHONPATH="$OURS_ROOT/runtime_compat" \
    PREFINE_PYTHON_INCLUDE="/home/minseo/miniconda3/include/python3.10" \
    PYTHONUNBUFFERED=1 \
    setsid "$vllm_bin" serve "$model_id" \
        --served-model-name "$model_id" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --tensor-parallel-size "$TP_SIZE" \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.95 \
        --enforce-eager \
        --model-impl "$model_impl" \
        --reasoning-parser "$reasoning_parser" \
        --trust-remote-code \
        "${extra_args[@]}" \
        >"$server_log" 2>&1 &
    SERVER_PID=$!
    log "vLLM pid=$SERVER_PID log=$server_log"
    wait_for_server
}

memory_complete() {
    local model_key="$1"
    local status
    status="$(python "$STEP1" status \
        --input "$INPUT" \
        --shard-root "$RUN_ROOT/$model_key/step1_shards")"
    [[ "$status" == *'"complete": true'* ]]
}

run_step1() {
    local model_key="$1"
    local model_id="$2"
    local model_root="$RUN_ROOT/$model_key"
    local shard_root="$model_root/step1_shards"

    mkdir -p "$shard_root/a" "$shard_root/b"
    if memory_complete "$model_key"; then
        log "Step1 already complete for $model_key"
    else
        log "Resuming step1 for $model_key"
        python "$STEP1" run-shard \
            --input "$INPUT" \
            --output "$shard_root/a/memory.jsonl" \
            --model "$model_id" \
            --api-base "$API_BASE" \
            --num-shards 2 \
            --shard-index 0 \
            --concurrency "$STEP1_CONCURRENCY" \
            --resume \
            >"$RUN_ROOT/run_logs/step1_${model_key}_a.log" 2>&1 &
        local shard_a_pid=$!

        python "$STEP1" run-shard \
            --input "$INPUT" \
            --output "$shard_root/b/memory.jsonl" \
            --model "$model_id" \
            --api-base "$API_BASE" \
            --num-shards 2 \
            --shard-index 1 \
            --concurrency "$STEP1_CONCURRENCY" \
            --resume \
            >"$RUN_ROOT/run_logs/step1_${model_key}_b.log" 2>&1 &
        local shard_b_pid=$!

        local failed=0
        wait "$shard_a_pid" || failed=1
        wait "$shard_b_pid" || failed=1
        if (( failed != 0 )); then
            log "Step1 shard failure for $model_key"
            return 1
        fi
    fi

    python "$STEP1" merge \
        --input "$INPUT" \
        --shard-root "$shard_root" \
        --output "$model_root/memory.jsonl"
    log "Step1 merged for $model_key"
}

run_action_cell() {
    local memory_key="$1"
    local action_key="$2"
    local action_model="$3"
    local turn="$4"
    local pref="$5"
    local query_path
    if [[ "$turn" == "single" ]]; then
        query_path="$ROOT/query_singleturn.json"
    else
        query_path="$ROOT/query_multiturn.json"
    fi

    local cell_root="$RUN_ROOT/action/${memory_key}__to__${action_key}/$turn/$pref"
    mkdir -p "$cell_root"
    python "$ACTION" \
        --turn "$turn" \
        --input-path "$INPUT" \
        --memory-path "$RUN_ROOT/$memory_key/memory.jsonl" \
        --output-path "$cell_root/output.json" \
        --log-path "$cell_root/requests.jsonl" \
        --query-path "$query_path" \
        --pref-list-path "$ROOT/pref_list.json" \
        --pref-group-path "$ROOT/pref_group.json" \
        --tools-schema-path "$ROOT/schema_all.json" \
        --pref-type "$pref" \
        --model-name "$action_model" \
        --api-base "$API_BASE" \
        --concurrency "$ACTION_CONCURRENCY" \
        --resume \
        >"$RUN_ROOT/run_logs/action_${memory_key}_to_${action_key}_${turn}_${pref}.log" 2>&1
}

run_action_model() {
    local action_key="$1"
    local action_model="$2"
    local memory_key turn pref
    for memory_key in gemma4_12b_it qwen3_32b gpt_oss_20b; do
        for turn in single multi; do
            for pref in easy medium hard; do
                log "Action memory=$memory_key model=$action_key turn=$turn pref=$pref reasoning=high"
                run_action_cell "$memory_key" "$action_key" "$action_model" "$turn" "$pref"
            done
        done
    done
}

run_model_step1() {
    local model_key="$1"
    local model_id="$2"
    local parser="$3"
    local vllm_bin="${4:-$VLLM_BIN_DEFAULT}"
    local model_impl="${5:-auto}"
    if memory_complete "$model_key"; then
        log "Skipping server load: complete step1 for $model_key"
        python "$STEP1" merge \
            --input "$INPUT" \
            --shard-root "$RUN_ROOT/$model_key/step1_shards" \
            --output "$RUN_ROOT/$model_key/memory.jsonl"
        return
    fi
    start_server "$model_key" "$model_id" "$parser" "$vllm_bin" "$model_impl"
    run_step1 "$model_key" "$model_id"
    stop_server
}

log "ours_memory 3x3 high-reasoning queue started"
log "Matrix axes: memory builder x action model"
run_model_step1 "qwen3_32b" "Qwen/Qwen3-32B" "qwen3"
run_model_step1 "gemma4_12b_it" "google/gemma-4-12B-it" "gemma4" "$VLLM_BIN_GEMMA4" "vllm"
run_model_step1 "gpt_oss_20b" "openai/gpt-oss-20b" "openai_gptoss"

start_server "qwen3_32b_action" "Qwen/Qwen3-32B" "qwen3"
run_action_model "qwen3_32b" "Qwen/Qwen3-32B"
stop_server

start_server "gemma4_12b_it_action" "google/gemma-4-12B-it" "gemma4" "$VLLM_BIN_GEMMA4" "vllm"
run_action_model "gemma4_12b_it" "google/gemma-4-12B-it"
stop_server

start_server "gpt_oss_20b_action" "openai/gpt-oss-20b" "openai_gptoss"
run_action_model "gpt_oss_20b" "openai/gpt-oss-20b"
stop_server

log "ours_memory 3x3 high-reasoning queue completed"
