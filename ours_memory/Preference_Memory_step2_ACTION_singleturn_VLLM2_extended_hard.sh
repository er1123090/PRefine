#!/bin/bash

# ==============================================================================
# 1. 환경 설정 및 경로
# ==============================================================================

# [중요] Reference에 명시된 Python 스크립트 경로
PYTHON_SCRIPT="/data/minseo/experiments4/ours_memory/Preference_Memory_step2_ACTION_singleturn_VLLM.py"

# 데이터 경로
INPUT_PATH="/data/minseo/experiments4/data/1229_dev_6.json"
QUERY_PATH="/data/minseo/experiments4/query_new_single.json"
PREF_LIST_PATH="/data/minseo/experiments4/pref_list_extended.json"
PREF_GROUP_PATH="/data/minseo/experiments4/pref_group_extended.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments4/schema_all_extended_complete.json"

# vLLM 서버 설정
GPU_IDS="0,1,2,3"       # 사용할 GPU ID
TP_SIZE=4               # Tensor Parallelism Size
PORT=8003
BASE_URL="http://localhost:$PORT/v1"
CONCURRENCY=50          # 비동기 요청 수
MAX_QUERIES="${MAX_QUERIES:-400}"

# ==============================================================================
# 2. 실험 변수 설정
# ==============================================================================

# [리스트 A] 실행할 로컬 LLM 모델 목록 (vLLM용)
MODELS=(
    ## "meta-llama/Llama-3.1-8B-Instruct"
    ##"Qwen/Qwen3-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
    "google/gemma-3-12b-it"
    "google/codegemma-7b-it"
    
)

# [리스트 B] 사용할 메모리가 저장된 폴더명 목록
MEMORY_FOLDERS=(
    # "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B"
    # "deepseek-ai_DeepSeek-R1-Distill-Llama-8B"
    "google_gemma-3-12b-it"
    "gpt-4o-mini"
    #"Qwen_Qwen3-8B"
)

# 실험 조건
CONTEXT_TYPES=("memory_api")     # choices: memory_only, memory_diag, memory_api, api-only
PREF_TYPES=("hard") # choices: easy, medium, hard
PROMPT_NAME="implicit_zs"

# ==============================================================================
# 3. 헬퍼 함수 (서버 제어 - Reference 로직 가져옴)
# ==============================================================================

# 스크립트 강제 종료 시 서버 프로세스 정리
trap cleanup SIGINT SIGTERM ERR

cleanup() {
    if [ -n "$SERVER_PID" ]; then
        echo ""
        echo "[WARN] Killing vLLM Server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null
        wait $SERVER_PID 2>/dev/null
    fi
    exit 1
}

# 서버가 준비될 때까지 대기
wait_for_server() {
    echo "Waiting for vLLM server at $BASE_URL..."
    MAX_RETRIES=120 # 10분 대기
    COUNT=0
    
    while ! curl -s "$BASE_URL/models" > /dev/null; do
        sleep 5
        echo -n "."
        COUNT=$((COUNT+1))
        
        # 서버 프로세스 생존 확인
        if ! ps -p $SERVER_PID > /dev/null; then
            echo ""
            echo "[ERROR] vLLM Server process died unexpectedly."
            cat vllm_server.log
            cleanup
        fi

        if [ $COUNT -ge $MAX_RETRIES ]; then
            echo ""
            echo "[ERROR] Timeout waiting for vLLM server."
            cleanup
        fi
    done
    echo ""
    echo ">> Server is READY!"
}

# ==============================================================================
# 4. 메인 루프 실행
# ==============================================================================

echo "========================================================"
echo "Starting Local vLLM Inference Loop (Single-turn Extended Hard)"
echo "========================================================"

# [Outer Loop] 모델 변경 -> 서버 재시작 필요
for model in "${MODELS[@]}"; do
    
    MODEL_SAFE_NAME="${model//\//_}"

    echo "####################################################################"
    echo "[STEP 1] Starting vLLM Server for: $model"
    echo "####################################################################"

    # 4-1. 모델별 Tool Parser 설정 (필요시 자동 설정)
    PARSER_FLAG="--tool-call-parser hermes" 
    if [[ "$model" == *"Llama-3"* ]]; then
        PARSER_FLAG="--tool-call-parser llama3_json"
    elif [[ "$model" == *"Mistral"* ]]; then
        PARSER_FLAG="--tool-call-parser mistral"
    fi

    # 4-2. vLLM 서버 백그라운드 실행
    CUDA_VISIBLE_DEVICES=$GPU_IDS nohup vllm serve "$model" \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP_SIZE \
        --enable-auto-tool-choice \
        $PARSER_FLAG \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code > vllm_server.log 2>&1 &
    
    SERVER_PID=$!
    echo ">> vLLM Server PID: $SERVER_PID"
    
    # 4-3. 서버 준비 대기
    wait_for_server

    # [Inner Loop] 메모리 폴더 변경 (서버 유지)
    for MEM_FOLDER in "${MEMORY_FOLDERS[@]}"; do
        
        # 메모리 경로 및 결과 저장 경로 설정
        MEMORY_PATH="/data/minseo/experiments4/ours_memory/inference/1231_MEMORY3/${MEM_FOLDER}/_memory1.jsonl"
        BASE_OUTPUT_DIR="/data/minseo/experiments4/ours_memory/inference/extended/outputs/singleturn/vllm/${MEM_FOLDER}"
        BASE_LOG_DIR="/data/minseo/experiments4/ours_memory/inference/extended/logs/singleturn/vllm/${MEM_FOLDER}"

        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do

                echo " >> [RUNNING]"
                echo "    - Model         : $model"
                echo "    - Memory Source : $MEM_FOLDER"
                echo "    - Context/Pref  : $context / $pref"

                # 결과 폴더 생성
                DATE_TAG="$(date +%m%d)"
                CURRENT_OUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/${MODEL_SAFE_NAME}_high/$PROMPT_NAME"
                CURRENT_LOG_DIR="$BASE_LOG_DIR/$context/$pref/${MODEL_SAFE_NAME}_high/$PROMPT_NAME"
                
                mkdir -p "$CURRENT_OUT_DIR"
                mkdir -p "$CURRENT_LOG_DIR"

                OUTPUT_FILE="$CURRENT_OUT_DIR/${DATE_TAG}_test1.json"
                LOG_FILE="$CURRENT_LOG_DIR/${DATE_TAG}_test1.log"

                # Python 스크립트 실행 (vLLM 모드)
                # base_url을 localhost로, api_key를 EMPTY로 설정
                python "$PYTHON_SCRIPT" \
                    --input_path "$INPUT_PATH" \
                    --memory_path "$MEMORY_PATH" \
                    --output_path "$OUTPUT_FILE" \
                    --log_path "$LOG_FILE" \
                    --query_path "$QUERY_PATH" \
                    --pref_list_path "$PREF_LIST_PATH" \
                    --pref_group_path "$PREF_GROUP_PATH" \
                    --tools_schema_path "$TOOLS_SCHEMA_PATH" \
                    --context_type "$context" \
                    --pref_type "$pref" \
                    --model_name "$model" \
                    --base_url "$BASE_URL" \
                    --api_key "EMPTY" \
                    --concurrency "$CONCURRENCY" \
                    --max_queries "$MAX_QUERIES"

            done
        done
    done # End of Memory Loop

    # 4-4. 서버 종료 (다음 모델을 위해)
    echo "[STEP 3] Stopping vLLM Server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    SERVER_PID=""
    
    echo ">> Server Stopped. Cooling down..."
    sleep 10

done # End of Model Loop

echo "========================================================"
echo "All Jobs Finished Successfully at $(date)"
echo "========================================================"
