#!/bin/bash

# ==============================================================================
# 1. 환경 설정 및 경로
# ==============================================================================

# Python 스크립트 절대 경로
PYTHON_SCRIPT="/data/minseo/experiments6/ours_memory/Preference_Memory_step2_ACTION_multiturn_VLLM.py"

# 고정 데이터 경로
INPUT_PATH="/data/minseo/experiments6/data/1229_dev_6.json"
QUERY_PATH="/data/minseo/experiments6/query_multiturn-domain.json"
PREF_LIST_PATH="/data/minseo/experiments6/pref_list.json"
PREF_GROUP_PATH="/data/minseo/experiments6/pref_group.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments6/schema_all.json"

# vLLM 서버 설정
GPU_IDS="0,1,2,3"
TP_SIZE=4
PORT=8003
BASE_URL="http://localhost:$PORT/v1"
CONCURRENCY=100

# ==============================================================================
# 2. 실험 변수 설정 (두 개의 리스트로 분리)
# ==============================================================================

# [리스트 A] 추론(Inference)을 수행할 모델 목록 (HuggingFace 경로)
# 이 모델들이 순서대로 vLLM 서버에 로드됩니다.
INFERENCE_MODELS=(
    ##"meta-llama/Llama-3.1-8B-Instruct"
    ##"Qwen/Qwen3-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
    "google/gemma-3-12b-it"
    "google/codegemma-7b-it"
)

# [리스트 B] 사용할 메모리가 저장된 폴더명 목록
# 현재 로드된 추론 모델 하나에 대해, 아래 메모리들을 하나씩 바꿔가며 모두 테스트합니다.
# 주의: 여기는 슬래시(/)가 없는 폴더명이어야 합니다 (예: google_gemma...)
MEMORY_FOLDERS=(
    "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B"
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B"
    "google_gemma-3-12b-it"
    "gpt-4o-mini"
    ##"meta-llama_Llama-3.1-8B-Instruct"
    "Qwen_Qwen3-8B"


)

# 실험 조건
CONTEXT_TYPES=("memory_api") 
PREF_TYPES=("easy" "medium" "hard")
PROMPT_NAME="implicit_zs"

# ==============================================================================
# 3. 헬퍼 함수
# ==============================================================================

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

wait_for_server() {
    echo "Waiting for vLLM server at $BASE_URL..."
    MAX_RETRIES=120
    COUNT=0
    
    while ! curl -s "$BASE_URL/models" > /dev/null; do
        sleep 5
        echo -n "."
        COUNT=$((COUNT+1))
        
        if ! ps -p $SERVER_PID > /dev/null; then
            echo ""
            echo "[ERROR] vLLM Server process died unexpectedly."
            cat vllm_server.log
            cleanup
        fi
        if [ $COUNT -ge $MAX_RETRIES ]; then
            echo "[ERROR] Timeout waiting for vLLM server."
            cleanup
        fi
    done
    echo ">> Server is READY!"
}

# ==============================================================================
# 4. 메인 루프 실행 (이중 루프 구조)
# ==============================================================================

# [Outer Loop] 추론 모델 변경
for INF_MODEL in "${INFERENCE_MODELS[@]}"; do
    
    # 모델명 치환 (폴더 저장용): google/gemma -> google_gemma
    INF_SAFE_NAME="${INF_MODEL//\//_}"

    echo "####################################################################"
    echo "[STEP 1] Starting vLLM Server for Inference Model: $INF_MODEL"
    echo "####################################################################"

    # 4-1. 모델별 Tool Parser 설정
    PARSER_FLAG=""
    if [[ "$INF_MODEL" == *"Llama-3"* ]]; then
        PARSER_FLAG="--tool-call-parser llama3_json"
    elif [[ "$INF_MODEL" == *"Mistral"* ]]; then
        PARSER_FLAG="--tool-call-parser mistral"
    elif [[ "$INF_MODEL" == *"Qwen"* ]]; then
        PARSER_FLAG="--tool-call-parser hermes"
    else
        PARSER_FLAG="--tool-call-parser hermes" 
    fi

    # 4-2. vLLM 서버 실행
    CUDA_VISIBLE_DEVICES=$GPU_IDS nohup vllm serve "$INF_MODEL" \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP_SIZE \
        --enable-auto-tool-choice \
        $PARSER_FLAG \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.9 \
        --trust-remote-code > vllm_server.log 2>&1 &
    
    SERVER_PID=$!
    wait_for_server

    # [Inner Loop] 메모리 폴더 변경
    # 서버는 띄워둔 채로, 메모리 파일 경로만 바꿔가며 Python 스크립트 실행
    echo "[STEP 2] Running Experiments across Memory List..."

    for MEM_FOLDER in "${MEMORY_FOLDERS[@]}"; do
        
        # ----------------------------------------------------------------------
        # 경로 설정
        # 1. 메모리 경로: 리스트 B의 폴더명 사용
        MEMORY_PATH="/data/minseo/experiments6/ours_memory/inference/0312_MEMORY1/${MEM_FOLDER}/_memory1.jsonl"
        
        # 2. 결과 저장 경로: [추론모델명]/[메모리출처] 형태로 계층화
        # 예: .../inference/google_gemma-3-12b-it/mem_source_llama-3/...
        BASE_OUTPUT_DIR="/data/minseo/experiments6/ours_memory/inference/0312_MEMORY1_0317_MEM-DIAG/multiturn/${MEM_FOLDER}"
        BASE_LOG_DIR="/data/minseo/experiments6/ours_memory/inference/0312_MEMORY1_0317_MEM-DIAG/multiturn/${MEM_FOLDER}"

        mkdir -p "$BASE_OUTPUT_DIR"
        mkdir -p "$BASE_LOG_DIR"
        
        # ----------------------------------------------------------------------

        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do

                echo " >> [Processing]"
                echo "    - Inference Model : $INF_SAFE_NAME"
                echo "    - Memory Source   : $MEM_FOLDER"
                echo "    - Context/Pref    : $context / $pref"

                DATE_TAG="$(date +%m%d)"
                CURRENT_OUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/$INF_MODEL/$PROMPT_NAME"
                CURRENT_LOG_DIR="$BASE_LOG_DIR/$context/$pref/$INF_MODEL/$PROMPT_NAME"
                
                mkdir -p "$CURRENT_OUT_DIR"
                mkdir -p "$CURRENT_LOG_DIR"

                OUTPUT_FILE="$CURRENT_OUT_DIR/${DATE_TAG}_test1.json"
                LOG_FILE="$CURRENT_LOG_DIR/${DATE_TAG}_test1.log"

                python "$PYTHON_SCRIPT" \
                    --input_path "$INPUT_PATH" \
                    --memory_path "$MEMORY_PATH" \
                    --output_path "$OUTPUT_FILE" \
                    --log_path "$LOG_FILE" \
                    --multiturn_path "$QUERY_PATH" \
                    --pref_list_path "$PREF_LIST_PATH" \
                    --pref_group_path "$PREF_GROUP_PATH" \
                    --tools_schema_path "$TOOLS_SCHEMA_PATH" \
                    --context_type "$context" \
                    --pref_type "$pref" \
                    --model_name "$INF_MODEL" \
                    --base_url "$BASE_URL" \
                    --api_key "EMPTY" \
                    --concurrency "$CONCURRENCY"

            done
        done
    done # End of Memory Loop

    # 4-3. 현재 모델 서버 종료
    echo "[STEP 3] Stopping vLLM Server for $INF_MODEL..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    SERVER_PID=""
    echo "--------------------------------------------------------"
    sleep 10

done # End of Inference Model Loop

echo "All Jobs Finished."
