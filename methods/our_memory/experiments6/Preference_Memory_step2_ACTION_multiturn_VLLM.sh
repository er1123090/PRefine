.#!/bin/bash

# ==============================================================================
# 1. 환경 설정 및 경로 (사용자 환경에 맞게 수정)
# ==============================================================================

# Python 스크립트 절대 경로
PYTHON_SCRIPT="/data/minseo/experiments6/ours_memory/Preference_Memory_step2_ACTION_multiturn_VLLM.py"  # 저장한 파이썬 파일명으로 변경하세요

# 데이터 및 리소스 경로
INPUT_PATH="/data/minseo/experiments6/data/1229_dev_6.json"
MEMORY_PATH="/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3/google_gemma-3-12b-it/_memory1.jsonl" # [중요] Python 코드에서 요구하는 메모리 파일 경로
QUERY_PATH="/data/minseo/experiments6/query_multiturn-domain.json"
PREF_LIST_PATH="/data/minseo/experiments6/pref_list.json"
PREF_GROUP_PATH="/data/minseo/experiments6/pref_group.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments6/schema_all.json"

# 결과 저장 루트 디렉토리
BASE_OUTPUT_DIR="/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3_inference1_multi/gemma-3-12b-it"
BASE_LOG_DIR="/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3_logs1_multi/gemma-3-12b-it"

# vLLM 서버 설정
GPU_IDS="0,1,2,3"       # 사용할 GPU ID 목록
TP_SIZE=4               # Tensor Parallelism Size (GPU 개수와 맞추세요)
PORT=8003
BASE_URL="http://localhost:$PORT/v1"
CONCURRENCY=100          # Python 클라이언트 비동기 처리 수

# ==============================================================================
# 2. 실험 변수 설정
# ==============================================================================

# 실행할 모델 목록 (주석 해제/제거하여 선택)
MODELS=(
    # "meta-llama/Llama-3.1-8B-Instruct"
    # "Qwen/Qwen3-8B"
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" 
    # "google/gemma-3-12b-it"
    #"mistralai/Mistral-7B-Instruct-v0.3"
    "google/codegemma-7b-it"
    
    )

# 실험 조건 (루프용)
CONTEXT_TYPES=("memory_api" ) # python 코드의 choices: memory_only, memory_diag, memory_api
PREF_TYPES=("easy" "medium" "hard")                # python 코드의 choices: easy, medium, hard
PROMPT_NAME="implicit_zs"                   # 폴더명 생성을 위한 태그 (Python 코드엔 인자로 안 들어감)

# ==============================================================================
# 3. 헬퍼 함수 (서버 제어)
# ==============================================================================

# 스크립트 종료 시 서버 프로세스 정리 (Trap)
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

# 서버가 준비될 때까지 대기하는 함수
wait_for_server() {
    echo "Waiting for vLLM server at $BASE_URL..."
    MAX_RETRIES=120 # 10분 대기 (5s * 120)
    COUNT=0
    
    while ! curl -s "$BASE_URL/models" > /dev/null; do
        sleep 5
        echo -n "."
        COUNT=$((COUNT+1))
        
        # 서버 프로세스가 죽었는지 확인
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

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$BASE_LOG_DIR"

for model in "${MODELS[@]}"; do
    # 모델명에서 슬래시(/)를 언더바(_)로 치환하여 폴더명/로그에 사용
    MODEL_SAFE_NAME="${model//\//_}"
    
    echo "####################################################################"
    echo "[STEP 1] Starting vLLM Server for: $model"
    echo "####################################################################"

    # 4-1. 모델별 Tool Parser 설정 (필요시)
    PARSER_FLAG=""
    if [[ "$model" == *"Llama-3"* ]]; then
        PARSER_FLAG="--tool-call-parser llama3_json"
    elif [[ "$model" == *"Mistral"* ]]; then
        PARSER_FLAG="--tool-call-parser mistral"
    elif [[ "$model" == *"Qwen"* ]]; then
        PARSER_FLAG="--tool-call-parser hermes"
    else
        # 기본적으로 자동 선택에 맡기거나, 필요시 추가
        PARSER_FLAG="--tool-call-parser hermes" 
    fi

    # 4-2. vLLM 서버 백그라운드 실행
    # nohup을 사용하여 백그라운드에서 실행하고 로그를 남김
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

    # 4-4. Python Client 실행 (조건별 루프)
    echo "[STEP 2] Running Python Client Scripts..."
    
    for context in "${CONTEXT_TYPES[@]}"; do
        for pref in "${PREF_TYPES[@]}"; do

            echo " >> [Processing] Context: $context | Pref: $pref"

            # 결과 저장 경로 생성 (구조화)
            DATE_TAG="$(date +%m%d)"
            CURRENT_OUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/$MODEL_SAFE_NAME/$PROMPT_NAME"
            CURRENT_LOG_DIR="$BASE_LOG_DIR/$context/$pref/$MODEL_SAFE_NAME/$PROMPT_NAME"
            
            mkdir -p "$CURRENT_OUT_DIR"
            mkdir -p "$CURRENT_LOG_DIR"

            OUTPUT_FILE="$CURRENT_OUT_DIR/${DATE_TAG}_test1.json"
            LOG_FILE="$CURRENT_LOG_DIR/${DATE_TAG}_test1.log"

            # Python 스크립트 실행
            # [주의] Python 코드의 argparse 이름과 정확히 일치해야 함
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
                --model_name "$model" \
                --base_url "$BASE_URL" \
                --api_key "EMPTY" \
                --concurrency "$CONCURRENCY"

        done
    done

    # 4-5. 서버 종료 및 정리
    echo "[STEP 3] Stopping vLLM Server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    
    SERVER_PID="" # PID 초기화
    echo ">> Server Stopped."
    echo "--------------------------------------------------------"
    
    # 다음 모델 로딩 전 GPU 메모리 해제를 위해 잠시 대기
    sleep 10

done

echo "========================================================"
echo "All Jobs Finished Successfully at $(date)"
echo "========================================================"