#!/bin/bash

# ==============================================================================
# 1. 환경 설정 및 경로 (사용자 환경에 맞게 수정 필수)
# ==============================================================================

# [중요] 작성한 파이썬 스크립트 파일명 (예: generate_memory.py)
PYTHON_SCRIPT="/data/minseo/experiments6/ours_memory/Preference_Memory_step1_LATENTPREF_VLLM.py"

# 입력 데이터 경로
INPUT_DATA="/data/minseo/experiments6/data/1229_dev_6.json"

# 결과 저장 루트 디렉토리
BASE_OUTPUT_DIR="/data/minseo/experiments6/ours_memory/inference/0312_MEMORY1"

# GPU 및 서버 설정
GPU_ID=0,1,2,3
PORT=8002
VLLM_URL="http://localhost:$PORT/v1"
TP_SIZE=4      # Tensor Parallelism (GPU 개수)
CONCURRENCY=20 # 동시 처리 요청 수

# ==============================================================================
# 2. 실행할 모델 리스트
# ==============================================================================

MODELS=(
    #"meta-llama/Llama-3.1-8B-Instruct"
    "Qwen/Qwen3-8B"
    #"google/gemma-3-12b-it"
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"


    
)

# ==============================================================================
# 3. 헬퍼 함수 (서버 대기 및 종료)
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
    echo "Waiting for vLLM server at $VLLM_URL..."
    MAX_RETRIES=60 # 5분 대기
    COUNT=0
    
    while ! curl -s "$VLLM_URL/models" > /dev/null; do
        sleep 5
        echo -n "."
        COUNT=$((COUNT+1))
        
        # 서버 프로세스가 죽었는지 확인
        if ! ps -p $SERVER_PID > /dev/null; then
            echo ""
            echo "[ERROR] vLLM Server process died unexpectedly."
            cat vllm_server.log
            exit 1
        fi

        if [ $COUNT -ge $MAX_RETRIES ]; then
            echo ""
            echo "[ERROR] Timeout waiting for vLLM server."
            kill $SERVER_PID
            exit 1
        fi
    done
    echo ""
    echo ">> Server is READY!"
}

# ==============================================================================
# 4. 메인 루프 실행
# ==============================================================================

echo "========================================================"
echo "Memory Generation Batch Started at $(date)"
echo "GPU: $GPU_ID | TP Size: $TP_SIZE"
echo "========================================================"

mkdir -p "$BASE_OUTPUT_DIR"

for model in "${MODELS[@]}"; do
    # 모델명 슬래시 치환 (폴더명 생성용)
    MODEL_SAFE_NAME="${model//\//_}"
    
    # 결과 저장 경로 설정 (모델별로 폴더 분리)
    MODEL_OUTPUT_DIR="$BASE_OUTPUT_DIR/$MODEL_SAFE_NAME/"
    mkdir -p "$MODEL_OUTPUT_DIR"
    
    OUTPUT_FILE="$MODEL_OUTPUT_DIR/${DATE_TAG}_memory1.jsonl"
    VERIFIER_FILE="$MODEL_OUTPUT_DIR/${DATE_TAG}_verifier_logs1.jsonl"
    REFINEMENT_FILE="$MODEL_OUTPUT_DIR/${DATE_TAG}_refinement_logs1.jsonl"

    echo "####################################################################"
    echo "[STEP 1] Starting vLLM Server for: $model"
    echo "####################################################################"

    # 4-1. vLLM 서버 실행
    # 파이썬 코드가 JSON 모드를 사용하므로, vLLM이 이를 지원하도록 설정 (기본적으로 지원함)
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup vllm serve "$model" \
        --host 0.0.0.0 \
        --port $PORT \
        --tensor-parallel-size $TP_SIZE \
        --max-model-len 16384 \
        --gpu-memory-utilization 0.95 \
        --trust-remote-code > vllm_server.log 2>&1 &
    #16384
    SERVER_PID=$!
    echo ">> vLLM Server PID: $SERVER_PID"

    # 4-2. 서버 대기
    wait_for_server

    # 4-3. 파이썬 스크립트 실행
    echo "[STEP 2] Running Memory Generation Script..."
    echo "Saving to: $MODEL_OUTPUT_DIR"

    python "$PYTHON_SCRIPT" \
        --input "$INPUT_DATA" \
        --output "$OUTPUT_FILE" \
        --verifier_output "$VERIFIER_FILE" \
        --refinement_output "$REFINEMENT_FILE" \
        --model "$model" \
        --api_base "$VLLM_URL" \
        --api_key "EMPTY" \
        --concurrency "$CONCURRENCY"

    # 4-4. 서버 종료
    echo "[STEP 3] Stopping vLLM Server..."
    kill $SERVER_PID
    wait $SERVER_PID 2>/dev/null
    SERVER_PID="" # PID 초기화
    
    echo ">> Server Stopped."
    echo "--------------------------------------------------------"
    
    # GPU 메모리 정리 대기
    sleep 10
done

echo "========================================================"
echo "All Memory Generation Jobs Finished at $(date)"
echo "========================================================"