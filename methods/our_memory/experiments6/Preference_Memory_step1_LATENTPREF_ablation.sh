#!/bin/bash

# ==============================================================================
# 1. 환경 설정 및 경로
# ==============================================================================

# Python 스크립트 절대 경로
PYTHON_SCRIPT="/data/minseo/experiments6/ours_memory/Preference_Memory_step1_LATENTPREF_ablation.py"

# 입력 데이터 경로
INPUT_DATA="/data/minseo/experiments6/data/1229_dev_6.json"

# 결과 저장 최상위 루트
BASE_OUTPUT_DIR="/data/minseo/experiments6/ours_memory/inference/0216_MEMORY"

# 모델 설정
PROVIDER="openai"         # or "google"
MODEL_NAME="gpt-4o-mini"  
# API_KEY="sk-..."       # 필요한 경우 여기에 직접 입력 (보안상 환경변수 권장)
CONCURRENCY=20

# ==============================================================================
# 2. 메인 루프 (Verifier 0회 ~ 5회 반복)
# ==============================================================================

echo "========================================================"
echo "Starting Verifier Loop Experiment (0 to 5)"
echo "Start Time: $(date)"
echo "========================================================"

mkdir -p "$BASE_OUTPUT_DIR"

# 0부터 5까지 순회
for RETRY_COUNT in {0..5}; do

    echo "####################################################################"
    echo "[RUNNING] Max Retries: $RETRY_COUNT"
    echo "####################################################################"

    # 1. 결과 저장할 폴더 생성 (retries_0, retries_1, ...)
    CURRENT_DIR="$BASE_OUTPUT_DIR/retries_${RETRY_COUNT}"
    mkdir -p "$CURRENT_DIR"

    # 2. 파일 경로 설정
    OUTPUT_FILE="$CURRENT_DIR/memory_result.jsonl"
    VERIFIER_LOG="$CURRENT_DIR/verifier_log.jsonl"
    REFINEMENT_LOG="$CURRENT_DIR/refinement_log.jsonl"
    LOG_FILE="$CURRENT_DIR/run_output.log"

    # 3. Python 스크립트 실행
    # nohup 없이 실행 (순차 진행), 로그는 파일로 저장
    python "$PYTHON_SCRIPT" \
        --input "$INPUT_DATA" \
        --output "$OUTPUT_FILE" \
        --verifier_output "$VERIFIER_LOG" \
        --refinement_output "$REFINEMENT_LOG" \
        --provider "$PROVIDER" \
        --model "$MODEL_NAME" \
        --concurrency "$CONCURRENCY" \
        --max_retries "$RETRY_COUNT" \
        > "$LOG_FILE" 2>&1

    echo " >> [DONE] Saved to: $CURRENT_DIR"
    echo "--------------------------------------------------------"
    
    # API Rate Limit 보호를 위한 쿨다운 (필요시 조절)
    sleep 3

done

echo "========================================================"
echo "All Experiments Finished at $(date)"
echo "========================================================"