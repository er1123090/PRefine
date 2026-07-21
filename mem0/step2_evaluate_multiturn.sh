#!/bin/bash

# ==============================================================================
# [설정 영역]
# ==============================================================================

# 1. Python 스크립트 파일명
PYTHON_SCRIPT="/data/minseo/experiments4/mem0/step2_evaluate_multiturn.py"

# 2. 실험 식별자 (파일명 생성용)
TAG="test1"                                  # 예: test1
DATE="0309"                         # 오늘 날짜 (예: 0216)
MODEL_NAME="gpt-5" # 실제 API 호출에 쓸 모델명
MODEL_FILE_NAME="gpt-5"
# 3. 경로 설정 (고정 경로)
BASE_DIR="/data/minseo/experiments4"
OUTPUT_ROOT="${BASE_DIR}/mem0/inference/multiturn"

# 데이터 파일 경로들
INPUT_PATH="${BASE_DIR}/data/1229_dev_6.json"
MULTITURN_PATH="${BASE_DIR}/query_multiturn-domain.json"
PREF_LIST_PATH="${BASE_DIR}/pref_list.json"
PREF_GROUP_PATH="${BASE_DIR}/pref_group.json"
TOOLS_SCHEMA_PATH="${BASE_DIR}/schema_all.json"

# 4. 기타 설정
CONTEXT_TYPE="memory_only"
CONCURRENCY=5

# ==============================================================================
# [실행 로직]
# ==============================================================================

# 실행할 pref_type 목록
PREF_TYPES=("easy" "medium" "hard")

for PREF in "${PREF_TYPES[@]}"; do
    
    # 1. 결과 저장용 디렉토리 생성 (없으면 생성)
    # 예: /data/minseo/experiments4/mem0/inference/multiturn/easy/
    TARGET_DIR="${OUTPUT_ROOT}/${PREF}"
    mkdir -p "$TARGET_DIR"

    # 2. 파일명 조합
    # 형식: {날짜}_{모델명}_{태그}.json
    # 예: 0216_gemini-flash-thinking_test1.json
    FILENAME="${DATE}_${MODEL_FILE_NAME}_${TAG}"
    
    OUTPUT_FILE="${TARGET_DIR}/${FILENAME}.json"
    LOG_FILE="${TARGET_DIR}/${FILENAME}.log"

    echo "======================================================================"
    echo " [START] Pref Type: $PREF"
    echo " Save Path: $OUTPUT_FILE"
    echo "======================================================================"

    # 3. Python 실행
    python "$PYTHON_SCRIPT" \
        --input_path "$INPUT_PATH" \
        --output_path "$OUTPUT_FILE" \
        --log_path "$LOG_FILE" \
        --pref_type "$PREF" \
        --context_type "$CONTEXT_TYPE" \
        --model_name "$MODEL_NAME" \
        --concurrency "$CONCURRENCY" \
        --reasoning_effort "minimal" 

    # 4. 종료 코드 확인
    if [ $? -eq 0 ]; then
        echo " [SUCCESS] Completed: $PREF"
    else
        echo " [ERROR] Failed: $PREF"
        # 에러 발생 시 중단하려면 아래 주석 해제
        # exit 1 
    fi
    
    echo ""
    sleep 2
done

echo "All tasks finished."