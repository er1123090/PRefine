#!/bin/bash

# ==============================================================================
# [설정 영역]
# ==============================================================================

# 1. Python 스크립트 파일명
PYTHON_SCRIPT="/data/minseo/experiments4/mem0/step2_evaluate_singleturn.py"

# 2. 실험 식별자 (파일명 생성용)
TAG="test1"                     # 예: test1
DATE="$(date +%m%d)"            # 예: 0309
MODEL_NAME="gpt-5"              # 실제 API 호출에 쓸 모델명
MODEL_FILE_NAME="${MODEL_NAME//\//_}"  # 파일명에 사용할 모델명

# 3. 경로 설정 (고정 경로)
BASE_DIR="/data/minseo/experiments4"
OUTPUT_ROOT="${BASE_DIR}/mem0/inference_0309/singleturn"

# 데이터 파일 경로들
INPUT_PATH="${BASE_DIR}/data/1229_dev_6.json"
QUERY_PATH="${BASE_DIR}/query_singleturn.json"
PREF_LIST_PATH="${BASE_DIR}/pref_list.json"
PREF_GROUP_PATH="${BASE_DIR}/pref_group.json"
TOOLS_SCHEMA_PATH="${BASE_DIR}/schema_easy.json"

# 4. 기타 설정
CONCURRENCY=5
REASONING_EFFORT="medium"

# ==============================================================================
# [실행 로직]
# ==============================================================================

# 실행할 케이스 목록
CONTEXT_TYPES=("memory_api")      #"memory_only" "memory_api"
PREF_TYPES=("easy" "medium" "hard")

for CONTEXT in "${CONTEXT_TYPES[@]}"; do
    for PREF in "${PREF_TYPES[@]}"; do

        # 결과 저장용 디렉토리 생성
        # 예: /data/minseo/experiments4/mem0/inference/singleturn/memory_api/easy/
        TARGET_DIR="${OUTPUT_ROOT}/${CONTEXT}/${PREF}"
        mkdir -p "$TARGET_DIR"

        # 파일명 조합
        FILENAME="${DATE}_${MODEL_FILE_NAME}_${TAG}"
        OUTPUT_FILE="${TARGET_DIR}/${FILENAME}.json"
        LOG_FILE="${TARGET_DIR}/${FILENAME}.log"

        echo "======================================================================"
        echo " [START] Context: $CONTEXT | Pref Type: $PREF"
        echo " Save Path: $OUTPUT_FILE"
        echo "======================================================================"

        python "$PYTHON_SCRIPT" \
            --input_path "$INPUT_PATH" \
            --output_path "$OUTPUT_FILE" \
            --log_path "$LOG_FILE" \
            --query_path "$QUERY_PATH" \
            --pref_list_path "$PREF_LIST_PATH" \
            --pref_group_path "$PREF_GROUP_PATH" \
            --tools_schema_path "$TOOLS_SCHEMA_PATH" \
            --pref_type "$PREF" \
            --context_type "$CONTEXT" \
            --model_name "$MODEL_NAME" \
            --concurrency "$CONCURRENCY" \
            --reasoning_effort "$REASONING_EFFORT"

        if [ $? -eq 0 ]; then
            echo " [SUCCESS] Completed: $CONTEXT / $PREF"
        else
            echo " [ERROR] Failed: $CONTEXT / $PREF"
            # 에러 발생 시 중단하려면 아래 주석 해제
            # exit 1
        fi

        echo ""
        sleep 2
    done
done

echo "All single-turn tasks finished."
