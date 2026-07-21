#!/bin/bash

# ==============================================================================
# 2. 경로 및 환경 설정
# ==============================================================================

# Python 스크립트 경로 (저장한 파이썬 파일명으로 변경)
PYTHON_SCRIPT="/data/minseo/experiments6/ours_memory/Preference_Memory_step2_ACTION_singleturn_api.py"

# 데이터 경로
INPUT_PATH="/data/minseo/experiments6/data/1229_dev_6.json"
MEMORY_PATH="/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3/google_gemma-3-12b-it/_memory1.jsonl"
QUERY_PATH="/data/minseo/experiments6/query_singleturn.json"
PREF_LIST_PATH="/data/minseo/experiments6/pref_list.json"
PREF_GROUP_PATH="/data/minseo/experiments6/pref_group.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments6/schema_easy.json"

# 결과 저장 루트 디렉토리
BASE_OUTPUT_DIR="/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3_inference1_single/google_gemma-3-12b-it"
BASE_LOG_DIR="/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3_logs1_single/google_gemma-3-12b-it"

# 동시 처리 수 (OpenAI는 vLLM보다 높게 설정 가능하나, Rate Limit 주의)
CONCURRENCY=10

# ==============================================================================
# 3. 실험 변수 설정
# ==============================================================================

# 실행할 OpenAI 모델 목록
MODELS=(
    #"gpt-5-mini"
    #"gpt-4o-mini"
    "gemini-3-flash-preview"
)

# 실험 조건 (루프용)
CONTEXT_TYPES=("memory_api")   # choices: memory_only, memory_diag, memory_api
PREF_TYPES=("medium" "hard") # choices: easy, medium, hard
PROMPT_NAME="implicit_zs"

# ==============================================================================
# 4. 메인 루프 실행
# ==============================================================================

mkdir -p "$BASE_OUTPUT_DIR"
mkdir -p "$BASE_LOG_DIR"

echo "####################################################################"
echo "Starting Inference using OpenAI API"
echo "####################################################################"

for model in "${MODELS[@]}"; do
    echo ">> [Model] Current Model: $model"

    for context in "${CONTEXT_TYPES[@]}"; do
        for pref in "${PREF_TYPES[@]}"; do

            echo "   >> [Processing] Context: $context | Pref: $pref"

            # 결과 저장 경로 생성
            DATE_TAG="$(date +%m%d)"
            
            # 폴더명에 모델명 포함 (슬래시가 없으므로 치환 불필요)
            CURRENT_OUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/$model/$PROMPT_NAME"
            CURRENT_LOG_DIR="$BASE_LOG_DIR/$context/$pref/$model/$PROMPT_NAME"
            
            mkdir -p "$CURRENT_OUT_DIR"
            mkdir -p "$CURRENT_LOG_DIR"

            OUTPUT_FILE="$CURRENT_OUT_DIR/${DATE_TAG}_test1.json"
            LOG_FILE="$CURRENT_LOG_DIR/${DATE_TAG}_test1.log"

            # Python 스크립트 실행
            # --base_url 및 --api_key 인자를 제거했습니다. 
            # (파이썬 코드가 환경변수 OPENAI_API_KEY를 자동으로 인식합니다)
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
                --concurrency "$CONCURRENCY" \
                --reasoning_effort "high"

        done
    done
    
    echo "--------------------------------------------------------"
done

echo "========================================================"
echo "All OpenAI API Jobs Finished Successfully"
echo "========================================================"