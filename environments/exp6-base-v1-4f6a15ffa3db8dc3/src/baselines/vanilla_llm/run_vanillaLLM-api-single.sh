#!/bin/bash

# ==============================================================================
# 1. 환경 설정 및 경로
# ==============================================================================
# [중요] 실행할 파이썬 파일명 확인
PYTHON_SCRIPT="/data/minseo/experiments6/vanillaLLM/vanillaLLM_inference-api-single.py"

# 데이터 및 스키마 경로
INPUT_PATH="/data/minseo/experiments6/data/1229_dev_6.json"
QUERY_PATH="/data/minseo/experiments6/query_singleturn.json"
PREF_LIST_PATH="/data/minseo/experiments6/pref_list.json"
PREF_GROUP_PATH="/data/minseo/experiments6/pref_group.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments6/schema_easy.json"

# 출력 및 로그 디렉토리
BASE_OUTPUT_DIR="/data/minseo/experiments6/vanillaLLM/inference/1231-1_output_singleturn"
BASE_LOG_DIR="/data/minseo/experiments6/vanillaLLM/inference/1231-1_logs_singleturn"

# 태그 설정
DATE_TAG="$(date +%m%d)"
TEST_TAG="test_1"

# 동시성 설정
CONCURRENCY=2

# ==============================================================================
# 2. 실험 변수 (모델 및 프롬프트)
# ==============================================================================

# [Model List]
# Claude 모델명 예시: "claude-3-7-sonnet-20250219", "claude-opus-4-5-20251101" 등
MODELS=(
    #"gpt-4o-mini-2024-07-18"  # 일반 모델
    #"gpt-5-mini"
    #"gpt-5"
    #"gemini-3-flash-preview"
    "gemini-3-pro-preview"

    #"claude-opus-4-5-20251101" # [NEW] Claude 모델 추가 예시
    #"claude-3-5-sonnet-20241022"
)

# [Prompt Types]
PROMPT_TYPES=("imp-zs") #"imp-pref-group"

# [Context Types]
CONTEXT_TYPES=("diag-apilist")

# [Preference Types]
PREF_TYPES=("easy" "medium" "hard") # "medium" "hard"

# ==============================================================================
# 3. 배치 실행 로직
# ==============================================================================

echo "========================================================"
echo "Batch Inference Started at $(date)"
echo "Models: ${MODELS[*]}"
echo "Prompts: ${PROMPT_TYPES[*]}"
echo "========================================================"

for model in "${MODELS[@]}"; do
    # 모델명 안전 변환 (슬래시를 언더스코어로)
    MODEL_SAFE_NAME="${model//\//_}"

    # ------------------------------------------------------------------
    # [Reasoning Effort 설정 로직]
    # 모델명에 gpt-5, gemini-3, o1, o3, claude 등이 포함되면 Reasoning 레벨 루프를 돔
    # Python 스크립트 내부에서 모델명에 따라 적절한 API 파라미터(thinking/budget 등)로 변환함
    # ------------------------------------------------------------------
    EFFORT_LEVELS=("default")  # 기본값 (설정 없음)
    
    if [[ "$model" == *"gpt-5"* ]] || \
       [[ "$model" == *"gemini-3"* ]] || \
       [[ "$model" == *"o1"* ]] || \
       [[ "$model" == *"o3"* ]] || \
       [[ "$model" == *"claude"* ]]; then  # [NEW] claude 조건 추가
       
        # Python 코드의 매핑에 따라:
        # - Claude 3.7 등: 'medium' -> budget_tokens=8192
        # - Opus 4.5: 'medium' -> effort='medium'
        EFFORT_LEVELS=("low") # 필요에 따라 "low", "high", "minimal" 로 변경 가능
    fi

    for prompt_type in "${PROMPT_TYPES[@]}"; do
        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do
                for effort in "${EFFORT_LEVELS[@]}"; do

                    echo ""
                    echo "--------------------------------------------------------------------------------"
                    echo "[RUNNING] Model: $model | Prompt: $prompt_type | Pref: $pref | Effort: $effort"
                    echo "--------------------------------------------------------------------------------"

                    # ------------------------------------------------------------------
                    # 디렉토리 및 파일명 생성 (Effort 폴더 추가)
                    # ------------------------------------------------------------------
                    # 구조: BASE / context / pref / model / prompt / effort / filename
                    OUTPUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/singleturn-query/$MODEL_SAFE_NAME-$effort/$prompt_type"
                    LOG_DIR="$BASE_LOG_DIR/$context/$pref/singleturn-query/$MODEL_SAFE_NAME-$effort/$prompt_type"
                    
                    mkdir -p "$OUTPUT_DIR"
                    mkdir -p "$LOG_DIR"

                    FILENAME="${DATE_TAG}_${TEST_TAG}.json"
                    LOGNAME="${DATE_TAG}_${TEST_TAG}.jsonl"

                    OUTPUT_FILE="$OUTPUT_DIR/$FILENAME"
                    LOG_FILE="$LOG_DIR/$LOGNAME"

                    # ------------------------------------------------------------------
                    # Python 명령어 구성 (effort 인자 처리)
                    # ------------------------------------------------------------------
                    CMD="python $PYTHON_SCRIPT \
                        --input_path $INPUT_PATH \
                        --query_path $QUERY_PATH \
                        --pref_list_path $PREF_LIST_PATH \
                        --pref_group_path $PREF_GROUP_PATH \
                        --tools_schema_path $TOOLS_SCHEMA_PATH \
                        --context_type $context \
                        --pref_type $pref \
                        --prompt_type $prompt_type \
                        --model_name $model \
                        --output_path $OUTPUT_FILE \
                        --log_path $LOG_FILE \
                        --concurrency $CONCURRENCY"

                    # default가 아니면 reasoning_effort 인자 추가
                    if [ "$effort" != "default" ]; then
                        CMD="$CMD --reasoning_effort $effort"
                    fi

                    # ------------------------------------------------------------------
                    # 실행
                    # ------------------------------------------------------------------
                    eval $CMD

                    # 실행 결과 확인
                    if [ $? -eq 0 ]; then
                        echo ">> [SUCCESS] Saved to: $OUTPUT_FILE"
                    else
                        echo ">> [ERROR] Failed at Model: $model (Effort: $effort)"
                    fi
                done
            done
        done
    done
done

echo ""
echo "========================================================"
echo "All Jobs Finished at $(date)"
echo "========================================================"