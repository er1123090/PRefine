#!/bin/bash

# ==============================================================================
# 1. 환경 설정 및 경로
# ==============================================================================

# Python 스크립트 절대 경로 (Multi-turn API용)
PYTHON_SCRIPT="/data/minseo/experiments6/ours_memory/Preference_Memory_step2_ACTION_multiturn_api.py"

# 고정 데이터 경로
INPUT_PATH="/data/minseo/experiments6/data/1229_dev_6.json"
QUERY_PATH="/data/minseo/experiments6/query_multiturn-domain.json"
PREF_LIST_PATH="/data/minseo/experiments6/pref_list.json"
PREF_GROUP_PATH="/data/minseo/experiments6/pref_group.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments6/schema_all.json"

# 동시 처리 수
CONCURRENCY=10

# ==============================================================================
# 2. 실험 변수 설정
# ==============================================================================

# [리스트 A] 실행할 클라우드 모델 목록
MODELS=(
    "gpt-5"
    #"gpt-4o-mini"
    #"gpt-5-mini"
    #"gemini-3-flash-preview"
)

# [리스트 B] 사용할 메모리가 저장된 폴더명 목록
MEMORY_FOLDERS=(
    "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B"
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B"
    "google_gemma-3-12b-it"
    "gpt-4o-mini"
    ##"meta-llama_Llama-3.1-8B-Instruct"
    ##"Qwen_Qwen3-8B"
)

# 실험 조건
CONTEXT_TYPES=("memory_api") # choices: memory_only, memory_diag, memory_api
PREF_TYPES=("medium" "hard" "easy")      # choices: easy, medium, hard
PROMPT_NAME="implicit_zs"

# ==============================================================================
# 3. 메인 루프 실행
# ==============================================================================

echo "========================================================"
echo "Starting Cloud API Inference (Multi-turn)"
echo "========================================================"

# [Outer Loop] 모델 변경
for model in "${MODELS[@]}"; do
    
    # 모델명 슬래시 처리 (파일 경로용)
    MODEL_SAFE_NAME="${model//\//_}"

    echo "####################################################################"
    echo "[Processing Model]: $model"
    echo "####################################################################"

    # [Inner Loop] 메모리 폴더 변경
    for MEM_FOLDER in "${MEMORY_FOLDERS[@]}"; do

        # ----------------------------------------------------------------------
        # 경로 동적 설정
        # ----------------------------------------------------------------------
        
        # 1. 메모리 파일 경로 (폴더명에 따라 변경)
        MEMORY_PATH="/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3/${MEM_FOLDER}/_memory1.jsonl"
        
        # 2. 결과 저장 루트 (메모리 소스별로 폴더 구분, multi 폴더 사용)
        BASE_OUTPUT_DIR="/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3_0306-inference_multi/${MEM_FOLDER}"
        BASE_LOG_DIR="/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3_0306-logs_multi/${MEM_FOLDER}"
        
        mkdir -p "$BASE_OUTPUT_DIR"
        mkdir -p "$BASE_LOG_DIR"

        # ----------------------------------------------------------------------

        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do

                echo " >> [RUNNING]"
                echo "    - Memory Source : $MEM_FOLDER"
                echo "    - Context/Pref  : $context / $pref"

                # 결과 저장 경로 생성
                DATE_TAG="$(date +%m%d)"
                CURRENT_OUT_DIR="$BASE_OUTPUT_DIR/$context/$pref/${MODEL_SAFE_NAME}_high/$PROMPT_NAME"
                CURRENT_LOG_DIR="$BASE_LOG_DIR/$context/$pref/${MODEL_SAFE_NAME}_high/$PROMPT_NAME"
                
                mkdir -p "$CURRENT_OUT_DIR"
                mkdir -p "$CURRENT_LOG_DIR"

                OUTPUT_FILE="$CURRENT_OUT_DIR/${DATE_TAG}_test1.json"
                LOG_FILE="$CURRENT_LOG_DIR/${DATE_TAG}_test1.log"

                # Python 스크립트 실행
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
                    --base_url "" \
                    --api_key "ENV" \
                    --concurrency "$CONCURRENCY" \
                    #--reasoning_effort "high"

            done
        done
    done # End of Memory Loop

    echo "--------------------------------------------------------"
    # API Rate Limit 고려 잠시 대기
    sleep 2

done # End of Model Loop

echo "========================================================"
echo "All Cloud API Jobs Finished at $(date)"
echo "========================================================"