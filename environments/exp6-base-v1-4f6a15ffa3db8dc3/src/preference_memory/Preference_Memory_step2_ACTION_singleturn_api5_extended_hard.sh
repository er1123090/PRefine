#!/bin/bash

set -euo pipefail

# ==============================================================================
# 1. 환경 설정 및 경로
# ==============================================================================

# Python 스크립트 경로
PYTHON_SCRIPT="/data/minseo/experiments6/ours_memory/Preference_Memory_step2_ACTION_singleturn_api.py"

# 고정 데이터 경로
INPUT_PATH="/data/minseo/experiments6/data/1229_dev_6.json"
# extended single-turn query만 사용
QUERY_PATH="/data/minseo/experiments6/query_new_single.json"
PREF_LIST_PATH="/data/minseo/experiments6/pref_list_extended.json"
PREF_GROUP_PATH="/data/minseo/experiments6/pref_group_extended.json"
TOOLS_SCHEMA_PATH="/data/minseo/experiments6/schema_easy_extended.json"

# 동시 처리 수 (API Rate Limit 주의)
CONCURRENCY=10
MAX_QUERIES="${MAX_QUERIES:-400}"

# ==============================================================================
# 2. 실험 변수 설정
# ==============================================================================

# [리스트 A] 사용할 OpenAI/API 모델 목록
MODELS=(
    ##"gpt-5-mini"
    ##"gpt-4o-mini"
    #"gemini-3-flash-preview"
    "gpt-5"
)

# [리스트 B] 사용할 메모리가 저장된 폴더명 목록 (스크립트 1과 동일하게 맞춤)
MEMORY_FOLDERS=(
    #"deepseek-ai_DeepSeek-R1-0528-Qwen3-8B"
    # "deepseek-ai_DeepSeek-R1-Distill-Llama-8B"
    "google_gemma-3-12b-it"
    "gpt-4o-mini"
    ##"meta-llama_Llama-3.1-8B-Instruct"
    # "Qwen_Qwen3-8B"
)

# gpt-4o-mini 메모리까지 완료하면 이후 메모리 추론은 중단
STOP_AFTER_MEMORY_FOLDER="${STOP_AFTER_MEMORY_FOLDER:-gpt-4o-mini}"
STOP_RUN_AFTER_TARGET_MEMORY="${STOP_RUN_AFTER_TARGET_MEMORY:-1}"

# 실험 조건
CONTEXT_TYPES=("memory_api")   # choices: memory_only, memory_diag, memory_api
PREF_TYPES=("hard")            # choices: easy, medium, hard
PROMPT_NAME="implicit_zs"

# ==============================================================================
# 3. 메인 루프 실행
# ==============================================================================

echo "####################################################################"
echo "Starting Inference using OpenAI/Gemini API (Single-turn Hard, New Query Only)"
echo "####################################################################"

STOP_REQUESTED=0

# [Outer Loop] 모델 변경
for model in "${MODELS[@]}"; do
    MODEL_SAFE_NAME="${model//\//_}"
    
    echo ">> [Model] Current Inference Model: $model"

    if [[ "$model" == *"gemini"* ]] && [[ -z "${GOOGLE_API_KEY:-}" ]]; then
        echo "[ERROR] GOOGLE_API_KEY is not set. Gemini inference cannot start."
        exit 1
    fi

    # [Inner Loop] 메모리 폴더 변경
    for MEM_FOLDER in "${MEMORY_FOLDERS[@]}"; do

        # ----------------------------------------------------------------------
        # 경로 동적 설정
        # ----------------------------------------------------------------------
        # 1. 메모리 파일 경로 설정
        MEMORY_PATH="/data/minseo/experiments6/ours_memory/inference/1231_MEMORY3/${MEM_FOLDER}/_memory1.jsonl"
        
        # 2. 결과 저장 루트 (메모리 소스별로 폴더 구분)
        # 예: .../inference/1231_MEMORY3_inference1_single/deepseek-ai_DeepSeek.../
        BASE_OUTPUT_DIR="/data/minseo/experiments6/ours_memory/inference/extended/outputs/singleturn/api/${MEM_FOLDER}"
        BASE_LOG_DIR="/data/minseo/experiments6/ours_memory/inference/extended/logs/singleturn/api/${MEM_FOLDER}"
        
        mkdir -p "$BASE_OUTPUT_DIR"
        mkdir -p "$BASE_LOG_DIR"

        # ----------------------------------------------------------------------

        for context in "${CONTEXT_TYPES[@]}"; do
            for pref in "${PREF_TYPES[@]}"; do

                echo "   >> [Processing]"
                echo "      - Memory Source : $MEM_FOLDER"
                echo "      - Context/Pref  : $context / $pref"

                # 결과 저장 경로 생성
                DATE_TAG="$(date +%m%d)"
                
                # 최종 저장 경로: [Base]/[Context]/[Pref]/[ModelName]/[PromptName]
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
                    --query_path "$QUERY_PATH" \
                    --pref_list_path "$PREF_LIST_PATH" \
                    --pref_group_path "$PREF_GROUP_PATH" \
                    --tools_schema_path "$TOOLS_SCHEMA_PATH" \
                    --context_type "$context" \
                    --pref_type "$pref" \
                    --model_name "$model" \
                    --concurrency "$CONCURRENCY" \
                    --reasoning_effort "high" \
                    --max_queries "$MAX_QUERIES"

            done
        done

        if [ "${STOP_RUN_AFTER_TARGET_MEMORY}" = "1" ] && [ "$MEM_FOLDER" = "$STOP_AFTER_MEMORY_FOLDER" ]; then
            echo ">> Stop requested after memory source: $MEM_FOLDER"
            STOP_REQUESTED=1
            break
        fi
    done # End of Memory Loop

    if [ "${STOP_REQUESTED}" = "1" ]; then
        echo ">> Exiting after requested memory boundary: ${STOP_AFTER_MEMORY_FOLDER}"
        break
    fi
    
    echo "--------------------------------------------------------"
done # End of Model Loop

echo "========================================================"
echo "All API Jobs Finished Successfully"
echo "========================================================"
