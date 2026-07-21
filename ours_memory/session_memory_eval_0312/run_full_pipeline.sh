#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$SCRIPT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"

MEMORY_ROOT="${MEMORY_ROOT:-/data/minseo/experiments4/ours_memory/inference/0312_MEMORY1}"
DATASET_PATH="${DATASET_PATH:-/data/minseo/experiments4/data/1229_dev_6.json}"
PREP_OUTPUT_ROOT="${PREP_OUTPUT_ROOT:-$PACKAGE_ROOT/outputs/prepared}"
RUN_OUTPUT_ROOT="${RUN_OUTPUT_ROOT:-$PACKAGE_ROOT/outputs/runs}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-$PACKAGE_ROOT/outputs/eval}"
PLOT_OUTPUT_ROOT="${PLOT_OUTPUT_ROOT:-$PACKAGE_ROOT/outputs/plots}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
TASK="${TASK:-singleturn}"
TASK_EXPLICIT=0
COHORTS_CSV="${COHORTS_CSV:-all}"
MAX_SESSION="${MAX_SESSION:-}"
SAVE_OR_LOGS="${SAVE_OR_LOGS:-0}"
SAVE_PARSING_FAILURES="${SAVE_PARSING_FAILURES:-0}"
EVAL_LOGS_DIR="${EVAL_LOGS_DIR:-}"

RUNNER_ARGS=()

print_help() {
    cat <<EOF
Usage: $(basename "$0") [pipeline options] [-- runner options]

Pipeline options:
  --task singleturn|multiturn
  --memory-root PATH
  --dataset-path PATH
  --prep-output-root PATH
  --run-output-root PATH
  --eval-output-root PATH
  --plot-output-root PATH
  --run-tag TAG
  --cohorts CSV
  --max-session N
  --help

Any additional arguments after -- are forwarded to run_session_memory_eval.sh.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK="$2"
            TASK_EXPLICIT=1
            shift 2
            ;;
        --task=*)
            TASK="${1#*=}"
            TASK_EXPLICIT=1
            shift
            ;;
        --memory-root)
            MEMORY_ROOT="$2"
            shift 2
            ;;
        --dataset-path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --prep-output-root)
            PREP_OUTPUT_ROOT="$2"
            shift 2
            ;;
        --run-output-root)
            RUN_OUTPUT_ROOT="$2"
            shift 2
            ;;
        --eval-output-root)
            EVAL_OUTPUT_ROOT="$2"
            shift 2
            ;;
        --plot-output-root)
            PLOT_OUTPUT_ROOT="$2"
            shift 2
            ;;
        --run-tag)
            RUN_TAG="$2"
            shift 2
            ;;
        --cohorts)
            COHORTS_CSV="$2"
            shift 2
            ;;
        --max-session)
            MAX_SESSION="$2"
            shift 2
            ;;
        --help)
            print_help
            exit 0
            ;;
        --)
            shift
            RUNNER_ARGS+=("$@")
            break
            ;;
        *)
            RUNNER_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$TASK" in
    singleturn|multiturn)
        ;;
    *)
        echo "[ERROR] Unsupported task: $TASK" >&2
        exit 1
        ;;
esac

if [[ "$TASK_EXPLICIT" == "0" ]]; then
    echo "[WARN] --task was not provided. Defaulting to singleturn."
fi

IFS=',' read -r -a COHORTS <<< "$COHORTS_CSV"

PREP_ARGS=(
    --memory_root "$MEMORY_ROOT"
    --dataset_path "$DATASET_PATH"
    --output_root "$PREP_OUTPUT_ROOT"
    --cohorts "${COHORTS[@]}"
)

if [[ -n "$MAX_SESSION" ]]; then
    PREP_ARGS+=(--max_session "$MAX_SESSION")
fi

mkdir -p "$EVAL_OUTPUT_ROOT" "$PLOT_OUTPUT_ROOT"

TASK_RUN_OUTPUT_ROOT="$RUN_OUTPUT_ROOT/$TASK"
TASK_EVAL_OUTPUT_ROOT="$EVAL_OUTPUT_ROOT/$TASK"
TASK_PLOT_OUTPUT_ROOT="$PLOT_OUTPUT_ROOT/$TASK"

if [[ -z "$EVAL_LOGS_DIR" ]]; then
    EVAL_LOGS_DIR="$TASK_EVAL_OUTPUT_ROOT/logs/$RUN_TAG"
fi

mkdir -p "$TASK_RUN_OUTPUT_ROOT" "$TASK_EVAL_OUTPUT_ROOT" "$TASK_PLOT_OUTPUT_ROOT"

echo "[PIPELINE] Stage 1/4: prepare session memory artifacts"
"$PYTHON_BIN" "$PACKAGE_ROOT/prepare_session_memory_eval.py" "${PREP_ARGS[@]}"

echo "[PIPELINE] Stage 2/4: run step2 inference"
"$PACKAGE_ROOT/run_session_memory_eval.sh" \
    --task "$TASK" \
    --prep-manifest "$PREP_OUTPUT_ROOT/prep_manifest.jsonl" \
    --output-root "$TASK_RUN_OUTPUT_ROOT" \
    --run-tag "$RUN_TAG" \
    "${RUNNER_ARGS[@]}"

RUN_MANIFEST="$TASK_RUN_OUTPUT_ROOT/$RUN_TAG/run_manifest.jsonl"
DETAIL_CSV="$TASK_EVAL_OUTPUT_ROOT/${RUN_TAG}_detailed.csv"
SUMMARY_CSV="$TASK_EVAL_OUTPUT_ROOT/${RUN_TAG}_summary.csv"
PLOT_DIR="$TASK_PLOT_OUTPUT_ROOT/$RUN_TAG"

EVAL_ARGS=(
    --task "$TASK"
    --run_manifest "$RUN_MANIFEST"
    --out_csv "$DETAIL_CSV"
    --out_summary_csv "$SUMMARY_CSV"
    --logs_dir "$EVAL_LOGS_DIR"
)

if [[ "$SAVE_OR_LOGS" == "1" ]]; then
    EVAL_ARGS+=(--save_or_logs)
fi
if [[ "$SAVE_PARSING_FAILURES" == "1" ]]; then
    EVAL_ARGS+=(--save_parsing_failures)
fi

echo "[PIPELINE] Stage 3/4: evaluate outputs"
"$PYTHON_BIN" "$PACKAGE_ROOT/evaluate_session_memory_results.py" "${EVAL_ARGS[@]}"

echo "[PIPELINE] Stage 4/4: plot session curves"
"$PYTHON_BIN" "$PACKAGE_ROOT/plot_session_memory_curves.py" \
    --task "$TASK" \
    --summary_csv "$SUMMARY_CSV" \
    --plot_dir "$PLOT_DIR"

echo "[PIPELINE DONE]"
echo "  - Run manifest : $RUN_MANIFEST"
echo "  - Detail CSV   : $DETAIL_CSV"
echo "  - Summary CSV  : $SUMMARY_CSV"
echo "  - Plot dir     : $PLOT_DIR"
