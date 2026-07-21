#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASK="${SESSION_MEMORY_EVAL_TASK:-singleturn}"
FORWARDED_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)
            TASK="$2"
            shift 2
            ;;
        --task=*)
            TASK="${1#*=}"
            shift
            ;;
        *)
            FORWARDED_ARGS+=("$1")
            shift
            ;;
    esac
done

case "$TASK" in
    singleturn)
        TARGET="$SCRIPT_DIR/run_session_memory_eval_singleturn.sh"
        ;;
    multiturn)
        TARGET="$SCRIPT_DIR/run_session_memory_eval_multiturn.sh"
        ;;
    *)
        echo "[ERROR] Unsupported task: $TASK" >&2
        echo "Expected one of: singleturn, multiturn" >&2
        exit 1
        ;;
esac

exec "$TARGET" "${FORWARDED_ARGS[@]}"
