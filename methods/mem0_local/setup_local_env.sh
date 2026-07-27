#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/data/minseo/.venvs/experiment8/bin/python}"
UPSTREAM_DIR="${UPSTREAM_DIR:-${SCRIPT_DIR}/upstream/mem0}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/bootstrap_upstream.py" \
    --repo_path "${UPSTREAM_DIR}"
"${PYTHON_BIN}" -m pip install --editable "${UPSTREAM_DIR}[nlp]"
"${PYTHON_BIN}" -m pip install "fastembed>=0.3.1"
"${PYTHON_BIN}" -m spacy download en_core_web_sm

printf 'mem0_local environment ready: python=%s upstream=%s\n' \
    "${PYTHON_BIN}" "${UPSTREAM_DIR}"
