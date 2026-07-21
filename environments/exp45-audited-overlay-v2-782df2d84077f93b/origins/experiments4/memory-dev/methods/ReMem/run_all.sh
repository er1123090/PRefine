#!/usr/bin/env bash
set -euo pipefail

ts="$(date +%y%m%d_%H%M%S)"
log_dir="memory-dev/methods/ReMem/logs"
model="${1:-gpt-5-mini}"

mkdir -p "$log_dir"

nohup python memory-dev/methods/ReMem/script.py --pref_type easy --model "$model" \
  > "$log_dir/run_easy_${ts}.log" 2>&1 &

nohup python memory-dev/methods/ReMem/script.py --pref_type medium --model "$model" \
  > "$log_dir/run_medium_${ts}.log" 2>&1 &

nohup python memory-dev/methods/ReMem/script.py --pref_type hard --model "$model" \
  > "$log_dir/run_hard_${ts}.log" 2>&1 &

echo "Started easy/medium/hard runs with timestamp ${ts} (model=${model})"
