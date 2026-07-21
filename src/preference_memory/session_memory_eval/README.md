# Session Memory Eval

Standalone package for evaluating session-index memory snapshots from the
preference-memory outputs without changing the main inference code.

## Files
- `prepare_session_memory_eval.py`: builds subset datasets, session memory JSONL files, and a prep manifest.
- `run_session_memory_eval_singleturn.sh`: single-turn runner over prepared session artifacts.
- `run_session_memory_eval_multiturn.sh`: multi-turn runner over prepared session artifacts.
- `run_session_memory_eval.sh`: dispatcher that routes to `singleturn` or `multiturn`.
- `evaluate_session_memory_results_singleturn.py`: single-turn evaluator.
- `evaluate_session_memory_results_multiturn.py`: multi-turn evaluator.
- `evaluate_session_memory_results.py`: dispatcher that routes to `singleturn` or `multiturn`.
- `plot_session_memory_curves_singleturn.py`: single-turn plotting logic.
- `plot_session_memory_curves_multiturn.py`: multi-turn plotting logic.
- `plot_session_memory_curves.py`: dispatcher that routes to `singleturn` or `multiturn`.
- `run_full_pipeline.sh`: chained entrypoint for prepare -> run -> evaluate -> plot.

## Quick Start

Run these commands from the `experiments6` release root.

```bash
python src/preference_memory/session_memory_eval/prepare_session_memory_eval.py

bash src/preference_memory/session_memory_eval/run_session_memory_eval_singleturn.sh \
  --prep-manifest src/preference_memory/session_memory_eval/outputs/prepared/prep_manifest.jsonl

python src/preference_memory/session_memory_eval/evaluate_session_memory_results_singleturn.py \
  --run_manifest src/preference_memory/session_memory_eval/outputs/runs/singleturn/<RUN_TAG>/run_manifest.jsonl \
  --out_csv src/preference_memory/session_memory_eval/outputs/eval/singleturn/<RUN_TAG>_detailed.csv \
  --out_summary_csv src/preference_memory/session_memory_eval/outputs/eval/singleturn/<RUN_TAG>_summary.csv

python src/preference_memory/session_memory_eval/plot_session_memory_curves_singleturn.py \
  --summary_csv src/preference_memory/session_memory_eval/outputs/eval/singleturn/<RUN_TAG>_summary.csv \
  --plot_dir src/preference_memory/session_memory_eval/outputs/plots/singleturn/<RUN_TAG>
```

## Direct Multiturn Runner
```bash
bash src/preference_memory/session_memory_eval/run_session_memory_eval_multiturn.sh \
  --prep-manifest src/preference_memory/session_memory_eval/outputs/prepared/prep_manifest.jsonl
```

## Full Pipeline
```bash
bash src/preference_memory/session_memory_eval/run_full_pipeline.sh --task singleturn

bash src/preference_memory/session_memory_eval/run_full_pipeline.sh --task multiturn
```

Use `--` with `run_full_pipeline.sh` to forward extra runner options. Example:

```bash
bash src/preference_memory/session_memory_eval/run_full_pipeline.sh \
  --task singleturn \
  --run-tag smoke_test \
  -- --limit-memory-model Qwen_Qwen3-8B --limit-cohort false_anytime --limit-session-index 1 --limit-pref-type easy
```

Dispatcher examples:

```bash
bash src/preference_memory/session_memory_eval/run_session_memory_eval.sh \
  --task multiturn \
  --prep-manifest src/preference_memory/session_memory_eval/outputs/prepared/prep_manifest.jsonl

python src/preference_memory/session_memory_eval/evaluate_session_memory_results.py \
  --task multiturn \
  --run_manifest src/preference_memory/session_memory_eval/outputs/runs/multiturn/<RUN_TAG>/run_manifest.jsonl \
  --out_csv src/preference_memory/session_memory_eval/outputs/eval/multiturn/<RUN_TAG>_detailed.csv \
  --out_summary_csv src/preference_memory/session_memory_eval/outputs/eval/multiturn/<RUN_TAG>_summary.csv
```
