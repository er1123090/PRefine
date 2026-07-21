#!/usr/bin/env python3
"""Validate true-blind memory, inference, and evaluation artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path("/data/minseo/experiments5")
E4_ROOT = Path("/data/minseo/experiments4")
MEMORY_RUN_ID = "1229_dev6_memory_variants_true_blind_20260528"
INFERENCE_RUN_ID = "1229_dev6_true_blind_ablation_e4_ours_memory_inference_20260528"
EVAL_RUN_ID = "1229_dev6_true_blind_ablation_e4_ours_memory_eval_20260528"

MEMORY_MODES = [
    "generation_only",
    "generation_only_accum",
    "blind_refine_1",
    "blind_refine_2",
    "blind_refine_3",
]
MEMORY_MODELS = [
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "google/gemma-3-12b-it",
]
INFERENCE_MODELS = [
    "google/codegemma-7b-it",
    "google/gemma-3-12b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]
TURNS = ["singleturn", "multiturn"]
DIFFICULTIES = ["easy", "medium", "hard"]
CONTEXT = "memory_api"
PROMPT = "implicit_zs"
EXPECTED_MEMORY_ROWS = 265


def model_safe(model: str) -> str:
    return model.replace("/", "_")


def memory_path(mode: str, model: str) -> Path:
    return (
        ROOT
        / "outputs/our_memory"
        / MEMORY_RUN_ID
        / "memories"
        / mode
        / model_safe(model)
        / "_memory1.jsonl"
    )


def inference_path(mode: str, turn: str, inference_model: str, difficulty: str, memory_model: str) -> Path:
    return (
        ROOT
        / "outputs/our_memory"
        / INFERENCE_RUN_ID
        / "inference"
        / mode
        / turn
        / model_safe(inference_model)
        / CONTEXT
        / difficulty
        / model_safe(memory_model)
        / PROMPT
        / "result.json"
    )


def check_experiments4_clean(errors: list[str]) -> str:
    proc = subprocess.run(
        ["git", "status", "--short", "--", str(E4_ROOT)],
        cwd="/data/minseo",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        errors.append(f"git status for experiments4 failed: {proc.stderr.strip()}")
    elif proc.stdout.strip():
        errors.append(f"experiments4 has git changes: {proc.stdout.strip()}")
    return proc.stdout.strip()


def check_memory_files(errors: list[str]) -> dict[str, int]:
    checked = 0
    true_blind_rows = 0
    empty_api_rows = 0
    for mode in MEMORY_MODES:
        for model in MEMORY_MODELS:
            path = memory_path(mode, model)
            if not path.exists():
                errors.append(f"missing memory file: {path}")
                continue
            rows = path.read_text(encoding="utf-8").splitlines()
            if len(rows) < EXPECTED_MEMORY_ROWS:
                errors.append(f"incomplete memory file: {path} rows={len(rows)}")
            checked += 1
            for idx, line in enumerate(rows, start=1):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as exc:
                    errors.append(f"invalid memory json: {path}:{idx}: {exc}")
                    continue
                if rec.get("true_blind") is True:
                    true_blind_rows += 1
                else:
                    errors.append(f"memory row not true_blind: {path}:{idx}")
                if rec.get("final_accumulated_api_calls") == []:
                    empty_api_rows += 1
                else:
                    errors.append(f"memory row has accumulated API calls: {path}:{idx}")
    return {
        "checked_memory_files": checked,
        "expected_memory_files": len(MEMORY_MODES) * len(MEMORY_MODELS),
        "true_blind_rows": true_blind_rows,
        "empty_api_rows": empty_api_rows,
    }


def check_inference_files(errors: list[str]) -> dict[str, int]:
    checked = 0
    missing = 0
    for mode in MEMORY_MODES:
        for memory_model in MEMORY_MODELS:
            for inference_model in INFERENCE_MODELS:
                for turn in TURNS:
                    for difficulty in DIFFICULTIES:
                        path = inference_path(mode, turn, inference_model, difficulty, memory_model)
                        if path.exists() and path.stat().st_size > 0:
                            checked += 1
                        else:
                            missing += 1
                            errors.append(f"missing inference result: {path}")
    return {
        "checked_inference_files": checked,
        "missing_inference_files": missing,
        "expected_inference_files": (
            len(MEMORY_MODES)
            * len(MEMORY_MODELS)
            * len(INFERENCE_MODELS)
            * len(TURNS)
            * len(DIFFICULTIES)
        ),
    }


def check_eval_outputs(errors: list[str]) -> dict[str, object]:
    eval_dir = ROOT / "results/our_memory" / EVAL_RUN_ID
    manifest_path = eval_dir / "manifest.json"
    summary_path = eval_dir / "summary_by_method_inference_model_turn_difficulty.csv"
    per_file_path = eval_dir / "per_file_metrics.csv"
    if not manifest_path.exists():
        errors.append(f"missing eval manifest: {manifest_path}")
        return {"eval_dir": str(eval_dir), "manifest_exists": False}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("expected_files") != 360:
        errors.append(f"unexpected eval expected_files: {manifest.get('expected_files')}")
    if manifest.get("ok_files") != 360:
        errors.append(f"unexpected eval ok_files: {manifest.get('ok_files')}")
    for path in [summary_path, per_file_path]:
        if not path.exists() or path.stat().st_size == 0:
            errors.append(f"missing eval output: {path}")
    return {
        "eval_dir": str(eval_dir),
        "manifest_exists": True,
        "expected_files": manifest.get("expected_files"),
        "ok_files": manifest.get("ok_files"),
        "summary_exists": summary_path.exists(),
        "per_file_exists": per_file_path.exists(),
    }


def main() -> int:
    errors: list[str] = []
    evidence = {
        "memory_run_id": MEMORY_RUN_ID,
        "inference_run_id": INFERENCE_RUN_ID,
        "eval_run_id": EVAL_RUN_ID,
        "experiments4_git_status": check_experiments4_clean(errors),
        "memory": check_memory_files(errors),
        "inference": check_inference_files(errors),
        "evaluation": check_eval_outputs(errors),
    }
    evidence["status"] = "pass" if not errors else "fail"
    evidence["errors"] = errors[:50]
    print(json.dumps(evidence, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
