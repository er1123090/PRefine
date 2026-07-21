#!/usr/bin/env python3
"""Evaluate ours_memory_v1 MEMORY3 seed-variance inference outputs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.evaluation import metrics  # noqa: E402

if hasattr(metrics, "_normalize_date_time_value"):
    metrics._normalize_date_time_value = lru_cache(maxsize=200_000)(metrics._normalize_date_time_value)

CONTEXT = "memory_api"
PROMPT = "implicit_zs"
PRIMARY_METRICS = ["overall_f1", "pref_em", "nonpref_f1", "parse_failure_rate"]


def model_safe(model: str) -> str:
    return model.replace("/", "_").replace("[", "__").replace("]", "").replace(":", "_")


def read_split_env(value: str | None, default: List[str]) -> List[str]:
    return value.split() if value else default


def write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def result_path(run_root: Path, repeat: str, turn: str, inference_model: str, difficulty: str, memory_safe: str) -> Path:
    return (
        run_root
        / "repeats"
        / repeat
        / "inference"
        / turn
        / model_safe(inference_model)
        / CONTEXT
        / difficulty
        / memory_safe
        / PROMPT
        / "result.json"
    )


def slot_counts(example: Dict[str, Any], pref_map: Dict[str, set]) -> Dict[str, Any]:
    gt = metrics.build_gt_allowed_map(example.get("reference_ground_truth"))
    pred = metrics.build_pred_map(example.get("llm_output"))
    tp, fp, fn = metrics.counts_slot_and_value_or(gt, pred)
    gt_nonpref = metrics.filter_map_by_pref(gt, pref_map, want_pref=False)
    pred_nonpref = metrics.filter_map_by_pref(pred, pref_map, want_pref=False)
    ntp, nfp, nfn = metrics.counts_slot_and_value_or(gt_nonpref, pred_nonpref)
    gt_pref = metrics.filter_map_by_pref(gt, pref_map, want_pref=True)
    pred_pref = metrics.filter_map_by_pref(pred, pref_map, want_pref=True)
    _, pref_fp, pref_fn = metrics.counts_slot_and_value_or(gt_pref, pred_pref)
    parse_failed, parse_reason = metrics.is_parsing_failed(example)
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "nonpref_tp": ntp,
        "nonpref_fp": nfp,
        "nonpref_fn": nfn,
        "pref_em": 1 if (pref_fp == 0 and pref_fn == 0) else 0,
        "parse_failed": int(parse_failed),
        "parse_reason": parse_reason,
    }


def evaluate_count_rows(counts_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    overall = metrics.prf_from_counts(
        sum(int(row["tp"]) for row in counts_rows),
        sum(int(row["fp"]) for row in counts_rows),
        sum(int(row["fn"]) for row in counts_rows),
    )
    nonpref = metrics.prf_from_counts(
        sum(int(row["nonpref_tp"]) for row in counts_rows),
        sum(int(row["nonpref_fp"]) for row in counts_rows),
        sum(int(row["nonpref_fn"]) for row in counts_rows),
    )
    n_examples = len(counts_rows)
    parse_failures = sum(int(row["parse_failed"]) for row in counts_rows)
    pref_em_count = sum(int(row["pref_em"]) for row in counts_rows)
    return {
        "n_examples": n_examples,
        "parse_failures": parse_failures,
        "parse_failure_rate": parse_failures / n_examples if n_examples else 0.0,
        "overall_tp": overall.tp,
        "overall_fp": overall.fp,
        "overall_fn": overall.fn,
        "overall_precision": overall.precision,
        "overall_recall": overall.recall,
        "overall_f1": overall.f1,
        "pref_em": pref_em_count / n_examples if n_examples else 0.0,
        "pref_em_count": pref_em_count,
        "nonpref_tp": nonpref.tp,
        "nonpref_fp": nonpref.fp,
        "nonpref_fn": nonpref.fn,
        "nonpref_precision": nonpref.precision,
        "nonpref_recall": nonpref.recall,
        "nonpref_f1": nonpref.f1,
    }


def aggregate(rows: List[Dict[str, Any]], keys: Tuple[str, ...]) -> Dict[str, Any]:
    overall = metrics.prf_from_counts(
        sum(int(row["overall_tp"]) for row in rows),
        sum(int(row["overall_fp"]) for row in rows),
        sum(int(row["overall_fn"]) for row in rows),
    )
    nonpref = metrics.prf_from_counts(
        sum(int(row["nonpref_tp"]) for row in rows),
        sum(int(row["nonpref_fp"]) for row in rows),
        sum(int(row["nonpref_fn"]) for row in rows),
    )
    n_examples = sum(int(row["n_examples"]) for row in rows)
    pref_em_count = sum(int(row["pref_em_count"]) for row in rows)
    parse_failures = sum(int(row["parse_failures"]) for row in rows)
    out = {key: rows[0][key] for key in keys}
    out.update(
        {
            "n_files": len(rows),
            "n_examples": n_examples,
            "parse_failures": parse_failures,
            "parse_failure_rate": parse_failures / n_examples if n_examples else 0.0,
            "overall_precision": overall.precision,
            "overall_recall": overall.recall,
            "overall_f1": overall.f1,
            "pref_em": pref_em_count / n_examples if n_examples else 0.0,
            "nonpref_precision": nonpref.precision,
            "nonpref_recall": nonpref.recall,
            "nonpref_f1": nonpref.f1,
        }
    )
    return out


def write_summary(path: Path, rows: List[Dict[str, Any]], keys: Tuple[str, ...]) -> None:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summary = [aggregate(group_rows, keys) for group_rows in grouped.values()]
    summary.sort(key=lambda row: tuple(row[key] for key in keys))
    write_csv(path, summary)


def write_seed_outputs(out_root: Path, per_file: List[Dict[str, Any]], per_example: List[Dict[str, Any]], coverage: List[Dict[str, Any]]) -> None:
    write_summary(out_root / "summary_by_seed.csv", per_file, ("repeat_id",))
    write_summary(out_root / "summary_by_seed_model.csv", per_file, ("repeat_id", "inference_model"))
    write_summary(out_root / "summary_by_seed_memory.csv", per_file, ("repeat_id", "memory_safe_name"))
    write_summary(out_root / "summary_by_seed_turn_difficulty.csv", per_file, ("repeat_id", "turn", "difficulty"))

    repeats = sorted({row["repeat_id"] for row in per_file})
    for repeat in repeats:
        seed_root = out_root / "by_seed" / repeat
        seed_files = [row for row in per_file if row["repeat_id"] == repeat]
        seed_examples = [row for row in per_example if row["repeat_id"] == repeat]
        seed_coverage = [row for row in coverage if row["repeat_id"] == repeat]
        write_csv(seed_root / "per_file_metrics.csv", seed_files)
        write_csv(seed_root / "per_example_correctness.csv", seed_examples)
        write_csv(seed_root / "coverage.csv", seed_coverage)
        write_summary(seed_root / "summary_overall.csv", seed_files, ("repeat_id",))
        write_summary(seed_root / "summary_by_model.csv", seed_files, ("inference_model",))
        write_summary(seed_root / "summary_by_memory.csv", seed_files, ("memory_safe_name",))
        write_summary(seed_root / "summary_by_model_memory.csv", seed_files, ("inference_model", "memory_safe_name"))
        write_summary(seed_root / "summary_by_model_turn_difficulty.csv", seed_files, ("inference_model", "memory_safe_name", "turn", "difficulty"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run-id", required=True)
    parser.add_argument("--allow-missing", action="store_true")
    args = parser.parse_args()

    run_root = ROOT / "outputs" / "our_memory" / args.parent_run_id
    out_root = ROOT / "results" / "our_memory" / args.parent_run_id
    metadata_path = run_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}

    inference_models = read_split_env(
        None,
        metadata.get(
            "inference_models",
            [
                "google/gemma-3-12b-it",
                "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "google/codegemma-7b-it",
            ],
        ),
    )
    memory_sources = read_split_env(
        None,
        metadata.get(
            "memory_sources",
            [
                "google_gemma-3-12b-it",
                "gpt-4o-mini",
                "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B",
                "deepseek-ai_DeepSeek-R1-Distill-Llama-8B",
            ],
        ),
    )
    repeats = read_split_env(None, metadata.get("repeats", ["seed_001", "seed_002", "seed_003"]))
    turns = read_split_env(None, metadata.get("turns", ["singleturn", "multiturn"]))
    difficulties = read_split_env(None, metadata.get("difficulties", ["easy", "medium", "hard"]))
    pref_map = metrics.load_pref_list(str(ROOT / "config" / "pref_list.json"))

    per_file: List[Dict[str, Any]] = []
    per_example: List[Dict[str, Any]] = []
    coverage: List[Dict[str, Any]] = []
    missing: List[str] = []

    for repeat in repeats:
        for inference_model in inference_models:
            for memory_safe in memory_sources:
                for turn in turns:
                    for difficulty in difficulties:
                        path = result_path(run_root, repeat, turn, inference_model, difficulty, memory_safe)
                        base = {
                            "repeat_id": repeat,
                            "turn": turn,
                            "difficulty": difficulty,
                            "inference_model": model_safe(inference_model),
                            "canonical_inference_model_id": inference_model,
                            "memory_method": "ours_memory_v1",
                            "memory_variant": "verified_refine",
                            "memory_safe_name": memory_safe,
                            "context": CONTEXT,
                            "prompt_type": PROMPT,
                            "file": str(path),
                        }
                        if not path.exists():
                            missing.append(str(path))
                            coverage.append({**base, "status": "missing"})
                            continue
                        examples = metrics.load_json_examples(str(path))
                        file_counts = [slot_counts(ex, pref_map) for ex in examples]
                        row = {**base, "status": "ok"}
                        row.update(evaluate_count_rows(file_counts))
                        per_file.append(row)
                        coverage.append({**base, "status": "ok", "n_examples": len(examples)})
                        for ex, counts in zip(examples, file_counts):
                            example_id_sub = ex.get("example_id_sub", "")
                            per_example.append(
                                {
                                    **base,
                                    "example_id": ex.get("example_id", ""),
                                    "example_id_sub": example_id_sub,
                                    "pair_key": f"{turn}|{difficulty}|{repeat}|{memory_safe}|{example_id_sub}",
                                    **counts,
                                }
                            )

    if missing and not args.allow_missing:
        preview = "\n".join(missing[:20])
        raise SystemExit(f"missing {len(missing)} expected result files; first paths:\n{preview}")

    write_csv(out_root / "per_file_metrics.csv", per_file)
    write_csv(out_root / "per_example_correctness.csv", per_example)
    write_csv(out_root / "coverage.csv", coverage)

    write_summary(out_root / "summary_by_model_turn_difficulty.csv", per_file, ("inference_model", "memory_safe_name", "turn", "difficulty"))
    write_seed_outputs(out_root, per_file, per_example, coverage)

    print(f"Wrote {out_root}")


if __name__ == "__main__":
    main()
