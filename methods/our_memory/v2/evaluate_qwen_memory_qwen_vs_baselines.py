#!/usr/bin/env python3
"""Evaluate Qwen memory -> Qwen inference and compare with Qwen/Qwen baselines."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "1229_dev6_ours_memory_v2_p0_20260520"
MODEL_SAFE = "Qwen_Qwen3-8B"
INFERENCE_ROOT = ROOT / "outputs" / "our_memory" / RUN_ID / "inference"
OUT_ROOT = ROOT / "results" / "our_memory" / "1229_qwen_memory_qwen_compare_v2_p0_20260520"
PREF_LIST = ROOT / "config" / "pref_list.json"
V1_PER_FILE = ROOT / "results" / "our_memory" / "1229_dev6_autoresearch_20260515_eval_4opensource_20260520" / "per_file_metrics.csv"
GEN_PER_FILE = ROOT / "results" / "our_memory" / "1229_dev6_memory_variants_20260517_eval_20260520" / "per_file_metrics.csv"

sys.path.insert(0, str(ROOT))
import src.evaluation.metrics as metrics  # noqa: E402

metrics._normalize_date_time_value = lru_cache(maxsize=None)(metrics._normalize_date_time_value)


@lru_cache(maxsize=500000)
def _slotvals_for_call(call: str):
    return tuple(metrics.parse_call_to_slotvals(call))


@lru_cache(maxsize=500000)
def _calls_from_string(text: str):
    return tuple(metrics.extract_calls(text))


def extract_calls_cached(value):
    if value is None:
        return []
    if isinstance(value, str):
        return list(_calls_from_string(value))
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, str) and item.strip():
                out.extend(_calls_from_string(item))
        return out
    return metrics.extract_calls(value)


def build_map_cached(field):
    out = defaultdict(set)
    for call in extract_calls_cached(field):
        for domain, slot, val in _slotvals_for_call(call):
            out[(domain, slot)].add(val)
    return dict(out)


def filter_pref(slotval_map, pref_map, want_pref):
    return {
        key: vals
        for key, vals in slotval_map.items()
        if metrics.is_pref_slot(key[0], key[1], pref_map) == want_pref
    }


def eval_examples(examples, pref_map):
    tp = fp = fn = 0
    ntp = nfp = nfn = 0
    em_count = 0
    parse_failures = 0
    for ex in examples:
        gt = build_map_cached(ex.get("reference_ground_truth"))
        pred = build_map_cached(ex.get("llm_output"))
        ctp, cfp, cfn = metrics.counts_slot_and_value_or(gt, pred)
        tp += ctp
        fp += cfp
        fn += cfn

        gt_pref = filter_pref(gt, pref_map, True)
        pred_pref = filter_pref(pred, pref_map, True)
        _, pfp, pfn = metrics.counts_slot_and_value_or(gt_pref, pred_pref)
        if pfp == 0 and pfn == 0:
            em_count += 1

        gt_nonpref = filter_pref(gt, pref_map, False)
        pred_nonpref = filter_pref(pred, pref_map, False)
        cntp, cnfp, cnfn = metrics.counts_slot_and_value_or(gt_nonpref, pred_nonpref)
        ntp += cntp
        nfp += cnfp
        nfn += cnfn

        if not extract_calls_cached(ex.get("llm_output")):
            parse_failures += 1

    overall = metrics.prf_from_counts(tp, fp, fn)
    nonpref = metrics.prf_from_counts(ntp, nfp, nfn)
    return {
        "n_examples": len(examples),
        "parse_failures": parse_failures,
        "overall_tp": overall.tp,
        "overall_fp": overall.fp,
        "overall_fn": overall.fn,
        "overall_precision": overall.precision,
        "overall_recall": overall.recall,
        "overall_f1": overall.f1,
        "pref_em": em_count / len(examples) if examples else 0.0,
        "pref_em_count": em_count,
        "nonpref_tp": nonpref.tp,
        "nonpref_fp": nonpref.fp,
        "nonpref_fn": nonpref.fn,
        "nonpref_precision": nonpref.precision,
        "nonpref_recall": nonpref.recall,
        "nonpref_f1": nonpref.f1,
    }


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def aggregate(rows, method):
    overall = metrics.prf_from_counts(
        sum(int(r["overall_tp"]) for r in rows),
        sum(int(r["overall_fp"]) for r in rows),
        sum(int(r["overall_fn"]) for r in rows),
    )
    nonpref = metrics.prf_from_counts(
        sum(int(r["nonpref_tp"]) for r in rows),
        sum(int(r["nonpref_fp"]) for r in rows),
        sum(int(r["nonpref_fn"]) for r in rows),
    )
    n_examples = sum(int(r["n_examples"]) for r in rows)
    pref_em_count = sum(int(r["pref_em_count"]) for r in rows)
    parse_failures = sum(int(r["parse_failures"]) for r in rows)
    return {
        "method": method,
        "n_files": len(rows),
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


def load_baseline_rows(path: Path, method: str, memory_mode: str | None = None):
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if memory_mode is not None and row.get("memory_mode") != memory_mode:
                continue
            if row.get("inference_model") != MODEL_SAFE:
                continue
            if row.get("memory_model") != MODEL_SAFE:
                continue
            if row.get("context") != "memory_api":
                continue
            if row.get("prompt_type") != "implicit_zs":
                continue
            if row.get("turn") not in {"singleturn", "multiturn"}:
                continue
            if row.get("difficulty") not in {"easy", "medium", "hard"}:
                continue
            if row.get("status") != "ok":
                continue
            row = dict(row)
            row["method"] = method
            rows.append(row)
    if len(rows) != 6:
        raise RuntimeError(f"expected 6 baseline rows for {method}, found {len(rows)} in {path}")
    return rows


def load_v2_rows():
    pref_map = metrics.load_pref_list(str(PREF_LIST))
    rows = []
    for turn in ["singleturn", "multiturn"]:
        for difficulty in ["easy", "medium", "hard"]:
            path = INFERENCE_ROOT / turn / MODEL_SAFE / "memory_api" / difficulty / MODEL_SAFE / "implicit_zs" / "result.json"
            if not path.exists():
                raise RuntimeError(f"missing v2 inference file: {path}")
            examples = metrics.load_json_examples(str(path))
            vals = eval_examples(examples, pref_map)
            row = {
                "method": "ours_memory_v2_p0",
                "memory_mode": "ours_memory_v2_p0",
                "turn": turn,
                "inference_model": MODEL_SAFE,
                "context": "memory_api",
                "difficulty": difficulty,
                "memory_model": MODEL_SAFE,
                "prompt_type": "implicit_zs",
                "file": str(path),
                "status": "ok",
                "error": "",
            }
            row.update(vals)
            rows.append(row)
    return rows


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    gen_rows = load_baseline_rows(GEN_PER_FILE, "generation_only", "generation_only")
    v1_rows = load_baseline_rows(V1_PER_FILE, "ours_memory_v1")
    v2_rows = load_v2_rows()

    write_csv(OUT_ROOT / "per_file_metrics.csv", gen_rows + v1_rows + v2_rows)
    summary = [
        aggregate(gen_rows, "generation_only"),
        aggregate(v1_rows, "ours_memory_v1"),
        aggregate(v2_rows, "ours_memory_v2_p0"),
    ]
    base_gen = float(summary[0]["overall_f1"])
    base_v1 = float(summary[1]["overall_f1"])
    for row in summary:
        row["overall_f1_minus_generation_only"] = float(row["overall_f1"]) - base_gen
        row["overall_f1_minus_v1"] = float(row["overall_f1"]) - base_v1
    write_csv(OUT_ROOT / "comparison_summary.csv", summary)

    lines = [
        "# Qwen memory -> Qwen inference comparison",
        "",
        "| method | overall_F1 | gap_vs_generation | gap_vs_v1 | pref_EM | nonpref_F1 | parse_fail_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['method']} | {float(row['overall_f1']):.6f} | "
            f"{float(row['overall_f1_minus_generation_only']):+.6f} | "
            f"{float(row['overall_f1_minus_v1']):+.6f} | "
            f"{float(row['pref_em']):.6f} | "
            f"{float(row['nonpref_f1']):.6f} | "
            f"{float(row['parse_failure_rate']):.6f} |"
        )
    (OUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"[OUT] {OUT_ROOT}")


if __name__ == "__main__":
    main()
