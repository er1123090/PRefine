#!/usr/bin/env python3
"""Evaluate Gemma3 memory -> Gemma3 inference across memory variants."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL_SAFE = "google_gemma-3-12b-it"
V2_RUN_ID = "1229_dev6_ours_memory_v2_p0_20260520"
V2_INFERENCE_ROOT = ROOT / "outputs" / "our_memory" / V2_RUN_ID / "inference"
V1_INFERENCE_ROOT = ROOT / "outputs" / "our_memory" / "1229_dev6_autoresearch_20260515" / "inference"
VARIANT_INFERENCE_ROOT = ROOT / "outputs" / "our_memory" / "1229_dev6_memory_variants_20260517" / "inference"
OUT_ROOT = ROOT / "results" / "our_memory" / "1229_gemma3_memory_gemma3_compare_v2_p0_20260522"
PREF_LIST = ROOT / "config" / "pref_list.json"

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


def load_raw_rows(method: str, memory_mode: str, root: Path, pref_map):
    rows = []
    for turn in ["singleturn", "multiturn"]:
        for difficulty in ["easy", "medium", "hard"]:
            if memory_mode == "ours_memory":
                path = root / turn / MODEL_SAFE / "memory_api" / difficulty / MODEL_SAFE / "implicit_zs" / "result.json"
            else:
                path = root / memory_mode / turn / MODEL_SAFE / "memory_api" / difficulty / MODEL_SAFE / "implicit_zs" / "result.json"
            if not path.exists():
                raise RuntimeError(f"missing inference file for {method}: {path}")
            examples = metrics.load_json_examples(str(path))
            vals = eval_examples(examples, pref_map)
            row = {
                "method": method,
                "memory_mode": memory_mode,
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
    if len(rows) != 6:
        raise RuntimeError(f"expected 6 rows for {method}, found {len(rows)}")
    return rows


def load_v2_rows(pref_map):
    rows = []
    for turn in ["singleturn", "multiturn"]:
        for difficulty in ["easy", "medium", "hard"]:
            path = V2_INFERENCE_ROOT / turn / MODEL_SAFE / "memory_api" / difficulty / MODEL_SAFE / "implicit_zs" / "result.json"
            if not path.exists():
                raise RuntimeError(f"missing v2 inference file: {path}")
            examples = metrics.load_json_examples(str(path))
            vals = eval_examples(examples, pref_map)
            row = {
                "method": "ours_mem_v2",
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


def aggregate(rows, method: str, turn: str):
    selected = [r for r in rows if r["turn"] == turn]
    overall = metrics.prf_from_counts(
        sum(int(r["overall_tp"]) for r in selected),
        sum(int(r["overall_fp"]) for r in selected),
        sum(int(r["overall_fn"]) for r in selected),
    )
    nonpref = metrics.prf_from_counts(
        sum(int(r["nonpref_tp"]) for r in selected),
        sum(int(r["nonpref_fp"]) for r in selected),
        sum(int(r["nonpref_fn"]) for r in selected),
    )
    n_examples = sum(int(r["n_examples"]) for r in selected)
    pref_em_count = sum(int(r["pref_em_count"]) for r in selected)
    parse_failures = sum(int(r["parse_failures"]) for r in selected)
    return {
        "method": method,
        "turn": turn,
        "n_files": len(selected),
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


def main():
    pref_map = metrics.load_pref_list(str(PREF_LIST))
    method_sources = [
        ("gen_only", load_raw_rows("gen_only", "generation_only", VARIANT_INFERENCE_ROOT, pref_map)),
        ("refine_1", load_raw_rows("refine_1", "blind_refine_1", VARIANT_INFERENCE_ROOT, pref_map)),
        ("refine_2", load_raw_rows("refine_2", "blind_refine_2", VARIANT_INFERENCE_ROOT, pref_map)),
        ("refine_3", load_raw_rows("refine_3", "blind_refine_3", VARIANT_INFERENCE_ROOT, pref_map)),
        ("ours_mem_v1", load_raw_rows("ours_mem_v1", "ours_memory", V1_INFERENCE_ROOT, pref_map)),
        ("ours_mem_v2", load_v2_rows(pref_map)),
    ]

    all_rows = [row for _, rows in method_sources for row in rows]
    summary = []
    for method, rows in method_sources:
        for turn in ["singleturn", "multiturn"]:
            summary.append(aggregate(rows, method, turn))

    write_csv(OUT_ROOT / "per_file_metrics.csv", all_rows)
    write_csv(OUT_ROOT / "summary_by_turn.csv", summary)

    lines = ["# Gemma3 memory -> Gemma3 inference comparison", ""]
    for turn in ["singleturn", "multiturn"]:
        lines.append(f"## {turn}")
        if turn == "singleturn":
            lines.append("| method | precision | recall | f1 | pref_em | parse_fail |")
            lines.append("|---|---:|---:|---:|---:|---:|")
        else:
            lines.append("| method | precision | recall | f1 | pref_em | nonpref_f1 | parse_fail |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for row in [r for r in summary if r["turn"] == turn]:
            if turn == "singleturn":
                lines.append(
                    f"| {row['method']} | {row['overall_precision']:.6f} | "
                    f"{row['overall_recall']:.6f} | {row['overall_f1']:.6f} | "
                    f"{row['pref_em']:.6f} | {row['parse_failure_rate']:.6f} |"
                )
            else:
                lines.append(
                    f"| {row['method']} | {row['overall_precision']:.6f} | "
                    f"{row['overall_recall']:.6f} | {row['overall_f1']:.6f} | "
                    f"{row['pref_em']:.6f} | {row['nonpref_f1']:.6f} | "
                    f"{row['parse_failure_rate']:.6f} |"
                )
        lines.append("")
    (OUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"[OUT] {OUT_ROOT}")


if __name__ == "__main__":
    main()
