#!/usr/bin/env python3
"""Evaluate ours_memory_v3 Gemma3/DeepSeek-Distill-Qwen memory-inference combos."""

from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = os.environ.get("RUN_ID", "1229_dev6_ours_memory_v3_typed_20260523")
INFERENCE_ROOT = ROOT / "outputs" / "our_memory" / RUN_ID / "inference"
OUT_ROOT = ROOT / "results" / "our_memory" / RUN_ID
PREF_LIST = ROOT / "config" / "pref_list.json"
GEN_PER_FILE = (
    ROOT / "results" / "our_memory"
    / "1229_dev6_memory_variants_20260517_eval_20260520"
    / "per_file_metrics.csv"
)
V1_PER_FILE = (
    ROOT / "results" / "our_memory"
    / "1229_dev6_autoresearch_20260515_eval_4opensource_20260520"
    / "per_file_metrics.csv"
)

MEMORY_MODELS = [
    "google/gemma-3-12b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]
INFERENCE_MODELS = [
    "google/gemma-3-12b-it",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
]
TURNS = ["singleturn", "multiturn"]
DIFFICULTIES = ["easy", "medium", "hard"]
CONTEXT = "memory_api"
PROMPT = "implicit_zs"

if os.environ.get("ONLY_MEMORY_MODELS"):
    MEMORY_MODELS = os.environ["ONLY_MEMORY_MODELS"].split()
if os.environ.get("ONLY_MODELS"):
    INFERENCE_MODELS = os.environ["ONLY_MODELS"].split()
if os.environ.get("ONLY_TURNS"):
    TURNS = os.environ["ONLY_TURNS"].split()
if os.environ.get("ONLY_PREFS"):
    DIFFICULTIES = os.environ["ONLY_PREFS"].split()

sys.path.insert(0, str(ROOT))
import src.evaluation.metrics as metrics  # noqa: E402

metrics._normalize_date_time_value = lru_cache(maxsize=None)(metrics._normalize_date_time_value)


def model_safe(model: str) -> str:
    return model.replace("/", "_")


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


def output_path(turn: str, inference_model: str, difficulty: str, memory_model: str) -> Path:
    return (
        INFERENCE_ROOT / turn / model_safe(inference_model) / CONTEXT / difficulty
        / model_safe(memory_model) / PROMPT / "result.json"
    )


def load_v3_rows(pref_map):
    rows = []
    missing = []
    for memory_model in MEMORY_MODELS:
        for inference_model in INFERENCE_MODELS:
            for turn in TURNS:
                for difficulty in DIFFICULTIES:
                    path = output_path(turn, inference_model, difficulty, memory_model)
                    if not path.exists():
                        missing.append(str(path))
                        continue
                    examples = metrics.load_json_examples(str(path))
                    row = {
                        "method": "ours_memory_v3",
                        "memory_mode": "typed_slots_v3",
                        "turn": turn,
                        "inference_model": model_safe(inference_model),
                        "context": CONTEXT,
                        "difficulty": difficulty,
                        "memory_model": model_safe(memory_model),
                        "prompt_type": PROMPT,
                        "file": str(path),
                        "status": "ok",
                        "error": "",
                    }
                    row.update(eval_examples(examples, pref_map))
                    rows.append(row)
    if missing:
        joined = "\n".join(missing[:20])
        raise RuntimeError(f"missing {len(missing)} v3 inference files; first paths:\n{joined}")
    return rows


def aggregate(rows, keys):
    selected = rows
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
    out = {key: selected[0][key] for key in keys}
    out.update({
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
    })
    return out


def load_baseline_rows(path: Path, method: str, memory_mode: str | None):
    if not path.exists():
        return []
    out = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if memory_mode is not None and row.get("memory_mode") != memory_mode:
                continue
            if row.get("context") != CONTEXT or row.get("prompt_type") != PROMPT:
                continue
            if row.get("turn") not in TURNS or row.get("difficulty") not in DIFFICULTIES:
                continue
            if row.get("status") != "ok":
                continue
            row = dict(row)
            row["method"] = method
            out.append(row)
    return out


def baseline_lookup(rows):
    lookup = {}
    for row in rows:
        key = (
            row.get("method"),
            row.get("memory_model"),
            row.get("inference_model"),
            row.get("turn"),
        )
        lookup.setdefault(key, []).append(row)
    return {
        key: aggregate(group, ["method", "memory_model", "inference_model", "turn"])
        for key, group in lookup.items()
        if len(group) == len(DIFFICULTIES)
    }


def main():
    pref_map = metrics.load_pref_list(str(PREF_LIST))
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    per_file = load_v3_rows(pref_map)
    write_csv(OUT_ROOT / "per_file_metrics.csv", per_file)

    grouped = defaultdict(list)
    for row in per_file:
        grouped[(row["memory_model"], row["inference_model"], row["turn"])].append(row)
    summary = [
        aggregate(rows, ["memory_model", "inference_model", "turn"])
        for rows in grouped.values()
    ]
    summary.sort(key=lambda r: (r["inference_model"], r["memory_model"], r["turn"]))

    baselines = baseline_lookup(
        load_baseline_rows(GEN_PER_FILE, "generation_only", "generation_only")
        + load_baseline_rows(V1_PER_FILE, "ours_memory_v1", "ours_memory")
    )
    for row in summary:
        for method in ["generation_only", "ours_memory_v1"]:
            key = (method, row["memory_model"], row["inference_model"], row["turn"])
            base = baselines.get(key)
            row[f"f1_minus_{method}"] = (
                float(row["overall_f1"]) - float(base["overall_f1"]) if base else ""
            )

    write_csv(OUT_ROOT / "summary_by_turn.csv", summary)

    overall_groups = defaultdict(list)
    for row in per_file:
        overall_groups[(row["memory_model"], row["inference_model"])].append(row)
    overall = [
        aggregate(rows, ["memory_model", "inference_model"])
        for rows in overall_groups.values()
    ]
    overall.sort(key=lambda r: (r["inference_model"], r["memory_model"]))
    write_csv(OUT_ROOT / "comparison_summary.csv", overall)

    lines = ["# ours_memory_v3 typed-slot evaluation", ""]
    lines.append("## By Turn")
    lines.append("| memory | inference | turn | precision | recall | f1 | pref_em | nonpref_f1 | parse_fail | gap_gen | gap_v1 |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary:
        gap_gen = row.get("f1_minus_generation_only")
        gap_v1 = row.get("f1_minus_ours_memory_v1")
        gap_gen_str = f"{float(gap_gen):+.6f}" if gap_gen != "" else ""
        gap_v1_str = f"{float(gap_v1):+.6f}" if gap_v1 != "" else ""
        lines.append(
            f"| {row['memory_model']} | {row['inference_model']} | {row['turn']} | "
            f"{float(row['overall_precision']):.6f} | {float(row['overall_recall']):.6f} | "
            f"{float(row['overall_f1']):.6f} | {float(row['pref_em']):.6f} | "
            f"{float(row['nonpref_f1']):.6f} | {int(row['parse_failures'])} | "
            f"{gap_gen_str} | {gap_v1_str} |"
        )
    lines.append("")
    lines.append("## Overall")
    lines.append("| memory | inference | precision | recall | f1 | pref_em | nonpref_f1 | parse_fail |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for row in overall:
        lines.append(
            f"| {row['memory_model']} | {row['inference_model']} | "
            f"{float(row['overall_precision']):.6f} | {float(row['overall_recall']):.6f} | "
            f"{float(row['overall_f1']):.6f} | {float(row['pref_em']):.6f} | "
            f"{float(row['nonpref_f1']):.6f} | {int(row['parse_failures'])} |"
        )
    (OUT_ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()
