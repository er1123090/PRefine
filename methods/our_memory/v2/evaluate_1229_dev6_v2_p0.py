#!/usr/bin/env python3
"""Evaluate ours_memory_v2 P0 outputs and compare with v1/generation_only."""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUN_ID = "1229_dev6_ours_memory_v2_p0_20260520"
EXPECTED_FILES = 96
INFERENCE_ROOT = ROOT / "outputs" / "our_memory" / RUN_ID / "inference"
EVAL_ROOT = ROOT / "results" / "our_memory" / f"{RUN_ID}_eval_20260520"
COMPARE_ROOT = ROOT / "results" / "our_memory" / "1229_compare_generation_v1_v2_p0_20260520"
PREF_LIST = ROOT / "config" / "pref_list.json"
V1_SUMMARY = ROOT / "results" / "our_memory" / "1229_dev6_autoresearch_20260515_eval_4opensource_20260520" / "summary_overall.csv"
GEN_SUMMARY = ROOT / "results" / "our_memory" / "1229_dev6_memory_variants_20260517_eval_20260520" / "summary_by_memory_mode.csv"

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
    return overall, em_count, len(examples), nonpref, parse_failures


def aggregate(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        if row["status"] == "ok":
            groups[tuple(row[k] for k in keys)].append(row)

    out_rows = []
    for key_vals, group_rows in sorted(groups.items()):
        overall = metrics.prf_from_counts(
            sum(int(r["overall_tp"]) for r in group_rows),
            sum(int(r["overall_fp"]) for r in group_rows),
            sum(int(r["overall_fn"]) for r in group_rows),
        )
        nonpref = metrics.prf_from_counts(
            sum(int(r["nonpref_tp"]) for r in group_rows),
            sum(int(r["nonpref_fp"]) for r in group_rows),
            sum(int(r["nonpref_fn"]) for r in group_rows),
        )
        n_examples = sum(int(r["n_examples"]) for r in group_rows)
        em_count = sum(int(r["pref_em_count"]) for r in group_rows)
        parse_failures = sum(int(r["parse_failures"]) for r in group_rows)
        row = {k: v for k, v in zip(keys, key_vals)}
        row.update({
            "n_files": len(group_rows),
            "n_examples": n_examples,
            "parse_failures": parse_failures,
            "parse_failure_rate": parse_failures / n_examples if n_examples else 0.0,
            "overall_tp": overall.tp,
            "overall_fp": overall.fp,
            "overall_fn": overall.fn,
            "overall_precision": overall.precision,
            "overall_recall": overall.recall,
            "overall_f1": overall.f1,
            "pref_em": em_count / n_examples if n_examples else 0.0,
            "pref_em_count": em_count,
            "nonpref_tp": nonpref.tp,
            "nonpref_fp": nonpref.fp,
            "nonpref_fn": nonpref.fn,
            "nonpref_precision": nonpref.precision,
            "nonpref_recall": nonpref.recall,
            "nonpref_f1": nonpref.f1,
        })
        out_rows.append(row)
    return out_rows


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


def load_summary_row(path: Path, method: str, filter_key: str | None = None, filter_value: str | None = None):
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if filter_key is None or row.get(filter_key) == filter_value:
                row = dict(row)
                row["method"] = method
                return row
    raise RuntimeError(f"Could not load summary row from {path}")


def main() -> None:
    pref_map = metrics.load_pref_list(str(PREF_LIST))
    files = sorted(INFERENCE_ROOT.rglob("result.json"))
    if len(files) != EXPECTED_FILES:
        raise SystemExit(f"expected {EXPECTED_FILES} result files, found {len(files)} under {INFERENCE_ROOT}")

    rows = []
    errors = []
    for path in files:
        rel = path.relative_to(INFERENCE_ROOT).parts
        if len(rel) != 7:
            errors.append((str(path), f"unexpected relpath parts: {rel}"))
            continue
        turn, inference_model, context, difficulty, memory_model, prompt_type, _ = rel
        try:
            examples = metrics.load_json_examples(str(path))
            overall, em_count, n_examples, nonpref, parse_failures = eval_examples(examples, pref_map)
            status = "ok"
            error = ""
        except Exception as exc:  # pragma: no cover - diagnostic path
            overall = metrics.PRF(0.0, 0.0, 0.0, 0, 0, 0)
            nonpref = metrics.PRF(0.0, 0.0, 0.0, 0, 0, 0)
            em_count = n_examples = parse_failures = 0
            status = "error"
            error = repr(exc)
            errors.append((str(path), error))

        rows.append({
            "memory_mode": "ours_memory_v2_p0",
            "turn": turn,
            "inference_model": inference_model,
            "context": context,
            "difficulty": difficulty,
            "memory_model": memory_model,
            "prompt_type": prompt_type,
            "file": str(path),
            "rel_path": str(path.relative_to(INFERENCE_ROOT)),
            "status": status,
            "error": error,
            "n_examples": n_examples,
            "parse_failures": parse_failures,
            "overall_tp": overall.tp,
            "overall_fp": overall.fp,
            "overall_fn": overall.fn,
            "overall_precision": overall.precision,
            "overall_recall": overall.recall,
            "overall_f1": overall.f1,
            "pref_em": em_count / n_examples if n_examples else 0.0,
            "pref_em_count": em_count,
            "nonpref_tp": nonpref.tp,
            "nonpref_fp": nonpref.fp,
            "nonpref_fn": nonpref.fn,
            "nonpref_precision": nonpref.precision,
            "nonpref_recall": nonpref.recall,
            "nonpref_f1": nonpref.f1,
        })

    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    write_csv(EVAL_ROOT / "per_file_metrics.csv", rows)

    summaries = {
        "summary_overall.csv": [],
        "summary_by_turn.csv": ["turn"],
        "summary_by_difficulty.csv": ["difficulty"],
        "summary_by_inference_model.csv": ["inference_model"],
        "summary_by_memory_model.csv": ["memory_model"],
        "summary_by_turn_difficulty.csv": ["turn", "difficulty"],
    }
    for name, keys in summaries.items():
        write_csv(EVAL_ROOT / name, aggregate(rows, keys))

    if errors:
        (EVAL_ROOT / "errors.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")

    overall = aggregate(rows, [])[0]
    report_lines = [
        "# ours_memory_v2 P0 evaluation",
        "",
        f"- inference root: `{INFERENCE_ROOT}`",
        f"- result files evaluated: `{len(rows)}` / expected `{EXPECTED_FILES}`",
        f"- status ok/error: `{sum(1 for r in rows if r['status'] == 'ok')}` / `{sum(1 for r in rows if r['status'] != 'ok')}`",
        f"- overall: F1 `{overall['overall_f1']:.6f}`, P `{overall['overall_precision']:.6f}`, R `{overall['overall_recall']:.6f}`, pref_EM `{overall['pref_em']:.6f}`, nonpref_F1 `{overall['nonpref_f1']:.6f}`, parse_fail `{overall['parse_failures']}/{overall['n_examples']}`",
    ]
    (EVAL_ROOT / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    COMPARE_ROOT.mkdir(parents=True, exist_ok=True)
    generation = load_summary_row(GEN_SUMMARY, "generation_only", "memory_mode", "generation_only")
    v1 = load_summary_row(V1_SUMMARY, "ours_memory_v1")
    v2 = dict(overall)
    v2["method"] = "ours_memory_v2_p0"
    compare_rows = [generation, v1, v2]
    base_gen = float(generation["overall_f1"])
    base_v1 = float(v1["overall_f1"])
    for row in compare_rows:
        row["overall_f1_minus_generation_only"] = float(row["overall_f1"]) - base_gen
        row["overall_f1_minus_v1"] = float(row["overall_f1"]) - base_v1
    write_csv(COMPARE_ROOT / "comparison_summary.csv", compare_rows)

    compare_lines = [
        "# generation_only vs ours_memory_v1 vs ours_memory_v2_p0",
        "",
        "| method | overall_F1 | gap_vs_generation | gap_vs_v1 | pref_EM | nonpref_F1 | parse_fail_rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in compare_rows:
        compare_lines.append(
            f"| {row['method']} | {float(row['overall_f1']):.6f} | "
            f"{float(row['overall_f1_minus_generation_only']):+.6f} | "
            f"{float(row['overall_f1_minus_v1']):+.6f} | "
            f"{float(row['pref_em']):.6f} | {float(row['nonpref_f1']):.6f} | "
            f"{float(row['parse_failure_rate']):.6f} |"
        )
    (COMPARE_ROOT / "README.md").write_text("\n".join(compare_lines) + "\n", encoding="utf-8")

    print(f"[DONE] rows={len(rows)} errors={len(errors)}")
    print(f"[EVAL] {EVAL_ROOT}")
    print(f"[COMPARE] {COMPARE_ROOT / 'comparison_summary.csv'}")
    print(
        "[OVERALL] F1={overall_f1:.6f} P={overall_precision:.6f} R={overall_recall:.6f} "
        "pref_EM={pref_em:.6f} nonpref_F1={nonpref_f1:.6f} parse_failures={parse_failures}/{n_examples}".format(**overall)
    )


if __name__ == "__main__":
    main()
