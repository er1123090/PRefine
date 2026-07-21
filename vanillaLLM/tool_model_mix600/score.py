from __future__ import annotations

import argparse
import csv
import io
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from core import (
    ContractError,
    DIFFICULTIES,
    MODES,
    SEED,
    canonical_calls,
    canonical_json,
    load_json,
    read_jsonl,
    sha256_bytes,
    write_json_atomic,
)


Key = tuple[str, str]
ValueMap = dict[Key, set[str]]
STATUSES = ("success", "model_no_tool", "model_parse_failure", "infra_failure")
BOOTSTRAP_DRAWS = 10_000
CI_INDICES = (249, 9749)


def calls_to_map(calls: Iterable[Mapping[str, Any]]) -> ValueMap:
    result: ValueMap = defaultdict(set)
    for call in calls:
        name = str(call["name"])
        for slot, value in (call.get("arguments") or {}).items():
            result[(name, str(slot))].add(canonical_json(value))
    return dict(result)


def project(values: ValueMap, pref_list: Mapping[str, Sequence[str]], preference: bool) -> ValueMap:
    return {
        key: set(items)
        for key, items in values.items()
        if (key[1] in set(pref_list.get(key[0], []))) is preference
    }


def value_or_counts(gt: ValueMap, pred: ValueMap) -> tuple[int, int, int]:
    tp = sum(bool(pred.get(key, set()) & allowed) for key, allowed in gt.items())
    fn = sum(not bool(pred.get(key, set()) & allowed) for key, allowed in gt.items())
    fp = sum(key not in gt or not bool(values & gt[key]) for key, values in pred.items())
    return int(tp), int(fp), int(fn)


def prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def preference_exact(gt: ValueMap, pred: ValueMap) -> int:
    return int(
        set(gt) == set(pred)
        and all(bool(pred[key] & gt[key]) and pred[key] <= gt[key] for key in gt)
    )


def evaluate_case(case: Mapping[str, Any], record: Mapping[str, Any], pref_list: Mapping[str, Sequence[str]]) -> dict[str, Any]:
    pred_calls = canonical_calls(record.get("normalized_calls") or [])
    pred_all = calls_to_map(pred_calls)
    gt_pref = calls_to_map(case["reference_ground_truth_preference"])
    gt_full = calls_to_map(case["reference_ground_truth_full"])
    pred_pref = project(pred_all, pref_list, True)
    pred_nonpref = project(pred_all, pref_list, False)
    gt_nonpref = project(gt_full, pref_list, False)
    p_tp, p_fp, p_fn = value_or_counts(gt_pref, pred_pref)
    o_tp, o_fp, o_fn = value_or_counts(gt_full, pred_all)
    n_tp, n_fp, n_fn = value_or_counts(gt_nonpref, pred_nonpref)
    allowed_full = {canonical_json(call) for call in canonical_calls(case["reference_ground_truth_full"])}
    full_em = int(len(pred_calls) == 1 and canonical_json(pred_calls[0]) in allowed_full)
    function_acc = int({call["name"] for call in pred_calls} == {case["target_domain"]})
    return {
        "case_id": case["case_id"],
        "pair_id": case["pair_id"],
        "source_example_id": case["source_example_id"],
        "mode": case["mode"],
        "difficulty": case["difficulty"],
        "target_domain": case["target_domain"],
        "preference_group": case.get("preference_group"),
        "status": record["status"],
        "primary_tp": p_tp,
        "primary_fp": p_fp,
        "primary_fn": p_fn,
        "overall_tp": o_tp,
        "overall_fp": o_fp,
        "overall_fn": o_fn,
        "nonpref_tp": n_tp,
        "nonpref_fp": n_fp,
        "nonpref_fn": n_fn,
        "preference_exact": preference_exact(gt_pref, pred_pref),
        "full_call_exact": full_em,
        "function_accuracy": function_acc,
    }


def aggregate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    result = {
        "cases": count,
        "pairs": len({row["pair_id"] for row in rows}),
        "source_clusters": len({row["source_example_id"] for row in rows}),
        "primary_preference": prf(
            sum(row["primary_tp"] for row in rows),
            sum(row["primary_fp"] for row in rows),
            sum(row["primary_fn"] for row in rows),
        ),
        "overall_full": prf(
            sum(row["overall_tp"] for row in rows),
            sum(row["overall_fp"] for row in rows),
            sum(row["overall_fn"] for row in rows),
        ),
        "nonpreference": prf(
            sum(row["nonpref_tp"] for row in rows),
            sum(row["nonpref_fp"] for row in rows),
            sum(row["nonpref_fn"] for row in rows),
        ),
        "preference_exact": {"numerator": sum(row["preference_exact"] for row in rows), "denominator": count},
        "full_call_exact": {"numerator": sum(row["full_call_exact"] for row in rows), "denominator": count},
        "function_accuracy": {"numerator": sum(row["function_accuracy"] for row in rows), "denominator": count},
        "terminal_status": {},
    }
    for key in ("preference_exact", "full_call_exact", "function_accuracy"):
        denominator = result[key]["denominator"]
        result[key]["rate"] = result[key]["numerator"] / denominator if denominator else 0.0
    statuses = Counter(row["status"] for row in rows)
    result["terminal_status"] = {
        status: {"count": statuses[status], "denominator": count, "rate": statuses[status] / count if count else 0.0}
        for status in STATUSES
    }
    return result


def _sum_stat(rows: Sequence[Mapping[str, Any]], field: str) -> int:
    return sum(int(row[field]) for row in rows)


def _bootstrap_stratum(rows: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any]:
    sources = sorted({str(row["source_example_id"]) for row in rows})
    if not sources:
        raise ContractError(f"empty bootstrap stratum {label}")
    by_source_mode: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_source_mode[(str(row["source_example_id"]), str(row["mode"]))].append(row)
    source_stats: dict[tuple[str, str], tuple[int, int, int, int, int]] = {}
    for source in sources:
        for mode in MODES:
            group = by_source_mode[(source, mode)]
            source_stats[(source, mode)] = (
                _sum_stat(group, "primary_tp"),
                _sum_stat(group, "primary_fp"),
                _sum_stat(group, "primary_fn"),
                _sum_stat(group, "preference_exact"),
                len(group),
            )

    def point(sampled_sources: Sequence[str]) -> tuple[float, float]:
        metrics: dict[str, tuple[float, float]] = {}
        for mode in MODES:
            values = [source_stats[(source, mode)] for source in sampled_sources]
            tp = sum(value[0] for value in values)
            fp = sum(value[1] for value in values)
            fn = sum(value[2] for value in values)
            exact = sum(value[3] for value in values)
            cases = sum(value[4] for value in values)
            metrics[mode] = (prf(tp, fp, fn)["f1"], exact / cases if cases else 0.0)
        return metrics["multi"][0] - metrics["single"][0], metrics["multi"][1] - metrics["single"][1]

    rng = random.Random(SEED)
    f1_deltas: list[float] = []
    exact_deltas: list[float] = []
    for _ in range(BOOTSTRAP_DRAWS):
        sampled = [sources[rng.randrange(len(sources))] for _ in sources]
        f1_delta, exact_delta = point(sampled)
        f1_deltas.append(f1_delta)
        exact_deltas.append(exact_delta)
    f1_deltas.sort()
    exact_deltas.sort()
    point_f1, point_exact = point(sources)
    source_hash = sha256_bytes((canonical_json(sources) + "\n").encode("utf-8"))
    return {
        "label": label,
        "delta_definition": "multi-single",
        "cluster_count": len(sources),
        "pair_count": len({row["pair_id"] for row in rows}),
        "point_delta_preference_f1": point_f1,
        "point_delta_preference_exact": point_exact,
        "preference_f1_ci95": [f1_deltas[CI_INDICES[0]], f1_deltas[CI_INDICES[1]]],
        "preference_exact_ci95": [exact_deltas[CI_INDICES[0]], exact_deltas[CI_INDICES[1]]],
        "rng": "python.random.Random/MT19937",
        "seed": SEED,
        "independent_reset_per_stratum": True,
        "sorted_cluster_id_sha256": source_hash,
        "draws": BOOTSTRAP_DRAWS,
        "ci_method": "empirical type-1 percentile",
        "ci_indices": list(CI_INDICES),
    }


def _slice(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, Any]:
    values = sorted({str(row[field]) for row in rows if row.get(field) is not None})
    result: dict[str, Any] = {}
    for value in values:
        group = [row for row in rows if str(row.get(field)) == value]
        pairs = len({row["pair_id"] for row in group})
        clusters = len({row["source_example_id"] for row in group})
        if pairs < 30 or clusters < 20:
            result[value] = {"status": "insufficient_support", "pairs": pairs, "source_clusters": clusters}
        else:
            result[value] = {
                "status": "descriptive",
                "pairs": pairs,
                "source_clusters": clusters,
                "by_mode": {mode: aggregate([row for row in group if row["mode"] == mode]) for mode in MODES},
            }
    return result


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with open(temp, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def _csv_text(cell_metrics: Mapping[str, Any]) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["mode", "difficulty", "cases", "tp", "fp", "fn", "precision", "recall", "f1", "pref_exact", "full_call_exact", "function_accuracy"])
    for key in sorted(cell_metrics):
        mode, difficulty = key.split(":", 1)
        metric = cell_metrics[key]
        primary = metric["primary_preference"]
        writer.writerow([
            mode, difficulty, metric["cases"], primary["tp"], primary["fp"], primary["fn"],
            f"{primary['precision']:.12f}", f"{primary['recall']:.12f}", f"{primary['f1']:.12f}",
            f"{metric['preference_exact']['rate']:.12f}", f"{metric['full_call_exact']['rate']:.12f}",
            f"{metric['function_accuracy']['rate']:.12f}",
        ])
    return output.getvalue()


def _report(results: Mapping[str, Any]) -> str:
    lines = [
        "# Gemma 4 12B Vanilla LLM Evaluation",
        "",
        "Primary metric: preference-slot value-OR micro F1. Deltas are multi − single.",
        "",
        "| Mode | Difficulty | Cases | P | R | F1 | Preference EM | Full-call EM | Function acc. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in sorted(results["cells"]):
        metric = results["cells"][key]
        mode, difficulty = key.split(":", 1)
        p = metric["primary_preference"]
        lines.append(
            f"| {mode} | {difficulty} | {metric['cases']} | {p['precision']:.4f} | {p['recall']:.4f} | {p['f1']:.4f} | "
            f"{metric['preference_exact']['rate']:.4f} | {metric['full_call_exact']['rate']:.4f} | {metric['function_accuracy']['rate']:.4f} |"
        )
    lines.extend(["", "## Paired source-cluster bootstrap", "", "| Stratum | Pairs | Sources | ΔF1 | 95% CI |", "|---|---:|---:|---:|---:|"])
    for label in ("pooled", *DIFFICULTIES):
        item = results["paired_bootstrap"][label]
        lines.append(
            f"| {label} | {item['pair_count']} | {item['cluster_count']} | {item['point_delta_preference_f1']:.4f} | "
            f"[{item['preference_f1_ci95'][0]:.4f}, {item['preference_f1_ci95'][1]:.4f}] |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--predictions", default="predictions/full/predictions.jsonl")
    args = parser.parse_args()
    run_root = Path(args.run_root).resolve()
    cases = read_jsonl(run_root / "prepared/cases.jsonl")
    records = read_jsonl(run_root / args.predictions)
    case_by_id = {case["case_id"]: case for case in cases}
    record_by_id = {record["case_id"]: record for record in records}
    if len(case_by_id) != len(cases) or len(record_by_id) != len(records):
        raise ContractError("duplicate case or prediction IDs")
    if set(case_by_id) != set(record_by_id):
        raise ContractError(f"prediction reconciliation failed: missing={len(set(case_by_id)-set(record_by_id))} foreign={len(set(record_by_id)-set(case_by_id))}")
    pref_list = load_json(run_root / "snapshots/pref_list/pref_list.json")
    evaluated = [evaluate_case(case_by_id[case_id], record_by_id[case_id], pref_list) for case_id in sorted(case_by_id)]
    cells = {
        f"{mode}:{difficulty}": aggregate([row for row in evaluated if row["mode"] == mode and row["difficulty"] == difficulty])
        for mode in MODES for difficulty in DIFFICULTIES
    }
    within_mode = {mode: aggregate([row for row in evaluated if row["mode"] == mode]) for mode in MODES}
    bootstrap_rows = {
        "pooled": evaluated,
        **{difficulty: [row for row in evaluated if row["difficulty"] == difficulty] for difficulty in DIFFICULTIES},
    }
    results = {
        "metric_contract": {
            "primary": "preference-slot value-OR micro P/R/F1",
            "failures_in_denominator": True,
            "bootstrap_delta": "multi-single",
            "slice_min_pairs": 30,
            "slice_min_source_clusters": 20,
        },
        "cases": len(evaluated),
        "pairs": len({row["pair_id"] for row in evaluated}),
        "cells": cells,
        "within_mode": within_mode,
        "difficulty_macro_f1": {
            mode: sum(cells[f"{mode}:{difficulty}"]["primary_preference"]["f1"] for difficulty in DIFFICULTIES) / len(DIFFICULTIES)
            for mode in MODES
        },
        "paired_bootstrap": {label: _bootstrap_stratum(bootstrap_rows[label], label) for label in ("pooled", *DIFFICULTIES)},
        "domain_slices": _slice(evaluated, "target_domain"),
        "preference_group_slices": _slice(evaluated, "preference_group"),
        "status_counts": dict(sorted(Counter(row["status"] for row in evaluated).items())),
    }
    if results["status_counts"].get("infra_failure", 0):
        raise ContractError("unresolved infrastructure failures suppress model comparison")
    output_root = run_root / "metrics"
    write_json_atomic(output_root / "metrics.json", results)
    write_json_atomic(output_root / "per_case_metrics.json", evaluated)
    _write_text_atomic(output_root / "metrics.csv", _csv_text(cells))
    _write_text_atomic(output_root / "report.md", _report(results))
    print(canonical_json({"metrics": str(output_root / "metrics.json"), "pairs": results["pairs"], "status_counts": results["status_counts"]}))


if __name__ == "__main__":
    main()
