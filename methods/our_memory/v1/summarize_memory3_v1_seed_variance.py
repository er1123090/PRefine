#!/usr/bin/env python3
"""Summarize repeat variance, confidence intervals, and pairwise-test eligibility."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Tuple

ROOT = Path(__file__).resolve().parents[3]
PRIMARY_METRICS = ["overall_f1", "pref_em", "nonpref_f1", "parse_failure_rate"]


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-run-id", required=True)
    args = parser.parse_args()

    out_root = ROOT / "results" / "our_memory" / args.parent_run_id
    per_file = read_csv(out_root / "per_file_metrics.csv")
    if not per_file:
        raise SystemExit(f"No per-file metrics found under {out_root}")

    grouped: Dict[Tuple[str, str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for row in per_file:
        key = (
            row["inference_model"],
            row["memory_safe_name"],
            row["turn"],
            row["difficulty"],
        )
        grouped[key].append(row)

    variance_rows: List[Dict[str, Any]] = []
    ci_rows: List[Dict[str, Any]] = []
    for key, rows in grouped.items():
        inference_model, memory_safe_name, turn, difficulty = key
        repeat_ids = sorted({row["repeat_id"] for row in rows})
        for metric in PRIMARY_METRICS:
            values = [float(row[metric]) for row in rows if row.get(metric) not in ("", None)]
            n = len(values)
            metric_mean = mean(values) if values else 0.0
            metric_std = stdev(values) if n > 1 else 0.0
            metric_se = metric_std / math.sqrt(n) if n else 0.0
            common = {
                "inference_model": inference_model,
                "memory_safe_name": memory_safe_name,
                "turn": turn,
                "difficulty": difficulty,
                "metric": metric,
                "n_repeats": n,
                "repeat_ids": " ".join(repeat_ids),
                "mean": metric_mean,
                "std": metric_std,
                "se": metric_se,
                "seed_policy": "deterministic_rerun",
            }
            variance_rows.append(common)
            ci_rows.append(
                {
                    **common,
                    "method": "repeat_normal_approx_small_n",
                    "confidence_level": 0.95,
                    "ci_lower": metric_mean - 1.96 * metric_se,
                    "ci_upper": metric_mean + 1.96 * metric_se,
                    "limitation": "Only three repeats; if temperature remains 0.0 this is deterministic rerun variance, not stochastic seed variance.",
                }
            )

    write_csv(out_root / "repeat_variance.csv", variance_rows)
    write_csv(out_root / "confidence_intervals.csv", ci_rows)

    pairwise_rows = [
        {
            "status": "not_applicable",
            "reason": "no_comparator",
            "method": "not_applicable",
            "note": "Define explicit comparator pairs before claiming statistical significance.",
        }
    ]
    write_csv(out_root / "pairwise_tests.csv", pairwise_rows)

    readme = [
        "# ours_memory_v1 MEMORY3 seed-variance summary",
        "",
        f"Parent run: `{args.parent_run_id}`",
        "",
        "Repeat mode: `deterministic_rerun` unless the runner metadata proves a stochastic request/server seed was used.",
        "",
        "Generated files:",
        "- `per_file_metrics.csv`",
        "- `per_example_correctness.csv`",
        "- `summary_by_model_turn_difficulty.csv`",
        "- `coverage.csv`",
        "- `repeat_variance.csv`",
        "- `confidence_intervals.csv`",
        "- `pairwise_tests.csv`",
        "",
        "Pairwise significance is marked not applicable until explicit comparator pairs are defined.",
    ]
    (out_root / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(f"Wrote summary artifacts under {out_root}")


if __name__ == "__main__":
    main()
