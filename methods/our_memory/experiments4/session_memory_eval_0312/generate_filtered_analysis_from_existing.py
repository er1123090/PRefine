#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import importlib
import shutil
from pathlib import Path
from typing import Any, Dict, List

import analyze_session_memory_run as analyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--task", type=str, choices=sorted(analyzer.TASK_CONFIGS), required=True)
    parser.add_argument("--analysis_dir", type=str, required=True)
    parser.add_argument(
        "--exclude-action-model",
        dest="exclude_action_models",
        action="append",
        default=[],
        help="Exclude rows for the given action model. Repeatable.",
    )
    return parser.parse_args()


def load_filtered_detailed_rows(path: Path, excluded: set[str]) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [row for row in csv.DictReader(f) if row.get("action_model", "") not in excluded]


def main() -> None:
    args = parse_args()
    excluded = {str(model) for model in args.exclude_action_models}

    run_dir = Path(args.run_dir).resolve()
    src_analysis = run_dir / "analysis"
    dst_analysis = Path(args.analysis_dir).resolve()
    config = analyzer.TASK_CONFIGS[args.task]
    _, plot_metric_func = analyzer.load_support_modules(config)
    plot_module = importlib.import_module(config.plot_module_name)

    if not src_analysis.exists():
        raise FileNotFoundError(f"Missing source analysis dir: {src_analysis}")

    if dst_analysis.exists():
        shutil.rmtree(dst_analysis)
    analyzer.ensure_dir(dst_analysis)

    detailed_rows = load_filtered_detailed_rows(src_analysis / "session_memory_metrics_detailed.csv", excluded)
    summary_rows = plot_module.load_summary_rows(str(src_analysis / "session_memory_metrics_summary.csv"))
    summary_rows = [row for row in summary_rows if str(row.get("action_model", "")) not in excluded]

    if not summary_rows:
        raise ValueError("No rows remain after applying action-model filters.")

    analyzer.write_csv(dst_analysis / "session_memory_metrics_detailed.csv", config.detail_fieldnames, detailed_rows)
    analyzer.write_csv(dst_analysis / "session_memory_metrics_summary.csv", config.summary_fieldnames, summary_rows)

    saved_paths = [
        dst_analysis / "session_memory_metrics_detailed.csv",
        dst_analysis / "session_memory_metrics_summary.csv",
    ]
    saved_paths.extend(analyzer.write_session_tables_and_plots(summary_rows, dst_analysis, config, plot_metric_func))

    print(f"[TASK] {args.task}")
    print(f"[SRC ANALYSIS DIR] {src_analysis}")
    print(f"[DST ANALYSIS DIR] {dst_analysis}")
    if excluded:
        print(f"[EXCLUDED ACTION MODELS] {', '.join(sorted(excluded))}")
    print(f"[FILTERED DETAILED ROWS] {len(detailed_rows)}")
    print(f"[FILTERED SUMMARY ROWS] {len(summary_rows)}")
    for path in saved_paths:
        print(f"[OUTPUT] {path}")


if __name__ == "__main__":
    main()
