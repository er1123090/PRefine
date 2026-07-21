#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
from collections import defaultdict
from dataclasses import dataclass
from math import isclose
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from session_memory_eval_common import DEFAULT_PREF_LIST_PATH, ensure_dir, iter_jsonl, safe_name, write_csv


SINGLETURN_DETAIL_FIELDNAMES = [
    "run_tag",
    "action_model",
    "action_model_safe",
    "memory_model",
    "cohort",
    "session_index",
    "pref_type",
    "prompt_name",
    "context_type",
    "eligible_count",
    "input_path",
    "memory_path",
    "json_path",
    "output_path",
    "log_path",
    "status",
    "error",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "parsing_fail_count",
    "or_log_path",
    "parsing_fail_path",
]

SINGLETURN_SUMMARY_FIELDNAMES = [
    "run_tag",
    "action_model",
    "action_model_safe",
    "memory_model",
    "cohort",
    "session_index",
    "pref_type",
    "prompt_name",
    "context_type",
    "eligible_count",
    "files_in_group",
    "ok_files",
    "error_files",
    "status",
    "tp",
    "fp",
    "fn",
    "precision",
    "recall",
    "f1",
    "parsing_fail_count",
]

MULTITURN_DETAIL_FIELDNAMES = [
    "run_tag",
    "action_model",
    "action_model_safe",
    "memory_model",
    "cohort",
    "session_index",
    "pref_type",
    "prompt_name",
    "context_type",
    "eligible_count",
    "input_path",
    "memory_path",
    "json_path",
    "output_path",
    "log_path",
    "status",
    "error",
    "overall_tp",
    "overall_fp",
    "overall_fn",
    "overall_precision",
    "overall_recall",
    "overall_f1",
    "pref_em",
    "pref_em_count",
    "n_examples",
    "nonpref_tp",
    "nonpref_fp",
    "nonpref_fn",
    "nonpref_precision",
    "nonpref_recall",
    "nonpref_f1",
    "parsing_fail_count",
    "or_log_path",
    "parsing_fail_path",
]

MULTITURN_SUMMARY_FIELDNAMES = [
    "run_tag",
    "action_model",
    "action_model_safe",
    "memory_model",
    "cohort",
    "session_index",
    "pref_type",
    "prompt_name",
    "context_type",
    "eligible_count",
    "files_in_group",
    "ok_files",
    "error_files",
    "status",
    "overall_tp",
    "overall_fp",
    "overall_fn",
    "overall_precision",
    "overall_recall",
    "overall_f1",
    "pref_em",
    "pref_em_count",
    "n_examples",
    "nonpref_tp",
    "nonpref_fp",
    "nonpref_fn",
    "nonpref_precision",
    "nonpref_recall",
    "nonpref_f1",
    "parsing_fail_count",
]

SESSION_TABLE_FIELDNAMES = [
    "session_index",
    "overall_precision",
    "overall_recall",
    "overall_f1",
    "easy_f1",
    "medium_f1",
    "hard_f1",
    "overall_parsing_fail_count",
    "easy_parsing_fail_count",
    "medium_parsing_fail_count",
    "hard_parsing_fail_count",
    "overall_tp",
    "overall_fp",
    "overall_fn",
    "eligible_count",
    "group_count",
    "delta_overall_f1_from_prev_session",
]

PREF_ORDER = ["easy", "medium", "hard"]
COMBO_KEYS = ("cohort", "context_type", "prompt_name")


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class PlotSpec:
    metric_key: str
    ylabel: str
    subdir: str
    filename_suffix: str
    compact_legend: bool = False


@dataclass(frozen=True)
class TaskConfig:
    task: str
    detail_fieldnames: Sequence[str]
    summary_fieldnames: Sequence[str]
    tp_key: str
    fp_key: str
    fn_key: str
    precision_key: str
    recall_key: str
    f1_key: str
    aggregate_title_metric: str
    aggregate_ylabel: str
    eval_module_name: str
    plot_module_name: str
    detailed_plots: Sequence[PlotSpec]


TASK_CONFIGS: Dict[str, TaskConfig] = {
    "singleturn": TaskConfig(
        task="singleturn",
        detail_fieldnames=SINGLETURN_DETAIL_FIELDNAMES,
        summary_fieldnames=SINGLETURN_SUMMARY_FIELDNAMES,
        tp_key="tp",
        fp_key="fp",
        fn_key="fn",
        precision_key="precision",
        recall_key="recall",
        f1_key="f1",
        aggregate_title_metric="F1",
        aggregate_ylabel="F1",
        eval_module_name="evaluate_session_memory_results_singleturn",
        plot_module_name="plot_session_memory_curves_singleturn",
        detailed_plots=(
            PlotSpec("f1", "F1", "f1", "f1"),
            PlotSpec("f1", "F1", "f1_legend_safe", "f1_legend_safe", compact_legend=True),
            PlotSpec("precision", "Precision", "precision", "precision"),
            PlotSpec("recall", "Recall", "recall", "recall"),
            PlotSpec("parsing_fail_count", "Parsing fail count", "parsing_fail_count", "parsing_fail_count"),
        ),
    ),
    "multiturn": TaskConfig(
        task="multiturn",
        detail_fieldnames=MULTITURN_DETAIL_FIELDNAMES,
        summary_fieldnames=MULTITURN_SUMMARY_FIELDNAMES,
        tp_key="overall_tp",
        fp_key="overall_fp",
        fn_key="overall_fn",
        precision_key="overall_precision",
        recall_key="overall_recall",
        f1_key="overall_f1",
        aggregate_title_metric="Overall F1",
        aggregate_ylabel="Overall F1",
        eval_module_name="evaluate_session_memory_results_multiturn",
        plot_module_name="plot_session_memory_curves_multiturn",
        detailed_plots=(
            PlotSpec("overall_f1", "Overall F1", "overall_f1", "overall_f1"),
            PlotSpec("overall_precision", "Overall Precision", "overall_precision", "overall_precision"),
            PlotSpec("overall_recall", "Overall Recall", "overall_recall", "overall_recall"),
            PlotSpec("pref_em", "Preference EM", "pref_em", "pref_em"),
            PlotSpec("nonpref_f1", "Non-pref F1", "nonpref_f1", "nonpref_f1"),
            PlotSpec("parsing_fail_count", "Parsing fail count", "parsing_fail_count", "parsing_fail_count"),
        ),
    ),
}


def prf_from_counts(tp: int, fp: int, fn: int) -> PRF:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return PRF(precision, recall, f1, tp, fp, fn)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--analysis_dir", type=str, default=None)
    parser.add_argument("--task", type=str, choices=sorted(TASK_CONFIGS), default=None)
    parser.add_argument(
        "--exclude-action-model",
        dest="exclude_action_models",
        action="append",
        default=[],
        help="Exclude rows for the given action model. Repeatable.",
    )
    return parser.parse_args()


def infer_task_from_run_dir(run_dir: Path) -> str:
    text = str(run_dir).lower()
    matches = [task for task in TASK_CONFIGS if task in text]
    unique_matches = sorted(set(matches))
    if len(unique_matches) == 1:
        return unique_matches[0]
    if not unique_matches:
        raise ValueError(
            f"Could not infer task from run_dir: {run_dir}. "
            "Pass --task singleturn or --task multiturn."
        )
    raise ValueError(
        f"Ambiguous task inference from run_dir: {run_dir}. "
        f"Matched {unique_matches}. Pass --task explicitly."
    )


def resolve_config(task_arg: str | None, run_dir: Path) -> TaskConfig:
    task = task_arg or infer_task_from_run_dir(run_dir)
    return TASK_CONFIGS[task]


def load_support_modules(config: TaskConfig) -> Tuple[Any, Callable[..., None]]:
    eval_module = importlib.import_module(config.eval_module_name)
    plot_module = importlib.import_module(config.plot_module_name)
    return eval_module, plot_module.plot_metric


def combo_key(row: Dict[str, Any]) -> Tuple[str, str, str]:
    return tuple(str(row.get(key, "")) for key in COMBO_KEYS)


def combo_suffix(combo: Tuple[str, str, str]) -> str:
    return "__".join(safe_name(part) for part in combo)


def combo_title(combo: Tuple[str, str, str]) -> str:
    labels = [f"{key}={value}" for key, value in zip(COMBO_KEYS, combo)]
    return ", ".join(labels)


def output_path_with_combo(path: Path, combo: Tuple[str, str, str], use_suffix: bool) -> Path:
    if not use_suffix:
        return path
    return path.with_name(f"{path.stem}__{combo_suffix(combo)}{path.suffix}")


def build_eval_args(run_manifest: Path, out_csv: Path, out_summary_csv: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_manifest=str(run_manifest),
        out_csv=str(out_csv),
        out_summary_csv=str(out_summary_csv),
        pref_list_path=str(DEFAULT_PREF_LIST_PATH),
        save_or_logs=False,
        save_parsing_failures=False,
        logs_dir=None,
    )


def evaluate_outputs(
    config: TaskConfig,
    eval_module: Any,
    run_manifest: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    args = build_eval_args(run_manifest, Path("/dev/null"), Path("/dev/null"))
    return eval_module.evaluate_rows(args)


def filter_rows_by_action_model(
    rows: Sequence[Dict[str, Any]],
    excluded_action_models: Sequence[str],
) -> List[Dict[str, Any]]:
    if not excluded_action_models:
        return list(rows)

    excluded = {str(model) for model in excluded_action_models}
    return [row for row in rows if str(row.get("action_model", "")) not in excluded]


def aggregate_counts(rows: Iterable[Dict[str, Any]], config: TaskConfig) -> Dict[str, int]:
    total = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "parsing_fail_count": 0,
        "eligible_count": 0,
        "group_count": 0,
    }
    for row in rows:
        total["tp"] += int(row.get(config.tp_key, 0) or 0)
        total["fp"] += int(row.get(config.fp_key, 0) or 0)
        total["fn"] += int(row.get(config.fn_key, 0) or 0)
        total["parsing_fail_count"] += int(row.get("parsing_fail_count", 0) or 0)
        total["eligible_count"] += int(row.get("eligible_count", 0) or 0)
        total["group_count"] += 1
    return total


def metric_value_or_blank(counts: Dict[str, int], metric_name: str) -> float | str:
    if counts["group_count"] == 0:
        return ""
    metrics = prf_from_counts(counts["tp"], counts["fp"], counts["fn"])
    return getattr(metrics, metric_name)


def build_session_table_rows(summary_rows: Sequence[Dict[str, Any]], config: TaskConfig) -> List[Dict[str, Any]]:
    session_to_pref_rows: Dict[int, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in summary_rows:
        session_to_pref_rows[int(row["session_index"])][str(row["pref_type"])].append(row)

    table_rows: List[Dict[str, Any]] = []
    previous_overall_f1: float | None = None

    for session_index in sorted(session_to_pref_rows):
        pref_rows = session_to_pref_rows[session_index]
        overall_counts = aggregate_counts((row for rows in pref_rows.values() for row in rows), config)
        overall_metrics = prf_from_counts(overall_counts["tp"], overall_counts["fp"], overall_counts["fn"])

        table_row: Dict[str, Any] = {
            "session_index": session_index,
            "overall_precision": overall_metrics.precision,
            "overall_recall": overall_metrics.recall,
            "overall_f1": overall_metrics.f1,
            "overall_parsing_fail_count": overall_counts["parsing_fail_count"],
            "overall_tp": overall_counts["tp"],
            "overall_fp": overall_counts["fp"],
            "overall_fn": overall_counts["fn"],
            "eligible_count": overall_counts["eligible_count"],
            "group_count": overall_counts["group_count"],
        }

        for pref_type in PREF_ORDER:
            counts = aggregate_counts(pref_rows.get(pref_type, []), config)
            table_row[f"{pref_type}_f1"] = metric_value_or_blank(counts, "f1")
            table_row[f"{pref_type}_parsing_fail_count"] = (
                counts["parsing_fail_count"] if counts["group_count"] > 0 else ""
            )

        if previous_overall_f1 is None:
            table_row["delta_overall_f1_from_prev_session"] = 0.0
        else:
            table_row["delta_overall_f1_from_prev_session"] = overall_metrics.f1 - previous_overall_f1
        previous_overall_f1 = overall_metrics.f1
        table_rows.append(table_row)

    return table_rows


def plot_session_aggregate(
    session_table_rows: Sequence[Dict[str, Any]],
    run_tag: str,
    combo: Tuple[str, str, str],
    output_path: Path,
    config: TaskConfig,
) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(10, 6))

    x_values = [int(row["session_index"]) for row in session_table_rows]
    series_specs = [
        ("overall_f1", "Overall"),
        ("easy_f1", "Easy"),
        ("medium_f1", "Medium"),
        ("hard_f1", "Hard"),
    ]

    for key, label in series_specs:
        xs: List[int] = []
        ys: List[float] = []
        for row in session_table_rows:
            value = row.get(key, "")
            if value == "":
                continue
            xs.append(int(row["session_index"]))
            ys.append(float(value))
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, markersize=5, label=label)

    ax.set_xlabel("Session index")
    ax.set_ylabel(config.aggregate_ylabel)
    ax.set_xticks(x_values)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.set_title(
        f"Session memory aggregate {config.aggregate_title_metric} by session\n"
        f"Run: {run_tag} | {combo_title(combo)}"
    )
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_session_tables_and_plots(
    summary_rows: Sequence[Dict[str, Any]],
    analysis_dir: Path,
    config: TaskConfig,
    plot_metric_func: Callable[..., None],
) -> List[Path]:
    ok_rows = [row for row in summary_rows if row.get("status") == "ok"]
    combos = sorted({combo_key(row) for row in summary_rows})
    use_combo_suffix = len(combos) > 1
    saved_paths: List[Path] = []

    for combo in combos:
        combo_ok_rows = [row for row in ok_rows if combo_key(row) == combo]
        if not combo_ok_rows:
            continue

        session_table_rows = build_session_table_rows(combo_ok_rows, config)
        table_path = output_path_with_combo(analysis_dir / "session_memory_session_table.csv", combo, use_combo_suffix)
        write_csv(table_path, SESSION_TABLE_FIELDNAMES, session_table_rows)
        saved_paths.append(table_path)

        aggregate_plot_path = output_path_with_combo(
            analysis_dir / "plots" / "session_f1_aggregate.png",
            combo,
            use_combo_suffix,
        )
        plot_session_aggregate(
            session_table_rows=session_table_rows,
            run_tag=str(combo_ok_rows[0].get("run_tag", "run")),
            combo=combo,
            output_path=aggregate_plot_path,
            config=config,
        )
        saved_paths.append(aggregate_plot_path)

        detailed_root = analysis_dir / "plots" / "detailed"
        if use_combo_suffix:
            detailed_root = detailed_root / combo_suffix(combo)
        saved_paths.extend(write_detailed_plots(combo_ok_rows, detailed_root, config, plot_metric_func))

    return saved_paths


def write_detailed_plots(
    summary_rows: Sequence[Dict[str, Any]],
    plot_dir: Path,
    config: TaskConfig,
    plot_metric_func: Callable[..., None],
) -> List[Path]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        key = (str(row.get("run_tag", "")), str(row.get("action_model", "")), str(row.get("cohort", "")))
        groups[key].append(row)

    saved_files: List[Path] = []
    for run_tag, action_model, cohort in groups:
        group_rows = groups[(run_tag, action_model, cohort)]
        stem = f"{safe_name(run_tag)}__{safe_name(action_model)}__{safe_name(cohort)}"
        for plot_spec in config.detailed_plots:
            output_path = plot_dir / plot_spec.subdir / f"{stem}__{plot_spec.filename_suffix}.png"
            plot_metric_func(
                group_rows,
                plot_spec.metric_key,
                plot_spec.ylabel,
                output_path,
                compact_legend=plot_spec.compact_legend,
            )
            saved_files.append(output_path)
    return saved_files


def validate_manifest_row_count(run_manifest: Path, detailed_rows: Sequence[Dict[str, Any]]) -> None:
    manifest_count = sum(1 for _ in iter_jsonl(run_manifest))
    if manifest_count != len(detailed_rows):
        raise ValueError(
            f"Manifest rows ({manifest_count}) do not match detailed CSV rows ({len(detailed_rows)})"
        )


def validate_overall_metrics(
    detailed_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    config: TaskConfig,
) -> None:
    detailed_ok = [row for row in detailed_rows if row.get("status") == "ok"]
    summary_ok = [row for row in summary_rows if row.get("status") == "ok"]

    detailed_metrics = prf_from_counts(
        sum(int(row[config.tp_key]) for row in detailed_ok),
        sum(int(row[config.fp_key]) for row in detailed_ok),
        sum(int(row[config.fn_key]) for row in detailed_ok),
    )
    summary_metrics = prf_from_counts(
        sum(int(row[config.tp_key]) for row in summary_ok),
        sum(int(row[config.fp_key]) for row in summary_ok),
        sum(int(row[config.fn_key]) for row in summary_ok),
    )

    for attr in ("precision", "recall", "f1"):
        if not isclose(getattr(detailed_metrics, attr), getattr(summary_metrics, attr), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(
                f"Overall {attr} mismatch between detailed and summary rows: "
                f"{getattr(detailed_metrics, attr)} vs {getattr(summary_metrics, attr)}"
            )


def validate_output_paths(paths: Sequence[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Expected output files were not created: {missing}")


def print_report(
    analysis_dir: Path,
    detailed_rows: Sequence[Dict[str, Any]],
    summary_rows: Sequence[Dict[str, Any]],
    saved_paths: Sequence[Path],
    config: TaskConfig,
    excluded_action_models: Sequence[str] | None = None,
) -> None:
    error_rows = [row for row in summary_rows if row.get("status") == "error"]
    partial_rows = [row for row in summary_rows if row.get("status") == "partial"]
    ok_rows = [row for row in summary_rows if row.get("status") == "ok"]
    overall = prf_from_counts(
        sum(int(row[config.tp_key]) for row in ok_rows),
        sum(int(row[config.fp_key]) for row in ok_rows),
        sum(int(row[config.fn_key]) for row in ok_rows),
    )
    total_failures = sum(int(row["parsing_fail_count"]) for row in ok_rows)

    print(f"[TASK] {config.task}")
    print(f"[ANALYSIS DIR] {analysis_dir}")
    if excluded_action_models:
        print(f"[EXCLUDED ACTION MODELS] {', '.join(excluded_action_models)}")
    print(f"[DETAILED ROWS] {len(detailed_rows)}")
    print(f"[SUMMARY ROWS] ok={len(ok_rows)} partial={len(partial_rows)} error={len(error_rows)}")
    print(
        f"[OVERALL] TP={overall.tp} FP={overall.fp} FN={overall.fn} "
        f"P={overall.precision:.4f} R={overall.recall:.4f} F1={overall.f1:.4f}"
    )
    print(f"[PARSING FAILURES] {total_failures}")
    for path in saved_paths:
        print(f"[OUTPUT] {path}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    analysis_dir = Path(args.analysis_dir).resolve() if args.analysis_dir else run_dir / "analysis"
    config = resolve_config(args.task, run_dir)
    eval_module, plot_metric_func = load_support_modules(config)
    ensure_dir(analysis_dir)

    run_manifest = run_dir / "run_manifest.jsonl"
    if not run_manifest.exists():
        raise FileNotFoundError(f"Missing run manifest: {run_manifest}")

    detailed_csv_path = analysis_dir / "session_memory_metrics_detailed.csv"
    summary_csv_path = analysis_dir / "session_memory_metrics_summary.csv"

    all_detailed_rows, all_summary_rows = evaluate_outputs(
        config=config,
        eval_module=eval_module,
        run_manifest=run_manifest,
    )
    validate_manifest_row_count(run_manifest, all_detailed_rows)
    validate_overall_metrics(all_detailed_rows, all_summary_rows, config)

    detailed_rows = filter_rows_by_action_model(all_detailed_rows, args.exclude_action_models)
    summary_rows = filter_rows_by_action_model(all_summary_rows, args.exclude_action_models)

    if not summary_rows:
        raise ValueError("No rows remain after applying action-model filters.")

    write_csv(detailed_csv_path, config.detail_fieldnames, detailed_rows)
    write_csv(summary_csv_path, config.summary_fieldnames, summary_rows)

    saved_paths = [detailed_csv_path, summary_csv_path]
    saved_paths.extend(write_session_tables_and_plots(summary_rows, analysis_dir, config, plot_metric_func))
    validate_output_paths(saved_paths)
    print_report(
        analysis_dir,
        detailed_rows,
        summary_rows,
        saved_paths,
        config,
        excluded_action_models=args.exclude_action_models,
    )


if __name__ == "__main__":
    main()
