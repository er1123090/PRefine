#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analyze_session_memory_run as analyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, choices=sorted(analyzer.TASK_CONFIGS), required=True)
    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--analysis_dir", type=str, required=True)
    parser.add_argument(
        "--exclude-action-model",
        dest="exclude_action_models",
        action="append",
        default=[],
        help="Exclude rows for the given action model. Repeatable.",
    )
    return parser.parse_args()


def shorten_action_model_label(action_model: str) -> str:
    label = str(action_model)
    if "/" in label:
        return label.split("/")[-1]
    return label


def shorten_memory_model_label(memory_model: str) -> str:
    label = str(memory_model)
    if "/" in label:
        label = label.split("/")[-1]
    if "_" in label:
        prefix, rest = label.split("_", 1)
        if prefix.lower() in {"qwen", "google", "deepseek-ai", "meta-llama", "openai"}:
            label = rest
    return label


def bounded_metrics_for_task(task: str) -> set[str]:
    if task == "singleturn":
        return {"f1", "precision", "recall"}
    return {
        "overall_precision",
        "overall_recall",
        "overall_f1",
        "pref_em",
        "nonpref_precision",
        "nonpref_recall",
        "nonpref_f1",
    }


def group_rows_by_memory_model(
    rows: Sequence[Dict[str, Any]],
) -> Dict[Tuple[str, str, Tuple[str, str, str]], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, Tuple[str, str, str]], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "error":
            continue
        key = (
            str(row.get("run_tag", "")),
            str(row.get("memory_model", "")),
            analyzer.combo_key(row),
        )
        grouped[key].append(row)
    return grouped


def aggregate_rows_across_action_models(
    rows: Sequence[Dict[str, Any]],
    config: analyzer.TaskConfig,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, int, str, Tuple[str, str, str]], Dict[str, Any]] = {}

    for row in rows:
        if row.get("status") == "error":
            continue

        combo = analyzer.combo_key(row)
        key = (
            str(row.get("memory_model", "")),
            int(row.get("session_index", 0)),
            str(row.get("pref_type", "")),
            combo,
        )
        if key not in grouped:
            grouped[key] = {
                "run_tag": str(row.get("run_tag", "")),
                "memory_model": str(row.get("memory_model", "")),
                "cohort": combo[0],
                "context_type": combo[1],
                "prompt_name": combo[2],
                "session_index": int(row.get("session_index", 0)),
                "pref_type": str(row.get("pref_type", "")),
                "eligible_count": 0,
                "parsing_fail_count": 0,
                "group_count": 0,
            }
            if config.task == "singleturn":
                grouped[key].update(
                    {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                    }
                )
            else:
                grouped[key].update(
                    {
                        "overall_tp": 0,
                        "overall_fp": 0,
                        "overall_fn": 0,
                        "pref_em_count": 0,
                        "n_examples": 0,
                        "nonpref_tp": 0,
                        "nonpref_fp": 0,
                        "nonpref_fn": 0,
                    }
                )

        group = grouped[key]
        group["eligible_count"] += int(row.get("eligible_count", 0) or 0)
        group["parsing_fail_count"] += int(row.get("parsing_fail_count", 0) or 0)
        group["group_count"] += 1

        if config.task == "singleturn":
            group["tp"] += int(row.get("tp", 0) or 0)
            group["fp"] += int(row.get("fp", 0) or 0)
            group["fn"] += int(row.get("fn", 0) or 0)
        else:
            group["overall_tp"] += int(row.get("overall_tp", 0) or 0)
            group["overall_fp"] += int(row.get("overall_fp", 0) or 0)
            group["overall_fn"] += int(row.get("overall_fn", 0) or 0)
            group["pref_em_count"] += int(row.get("pref_em_count", 0) or 0)
            group["n_examples"] += int(row.get("n_examples", 0) or 0)
            group["nonpref_tp"] += int(row.get("nonpref_tp", 0) or 0)
            group["nonpref_fp"] += int(row.get("nonpref_fp", 0) or 0)
            group["nonpref_fn"] += int(row.get("nonpref_fn", 0) or 0)

    aggregated_rows: List[Dict[str, Any]] = []
    for group in grouped.values():
        if config.task == "singleturn":
            metrics = analyzer.prf_from_counts(group["tp"], group["fp"], group["fn"])
            aggregated_rows.append(
                {
                    **group,
                    "precision": metrics.precision,
                    "recall": metrics.recall,
                    "f1": metrics.f1,
                }
            )
        else:
            overall_metrics = analyzer.prf_from_counts(group["overall_tp"], group["overall_fp"], group["overall_fn"])
            nonpref_metrics = analyzer.prf_from_counts(group["nonpref_tp"], group["nonpref_fp"], group["nonpref_fn"])
            aggregated_rows.append(
                {
                    **group,
                    "overall_precision": overall_metrics.precision,
                    "overall_recall": overall_metrics.recall,
                    "overall_f1": overall_metrics.f1,
                    "pref_em": (
                        group["pref_em_count"] / group["n_examples"] if group["n_examples"] > 0 else 0.0
                    ),
                    "nonpref_precision": nonpref_metrics.precision,
                    "nonpref_recall": nonpref_metrics.recall,
                    "nonpref_f1": nonpref_metrics.f1,
                }
            )

    aggregated_rows.sort(
        key=lambda row: (
            str(row.get("run_tag", "")),
            str(row.get("cohort", "")),
            str(row.get("context_type", "")),
            str(row.get("prompt_name", "")),
            str(row.get("pref_type", "")),
            str(row.get("memory_model", "")),
            int(row.get("session_index", 0)),
        )
    )
    return aggregated_rows


def plot_metric_aggregate_by_memory_model(
    rows: Sequence[Dict[str, Any]],
    metric_key: str,
    ylabel: str,
    output_path: Path,
    task: str,
    combo: Tuple[str, str, str],
    compact_legend: bool = False,
) -> None:
    analyzer.ensure_dir(output_path.parent)

    run_tag = str(rows[0].get("run_tag", "run"))
    memory_models = sorted({str(row["memory_model"]) for row in rows})
    bounded_metrics = bounded_metrics_for_task(task)

    fig_height = 5.8 if compact_legend else 5.0
    fig, axes = plt.subplots(1, 3, figsize=(18, fig_height), sharey=(metric_key in bounded_metrics))

    for ax, pref_type in zip(axes, analyzer.PREF_ORDER):
        pref_rows = [row for row in rows if str(row.get("pref_type", "")) == pref_type]
        ax.set_title(pref_type.capitalize())
        ax.set_xlabel("Session index")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        if ax is axes[0]:
            ax.set_ylabel(ylabel)

        if not pref_rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        for memory_model in memory_models:
            series = sorted(
                (row for row in pref_rows if str(row.get("memory_model", "")) == memory_model),
                key=lambda row: int(row["session_index"]),
            )
            if not series:
                continue
            x_values = [int(row["session_index"]) for row in series]
            y_values = [float(row[metric_key]) for row in series]
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2,
                markersize=5,
                label=shorten_memory_model_label(memory_model) if compact_legend else memory_model,
            )

        if metric_key in bounded_metrics:
            ax.set_ylim(0.0, 1.0)

    fig.suptitle(
        f"Action models aggregated | {analyzer.combo_title(combo)} | {ylabel} by session\nRun: {run_tag}",
        y=0.98,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        if compact_legend:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.90),
                ncol=min(3, len(labels)),
                frameon=False,
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.76))
        else:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.92),
                ncol=min(3, len(labels)),
                frameon=False,
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.80))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.88))

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_metric_by_memory_model(
    rows: Sequence[Dict[str, Any]],
    metric_key: str,
    ylabel: str,
    output_path: Path,
    task: str,
    combo: Tuple[str, str, str],
    compact_legend: bool = False,
) -> None:
    analyzer.ensure_dir(output_path.parent)

    run_tag = str(rows[0].get("run_tag", "run"))
    memory_model = str(rows[0].get("memory_model", "memory_model"))
    action_models = sorted({str(row["action_model"]) for row in rows})
    bounded_metrics = bounded_metrics_for_task(task)

    fig_height = 5.8 if compact_legend else 5.0
    fig, axes = plt.subplots(1, 3, figsize=(18, fig_height), sharey=(metric_key in bounded_metrics))

    for ax, pref_type in zip(axes, analyzer.PREF_ORDER):
        pref_rows = [row for row in rows if str(row.get("pref_type", "")) == pref_type]
        ax.set_title(pref_type.capitalize())
        ax.set_xlabel("Session index")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        if ax is axes[0]:
            ax.set_ylabel(ylabel)

        if not pref_rows:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        for action_model in action_models:
            series = sorted(
                (row for row in pref_rows if str(row.get("action_model", "")) == action_model),
                key=lambda row: int(row["session_index"]),
            )
            if not series:
                continue
            x_values = [int(row["session_index"]) for row in series]
            y_values = [float(row[metric_key]) for row in series]
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2,
                markersize=5,
                label=shorten_action_model_label(action_model) if compact_legend else action_model,
            )

        if metric_key in bounded_metrics:
            ax.set_ylim(0.0, 1.0)

    title_memory_model = shorten_memory_model_label(memory_model) if compact_legend else memory_model
    fig.suptitle(
        f"{title_memory_model} | {analyzer.combo_title(combo)} | {ylabel} by session\nRun: {run_tag}",
        y=0.98,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        if compact_legend:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.90),
                ncol=min(3, len(labels)),
                frameon=False,
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.76))
        else:
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.92),
                ncol=min(3, len(labels)),
                frameon=False,
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.80))
    else:
        fig.tight_layout(rect=(0, 0, 1, 0.88))

    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_memory_model_aggregate_and_detailed_plots(
    summary_rows: Sequence[Dict[str, Any]],
    analysis_dir: Path,
    config: analyzer.TaskConfig,
) -> List[Path]:
    grouped = group_rows_by_memory_model(summary_rows)
    saved_paths: List[Path] = []

    for (run_tag, memory_model, combo), group_rows in sorted(grouped.items()):
        stem = f"{analyzer.safe_name(run_tag)}__{analyzer.safe_name(memory_model)}__{analyzer.combo_suffix(combo)}"

        session_table_rows = analyzer.build_session_table_rows(group_rows, config)
        session_table_path = analysis_dir / "memory_model_session_tables" / f"{stem}__session_table.csv"
        analyzer.write_csv(session_table_path, analyzer.SESSION_TABLE_FIELDNAMES, session_table_rows)
        saved_paths.append(session_table_path)

        aggregate_plot_path = (
            analysis_dir / "plots" / "by_memory_model" / "aggregate" / f"{stem}__session_f1_aggregate.png"
        )
        analyzer.plot_session_aggregate(
            session_table_rows=session_table_rows,
            run_tag=run_tag,
            combo=combo,
            output_path=aggregate_plot_path,
            config=config,
        )
        saved_paths.append(aggregate_plot_path)

        for plot_spec in config.detailed_plots:
            output_path = (
                analysis_dir
                / "plots"
                / "by_memory_model"
                / "detailed"
                / plot_spec.subdir
                / f"{stem}__{plot_spec.filename_suffix}.png"
            )
            plot_metric_by_memory_model(
                rows=group_rows,
                metric_key=plot_spec.metric_key,
                ylabel=plot_spec.ylabel,
                output_path=output_path,
                task=config.task,
                combo=combo,
                compact_legend=plot_spec.compact_legend,
            )
            saved_paths.append(output_path)

    combos = sorted({analyzer.combo_key(row) for row in summary_rows if row.get("status") != "error"})
    for combo in combos:
        combo_rows = [row for row in summary_rows if row.get("status") != "error" and analyzer.combo_key(row) == combo]
        if not combo_rows:
            continue

        aggregate_rows = aggregate_rows_across_action_models(combo_rows, config)
        for plot_spec in config.detailed_plots:
            output_path = (
                analysis_dir
                / "plots"
                / "by_memory_model"
                / "detailed"
                / plot_spec.subdir
                / (
                    f"aggregate_by_memory_model__{analyzer.combo_suffix(combo)}"
                    f"__{plot_spec.filename_suffix}.png"
                )
            )
            plot_metric_aggregate_by_memory_model(
                rows=aggregate_rows,
                metric_key=plot_spec.metric_key,
                ylabel=plot_spec.ylabel,
                output_path=output_path,
                task=config.task,
                combo=combo,
                compact_legend=plot_spec.compact_legend,
            )
            saved_paths.append(output_path)

    return saved_paths


def main() -> None:
    args = parse_args()
    config = analyzer.TASK_CONFIGS[args.task]
    plot_module = importlib.import_module(config.plot_module_name)

    summary_rows = plot_module.load_summary_rows(args.summary_csv)
    excluded = {str(model) for model in args.exclude_action_models}
    if excluded:
        summary_rows = [row for row in summary_rows if str(row.get("action_model", "")) not in excluded]

    if not summary_rows:
        raise ValueError("No rows remain after filtering summary rows.")

    analysis_dir = Path(args.analysis_dir).resolve()
    saved_paths = write_memory_model_aggregate_and_detailed_plots(summary_rows, analysis_dir, config)

    print(f"[TASK] {config.task}")
    print(f"[ANALYSIS DIR] {analysis_dir}")
    if excluded:
        print(f"[EXCLUDED ACTION MODELS] {', '.join(sorted(excluded))}")
    print(f"[SUMMARY ROWS] {len(summary_rows)}")
    for path in saved_paths:
        print(f"[OUTPUT] {path}")


if __name__ == "__main__":
    main()
