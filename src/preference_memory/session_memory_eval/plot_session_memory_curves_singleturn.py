#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from session_memory_eval_common import DEFAULT_PLOT_ROOT, ensure_dir, safe_name


PREF_ORDER = ["easy", "medium", "hard"]


def display_action_label(action_model: str, shorten: bool = False) -> str:
    label = str(action_model)
    if not shorten:
        return label
    if "/" in label:
        return label.split("/")[-1]
    return label


def display_model_label(memory_model: str, shorten: bool = False) -> str:
    label = str(memory_model)
    if not shorten:
        return label

    if "/" in label:
        label = label.split("/")[-1]
    if "_" in label:
        prefix, rest = label.split("_", 1)
        if prefix.lower() in {"qwen", "google", "deepseek-ai", "meta-llama", "openai"}:
            label = rest
    return label


def load_summary_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for key in ("session_index", "eligible_count", "files_in_group", "ok_files", "error_files", "tp", "fp", "fn", "parsing_fail_count"):
                parsed[key] = int(parsed.get(key, 0) or 0)
            for key in ("precision", "recall", "f1"):
                parsed[key] = float(parsed.get(key, 0.0) or 0.0)
            rows.append(parsed)
    return rows


def group_rows(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "error":
            continue
        key = (row.get("run_tag", ""), row.get("action_model", ""), row.get("cohort", ""))
        grouped[key].append(row)
    return grouped


def group_rows_for_aggregate(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "error":
            continue
        key = (row.get("run_tag", ""), row.get("cohort", ""))
        grouped[key].append(row)
    return grouped


def aggregate_rows_across_memory_models(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, int, str], Dict[str, Any]] = {}

    for row in rows:
        key = (
            str(row.get("run_tag", "")),
            str(row.get("action_model", "")),
            int(row.get("session_index", 0)),
            str(row.get("pref_type", "")),
        )
        if key not in grouped:
            grouped[key] = {
                "run_tag": str(row.get("run_tag", "")),
                "action_model": str(row.get("action_model", "")),
                "cohort": str(row.get("cohort", "")),
                "session_index": int(row.get("session_index", 0)),
                "pref_type": str(row.get("pref_type", "")),
                "eligible_count": 0,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "parsing_fail_count": 0,
            }

        group = grouped[key]
        group["eligible_count"] += int(row.get("eligible_count", 0) or 0)
        group["tp"] += int(row.get("tp", 0) or 0)
        group["fp"] += int(row.get("fp", 0) or 0)
        group["fn"] += int(row.get("fn", 0) or 0)
        group["parsing_fail_count"] += int(row.get("parsing_fail_count", 0) or 0)

    aggregated_rows: List[Dict[str, Any]] = []
    for group in grouped.values():
        tp = group["tp"]
        fp = group["fp"]
        fn = group["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        aggregated_rows.append(
            {
                **group,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    aggregated_rows.sort(
        key=lambda row: (
            str(row.get("run_tag", "")),
            str(row.get("pref_type", "")),
            str(row.get("action_model", "")),
            int(row.get("session_index", 0)),
        )
    )
    return aggregated_rows


def plot_metric(
    rows: List[Dict[str, Any]],
    metric_key: str,
    ylabel: str,
    output_path: Path,
    compact_legend: bool = False,
) -> None:
    run_tag = rows[0].get("run_tag", "run")
    action_model = rows[0].get("action_model", "action_model")
    cohort = rows[0].get("cohort", "cohort")
    memory_models = sorted({row["memory_model"] for row in rows})

    fig_height = 5.8 if compact_legend else 5.0
    fig, axes = plt.subplots(1, 3, figsize=(18, fig_height), sharey=(metric_key in {"f1", "precision", "recall"}))
    for ax, pref_type in zip(axes, PREF_ORDER):
        pref_rows = [row for row in rows if row["pref_type"] == pref_type]
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
                (row for row in pref_rows if row["memory_model"] == memory_model),
                key=lambda row: row["session_index"],
            )
            if not series:
                continue
            x_values = [row["session_index"] for row in series]
            y_values = [row[metric_key] for row in series]
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2,
                markersize=5,
                label=display_model_label(memory_model, shorten=compact_legend),
            )

    title_action_model = action_model.split("/")[-1] if compact_legend else action_model
    fig.suptitle(f"{title_action_model} | {cohort} | {ylabel} by session\nRun: {run_tag}", y=0.98)
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

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_aggregate_metric(
    rows: List[Dict[str, Any]],
    metric_key: str,
    ylabel: str,
    output_path: Path,
    compact_legend: bool = False,
) -> None:
    run_tag = rows[0].get("run_tag", "run")
    cohort = rows[0].get("cohort", "cohort")
    action_models = sorted({str(row["action_model"]) for row in rows})

    fig_height = 5.8 if compact_legend else 5.0
    fig, axes = plt.subplots(1, 3, figsize=(18, fig_height), sharey=(metric_key in {"f1", "precision", "recall"}))
    for ax, pref_type in zip(axes, PREF_ORDER):
        pref_rows = [row for row in rows if row["pref_type"] == pref_type]
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
                (row for row in pref_rows if row["action_model"] == action_model),
                key=lambda row: row["session_index"],
            )
            if not series:
                continue
            x_values = [row["session_index"] for row in series]
            y_values = [row[metric_key] for row in series]
            ax.plot(
                x_values,
                y_values,
                marker="o",
                linewidth=2,
                markersize=5,
                label=display_action_label(action_model, shorten=compact_legend),
            )

    fig.suptitle(f"Memory models aggregated | {cohort} | {ylabel} by session\nRun: {run_tag}", y=0.98)
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

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--plot_dir", type=str, default=str(DEFAULT_PLOT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_summary_rows(args.summary_csv)
    grouped = group_rows(rows)
    aggregate_groups = group_rows_for_aggregate(rows)
    plot_dir = Path(args.plot_dir)
    saved_files: List[Path] = []

    for (run_tag, action_model, cohort), group_rows_for_plot in grouped.items():
        stem = f"{safe_name(run_tag)}__{safe_name(action_model)}__{safe_name(cohort)}"
        f1_path = plot_dir / "f1" / f"{stem}__f1.png"
        f1_legend_safe_path = plot_dir / "f1_legend_safe" / f"{stem}__f1_legend_safe.png"
        precision_path = plot_dir / "precision" / f"{stem}__precision.png"
        recall_path = plot_dir / "recall" / f"{stem}__recall.png"
        fail_path = plot_dir / "parsing_fail_count" / f"{stem}__parsing_fail_count.png"
        plot_metric(group_rows_for_plot, "f1", "F1", f1_path)
        plot_metric(group_rows_for_plot, "f1", "F1", f1_legend_safe_path, compact_legend=True)
        plot_metric(group_rows_for_plot, "precision", "Precision", precision_path)
        plot_metric(group_rows_for_plot, "recall", "Recall", recall_path)
        plot_metric(group_rows_for_plot, "parsing_fail_count", "Parsing fail count", fail_path)
        saved_files.extend([f1_path, f1_legend_safe_path, precision_path, recall_path, fail_path])

    for (run_tag, cohort), group_rows_for_plot in aggregate_groups.items():
        aggregate_rows = aggregate_rows_across_memory_models(group_rows_for_plot)
        f1_path = plot_dir / "f1" / f"aggregate_by_action_model__{safe_name(run_tag)}__{safe_name(cohort)}__f1.png"
        f1_legend_safe_path = (
            plot_dir
            / "f1_legend_safe"
            / f"aggregate_by_action_model__{safe_name(run_tag)}__{safe_name(cohort)}__f1_legend_safe.png"
        )
        precision_path = (
            plot_dir / "precision" / f"aggregate_by_action_model__{safe_name(run_tag)}__{safe_name(cohort)}__precision.png"
        )
        recall_path = (
            plot_dir / "recall" / f"aggregate_by_action_model__{safe_name(run_tag)}__{safe_name(cohort)}__recall.png"
        )
        fail_path = (
            plot_dir
            / "parsing_fail_count"
            / f"aggregate_by_action_model__{safe_name(run_tag)}__{safe_name(cohort)}__parsing_fail_count.png"
        )
        plot_aggregate_metric(aggregate_rows, "f1", "F1", f1_path)
        plot_aggregate_metric(aggregate_rows, "f1", "F1", f1_legend_safe_path, compact_legend=True)
        plot_aggregate_metric(aggregate_rows, "precision", "Precision", precision_path)
        plot_aggregate_metric(aggregate_rows, "recall", "Recall", recall_path)
        plot_aggregate_metric(aggregate_rows, "parsing_fail_count", "Parsing fail count", fail_path)
        saved_files.extend([f1_path, f1_legend_safe_path, precision_path, recall_path, fail_path])

    print(f"[DONE] Saved {len(saved_files)} plot files to {plot_dir}")
    for path in saved_files:
        print(f"[PLOT] {path}")


if __name__ == "__main__":
    main()
