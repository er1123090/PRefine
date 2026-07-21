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
            for key in (
                "session_index",
                "eligible_count",
                "files_in_group",
                "ok_files",
                "error_files",
                "overall_tp",
                "overall_fp",
                "overall_fn",
                "pref_em_count",
                "n_examples",
                "nonpref_tp",
                "nonpref_fp",
                "nonpref_fn",
                "parsing_fail_count",
            ):
                parsed[key] = int(parsed.get(key, 0) or 0)
            for key in (
                "overall_precision",
                "overall_recall",
                "overall_f1",
                "pref_em",
                "nonpref_precision",
                "nonpref_recall",
                "nonpref_f1",
            ):
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

    bounded_metrics = {
        "overall_precision",
        "overall_recall",
        "overall_f1",
        "pref_em",
        "nonpref_precision",
        "nonpref_recall",
        "nonpref_f1",
    }

    fig_height = 5.8 if compact_legend else 5.0
    fig, axes = plt.subplots(1, 3, figsize=(18, fig_height), sharey=(metric_key in bounded_metrics))
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
        if metric_key in bounded_metrics:
            ax.set_ylim(0.0, 1.0)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--plot_dir", type=str, default=str(DEFAULT_PLOT_ROOT))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_summary_rows(args.summary_csv)
    grouped = group_rows(rows)
    plot_dir = Path(args.plot_dir)
    saved_files: List[Path] = []

    for (run_tag, action_model, cohort), group_rows_for_plot in grouped.items():
        stem = f"{safe_name(run_tag)}__{safe_name(action_model)}__{safe_name(cohort)}"
        overall_f1_path = plot_dir / "overall_f1" / f"{stem}__overall_f1.png"
        overall_precision_path = plot_dir / "overall_precision" / f"{stem}__overall_precision.png"
        overall_recall_path = plot_dir / "overall_recall" / f"{stem}__overall_recall.png"
        pref_em_path = plot_dir / "pref_em" / f"{stem}__pref_em.png"
        nonpref_f1_path = plot_dir / "nonpref_f1" / f"{stem}__nonpref_f1.png"
        fail_path = plot_dir / "parsing_fail_count" / f"{stem}__parsing_fail_count.png"
        plot_metric(group_rows_for_plot, "overall_f1", "Overall F1", overall_f1_path)
        plot_metric(group_rows_for_plot, "overall_precision", "Overall Precision", overall_precision_path)
        plot_metric(group_rows_for_plot, "overall_recall", "Overall Recall", overall_recall_path)
        plot_metric(group_rows_for_plot, "pref_em", "Preference EM", pref_em_path)
        plot_metric(group_rows_for_plot, "nonpref_f1", "Non-pref F1", nonpref_f1_path)
        plot_metric(group_rows_for_plot, "parsing_fail_count", "Parsing fail count", fail_path)
        saved_files.extend(
            [
                overall_f1_path,
                overall_precision_path,
                overall_recall_path,
                pref_em_path,
                nonpref_f1_path,
                fail_path,
            ]
        )

    print(f"[DONE] Saved {len(saved_files)} plot files to {plot_dir}")
    for path in saved_files:
        print(f"[PLOT] {path}")


if __name__ == "__main__":
    main()
