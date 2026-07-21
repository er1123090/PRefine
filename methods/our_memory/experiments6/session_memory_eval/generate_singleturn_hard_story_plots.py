#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--focus-memory-model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--exclude-action-model",
        dest="exclude_action_models",
        action="append",
        default=[],
        help="Exclude rows for the given action model. Repeatable.",
    )
    parser.add_argument("--title-suffix", type=str, default="")
    return parser.parse_args()


def load_summary_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = dict(row)
            for key in ("session_index", "tp", "fp", "fn", "parsing_fail_count"):
                parsed[key] = int(parsed.get(key, 0) or 0)
            for key in ("precision", "recall", "f1"):
                parsed[key] = float(parsed.get(key, 0.0) or 0.0)
            rows.append(parsed)
    return rows


def shorten_memory_label(label: str) -> str:
    if "/" in label:
        label = label.split("/")[-1]
    if "_" in label:
        prefix, rest = label.split("_", 1)
        if prefix.lower() in {"qwen", "google", "deepseek-ai", "meta-llama", "openai"}:
            return rest
    return label


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate_counts(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    total = {"tp": 0, "fp": 0, "fn": 0, "parsing_fail_count": 0}
    for row in rows:
        total["tp"] += int(row["tp"])
        total["fp"] += int(row["fp"])
        total["fn"] += int(row["fn"])
        total["parsing_fail_count"] += int(row["parsing_fail_count"])
    return total


def build_overall_hard_by_session(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    by_session: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("pref_type") != "hard" or row.get("status") == "error":
            continue
        by_session[int(row["session_index"])].append(row)

    for session_index in sorted(by_session):
        counts = aggregate_counts(by_session[session_index])
        metrics = prf(counts["tp"], counts["fp"], counts["fn"])
        out.append(
            {
                "session_index": session_index,
                "hard_precision": metrics["precision"],
                "hard_recall": metrics["recall"],
                "hard_f1": metrics["f1"],
                "hard_parsing_fail_count": counts["parsing_fail_count"],
            }
        )
    return out


def build_hard_by_memory_model(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_memory_and_session: Dict[str, Dict[int, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("pref_type") != "hard" or row.get("status") == "error":
            continue
        by_memory_and_session[str(row["memory_model"])][int(row["session_index"])].append(row)

    out: Dict[str, List[Dict[str, Any]]] = {}
    for memory_model, session_rows in by_memory_and_session.items():
        series: List[Dict[str, Any]] = []
        for session_index in sorted(session_rows):
            counts = aggregate_counts(session_rows[session_index])
            metrics = prf(counts["tp"], counts["fp"], counts["fn"])
            series.append(
                {
                    "session_index": session_index,
                    "hard_precision": metrics["precision"],
                    "hard_recall": metrics["recall"],
                    "hard_f1": metrics["f1"],
                    "hard_parsing_fail_count": counts["parsing_fail_count"],
                }
            )
        out[memory_model] = series
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_hard_f1_by_memory_model(
    series_by_memory: Dict[str, List[Dict[str, Any]]],
    output_path: Path,
    title_suffix: str,
) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(10, 6))

    for memory_model in sorted(series_by_memory):
        series = series_by_memory[memory_model]
        x = [row["session_index"] for row in series]
        y = [row["hard_f1"] for row in series]
        ax.plot(x, y, marker="o", linewidth=2, markersize=5, label=shorten_memory_label(memory_model))

    ax.set_xlabel("Session index")
    ax.set_ylabel("Hard F1")
    ax.set_ylim(0.0, 0.22)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    title = "Singleturn Hard F1 by Memory Generator"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_overall_vs_focus(
    overall_series: List[Dict[str, Any]],
    focus_series: List[Dict[str, Any]],
    focus_memory_model: str,
    output_path: Path,
    title_suffix: str,
) -> None:
    ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(10, 6))

    x1 = [row["session_index"] for row in overall_series]
    y1 = [row["hard_f1"] for row in overall_series]
    x2 = [row["session_index"] for row in focus_series]
    y2 = [row["hard_f1"] for row in focus_series]

    ax.plot(x1, y1, marker="o", linewidth=2, markersize=5, label="Overall aggregate")
    ax.plot(x2, y2, marker="o", linewidth=2, markersize=5, label=f"{focus_memory_model} aggregate")

    ax.set_xlabel("Session index")
    ax.set_ylabel("Hard F1")
    ax.set_ylim(0.0, 0.22)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    title = "Singleturn Hard F1: Overall vs Best Memory Generator"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_focus_breakdown(
    focus_series: List[Dict[str, Any]],
    focus_memory_model: str,
    output_path: Path,
    title_suffix: str,
) -> None:
    ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    x = [row["session_index"] for row in focus_series]
    precision = [row["hard_precision"] for row in focus_series]
    recall = [row["hard_recall"] for row in focus_series]
    fail = [row["hard_parsing_fail_count"] for row in focus_series]

    axes[0].plot(x, precision, marker="o", linewidth=2, markersize=5)
    axes[0].set_title("Hard Precision")
    axes[0].set_ylim(0.0, 0.22)

    axes[1].plot(x, recall, marker="o", linewidth=2, markersize=5)
    axes[1].set_title("Hard Recall")
    axes[1].set_ylim(0.0, 0.22)

    axes[2].plot(x, fail, marker="o", linewidth=2, markersize=5)
    axes[2].set_title("Hard Parsing Fail Count")

    for ax in axes:
        ax.set_xlabel("Session index")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    axes[0].set_ylabel("Score")
    axes[2].set_ylabel("Count")

    title = f"Singleturn Hard Breakdown for {focus_memory_model}"
    if title_suffix:
        title += f"\n{title_suffix}"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    rows = load_summary_rows(Path(args.summary_csv))
    excluded = {str(model) for model in args.exclude_action_models}
    if excluded:
        rows = [row for row in rows if str(row.get("action_model", "")) not in excluded]

    overall_series = build_overall_hard_by_session(rows)
    series_by_memory = build_hard_by_memory_model(rows)
    if args.focus_memory_model not in series_by_memory:
        raise ValueError(f"Focus memory model not found in summary rows: {args.focus_memory_model}")

    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    hard_by_memory_path = out_dir / "singleturn_hard_f1_by_memory_model.png"
    overall_vs_focus_path = out_dir / f"singleturn_hard_overall_vs_{args.focus_memory_model}_memory.png"
    focus_breakdown_path = out_dir / f"singleturn_hard_{args.focus_memory_model}_breakdown.png"

    plot_hard_f1_by_memory_model(series_by_memory, hard_by_memory_path, args.title_suffix)
    plot_overall_vs_focus(
        overall_series,
        series_by_memory[args.focus_memory_model],
        args.focus_memory_model,
        overall_vs_focus_path,
        args.title_suffix,
    )
    plot_focus_breakdown(
        series_by_memory[args.focus_memory_model],
        args.focus_memory_model,
        focus_breakdown_path,
        args.title_suffix,
    )

    print(f"[OUTPUT] {hard_by_memory_path}")
    print(f"[OUTPUT] {overall_vs_focus_path}")
    print(f"[OUTPUT] {focus_breakdown_path}")


if __name__ == "__main__":
    main()
