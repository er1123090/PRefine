from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot session-by-session LangMem stored memory token growth directly "
            "from snapshot JSONL files."
        )
    )
    parser.add_argument(
        "--snapshots_root",
        type=str,
        default="/data/minseo/experiments6/langmem/memory_snapshots",
        help="Root directory that contains one subdirectory per model snapshot.",
    )
    parser.add_argument(
        "--snapshot_name",
        type=str,
        default="langmem_1229_dev_6.jsonl",
        help="Snapshot filename to read from each model directory.",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=(
            "/data/minseo/experiments6/langmem/memory_snapshots/"
            "langmem_1229_dev_6_snapshot_token_growth_by_model.png"
        ),
        help="Where to save the output plot.",
    )
    parser.add_argument(
        "--output_summary_csv",
        type=str,
        default=(
            "/data/minseo/experiments6/langmem/memory_snapshots/"
            "langmem_1229_dev_6_snapshot_token_growth_by_model.summary.csv"
        ),
        help="Where to save the per-session summary CSV.",
    )
    return parser.parse_args()


def prettify_model_name(model_name: str) -> str:
    if "_" not in model_name:
        return model_name
    prefix, suffix = model_name.split("_", 1)
    if prefix in {"Qwen", "google", "deepseek-ai"}:
        return suffix
    return model_name


def discover_snapshot_paths(root: Path, snapshot_name: str) -> List[Path]:
    return sorted(path for path in root.glob(f"*/{snapshot_name}") if path.is_file())


def load_snapshot_summaries(
    snapshot_paths: List[Path],
) -> Tuple[List[Dict[str, float | int | str]], Dict[int, float]]:
    per_model_session: Dict[str, Dict[int, Dict[str, List[int]]]] = defaultdict(
        lambda: defaultdict(lambda: {"before": [], "after": [], "delta": []})
    )
    unique_dialogue_tokens: Dict[Tuple[str, int], int] = {}

    for snapshot_path in snapshot_paths:
        model_name = snapshot_path.parent.name
        with snapshot_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                example_id = str(record.get("example_id", ""))
                prev_memory_tokens = 0
                cumulative_dialogue_tokens = 0

                for export in record.get("session_exports") or []:
                    if export.get("status") != "OK":
                        continue

                    session_index = int(export["session_index"])
                    after_memory_tokens = export.get(
                        "stored_memory_tokens_after_session"
                    )
                    if after_memory_tokens in (None, ""):
                        continue
                    after_memory_tokens = int(after_memory_tokens)

                    before_memory_tokens = prev_memory_tokens
                    delta_memory_tokens = after_memory_tokens - before_memory_tokens
                    prev_memory_tokens = after_memory_tokens

                    bucket = per_model_session[model_name][session_index]
                    bucket["before"].append(before_memory_tokens)
                    bucket["after"].append(after_memory_tokens)
                    bucket["delta"].append(delta_memory_tokens)

                    session_input_tokens = export.get("session_input_tokens")
                    if session_input_tokens not in (None, ""):
                        cumulative_dialogue_tokens += int(session_input_tokens)
                        unique_dialogue_tokens.setdefault(
                            (example_id, session_index),
                            cumulative_dialogue_tokens,
                        )

    dialogue_by_session: Dict[int, float] = {}
    raw_dialogue_by_session: Dict[int, List[int]] = defaultdict(list)
    for (_, session_index), token_count in unique_dialogue_tokens.items():
        raw_dialogue_by_session[session_index].append(token_count)

    for session_index, values in raw_dialogue_by_session.items():
        if values:
            dialogue_by_session[session_index] = float(np.mean(values))

    summary_rows: List[Dict[str, float | int | str]] = []
    for model_name in sorted(per_model_session):
        for session_index in sorted(per_model_session[model_name]):
            bucket = per_model_session[model_name][session_index]
            after_values = bucket["after"]
            before_values = bucket["before"]
            delta_values = bucket["delta"]
            dialogue_tokens = dialogue_by_session.get(session_index, np.nan)
            avg_after_memory_tokens = float(np.mean(after_values))
            ratio = (
                avg_after_memory_tokens / dialogue_tokens
                if dialogue_tokens not in (None, 0) and not np.isnan(dialogue_tokens)
                else np.nan
            )
            summary_rows.append(
                {
                    "model_name": model_name,
                    "model_label": prettify_model_name(model_name),
                    "snapshot_path": str(
                        next(
                            path
                            for path in snapshot_paths
                            if path.parent.name == model_name
                        )
                    ),
                    "session_index": session_index,
                    "n_examples": len(after_values),
                    "avg_before_memory_tokens": float(np.mean(before_values)),
                    "avg_after_memory_tokens": avg_after_memory_tokens,
                    "avg_delta_memory_tokens": float(np.mean(delta_values)),
                    "median_after_memory_tokens": float(np.median(after_values)),
                    "min_after_memory_tokens": int(np.min(after_values)),
                    "max_after_memory_tokens": int(np.max(after_values)),
                    "avg_dialogue_tokens": (
                        round(dialogue_tokens, 2)
                        if not np.isnan(dialogue_tokens)
                        else ""
                    ),
                    "memory_vs_dialogue_ratio": (
                        round(float(ratio), 4) if not np.isnan(ratio) else ""
                    ),
                }
            )

    return summary_rows, dialogue_by_session


def write_summary_csv(
    output_summary_csv: Path,
    summary_rows: List[Dict[str, float | int | str]],
) -> None:
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "model_label",
        "snapshot_path",
        "session_index",
        "n_examples",
        "avg_before_memory_tokens",
        "avg_after_memory_tokens",
        "avg_delta_memory_tokens",
        "median_after_memory_tokens",
        "min_after_memory_tokens",
        "max_after_memory_tokens",
        "avg_dialogue_tokens",
        "memory_vs_dialogue_ratio",
    ]
    with output_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def plot_growth(
    output_png: Path,
    summary_rows: List[Dict[str, float | int | str]],
    dialogue_by_session: Dict[int, float],
    title_stem: str,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    rows_by_model: Dict[str, List[Dict[str, float | int | str]]] = defaultdict(list)
    for row in summary_rows:
        rows_by_model[str(row["model_name"])].append(row)

    for rows in rows_by_model.values():
        rows.sort(key=lambda row: int(row["session_index"]))

    ordered_models = sorted(
        rows_by_model,
        key=lambda model_name: (
            -float(rows_by_model[model_name][-1]["avg_after_memory_tokens"]),
            str(rows_by_model[model_name][0]["model_label"]).lower(),
        ),
    )
    sessions = sorted(
        {
            int(row["session_index"])
            for rows in rows_by_model.values()
            for row in rows
        }
    )
    x = np.array(sessions, dtype=float)
    n_models = len(ordered_models)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    top_ax, bottom_ax = axes

    if dialogue_by_session:
        dialogue_values = [dialogue_by_session.get(session, np.nan) for session in sessions]
        top_ax.plot(
            sessions,
            dialogue_values,
            marker="o",
            linewidth=2.8,
            color="black",
            label="Accumulated Dialogue Context (Matched Snapshot Examples)",
            zorder=10,
        )
        for session, value in zip(sessions, dialogue_values):
            if not np.isnan(value):
                top_ax.annotate(
                    f"{value:.2f}",
                    (session, value),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                    color="black",
                )

    colors = sns.color_palette("tab10", n_colors=max(n_models, 3))
    markers = ["s", "^", "D", "P", "X", "v", "<", ">"]
    linestyles = ["--", "-.", "-", ":", (0, (3, 1, 1, 1)), (0, (5, 2))]
    bar_width = 0.8 / max(n_models, 1)
    offsets = (np.arange(n_models) - (n_models - 1) / 2.0) * bar_width

    for idx, model_name in enumerate(ordered_models):
        rows = rows_by_model[model_name]
        row_by_session = {int(row["session_index"]): row for row in rows}
        label = str(rows[0]["model_label"])
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]
        avg_after = [
            float(row_by_session[session]["avg_after_memory_tokens"])
            if session in row_by_session
            else np.nan
            for session in sessions
        ]
        avg_delta = [
            float(row_by_session[session]["avg_delta_memory_tokens"])
            if session in row_by_session
            else np.nan
            for session in sessions
        ]

        top_ax.plot(
            sessions,
            avg_after,
            marker=marker,
            linewidth=2.4,
            linestyle=linestyle,
            color=color,
            label=f"{label} Stored Memory",
        )
        bottom_ax.bar(
            x + offsets[idx],
            avg_delta,
            width=bar_width * 0.95,
            color=color,
            alpha=0.82,
            label=label,
        )

    top_ax.set_ylabel("Average Token Count")
    top_ax.set_title(f"LangMem Snapshot Session Memory Token Growth ({title_stem})")
    top_ax.legend(fontsize=9, ncol=2)
    top_ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    bottom_ax.axhline(0, color="black", linewidth=1.0)
    bottom_ax.set_ylabel("Avg Delta Tokens")
    bottom_ax.set_xlabel("Session Index")
    bottom_ax.set_xticks(sessions)
    bottom_ax.set_xlim(min(sessions) - 0.6, max(sessions) + 0.6)
    bottom_ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    snapshots_root = Path(args.snapshots_root)
    snapshot_paths = discover_snapshot_paths(snapshots_root, args.snapshot_name)
    if not snapshot_paths:
        raise SystemExit(
            f"No snapshot files named {args.snapshot_name!r} were found under {snapshots_root}"
        )

    summary_rows, dialogue_by_session = load_snapshot_summaries(snapshot_paths)
    if not summary_rows:
        raise SystemExit("No valid session_exports with stored memory tokens were found.")

    output_png = Path(args.output_png)
    output_summary_csv = Path(args.output_summary_csv)
    write_summary_csv(output_summary_csv, summary_rows)
    plot_growth(
        output_png=output_png,
        summary_rows=summary_rows,
        dialogue_by_session=dialogue_by_session,
        title_stem=Path(args.snapshot_name).stem,
    )
    print(f"Plot saved to {output_png}")
    print(f"Summary saved to {output_summary_csv}")
    print(f"Models processed: {len({row['model_name'] for row in summary_rows})}")


if __name__ == "__main__":
    main()
