from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


SUMMARY_FIELDNAMES = [
    "model_name",
    "model_label",
    "snapshot_path",
    "session_index",
    "n_examples",
    "avg_after_memory_tokens",
    "avg_delta_memory_tokens",
    "avg_dialogue_tokens",
    "median_after_memory_tokens",
    "min_after_memory_tokens",
    "max_after_memory_tokens",
    "min_delta_memory_tokens",
    "max_delta_memory_tokens",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create combined and per-model token-growth reports directly from "
            "LangMem snapshot JSONL files."
        )
    )
    parser.add_argument(
        "--snapshots_root",
        type=str,
        default="/data/minseo/experiments5/methods/langmem/memory_snapshots/semantic-custom",
        help="Root directory containing one subdirectory per model snapshot.",
    )
    parser.add_argument(
        "--snapshot_name",
        type=str,
        default="langmem_1229_dev_6.jsonl",
        help="Snapshot filename to read from each model directory.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "/data/minseo/experiments5/methods/langmem/memory_snapshots/"
            "semantic-custom/0325_main_token_growth"
        ),
        help="Directory where combined and per-model plots/CSVs will be saved.",
    )
    parser.add_argument(
        "--title_prefix",
        type=str,
        default="Semantic-Custom Main Snapshot",
        help="Title prefix used for the combined plot.",
    )
    return parser.parse_args()


def prettify_model_name(model_name: str) -> str:
    if "_" not in model_name:
        return model_name
    prefix, suffix = model_name.split("_", 1)
    if prefix in {"Qwen", "google", "deepseek-ai"}:
        return suffix
    return model_name


def sanitize_filename(name: str) -> str:
    return "".join(
        ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in name
    )


def discover_snapshot_paths(root: Path, snapshot_name: str) -> List[Path]:
    return sorted(path for path in root.glob(f"*/{snapshot_name}") if path.is_file())


def summarize_snapshot(
    snapshot_path: Path,
) -> Tuple[List[Dict[str, float | int | str]], Dict[Tuple[str, int], int]]:
    model_name = snapshot_path.parent.name
    model_label = prettify_model_name(model_name)

    after_by_session: Dict[int, List[int]] = defaultdict(list)
    delta_by_session: Dict[int, List[int]] = defaultdict(list)
    dialogue_by_session: Dict[int, List[int]] = defaultdict(list)
    unique_dialogue_tokens: Dict[Tuple[str, int], int] = {}

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
                after_memory_tokens = export.get("stored_memory_tokens_after_session")
                if after_memory_tokens in (None, ""):
                    continue
                after_memory_tokens = int(after_memory_tokens)

                delta_memory_tokens = after_memory_tokens - prev_memory_tokens
                prev_memory_tokens = after_memory_tokens

                after_by_session[session_index].append(after_memory_tokens)
                delta_by_session[session_index].append(delta_memory_tokens)

                session_input_tokens = export.get("session_input_tokens")
                if session_input_tokens not in (None, ""):
                    cumulative_dialogue_tokens += int(session_input_tokens)
                    dialogue_by_session[session_index].append(cumulative_dialogue_tokens)
                    unique_dialogue_tokens.setdefault(
                        (example_id, session_index), cumulative_dialogue_tokens
                    )

    summary_rows: List[Dict[str, float | int | str]] = []
    all_sessions = sorted(
        set(after_by_session) | set(delta_by_session) | set(dialogue_by_session)
    )

    for session_index in all_sessions:
        after_values = after_by_session[session_index]
        delta_values = delta_by_session[session_index]
        dialogue_values = dialogue_by_session[session_index]
        summary_rows.append(
            {
                "model_name": model_name,
                "model_label": model_label,
                "snapshot_path": str(snapshot_path),
                "session_index": session_index,
                "n_examples": len(after_values),
                "avg_after_memory_tokens": float(np.mean(after_values)),
                "avg_delta_memory_tokens": float(np.mean(delta_values)),
                "avg_dialogue_tokens": (
                    float(np.mean(dialogue_values)) if dialogue_values else np.nan
                ),
                "median_after_memory_tokens": float(np.median(after_values)),
                "min_after_memory_tokens": int(np.min(after_values)),
                "max_after_memory_tokens": int(np.max(after_values)),
                "min_delta_memory_tokens": float(np.min(delta_values)),
                "max_delta_memory_tokens": float(np.max(delta_values)),
            }
        )

    return summary_rows, unique_dialogue_tokens


def write_summary_csv(
    output_csv: Path, summary_rows: List[Dict[str, float | int | str]]
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_individual_model(
    output_png: Path,
    model_label: str,
    summary_rows: List[Dict[str, float | int | str]],
) -> None:
    sns.set_theme(style="whitegrid")
    output_png.parent.mkdir(parents=True, exist_ok=True)

    summary_rows = sorted(summary_rows, key=lambda row: int(row["session_index"]))
    sessions = [int(row["session_index"]) for row in summary_rows]
    after_values = [float(row["avg_after_memory_tokens"]) for row in summary_rows]
    delta_values = [float(row["avg_delta_memory_tokens"]) for row in summary_rows]
    dialogue_values = [float(row["avg_dialogue_tokens"]) for row in summary_rows]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(13, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    top_ax, bottom_ax = axes

    top_ax.plot(
        sessions,
        dialogue_values,
        marker="o",
        linewidth=2.5,
        color="black",
        label="Accumulated Dialogue Context (Matched Examples)",
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

    top_ax.plot(
        sessions,
        after_values,
        marker="s",
        linewidth=3.0,
        linestyle="--",
        color="red",
        label=f"{model_label} Stored Memory (Avg)",
    )
    for session, value in zip(sessions, after_values):
        if not np.isnan(value):
            top_ax.annotate(
                f"{value:.2f}",
                (session, value),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=8,
                color="red",
                fontweight="bold",
            )

    bars = bottom_ax.bar(
        sessions,
        delta_values,
        width=0.6,
        color="#4C78A8",
        alpha=0.85,
        label="Average Delta Memory Tokens",
    )
    bottom_ax.plot(sessions, delta_values, marker="o", linewidth=1.8, color="#2F4B7C")
    for bar, value in zip(bars, delta_values):
        x = bar.get_x() + bar.get_width() / 2.0
        y_offset = 3 if value >= 0 else -12
        bottom_ax.annotate(
            f"{value:.2f}",
            (x, value),
            textcoords="offset points",
            xytext=(0, y_offset),
            ha="center",
            fontsize=8,
            color="#2F4B7C",
        )

    top_ax.set_ylabel("Average Token Count")
    top_ax.set_title(f"{model_label} Session Memory Token Growth")
    top_ax.legend(fontsize=10)
    top_ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    bottom_ax.axhline(0, color="black", linewidth=1.0)
    bottom_ax.set_ylabel("Avg Delta Tokens")
    bottom_ax.set_xlabel("Session Index")
    bottom_ax.set_xticks(sessions)
    bottom_ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close(fig)


def plot_combined(
    output_png: Path,
    per_model_rows: Dict[str, List[Dict[str, float | int | str]]],
    dialogue_by_session: Dict[int, float],
    title_prefix: str,
) -> None:
    sns.set_theme(style="whitegrid")
    output_png.parent.mkdir(parents=True, exist_ok=True)

    ordered_models = sorted(
        per_model_rows,
        key=lambda model_name: -float(
            sorted(
                per_model_rows[model_name], key=lambda row: int(row["session_index"])
            )[-1]["avg_after_memory_tokens"]
        ),
    )
    all_sessions = sorted(
        {
            int(row["session_index"])
            for rows in per_model_rows.values()
            for row in rows
        }
    )
    x = np.array(all_sessions, dtype=float)
    n_models = len(ordered_models)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    top_ax, bottom_ax = axes

    dialogue_values = [dialogue_by_session.get(session, np.nan) for session in all_sessions]
    top_ax.plot(
        all_sessions,
        dialogue_values,
        marker="o",
        linewidth=2.8,
        color="black",
        label="Accumulated Dialogue Context (Matched Examples)",
        zorder=10,
    )
    for session, value in zip(all_sessions, dialogue_values):
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
    mean_after_by_session: Dict[int, List[float]] = defaultdict(list)
    mean_delta_by_session: Dict[int, List[float]] = defaultdict(list)

    for idx, model_name in enumerate(ordered_models):
        rows = sorted(per_model_rows[model_name], key=lambda row: int(row["session_index"]))
        row_by_session = {int(row["session_index"]): row for row in rows}
        label = str(rows[0]["model_label"])
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        linestyle = linestyles[idx % len(linestyles)]
        avg_after = [
            float(row_by_session[session]["avg_after_memory_tokens"])
            if session in row_by_session
            else np.nan
            for session in all_sessions
        ]
        avg_delta = [
            float(row_by_session[session]["avg_delta_memory_tokens"])
            if session in row_by_session
            else np.nan
            for session in all_sessions
        ]
        for session, value in zip(all_sessions, avg_after):
            if not np.isnan(value):
                mean_after_by_session[session].append(value)
        for session, value in zip(all_sessions, avg_delta):
            if not np.isnan(value):
                mean_delta_by_session[session].append(value)

        top_ax.plot(
            all_sessions,
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

    mean_after_values = [
        float(np.mean(mean_after_by_session[session]))
        if mean_after_by_session[session]
        else np.nan
        for session in all_sessions
    ]
    mean_delta_values = [
        float(np.mean(mean_delta_by_session[session]))
        if mean_delta_by_session[session]
        else np.nan
        for session in all_sessions
    ]

    avg_color = "#6B4C9A"
    top_ax.plot(
        all_sessions,
        mean_after_values,
        marker="o",
        linewidth=3.0,
        linestyle=":",
        color=avg_color,
        label="Mean Stored Memory Across Models",
        zorder=11,
    )
    for session, value in zip(all_sessions, mean_after_values):
        if not np.isnan(value):
            top_ax.annotate(
                f"{value:.2f}",
                (session, value),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=8,
                color=avg_color,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            )

    bottom_ax.plot(
        all_sessions,
        mean_delta_values,
        marker="o",
        linewidth=2.3,
        linestyle=":",
        color=avg_color,
        label="Mean Delta Across Models",
        zorder=12,
    )
    for session, value in zip(all_sessions, mean_delta_values):
        if not np.isnan(value):
            bottom_ax.annotate(
                f"{value:.2f}",
                (session, value),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=8,
                color=avg_color,
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                },
            )

    top_ax.set_ylabel("Average Token Count")
    top_ax.set_title(f"{title_prefix} Session Memory Token Growth")
    top_ax.legend(fontsize=9, ncol=2)
    top_ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    bottom_ax.axhline(0, color="black", linewidth=1.0)
    bottom_ax.set_ylabel("Avg Delta Tokens")
    bottom_ax.set_xlabel("Session Index")
    bottom_ax.set_xticks(all_sessions)
    bottom_ax.set_xlim(min(all_sessions) - 0.6, max(all_sessions) + 0.6)
    bottom_ax.grid(True, axis="y", linestyle="--", linewidth=0.5)
    bottom_ax.legend(fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    snapshots_root = Path(args.snapshots_root)
    output_dir = Path(args.output_dir)
    snapshot_paths = discover_snapshot_paths(snapshots_root, args.snapshot_name)

    if not snapshot_paths:
        raise SystemExit(
            f"No snapshot files named {args.snapshot_name!r} were found under {snapshots_root}"
        )

    combined_rows: List[Dict[str, float | int | str]] = []
    per_model_rows: Dict[str, List[Dict[str, float | int | str]]] = {}
    unique_dialogue_tokens: Dict[Tuple[str, int], int] = {}

    for snapshot_path in snapshot_paths:
        summary_rows, snapshot_dialogue = summarize_snapshot(snapshot_path)
        if not summary_rows:
            continue
        model_name = str(summary_rows[0]["model_name"])
        per_model_rows[model_name] = summary_rows
        combined_rows.extend(summary_rows)
        for key, value in snapshot_dialogue.items():
            unique_dialogue_tokens.setdefault(key, value)

        stem = sanitize_filename(model_name)
        write_summary_csv(output_dir / f"{stem}_token_growth.summary.csv", summary_rows)
        plot_individual_model(
            output_dir / f"{stem}_token_growth.png",
            str(summary_rows[0]["model_label"]),
            summary_rows,
        )

    if not combined_rows:
        raise SystemExit("No valid session exports with stored memory tokens were found.")

    dialogue_values_by_session: Dict[int, List[int]] = defaultdict(list)
    for (_, session_index), token_count in unique_dialogue_tokens.items():
        dialogue_values_by_session[session_index].append(token_count)
    dialogue_by_session = {
        session_index: float(np.mean(values))
        for session_index, values in dialogue_values_by_session.items()
        if values
    }

    write_summary_csv(
        output_dir / "semantic_custom_main_token_growth_by_model.summary.csv",
        sorted(combined_rows, key=lambda row: (str(row["model_name"]), int(row["session_index"]))),
    )
    plot_combined(
        output_dir / "semantic_custom_main_token_growth_by_model.png",
        per_model_rows=per_model_rows,
        dialogue_by_session=dialogue_by_session,
        title_prefix=args.title_prefix,
    )

    print(f"Report directory: {output_dir}")
    for path in sorted(output_dir.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
