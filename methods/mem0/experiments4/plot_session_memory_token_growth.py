from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tiktoken


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot average Mem0 stored-memory token growth by session."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Merged Mem0 session token delta CSV.",
    )
    parser.add_argument(
        "--dialogue_data_path",
        type=str,
        default="/data/minseo/experiments4/data/1229_dev_6.json",
        help="Dataset JSON used to build the accumulated dialogue baseline.",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        required=True,
        help="Where to save the output plot.",
    )
    parser.add_argument(
        "--output_summary_csv",
        type=str,
        required=True,
        help="Where to save the per-session summary CSV.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="cl100k_base",
    )
    parser.add_argument(
        "--baseline_scope",
        choices=("matched", "all", "none"),
        default="matched",
        help=(
            "How to compute the dialogue baseline: only examples present in the Mem0 CSV, "
            "the whole dataset, or skip the baseline entirely."
        ),
    )
    return parser.parse_args()


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text)) if text else 0


def load_mem0_summary(input_csv: Path) -> tuple[List[Dict[str, float | int]], Set[str]]:
    before_by_session: Dict[int, List[int]] = defaultdict(list)
    after_by_session: Dict[int, List[int]] = defaultdict(list)
    delta_by_session: Dict[int, List[int]] = defaultdict(list)
    example_ids: Set[str] = set()

    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "OK":
                continue
            session_index = int(row["session_index"]) + 1
            example_ids.add(row["example_id"])
            before_by_session[session_index].append(int(row["before_memory_tokens"]))
            after_by_session[session_index].append(int(row["after_memory_tokens"]))
            delta_by_session[session_index].append(int(row["delta_memory_tokens"]))

    summary_rows: List[Dict[str, float | int]] = []
    for session_index in sorted(after_by_session):
        before_values = before_by_session[session_index]
        after_values = after_by_session[session_index]
        delta_values = delta_by_session[session_index]
        summary_rows.append(
            {
                "session_index": session_index,
                "n_examples": len(after_values),
                "avg_before_memory_tokens": float(np.mean(before_values)),
                "avg_after_memory_tokens": float(np.mean(after_values)),
                "avg_delta_memory_tokens": float(np.mean(delta_values)),
                "median_after_memory_tokens": float(np.median(after_values)),
                "min_after_memory_tokens": int(np.min(after_values)),
                "max_after_memory_tokens": int(np.max(after_values)),
            }
        )
    return summary_rows, example_ids


def load_dialogue_baseline(
    dialogue_data_path: Path,
    enc,
    example_ids: Set[str],
    scope: str,
) -> Dict[int, float]:
    if scope == "none":
        return {}

    data = json.loads(dialogue_data_path.read_text(encoding="utf-8"))
    session_tokens: Dict[int, List[int]] = defaultdict(list)

    for item in data:
        if scope == "matched" and item.get("example_id") not in example_ids:
            continue

        accumulated_text = ""
        for session_index, session in enumerate(item.get("sessions", []), start=1):
            session_text = ""
            for turn in session.get("dialogue", []):
                role = turn.get("role", "")
                message = turn.get("message", "")
                session_text += f"{role}: {message}\n"
            accumulated_text += session_text
            session_tokens[session_index].append(count_tokens(accumulated_text, enc))

    return {
        session_index: float(np.mean(values))
        for session_index, values in session_tokens.items()
        if values
    }


def write_summary_csv(
    output_summary_csv: Path,
    summary_rows: List[Dict[str, float | int]],
    dialogue_baseline: Dict[int, float],
) -> None:
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
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
            session_index = int(row["session_index"])
            dialogue_tokens = dialogue_baseline.get(session_index)
            memory_tokens = float(row["avg_after_memory_tokens"])
            ratio = (
                memory_tokens / dialogue_tokens
                if dialogue_tokens not in (None, 0)
                else np.nan
            )
            writer.writerow(
                {
                    **row,
                    "avg_dialogue_tokens": (
                        round(dialogue_tokens, 2) if dialogue_tokens is not None else ""
                    ),
                    "memory_vs_dialogue_ratio": (
                        round(float(ratio), 4) if not np.isnan(ratio) else ""
                    ),
                }
            )


def plot_growth(
    output_png: Path,
    summary_rows: List[Dict[str, float | int]],
    dialogue_baseline: Dict[int, float],
    baseline_scope: str,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    sessions = [int(row["session_index"]) for row in summary_rows]
    avg_after = [float(row["avg_after_memory_tokens"]) for row in summary_rows]
    avg_delta = [float(row["avg_delta_memory_tokens"]) for row in summary_rows]
    counts = [int(row["n_examples"]) for row in summary_rows]
    dialogue_values = [dialogue_baseline.get(session, np.nan) for session in sessions]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    top_ax = axes[0]
    if dialogue_baseline:
        label = (
            "Accumulated Dialogue Context (Matched Examples)"
            if baseline_scope == "matched"
            else "Accumulated Dialogue Context (All Examples)"
        )
        top_ax.plot(
            sessions,
            dialogue_values,
            marker="o",
            linewidth=2.8,
            color="black",
            label=label,
        )

    top_ax.plot(
        sessions,
        avg_after,
        marker="s",
        linewidth=3.2,
        linestyle="--",
        color="red",
        label="Mem0 Stored Memory (Avg)",
    )
    top_ax.set_ylabel("Average Token Count")
    top_ax.set_title("Mem0 Session Memory Token Growth")
    top_ax.legend()
    top_ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    for session, dialogue_value in zip(sessions, dialogue_values):
        if not np.isnan(dialogue_value):
            top_ax.annotate(
                f"{dialogue_value:.2f}",
                (session, dialogue_value),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=9,
                color="black",
            )

    for session, after_value in zip(sessions, avg_after):
        top_ax.annotate(
            f"{after_value:.2f}",
            (session, after_value),
            textcoords="offset points",
            xytext=(0, -14),
            ha="center",
            fontsize=9,
            color="red",
            fontweight="bold",
        )

    bottom_ax = axes[1]
    bottom_ax.bar(
        sessions,
        avg_delta,
        color="#4C78A8",
        alpha=0.9,
        width=0.65,
        label="Average Delta Memory Tokens",
    )
    bottom_ax.plot(
        sessions,
        avg_delta,
        color="#1F3B73",
        linewidth=2.0,
        marker="o",
    )
    for session, delta, count in zip(sessions, avg_delta, counts):
        bottom_ax.text(
            session,
            delta + max(avg_delta) * 0.02,
            f"n={count}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        bottom_ax.annotate(
            f"{delta:.2f}",
            (session, delta),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            color="#1F3B73",
            fontweight="bold",
        )
    bottom_ax.set_xlabel("Session Index")
    bottom_ax.set_ylabel("Avg Delta Tokens")
    bottom_ax.set_xticks(sessions)
    bottom_ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_png = Path(args.output_png)
    output_summary_csv = Path(args.output_summary_csv)
    dialogue_data_path = Path(args.dialogue_data_path)
    enc = tiktoken.get_encoding(args.encoding)

    summary_rows, example_ids = load_mem0_summary(input_csv)
    if not summary_rows:
        raise RuntimeError(f"No successful rows found in {input_csv}")

    dialogue_baseline = load_dialogue_baseline(
        dialogue_data_path=dialogue_data_path,
        enc=enc,
        example_ids=example_ids,
        scope=args.baseline_scope,
    )
    write_summary_csv(output_summary_csv, summary_rows, dialogue_baseline)
    plot_growth(output_png, summary_rows, dialogue_baseline, args.baseline_scope)

    print(f"Plot saved to {output_png}")
    print(f"Summary saved to {output_summary_csv}")


if __name__ == "__main__":
    main()
