from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tiktoken


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot LangMem session-by-session stored/retrieved memory token growth."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="/data/minseo/experiments4/langmem/session_memory_token_deltas_1229_dev_6.csv",
    )
    parser.add_argument(
        "--dialogue_data_path",
        type=str,
        default="/data/minseo/experiments4/data/1229_dev_6.json",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default="/data/minseo/experiments4/langmem/session_memory_token_growth.png",
    )
    parser.add_argument(
        "--output_summary_csv",
        type=str,
        default="/data/minseo/experiments4/langmem/session_memory_token_growth.summary.csv",
    )
    parser.add_argument("--encoding", type=str, default="cl100k_base")
    parser.add_argument(
        "--skip_dialogue_baseline",
        action="store_true",
        help="Do not plot accumulated dialogue token baseline.",
    )
    return parser.parse_args()


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text)) if text else 0


def load_dialogue_baseline(path: str, enc) -> Dict[int, float]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    session_tokens = defaultdict(list)
    for example in data:
        accumulated_text = ""
        for idx, session in enumerate(example.get("sessions", []), start=1):
            session_text = ""
            for turn in session.get("dialogue", []):
                role = turn.get("role", "")
                message = turn.get("message", "")
                session_text += f"{role}: {message}\n"
            accumulated_text += session_text
            session_tokens[idx].append(count_tokens(accumulated_text, enc))
    return {
        session_index: float(np.mean(values))
        for session_index, values in session_tokens.items()
        if values
    }


def main() -> None:
    args = parse_args()
    enc = tiktoken.get_encoding(args.encoding)
    input_csv = Path(args.input_csv)
    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_summary_csv = Path(args.output_summary_csv)
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)

    stored_by_session = defaultdict(list)
    retrieved_by_session = defaultdict(list)
    delta_stored_by_session = defaultdict(list)
    delta_retrieved_by_session = defaultdict(list)

    with input_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "OK":
                continue
            session_index = int(row["session_index"])
            stored_by_session[session_index].append(
                int(row["after_stored_memory_tokens"])
            )
            retrieved_by_session[session_index].append(
                int(row["after_retrieved_memory_tokens"])
            )
            delta_stored_by_session[session_index].append(
                int(row["delta_stored_memory_tokens"])
            )
            delta_retrieved_by_session[session_index].append(
                int(row["delta_retrieved_memory_tokens"])
            )

    summary_rows: List[Dict[str, float | int]] = []
    all_sessions = sorted(
        set(stored_by_session.keys())
        | set(retrieved_by_session.keys())
        | set(delta_stored_by_session.keys())
        | set(delta_retrieved_by_session.keys())
    )
    for session_index in all_sessions:
        summary_rows.append(
            {
                "session_index": session_index,
                "avg_after_stored_memory_tokens": float(
                    np.mean(stored_by_session[session_index])
                )
                if stored_by_session[session_index]
                else np.nan,
                "avg_after_retrieved_memory_tokens": float(
                    np.mean(retrieved_by_session[session_index])
                )
                if retrieved_by_session[session_index]
                else np.nan,
                "avg_delta_stored_memory_tokens": float(
                    np.mean(delta_stored_by_session[session_index])
                )
                if delta_stored_by_session[session_index]
                else np.nan,
                "avg_delta_retrieved_memory_tokens": float(
                    np.mean(delta_retrieved_by_session[session_index])
                )
                if delta_retrieved_by_session[session_index]
                else np.nan,
            }
        )

    with output_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()) if summary_rows else [
            "session_index",
            "avg_after_stored_memory_tokens",
            "avg_after_retrieved_memory_tokens",
            "avg_delta_stored_memory_tokens",
            "avg_delta_retrieved_memory_tokens",
        ])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    if not args.skip_dialogue_baseline:
        dialogue_baseline = load_dialogue_baseline(args.dialogue_data_path, enc)
        dialogue_values = [dialogue_baseline.get(session, np.nan) for session in all_sessions]
        plt.plot(
            all_sessions,
            dialogue_values,
            marker="o",
            linewidth=2.5,
            color="black",
            label="Accumulated Dialogue Context",
        )

    stored_values = [
        next(
            (
                row["avg_after_stored_memory_tokens"]
                for row in summary_rows
                if row["session_index"] == session
            ),
            np.nan,
        )
        for session in all_sessions
    ]
    retrieved_values = [
        next(
            (
                row["avg_after_retrieved_memory_tokens"]
                for row in summary_rows
                if row["session_index"] == session
            ),
            np.nan,
        )
        for session in all_sessions
    ]

    plt.plot(
        all_sessions,
        stored_values,
        marker="s",
        linewidth=3.0,
        linestyle="--",
        color="red",
        label="LangMem Stored Memory",
    )
    plt.plot(
        all_sessions,
        retrieved_values,
        marker="^",
        linewidth=3.0,
        linestyle="-.",
        color="blue",
        label="LangMem Retrieved Memory",
    )

    plt.xlabel("Session Index")
    plt.ylabel("Average Token Count")
    plt.title("LangMem Session Memory Token Growth")
    plt.xticks(all_sessions)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    print(f"Plot saved to {output_png}")
    print(f"Summary saved to {output_summary_csv}")


if __name__ == "__main__":
    main()
