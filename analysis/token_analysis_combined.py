"""
Token analysis: combined plot comparing accumulated dialogue context vs.
PREFINE memory size (averaged across all step-1 models).

Supports linear and symlog y-axis scales via --symlog flag.

Usage:
    python analysis/token_analysis_combined.py \
        --data_path data/dev.json \
        --memory_base outputs/our_memory \
        --output_png analysis/token_growth_combined.png \
        [--symlog]
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tiktoken


def count_tokens(text: str, enc) -> int:
    if not text:
        return 0
    return len(enc.encode(text))


def process_dialogue_data(file_path: str, enc) -> dict:
    """Return {session_index: avg_cumulative_token_count}."""
    with open(file_path, "r") as f:
        data = json.load(f)

    session_tokens: dict[int, list] = defaultdict(list)
    for item in data:
        accumulated = ""
        for i, session in enumerate(item.get("sessions", [])):
            session_text = "".join(
                f"{t.get('role', '')}: {t.get('message', '')}\n"
                for t in session.get("dialogue", [])
            )
            accumulated += session_text
            session_tokens[i + 1].append(count_tokens(accumulated, enc))

    return {idx: float(np.mean(v)) for idx, v in session_tokens.items()}


def process_memory_data(memory_base: str, enc) -> dict:
    """Return {model_name: {session_index: avg_token_count}}."""
    model_avg: dict[str, dict] = {}

    for model_dir in os.listdir(memory_base):
        memory_file = os.path.join(memory_base, model_dir, "memory_result.jsonl")
        if not os.path.exists(memory_file):
            continue

        session_tokens: dict[int, list] = defaultdict(list)
        with open(memory_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for entry in item.get("preference_evolution_history", []):
                    idx = entry.get("session_index")
                    if idx is None:
                        continue
                    pref = entry.get("final_preference_at_session") or {}
                    implicit = pref.get("implicit_pref", "") if isinstance(pref, dict) else ""
                    session_tokens[idx].append(count_tokens(implicit, enc))

        if session_tokens:
            model_avg[model_dir] = {
                idx: float(np.mean(v)) for idx, v in session_tokens.items()
            }

    return model_avg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="data/dev.json")
    parser.add_argument("--memory_base", default="outputs/our_memory",
                        help="Directory containing per-model subdirs with memory_result.jsonl")
    parser.add_argument("--output_png", default="analysis/token_growth_combined.png")
    parser.add_argument("--symlog", action="store_true",
                        help="Use symlog y-axis scale to handle large value ranges")
    args = parser.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")

    dialogue_avg = process_dialogue_data(args.data_path, enc)
    model_memory = process_memory_data(args.memory_base, enc)

    # Aggregate memory average across all models
    all_memory_sessions: set[int] = set()
    for avg in model_memory.values():
        all_memory_sessions.update(avg.keys())

    memory_agg: dict[int, float] = {}
    for idx in all_memory_sessions:
        vals = [v[idx] for v in model_memory.values() if idx in v]
        if vals:
            memory_agg[idx] = float(np.mean(vals))

    all_sessions = sorted(
        set(dialogue_avg) | set(memory_agg) | {k for v in model_memory.values() for k in v}
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))

    dialogue_vals = [dialogue_avg.get(s, np.nan) for s in all_sessions]
    plt.plot(all_sessions, dialogue_vals, marker="o", label="Accumulated Dialogue Context",
             color="black", linewidth=3.0, zorder=10)

    memory_vals = [memory_agg.get(s, np.nan) for s in all_sessions]
    plt.plot(all_sessions, memory_vals, marker="s", label="PREFINE (Avg)",
             color="red", linewidth=4.0, linestyle="--", zorder=11)

    if args.symlog:
        plt.yscale("symlog", linthresh=50)
        yticks = [0, 10, 25, 50, 100, 500, 1000, 5000, 10000]
        plt.yticks(yticks, [str(y) for y in yticks])
        plt.ylabel("Average Token Count (Symlog Scale)", fontsize=12)
    else:
        plt.ylabel("Average Token Count", fontsize=12)

    plt.xlabel("Session Index", fontsize=12)
    plt.title("Token Growth: Accumulated Dialogue vs. PREFINE Memory", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.xticks(all_sessions)

    os.makedirs(os.path.dirname(args.output_png) or ".", exist_ok=True)
    plt.savefig(args.output_png, dpi=300)
    print(f"Plot saved -> {args.output_png}")


if __name__ == "__main__":
    main()
