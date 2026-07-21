"""
Token analysis: dialogue context accumulation vs. per-model implicit preference memory size.

Compares the average cumulative token count of raw dialogue history (baseline)
against the token count of extracted implicit preference memories across sessions,
broken down by the model that produced the memory.

Usage:
    python analysis/token_analysis_per_model.py \
        --data_path data/dev.json \
        --memory_base outputs/our_memory \
        --output_png analysis/token_evolution_per_model.png
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tiktoken


def count_tokens(text: str, enc) -> int:
    if not text:
        return 0
    return len(enc.encode(text))


def process_dialogue_data(file_path: str, enc) -> dict:
    """Return {session_index: avg_cumulative_token_count} for accumulated dialogue."""
    with open(file_path, "r") as f:
        data = json.load(f)

    session_tokens: dict[int, list] = {}
    for item in data:
        sessions = item.get("sessions", [])
        cumulative_text = ""
        for i, session in enumerate(sessions):
            session_text = "".join(
                f"{t.get('role', '')}: {t.get('message', '')}\n"
                for t in session.get("dialogue", [])
            )
            cumulative_text += session_text
            idx = i + 1
            session_tokens.setdefault(idx, []).append(count_tokens(cumulative_text, enc))

    return {idx: float(np.mean(v)) for idx, v in session_tokens.items()}


def process_memory_data(memory_base: str, enc) -> dict:
    """Return {model_name: {session_index: avg_token_count}} from memory JSONL files."""
    model_avg: dict[str, dict] = {}

    for model_dir in os.listdir(memory_base):
        memory_file = os.path.join(memory_base, model_dir, "memory_result.jsonl")
        if not os.path.exists(memory_file):
            continue

        session_tokens: dict[int, list] = {}
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
                    session_tokens.setdefault(idx, []).append(count_tokens(implicit, enc))

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
    parser.add_argument("--output_png", default="analysis/token_evolution_per_model.png")
    args = parser.parse_args()

    enc = tiktoken.get_encoding("cl100k_base")

    dialogue_avg = process_dialogue_data(args.data_path, enc)
    model_memory = process_memory_data(args.memory_base, enc)

    print("Average cumulative dialogue tokens per session:")
    for idx in sorted(dialogue_avg):
        print(f"  Session {idx}: {dialogue_avg[idx]:.2f}")

    for model, avg in model_memory.items():
        print(f"\n{model}:")
        for idx in sorted(avg):
            print(f"  Session {idx}: {avg[idx]:.2f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    x_data = sorted(dialogue_avg)
    ax1.plot(x_data, [dialogue_avg[i] for i in x_data], marker="o", color="blue")
    ax1.set_title("Dialogue Context Accumulation")
    ax1.set_xlabel("Session Index")
    ax1.set_ylabel("Avg Cumulative Token Count")
    ax1.set_xticks(x_data)
    ax1.grid(True)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(model_memory), 1)))
    for i, (model, avg) in enumerate(model_memory.items()):
        x = sorted(avg)
        ax2.plot(x, [avg[j] for j in x], marker="s", label=model, color=colors[i])

    ax2.set_title("Implicit Preference Token Count per Model")
    ax2.set_xlabel("Session Index")
    ax2.set_ylabel("Avg Implicit Pref Token Count")
    ax2.legend()
    ax2.grid(True)
    all_x = sorted({k for v in model_memory.values() for k in v})
    if all_x:
        ax2.set_xticks(all_x)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output_png) or ".", exist_ok=True)
    plt.savefig(args.output_png, dpi=300)
    print(f"\nPlot saved -> {args.output_png}")


if __name__ == "__main__":
    main()
