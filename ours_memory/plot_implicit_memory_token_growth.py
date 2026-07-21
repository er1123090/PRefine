from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tiktoken


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot session-by-session token growth for implicit-preference memories "
            "stored in ours_memory inference outputs."
        )
    )
    parser.add_argument(
        "--memory_root",
        type=str,
        required=True,
        help="Root directory containing one subdirectory per memory model.",
    )
    parser.add_argument(
        "--memory_filename",
        type=str,
        default="_memory1.jsonl",
        help="Memory JSONL filename expected under each model directory.",
    )
    parser.add_argument(
        "--dialogue_data_path",
        type=str,
        default="/data/minseo/experiments4/data/1229_dev_6.json",
        help="Dataset JSON used to compute accumulated dialogue-token baselines.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where summary CSVs and PNGs will be saved.",
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="cl100k_base",
        help="Tokenizer encoding name used to count tokens.",
    )
    parser.add_argument(
        "--title_prefix",
        type=str,
        default="Ours Memory",
        help="Prefix for plot titles and combined output filenames.",
    )
    return parser.parse_args()


def prettify_model_name(model_name: str) -> str:
    if "_" not in model_name:
        return model_name
    prefix, suffix = model_name.split("_", 1)
    if prefix in {"Qwen", "google", "deepseek-ai", "meta-llama"}:
        return suffix
    return model_name


def sanitize_filename(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def count_tokens(text: str, enc: tiktoken.Encoding) -> int:
    return len(enc.encode(text)) if text else 0


def load_dialogue_baselines(
    dialogue_data_path: Path,
    enc: tiktoken.Encoding,
    example_ids_by_model: Dict[str, Set[str]],
) -> Dict[str, Dict[int, float]]:
    data = json.loads(dialogue_data_path.read_text(encoding="utf-8"))
    dialogue_tokens_by_model: Dict[str, Dict[int, List[int]]] = {
        model_name: defaultdict(list) for model_name in example_ids_by_model
    }

    for item in data:
        example_id = str(item.get("example_id", ""))
        accumulated_text = ""
        sessions = item.get("sessions") or []

        membership = [
            model_name
            for model_name, example_ids in example_ids_by_model.items()
            if example_id in example_ids
        ]
        if not membership:
            continue

        for session_index, session in enumerate(sessions, start=1):
            session_text = ""
            for turn in session.get("dialogue", []):
                role = turn.get("role", "")
                message = turn.get("message", "")
                session_text += f"{role}: {message}\n"
            accumulated_text += session_text
            token_count = count_tokens(accumulated_text, enc)
            for model_name in membership:
                dialogue_tokens_by_model[model_name][session_index].append(token_count)

    return {
        model_name: {
            session_index: float(np.mean(values))
            for session_index, values in sessions.items()
            if values
        }
        for model_name, sessions in dialogue_tokens_by_model.items()
    }


def load_memory_summaries(
    memory_root: Path,
    memory_filename: str,
    enc: tiktoken.Encoding,
) -> Tuple[
    List[Dict[str, float | int | str]],
    Dict[str, Set[str]],
    Dict[str, Path],
]:
    per_model_session: Dict[str, Dict[int, Dict[str, List[int]]]] = defaultdict(
        lambda: defaultdict(lambda: {"after": [], "delta": []})
    )
    example_ids_by_model: Dict[str, Set[str]] = defaultdict(set)
    memory_paths: Dict[str, Path] = {}

    for memory_path in sorted(memory_root.glob(f"*/{memory_filename}")):
        if not memory_path.is_file():
            continue
        model_name = memory_path.parent.name
        memory_paths[model_name] = memory_path

        with memory_path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                example_id = str(record.get("example_id", ""))
                if example_id:
                    example_ids_by_model[model_name].add(example_id)

                prev_memory_tokens = 0
                for export in record.get("preference_evolution_history") or []:
                    session_index = int(export.get("session_index", 0))
                    if session_index <= 0:
                        continue

                    final_pref = export.get("final_preference_at_session") or {}
                    implicit_pref = ""
                    if isinstance(final_pref, dict):
                        implicit_pref = str(final_pref.get("implicit_pref", "") or "")
                    elif isinstance(final_pref, str):
                        implicit_pref = final_pref

                    after_memory_tokens = count_tokens(implicit_pref, enc)
                    delta_memory_tokens = after_memory_tokens - prev_memory_tokens
                    prev_memory_tokens = after_memory_tokens

                    bucket = per_model_session[model_name][session_index]
                    bucket["after"].append(after_memory_tokens)
                    bucket["delta"].append(delta_memory_tokens)

    summary_rows: List[Dict[str, float | int | str]] = []
    for model_name in sorted(per_model_session):
        memory_path = memory_paths[model_name]
        for session_index in sorted(per_model_session[model_name]):
            bucket = per_model_session[model_name][session_index]
            after_values = bucket["after"]
            delta_values = bucket["delta"]
            summary_rows.append(
                {
                    "model_name": model_name,
                    "model_label": prettify_model_name(model_name),
                    "memory_path": str(memory_path),
                    "session_index": session_index,
                    "n_examples": len(after_values),
                    "avg_after_memory_tokens": float(np.mean(after_values)),
                    "avg_delta_memory_tokens": float(np.mean(delta_values)),
                    "avg_dialogue_tokens": "",
                    "median_after_memory_tokens": float(np.median(after_values)),
                    "min_after_memory_tokens": int(np.min(after_values)),
                    "max_after_memory_tokens": int(np.max(after_values)),
                    "min_delta_memory_tokens": float(np.min(delta_values)),
                    "max_delta_memory_tokens": float(np.max(delta_values)),
                }
            )

    return summary_rows, dict(example_ids_by_model), memory_paths


def write_summary_csv(
    output_summary_csv: Path,
    summary_rows: List[Dict[str, float | int | str]],
) -> None:
    output_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "model_label",
        "memory_path",
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
    with output_summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)


def plot_single_model(
    output_png: Path,
    rows: List[Dict[str, float | int | str]],
    dialogue_by_session: Dict[int, float],
    title_prefix: str,
) -> None:
    output_png.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    rows = sorted(rows, key=lambda row: int(row["session_index"]))
    sessions = [int(row["session_index"]) for row in rows]
    avg_after = [float(row["avg_after_memory_tokens"]) for row in rows]
    avg_delta = [float(row["avg_delta_memory_tokens"]) for row in rows]
    counts = [int(row["n_examples"]) for row in rows]
    dialogue_values = [dialogue_by_session.get(session, np.nan) for session in sessions]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )
    top_ax, bottom_ax = axes

    if dialogue_by_session:
        top_ax.plot(
            sessions,
            dialogue_values,
            marker="o",
            linewidth=2.8,
            color="black",
            label="Accumulated Dialogue Context (Matched Examples)",
        )
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

    top_ax.plot(
        sessions,
        avg_after,
        marker="s",
        linewidth=3.2,
        linestyle="--",
        color="red",
        label="Stored Memory (Avg)",
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

    top_ax.set_ylabel("Average Token Count")
    top_ax.set_title(title_prefix)
    top_ax.legend()
    top_ax.grid(True, which="both", linestyle="--", linewidth=0.5)

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
    offset = max(max(avg_delta), 1.0) * 0.02
    for session, delta, count in zip(sessions, avg_delta, counts):
        bottom_ax.text(
            session,
            delta + offset,
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

    bottom_ax.axhline(0, color="black", linewidth=1.0)
    bottom_ax.set_xlabel("Session Index")
    bottom_ax.set_ylabel("Avg Delta Tokens")
    bottom_ax.set_xticks(sessions)
    bottom_ax.grid(True, axis="y", linestyle="--", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close(fig)


def plot_combined(
    output_png: Path,
    summary_rows: List[Dict[str, float | int | str]],
    dialogue_by_model: Dict[str, Dict[int, float]],
    title_prefix: str,
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

    dialogue_reference: Dict[int, float] = {}
    if ordered_models:
        reference_model = ordered_models[0]
        dialogue_reference = dialogue_by_model.get(reference_model, {})
    if dialogue_reference:
        dialogue_values = [dialogue_reference.get(session, np.nan) for session in sessions]
        top_ax.plot(
            sessions,
            dialogue_values,
            marker="o",
            linewidth=2.8,
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
    top_ax.set_title(title_prefix)
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

    memory_root = Path(args.memory_root)
    output_dir = Path(args.output_dir)
    enc = tiktoken.get_encoding(args.encoding)

    summary_rows, example_ids_by_model, _memory_paths = load_memory_summaries(
        memory_root=memory_root,
        memory_filename=args.memory_filename,
        enc=enc,
    )
    if not summary_rows:
        raise SystemExit(
            f"No memory summaries could be loaded from {memory_root / args.memory_filename}"
        )

    dialogue_by_model = load_dialogue_baselines(
        dialogue_data_path=Path(args.dialogue_data_path),
        enc=enc,
        example_ids_by_model=example_ids_by_model,
    )

    for row in summary_rows:
        model_name = str(row["model_name"])
        session_index = int(row["session_index"])
        dialogue_tokens = dialogue_by_model.get(model_name, {}).get(session_index, "")
        row["avg_dialogue_tokens"] = dialogue_tokens

    output_dir.mkdir(parents=True, exist_ok=True)

    rows_by_model: Dict[str, List[Dict[str, float | int | str]]] = defaultdict(list)
    for row in summary_rows:
        rows_by_model[str(row["model_name"])].append(row)

    for model_name, rows in sorted(rows_by_model.items()):
        safe_name = sanitize_filename(model_name)
        write_summary_csv(output_dir / f"{safe_name}_token_growth.summary.csv", rows)
        plot_single_model(
            output_png=output_dir / f"{safe_name}_token_growth.png",
            rows=rows,
            dialogue_by_session=dialogue_by_model.get(model_name, {}),
            title_prefix=f"{prettify_model_name(model_name)} Session Memory Token Growth",
        )

    combined_prefix = sanitize_filename(args.title_prefix.lower())
    write_summary_csv(output_dir / f"{combined_prefix}_main_token_growth_by_model.summary.csv", summary_rows)
    plot_combined(
        output_png=output_dir / f"{combined_prefix}_main_token_growth_by_model.png",
        summary_rows=summary_rows,
        dialogue_by_model=dialogue_by_model,
        title_prefix=f"{args.title_prefix} Session Memory Token Growth",
    )

    print(f"Output directory: {output_dir}")
    print(f"Models processed: {len(rows_by_model)}")


if __name__ == "__main__":
    main()
