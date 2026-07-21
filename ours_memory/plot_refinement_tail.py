#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_MEMORY_ROOT = Path("/data/minseo/experiments4/ours_memory/inference/0312_MEMORY1")
DEFAULT_OUTPUT_DIR = Path("/data/minseo/experiments4/ours_memory/iteration_cap_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot how little additional value comes from refinement beyond step 3."
    )
    parser.add_argument("--memory-root", type=Path, default=DEFAULT_MEMORY_ROOT)
    parser.add_argument("--label", default="0312_MEMORY1_max10")
    parser.add_argument("--cap-step", type=int, default=3)
    parser.add_argument("--max-step", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--panel", choices=["full", "right"], default="full")
    return parser.parse_args()


def discover_model_dirs(root: Path) -> Iterable[Path]:
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "_refinement_logs1.jsonl").exists():
            yield child


def load_step_statistics(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    last_step: Dict[Tuple[str, str, int], int] = {}
    first_valid: Dict[Tuple[str, str, int], int] = {}

    for model_dir in discover_model_dirs(root):
        model_name = model_dir.name
        refinement_path = model_dir / "_refinement_logs1.jsonl"
        verifier_path = model_dir / "_verifier_logs1.jsonl"

        with refinement_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                key = (model_name, obj["example_id"], int(obj["session_index"]))
                step = int(obj["step"])
                if step > last_step.get(key, 0):
                    last_step[key] = step

        with verifier_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("is_valid") is not True:
                    continue
                key = (model_name, obj["example_id"], int(obj["session_index"]))
                step = int(obj["step"])
                current = first_valid.get(key)
                if current is None or step < current:
                    first_valid[key] = step

    total_pairs = len(last_step)
    valid_pairs = len(first_valid)
    invalid_pairs = total_pairs - valid_pairs

    first_valid_counter = Counter(first_valid.values())
    last_step_counter = Counter(last_step.values())

    first_valid_rows = []
    running_valid = 0
    for step in sorted(first_valid_counter):
        count = first_valid_counter[step]
        running_valid += count
        first_valid_rows.append(
            {
                "step": step,
                "count": count,
                "share_of_total_pairs": count / total_pairs if total_pairs else 0.0,
                "cumulative_valid_count": running_valid,
                "cumulative_valid_share_of_total_pairs": running_valid / total_pairs if total_pairs else 0.0,
            }
        )

    last_step_rows = []
    for step in sorted(last_step_counter):
        count = last_step_counter[step]
        last_step_rows.append(
            {
                "step": step,
                "count": count,
                "share_of_total_pairs": count / total_pairs if total_pairs else 0.0,
            }
        )

    summary = {
        "total_pairs": total_pairs,
        "valid_pairs": valid_pairs,
        "invalid_pairs": invalid_pairs,
        "valid_rate": valid_pairs / total_pairs if total_pairs else 0.0,
    }
    return pd.DataFrame(first_valid_rows), pd.DataFrame(last_step_rows), summary


def build_category_table(
    first_valid_df: pd.DataFrame,
    summary: Dict[str, float],
    cap_step: int,
) -> pd.DataFrame:
    total_pairs = int(summary["total_pairs"])
    valid_by_step = {int(row["step"]): int(row["count"]) for _, row in first_valid_df.iterrows()}

    rows = []
    for step in range(1, cap_step + 1):
        count = valid_by_step.get(step, 0)
        rows.append(
            {
                "category": f"valid@{step}",
                "count": count,
                "share": count / total_pairs if total_pairs else 0.0,
            }
        )

    after_cap_count = sum(
        count for step, count in valid_by_step.items() if step > cap_step
    )
    rows.append(
        {
            "category": f"valid@>{cap_step}",
            "count": after_cap_count,
            "share": after_cap_count / total_pairs if total_pairs else 0.0,
        }
    )
    rows.append(
        {
            "category": "never_valid",
            "count": int(summary["invalid_pairs"]),
            "share": summary["invalid_pairs"] / total_pairs if total_pairs else 0.0,
        }
    )
    return pd.DataFrame(rows)


def plot(
    first_valid_df: pd.DataFrame,
    category_df: pd.DataFrame,
    summary: Dict[str, float],
    label: str,
    cap_step: int,
    max_step: int,
    output_png: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.6), constrained_layout=True)

    # Left: stacked horizontal bar for exact first-valid-step buckets.
    ax = axes[0]
    colors = ["#2446A8", "#4E7CF0", "#8DB2FF", "#F28E2B", "#C9CDD4"]
    left = 0.0
    legend_labels = []
    for (idx, row), color in zip(category_df.iterrows(), colors):
        width = float(row["share"])
        ax.barh([0], [width], left=left, color=color, height=0.55, edgecolor="white")
        legend_labels.append(f"{row['category']} ({row['share'] * 100:.2f}%)")
        if width >= 0.08:
            ax.text(
                left + width / 2,
                0,
                f"{row['category']}\n{row['share'] * 100:.2f}%",
                ha="center",
                va="center",
                fontsize=10,
                color="white",
                fontweight="bold",
            )
        left += width

    after_cap = category_df.loc[category_df["category"] == f"valid@>{cap_step}", "count"].iloc[0]
    total_pairs = int(summary["total_pairs"])
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Share of example-session pairs")
    ax.set_title("Where Validity Was Reached", fontsize=13, fontweight="bold")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, ec="white")
        for color in colors[: len(legend_labels)]
    ]
    ax.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=2,
        fontsize=9,
        frameon=False,
    )
    ax.text(
        0.02,
        -0.12,
        f"Only {after_cap} / {total_pairs} pairs ({after_cap / total_pairs * 100:.2f}%)\n"
        f"needed refinement beyond step {cap_step}.",
        ha="left",
        va="top",
        fontsize=11,
        transform=ax.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#FFF4E8", "edgecolor": "#F28E2B"},
    )

    # Right: cumulative valid rate by allowed refinement cap.
    ax = axes[1]
    cumulative = {int(row["step"]): float(row["cumulative_valid_share_of_total_pairs"]) for _, row in first_valid_df.iterrows()}
    xs = list(range(1, max_step + 1))
    ys = []
    running = 0.0
    for step in xs:
        running = cumulative.get(step, running)
        ys.append(running)

    cap_share = ys[cap_step - 1]
    final_share = ys[-1]
    gain_after_cap = final_share - cap_share

    ax.plot(xs, ys, marker="o", linewidth=2.5, color="#2446A8")
    ax.axvline(cap_step, color="#F28E2B", linestyle="--", linewidth=1.5)
    ax.fill_between([cap_step, max_step], [cap_share, cap_share], [final_share, final_share], color="#F28E2B", alpha=0.12)
    ax.scatter([cap_step, max_step], [cap_share, final_share], color=["#F28E2B", "#2446A8"], s=60, zorder=3)
    ax.text(cap_step + 0.12, cap_share - 0.018, f"step {cap_step}: {cap_share * 100:.2f}%", color="#B86100", fontsize=10)
    ax.text(max_step - 1.15, final_share - 0.03, f"step {max_step}: {final_share * 100:.2f}%", color="#2446A8", fontsize=10)
    ax.text(
        cap_step + 0.45,
        cap_share + max(gain_after_cap / 2, 0.003),
        f"+{gain_after_cap * 100:.2f}% valid pairs\nfrom steps {cap_step + 1}-{max_step}",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D0D7E2"},
    )
    ax.set_xlim(1, max_step)
    ax.set_ylim(0.93, 1.001)
    ax.set_xticks(xs)
    ax.set_xlabel("Allowed refinement steps")
    ax.set_ylabel("Cumulative valid share of all pairs")
    ax.set_title("Validity Saturates By Step 3", fontsize=13, fontweight="bold")

    fig.suptitle(
        f"{label}: Extra Refinement Beyond Step {cap_step} Has Very Small Payoff",
        fontsize=15,
        fontweight="bold",
    )
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_right_only(
    first_valid_df: pd.DataFrame,
    label: str,
    cap_step: int,
    max_step: int,
    output_png: Path,
) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(7.4, 5.4), constrained_layout=True)

    cumulative = {
        int(row["step"]): float(row["cumulative_valid_share_of_total_pairs"])
        for _, row in first_valid_df.iterrows()
    }
    xs = list(range(1, max_step + 1))
    ys = []
    running = 0.0
    for step in xs:
        running = cumulative.get(step, running)
        ys.append(running)

    cap_share = ys[cap_step - 1]
    final_share = ys[-1]
    gain_after_cap = final_share - cap_share

    ax.plot(xs, ys, marker="o", linewidth=2.5, color="#2446A8")
    ax.axvline(cap_step, color="#F28E2B", linestyle="--", linewidth=1.5)
    ax.fill_between(
        [cap_step, max_step],
        [cap_share, cap_share],
        [final_share, final_share],
        color="#F28E2B",
        alpha=0.12,
    )
    ax.scatter([cap_step, max_step], [cap_share, final_share], color=["#F28E2B", "#2446A8"], s=60, zorder=3)
    ax.text(cap_step + 0.12, cap_share - 0.018, f"step {cap_step}: {cap_share * 100:.2f}%", color="#B86100", fontsize=10)
    ax.text(max_step - 1.15, final_share - 0.03, f"step {max_step}: {final_share * 100:.2f}%", color="#2446A8", fontsize=10)
    ax.text(
        cap_step + 0.55,
        cap_share + max(gain_after_cap / 2, 0.003),
        f"+{gain_after_cap * 100:.2f}% valid pairs\nfrom steps {cap_step + 1}-{max_step}",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D0D7E2"},
    )
    ymin = min(min(ys) - 0.01, 0.93)
    ymax = min(1.001, max(ys) + 0.01)
    ax.set_xlim(1, max_step)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(xs)
    ax.set_xlabel("Allowed refinement steps")
    ax.set_ylabel("Cumulative valid share of all pairs")
    ax.set_title(f"{label}: Validity Saturates By Step {cap_step}", fontsize=14, fontweight="bold")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    first_valid_df, last_step_df, summary = load_step_statistics(args.memory_root)
    category_df = build_category_table(first_valid_df, summary, args.cap_step)

    prefix = args.label.replace("/", "_")
    first_valid_df.to_csv(args.output_dir / f"{prefix}_first_valid_steps.csv", index=False)
    last_step_df.to_csv(args.output_dir / f"{prefix}_last_step_distribution.csv", index=False)
    category_df.to_csv(args.output_dir / f"{prefix}_step_bucket_summary.csv", index=False)
    pd.DataFrame([summary]).to_csv(args.output_dir / f"{prefix}_step_summary.csv", index=False)

    if args.panel == "right":
        output_png = args.output_dir / f"{prefix}_cumulative_validity.png"
        plot_right_only(
            first_valid_df=first_valid_df,
            label=args.label,
            cap_step=args.cap_step,
            max_step=args.max_step,
            output_png=output_png,
        )
    else:
        output_png = args.output_dir / f"{prefix}_refinement_tail.png"
        plot(
            first_valid_df=first_valid_df,
            category_df=category_df,
            summary=summary,
            label=args.label,
            cap_step=args.cap_step,
            max_step=args.max_step,
            output_png=output_png,
        )
    print(f"[saved] {output_png}")


if __name__ == "__main__":
    main()
