#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_MEMORY_ROOT = Path("/data/minseo/experiments6/ours_memory/inference/0216_MEMORY")
DEFAULT_OUTPUT_DIR = Path("/data/minseo/experiments6/ours_memory/iteration_cap_comparison")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cumulative valid share across retries_k runs."
    )
    parser.add_argument("--memory-root", type=Path, default=DEFAULT_MEMORY_ROOT)
    parser.add_argument("--label", default="0216_MEMORY")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def retry_index(path: Path) -> int:
    match = re.fullmatch(r"retries_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Unexpected retry dir name: {path.name}")
    return int(match.group(1))


def load_retry_sweep(root: Path) -> pd.DataFrame:
    rows: List[Dict[str, float]] = []
    all_pair_counts = []

    for retry_dir in sorted(root.glob("retries_*"), key=retry_index):
        verifier_path = retry_dir / "verifier_log.jsonl"
        valid_count = 0
        pair_keys = set()
        verifier_lines = 0

        if verifier_path.exists():
            with verifier_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    verifier_lines += 1
                    obj = json.loads(line)
                    pair_keys.add((obj.get("example_id"), obj.get("session_index")))
                    if obj.get("is_valid") is True:
                        valid_count += 1

        pair_count = len(pair_keys)
        if pair_count:
            all_pair_counts.append(pair_count)

        rows.append(
            {
                "retry_cap": retry_index(retry_dir),
                "verifier_lines": verifier_lines,
                "valid_count": valid_count,
                "pair_count": pair_count,
            }
        )

    df = pd.DataFrame(rows).sort_values("retry_cap").reset_index(drop=True)
    total_pairs = max(all_pair_counts) if all_pair_counts else 0
    df["total_pairs"] = total_pairs
    df["valid_share_of_total_pairs"] = df["valid_count"] / total_pairs if total_pairs else 0.0
    return df


def plot_retry_curve(df: pd.DataFrame, label: str, output_png: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.4), constrained_layout=True)

    plot_df = df[df["pair_count"] > 0].copy()
    xs = plot_df["retry_cap"].tolist()
    ys = plot_df["valid_share_of_total_pairs"].tolist()
    if not xs:
        raise ValueError("No retries_k directories were found.")

    best_idx = int(plot_df["valid_count"].idxmax())
    saturate_retry = int(plot_df.loc[best_idx, "retry_cap"])
    saturate_share = float(plot_df.loc[best_idx, "valid_share_of_total_pairs"])

    prev_retry = max(saturate_retry - 1, xs[0])
    prev_share = float(plot_df.loc[plot_df["retry_cap"] == prev_retry, "valid_share_of_total_pairs"].iloc[0])
    gain_after_prev = saturate_share - prev_share

    ax.plot(xs, ys, marker="o", linewidth=2.5, color="#2446A8")
    ax.axvline(saturate_retry, color="#F28E2B", linestyle="--", linewidth=1.5)
    ax.scatter([saturate_retry, xs[-1]], [saturate_share, ys[-1]], color=["#F28E2B", "#2446A8"], s=60, zorder=3)
    ax.fill_between(
        [saturate_retry, xs[-1]],
        [saturate_share, saturate_share],
        [ys[-1], ys[-1]],
        color="#F28E2B",
        alpha=0.12,
    )

    ax.text(
        saturate_retry + 0.08,
        saturate_share - 0.018,
        f"retry {saturate_retry}: {saturate_share * 100:.2f}%",
        color="#B86100",
        fontsize=10,
    )
    ax.text(
        xs[-1] - 0.55,
        ys[-1] - 0.03,
        f"retry {xs[-1]}: {ys[-1] * 100:.2f}%",
        color="#2446A8",
        fontsize=10,
    )
    ax.text(
        saturate_retry + 0.28,
        saturate_share + max((ys[-1] - saturate_share) / 2, 0.003),
        f"+{(ys[-1] - saturate_share) * 100:.2f}% valid pairs\nfrom retries {saturate_retry + 1}-{xs[-1]}",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#D0D7E2"},
    )
    ax.text(
        0.50,
        0.93,
        f"Last meaningful gain: +{gain_after_prev * 100:.2f}%\nfrom retry {prev_retry} -> {saturate_retry}",
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D7DCE3"},
    )

    ymin = min(min(ys) - 0.01, 0.93)
    ymax = min(1.001, max(ys) + 0.01)
    ax.set_xlim(min(xs), max(xs))
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(xs)
    ax.set_xlabel("Allowed refinement retries")
    ax.set_ylabel("Cumulative valid share of all pairs")
    ax.set_title(f"{label}: Validity Saturates By Retry {saturate_retry}", fontsize=14, fontweight="bold")
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_retry_sweep(args.memory_root)
    prefix = args.label.replace("/", "_")
    csv_path = args.output_dir / f"{prefix}_retry_validity.csv"
    png_path = args.output_dir / f"{prefix}_retry_validity.png"
    df.to_csv(csv_path, index=False)
    plot_retry_curve(df, args.label, png_path)
    print(f"[saved] {csv_path}")
    print(f"[saved] {png_path}")


if __name__ == "__main__":
    main()
