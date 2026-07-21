#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DIR = Path("/data/minseo/experiments4/ours_memory/iteration_cap_comparison")
DEFAULT_MULTI_BY_PREF = DEFAULT_DIR / "multiturn_by_pref.csv"
DEFAULT_SINGLE_BY_PREF = DEFAULT_DIR / "singleturn_by_pref.csv"
DEFAULT_MULTI_COMMON = DEFAULT_DIR / "multiturn_common_subset.csv"
DEFAULT_SINGLE_COMMON = DEFAULT_DIR / "singleturn_common_subset.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot how small the performance gap is between 0312 and 1231 memory settings."
    )
    parser.add_argument("--multi-by-pref", type=Path, default=DEFAULT_MULTI_BY_PREF)
    parser.add_argument("--single-by-pref", type=Path, default=DEFAULT_SINGLE_BY_PREF)
    parser.add_argument("--multi-common", type=Path, default=DEFAULT_MULTI_COMMON)
    parser.add_argument("--single-common", type=Path, default=DEFAULT_SINGLE_COMMON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--label-a", default="0312 max10")
    parser.add_argument("--label-b", default="1231 max3")
    return parser.parse_args()


def detect_f1_columns(df: pd.DataFrame) -> Tuple[str, str]:
    cols = [col for col in df.columns if col.startswith("f1_")]
    if len(cols) != 2:
        raise ValueError(f"Expected exactly 2 f1 columns, found {cols}")
    return cols[0], cols[1]


def detect_suffixes_from_f1_columns(df: pd.DataFrame) -> Tuple[str, str]:
    f1_a_col, f1_b_col = detect_f1_columns(df)
    return f1_a_col[len("f1_") :], f1_b_col[len("f1_") :]


def f1_from_counts(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def build_summary_rows(
    by_pref_df: pd.DataFrame,
    common_df: pd.DataFrame,
    task_name: str,
    suffix_a: str,
    suffix_b: str,
) -> pd.DataFrame:
    f1_a_col, f1_b_col = detect_f1_columns(by_pref_df)
    tp_a_col = f"tp_{suffix_a}"
    fp_a_col = f"fp_{suffix_a}"
    fn_a_col = f"fn_{suffix_a}"
    tp_b_col = f"tp_{suffix_b}"
    fp_b_col = f"fp_{suffix_b}"
    fn_b_col = f"fn_{suffix_b}"

    rows: List[Dict[str, float]] = []
    tp_a = int(common_df[tp_a_col].sum())
    fp_a = int(common_df[fp_a_col].sum())
    fn_a = int(common_df[fn_a_col].sum())
    tp_b = int(common_df[tp_b_col].sum())
    fp_b = int(common_df[fp_b_col].sum())
    fn_b = int(common_df[fn_b_col].sum())
    rows.append(
        {
            "task": task_name,
            "subset": "overall",
            f"f1_{suffix_a}": f1_from_counts(tp_a, fp_a, fn_a),
            f"f1_{suffix_b}": f1_from_counts(tp_b, fp_b, fn_b),
        }
    )
    for _, row in by_pref_df.iterrows():
        rows.append(
            {
                "task": task_name,
                "subset": str(row["pref_type"]),
                f"f1_{suffix_a}": float(row[f1_a_col]),
                f"f1_{suffix_b}": float(row[f1_b_col]),
            }
        )
    summary_df = pd.DataFrame(rows)
    summary_df["delta"] = summary_df[f"f1_{suffix_a}"] - summary_df[f"f1_{suffix_b}"]
    return summary_df


def build_delta_stats(
    common_df: pd.DataFrame,
    task_name: str,
) -> Dict[str, float]:
    delta = common_df["f1_diff"].astype(float)
    return {
        "task": task_name,
        "rows": len(delta),
        "mean_delta": float(delta.mean()),
        "median_delta": float(delta.median()),
        "mean_abs_delta": float(delta.abs().mean()),
        "share_within_0.01": float((delta.abs() <= 0.01).mean()),
        "share_within_0.02": float((delta.abs() <= 0.02).mean()),
        "share_within_0.03": float((delta.abs() <= 0.03).mean()),
    }


def plot_dumbbell(
    ax: plt.Axes,
    summary_df: pd.DataFrame,
    suffix_a: str,
    suffix_b: str,
    label_a: str,
    label_b: str,
    color_a: str,
    color_b: str,
) -> None:
    order = [
        ("multiturn", "overall"),
        ("multiturn", "easy"),
        ("multiturn", "medium"),
        ("multiturn", "hard"),
        ("singleturn", "overall"),
        ("singleturn", "easy"),
        ("singleturn", "medium"),
        ("singleturn", "hard"),
    ]
    order_map = {key: idx for idx, key in enumerate(order)}
    summary_df = summary_df.copy()
    summary_df["ypos"] = summary_df.apply(lambda row: order_map[(row["task"], row["subset"])], axis=1)
    summary_df = summary_df.sort_values("ypos")

    for _, row in summary_df.iterrows():
        y = row["ypos"]
        x_a = row[f"f1_{suffix_a}"]
        x_b = row[f"f1_{suffix_b}"]
        ax.plot([x_b, x_a], [y, y], color="#B7BFCC", linewidth=2.4, zorder=1)
        ax.scatter([x_b], [y], color=color_b, s=62, zorder=3)
        ax.scatter([x_a], [y], color=color_a, s=62, zorder=3)
        ax.text(
            max(x_a, x_b) + 0.012,
            y,
            f"{row['delta']:+.3f}",
            va="center",
            ha="left",
            fontsize=9.5,
            color="#444444",
        )

    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(
        [
            "Multi overall",
            "Multi easy",
            "Multi medium",
            "Multi hard",
            "Single overall",
            "Single easy",
            "Single medium",
            "Single hard",
        ]
    )
    ax.invert_yaxis()
    ax.set_xlim(0.08, 0.66)
    ax.set_xlabel("F1 on common comparison subset")
    ax.set_title("Task-Level F1 Is Very Similar", fontsize=13, fontweight="bold")
    ax.axhspan(-0.5, 3.5, color="#F4F7FB", zorder=0)
    ax.text(0.085, 3.8, "Multiturn", fontsize=10, color="#666666")
    ax.text(0.085, 7.8, "Singleturn", fontsize=10, color="#666666")
    ax.legend(
        [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_a, markersize=8),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_b, markersize=8),
        ],
        [label_a, label_b],
        loc="lower right",
        frameon=False,
    )


def plot_delta_strip(
    ax: plt.Axes,
    multi_df: pd.DataFrame,
    single_df: pd.DataFrame,
    multi_stats: Dict[str, float],
    single_stats: Dict[str, float],
    color_multi: str,
    color_single: str,
) -> None:
    rng = np.random.default_rng(0)
    ax.axvspan(-0.02, 0.02, color="#EAF3EA", alpha=1.0, zorder=0)
    ax.axvline(0.0, color="#5D6672", linewidth=1.3, linestyle="--", zorder=1)

    for ypos, df, color, label in [
        (1, multi_df, color_multi, "Multiturn"),
        (0, single_df, color_single, "Singleturn"),
    ]:
        jitter = rng.normal(0, 0.055, size=len(df))
        ax.scatter(df["f1_diff"], ypos + jitter, color=color, s=42, alpha=0.82, edgecolors="white", linewidths=0.4)
        ax.scatter([df["f1_diff"].mean()], [ypos], color="black", s=72, marker="D", zorder=4)
        ax.text(
            0.056,
            ypos + 0.19,
            f"{label} mean |Δ| = {df['f1_diff'].abs().mean():.3f}\n"
            f"within ±0.02: {(df['f1_diff'].abs() <= 0.02).mean() * 100:.0f}%",
            ha="left",
            va="center",
            fontsize=9.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#D7DCE3"},
        )

    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Singleturn", "Multiturn"])
    ax.set_xlim(-0.06, 0.10)
    ax.set_xlabel("Row-level ΔF1 (0312 max10 - 1231 max3)")
    ax.set_title("Most Row-Level Gaps Stay Near Zero", fontsize=13, fontweight="bold")
    ax.text(-0.0195, 1.37, "near-equivalent band", fontsize=9, color="#2E6B2E")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    multi_by_pref = pd.read_csv(args.multi_by_pref)
    single_by_pref = pd.read_csv(args.single_by_pref)
    multi_common = pd.read_csv(args.multi_common)
    single_common = pd.read_csv(args.single_common)

    suffix_a, suffix_b = detect_suffixes_from_f1_columns(multi_by_pref)
    summary_multi = build_summary_rows(multi_by_pref, multi_common, "multiturn", suffix_a, suffix_b)
    summary_single = build_summary_rows(single_by_pref, single_common, "singleturn", suffix_a, suffix_b)
    summary_df = pd.concat([summary_multi, summary_single], ignore_index=True)
    summary_df.to_csv(args.output_dir / "performance_gap_summary.csv", index=False)

    multi_stats = build_delta_stats(multi_common, "multiturn")
    single_stats = build_delta_stats(single_common, "singleturn")
    pd.DataFrame([multi_stats, single_stats]).to_csv(args.output_dir / "performance_gap_delta_stats.csv", index=False)

    color_a = "#2446A8"
    color_b = "#F28E2B"
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.1), constrained_layout=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    plot_dumbbell(
        ax=axes[0],
        summary_df=summary_df,
        suffix_a=suffix_a,
        suffix_b=suffix_b,
        label_a=args.label_a,
        label_b=args.label_b,
        color_a=color_a,
        color_b=color_b,
    )
    plot_delta_strip(
        ax=axes[1],
        multi_df=multi_common,
        single_df=single_common,
        multi_stats=multi_stats,
        single_stats=single_stats,
        color_multi=color_a,
        color_single=color_b,
    )

    fig.suptitle(
        "0312 vs 1231 Memory: F1 Differences Are Small on the Common Comparison Subset",
        fontsize=15,
        fontweight="bold",
    )
    output_png = args.output_dir / "0312_vs_1231_performance_gap.png"
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {output_png}")


if __name__ == "__main__":
    main()
