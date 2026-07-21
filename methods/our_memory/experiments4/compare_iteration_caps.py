#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


DEFAULT_MEMORY_A = Path("/data/minseo/experiments4/ours_memory/inference/0312_MEMORY1")
DEFAULT_MEMORY_B = Path("/data/minseo/experiments4/ours_memory/inference/1231_MEMORY3")
DEFAULT_MULTI_A = Path(
    "/data/minseo/experiments4/ours_memory/inference/0312_MEMORY1_0317_MEM-DIAG/analysis/multiturn_performance_table.csv"
)
DEFAULT_SINGLE_A = Path(
    "/data/minseo/experiments4/ours_memory/inference/0312_MEMORY1_0317_MEM-DIAG/analysis/singleturn_performance_table.csv"
)
DEFAULT_MULTI_B = Path("/data/minseo/experiments4/evaluation/eval_results/results_memory_multiturn2.csv")
DEFAULT_SINGLE_B = Path("/data/minseo/experiments4/evaluation/eval_results/results_memory_singleturn2.csv")

MEMORY_NAME_MAP_A = {
    "Qwen_Qwen3-8B": "Qwen3-8B",
    "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill-Llama-8B",
    "google_gemma-3-12b-it": "gemma-3-12b-it",
    "gpt-4o-mini": "gpt-4o-mini",
}

MEMORY_NAME_MAP_B_SINGLE = {
    "Qwen_Qwen3-8B": "Qwen3-8B",
    "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B": "DeepSeek-R1-0528-Qwen3-8B",
    "deepseek-ai_DeepSeek-R1-Distill-Llama-8B": "DeepSeek-R1-Distill-Llama-8B",
    "google_gemma-3-12b-it": "gemma-3-12b-it",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two memory roots with different refinement caps."
    )
    parser.add_argument("--memory-a-root", type=Path, default=DEFAULT_MEMORY_A)
    parser.add_argument("--memory-b-root", type=Path, default=DEFAULT_MEMORY_B)
    parser.add_argument("--memory-a-label", default="0312_MEMORY1_max10")
    parser.add_argument("--memory-b-label", default="1231_MEMORY3_max3")
    parser.add_argument("--multi-a-csv", type=Path, default=DEFAULT_MULTI_A)
    parser.add_argument("--single-a-csv", type=Path, default=DEFAULT_SINGLE_A)
    parser.add_argument("--multi-b-csv", type=Path, default=DEFAULT_MULTI_B)
    parser.add_argument("--single-b-csv", type=Path, default=DEFAULT_SINGLE_B)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def discover_model_dirs(root: Path) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    for child in sorted(root.iterdir()):
        if child.is_dir() and (child / "_refinement_logs1.jsonl").exists():
            discovered[child.name] = child
    return discovered


def load_refinement_stats(
    root: Path, label: str, include_model_dirs: Iterable[str] | None = None
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    per_model_rows: List[Dict[str, float]] = []
    total_pairs = 0
    total_attempts = 0
    total_valid_pairs = 0
    total_valid_after3 = 0
    total_pairs_reaching_step4 = 0
    total_extra_attempts_over3 = 0

    discovered = discover_model_dirs(root)
    if include_model_dirs is None:
        model_names = sorted(discovered)
    else:
        model_names = sorted(name for name in include_model_dirs if name in discovered)

    for model_name in model_names:
        model_dir = discovered[model_name]
        refinement_path = model_dir / "_refinement_logs1.jsonl"
        verifier_path = model_dir / "_verifier_logs1.jsonl"

        last_step: Dict[Tuple[str, int], int] = {}
        first_valid: Dict[Tuple[str, int], int] = {}

        with refinement_path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                key = (obj["example_id"], int(obj["session_index"]))
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
                key = (obj["example_id"], int(obj["session_index"]))
                step = int(obj["step"])
                current = first_valid.get(key)
                if current is None or step < current:
                    first_valid[key] = step

        pair_count = len(last_step)
        attempts = sum(last_step.values())
        valid_pairs = len(first_valid)
        valid_after3 = sum(1 for value in first_valid.values() if value > 3)
        pairs_reaching_step4 = sum(1 for value in last_step.values() if value > 3)
        extra_attempts_over3 = sum(max(value - 3, 0) for value in last_step.values())

        total_pairs += pair_count
        total_attempts += attempts
        total_valid_pairs += valid_pairs
        total_valid_after3 += valid_after3
        total_pairs_reaching_step4 += pairs_reaching_step4
        total_extra_attempts_over3 += extra_attempts_over3

        per_model_rows.append(
            {
                "setting": label,
                "model_dir": model_dir.name,
                "pair_count": pair_count,
                "avg_attempts": attempts / pair_count if pair_count else 0.0,
                "valid_rate": valid_pairs / pair_count if pair_count else 0.0,
                "valid_after3_count": valid_after3,
                "valid_after3_rate": valid_after3 / pair_count if pair_count else 0.0,
                "pairs_reaching_step4_count": pairs_reaching_step4,
                "pairs_reaching_step4_rate": pairs_reaching_step4 / pair_count if pair_count else 0.0,
                "extra_attempts_over3": extra_attempts_over3,
            }
        )

    totals = {
        "setting": label,
        "total_pairs": total_pairs,
        "total_attempts": total_attempts,
        "avg_attempts": total_attempts / total_pairs if total_pairs else 0.0,
        "total_valid_pairs": total_valid_pairs,
        "valid_rate": total_valid_pairs / total_pairs if total_pairs else 0.0,
        "valid_after3_count": total_valid_after3,
        "valid_after3_rate": total_valid_after3 / total_pairs if total_pairs else 0.0,
        "pairs_reaching_step4_count": total_pairs_reaching_step4,
        "pairs_reaching_step4_rate": total_pairs_reaching_step4 / total_pairs if total_pairs else 0.0,
        "extra_attempts_over3": total_extra_attempts_over3,
        "avg_extra_attempts_over3_per_pair": (
            total_extra_attempts_over3 / total_pairs if total_pairs else 0.0
        ),
    }
    return pd.DataFrame(per_model_rows), totals


def load_multiturn_a(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).rename(
        columns={
            "tp": "tp",
            "fp": "fp",
            "fn": "fn",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        }
    )
    df["memory_model_norm"] = df["memory_model"].map(MEMORY_NAME_MAP_A)
    return df[
        [
            "memory_model_norm",
            "pref_type",
            "action_model",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "pref_em",
            "pref_em_count",
            "n_examples",
        ]
    ].copy()


def load_multiturn_b(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).rename(
        columns={
            "context": "memory_model_norm",
            "pref_type": "context_type",
            "query_turns": "pref_type",
            "model_name": "action_model",
            "overall_tp": "tp",
            "overall_fp": "fp",
            "overall_fn": "fn",
            "overall_precision": "precision",
            "overall_recall": "recall",
            "overall_f1": "f1",
        }
    )
    df = df[(df["context_type"] == "memory_api") & (df["prompt_type"] == "implicit_zs")].copy()
    return df[
        [
            "memory_model_norm",
            "pref_type",
            "action_model",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "pref_em",
            "pref_em_count",
            "n_examples",
        ]
    ].copy()


def load_singleturn_a(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).rename(
        columns={
            "tp": "tp",
            "fp": "fp",
            "fn": "fn",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        }
    )
    df["memory_model_norm"] = df["memory_model"].map(MEMORY_NAME_MAP_A)
    return df[
        [
            "memory_model_norm",
            "pref_type",
            "action_model",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "n_examples",
        ]
    ].copy()


def load_singleturn_b(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[~df["file_name"].astype(str).str.contains(r"_format\.json$", na=False)].copy()
    df = df.rename(
        columns={
            "context": "memory_model_norm",
            "pref_type": "context_type",
            "query_turns": "pref_type",
            "model_name": "action_model",
        }
    )
    df["memory_model_norm"] = df["memory_model_norm"].replace(MEMORY_NAME_MAP_B_SINGLE)
    df = df[(df["context_type"] == "memory_api") & (df["prompt_type"] == "implicit_zs")].copy()
    df = (
        df.sort_values(["memory_model_norm", "pref_type", "action_model", "file_name"])
        .drop_duplicates(["memory_model_norm", "pref_type", "action_model"], keep="first")
        .copy()
    )
    if "n_examples" not in df.columns:
        df["n_examples"] = 0
    return df[
        [
            "memory_model_norm",
            "pref_type",
            "action_model",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "n_examples",
        ]
    ].copy()


def compare_common_subset(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    include_pref_em: bool,
) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    common_mems = sorted(set(df_a["memory_model_norm"].dropna()) & set(df_b["memory_model_norm"].dropna()))
    common_actions = sorted(set(df_a["action_model"]) & set(df_b["action_model"]))

    a = df_a[
        df_a["memory_model_norm"].isin(common_mems) & df_a["action_model"].isin(common_actions)
    ].copy()
    b = df_b[
        df_b["memory_model_norm"].isin(common_mems) & df_b["action_model"].isin(common_actions)
    ].copy()

    merged = a.merge(
        b,
        on=["memory_model_norm", "pref_type", "action_model"],
        suffixes=(f"_{label_a}", f"_{label_b}"),
        how="inner",
    )
    merged["f1_diff"] = merged[f"f1_{label_a}"] - merged[f"f1_{label_b}"]
    if include_pref_em:
        merged["pref_em_diff"] = merged[f"pref_em_{label_a}"] - merged[f"pref_em_{label_b}"]

    summary = {
        "rows_compared": len(merged),
        "common_memory_models": len(common_mems),
        "common_action_models": len(common_actions),
        f"f1_row_wins_{label_a}": int((merged["f1_diff"] > 0).sum()),
        f"f1_row_wins_{label_b}": int((merged["f1_diff"] < 0).sum()),
    }

    for label in (label_a, label_b):
        tp = int(merged[f"tp_{label}"].sum())
        fp = int(merged[f"fp_{label}"].sum())
        fn = int(merged[f"fn_{label}"].sum())
        precision, recall, f1 = prf(tp, fp, fn)
        summary[f"tp_{label}"] = tp
        summary[f"fp_{label}"] = fp
        summary[f"fn_{label}"] = fn
        summary[f"precision_{label}"] = precision
        summary[f"recall_{label}"] = recall
        summary[f"f1_{label}"] = f1
        if include_pref_em:
            summary[f"pref_em_mean_{label}"] = float(merged[f"pref_em_{label}"].mean())

    per_pref_rows: List[Dict[str, float]] = []
    for pref in ("easy", "medium", "hard"):
        sub = merged[merged["pref_type"] == pref]
        if sub.empty:
            continue
        row: Dict[str, float] = {"pref_type": pref, "rows": len(sub)}
        for label in (label_a, label_b):
            tp = int(sub[f"tp_{label}"].sum())
            fp = int(sub[f"fp_{label}"].sum())
            fn = int(sub[f"fn_{label}"].sum())
            _precision, _recall, f1 = prf(tp, fp, fn)
            row[f"f1_{label}"] = f1
            if include_pref_em:
                row[f"pref_em_mean_{label}"] = float(sub[f"pref_em_{label}"].mean())
        per_pref_rows.append(row)
    return merged, summary, pd.DataFrame(per_pref_rows)


def ensure_output_dir(path: Path | None) -> Path | None:
    if path is None:
        return None
    path.mkdir(parents=True, exist_ok=True)
    return path


def print_refinement_summary(totals_a: Dict[str, float], totals_b: Dict[str, float]) -> None:
    print("## Refinement Efficiency")
    for totals in (totals_a, totals_b):
        print(
            f"- {totals['setting']}: avg_attempts={totals['avg_attempts']:.4f}, "
            f"valid_rate={totals['valid_rate']:.4%}, "
            f"valid_after3={int(totals['valid_after3_count'])}/{int(totals['total_pairs'])} "
            f"({totals['valid_after3_rate']:.4%}), "
            f"extra_attempts_over3={int(totals['extra_attempts_over3'])}"
        )
    print()


def print_comparison_block(
    title: str,
    summary: Dict[str, float],
    by_pref: pd.DataFrame,
    label_a: str,
    label_b: str,
    include_pref_em: bool,
) -> None:
    print(f"## {title}")
    print(
        f"- Common subset: {summary['rows_compared']} rows, "
        f"{summary['common_memory_models']} memory models, "
        f"{summary['common_action_models']} action models"
    )
    print(
        f"- F1: {label_a}={summary[f'f1_{label_a}']:.6f}, "
        f"{label_b}={summary[f'f1_{label_b}']:.6f}, "
        f"delta={summary[f'f1_{label_a}'] - summary[f'f1_{label_b}']:+.6f}"
    )
    print(
        f"- Row-wise wins by F1: {label_a}={summary[f'f1_row_wins_{label_a}']}, "
        f"{label_b}={summary[f'f1_row_wins_{label_b}']}"
    )
    if include_pref_em:
        print(
            f"- Mean pref_EM: {label_a}={summary[f'pref_em_mean_{label_a}']:.6f}, "
            f"{label_b}={summary[f'pref_em_mean_{label_b}']:.6f}, "
            f"delta={summary[f'pref_em_mean_{label_a}'] - summary[f'pref_em_mean_{label_b}']:+.6f}"
        )
    if not by_pref.empty:
        print("- By difficulty:")
        for _, row in by_pref.iterrows():
            line = (
                f"  - {row['pref_type']}: "
                f"F1 {label_a}={row[f'f1_{label_a}']:.6f}, "
                f"{label_b}={row[f'f1_{label_b}']:.6f}"
            )
            if include_pref_em:
                line += (
                    f"; pref_EM {label_a}={row[f'pref_em_mean_{label_a}']:.6f}, "
                    f"{label_b}={row[f'pref_em_mean_{label_b}']:.6f}"
                )
            print(line)
    print()


def main() -> None:
    args = parse_args()
    output_dir = ensure_output_dir(args.output_dir)

    label_a = args.memory_a_label
    label_b = args.memory_b_label

    common_model_dirs = sorted(
        set(discover_model_dirs(args.memory_a_root)) & set(discover_model_dirs(args.memory_b_root))
    )
    refinement_a, refinement_totals_a = load_refinement_stats(
        args.memory_a_root, label_a, include_model_dirs=common_model_dirs
    )
    refinement_b, refinement_totals_b = load_refinement_stats(
        args.memory_b_root, label_b, include_model_dirs=common_model_dirs
    )

    multi_a = load_multiturn_a(args.multi_a_csv)
    multi_b = load_multiturn_b(args.multi_b_csv)
    multi_merged, multi_summary, multi_by_pref = compare_common_subset(
        multi_a, multi_b, label_a, label_b, include_pref_em=True
    )

    single_a = load_singleturn_a(args.single_a_csv)
    single_b = load_singleturn_b(args.single_b_csv)
    single_merged, single_summary, single_by_pref = compare_common_subset(
        single_a, single_b, label_a, label_b, include_pref_em=False
    )

    print_refinement_summary(refinement_totals_a, refinement_totals_b)
    print_comparison_block("Multiturn Downstream", multi_summary, multi_by_pref, label_a, label_b, True)
    print_comparison_block("Singleturn Downstream", single_summary, single_by_pref, label_a, label_b, False)

    if output_dir is not None:
        refinement_all = pd.concat([refinement_a, refinement_b], ignore_index=True)
        pd.DataFrame([refinement_totals_a, refinement_totals_b]).to_csv(
            output_dir / "refinement_totals.csv", index=False
        )
        refinement_all.to_csv(output_dir / "refinement_by_model.csv", index=False)
        multi_merged.to_csv(output_dir / "multiturn_common_subset.csv", index=False)
        multi_by_pref.to_csv(output_dir / "multiturn_by_pref.csv", index=False)
        single_merged.to_csv(output_dir / "singleturn_common_subset.csv", index=False)
        single_by_pref.to_csv(output_dir / "singleturn_by_pref.csv", index=False)
        print(f"[saved] {output_dir}")


if __name__ == "__main__":
    main()
