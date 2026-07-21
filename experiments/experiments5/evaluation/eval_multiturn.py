"""
Multi-turn evaluation script.

Same metrics as eval_singleturn.py but also accepts glob patterns so you can
evaluate multiple output files in one pass and aggregate results.

Usage:
    # Single file
    python evaluation/eval_multiturn.py \
        --input_path outputs/vanilla_llm/api/multiturn/.../result.json \
        --pref_list_path config/pref_list.json

    # Multiple files via glob
    python evaluation/eval_multiturn.py \
        --input_glob "outputs/vanilla_llm/api/multiturn/**/*.json" \
        --pref_list_path config/pref_list.json \
        --csv_output results_summary.csv
"""

import argparse
import csv
import glob
import json
import os
import sys
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import eval_one_file, load_pref_list


def _eval_file(path: str, pref_map) -> dict:
    overall_f1, pref_emr, nonpref_f1 = eval_one_file(path, pref_map)
    return {
        "file": path,
        "overall_P": round(overall_f1.precision, 4),
        "overall_R": round(overall_f1.recall, 4),
        "overall_F1": round(overall_f1.f1, 4),
        "pref_EM_rate": round(pref_emr.em, 4),
        "pref_EM_exact": pref_emr.em_count,
        "pref_EM_total": pref_emr.n,
        "nonpref_P": round(nonpref_f1.precision, 4),
        "nonpref_R": round(nonpref_f1.recall, 4),
        "nonpref_F1": round(nonpref_f1.f1, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-turn evaluation (single or batch).")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_path", help="Single output JSON file.")
    group.add_argument("--input_glob", help='Glob pattern, e.g. "outputs/**/*.json".')
    parser.add_argument("--pref_list_path", required=True,
                        help="JSON mapping domain -> list of preference slots.")
    parser.add_argument("--csv_output", default=None,
                        help="Optional path to write CSV summary.")
    args = parser.parse_args()

    pref_map = load_pref_list(args.pref_list_path)

    if args.input_path:
        files: List[str] = [args.input_path]
    else:
        files = sorted(glob.glob(args.input_glob, recursive=True))
        if not files:
            print(f"[Warning] No files matched: {args.input_glob}")
            return

    rows = []
    for fpath in files:
        if not os.path.exists(fpath):
            print(f"[Warning] File not found, skipping: {fpath}")
            continue
        row = _eval_file(fpath, pref_map)
        rows.append(row)

        print(f"\n{'=' * 70}")
        print(f"  File: {fpath}")
        print(f"{'=' * 70}")
        print(f"  Overall Micro-F1  : P={row['overall_P']:.4f}  "
              f"R={row['overall_R']:.4f}  F1={row['overall_F1']:.4f}")
        print(f"  Pref EM Rate      : {row['pref_EM_rate']:.4f}  "
              f"({row['pref_EM_exact']}/{row['pref_EM_total']})")
        print(f"  Non-pref Micro-F1 : P={row['nonpref_P']:.4f}  "
              f"R={row['nonpref_R']:.4f}  F1={row['nonpref_F1']:.4f}")

    if args.csv_output and rows:
        os.makedirs(os.path.dirname(args.csv_output) or ".", exist_ok=True)
        with open(args.csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV saved -> {args.csv_output}")


if __name__ == "__main__":
    main()
