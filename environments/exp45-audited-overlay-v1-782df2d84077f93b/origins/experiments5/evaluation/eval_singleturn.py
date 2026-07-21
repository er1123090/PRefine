"""
Single-turn evaluation script.

Computes:
  - Overall Micro-F1 (slot & value, OR-allowed)
  - Pref-slot Exact Match Rate
  - Non-pref-slot Micro-F1

Usage:
    python evaluation/eval_singleturn.py \
        --input_path outputs/vanilla_llm/api/singleturn/.../result.json \
        --pref_list_path config/pref_list.json
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.metrics import (
    eval_one_file,
    load_pref_list,
    micro_f1_slot_and_value_or,
    pref_em_rate,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-turn evaluation.")
    parser.add_argument("--input_path", required=True,
                        help="JSON file with model outputs (list of dicts).")
    parser.add_argument("--pref_list_path", required=True,
                        help="JSON file mapping domain -> list of preference slots.")
    args = parser.parse_args()

    if not os.path.exists(args.input_path):
        print(f"[Error] Input file not found: {args.input_path}")
        return

    pref_map = load_pref_list(args.pref_list_path)

    overall_f1, pref_emr, nonpref_f1 = eval_one_file(args.input_path, pref_map)

    print(f"\n{'=' * 60}")
    print(f"  File: {args.input_path}")
    print(f"{'=' * 60}")
    print(f"  Overall Micro-F1  : P={overall_f1.precision:.4f}  "
          f"R={overall_f1.recall:.4f}  F1={overall_f1.f1:.4f}")
    print(f"  Pref EM Rate      : {pref_emr.em:.4f}  "
          f"({pref_emr.em_count}/{pref_emr.n})")
    print(f"  Non-pref Micro-F1 : P={nonpref_f1.precision:.4f}  "
          f"R={nonpref_f1.recall:.4f}  F1={nonpref_f1.f1:.4f}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
