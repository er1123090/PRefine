"""Per-API-argument matching audit using the experiment5 canonical parser."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (  # noqa: E402
    build_gt_allowed_map,
    build_pred_map,
    counts_slot_and_value_or,
    is_parsing_failed,
    load_json_examples,
    load_pref_list,
)


Slot = Tuple[str, str]


def render_values(values: Set[str]) -> str:
    return " | ".join(sorted(values))


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", required=True)
    parser.add_argument(
        "--pref_list_path", default=str(ROOT / "config/pref_list.json")
    )
    parser.add_argument("--gt_key", default="reference_ground_truth")
    parser.add_argument("--pred_key", default="llm_output")
    parser.add_argument("--csv_output", default=None)
    parser.add_argument("--json_output", default=None)
    args = parser.parse_args()

    rows = load_json_examples(args.input_path)
    pref_map = load_pref_list(args.pref_list_path)
    details: List[Dict[str, Any]] = []
    totals = Counter()

    for row_index, row in enumerate(rows):
        gt = build_gt_allowed_map(row.get(args.gt_key))
        pred = build_pred_map(row.get(args.pred_key))
        parsing_failed, parse_reason = is_parsing_failed(row, args.pred_key)
        tp, fp, fn = counts_slot_and_value_or(gt, pred)
        totals.update({"tp": tp, "fp": fp, "fn": fn})
        totals["rows"] += 1
        totals["parse_failures"] += int(parsing_failed)

        for domain, slot in sorted(set(gt) | set(pred)):
            allowed = gt.get((domain, slot), set())
            predicted = pred.get((domain, slot), set())
            matched = bool(allowed & predicted)
            is_pref = slot in pref_map.get(domain, set())
            kind = "pref" if is_pref else "nonpref"
            totals[f"{kind}_slots"] += 1
            totals[f"{kind}_matched"] += int(matched)
            details.append(
                {
                    "row": row_index,
                    "example_id": row.get("example_id"),
                    "example_id_sub": row.get("example_id_sub"),
                    "domain": domain,
                    "slot": slot,
                    "is_preference_slot": is_pref,
                    "ground_truth_values": render_values(allowed),
                    "predicted_values": render_values(predicted),
                    "matched": matched,
                    "parse_failed": parsing_failed,
                    "parse_reason": parse_reason,
                }
            )

    precision = safe_ratio(totals["tp"], totals["tp"] + totals["fp"])
    recall = safe_ratio(totals["tp"], totals["tp"] + totals["fn"])
    summary = {
        **totals,
        "precision": precision,
        "recall": recall,
        "f1": safe_ratio(2 * precision * recall, precision + recall),
        "pref_argument_match_rate": safe_ratio(
            totals["pref_matched"], totals["pref_slots"]
        ),
        "nonpref_argument_match_rate": safe_ratio(
            totals["nonpref_matched"], totals["nonpref_slots"]
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.csv_output:
        output = Path(args.csv_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as handle:
            if details:
                writer = csv.DictWriter(handle, fieldnames=list(details[0]))
                writer.writeheader()
                writer.writerows(details)
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps({"summary": summary, "rows": details}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
