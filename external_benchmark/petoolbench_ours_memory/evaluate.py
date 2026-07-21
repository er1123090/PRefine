"""Evaluate PEToolBench predictions."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from common import first_json_object, read_json, write_json


def normalize_prediction(row: Dict[str, Any]) -> Dict[str, Any]:
    parsed = row.get("parsed_response")
    if isinstance(parsed, dict) and parsed:
        return parsed
    response = row.get("response", "")
    return first_json_object(response) or {}


def evaluate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    details = []
    tool_correct = 0
    parameter_correct = 0
    parse_failures = 0
    for row in rows:
        gt = row.get("api_call_ground_truth") or {}
        pred = normalize_prediction(row)
        if not pred:
            parse_failures += 1
        is_tool_correct = gt.get("tool_name") == pred.get("tool_name")
        is_parameter_correct = gt.get("parameters") == pred.get("parameters")
        tool_correct += int(is_tool_correct)
        parameter_correct += int(is_parameter_correct)
        details.append(
            {
                "example_id": row.get("example_id"),
                "history_type": row.get("history_type"),
                "tool_correct": is_tool_correct,
                "parameter_correct": is_parameter_correct,
                "ground_truth": gt,
                "prediction": pred,
                "error": row.get("error"),
            }
        )

    n = len(rows)
    return {
        "n": n,
        "tool_correct": tool_correct,
        "parameter_correct": parameter_correct,
        "tool_accuracy": tool_correct / n if n else 0.0,
        "parameter_accuracy": parameter_correct / n if n else 0.0,
        "parse_failures": parse_failures,
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PEToolBench predictions.")
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output_summary", required=True)
    args = parser.parse_args()

    rows = read_json(args.predictions)
    if not isinstance(rows, list):
        raise SystemExit("predictions file must contain a JSON list")
    summary = evaluate_rows(rows)
    write_json(args.output_summary, summary)
    print(
        "n={n} tool_accuracy={tool_accuracy:.4f} parameter_accuracy={parameter_accuracy:.4f} "
        "parse_failures={parse_failures} output={output}".format(
            output=args.output_summary,
            **summary,
        )
    )


if __name__ == "__main__":
    main()

