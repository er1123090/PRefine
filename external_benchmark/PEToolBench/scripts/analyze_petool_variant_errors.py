#!/usr/bin/env python3
"""Create lightweight diagnostics for failed PEToolBench variants."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


TOOL_RE = re.compile(r"<([^>]+)>")


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def namespace(tool_name: str) -> str:
    parts = TOOL_RE.findall(tool_name or "")
    if len(parts) >= 2:
        return f"<{parts[0]}>.<{parts[1]}>"
    return tool_name or ""


def operation(tool_name: str) -> str:
    parts = TOOL_RE.findall(tool_name or "")
    return parts[2] if len(parts) >= 3 else (parts[-1] if parts else tool_name or "")


def classify(row: Dict[str, Any], baseline_row: Dict[str, Any]) -> str:
    gt = row.get("api_call_ground_truth", {}) or {}
    var = row.get("parsed_response", {}) or {}
    base = baseline_row.get("parsed_response", {}) or {}
    if not isinstance(var, dict) or not var:
        return "variant_parse_failure"
    if not isinstance(base, dict) or not base:
        base = {}
    gt_tool = gt.get("tool_name")
    var_tool = var.get("tool_name")
    base_tool = base.get("tool_name")
    var_correct = var_tool == gt_tool
    base_correct = base_tool == gt_tool
    if base_correct and not var_correct:
        if namespace(var_tool) != namespace(gt_tool):
            return "regression_provider_namespace_mismatch"
        if operation(var_tool) != operation(gt_tool):
            return "regression_operation_mismatch"
        return "regression_other_wrong_tool"
    if (not base_correct) and var_correct:
        return "variant_fixed_baseline_error"
    if var_correct and var.get("parameters") != gt.get("parameters"):
        return "tool_correct_params_wrong"
    if not var_correct:
        if namespace(var_tool) != namespace(gt_tool):
            return "provider_namespace_mismatch"
        if operation(var_tool) != operation(gt_tool):
            return "operation_mismatch"
        return "wrong_tool_other"
    return "correct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--variant-dir", type=Path, required=True)
    parser.add_argument("--history-types", nargs="+", default=["p", "r", "c"])
    args = parser.parse_args()

    summary = {}
    for history_type in args.history_types:
        baseline_rows = read_json(args.baseline_dir / f"predictions_{history_type}.json")
        variant_rows = read_json(args.variant_dir / f"predictions_{history_type}.json")
        baseline_by_id = {row["example_id"]: row for row in baseline_rows}
        details: List[Dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for row in variant_rows:
            base = baseline_by_id.get(row["example_id"], {})
            category = classify(row, base)
            counts[category] += 1
            if category != "correct":
                details.append(
                    {
                        "example_id": row.get("example_id"),
                        "source_index": row.get("source_index"),
                        "category": category,
                        "query": row.get("query"),
                        "ground_truth": row.get("api_call_ground_truth"),
                        "baseline_parsed": base.get("parsed_response"),
                        "variant_parsed": row.get("parsed_response"),
                    }
                )
        payload = {"history_type": history_type, "counts": dict(counts), "examples": details[:50]}
        write_json(args.variant_dir / f"diagnostics_{history_type}.json", payload)
        summary[history_type] = payload["counts"]
    write_json(args.variant_dir / "diagnostics_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
