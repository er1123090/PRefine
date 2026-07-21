#!/usr/bin/env python3
"""Compare same-sample PEToolBench baseline and variant runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def split_delta(base: Dict[str, Any], variant: Dict[str, Any]) -> Dict[str, Any]:
    base_acc = float(base.get("tool_accuracy", 0.0) or 0.0)
    var_acc = float(variant.get("tool_accuracy", 0.0) or 0.0)
    return {
        "baseline_tool_accuracy": base_acc,
        "variant_tool_accuracy": var_acc,
        "tool_acc_delta": (var_acc - base_acc) * 100,
        "baseline_parameter_accuracy": float(base.get("parameter_accuracy", 0.0) or 0.0),
        "variant_parameter_accuracy": float(variant.get("parameter_accuracy", 0.0) or 0.0),
        "parameter_acc_delta": (
            float(variant.get("parameter_accuracy", 0.0) or 0.0)
            - float(base.get("parameter_accuracy", 0.0) or 0.0)
        )
        * 100,
        "baseline_parse_failures": int(base.get("parse_failures", 0) or 0),
        "variant_parse_failures": int(variant.get("parse_failures", 0) or 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--variant-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-tool-acc-delta", type=float, default=5.0)
    args = parser.parse_args()

    baseline = read_json(args.baseline_dir / "summary_all.json")
    variant = read_json(args.variant_dir / "summary_all.json")
    split_results = {}
    for history_type in ("p", "r", "c"):
        split_results[history_type] = split_delta(
            baseline.get("summaries", {}).get(history_type, {}),
            variant.get("summaries", {}).get(history_type, {}),
        )
    p_pass = split_results["p"]["tool_acc_delta"] >= args.min_tool_acc_delta
    r_pass = split_results["r"]["tool_acc_delta"] >= args.min_tool_acc_delta
    result = {
        "baseline_dir": str(args.baseline_dir),
        "variant_dir": str(args.variant_dir),
        "min_tool_acc_delta": args.min_tool_acc_delta,
        "splits": split_results,
        "p_pass": p_pass,
        "r_pass": r_pass,
        "success": p_pass and r_pass,
        "c_report_only": split_results["c"],
    }
    output = args.output or (args.variant_dir / "comparison_pr_gate.json")
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
