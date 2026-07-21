#!/usr/bin/env python3
"""Validate completion of the experiments4-code ablation rerun."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path("/data/minseo")
EXP5 = ROOT / "experiments5"
RUN_ID = "1229_dev6_ablation_e4_ours_memory_inference_20260528"
RESULT_ID = "1229_dev6_ablation_e4_ours_memory_eval_20260528"
OUTPUT_ROOT = EXP5 / "outputs/our_memory" / RUN_ID / "inference"
RESULT_ROOT = EXP5 / "results/our_memory" / RESULT_ID
EXPECTED_FILES = 4 * 3 * 4 * 2 * 3


def fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def main() -> int:
    status = subprocess.check_output(
        ["git", "-C", str(ROOT), "status", "--short", "--", "experiments4"],
        text=True,
    ).strip()
    if status:
        return fail(f"experiments4 has git changes: {status}")

    result_files = list(OUTPUT_ROOT.glob("**/result.json"))
    if len(result_files) != EXPECTED_FILES:
        return fail(f"expected {EXPECTED_FILES} result.json files, found {len(result_files)}")

    coverage_path = RESULT_ROOT / "coverage.csv"
    summary_path = RESULT_ROOT / "summary_by_method.csv"
    manifest_path = RESULT_ROOT / "manifest.json"
    for path in [coverage_path, summary_path, manifest_path]:
        if not path.exists():
            return fail(f"missing evaluation artifact: {path}")

    with coverage_path.open(encoding="utf-8", newline="") as f:
        coverage = list(csv.DictReader(f))
    bad = [row for row in coverage if row.get("status") != "ok"]
    if len(coverage) != EXPECTED_FILES:
        return fail(f"coverage rows expected {EXPECTED_FILES}, found {len(coverage)}")
    if bad:
        return fail(f"coverage has {len(bad)} missing/error rows")

    with summary_path.open(encoding="utf-8", newline="") as f:
        summary = list(csv.DictReader(f))
    methods = {row.get("method") for row in summary}
    expected_methods = {"gen_only_accum", "refine1", "refine2", "refine3"}
    if methods != expected_methods:
        return fail(f"summary methods mismatch: {sorted(methods)}")
    if any(int(row.get("file_count", 0)) != 72 for row in summary):
        return fail("each method should summarize 72 files")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("evaluation_parser_source") != "/data/minseo/experiments4/evaluation/evaluation_multiturn-f1-parse-aggregate.py":
        return fail("manifest does not point to experiments4 evaluator")
    if manifest.get("ok_files") != EXPECTED_FILES:
        return fail(f"manifest ok_files mismatch: {manifest.get('ok_files')}")

    print(
        "PASS: experiments4 clean, "
        f"{EXPECTED_FILES}/{EXPECTED_FILES} result files, evaluator artifacts complete at {RESULT_ROOT}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
