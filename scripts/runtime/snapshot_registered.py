#!/usr/bin/env python3
"""Inspect the exact G2 snapshot plan; protected snapshot execution stays gated."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from facade.metadata import construct_metadata  # noqa: E402
from facade.registry import FacadeError, load_json  # noqa: E402
from facade.selection import canonical_bytes, sha256_bytes  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--json", action="store_true", required=True)
    args = parser.parse_args()
    plan_path = ROOT / "configs/snapshot-plan.json"
    expected = construct_metadata(ROOT)["configs/snapshot-plan.json"]
    actual = plan_path.read_bytes() if plan_path.is_file() and not plan_path.is_symlink() else b""
    if actual != expected:
        report = {
            "schema": "experiments7-g2-snapshot-check/v1",
            "state": "BLOCKED",
            "reason_codes": ["STALE_SNAPSHOT_PLAN"],
        }
        print(canonical_bytes(report).decode(), end="")
        return 2
    plan = load_json(plan_path)
    if args.execute:
        report = {
            "schema": "experiments7-g2-snapshot-check/v1",
            "state": "BLOCKED",
            "reason_codes": ["PROTECTED_SNAPSHOT_REQUIRES_SEALED_G0_EXECUTOR"],
            "record_count": len(plan["records"]),
            "snapshot_plan_sha256": sha256_bytes(actual),
            "protected_reads": 0,
        }
        print(canonical_bytes(report).decode(), end="")
        return 2
    report = {
        "schema": "experiments7-g2-snapshot-check/v1",
        "state": "PASS",
        "record_count": len(plan["records"]),
        "pending_records": sum(row["snapshot_state"] == "pending_protected_snapshot" for row in plan["records"]),
        "snapshot_plan_sha256": sha256_bytes(actual),
        "protected_reads": 0,
    }
    print(canonical_bytes(report).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
