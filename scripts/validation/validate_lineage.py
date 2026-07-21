#!/usr/bin/env python3
"""Validate exact G1 lineage closure and pending G2 snapshot bindings."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from facade.registry import FacadeError, load_bundle, load_json  # noqa: E402
from facade.selection import canonical_bytes, sha256_file  # noqa: E402


def main() -> int:
    try:
        bundle = load_bundle(ROOT)
        plan_path = ROOT / "configs/snapshot-plan.json"
        plan = load_json(plan_path)
        records = plan.get("records", [])
        expected_ids = set(bundle.lineage_by_id)
        actual_ids = {row.get("lineage_id") for row in records}
        if len(records) != 129 or actual_ids != expected_ids:
            raise FacadeError("SNAPSHOT_PLAN_LINEAGE_SET")
        if any(
            row.get("snapshot_state") != "pending_protected_snapshot"
            or row.get("final_sha256") is not None
            for row in records
        ):
            raise FacadeError("UNAUTHORIZED_SNAPSHOT_COMPLETION")
        report = {
            "schema": "experiments7-g2-lineage-validation/v1",
            "state": "PASS",
            "counts": {"code": 118, "config": 11, "total": 129, "pending": 129},
            "snapshot_plan_sha256": sha256_file(plan_path)[0],
            "protected_reads": 0,
            "protected_writes": 0,
        }
        sys.stdout.buffer.write(canonical_bytes(report))
        return 0
    except FacadeError as exc:
        sys.stdout.buffer.write(canonical_bytes({
            "schema": "experiments7-g2-lineage-validation/v1",
            "state": "BLOCKED",
            "reason_codes": [exc.code],
            "detail": exc.detail,
        }))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
