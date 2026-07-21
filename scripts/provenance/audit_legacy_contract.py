#!/usr/bin/env python3
"""Audit the immutable legacy admission/copy contract without protected reads."""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
COPY_ELIGIBLE = {"ADMITTED_FOR_COPY", "DERIVED_FROM_ADMITTED_RAW"}


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not an object")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def semantic_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("model"),
        row.get("setting"),
        row.get("preference_type"),
        row.get("metric"),
    )


def build_audit(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    inventory_rows = load_jsonl(root / "paper_outputs/inventory/results.jsonl")
    admission_rows = load_jsonl(root / "paper_outputs/admission/precopy/results.jsonl")
    node_rows = load_jsonl(root / "paper_outputs/provenance/nodes.jsonl")
    copy_rows = load_jsonl(root / "manifests/copies.jsonl")
    admission_summary = load_json(root / "paper_outputs/admission/precopy/summary.json")
    admission_seal = load_json(root / "paper_outputs/admission/precopy/seal.json")
    cp0_seal = load_json(root / "manifests/cp0-seal.json")

    inventory = {row["result_id"]: row for row in inventory_rows}
    admissions = {row["result_id"]: row for row in admission_rows}
    table3: dict[tuple[Any, ...], dict[str, str]] = defaultdict(dict)
    for row in inventory_rows:
        if row.get("evidence_id") == "table:3" and row.get("method") in {
            "Base Prompting", "PREFINE",
        }:
            table3[semantic_key(row)][row["method"]] = row["result_id"]

    defects: list[dict[str, Any]] = []
    incomplete_table10: list[dict[str, Any]] = []
    for row in inventory_rows:
        result_id = row["result_id"]
        admission = admissions[result_id]
        if (
            row.get("evidence_id") != "table:10"
            or admission.get("derived_state") not in COPY_ELIGIBLE
        ):
            continue
        parents = table3.get(semantic_key(row), {})
        parent_ids = [
            parents.get("Base Prompting"),
            parents.get("PREFINE"),
        ]
        parent_states = {
            parent_id: admissions.get(parent_id, {}).get("derived_state")
            for parent_id in parent_ids
            if parent_id is not None
        }
        if len(parent_ids) != 2 or any(parent is None for parent in parent_ids) or any(
            parent_states.get(parent) not in COPY_ELIGIBLE
            for parent in parent_ids
            if parent is not None
        ):
            defect = {
                "schema": "experiments7-legacy-contract-defect/v1",
                "defect_type": "TABLE10_INCOMPLETE_PARENT_ADMISSION",
                "result_id": result_id,
                "legacy_state": admission.get("derived_state"),
                "parent_result_ids": parent_ids,
                "parent_states": parent_states,
                "legacy_raw_artifact_ids": admission.get("raw_artifact_ids", []),
                "reason": "both Table 3 Base and PREFINE parents must be copy-eligible",
            }
            incomplete_table10.append(defect)
            defects.append(defect)

    result_ids_by_raw: dict[str, list[str]] = defaultdict(list)
    for admission in admission_rows:
        for raw_id in admission.get("raw_artifact_ids", []):
            result_ids_by_raw[raw_id].append(admission["result_id"])
    raw_nodes = {
        row["node_id"]: row
        for row in node_rows
        if row.get("node_type") == "raw_artifact"
    }
    drift_only: list[dict[str, Any]] = []
    for raw_id, result_ids in sorted(result_ids_by_raw.items()):
        if result_ids and all("declared_drift" in admissions[result_id] for result_id in result_ids):
            node = raw_nodes[raw_id]
            defect = {
                "schema": "experiments7-legacy-contract-defect/v1",
                "defect_type": "DRIFT_ONLY_RAW_WAS_COPIED",
                "raw_artifact_id": raw_id,
                "paper_result_ids": sorted(result_ids),
                "root_id": node["root_id"],
                "relative_path": base64.b64decode(node["relative_path_b64"]).decode("utf-8"),
                "sha256": node["sha256"],
                "size": node["size"],
                "reason": "strict exact-only copy cannot be justified only by declared-drift results",
            }
            drift_only.append(defect)
            defects.append(defect)

    run_identity_match = (
        cp0_seal.get("sealed_run_id") == admission_seal.get("sealed_run_id")
        and cp0_seal.get("envelope_id") == admission_seal.get("protected_envelope_id")
    )
    if not run_identity_match:
        defects.append({
            "schema": "experiments7-legacy-contract-defect/v1",
            "defect_type": "SEALED_RUN_IDENTITY_DISCONTINUITY",
            "cp0_sealed_run_id": cp0_seal.get("sealed_run_id"),
            "admission_sealed_run_id": admission_seal.get("sealed_run_id"),
            "cp0_envelope_id": cp0_seal.get("envelope_id"),
            "admission_envelope_id": admission_seal.get("protected_envelope_id"),
        })
    if admission_summary.get("unresolved_count", 0) != 0:
        defects.append({
            "schema": "experiments7-legacy-contract-defect/v1",
            "defect_type": "COPY_EXECUTED_BEFORE_ZERO_UNRESOLVED_GATE",
            "unresolved_count": admission_summary.get("unresolved_count"),
            "legacy_terminal_state": admission_summary.get("terminal_state"),
            "legacy_copy_records": len(copy_rows),
        })

    defects.sort(
        key=lambda row: (
            row["defect_type"],
            row.get("result_id", ""),
            row.get("raw_artifact_id", ""),
        )
    )
    summary = {
        "schema": "experiments7-legacy-contract-audit-summary/v1",
        "terminal_state": "BLOCKED" if defects else "PASS",
        "legacy_admission_results": len(admission_rows),
        "legacy_unresolved_results": admission_summary.get("unresolved_count"),
        "legacy_copy_records": len(copy_rows),
        "legacy_raw_nodes": len(result_ids_by_raw),
        "incomplete_table10_admissions": len(incomplete_table10),
        "drift_only_copied_raw_nodes": len(drift_only),
        "sealed_run_identity_match": run_identity_match,
        "defect_records": len(defects),
        "protected_source_reads_performed": 0,
        "protected_source_writes_performed": 0,
        "legacy_artifacts_modified": False,
        "strict_reuse_allowed": False,
        "reason_code": "LEGACY_RELAXED_EVIDENCE_NOT_STRICT_RELEASE_EVIDENCE",
    }
    return defects, summary


def write_noreplace(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, 0o444)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--json", action="store_true", required=True)
    args = parser.parse_args()
    defects, summary = build_audit(args.root.resolve(strict=True))
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        write_noreplace(
            args.output_dir / "defects.jsonl",
            b"".join(canonical_bytes(row) for row in defects),
        )
        write_noreplace(args.output_dir / "summary.json", pretty_bytes(summary))
    print(canonical_bytes(summary).decode("utf-8"), end="")
    return 2 if summary["terminal_state"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
