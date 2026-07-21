#!/usr/bin/env python3
"""Build a deterministic, no-replace audit of unresolved paper-result provenance.

This tool reads only experiments7 control artifacts. It never inspects the
protected experiments4/5/6 trees or the paper PDF.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def pretty_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"record {number} in {path} is not an object")
            rows.append(value)
    return rows


def evidence_requirement(reason_code: str) -> str:
    mapping = {
        "ambiguous_exact_raw_candidates": "unique_exact_candidate_disambiguation",
        "average_gain_parent_unresolved": "resolve_declared_parent_results",
        "derived_from_unresolved_figure4_results": "resolve_figure4_producer_bindings",
        "derived_parent_result_unresolved_or_without_raw": "resolve_declared_parent_results",
        "figure4_semantic_raw_set_not_fully_source_pre_bound": "protected_source_binding_required",
        "figure4_slotcount_semantic_set_empty": "figure_producer_binding_required",
        "no_exact_aggregate_row": "exact_producer_or_declared_derivation_required",
        "raw_path_not_source_pre_bound": "protected_source_binding_required",
        "table10_delta_does_not_equal_table3_prefine_minus_base": "paper_delta_producer_binding_required",
        "table3_delta_parents_have_no_admitted_raw": "resolve_declared_parent_results",
        "table3_prefine_cf_full_precision_raw_set_ambiguous_or_unbound": "unique_full_precision_parent_set_required",
        "table3_prefine_cf_full_precision_value_not_reproduced": "paper_value_producer_binding_required",
        "table3_prefine_cg_full_precision_parent_set_not_unique": "unique_full_precision_parent_set_required",
    }
    return mapping.get(reason_code, "manual_evidence_review_required")


def build_audit(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    admission_path = root / "paper_outputs/admission/precopy/results.jsonl"
    inventory_path = root / "paper_outputs/inventory/results.jsonl"
    admissions = load_jsonl(admission_path)
    inventory_rows = load_jsonl(inventory_path)
    inventory = {row["result_id"]: row for row in inventory_rows}
    states = {row["result_id"]: row.get("derived_state") for row in admissions}
    unresolved = [row for row in admissions if row.get("derived_state") == "UNRESOLVED"]

    memo: dict[str, tuple[str, ...]] = {}

    def blocking_roots(result_id: str, stack: tuple[str, ...] = ()) -> tuple[str, ...]:
        if result_id in memo:
            return memo[result_id]
        if result_id in stack:
            return (result_id,)
        admission = next((row for row in unresolved if row["result_id"] == result_id), None)
        paper = inventory.get(result_id, {})
        parents = [
            parent for parent in paper.get("derived_from", [])
            if states.get(parent) == "UNRESOLVED"
        ]
        if admission is None or admission.get("classification_basis") != "unresolved_derived":
            roots = (result_id,)
        elif not parents:
            roots = ()
        else:
            roots = tuple(sorted({
                root_id
                for parent in parents
                for root_id in blocking_roots(parent, stack + (result_id,))
            }))
        memo[result_id] = roots
        return roots

    records: list[dict[str, Any]] = []
    for admission in unresolved:
        result_id = admission["result_id"]
        paper = inventory[result_id]
        reason_codes = list(admission.get("reason_codes", []))
        parents = list(paper.get("derived_from", []))
        unresolved_parents = sorted(parent for parent in parents if states.get(parent) == "UNRESOLVED")
        admitted_parents = sorted(parent for parent in parents if states.get(parent) in {
            "ADMITTED_FOR_COPY", "DERIVED_FROM_ADMITTED_RAW",
        })
        roots = list(blocking_roots(result_id))
        direct_unresolved = admission.get("classification_basis") != "unresolved_derived"
        non_inference_derived = bool(parents) and all(
            inventory.get(parent, {}).get("evidence_id") == "figure:5"
            and states.get(parent) == "NON_INFERENCE"
            for parent in parents
        )
        records.append({
            "schema": "experiments7-unresolved-paper-result-audit/v1",
            "result_id": result_id,
            "evidence_id": admission.get("evidence_id"),
            "classification_basis": admission.get("classification_basis"),
            "reason_codes": reason_codes,
            "required_evidence_classes": sorted({
                evidence_requirement(reason) for reason in reason_codes
            }),
            "paper_result": {
                key: paper.get(key)
                for key in (
                    "page", "section", "result_kind", "method", "model", "setting",
                    "preference_type", "metric", "displayed_value", "displayed_marker",
                )
            },
            "declared_parent_result_ids": parents,
            "unresolved_parent_result_ids": unresolved_parents,
            "admitted_parent_result_ids": admitted_parents,
            "blocking_root_result_ids": roots,
            "is_direct_unresolved": direct_unresolved,
            "public_dependency_state": (
                "direct_root"
                if direct_unresolved
                else "serialized_unresolved_parents"
                if roots
                else "parents_not_serialized_as_unresolved"
            ),
            "resolution_state": (
                "RESOLVABLE_FROM_EXPERIMENTS7_ONLY"
                if non_inference_derived
                else "REQUIRES_ADDITIONAL_EVIDENCE"
            ),
            "proposed_reclassification": (
                "NON_INFERENCE_DERIVED"
                if non_inference_derived
                else None
            ),
            "protected_source_reads_performed": 0,
        })
    records.sort(key=lambda row: row["result_id"])

    reason_counts = Counter(
        reason
        for row in records
        for reason in row["reason_codes"]
    )
    evidence_counts = Counter(row["evidence_id"] for row in records)
    requirement_counts = Counter(
        requirement
        for row in records
        for requirement in row["required_evidence_classes"]
    )
    public_root_ids = sorted({
        root_id for row in records for root_id in row["blocking_root_result_ids"]
    })
    direct_root_ids = sorted(
        row["result_id"] for row in records if row["is_direct_unresolved"]
    )
    summary = {
        "schema": "experiments7-unresolved-paper-result-audit-summary/v1",
        "source_admission_path": "paper_outputs/admission/precopy/results.jsonl",
        "source_inventory_path": "paper_outputs/inventory/results.jsonl",
        "total_paper_results": len(admissions),
        "unresolved_results": len(records),
        "direct_unresolved_results": len(direct_root_ids),
        "derived_unresolved_results": sum(not row["is_direct_unresolved"] for row in records),
        "direct_unresolved_result_ids": direct_root_ids,
        "unique_public_blocking_root_results": len(public_root_ids),
        "public_blocking_root_result_ids": public_root_ids,
        "derived_results_without_serialized_unresolved_parents": sum(
            not row["is_direct_unresolved"]
            and row["public_dependency_state"] == "parents_not_serialized_as_unresolved"
            for row in records
        ),
        "resolvable_from_experiments7_only": sum(
            row["resolution_state"] == "RESOLVABLE_FROM_EXPERIMENTS7_ONLY"
            for row in records
        ),
        "public_dependency_graph_complete": False,
        "public_dependency_graph_limitation": (
            "the sealed admission serializer omits candidate diagnostics and some "
            "derived-parent bindings; empty public roots are not evidence of independence"
        ),
        "counts_by_evidence": dict(sorted(evidence_counts.items())),
        "counts_by_reason": dict(sorted(reason_counts.items())),
        "counts_by_required_evidence": dict(sorted(requirement_counts.items())),
        "protected_source_reads_performed": 0,
        "protected_source_writes_performed": 0,
        "admission_artifacts_modified": False,
        "terminal_state": "BLOCKED" if records else "PASS",
        "reason_code": "ZERO_UNRESOLVED_GATE_NOT_MET" if records else None,
    }
    return records, summary


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
    root = args.root.resolve(strict=True)
    records, summary = build_audit(root)
    if args.output_dir is not None:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=False)
        payload = b"".join(canonical_bytes(row) for row in records)
        write_noreplace(output_dir / "results.jsonl", payload)
        write_noreplace(output_dir / "summary.json", pretty_bytes(summary))
    print(canonical_bytes(summary).decode("utf-8"), end="")
    return 2 if summary["terminal_state"] == "BLOCKED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
