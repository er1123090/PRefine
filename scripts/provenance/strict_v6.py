"""Compatibility entrypoint for the package-backed strict provenance V6 core.

Pure admission, copy, and reconciliation contracts live in
``src/provenance/strict_v6.py``.  This wrapper preserves the historical script
import surface and owns only the pinned, read-only audit-oracle adapter.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from provenance import strict_v6 as _core  # noqa: E402
from provenance.strict_v6 import *  # noqa: E402,F403


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise V6ContractError(  # noqa: F405
                    f"invalid JSONL {path}:{line_number}"
                ) from exc
            if not isinstance(row, dict):
                raise V6ContractError(  # noqa: F405
                    f"non-object JSONL {path}:{line_number}"
                )
            rows.append(row)
    return rows


def validate_immutable_audit_oracles(experiments7_root: Path) -> dict[str, Any]:
    """Validate historical audits only as pinned negative regression oracles."""

    unresolved_path = experiments7_root / AUDIT_UNRESOLVED_RELATIVE  # noqa: F405
    defects_path = experiments7_root / AUDIT_DEFECTS_RELATIVE  # noqa: F405
    if _sha256_file(unresolved_path) != AUDIT_UNRESOLVED_SHA256:  # noqa: F405
        raise V6ContractError("immutable unresolved audit hash mismatch")  # noqa: F405
    if _sha256_file(defects_path) != AUDIT_DEFECTS_SHA256:  # noqa: F405
        raise V6ContractError("immutable defect audit hash mismatch")  # noqa: F405
    unresolved = _load_jsonl(unresolved_path)
    defects = _load_jsonl(defects_path)
    unresolved_ids = [_core._required_string(row, "result_id") for row in unresolved]
    if len(unresolved_ids) != 238 or len(set(unresolved_ids)) != 238:
        raise V6ContractError(  # noqa: F405
            "immutable unresolved audit is not 238 unique IDs"
        )
    id_set_payload = ("\n".join(sorted(unresolved_ids)) + "\n").encode("utf-8")
    if hashlib.sha256(id_set_payload).hexdigest() != AUDIT_UNRESOLVED_ID_SET_SHA256:  # noqa: F405,E501
        raise V6ContractError(  # noqa: F405
            "immutable unresolved result-ID set mismatch"
        )

    table10_ids = {
        _core._required_string(row, "result_id")
        for row in defects
        if row.get("defect_type") == "TABLE10_INCOMPLETE_PARENT_ADMISSION"
    }
    if table10_ids != TABLE10_INCOMPLETE_PARENT_RESULT_IDS or len(table10_ids) != 34:  # noqa: F405,E501
        raise V6ContractError("immutable Table10 34-ID defect set mismatch")  # noqa: F405
    drift_rows = [
        row for row in defects if row.get("defect_type") == "DRIFT_ONLY_RAW_WAS_COPIED"
    ]
    if len(drift_rows) != 1:
        raise V6ContractError("immutable drift-only defect cardinality mismatch")  # noqa: F405
    drift = drift_rows[0]
    if (
        drift.get("raw_artifact_id") != DRIFT_ONLY_RAW_ID  # noqa: F405
        or drift.get("sha256") != DRIFT_ONLY_RAW_SHA256  # noqa: F405
    ):
        raise V6ContractError("immutable drift-only raw contract mismatch")  # noqa: F405
    return {
        "schema": "experiments7-immutable-audit-oracles/v6",
        "positive_evidence_allowed": False,
        "unresolved_result_ids": sorted(unresolved_ids),
        "unresolved_result_id_count": 238,
        "table10_incomplete_parent_result_ids": sorted(table10_ids),
        "table10_incomplete_parent_result_id_count": 34,
        "figure5_zero_raw_result_ids": sorted(FIGURE5_ZERO_RAW_RESULT_IDS),  # noqa: F405
        "drift_only_raw_id": DRIFT_ONLY_RAW_ID,  # noqa: F405
        "drift_only_raw_sha256": DRIFT_ONLY_RAW_SHA256,  # noqa: F405
    }


__all__ = [*_core.__all__, "validate_immutable_audit_oracles"]
