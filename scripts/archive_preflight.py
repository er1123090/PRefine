#!/usr/bin/env python3
"""Read-only preflight for off-host raw backup and restored-copy evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp7.provenance.archive_preflight import (
    ArchivePreflightError,
    CANONICAL_RAW_COUNT,
    CANONICAL_RAW_MANIFEST,
    REPORT_SCHEMA,
    validate_archive_preflight,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--backup-proof", type=Path)
    value.add_argument("--object-manifest", type=Path)
    value.add_argument("--retention-evidence", type=Path)
    value.add_argument("--restore-transcript", type=Path)
    value.add_argument("--restore-root", type=Path)
    return value


def _blocked(error: str, *, missing: list[str] | None = None) -> dict[str, object]:
    value: dict[str, object] = {
        "archive_map_mutated": False,
        "cutover_receipt_verified": False,
        "evidence_bundle_structurally_valid": False,
        "error": error,
        "externalization_authorized": False,
        "independent_approval_verified": False,
        "no_mutations": True,
        "preflight_passed": False,
        "preflight_scope": "structural_only",
        "provider_authenticity_verified": False,
        "schema": REPORT_SCHEMA,
        "status": "blocked",
        "validation_domain": "archive_evidence_structure",
    }
    if missing:
        value["missing_arguments"] = missing
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    required = {
        "--backup-proof": args.backup_proof,
        "--object-manifest": args.object_manifest,
        "--retention-evidence": args.retention_evidence,
        "--restore-transcript": args.restore_transcript,
        "--restore-root": args.restore_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        print(
            json.dumps(
                _blocked("off-host backup and restore evidence is required", missing=missing),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    try:
        report = validate_archive_preflight(
            canonical_manifest=ROOT / CANONICAL_RAW_MANIFEST,
            backup_proof=args.backup_proof,
            object_manifest=args.object_manifest,
            retention_evidence=args.retention_evidence,
            restore_transcript=args.restore_transcript,
            restore_root=args.restore_root,
            expected_count=CANONICAL_RAW_COUNT,
        )
    except (ArchivePreflightError, OSError) as exc:
        print(json.dumps(_blocked(str(exc)), ensure_ascii=False, sort_keys=True))
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
