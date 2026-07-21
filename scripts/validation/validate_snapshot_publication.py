#!/usr/bin/env python3
"""Rehash and validate one exhaustive experiments7 snapshot publication."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from facade.registry import FacadeError, load_bundle  # noqa: E402
from facade.selection import canonical_bytes  # noqa: E402
from facade.snapshots import validate_snapshot_publication  # noqa: E402


def main(argv: list[str] | None = None, *, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--publication-id", required=True)
    parser.add_argument("--json", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        publication = validate_snapshot_publication(load_bundle(root), args.publication_id)
        report = {
            "schema": "experiments7-snapshot-publication-validation/v1",
            "state": "PASS",
            "publication": publication.summary(),
            "protected_reads": 0,
            "protected_writes": 0,
        }
        sys.stdout.buffer.write(canonical_bytes(report))
        return 0
    except FacadeError as exc:
        report = {
            "schema": "experiments7-snapshot-publication-validation/v1",
            "state": "BLOCKED",
            "reason_codes": [exc.code],
            "protected_reads": 0,
            "protected_writes": 0,
        }
        if exc.detail is not None:
            report["detail"] = exc.detail
        sys.stdout.buffer.write(canonical_bytes(report))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
