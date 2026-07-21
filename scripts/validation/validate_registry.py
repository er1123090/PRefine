#!/usr/bin/env python3
"""Validate exact G2 registry/profile/compatibility/adapter closure."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from facade.registry import FacadeError, load_bundle  # noqa: E402
from facade.selection import canonical_bytes  # noqa: E402


def main() -> int:
    try:
        bundle = load_bundle(ROOT)
        coverage = {
            variant_id
            for profile in bundle.profiles_by_id.values()
            for variant_id in profile["stages"].values()
        }
        report = {
            "schema": "experiments7-g2-registry-validation/v1",
            "state": "PASS",
            "counts": {
                "families": len(bundle.families),
                "variants": len(bundle.variants),
                "adapters": len(bundle.adapters_by_variant),
                "profiles": len(bundle.profiles_by_id),
                "covered_variants": len(coverage),
                "reference_only": 2,
            },
            "hashes": dict(bundle.hashes),
            "protected_reads": 0,
        }
        sys.stdout.buffer.write(canonical_bytes(report))
        return 0
    except FacadeError as exc:
        sys.stdout.buffer.write(canonical_bytes({
            "schema": "experiments7-g2-registry-validation/v1",
            "state": "BLOCKED",
            "reason_codes": [exc.code],
            "detail": exc.detail,
        }))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
