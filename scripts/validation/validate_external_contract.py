#!/usr/bin/env python3
"""Validate a publication, runtime closure, and external run config without execution."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from facade.external import validate_external_contract  # noqa: E402
from facade.planner import profile_selection  # noqa: E402
from facade.registry import FacadeError, load_bundle  # noqa: E402
from facade.selection import canonical_bytes  # noqa: E402


def main(argv: list[str] | None = None, *, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-id", required=True)
    parser.add_argument("--publication-id", required=True)
    parser.add_argument("--runtime-id", required=True)
    parser.add_argument("--run-config-id", required=True)
    parser.add_argument("--json", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        bundle = load_bundle(root)
        target = root / "runs/.external-contract-validation"
        selection = profile_selection(
            bundle,
            args.profile_id,
            execution="external_contract_validation",
            run_id=None,
            output_root=target,
        )
        contract = validate_external_contract(
            bundle,
            selection,
            target,
            publication_id=args.publication_id,
            runtime_id=args.runtime_id,
            run_config_id=args.run_config_id,
            require_environment=False,
        )
        report = {
            "schema": "experiments7-external-contract-validation/v1",
            "state": "PASS",
            "profile_id": args.profile_id,
            "contract": contract.summary(),
            "executed": False,
            "protected_reads": 0,
            "protected_writes": 0,
        }
        sys.stdout.buffer.write(canonical_bytes(report))
        return 0
    except FacadeError as exc:
        report = {
            "schema": "experiments7-external-contract-validation/v1",
            "state": "BLOCKED",
            "reason_codes": [exc.code],
            "executed": False,
            "protected_reads": 0,
            "protected_writes": 0,
        }
        if exc.detail is not None:
            report["detail"] = exc.detail
        sys.stdout.buffer.write(canonical_bytes(report))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
