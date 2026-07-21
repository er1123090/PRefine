#!/usr/bin/env python3
"""Create one fresh V6 strict run and seal its G0/CP0 bootstrap."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from strict_run import canonical_json  # noqa: E402
from strict_run.g0_bootstrap import bootstrap_g0_cp0  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reserve a fresh strict run, freeze G0, then emit CP0. "
            "Protected roots are opened only by the frozen child after its "
            "read-only envelope is active."
        )
    )
    parser.add_argument("--strict-parent", required=True)
    parser.add_argument("--sealed-run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--external-dir", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--owner-seed", required=True)
    parser.add_argument("--experiments4-root", required=True)
    parser.add_argument("--experiments5-root", required=True)
    parser.add_argument("--experiments6-root", required=True)
    parser.add_argument("--paper-path", required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    args = parser.parse_args()

    result = bootstrap_g0_cp0(
        strict_parent=args.strict_parent,
        sealed_run_id=args.sealed_run_id,
        run_root=args.run_root,
        external_dir=args.external_dir,
        project_id=args.project_id,
        owner_seed=args.owner_seed,
        source_roots={
            "experiments4": args.experiments4_root,
            "experiments5": args.experiments5_root,
            "experiments6": args.experiments6_root,
        },
        paper_path=args.paper_path,
        protected_executable=str(ROOT / "scripts" / "strict_run" / "g0_protected.py"),
        provider_path=str(ROOT / "scripts" / "g0" / "provider.py"),
        comparison_tool=str(ROOT / "scripts" / "g0" / "compare_manifests.py"),
        validator_path=str(ROOT / "src" / "provenance" / "strict_v6.py"),
        timeout_seconds=args.timeout_seconds,
    )
    sys.stdout.buffer.write(
        canonical_json(
            {
                "sealed_run_id": result.sealed_run_id,
                "run_root": result.run_root,
                "checkpoint": result.checkpoint.payload["checkpoint"],
                "checkpoint_sha256": result.checkpoint.publication.evidence.sha256,
                "external_paths": dict(result.external_paths),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


