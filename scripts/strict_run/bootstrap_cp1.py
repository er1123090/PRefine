#!/usr/bin/env python3
"""Seal CP1 from immutable external readonly paper-capture evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from strict_run import canonical_json  # noqa: E402
from strict_run.cp1_bootstrap import bootstrap_cp1  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate an already captured readonly paper layout, publish the "
            "CP1 registry/inventory stage, and seal checkpoint CP1."
        )
    )
    parser.add_argument("--strict-parent", required=True)
    parser.add_argument("--sealed-run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--external-capture", required=True)
    parser.add_argument("--external-capture-execution", required=True)
    parser.add_argument("--external-cp0-transcript", required=True)
    parser.add_argument("--external-task-evidence", required=True)
    parser.add_argument("--external-stage-bundle", required=True)
    parser.add_argument("--external-cp1-transcript", required=True)
    parser.add_argument("--external-stage-transcripts", required=True)
    args = parser.parse_args()

    result = bootstrap_cp1(
        strict_parent=args.strict_parent,
        sealed_run_id=args.sealed_run_id,
        run_root=args.run_root,
        external_capture=args.external_capture,
        external_capture_execution=args.external_capture_execution,
        external_cp0_transcript=args.external_cp0_transcript,
        external_task_evidence=args.external_task_evidence,
        external_stage_bundle=args.external_stage_bundle,
        external_cp1_transcript=args.external_cp1_transcript,
        external_stage_transcripts=args.external_stage_transcripts,
    )
    sys.stdout.buffer.write(
        canonical_json(
            {
                "sealed_run_id": result.sealed_run_id,
                "run_root": result.run_root,
                "checkpoint": result.checkpoint.payload["checkpoint"],
                "checkpoint_sha256": result.checkpoint.publication.evidence.sha256,
                "task_evidence_sha256": result.task_evidence_sha256,
                "stage_paths": sorted(
                    publication.evidence.relative_path
                    for publication in result.stage_publications
                ),
                "external_paths": dict(result.external_paths),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
