#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import preflight_existing_external_output, print_summary
from strict_run import (
    ExternalTranscriptRecord,
    ExternalTranscriptRegistry,
    WriterIdentity,
    accept_reservation,
    default_writer_policy,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-parent", required=True)
    parser.add_argument("--sealed-run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--reservation-transcript", required=True)
    parser.add_argument("--writer-json", required=True)
    parser.add_argument("--external-transcript", required=True)
    args = parser.parse_args()
    records = [
        ExternalTranscriptRecord(args.reservation_transcript),
        ExternalTranscriptRecord(args.writer_json),
    ]
    with (
        preflight_existing_external_output(
            args.external_transcript,
            strict_parent=args.strict_parent,
            sealed_run_id=args.sealed_run_id,
            run_root=args.run_root,
        ) as external_output,
        ExternalTranscriptRegistry(
            records,
            strict_parent=args.strict_parent,
            sealed_run_id=args.sealed_run_id,
            run_root=args.run_root,
        ) as registry,
    ):
        reservation_digest = registry.documents[args.reservation_transcript].sha256
        writer_digest = registry.documents[args.writer_json].sha256
        result = accept_reservation(
            args.strict_parent,
            args.sealed_run_id,
            args.run_root,
            registry,
            WriterIdentity.from_dict(registry.get_exact(writer_digest)),
            reservation_transcript_sha256=reservation_digest,
            policy=default_writer_policy(),
        )
        external_output.publish_json(result.transcript)
    print_summary(
        {
            "sealed_run_id": args.sealed_run_id,
            "run_root": args.run_root,
            "reservation_acceptance_sha256": result.evidence.sha256,
            "external_transcript": args.external_transcript,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
