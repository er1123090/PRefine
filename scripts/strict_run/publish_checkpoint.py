#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import ExitStack

from _common import (
    load_canonical_json,
    preflight_existing_external_output,
    print_summary,
)
from strict_run import (
    ArtifactEvidence,
    ExternalTranscriptRecord,
    ExternalTranscriptRegistry,
    Publication,
    WriterIdentity,
    default_writer_policy,
    publish_checkpoint,
    transcript_hash,
)
from strict_run.canonical import fail, require_exact_keys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-parent", required=True)
    parser.add_argument("--sealed-run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--checkpoint", type=int, choices=range(7), required=True)
    parser.add_argument("--stage-bundle", required=True)
    parser.add_argument("--controller-json", required=True)
    parser.add_argument("--previous-checkpoint-transcript")
    parser.add_argument("--prior-transcript-bundle")
    parser.add_argument("--prior-transcript-count", type=int)
    parser.add_argument(
        "--external-evidence-document",
        action="append",
        default=[],
        help="register one whole canonical external evidence document",
    )
    parser.add_argument("--external-transcript", required=True)
    parser.add_argument("--external-stage-transcripts", required=True)
    args = parser.parse_args()
    if args.external_transcript == args.external_stage_transcripts:
        fail(
            "EXTERNAL_OUTPUT_PATH_COLLISION",
            "checkpoint external output paths must be distinct",
        )
    prior_bundle_supplied = args.prior_transcript_bundle is not None
    prior_count_supplied = args.prior_transcript_count is not None
    if prior_bundle_supplied != prior_count_supplied:
        fail(
            "PRIOR_TRANSCRIPT_BUNDLE_INVALID",
            "prior transcript bundle and exact count must be supplied together",
        )
    if args.checkpoint >= 3 and not prior_bundle_supplied:
        fail(
            "PRIOR_TRANSCRIPT_BUNDLE_REQUIRED",
            "CP3 through CP6 require the complete typed prior transcript bundle",
        )
    if prior_count_supplied and not 0 < args.prior_transcript_count <= 1_000_000:
        fail(
            "PRIOR_TRANSCRIPT_BUNDLE_INVALID",
            "prior transcript count must be an exact positive bounded integer",
        )
    if len(set(args.external_evidence_document)) != len(
        args.external_evidence_document
    ):
        fail(
            "EXTERNAL_EVIDENCE_DOCUMENT_INVALID",
            "external evidence document paths must be unique",
        )
    with ExitStack() as stack:
        external_transcript = stack.enter_context(
            preflight_existing_external_output(
                args.external_transcript,
                strict_parent=args.strict_parent,
                sealed_run_id=args.sealed_run_id,
                run_root=args.run_root,
            )
        )
        external_stage_transcripts = stack.enter_context(
            preflight_existing_external_output(
                args.external_stage_transcripts,
                strict_parent=args.strict_parent,
                sealed_run_id=args.sealed_run_id,
                run_root=args.run_root,
            )
        )
        bundle = require_exact_keys(
            load_canonical_json(args.stage_bundle), {"publications"}, "stage-bundle"
        )
        if not isinstance(bundle["publications"], list):
            raise ValueError("publications must be an array")
        records = [
            ExternalTranscriptRecord(
                args.stage_bundle,
                ("publications", index, "transcript"),
            )
            for index in range(len(bundle["publications"]))
        ]
        records.extend(
            ExternalTranscriptRecord(path)
            for path in args.external_evidence_document
        )
        if args.previous_checkpoint_transcript is not None:
            records.append(
                ExternalTranscriptRecord(args.previous_checkpoint_transcript)
            )
        if args.prior_transcript_bundle is not None:
            records.extend(
                ExternalTranscriptRecord(
                    args.prior_transcript_bundle,
                    ("transcripts", index),
                )
                for index in range(args.prior_transcript_count)
            )
        registry = stack.enter_context(
            ExternalTranscriptRegistry(
                records,
                strict_parent=args.strict_parent,
                sealed_run_id=args.sealed_run_id,
                run_root=args.run_root,
            )
        )
        held_bundle = require_exact_keys(
            registry.documents[args.stage_bundle].value,
            {"publications"},
            "stage-bundle",
        )
        if not isinstance(held_bundle["publications"], list):
            raise ValueError("publications must be an array")
        if args.prior_transcript_bundle is not None:
            held_prior = require_exact_keys(
                registry.documents[args.prior_transcript_bundle].value,
                {"transcripts"},
                "prior-transcript-bundle",
            )
            if (
                not isinstance(held_prior["transcripts"], list)
                or len(held_prior["transcripts"]) != args.prior_transcript_count
            ):
                fail(
                    "PRIOR_TRANSCRIPT_BUNDLE_INVALID",
                    "held prior transcript bundle differs from its exact count",
                )
        publications = []
        for raw in held_bundle["publications"]:
            row = require_exact_keys(raw, {"evidence", "transcript"}, "stage-publication")
            if not isinstance(row["transcript"], dict):
                raise ValueError("stage transcript must be an object")
            digest = transcript_hash(row["transcript"])
            registered = registry.get_exact(digest)
            if registered != row["transcript"]:
                raise ValueError("stage transcript differs from typed registry")
            publications.append(
                Publication(ArtifactEvidence.from_dict(row["evidence"]), registered)
            )
        policy = default_writer_policy()
        previous_publication = None
        if args.previous_checkpoint_transcript is not None:
            previous = registry.documents[args.previous_checkpoint_transcript].value
            if not isinstance(previous, dict):
                raise ValueError("previous checkpoint transcript must be an object")
            previous_digest = transcript_hash(previous)
            registered = registry.get_exact(previous_digest)
            writer = WriterIdentity.from_dict(registered["writer"])
            relative_path = str(registered["relative_path"])
            previous_publication = Publication(
                ArtifactEvidence(
                    relative_path,
                    str(registered["artifact_type"]),
                    registered["sha256"],
                    registered["bytes"],
                    dict(registered["identity"]),
                    writer,
                    policy.authorize(relative_path, writer).rule_id,
                    previous_digest,
                ),
                registered,
            )
        result = publish_checkpoint(
            args.strict_parent,
            args.sealed_run_id,
            args.run_root,
            args.checkpoint,
            publications,
            WriterIdentity.from_dict(load_canonical_json(args.controller_json)),
            policy,
            transcript_registry=registry,
            previous_checkpoint_publication=previous_publication,
        )
        external_transcript.publish_json(result.transcript)
        external_stage_transcripts.publish_json(
            {"transcripts": [item.transcript for item in result.stage_publications]}
        )
    print_summary(
        {
            "sealed_run_id": args.sealed_run_id,
            "run_root": args.run_root,
            "checkpoint": args.checkpoint,
            "checkpoint_sha256": result.sha256,
            "external_transcript": args.external_transcript,
            "external_stage_transcripts": args.external_stage_transcripts,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
