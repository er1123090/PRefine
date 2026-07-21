#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import load_canonical_json, print_summary
from strict_run import (
    ArtifactEvidence,
    ExternalTranscriptRecord,
    ExternalTranscriptRegistry,
    Publication,
    WriterIdentity,
    default_writer_policy,
    transcript_hash,
    validate_final_tree,
)
from strict_run.canonical import require_exact_keys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-parent", required=True)
    parser.add_argument("--sealed-run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--transcript-bundle", required=True)
    parser.add_argument("--cp6-transcript", required=True)
    args = parser.parse_args()
    bundle = require_exact_keys(
        load_canonical_json(args.transcript_bundle), {"transcripts"}, "transcript-bundle"
    )
    if not isinstance(bundle["transcripts"], list):
        raise ValueError("transcripts must be an array")
    cp6 = load_canonical_json(args.cp6_transcript)
    if not isinstance(cp6, dict):
        raise ValueError("CP6 transcript must be an object")
    cp6_digest = transcript_hash(cp6)
    records = [
        ExternalTranscriptRecord(args.transcript_bundle, ("transcripts", index))
        for index, item in enumerate(bundle["transcripts"])
        if transcript_hash(item) != cp6_digest
    ]
    records.append(ExternalTranscriptRecord(args.cp6_transcript))
    with ExternalTranscriptRegistry(
        records,
        strict_parent=args.strict_parent,
        sealed_run_id=args.sealed_run_id,
        run_root=args.run_root,
    ) as registry:
        registered = registry.get_exact(cp6_digest)
        if not isinstance(registered, dict):
            raise ValueError("registered CP6 transcript must be an object")
        writer = WriterIdentity.from_dict(registered["writer"])
        relative_path = str(registered["relative_path"])
        policy = default_writer_policy()
        cp6_publication = Publication(
            ArtifactEvidence(
                relative_path,
                str(registered["artifact_type"]),
                registered["sha256"],
                registered["bytes"],
                dict(registered["identity"]),
                writer,
                policy.authorize(relative_path, writer).rule_id,
                cp6_digest,
            ),
            registered,
        )
        report = validate_final_tree(
            args.strict_parent,
            args.sealed_run_id,
            args.run_root,
            registry,
            cp6_publication,
        )
    print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
