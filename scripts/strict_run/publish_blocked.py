#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import load_canonical_json, print_summary, write_external_no_replace
from strict_run import WriterIdentity, default_writer_policy, publish_blocked


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-parent", required=True)
    parser.add_argument("--sealed-run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--reason-code", action="append", required=True)
    parser.add_argument("--controller-json", required=True)
    parser.add_argument("--external-transcripts", required=True)
    args = parser.parse_args()
    publications = publish_blocked(
        args.strict_parent,
        args.sealed_run_id,
        args.run_root,
        args.reason_code,
        WriterIdentity.from_dict(load_canonical_json(args.controller_json)),
        default_writer_policy(),
    )
    write_external_no_replace(
        args.external_transcripts,
        {"transcripts": [item.transcript for item in publications]},
        args.run_root,
    )
    print_summary(
        {
            "sealed_run_id": args.sealed_run_id,
            "run_root": args.run_root,
            "terminal": "BLOCKED",
            "reason_count": len(set(args.reason_code)),
            "external_transcripts": args.external_transcripts,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
