#!/usr/bin/env python3
from __future__ import annotations

import argparse

from _common import (
    open_pre_reservation_json,
    preflight_external_output,
    print_summary,
)
from strict_run import (
    ReservationInputs,
    WriterIdentity,
    default_writer_policy,
    reserve_strict_run,
)
from strict_run.canonical import require_exact_keys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-parent", required=True)
    parser.add_argument("--sealed-run-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--inputs-json", required=True)
    parser.add_argument("--external-transcript", required=True)
    args = parser.parse_args()
    with (
        open_pre_reservation_json(
            args.inputs_json,
            strict_parent=args.strict_parent,
            sealed_run_id=args.sealed_run_id,
            run_root=args.run_root,
        ) as input_document,
        preflight_external_output(
            args.external_transcript,
            strict_parent=args.strict_parent,
            sealed_run_id=args.sealed_run_id,
            run_root=args.run_root,
        ) as external_output,
    ):
        row = require_exact_keys(
            input_document.value,
            {
                "project_id", "owner_seed", "writer", "tool_sha256",
                "configuration_sha256", "runtime_sha256",
                "environment_path_policy_sha256", "canonical_argv",
            },
            "reservation-cli-input",
        )
        if not isinstance(row["canonical_argv"], list):
            raise ValueError("canonical_argv must be an array")
        inputs = ReservationInputs(
            project_id=str(row["project_id"]),
            owner_seed=str(row["owner_seed"]),
            writer=WriterIdentity.from_dict(row["writer"]),
            tool_sha256=str(row["tool_sha256"]),
            configuration_sha256=str(row["configuration_sha256"]),
            runtime_sha256=str(row["runtime_sha256"]),
            environment_path_policy_sha256=str(
                row["environment_path_policy_sha256"]
            ),
            canonical_argv=tuple(row["canonical_argv"]),
        )
        result = reserve_strict_run(
            args.strict_parent,
            args.sealed_run_id,
            args.run_root,
            inputs,
            policy=default_writer_policy(),
        )
        external_output.publish_json(result.transcript)
    print_summary(
        {
            "sealed_run_id": result.sealed_run_id,
            "run_root": result.run_root,
            "reservation_sha256": result.evidence.sha256,
            "external_transcript": args.external_transcript,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
