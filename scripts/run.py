#!/usr/bin/env python3
"""Run VanillaLLM over one verified canonical prepared JSONL file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from exp7.methods.vanilla_llm import OpenAICompatibleProvider, run_prepared_file
from exp7.provenance.admission import (
    StrictJSONError,
    admit_regular_file,
    decode_strict_json,
)


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description=(
            "Run one manifest-bound prepared dataset through an OpenAI-compatible "
            "chat-completions endpoint."
        )
    )
    value.add_argument("--prepared-jsonl", type=Path, required=True)
    value.add_argument("--manifest", type=Path)
    value.add_argument("--manifest-sha256")
    value.add_argument("--tools-schema", type=Path, required=True)
    value.add_argument("--tools-schema-sha256")
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--checkpoint", type=Path)
    value.add_argument("--model", required=True)
    value.add_argument("--reasoning-effort")
    value.add_argument("--base-url")
    value.add_argument("--api-key-env", default="OPENAI_API_KEY")
    value.add_argument("--temperature", type=float)
    value.add_argument("--max-tokens", type=int)
    value.add_argument("--timeout", type=float)
    value.add_argument("--resume", action="store_true")
    value.add_argument("--output-root", type=Path, help=argparse.SUPPRESS)
    value.add_argument("--output-root-device", type=int, help=argparse.SUPPRESS)
    value.add_argument("--output-root-inode", type=int, help=argparse.SUPPRESS)
    value.add_argument("--experiment-config-path", help=argparse.SUPPRESS)
    value.add_argument("--experiment-config-sha256", help=argparse.SUPPRESS)
    value.add_argument("--experiment-id", help=argparse.SUPPRESS)
    value.add_argument("--evaluator-variant", help=argparse.SUPPRESS)
    return value


def _default_manifest(prepared: Path) -> Path:
    for parent in prepared.resolve().parents:
        candidate = parent / "manifest.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError("cannot locate manifest.json above prepared input")


def _uses_experiment_config(arguments: list[str]) -> bool:
    return any(
        argument == "--config" or argument.startswith("--config=")
        for argument in arguments
    )


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if _uses_experiment_config(arguments):
        from run_suite import main as run_suite_main

        return run_suite_main(arguments)

    args = parser().parse_args(arguments)
    output_identity_values = (
        args.output_root,
        args.output_root_device,
        args.output_root_inode,
    )
    if any(value is not None for value in output_identity_values) and not all(
        value is not None for value in output_identity_values
    ):
        raise ValueError(
            "output root path, device, and inode must be supplied together"
        )
    manifest = args.manifest or _default_manifest(args.prepared_jsonl)
    admitted_schema = admit_regular_file(args.tools_schema, label="tools schema")
    if (
        args.tools_schema_sha256 is not None
        and admitted_schema.sha256 != args.tools_schema_sha256
    ):
        raise ValueError("tools schema SHA-256 does not match admitted bytes")
    try:
        schema = decode_strict_json(admitted_schema.payload, label="tools schema")
    except StrictJSONError as exc:
        raise ValueError(str(exc)) from exc
    if not isinstance(schema, list) or not all(isinstance(item, dict) for item in schema):
        raise ValueError("tools schema must be a JSON array of objects")
    checkpoint = args.checkpoint or args.output.with_suffix(
        args.output.suffix + ".checkpoint.jsonl"
    )
    provider = OpenAICompatibleProvider(
        api_key_env=args.api_key_env,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout=args.timeout,
    )
    summary = run_prepared_file(
        prepared_path=args.prepared_jsonl,
        manifest_path=manifest,
        output_path=args.output,
        checkpoint_path=checkpoint,
        tools_schema=schema,
        model_name=args.model,
        inference=provider,
        reasoning_effort=args.reasoning_effort,
        expected_manifest_sha256=args.manifest_sha256,
        resume=args.resume,
        output_root=args.output_root,
        expected_output_root_identity=(
            (args.output_root_device, args.output_root_inode)
            if args.output_root is not None
            else None
        ),
    )
    print(
        json.dumps(
            {
                "checkpoint": str(summary.checkpoint_path),
                "errors": summary.error_count,
                "inferred": summary.inferred_count,
                "input": summary.input_count,
                "output": str(summary.output_path),
                "resumed": summary.resumed_count,
                "status": summary.status,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
