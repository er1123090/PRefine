"""Unified validation-first execution surface for the standalone method."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
from typing import Sequence

from .contracts import (
    ContextMode,
    InputContractError,
    ProviderError,
    Step2OutputNames,
)
from .jsonl import append_jsonl_streams, read_examples_jsonl, validate_output_root
from .providers import OpenAICompatibleProvider
from .step1 import build_and_write_examples
from .step2_common import (
    _execute_step2,
    load_manifest_v1,
    normalize_context,
    normalize_difficulties,
)
from .step2_multi import build_multi_cases
from .step2_single import build_single_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ours-memory2")
    stages = parser.add_subparsers(dest="stage", required=True)

    step1 = stages.add_parser("step1", help="preference-memory construction")
    step1_commands = step1.add_subparsers(dest="step1_command", required=True)
    build = step1_commands.add_parser("build", help="build example memories")
    build.add_argument("--input", required=True, help="strict UTF-8 examples JSONL")
    _add_output_root(build)
    _add_provider_options(build)

    step2 = stages.add_parser("step2", help="memory-conditioned action inference")
    step2_commands = step2.add_subparsers(dest="step2_command", required=True)
    single = step2_commands.add_parser("single", help="single-turn inference")
    _add_step2_options(single, multi=False)
    multi = step2_commands.add_parser("multi", help="multi-turn inference")
    _add_step2_options(multi, multi=True)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        output_root = validate_output_root(args.output_root)
        _validate_execution_args(args)
        if args.stage == "step1":
            return _run_step1(args, output_root)
        return _run_step2(args, output_root)
    except InputContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except OSError:
        print("error: output operation failed", file=sys.stderr)
        return 1


def _add_output_root(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-root", required=True, help="new output directory")


def _add_provider_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--endpoint", required=True, help="OpenAI-compatible endpoint")
    parser.add_argument("--model", required=True, help="model identifier")
    parser.add_argument("--key-env", help="environment-variable name containing a key")
    parser.add_argument("--timeout", type=float, default=30.0, help="positive timeout seconds")


def _add_step2_options(parser: argparse.ArgumentParser, *, multi: bool) -> None:
    parser.add_argument("--manifest", required=True, help="manifest v1 JSON")
    if multi:
        parser.add_argument("--input-shape", choices=("auto", "grouped", "list"), default="auto")
    parser.add_argument("--difficulty", required=True)
    parser.add_argument("--context", required=True)
    _add_output_root(parser)
    _add_provider_options(parser)


def _validate_execution_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        raise InputContractError("must be finite and positive", path="provider.timeout")
    if not args.endpoint.strip():
        raise InputContractError("must be nonempty", path="provider.endpoint")
    if not args.model.strip():
        raise InputContractError("must be nonempty", path="provider.model")
    if args.key_env is not None and not args.key_env.strip():
        raise InputContractError("must be nonempty when supplied", path="provider.key_env")
    if args.stage == "step2":
        normalize_context(args.context)
        normalize_difficulties(args.difficulty)


def _provider(args: argparse.Namespace) -> OpenAICompatibleProvider:
    return OpenAICompatibleProvider(
        endpoint=args.endpoint,
        model=args.model,
        key_env=args.key_env,
        timeout=args.timeout,
    )


def _run_step1(args: argparse.Namespace, output_root: Path) -> int:
    examples = read_examples_jsonl(args.input)
    provider = _provider(args)
    build_and_write_examples(examples, provider, output_root)
    return 0


def _run_step2(args: argparse.Namespace, output_root: Path) -> int:
    context = ContextMode(args.context)
    input_shape = getattr(args, "input_shape", "auto")
    resources = load_manifest_v1(args.manifest, multi_shape=input_shape)
    if args.step2_command == "single":
        cases = build_single_cases(resources, difficulty=args.difficulty)
    else:
        cases = build_multi_cases(resources, difficulty=args.difficulty)

    provider = _provider(args)
    run = _execute_step2(
        resources=resources,
        cases=cases,
        context=context,
        model=args.model,
        provider=provider,
        mode=args.step2_command,
    )

    names = Step2OutputNames()
    append_jsonl_streams(
        output_root,
        {names.results: run.results, names.diagnostics: run.diagnostics},
    )
    return 0
