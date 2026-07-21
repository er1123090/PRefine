#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from experiment_env.audited import (
    audited_doctor,
    audited_variant,
    audited_variants,
    build_audited_plan,
    run_audited_smoke,
)


def emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="experiments7 audited experiments4/5 environment")
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="validate immutable base, audited overlay, hashes, and labels")
    commands.add_parser("list", help="list all 84 audited semantic variants")
    inspect = commands.add_parser("inspect", help="show one audited variant")
    inspect.add_argument("--variant", required=True)
    dry = commands.add_parser("dry-run", help="resolve a sealed entrypoint without executing legacy code")
    dry.add_argument("--variant", required=True)
    dry.add_argument("--entrypoint-index", type=int, default=0)
    dry.add_argument("--run-id", default="AUDITED_DRY_RUN")
    dry.add_argument("passthrough", nargs=argparse.REMAINDER)
    run = commands.add_parser("run", help="run the contained composite-environment smoke recipe")
    run.add_argument("--run-id", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "doctor":
        emit(audited_doctor(ROOT))
        return 0
    if args.command == "list":
        emit(audited_variants(ROOT))
        return 0
    if args.command == "inspect":
        emit(audited_variant(ROOT, args.variant))
        return 0
    if args.command == "dry-run":
        passthrough = args.passthrough[1:] if args.passthrough[:1] == ["--"] else args.passthrough
        emit(build_audited_plan(ROOT, args.variant, args.entrypoint_index, args.run_id, passthrough))
        return 0
    if args.command == "run":
        return run_audited_smoke(ROOT, args.run_id)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
