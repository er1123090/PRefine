"""Command-line interface for experiments7 navigation catalogs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .catalog import CATALOG_PATHS, LayoutError, build_catalogs, validate_layout, write_catalogs
from .readable_code import ReadableCodeError, sync_readable_code


def _emit(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="build and inspect experiments7 navigation catalogs")
    result.add_argument("--root", type=Path, required=True, help=argparse.SUPPRESS)
    commands = result.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build-indexes", help="regenerate deterministic catalog JSON")
    build.add_argument("--check", action="store_true", help="fail if checked-in catalogs are stale")
    sync = commands.add_parser("sync-code", help="materialize audited code in readable folders")
    sync.add_argument("--check", action="store_true", help="fail if readable code copies drift")
    commands.add_parser("validate", help="validate counts, containment, files, and determinism")
    commands.add_parser("list", help="list catalog files")
    show = commands.add_parser("show", help="print one catalog")
    show.add_argument(
        "category",
        choices=("data", "experiments", "methods", "paper-outputs", "results", "runs"),
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    root = args.root
    if args.command == "build-indexes":
        write_catalogs(root, check=args.check)
        _emit({"check": args.check, "status": "ok"})
        return 0
    if args.command == "sync-code":
        _emit({"check": args.check, **sync_readable_code(root, check=args.check)})
        return 0
    if args.command == "validate":
        _emit(validate_layout(root))
        return 0
    if args.command == "list":
        _emit({"catalogs": list(CATALOG_PATHS), "status": "ok"})
        return 0
    if args.command == "show":
        lookup = {
            "data": "data/datasets_index.json",
            "experiments": "experiments/index.json",
            "methods": "methods/methods_index.json",
            "paper-outputs": "outputs/paper_outputs_index.json",
            "results": "results/results_index.json",
            "runs": "outputs/runs_index.json",
        }
        documents = build_catalogs(root)
        _emit(documents[lookup[args.category]])
        return 0
    raise AssertionError(args.command)


def entrypoint(argv: list[str] | None = None) -> int:
    try:
        return main(argv)
    except (LayoutError, ReadableCodeError) as exc:
        print(f"layout error: {exc}")
        return 2
