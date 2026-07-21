"""Standard-library CLI for the neutral experiments7 facade."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .planner import inspect_variant, profile_selection
from .registry import FacadeError, load_bundle
from .runner import compare_reference, emit_selection, run_external, run_fixture
from .selection import canonical_bytes


def _emit(value: Any) -> None:
    sys.stdout.buffer.write(canonical_bytes(value))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list neutral families and selectable variants")
    list_parser.add_argument("--json", action="store_true", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect one exact variant")
    inspect_parser.add_argument("--variant-id", required=True)
    inspect_parser.add_argument("--json", action="store_true", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate and expand one exact profile")
    validate_parser.add_argument("--profile-id", required=True)
    validate_parser.add_argument("--json", action="store_true", required=True)

    plan_parser = subparsers.add_parser("plan", help="write a canonical selection plan")
    plan_parser.add_argument("--profile-id", required=True)
    plan_parser.add_argument("--output-dir", required=True)
    plan_parser.add_argument("--json", action="store_true", required=True)

    run_parser = subparsers.add_parser("run", help="run a fixture route or fail-closed external route")
    run_parser.add_argument("--profile-id", required=True)
    run_parser.add_argument("--output-dir", required=True)
    run_parser.add_argument("--execution", choices=("fixture", "external"), required=True)
    run_parser.add_argument("--fixture-id")
    run_parser.add_argument("--allow-external", action="store_true")
    run_parser.add_argument("--publication-id")
    run_parser.add_argument("--runtime-id")
    run_parser.add_argument("--run-config-id")
    run_parser.add_argument("--json", action="store_true", required=True)

    reference_parser = subparsers.add_parser(
        "compare-reference", help="compare reference routing without executing a historical parser"
    )
    reference_parser.add_argument("--profile-id", required=True)
    reference_parser.add_argument("--fixture-id", required=True)
    reference_parser.add_argument("--output-dir", required=True)
    reference_parser.add_argument("--json", action="store_true", required=True)
    return parser


def dispatch(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    bundle = load_bundle(root)
    if args.command == "list":
        return {
            "schema": "experiments7-g2-list/v1",
            "family_count": len(bundle.families),
            "variant_count": len(bundle.variants),
            "families": [bundle.families[key] for key in sorted(bundle.families)],
            "variants": [bundle.variants[key] for key in sorted(bundle.variants)],
            "control_hashes": dict(bundle.hashes),
        }
    if args.command == "inspect":
        return inspect_variant(bundle, args.variant_id)
    if args.command == "validate":
        selection = profile_selection(
            bundle, args.profile_id, execution="validate", run_id=None, output_root=None
        )
        return {"schema": "experiments7-g2-profile-validation/v1", "state": "PASS", "selection": selection}
    if args.command == "plan":
        _, selection = emit_selection(bundle, args.profile_id, args.output_dir, execution="plan")
        return {"schema": "experiments7-g2-plan-result/v1", "state": "PASS", "selection": selection}
    if args.command == "run" and args.execution == "fixture":
        if not args.fixture_id:
            raise FacadeError("FIXTURE_ID_REQUIRED")
        if args.publication_id or args.runtime_id or args.run_config_id:
            raise FacadeError("EXTERNAL_CONTRACT_NOT_ALLOWED_FOR_FIXTURE")
        return run_fixture(bundle, args.profile_id, args.output_dir, args.fixture_id)
    if args.command == "run":
        if args.fixture_id:
            raise FacadeError("FIXTURE_ID_NOT_ALLOWED_FOR_EXTERNAL")
        return run_external(
            bundle,
            args.profile_id,
            args.output_dir,
            allow_external=args.allow_external,
            publication_id=args.publication_id,
            runtime_id=args.runtime_id,
            run_config_id=args.run_config_id,
        )
    if args.command == "compare-reference":
        return compare_reference(bundle, args.profile_id, args.output_dir, args.fixture_id)
    raise FacadeError("UNKNOWN_COMMAND")


def main(argv: list[str] | None = None, *, root: Path | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    experiments7 = (root or Path(__file__).resolve().parents[2]).resolve()
    try:
        _emit(dispatch(args, experiments7))
        return 0
    except FacadeError as exc:
        report = {
            "schema": "experiments7-g2-facade-error/v1",
            "state": "BLOCKED",
            "reason_codes": [exc.code],
        }
        if exc.detail is not None:
            report["detail"] = exc.detail
        _emit(report)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
