from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .catalog import all_profiles, doctor, profile, recipes_for_profile
from .runner import build_plan, run_recipe


DEFAULT_PROFILE = "exp6_base_eval6"


def _emit(value: Any, as_json: bool) -> None:
    if as_json:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    elif isinstance(value, list):
        for item in value:
            print("\t".join(str(item[key]) for key in sorted(item)))
    else:
        print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _passthrough(value: list[str]) -> list[str]:
    return value[1:] if value[:1] == ["--"] else value


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="experiments7 runnable environment facade")
    result.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    subparsers = result.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="list published profiles and sealed recipes")
    list_parser.add_argument("--profile", default=None)
    list_parser.add_argument("--json", action="store_true")

    doctor_parser = subparsers.add_parser("doctor", help="verify snapshot hashes, seal, and containment")
    doctor_parser.add_argument("--profile", default=DEFAULT_PROFILE)
    doctor_parser.add_argument("--json", action="store_true")

    dry_parser = subparsers.add_parser("dry-run", help="render and validate a recipe without creating a run")
    dry_parser.add_argument("--profile", default=DEFAULT_PROFILE)
    dry_parser.add_argument("--recipe", required=True)
    dry_parser.add_argument("--run-id", default="DRY_RUN")
    dry_parser.add_argument("--json", action="store_true")
    dry_parser.add_argument("passthrough", nargs=argparse.REMAINDER)

    run_parser = subparsers.add_parser("run", help="execute a sealed recipe in a new experiments7 run directory")
    run_parser.add_argument("--profile", default=DEFAULT_PROFILE)
    run_parser.add_argument("--recipe", required=True)
    run_parser.add_argument("--run-id", required=True)
    run_parser.add_argument("passthrough", nargs=argparse.REMAINDER)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    repo_root = args.repo_root.resolve(strict=True)
    if args.command == "list":
        if args.profile:
            profile_value = profile(repo_root, args.profile)
            recipes = recipes_for_profile(repo_root, profile_value)
            value = [
                {
                    "description": recipe.get("description", ""),
                    "profile_id": args.profile,
                    "recipe_id": recipe_id,
                    "variant_label": profile_value["variant_label"],
                }
                for recipe_id, recipe in sorted(recipes["recipes"].items())
            ]
        else:
            value = all_profiles(repo_root)
        _emit(value, args.json)
        return 0
    if args.command == "doctor":
        _emit(doctor(repo_root, args.profile), args.json)
        return 0
    if args.command == "dry-run":
        runs_root = (repo_root / "runs").resolve(strict=True)
        plan = build_plan(
            repo_root,
            args.profile,
            args.recipe,
            runs_root / args.run_id,
            _passthrough(args.passthrough),
        )
        _emit(plan, args.json)
        return 0
    if args.command == "run":
        return run_recipe(
            repo_root,
            args.profile,
            args.recipe,
            args.run_id,
            _passthrough(args.passthrough),
        )
    raise AssertionError(args.command)


if __name__ == "__main__":
    sys.exit(main())
