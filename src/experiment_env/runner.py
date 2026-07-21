from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .catalog import doctor, profile, recipes_for_profile
from .util import credential_presence, is_relative_to, redact_argv, write_json_exclusive


class RunError(RuntimeError):
    pass


RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
OUTPUT_LIKE_RE = re.compile(r"(?:output|log|result|save|cache|db|manifest|out[_-]?csv)", re.IGNORECASE)


def _expand(value: str, *, snapshot: Path, run_dir: Path) -> str:
    return (
        value.replace("{snapshot}", str(snapshot))
        .replace("{run_dir}", str(run_dir))
        .replace("{python}", sys.executable)
    )


def _flag_values(argv: list[str], flag: str) -> list[str]:
    values: list[str] = []
    index = 0
    while index < len(argv):
        value = argv[index]
        if value == flag:
            if index + 1 >= len(argv):
                raise RunError(f"missing value for output flag {flag}")
            values.append(argv[index + 1])
            index += 2
            continue
        if value.startswith(flag + "="):
            values.append(value.split("=", 1)[1])
        index += 1
    return values


def _validate_output_paths(argv: list[str], recipe: dict[str, Any], snapshot: Path, run_dir: Path) -> None:
    declared = set(recipe.get("output_flags", []))
    for flag in declared:
        values = _flag_values(argv, flag)
        if flag in recipe.get("required_output_flags", []) and not values:
            raise RunError(f"recipe is missing required contained output flag: {flag}")
        for value in values:
            candidate = Path(value)
            if not candidate.is_absolute():
                candidate = snapshot / candidate
            resolved = candidate.resolve(strict=False)
            if not is_relative_to(resolved, run_dir.resolve(strict=False)):
                raise RunError(f"output flag {flag} escapes the experiments7 run directory: {value}")
    for value in argv:
        if not value.startswith("--"):
            continue
        flag = value.split("=", 1)[0]
        if OUTPUT_LIKE_RE.search(flag) and flag not in declared:
            raise RunError(f"output-like passthrough flag is not declared by the sealed recipe: {flag}")


def build_plan(
    repo_root: Path,
    profile_id: str,
    recipe_id: str,
    run_dir: Path,
    passthrough: list[str] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    profile_value = profile(repo_root, profile_id)
    recipes = recipes_for_profile(repo_root, profile_value)
    try:
        recipe = dict(recipes["recipes"][recipe_id])
    except KeyError as exc:
        raise RunError(f"unknown recipe for {profile_id}: {recipe_id}") from exc
    snapshot = profile_value["snapshot_path"].resolve(strict=True)
    command = [
        _expand(value, snapshot=snapshot, run_dir=run_dir)
        for value in recipe["command"] + recipe.get("default_args", []) + list(passthrough or [])
    ]
    if not command:
        raise RunError("empty recipe command")
    executable = Path(command[0])
    if not executable.is_absolute():
        executable = snapshot / executable
        command[0] = str(executable)
    resolved_executable = executable.resolve(strict=True)
    python_executable = Path(sys.executable).resolve(strict=True)
    if resolved_executable != python_executable and not is_relative_to(resolved_executable, snapshot):
        raise RunError("recipe executable is outside the sealed snapshot")
    if resolved_executable != python_executable and not os.access(resolved_executable, os.X_OK):
        raise RunError(f"recipe executable is not executable: {resolved_executable}")
    _validate_output_paths(command, recipe, snapshot, run_dir)
    return {
        "argv": command,
        "argv_redacted": redact_argv(command),
        "description": recipe.get("description", ""),
        "profile_id": profile_id,
        "recipe_id": recipe_id,
        "run_dir": str(run_dir),
        "snapshot": str(snapshot),
        "variant_label": profile_value["variant_label"],
    }


def run_recipe(
    repo_root: Path,
    profile_id: str,
    recipe_id: str,
    run_id: str,
    passthrough: list[str] | None = None,
) -> int:
    repo_root = repo_root.resolve(strict=True)
    if not RUN_ID_RE.fullmatch(run_id):
        raise RunError(f"invalid run id: {run_id!r}")
    runs_root = (repo_root / "runs").resolve(strict=True)
    run_dir = runs_root / run_id
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(run_dir)
    doctor_result = doctor(repo_root, profile_id)
    plan = build_plan(repo_root, profile_id, recipe_id, run_dir, passthrough)
    run_dir.mkdir(mode=0o700)
    if run_dir.is_symlink() or run_dir.resolve(strict=True).parent != runs_root:
        raise RunError("run directory containment check failed")
    (run_dir / "tmp").mkdir(mode=0o700)
    environment = dict(os.environ)
    environment.update(
        {
            "EXPERIMENTS7_ROOT": str(repo_root),
            "EXPERIMENT_RUN_DIR": str(run_dir),
            "EXPERIMENT_SNAPSHOT_ROOT": plan["snapshot"],
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "TMPDIR": str(run_dir / "tmp"),
        }
    )
    manifest = {
        "argv": plan["argv_redacted"],
        "credential_environment": credential_presence(environment),
        "doctor": doctor_result,
        "profile_id": profile_id,
        "recipe_id": recipe_id,
        "run_dir": str(run_dir),
        "schema": "experiments7-run-manifest/v1",
        "snapshot": plan["snapshot"],
        "variant_label": plan["variant_label"],
        "working_directory": plan["snapshot"],
    }
    write_json_exclusive(run_dir / "run-manifest.json", manifest)
    try:
        completed = subprocess.run(plan["argv"], cwd=plan["snapshot"], env=environment, check=False)
        exit_code = completed.returncode
    except OSError as exc:
        exit_code = 126
        error = f"{type(exc).__name__}: {exc}"
    else:
        error = None
    write_json_exclusive(
        run_dir / "result.json",
        {
            "error": error,
            "exit_code": exit_code,
            "profile_id": profile_id,
            "recipe_id": recipe_id,
            "schema": "experiments7-run-result/v1",
        },
    )
    return exit_code
