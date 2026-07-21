"""Owned-output containment and fixture/external dispatch."""

from __future__ import annotations

import json
import os
from pathlib import Path, PurePath
import re
import stat
from typing import Any

from .external import execute_external_contract, validate_external_contract
from .planner import profile_selection
from .registry import Bundle, FacadeError, require
from .selection import canonical_bytes, sha256_file


FIXTURE_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")


def _lexists(path: Path) -> bool:
    return os.path.lexists(path)


def contained_output(root: Path, value: str) -> tuple[Path, str]:
    require(isinstance(value, str) and value and "\x00" not in value, "INVALID_OUTPUT_DIR")
    raw = Path(value)
    require(".." not in PurePath(value).parts, "OUTPUT_DOT_DOT_FORBIDDEN", value)
    base = root / "runs"
    require(base.is_dir() and not base.is_symlink(), "UNSAFE_RUNS_ROOT", str(base))
    candidate = raw if raw.is_absolute() else root / raw
    normalized = Path(os.path.abspath(candidate))
    base_resolved = base.resolve(strict=True)
    try:
        relative = normalized.relative_to(base_resolved)
    except ValueError as exc:
        raise FacadeError("OUTPUT_ESCAPE", value) from exc
    require(relative.parts and all(part not in {"", ".", ".."} for part in relative.parts),
            "INVALID_RUN_ID", value)
    cursor = base_resolved
    for part in relative.parts[:-1]:
        cursor = cursor / part
        if _lexists(cursor):
            require(cursor.is_dir() and not cursor.is_symlink(), "OUTPUT_ANCESTOR_UNSAFE", str(cursor))
        else:
            cursor.mkdir(mode=0o700)
    target = base_resolved / relative
    require(not _lexists(target), "OUTPUT_COLLISION", str(target))
    return target, relative.as_posix()


def create_output_dir(root: Path, value: str) -> tuple[Path, str]:
    target, run_id = contained_output(root, value)
    target.mkdir(mode=0o700)
    return target, run_id


def write_noreplace(path: Path, payload: bytes) -> None:
    require(not _lexists(path), "OUTPUT_COLLISION", str(path))
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(temporary, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            require(written > 0, "OUTPUT_SHORT_WRITE", str(path))
            view = view[written:]
        os.fsync(fd)
        os.close(fd)
        fd = -1
        os.link(temporary, path, follow_symlinks=False)
        temporary.unlink()
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.chmod(path, 0o444)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def seal_output_tree(target: Path) -> None:
    """Make a terminal run tree read-only after its final result/block record."""

    require(target.is_dir() and not target.is_symlink(), "UNSAFE_OUTPUT_TREE", str(target))
    directories: list[Path] = []
    for current, names, files in os.walk(target, topdown=True, followlinks=False):
        directory = Path(current)
        directories.append(directory)
        for name in [*names, *files]:
            path = directory / name
            mode = path.lstat().st_mode
            require(not path.is_symlink(), "UNSAFE_OUTPUT_TREE_ENTRY", str(path))
            require(stat.S_ISDIR(mode) or stat.S_ISREG(mode),
                    "UNSAFE_OUTPUT_TREE_ENTRY", str(path))
        for name in files:
            os.chmod(directory / name, 0o444)
    for directory in reversed(directories):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.chmod(directory, 0o555)


def emit_selection(
    bundle: Bundle,
    profile_id: str,
    output_dir: str,
    *,
    execution: str,
) -> tuple[Path, dict[str, Any]]:
    target, run_id = create_output_dir(bundle.root, output_dir)
    selection = profile_selection(
        bundle, profile_id, execution=execution, run_id=run_id, output_root=target
    )
    write_noreplace(target / "selection.json", canonical_bytes(selection))
    return target, selection


def load_fixture(root: Path, fixture_id: str) -> tuple[dict[str, Any], str]:
    require(FIXTURE_ID.fullmatch(fixture_id or "") is not None, "INVALID_FIXTURE_ID", fixture_id)
    path = root / "runs/fixtures" / f"{fixture_id}.json"
    require(path.is_file() and not path.is_symlink(), "UNKNOWN_FIXTURE", fixture_id)
    digest, _ = sha256_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise FacadeError("INVALID_FIXTURE", fixture_id) from exc
    require(isinstance(value, dict), "INVALID_FIXTURE", fixture_id)
    return value, digest


def run_fixture(
    bundle: Bundle, profile_id: str, output_dir: str, fixture_id: str
) -> dict[str, Any]:
    target, selection = emit_selection(bundle, profile_id, output_dir, execution="fixture")
    fixture, fixture_hash = load_fixture(bundle.root, fixture_id)
    output = {
        "schema": "experiments7-g2-fixture-output/v1",
        "profile_id": profile_id,
        "fixture_id": fixture_id,
        "fixture_sha256": fixture_hash,
        "selection_sha256": selection["selection_sha256"],
        "selected_stages": selection["selected_stages"],
        "record_count": len(fixture.get("records", [])),
        "semantic_execution": False,
        "reason": "fixture validates deterministic routing only; origin semantics were not executed",
    }
    write_noreplace(target / "fixture-output.json", canonical_bytes(output))
    return {"state": "PASS", "selection": selection, "fixture_output": output}


def run_external(
    bundle: Bundle,
    profile_id: str,
    output_dir: str,
    *,
    allow_external: bool,
    publication_id: str | None = None,
    runtime_id: str | None = None,
    run_config_id: str | None = None,
) -> dict[str, Any]:
    require(allow_external, "ALLOW_EXTERNAL_REQUIRED")
    target, selection = emit_selection(bundle, profile_id, output_dir, execution="external")
    reference_only = [
        row["variant_id"]
        for row in selection["adapters"]
        if row["execution_kind"] == "reference_only"
    ]
    if reference_only:
        blocked = {
            "schema": "experiments7-g2-external-block/v1",
            "state": "BLOCKED",
            "reason_code": "REFERENCE_ONLY_EXECUTION_FORBIDDEN",
            "reference_variant_ids": reference_only,
            "selection_sha256": selection["selection_sha256"],
        }
        write_noreplace(target / "blocked.json", canonical_bytes(blocked))
        seal_output_tree(target)
        raise FacadeError("REFERENCE_ONLY_EXECUTION_FORBIDDEN", blocked)
    contract_ids = (publication_id, runtime_id, run_config_id)
    if not any(contract_ids):
        blocked = {
            "schema": "experiments7-g2-external-block/v1",
            "state": "BLOCKED",
            "reason_code": "EXTERNAL_PREREQUISITES_UNAVAILABLE",
            "external_prerequisites": selection["external_prerequisites"],
            "selection_sha256": selection["selection_sha256"],
        }
        write_noreplace(target / "blocked.json", canonical_bytes(blocked))
        seal_output_tree(target)
        raise FacadeError("EXTERNAL_PREREQUISITES_UNAVAILABLE", blocked)
    if not all(contract_ids):
        blocked = {
            "schema": "experiments7-g2-external-block/v1",
            "state": "BLOCKED",
            "reason_code": "EXTERNAL_CONTRACT_INCOMPLETE",
            "required_ids": ["publication_id", "runtime_id", "run_config_id"],
            "selection_sha256": selection["selection_sha256"],
        }
        write_noreplace(target / "blocked.json", canonical_bytes(blocked))
        seal_output_tree(target)
        raise FacadeError("EXTERNAL_CONTRACT_INCOMPLETE", blocked)
    try:
        contract = validate_external_contract(
            bundle,
            selection,
            target,
            publication_id=publication_id,
            runtime_id=runtime_id,
            run_config_id=run_config_id,
            source_environment=os.environ,
        )
        external_result = execute_external_contract(selection, target, contract)
        seal_output_tree(target)
        return {"state": "PASS", "selection": selection, "external_result": external_result}
    except FacadeError as exc:
        blocked_path = target / "blocked.json"
        if not _lexists(blocked_path):
            blocked = {
                "schema": "experiments7-g2-external-block/v1",
                "state": "BLOCKED",
                "reason_code": exc.code,
                "selection_sha256": selection["selection_sha256"],
            }
            if exc.detail is not None:
                blocked["detail"] = exc.detail
            write_noreplace(blocked_path, canonical_bytes(blocked))
        seal_output_tree(target)
        raise


def compare_reference(
    bundle: Bundle, profile_id: str, output_dir: str, fixture_id: str
) -> dict[str, Any]:
    target, selection = emit_selection(bundle, profile_id, output_dir, execution="reference_compare")
    reference = [row for row in selection["adapters"] if row["execution_kind"] == "reference_only"]
    require(len(reference) == 1, "REFERENCE_PROFILE_REQUIRED", profile_id)
    fixture, fixture_hash = load_fixture(bundle.root, fixture_id)
    comparison = {
        "schema": "experiments7-g2-reference-comparison/v1",
        "state": "PASS",
        "profile_id": profile_id,
        "reference_variant_id": reference[0]["variant_id"],
        "fixture_id": fixture_id,
        "fixture_sha256": fixture_hash,
        "selection_sha256": selection["selection_sha256"],
        "records_compared": len(fixture.get("records", [])),
        "live_parser_executed": False,
        "interpretation": "reference-only routing identity comparison",
    }
    write_noreplace(target / "reference-comparison.json", canonical_bytes(comparison))
    return {"state": "PASS", "selection": selection, "comparison": comparison}
