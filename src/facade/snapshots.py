"""Immutable snapshot-publication overlay for the pending G2 snapshot plan.

The generated registry, adapters, lineage, and snapshot plan intentionally remain
pending.  A publication is an additive, sealed overlay that binds every planned
destination to the exact bytes present in experiments7.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path, PurePosixPath
import stat
from typing import Any

from .registry import Bundle, FacadeError, HEX64, SAFE_ID, load_json, require
from .selection import canonical_bytes, sha256_bytes, sha256_file


PUBLICATION_SCHEMA = "experiments7-snapshot-publication/v1"
PUBLICATION_SEAL_SCHEMA = "experiments7-snapshot-publication-seal/v1"


def _safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
        and not any(character in value for character in "*?[")
    )


def _safe_directory(path: Path, *, root: Path, code: str) -> None:
    root = root.resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise FacadeError(code, str(path)) from exc
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError as exc:
            raise FacadeError(code, str(cursor)) from exc
        require(stat.S_ISDIR(mode) and not cursor.is_symlink(), code, str(cursor))


def _safe_snapshot(path: Path, *, root: Path) -> tuple[str, int]:
    root = root.resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise FacadeError("SNAPSHOT_PATH_ESCAPE", str(path)) from exc
    cursor = root
    for part in relative.parts[:-1]:
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError as exc:
            raise FacadeError("MISSING_PUBLISHED_SNAPSHOT", str(cursor)) from exc
        require(
            stat.S_ISDIR(mode) and not cursor.is_symlink(),
            "UNSAFE_SNAPSHOT_ANCESTOR",
            str(cursor),
        )
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise FacadeError("MISSING_PUBLISHED_SNAPSHOT", str(path)) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(), "UNSAFE_PUBLISHED_SNAPSHOT", str(path))
    require(mode & 0o222 == 0, "SNAPSHOT_NOT_IMMUTABLE", str(path))
    return sha256_file(path)


def _readonly_control(path: Path, *, code: str) -> tuple[str, int]:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise FacadeError(code, str(path)) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(), code, str(path))
    require(mode & 0o222 == 0, "PUBLICATION_CONTROL_NOT_IMMUTABLE", str(path))
    return sha256_file(path)


def _control_hashes(bundle: Bundle) -> dict[str, str]:
    return {
        "registry_sha256": bundle.hashes["registry"],
        "profiles_sha256": bundle.hashes["profiles"],
        "compatibility_sha256": bundle.hashes["compatibility"],
        "adapters_sha256": bundle.hashes["adapters"],
        "code_lineage_sha256": bundle.hashes["code_lineage"],
        "config_lineage_sha256": bundle.hashes["config_lineage"],
    }


def _plan(bundle: Bundle) -> tuple[dict[str, Any], str]:
    path = bundle.root / "configs/snapshot-plan.json"
    value = load_json(path)
    require(value.get("no_globs") is True, "SNAPSHOT_PLAN_GLOBS_ENABLED")
    require(value.get("protected_roots_writable") is False, "SNAPSHOT_PLAN_WRITES_ENABLED")
    records = value.get("records")
    require(isinstance(records, list) and len(records) == len(bundle.lineage_by_id),
            "SNAPSHOT_PLAN_RECORD_COUNT")
    require(
        {row.get("lineage_id") for row in records} == set(bundle.lineage_by_id),
        "SNAPSHOT_PLAN_LINEAGE_SET",
    )
    require(
        all(
            row.get("snapshot_state") == "pending_protected_snapshot"
            and row.get("final_sha256") is None
            for row in records
        ),
        "SNAPSHOT_PLAN_NOT_PENDING",
    )
    return value, sha256_file(path)[0]


@dataclass(frozen=True)
class SnapshotPublication:
    publication_id: str
    manifest_sha256: str
    snapshot_plan_sha256: str
    records_sha256: str
    records_by_lineage: dict[str, dict[str, Any]]

    def summary(self) -> dict[str, Any]:
        return {
            "publication_id": self.publication_id,
            "manifest_sha256": self.manifest_sha256,
            "snapshot_plan_sha256": self.snapshot_plan_sha256,
            "records_sha256": self.records_sha256,
            "record_count": len(self.records_by_lineage),
            "state": "VERIFIED",
        }


def validate_snapshot_publication(bundle: Bundle, publication_id: str) -> SnapshotPublication:
    require(SAFE_ID.fullmatch(publication_id or "") is not None,
            "INVALID_PUBLICATION_ID", publication_id)
    directory = bundle.root / "snapshots/publications" / publication_id
    _safe_directory(directory, root=bundle.root, code="MISSING_SNAPSHOT_PUBLICATION")
    require(directory.lstat().st_mode & 0o222 == 0,
            "PUBLICATION_DIRECTORY_NOT_IMMUTABLE", str(directory))
    manifest_path = directory / "manifest.json"
    seal_path = directory / "seal.json"
    manifest_hash, manifest_bytes = _readonly_control(
        manifest_path, code="MISSING_PUBLICATION_MANIFEST"
    )
    _readonly_control(seal_path, code="MISSING_PUBLICATION_SEAL")
    manifest = load_json(manifest_path)
    seal = load_json(seal_path)

    require(set(manifest) == {
        "schema", "publication_id", "snapshot_plan_sha256", "control_hashes",
        "record_count", "records_sha256", "records",
    }, "PUBLICATION_MANIFEST_FIELDS")
    require(set(seal) == {
        "schema", "publication_id", "manifest_sha256", "manifest_bytes",
    }, "PUBLICATION_SEAL_FIELDS")
    require(manifest.get("schema") == PUBLICATION_SCHEMA, "PUBLICATION_SCHEMA")
    require(seal.get("schema") == PUBLICATION_SEAL_SCHEMA, "PUBLICATION_SEAL_SCHEMA")
    require(
        manifest.get("publication_id") == publication_id
        and seal.get("publication_id") == publication_id,
        "PUBLICATION_ID_MISMATCH",
    )
    require(
        seal.get("manifest_sha256") == manifest_hash
        and seal.get("manifest_bytes") == manifest_bytes,
        "PUBLICATION_SEAL_MISMATCH",
    )

    plan, plan_hash = _plan(bundle)
    require(manifest.get("snapshot_plan_sha256") == plan_hash, "STALE_PUBLICATION_PLAN")
    require(manifest.get("control_hashes") == _control_hashes(bundle),
            "STALE_PUBLICATION_CONTROLS")
    records = manifest.get("records")
    require(isinstance(records, list) and manifest.get("record_count") == len(records),
            "PUBLICATION_RECORD_COUNT")
    require(len(records) == len(plan["records"]), "PUBLICATION_NOT_EXHAUSTIVE")
    records_hash = sha256_bytes(canonical_bytes(records))
    require(manifest.get("records_sha256") == records_hash, "PUBLICATION_RECORDS_SEAL_MISMATCH")
    require(records == sorted(records, key=lambda item: item.get("lineage_id", "")),
            "PUBLICATION_RECORD_ORDER")

    plan_by_lineage = {row["lineage_id"]: row for row in plan["records"]}
    records_by_lineage: dict[str, dict[str, Any]] = {}
    seen_paths: set[str] = set()
    for record in records:
        require(isinstance(record, dict), "INVALID_PUBLICATION_RECORD")
        require(set(record) == {
            "action", "bytes", "final_sha256", "lineage_id", "path", "source_sha256",
        }, "PUBLICATION_RECORD_FIELDS")
        lineage_id = record.get("lineage_id")
        require(lineage_id in plan_by_lineage, "UNKNOWN_PUBLICATION_LINEAGE", lineage_id)
        require(lineage_id not in records_by_lineage, "DUPLICATE_PUBLICATION_LINEAGE", lineage_id)
        planned = plan_by_lineage[lineage_id]
        path = record.get("path")
        require(_safe_relative(path), "INVALID_PUBLICATION_PATH", path)
        require(path == planned["planned_destination"], "PUBLICATION_PATH_DRIFT", lineage_id)
        require(path not in seen_paths, "DUPLICATE_PUBLICATION_PATH", path)
        require(record.get("action") == planned["action"], "PUBLICATION_ACTION_DRIFT", lineage_id)
        require(record.get("source_sha256") == planned["origin_sha256"],
                "PUBLICATION_SOURCE_HASH_DRIFT", lineage_id)
        require(HEX64.fullmatch(str(record.get("final_sha256"))) is not None,
                "INVALID_PUBLICATION_HASH", lineage_id)
        require(isinstance(record.get("bytes"), int) and record["bytes"] >= 0,
                "INVALID_PUBLICATION_SIZE", lineage_id)
        actual_hash, actual_bytes = _safe_snapshot(bundle.root / path, root=bundle.root)
        require(
            (record["final_sha256"], record["bytes"]) == (actual_hash, actual_bytes),
            "PUBLISHED_SNAPSHOT_HASH_MISMATCH",
            lineage_id,
        )
        records_by_lineage[lineage_id] = record
        seen_paths.add(path)

    require(set(records_by_lineage) == set(plan_by_lineage), "PUBLICATION_LINEAGE_SET")
    return SnapshotPublication(
        publication_id=publication_id,
        manifest_sha256=manifest_hash,
        snapshot_plan_sha256=plan_hash,
        records_sha256=records_hash,
        records_by_lineage=records_by_lineage,
    )


def _write_noreplace(path: Path, payload: bytes, *, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, mode)
    except FileExistsError as exc:
        raise FacadeError("PUBLICATION_COLLISION", str(path)) from exc
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, "PUBLICATION_SHORT_WRITE", str(path))
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.chmod(path, mode)


def publish_snapshot_overlay(bundle: Bundle, publication_id: str) -> SnapshotPublication:
    """Seal already-present snapshots without reading any origin experiment tree."""

    require(SAFE_ID.fullmatch(publication_id or "") is not None,
            "INVALID_PUBLICATION_ID", publication_id)
    plan, plan_hash = _plan(bundle)
    records: list[dict[str, Any]] = []
    for planned in plan["records"]:
        path = planned["planned_destination"]
        require(_safe_relative(path), "INVALID_PUBLICATION_PATH", path)
        final_hash, size = _safe_snapshot(bundle.root / path, root=bundle.root)
        records.append({
            "action": planned["action"],
            "bytes": size,
            "final_sha256": final_hash,
            "lineage_id": planned["lineage_id"],
            "path": path,
            "source_sha256": planned["origin_sha256"],
        })
    records.sort(key=lambda item: item["lineage_id"])
    manifest = {
        "schema": PUBLICATION_SCHEMA,
        "publication_id": publication_id,
        "snapshot_plan_sha256": plan_hash,
        "control_hashes": _control_hashes(bundle),
        "record_count": len(records),
        "records_sha256": sha256_bytes(canonical_bytes(records)),
        "records": records,
    }
    manifest_payload = canonical_bytes(manifest)
    seal = {
        "schema": PUBLICATION_SEAL_SCHEMA,
        "publication_id": publication_id,
        "manifest_sha256": sha256_bytes(manifest_payload),
        "manifest_bytes": len(manifest_payload),
    }

    snapshots = bundle.root / "snapshots"
    if os.path.lexists(snapshots):
        require(snapshots.is_dir() and not snapshots.is_symlink(),
                "UNSAFE_PUBLICATION_ROOT", str(snapshots))
    else:
        snapshots.mkdir(mode=0o755)
    parent = snapshots / "publications"
    if os.path.lexists(parent):
        require(parent.is_dir() and not parent.is_symlink(),
                "UNSAFE_PUBLICATION_ROOT", str(parent))
    else:
        parent.mkdir(mode=0o755)
    directory = parent / publication_id
    try:
        directory.mkdir(mode=0o755)
    except FileExistsError as exc:
        raise FacadeError("PUBLICATION_COLLISION", str(directory)) from exc
    try:
        _write_noreplace(directory / "manifest.json", manifest_payload)
        _write_noreplace(directory / "seal.json", canonical_bytes(seal))
        os.chmod(directory, 0o555)
    except Exception:
        # Keep any partial no-replace publication visibly incomplete and unusable.
        raise
    return validate_snapshot_publication(bundle, publication_id)
