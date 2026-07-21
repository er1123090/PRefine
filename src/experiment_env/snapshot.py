from __future__ import annotations

import base64
import ctypes
import errno
import hashlib
import json
import os
import posixpath
import shutil
import stat
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

from .util import (
    canonical_json_bytes,
    clean_relative_path,
    load_json,
    resolve_config_path,
    sha256_bytes,
    sha256_file,
)


class PublicationError(RuntimeError):
    pass


class ValidationError(RuntimeError):
    pass


def _excluded(relative_path: str, rules: dict[str, Any]) -> bool:
    path = PurePosixPath(relative_path)
    prefixes = tuple(value.rstrip("/") for value in rules.get("prefixes", []))
    components = set(rules.get("components", []))
    suffixes = tuple(rules.get("suffixes", []))
    if any(relative_path == prefix or relative_path.startswith(prefix + "/") for prefix in prefixes):
        return True
    if components.intersection(path.parts):
        return True
    return bool(suffixes and relative_path.endswith(suffixes))


def _decode_source_path(record: dict[str, Any]) -> str:
    try:
        raw = base64.b64decode(record["relative_path_b64"], validate=True)
        value = raw.decode("utf-8")
    except (KeyError, ValueError, UnicodeDecodeError) as exc:
        raise PublicationError(f"invalid source-pre relative path: {record.get('record_id')}") from exc
    clean_relative_path(value)
    return value


def _load_source_records(
    source_pre_path: Path,
    *,
    root_id: str,
    rules: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    root_count = 0
    seen_paths: set[str] = set()
    seen_ids: set[str] = set()
    with source_pre_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise PublicationError(f"invalid source-pre JSON at line {line_number}") from exc
            if record.get("root_id") != root_id:
                continue
            root_count += 1
            if record.get("type") != "regular":
                raise PublicationError(f"unsupported source record type: {record.get('type')}")
            relative_path = _decode_source_path(record)
            record_id = record.get("record_id")
            if relative_path in seen_paths or record_id in seen_ids:
                raise PublicationError(f"duplicate source-pre identity: {relative_path}")
            seen_paths.add(relative_path)
            seen_ids.add(record_id)
            item = dict(record)
            item["relative_path"] = relative_path
            (excluded if _excluded(relative_path, rules) else selected).append(item)
    selected.sort(key=lambda item: item["relative_path"])
    excluded.sort(key=lambda item: item["relative_path"])
    return selected, excluded, root_count


def _iter_symlinks(source_root: Path, rules: dict[str, Any]) -> Iterator[tuple[str, str]]:
    def visit(directory: Path, relative_dir: PurePosixPath | None) -> Iterator[tuple[str, str]]:
        with os.scandir(directory) as entries:
            ordered = sorted(entries, key=lambda entry: entry.name)
        for entry in ordered:
            relative = PurePosixPath(entry.name) if relative_dir is None else relative_dir / entry.name
            relative_text = relative.as_posix()
            if entry.is_symlink():
                yield relative_text, os.readlink(entry.path)
            elif entry.is_dir(follow_symlinks=False) and not _excluded(relative_text, rules):
                yield from visit(Path(entry.path), relative)

    yield from visit(source_root, None)


def _symlink_target_path(link_path: str, target: str) -> str | None:
    if not target or posixpath.isabs(target):
        return None
    candidate = posixpath.normpath(posixpath.join(posixpath.dirname(link_path), target))
    if candidate in {"", ".", ".."} or candidate.startswith("../"):
        return None
    try:
        clean_relative_path(candidate)
    except ValueError:
        return None
    return candidate


def _classify_symlinks(
    source_root: Path,
    rules: dict[str, Any],
    included_files: set[str],
) -> tuple[list[tuple[str, str, str]], list[dict[str, str]]]:
    safe: list[tuple[str, str, str]] = []
    skipped: list[dict[str, str]] = []
    for link_path, target in _iter_symlinks(source_root, rules):
        resolved = _symlink_target_path(link_path, target)
        target_is_included = resolved is not None and (
            resolved in included_files
            or any(item.startswith(resolved.rstrip("/") + "/") for item in included_files)
        )
        if _excluded(link_path, rules):
            skipped.append({"path": link_path, "target": target, "reason": "link_path_excluded"})
        elif not target_is_included:
            skipped.append({"path": link_path, "target": target, "reason": "target_not_in_snapshot"})
        else:
            safe.append((link_path, target, resolved))
    safe.sort()
    skipped.sort(key=lambda item: item["path"])
    return safe, skipped


def _descriptor_matches(record: dict[str, Any], info: os.stat_result) -> bool:
    identity = record.get("descriptor_identity", {})
    expected = {
        "st_dev": info.st_dev,
        "st_ino": info.st_ino,
        "size": info.st_size,
        "mtime_ns": info.st_mtime_ns,
        "mode": stat.S_IMODE(info.st_mode),
        "file_type": "regular" if stat.S_ISREG(info.st_mode) else "other",
    }
    return all(identity.get(key) == value for key, value in expected.items())


def _copy_record(source_root: Path, stage: Path, record: dict[str, Any]) -> dict[str, Any]:
    relative_path = record["relative_path"]
    source_path = source_root.joinpath(*PurePosixPath(relative_path).parts)
    destination = stage.joinpath(*PurePosixPath(relative_path).parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source_path, flags)
    except OSError as exc:
        raise PublicationError(f"cannot open source record without following symlinks: {source_path}") from exc
    digest = hashlib.sha256()
    copied = 0
    try:
        before = os.fstat(source_fd)
        if not _descriptor_matches(record, before):
            raise PublicationError(f"source descriptor drift: {source_path}")
        with destination.open("xb") as output:
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
                digest.update(chunk)
                copied += len(chunk)
            output.flush()
            os.fsync(output.fileno())
        after = os.fstat(source_fd)
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        if before_identity != after_identity:
            raise PublicationError(f"source changed during copy: {source_path}")
    finally:
        os.close(source_fd)
    actual_hash = digest.hexdigest()
    if copied != record.get("size") or actual_hash != record.get("sha256"):
        raise PublicationError(f"source content drift: {source_path}")
    source_mode = stat.S_IMODE(before.st_mode)
    frozen_mode = source_mode & ~0o222
    os.chmod(destination, frozen_mode)
    return {
        "bytes": copied,
        "frozen_mode": frozen_mode,
        "origin_path": str(source_path),
        "path": relative_path,
        "record_type": "regular",
        "schema": "experiments7-environment-file/v1",
        "sha256": actual_hash,
        "source_descriptor_identity": record["descriptor_identity"],
        "source_record_id": record["record_id"],
        "source_root_id": record["root_id"],
    }


def _write_bytes(path: Path, data: bytes, mode: int = 0o444) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, mode)


def _freeze_directories(root: Path) -> None:
    directories: list[Path] = []
    for current, names, _ in os.walk(root, topdown=True, followlinks=False):
        base = Path(current)
        directories.append(base)
        names[:] = [name for name in names if not (base / name).is_symlink()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        os.chmod(directory, 0o555)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise PublicationError("renameat2(RENAME_NOREPLACE) is required for exact publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise FileExistsError(destination)
        raise OSError(error, os.strerror(error), destination)


def _check_expected(name: str, actual: int, expected: Any) -> None:
    if expected is not None and actual != expected:
        raise PublicationError(f"{name} mismatch: expected {expected}, got {actual}")


def publish_snapshot(repo_root: Path, spec_path: Path) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    spec_path = spec_path.resolve(strict=True)
    spec_bytes = spec_path.read_bytes()
    spec = json.loads(spec_bytes)
    if spec.get("schema") != "experiments7-environment-publication/v1":
        raise PublicationError("unsupported environment publication schema")
    source_root = resolve_config_path(repo_root, spec["source_root"]).resolve(strict=True)
    source_pre_path = resolve_config_path(repo_root, spec["source_pre_path"]).resolve(strict=True)
    cp0_path = resolve_config_path(repo_root, spec["cp0_path"]).resolve(strict=True)
    recipes_path = resolve_config_path(repo_root, spec["recipes_path"]).resolve(strict=True)
    destination = resolve_config_path(repo_root, spec["destination"])
    environments_root = (repo_root / "environments").resolve(strict=False)
    if destination.parent.resolve(strict=False) != environments_root:
        raise PublicationError("environment snapshots must be direct children of experiments7/environments")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)

    source_pre_hash, _ = sha256_file(source_pre_path)
    cp0_hash, _ = sha256_file(cp0_path)
    recipes_bytes = recipes_path.read_bytes()
    recipes = json.loads(recipes_bytes)
    if recipes.get("schema") != "experiments7-environment-recipes/v1":
        raise PublicationError("unsupported recipe schema")
    if source_pre_hash != spec["expected_source_pre_sha256"]:
        raise PublicationError("source-pre hash does not match publication specification")
    if cp0_hash != spec["expected_cp0_sha256"]:
        raise PublicationError("CP0 hash does not match publication specification")
    cp0 = load_json(cp0_path)
    if cp0.get("checkpoint") != 0 or cp0.get("semantic_bindings", {}).get("source_pre_record_count") != spec["expected_source_pre_record_count"]:
        raise PublicationError("CP0 semantic source-pre binding does not match the publication specification")

    selected, excluded_records, root_record_count = _load_source_records(
        source_pre_path,
        root_id=spec["source_root_id"],
        rules=spec["exclude"],
    )
    selected_bytes = sum(record["size"] for record in selected)
    _check_expected("source root record count", root_record_count, spec.get("expected_root_record_count"))
    _check_expected("selected regular file count", len(selected), spec.get("expected_regular_file_count"))
    _check_expected("selected regular byte count", selected_bytes, spec.get("expected_regular_bytes"))
    _check_expected("excluded source record count", len(excluded_records), spec.get("expected_excluded_record_count"))

    included_files = {record["relative_path"] for record in selected}
    safe_symlinks, skipped_symlinks = _classify_symlinks(source_root, spec["exclude"], included_files)
    _check_expected("safe symlink count", len(safe_symlinks), spec.get("expected_safe_symlink_count"))
    _check_expected("skipped symlink count", len(skipped_symlinks), spec.get("expected_skipped_symlink_count"))

    environments_root.mkdir(parents=True, exist_ok=True)
    stage = environments_root / f".{destination.name}.staging.{os.getpid()}.{uuid.uuid4().hex}"
    stage.mkdir(mode=0o700)
    source_root_before = source_root.stat()
    try:
        records = [_copy_record(source_root, stage, record) for record in selected]
        for link_path, target, resolved_target in safe_symlinks:
            link_destination = stage.joinpath(*PurePosixPath(link_path).parts)
            link_destination.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(target, link_destination)
            records.append(
                {
                    "bytes": len(os.fsencode(target)),
                    "frozen_mode": None,
                    "origin_path": str(source_root.joinpath(*PurePosixPath(link_path).parts)),
                    "path": link_path,
                    "record_type": "symlink",
                    "resolved_snapshot_target": resolved_target,
                    "schema": "experiments7-environment-file/v1",
                    "sha256": sha256_bytes(os.fsencode(target)),
                    "source_descriptor_identity": None,
                    "source_record_id": None,
                    "source_root_id": spec["source_root_id"],
                    "target": target,
                }
            )
        records.sort(key=lambda record: record["path"])
        manifest_bytes = b"".join(canonical_json_bytes(record) + b"\n" for record in records)
        control_root = stage / ".experiment-env"
        _write_bytes(control_root / "manifest.jsonl", manifest_bytes)
        _write_bytes(control_root / "recipes.json", recipes_bytes)
        seal = {
            "cp0_path": spec["cp0_path"],
            "cp0_sha256": cp0_hash,
            "excluded_source_record_count": len(excluded_records),
            "manifest_record_count": len(records),
            "manifest_sha256": sha256_bytes(manifest_bytes),
            "profile_id": spec["profile_id"],
            "recipes_sha256": sha256_bytes(recipes_bytes),
            "regular_bytes": selected_bytes,
            "regular_file_count": len(selected),
            "safe_symlink_count": len(safe_symlinks),
            "schema": "experiments7-environment-seal/v1",
            "skipped_symlinks": skipped_symlinks,
            "snapshot_id": destination.name,
            "source_pre_path": spec["source_pre_path"],
            "source_pre_record_count": spec["expected_source_pre_record_count"],
            "source_pre_sha256": source_pre_hash,
            "source_root": str(source_root),
            "source_root_id": spec["source_root_id"],
            "spec_sha256": sha256_bytes(spec_bytes),
        }
        seal_bytes = canonical_json_bytes(seal) + b"\n"
        _write_bytes(control_root / "seal.json", seal_bytes)
        source_root_after = source_root.stat()
        before_identity = (source_root_before.st_dev, source_root_before.st_ino, source_root_before.st_mtime_ns)
        after_identity = (source_root_after.st_dev, source_root_after.st_ino, source_root_after.st_mtime_ns)
        if before_identity != after_identity:
            raise PublicationError("source root identity changed during publication")
        _freeze_directories(stage)
        _rename_noreplace(stage, destination)
    except Exception:
        if stage.exists():
            for current, _, files in os.walk(stage, topdown=False, followlinks=False):
                os.chmod(current, 0o700)
                for name in files:
                    path = Path(current) / name
                    if not path.is_symlink():
                        os.chmod(path, 0o600)
            shutil.rmtree(stage)
        raise
    return {
        "destination": str(destination),
        "seal": seal,
        "seal_sha256": sha256_bytes(seal_bytes),
    }


def validate_snapshot(
    repo_root: Path,
    snapshot_path: Path,
    *,
    expected_seal_sha256: str | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve(strict=True)
    snapshot = snapshot_path.resolve(strict=True)
    environments_root = (repo_root / "environments").resolve(strict=True)
    if snapshot.parent != environments_root or snapshot_path.is_symlink():
        raise ValidationError("snapshot is not a direct, non-symlink child of experiments7/environments")
    control_root = snapshot / ".experiment-env"
    seal_path = control_root / "seal.json"
    manifest_path = control_root / "manifest.jsonl"
    recipes_path = control_root / "recipes.json"
    for path in (seal_path, manifest_path, recipes_path):
        if path.is_symlink() or not path.is_file():
            raise ValidationError(f"missing regular control file: {path}")
    seal_bytes = seal_path.read_bytes()
    seal_sha = sha256_bytes(seal_bytes)
    if expected_seal_sha256 and seal_sha != expected_seal_sha256:
        raise ValidationError("snapshot seal hash does not match catalog")
    seal = json.loads(seal_bytes)
    if seal.get("schema") != "experiments7-environment-seal/v1":
        raise ValidationError("unsupported snapshot seal schema")
    manifest_bytes = manifest_path.read_bytes()
    recipes_bytes = recipes_path.read_bytes()
    if sha256_bytes(manifest_bytes) != seal.get("manifest_sha256"):
        raise ValidationError("snapshot manifest hash mismatch")
    if sha256_bytes(recipes_bytes) != seal.get("recipes_sha256"):
        raise ValidationError("snapshot recipes hash mismatch")
    cp0_path = resolve_config_path(repo_root, seal["cp0_path"])
    source_pre_path = resolve_config_path(repo_root, seal["source_pre_path"])
    if sha256_file(cp0_path)[0] != seal.get("cp0_sha256"):
        raise ValidationError("bound CP0 hash mismatch")
    if sha256_file(source_pre_path)[0] != seal.get("source_pre_sha256"):
        raise ValidationError("bound source-pre hash mismatch")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    regular_count = 0
    regular_bytes = 0
    symlink_count = 0
    for line_number, line in enumerate(manifest_bytes.splitlines(), start=1):
        try:
            record = json.loads(line)
            relative = clean_relative_path(record["path"])
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise ValidationError(f"invalid manifest record at line {line_number}") from exc
        relative_text = relative.as_posix()
        if relative_text in seen or relative_text.startswith(".experiment-env/"):
            raise ValidationError(f"duplicate or reserved manifest path: {relative_text}")
        seen.add(relative_text)
        path = snapshot.joinpath(*relative.parts)
        if record.get("record_type") == "regular":
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or path.is_symlink():
                raise ValidationError(f"expected regular snapshot file: {relative_text}")
            actual_hash, actual_size = sha256_file(path)
            if actual_hash != record.get("sha256") or actual_size != record.get("bytes"):
                raise ValidationError(f"snapshot content mismatch: {relative_text}")
            if stat.S_IMODE(info.st_mode) != record.get("frozen_mode") or info.st_mode & 0o222:
                raise ValidationError(f"snapshot file is not frozen: {relative_text}")
            regular_count += 1
            regular_bytes += actual_size
        elif record.get("record_type") == "symlink":
            if not path.is_symlink() or os.readlink(path) != record.get("target"):
                raise ValidationError(f"snapshot symlink mismatch: {relative_text}")
            if sha256_bytes(os.fsencode(record["target"])) != record.get("sha256"):
                raise ValidationError(f"snapshot symlink hash mismatch: {relative_text}")
            symlink_count += 1
        else:
            raise ValidationError(f"unknown snapshot record type: {relative_text}")
        records.append(record)

    all_regular_paths = {record["path"] for record in records if record["record_type"] == "regular"}
    for record in records:
        if record["record_type"] != "symlink":
            continue
        target = _symlink_target_path(record["path"], record["target"])
        target_exists = target in all_regular_paths or any(path.startswith(target.rstrip("/") + "/") for path in all_regular_paths)
        if not target_exists or target != record.get("resolved_snapshot_target"):
            raise ValidationError(f"symlink target is outside the sealed payload: {record['path']}")

    actual_payload: set[str] = set()
    for current, names, files in os.walk(snapshot, topdown=True, followlinks=False):
        base = Path(current)
        relative_base = base.relative_to(snapshot)
        if relative_base == Path(".experiment-env"):
            names[:] = []
            continue
        if base.stat().st_mode & 0o222:
            raise ValidationError(f"snapshot directory is writable: {base}")
        for name in list(names):
            path = base / name
            relative = (relative_base / name).as_posix()
            if path.is_symlink():
                actual_payload.add(relative)
                names.remove(name)
        for name in files:
            actual_payload.add((relative_base / name).as_posix())
    if actual_payload != seen:
        raise ValidationError("snapshot payload differs from sealed manifest")
    for path in (control_root, seal_path, manifest_path, recipes_path):
        if path.lstat().st_mode & 0o222:
            raise ValidationError(f"snapshot control path is writable: {path}")
    if regular_count != seal.get("regular_file_count") or regular_bytes != seal.get("regular_bytes"):
        raise ValidationError("snapshot regular-file totals differ from seal")
    if symlink_count != seal.get("safe_symlink_count") or len(records) != seal.get("manifest_record_count"):
        raise ValidationError("snapshot record totals differ from seal")
    return {
        "manifest_record_count": len(records),
        "regular_bytes": regular_bytes,
        "regular_file_count": regular_count,
        "safe_symlink_count": symlink_count,
        "seal_sha256": seal_sha,
        "snapshot": str(snapshot),
        "status": "ok",
    }
