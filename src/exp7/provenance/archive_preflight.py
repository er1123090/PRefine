"""Read-only admission of off-host backup and restore evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence
from urllib.parse import urlsplit

from .admission import AdmittedFile, lexical_absolute


PROOF_SCHEMA = "experiments7-archive-backup-proof/v1"
OBJECT_SCHEMA = "experiments7-archive-object/v1"
RETENTION_SCHEMA = "experiments7-archive-retention-evidence/v1"
TRANSCRIPT_SCHEMA = "experiments7-archive-restore-transcript/v1"
REPORT_SCHEMA = "experiments7-archive-preflight-report/v3"
CANONICAL_RAW_SCHEMA = "experiments7-final-paper-raw-manifest/v1"
_PAPER_OUTPUTS = "paper_" + "outputs"
_RAW_PREFIX = (_PAPER_OUTPUTS, "raw", "verified")
CANONICAL_RAW_MANIFEST = PurePosixPath(
    _PAPER_OUTPUTS, "final", "raw-manifest.jsonl"
).as_posix()
CANONICAL_RAW_COUNT = 521

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{7,255}\Z")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._~+=:/-]{7,255}\Z")
_OFF_HOST_SCHEMES = frozenset({"az", "gs", "r2", "s3"})
_IMMUTABILITY_MODES = frozenset(
    {"bucket_lock", "immutable_blob", "object_lock_compliance"}
)
_FAKE_VERSION_VALUES = frozenset(
    {"fake", "latest", "none", "null", "placeholder", "test", "unversioned"}
)


class ArchivePreflightError(RuntimeError):
    """Raised when externalization evidence is incomplete or untrusted."""


@dataclass(frozen=True)
class _OpenedFile:
    path: Path
    descriptor: int
    file_fingerprint: tuple[int, int, int, int, int, int]
    parents: tuple[
        tuple[Path, int, tuple[int, int, int, int, int, int]], ...
    ]


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _open_file_nofollow(path: Path, *, label: str) -> _OpenedFile:
    absolute = lexical_absolute(path)
    parts = absolute.parts
    if len(parts) < 2:
        raise ArchivePreflightError(f"{label} must identify a regular file: {absolute}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    parents: list[tuple[Path, int, tuple[int, int, int, int, int, int]]] = []
    descriptor: int | None = None
    try:
        current_path = Path(absolute.anchor)
        current_fd = os.open(absolute.anchor, directory_flags)
        current_value = os.fstat(current_fd)
        parents.append((current_path, current_fd, _fingerprint(current_value)))
        for part in parts[1:-1]:
            current_path = current_path / part
            try:
                before = os.lstat(current_path)
            except OSError as exc:
                raise ArchivePreflightError(
                    f"cannot inspect {label} parent {current_path}: {exc}"
                ) from exc
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise ArchivePreflightError(
                    f"{label} parent must be a non-symlink directory: {current_path}"
                )
            try:
                current_fd = os.open(part, directory_flags, dir_fd=current_fd)
            except OSError as exc:
                raise ArchivePreflightError(
                    f"cannot safely open {label} parent {current_path}: {exc}"
                ) from exc
            opened = os.fstat(current_fd)
            if _fingerprint(opened) != _fingerprint(before):
                raise ArchivePreflightError(
                    f"{label} parent changed while being opened: {current_path}"
                )
            parents.append((current_path, current_fd, _fingerprint(opened)))

        try:
            before_file = os.lstat(absolute)
        except OSError as exc:
            raise ArchivePreflightError(
                f"cannot inspect {label} {absolute}: {exc}"
            ) from exc
        if stat.S_ISLNK(before_file.st_mode) or not stat.S_ISREG(before_file.st_mode):
            raise ArchivePreflightError(
                f"{label} must be a non-symlink regular file: {absolute}"
            )
        try:
            descriptor = os.open(parts[-1], file_flags, dir_fd=parents[-1][1])
        except OSError as exc:
            raise ArchivePreflightError(
                f"cannot safely open {label} {absolute}: {exc}"
            ) from exc
        opened_file = os.fstat(descriptor)
        if _fingerprint(opened_file) != _fingerprint(before_file):
            raise ArchivePreflightError(
                f"{label} changed while being opened: {absolute}"
            )
        return _OpenedFile(
            path=absolute,
            descriptor=descriptor,
            file_fingerprint=_fingerprint(opened_file),
            parents=tuple(parents),
        )
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        for _, parent_fd, _ in reversed(parents):
            os.close(parent_fd)
        raise


def _close_and_verify(opened: _OpenedFile, *, label: str) -> None:
    error: ArchivePreflightError | None = None
    try:
        after = os.fstat(opened.descriptor)
        if _fingerprint(after) != opened.file_fingerprint:
            error = ArchivePreflightError(
                f"{label} changed while being read: {opened.path}"
            )
        try:
            current = os.lstat(opened.path)
        except OSError as exc:
            error = ArchivePreflightError(
                f"{label} pathname changed while being read: {opened.path}: {exc}"
            )
        else:
            if _fingerprint(current) != opened.file_fingerprint:
                error = ArchivePreflightError(
                    f"{label} pathname changed while being read: {opened.path}"
                )
        for parent_path, _, expected in opened.parents:
            try:
                current_parent = os.lstat(parent_path)
            except OSError as exc:
                error = ArchivePreflightError(
                    f"{label} parent changed while being read: {parent_path}: {exc}"
                )
                break
            if _fingerprint(current_parent) != expected:
                error = ArchivePreflightError(
                    f"{label} parent changed while being read: {parent_path}"
                )
                break
    finally:
        os.close(opened.descriptor)
        for _, parent_fd, _ in reversed(opened.parents):
            os.close(parent_fd)
    if error is not None:
        raise error


def _admit_file(path: Path, *, label: str) -> AdmittedFile:
    opened = _open_file_nofollow(path, label=label)
    chunks: list[bytes] = []
    read_error: BaseException | None = None
    try:
        while True:
            chunk = os.read(opened.descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    except BaseException as exc:
        read_error = exc
    try:
        _close_and_verify(opened, label=label)
    except BaseException:
        if read_error is None:
            raise
    if read_error is not None:
        raise ArchivePreflightError(
            f"cannot read {label} {opened.path}: {read_error}"
        ) from read_error
    return AdmittedFile(opened.path, b"".join(chunks))


def _hash_file(path: Path, *, label: str) -> tuple[int, str]:
    opened = _open_file_nofollow(path, label=label)
    digest = hashlib.sha256()
    size = 0
    read_error: BaseException | None = None
    try:
        while True:
            chunk = os.read(opened.descriptor, 1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    except BaseException as exc:
        read_error = exc
    try:
        _close_and_verify(opened, label=label)
    except BaseException:
        if read_error is None:
            raise
    if read_error is not None:
        raise ArchivePreflightError(
            f"cannot hash {label} {opened.path}: {read_error}"
        ) from read_error
    return size, digest.hexdigest()


def _json_pairs(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ArchivePreflightError(f"duplicate JSON key: {key!r}")
        value[key] = item
    return value


def _decode_json(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=_json_pairs)
    except ArchivePreflightError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchivePreflightError(f"invalid UTF-8 JSON in {label}: {exc}") from exc


def _decode_jsonl(payload: bytes, *, label: str) -> list[dict[str, Any]]:
    lines = payload.splitlines()
    if not lines:
        raise ArchivePreflightError(f"{label} must not be empty")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ArchivePreflightError(f"blank line in {label} at {line_number}")
        value = _decode_json(line, label=f"{label} line {line_number}")
        if not isinstance(value, dict):
            raise ArchivePreflightError(
                f"{label} line {line_number} must be a JSON object"
            )
        rows.append(value)
    return rows


def _object(value: Any, *, label: str, keys: Sequence[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ArchivePreflightError(f"{label} must be a JSON object")
    expected, actual = set(keys), set(value)
    if actual != expected:
        raise ArchivePreflightError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )
    return value


def _string(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ArchivePreflightError(f"{label} must be a non-empty string")
    return value


def _identifier(value: Any, *, label: str) -> str:
    text = _string(value, label=label)
    if _IDENTIFIER.fullmatch(text) is None:
        raise ArchivePreflightError(f"{label} is not a durable identifier")
    return text


def _sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise ArchivePreflightError(f"{label} must be a lowercase SHA-256")
    return value


def _count(value: Any, *, label: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ArchivePreflightError(f"{label} must be an integer >= {minimum}")
    return value


def _relative(value: Any, *, label: str) -> str:
    text = _string(value, label=label)
    path = PurePosixPath(text)
    if (
        "\\" in text
        or path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ArchivePreflightError(f"{label} must be a safe relative path")
    return path.as_posix()


def _timestamp(value: Any, *, label: str) -> datetime:
    text = _string(value, label=label)
    if not text.endswith("Z"):
        raise ArchivePreflightError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ArchivePreflightError(
            f"{label} must be an RFC3339 UTC timestamp"
        ) from exc
    if parsed.tzinfo is None:
        raise ArchivePreflightError(f"{label} must include UTC timezone")
    return parsed.astimezone(timezone.utc)


def _off_host_uri(value: Any, *, label: str) -> str:
    text = _string(value, label=label)
    parsed = urlsplit(text)
    if (
        parsed.scheme not in _OFF_HOST_SCHEMES
        or not parsed.netloc
        or not parsed.path.strip("/")
        or parsed.query
        or parsed.fragment
    ):
        raise ArchivePreflightError(
            f"{label} must be a versioned off-host object URI"
        )
    return text


def _version(value: Any, *, label: str) -> str:
    text = _string(value, label=label)
    lowered = text.lower()
    if (
        _VERSION.fullmatch(text) is None
        or lowered in _FAKE_VERSION_VALUES
        or lowered.startswith(("fake", "placeholder", "test-"))
    ):
        raise ArchivePreflightError(
            f"{label} must be a real immutable object version identifier"
        )
    return text


def _canonical_records(
    admitted: AdmittedFile,
    *,
    expected_count: int | None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    required = (
        "destination_relative_path",
        "fresh_source_record_id",
        "paper_result_ids",
        "raw_node_id",
        "root_id",
        "schema",
        "sha256",
        "size",
        "source_relative_path",
    )
    rows = _decode_jsonl(admitted.payload, label="canonical raw manifest")
    if expected_count is not None and len(rows) != expected_count:
        raise ArchivePreflightError(
            f"canonical raw manifest count must be {expected_count}, got {len(rows)}"
        )
    indexed: dict[str, dict[str, Any]] = {}
    raw_ids: set[str] = set()
    for index, raw in enumerate(rows):
        row = _object(raw, label=f"canonical row {index}", keys=required)
        if row["schema"] != CANONICAL_RAW_SCHEMA:
            raise ArchivePreflightError(f"canonical row {index} schema changed")
        path = _relative(
            row["destination_relative_path"],
            label=f"canonical row {index} destination_relative_path",
        )
        if PurePosixPath(path).parts[: len(_RAW_PREFIX)] != _RAW_PREFIX:
            raise ArchivePreflightError(
                f"canonical row {index} is outside the verified raw-output root"
            )
        if path in indexed:
            raise ArchivePreflightError(f"duplicate canonical path: {path}")
        source = _relative(
            row["source_relative_path"],
            label=f"canonical row {index} source_relative_path",
        )
        digest = _sha(row["sha256"], label=f"canonical row {index} sha256")
        size = _count(row["size"], label=f"canonical row {index} size", allow_zero=True)
        raw_id = _identifier(row["raw_node_id"], label=f"canonical row {index} raw_node_id")
        if raw_id in raw_ids:
            raise ArchivePreflightError(f"duplicate canonical raw_node_id: {raw_id}")
        raw_ids.add(raw_id)
        if not isinstance(row["paper_result_ids"], list) or not all(
            isinstance(item, str) and item for item in row["paper_result_ids"]
        ):
            raise ArchivePreflightError(
                f"canonical row {index} paper_result_ids must contain strings"
            )
        indexed[path] = {
            "path": path,
            "sha256": digest,
            "size": size,
            "source_relative_path": source,
        }
    return rows, indexed


def _retention(
    admitted: AdmittedFile,
    *,
    expected_count: int,
    now: datetime,
) -> dict[str, Any]:
    value = _object(
        _decode_json(admitted.payload, label="retention evidence"),
        label="retention evidence",
        keys=(
            "captured_at",
            "evidence_id",
            "immutability_enabled",
            "mode",
            "object_count",
            "provider_evidence_id",
            "retention_until",
            "schema",
            "storage_location",
            "versioning_enabled",
        ),
    )
    if value["schema"] != RETENTION_SCHEMA:
        raise ArchivePreflightError("unsupported retention evidence schema")
    evidence_id = _identifier(value["evidence_id"], label="retention evidence_id")
    provider_evidence_id = _identifier(
        value["provider_evidence_id"], label="retention provider_evidence_id"
    )
    if provider_evidence_id.lower().startswith(("fake", "placeholder", "test-")):
        raise ArchivePreflightError("retention provider_evidence_id is a placeholder")
    mode = _string(value["mode"], label="retention mode")
    if mode not in _IMMUTABILITY_MODES:
        raise ArchivePreflightError("retention mode is not an immutable storage mode")
    if value["versioning_enabled"] is not True:
        raise ArchivePreflightError("retention evidence requires versioning_enabled=true")
    if value["immutability_enabled"] is not True:
        raise ArchivePreflightError("retention evidence requires immutability_enabled=true")
    captured_at = _timestamp(value["captured_at"], label="retention captured_at")
    retention_until = _timestamp(
        value["retention_until"], label="retention retention_until"
    )
    if captured_at > now:
        raise ArchivePreflightError("retention evidence was captured in the future")
    if retention_until <= now or retention_until <= captured_at:
        raise ArchivePreflightError("retention evidence is expired or non-forward")
    if _count(value["object_count"], label="retention object_count") != expected_count:
        raise ArchivePreflightError("retention object_count differs from canonical count")
    return {
        "evidence_id": evidence_id,
        "mode": mode,
        "storage_location": _off_host_uri(
            value["storage_location"], label="retention storage_location"
        ),
    }


def _objects(
    admitted: AdmittedFile,
    *,
    canonical: Mapping[str, Mapping[str, Any]],
    retention: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    required = (
        "object_uri",
        "path",
        "retention_evidence_id",
        "schema",
        "sha256",
        "size",
        "version_id",
    )
    rows = _decode_jsonl(admitted.payload, label="object manifest")
    indexed: dict[str, dict[str, Any]] = {}
    object_versions: set[tuple[str, str]] = set()
    location = str(retention["storage_location"]).rstrip("/") + "/"
    for index, raw in enumerate(rows):
        row = _object(raw, label=f"object row {index}", keys=required)
        if row["schema"] != OBJECT_SCHEMA:
            raise ArchivePreflightError(f"object row {index} schema changed")
        path = _relative(row["path"], label=f"object row {index} path")
        if path in indexed:
            raise ArchivePreflightError(f"duplicate object path: {path}")
        object_uri = _off_host_uri(
            row["object_uri"], label=f"object row {index} object_uri"
        )
        if not object_uri.startswith(location):
            raise ArchivePreflightError(
                f"object row {index} is outside retention storage_location"
            )
        version_id = _version(
            row["version_id"], label=f"object row {index} version_id"
        )
        identity = (object_uri, version_id)
        if identity in object_versions:
            raise ArchivePreflightError(
                f"duplicate immutable object version: {object_uri}#{version_id}"
            )
        object_versions.add(identity)
        if row["retention_evidence_id"] != retention["evidence_id"]:
            raise ArchivePreflightError(
                f"object row {index} retention evidence binding changed"
            )
        indexed[path] = {
            "object_uri": object_uri,
            "path": path,
            "sha256": _sha(row["sha256"], label=f"object row {index} sha256"),
            "size": _count(
                row["size"], label=f"object row {index} size", allow_zero=True
            ),
            "version_id": version_id,
        }
    if set(indexed) != set(canonical):
        missing = sorted(set(canonical) - set(indexed))
        extra = sorted(set(indexed) - set(canonical))
        raise ArchivePreflightError(
            f"object manifest path set differs: missing={missing[:3]}, extra={extra[:3]}"
        )
    for path, expected in canonical.items():
        actual = indexed[path]
        if actual["size"] != expected["size"] or actual["sha256"] != expected["sha256"]:
            raise ArchivePreflightError(
                f"object manifest size/hash differs for {path}"
            )
    return indexed


def _transcript(
    admitted: AdmittedFile,
    *,
    canonical_sha256: str,
    object_manifest_sha256: str,
    retention_evidence_sha256: str,
    objects: Mapping[str, Mapping[str, Any]],
    total_bytes: int,
    now: datetime,
) -> dict[str, Any]:
    value = _object(
        _decode_json(admitted.payload, label="restore transcript"),
        label="restore transcript",
        keys=(
            "canonical_manifest_sha256",
            "files",
            "object_count",
            "object_manifest_sha256",
            "restore_id",
            "restored_at",
            "retention_evidence_sha256",
            "schema",
            "total_bytes",
        ),
    )
    if value["schema"] != TRANSCRIPT_SCHEMA:
        raise ArchivePreflightError("unsupported restore transcript schema")
    if value["canonical_manifest_sha256"] != canonical_sha256:
        raise ArchivePreflightError("restore transcript canonical manifest binding changed")
    if value["object_manifest_sha256"] != object_manifest_sha256:
        raise ArchivePreflightError("restore transcript object manifest binding changed")
    if value["retention_evidence_sha256"] != retention_evidence_sha256:
        raise ArchivePreflightError("restore transcript retention binding changed")
    _identifier(value["restore_id"], label="restore restore_id")
    if _timestamp(value["restored_at"], label="restore restored_at") > now:
        raise ArchivePreflightError("restore transcript was created in the future")
    if _count(value["object_count"], label="restore object_count") != len(objects):
        raise ArchivePreflightError("restore object_count differs from object manifest")
    if _count(value["total_bytes"], label="restore total_bytes", allow_zero=True) != total_bytes:
        raise ArchivePreflightError("restore total_bytes differs from canonical manifest")
    files = value["files"]
    if not isinstance(files, list):
        raise ArchivePreflightError("restore files must be a JSON array")
    indexed: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(files):
        row = _object(
            raw,
            label=f"restore file {index}",
            keys=("path", "sha256", "size", "status", "version_id"),
        )
        path = _relative(row["path"], label=f"restore file {index} path")
        if path in indexed:
            raise ArchivePreflightError(f"duplicate restore transcript path: {path}")
        if row["status"] != "restored_verified":
            raise ArchivePreflightError(
                f"restore file {index} status must be restored_verified"
            )
        indexed[path] = {
            "sha256": _sha(row["sha256"], label=f"restore file {index} sha256"),
            "size": _count(
                row["size"], label=f"restore file {index} size", allow_zero=True
            ),
            "version_id": _version(
                row["version_id"], label=f"restore file {index} version_id"
            ),
        }
    if set(indexed) != set(objects):
        missing = sorted(set(objects) - set(indexed))
        extra = sorted(set(indexed) - set(objects))
        raise ArchivePreflightError(
            f"restore transcript path set differs: missing={missing[:3]}, extra={extra[:3]}"
        )
    for path, expected in objects.items():
        actual = indexed[path]
        if any(actual[key] != expected[key] for key in ("sha256", "size", "version_id")):
            raise ArchivePreflightError(f"restore transcript differs for {path}")
    return dict(value)


def _proof(
    admitted: AdmittedFile,
    *,
    canonical_sha256: str,
    object_manifest_sha256: str,
    retention_evidence_sha256: str,
    restore_transcript_sha256: str,
    object_count: int,
    total_bytes: int,
    now: datetime,
) -> dict[str, Any]:
    value = _object(
        _decode_json(admitted.payload, label="backup proof"),
        label="backup proof",
        keys=(
            "backup_id",
            "canonical_manifest_sha256",
            "created_at",
            "independent_restore",
            "object_count",
            "object_manifest_sha256",
            "off_host",
            "restore_transcript_sha256",
            "retention_evidence_sha256",
            "schema",
            "total_bytes",
        ),
    )
    if value["schema"] != PROOF_SCHEMA:
        raise ArchivePreflightError("unsupported backup proof schema")
    _identifier(value["backup_id"], label="backup backup_id")
    if _timestamp(value["created_at"], label="backup created_at") > now:
        raise ArchivePreflightError("backup proof was created in the future")
    if value["off_host"] is not True:
        raise ArchivePreflightError("backup proof requires off_host=true")
    if value["independent_restore"] is not True:
        raise ArchivePreflightError("backup proof requires independent_restore=true")
    expected_hashes = {
        "canonical_manifest_sha256": canonical_sha256,
        "object_manifest_sha256": object_manifest_sha256,
        "retention_evidence_sha256": retention_evidence_sha256,
        "restore_transcript_sha256": restore_transcript_sha256,
    }
    for field, expected in expected_hashes.items():
        if _sha(value[field], label=f"backup {field}") != expected:
            raise ArchivePreflightError(f"backup proof {field} binding changed")
    if _count(value["object_count"], label="backup object_count") != object_count:
        raise ArchivePreflightError("backup object_count differs from canonical manifest")
    if _count(value["total_bytes"], label="backup total_bytes", allow_zero=True) != total_bytes:
        raise ArchivePreflightError("backup total_bytes differs from canonical manifest")
    return dict(value)


def _directory_snapshot(root: Path) -> dict[Path, tuple[int, int, int, int, int, int]]:
    absolute = lexical_absolute(root)
    try:
        root_value = os.lstat(absolute)
    except OSError as exc:
        raise ArchivePreflightError(f"cannot inspect restore root {absolute}: {exc}") from exc
    if stat.S_ISLNK(root_value.st_mode) or not stat.S_ISDIR(root_value.st_mode):
        raise ArchivePreflightError(
            f"restore root must be a non-symlink directory: {absolute}"
        )
    snapshots: dict[Path, tuple[int, int, int, int, int, int]] = {}
    for directory, directories, files in os.walk(absolute, followlinks=False):
        current = Path(directory)
        current_value = os.lstat(current)
        if stat.S_ISLNK(current_value.st_mode) or not stat.S_ISDIR(current_value.st_mode):
            raise ArchivePreflightError(
                f"restore tree contains an unsafe directory: {current}"
            )
        snapshots[current] = _fingerprint(current_value)
        for name in directories:
            child = current / name
            child_value = os.lstat(child)
            if stat.S_ISLNK(child_value.st_mode) or not stat.S_ISDIR(child_value.st_mode):
                raise ArchivePreflightError(
                    f"restore tree contains a symlink or unsafe directory: {child}"
                )
        for name in files:
            child = current / name
            child_value = os.lstat(child)
            if stat.S_ISLNK(child_value.st_mode) or not stat.S_ISREG(child_value.st_mode):
                raise ArchivePreflightError(
                    f"restore tree contains a symlink or unsafe file: {child}"
                )
    return snapshots


def _restore_paths(root: Path) -> set[str]:
    absolute = lexical_absolute(root)
    paths: set[str] = set()
    for directory, _, files in os.walk(absolute, followlinks=False):
        current = Path(directory)
        for name in files:
            path = current / name
            paths.add(path.relative_to(absolute).as_posix())
    return paths


def _verify_directories(
    snapshots: Mapping[Path, tuple[int, int, int, int, int, int]],
) -> None:
    for path, expected in snapshots.items():
        try:
            current = os.lstat(path)
        except OSError as exc:
            raise ArchivePreflightError(
                f"restore directory changed during validation: {path}: {exc}"
            ) from exc
        if _fingerprint(current) != expected:
            raise ArchivePreflightError(
                f"restore directory changed during validation: {path}"
            )


def validate_archive_preflight(
    *,
    canonical_manifest: Path,
    backup_proof: Path,
    object_manifest: Path,
    retention_evidence: Path,
    restore_transcript: Path,
    restore_root: Path,
    expected_count: int | None = CANONICAL_RAW_COUNT,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Validate evidence and restored bytes without writing or authorizing changes."""

    current_time = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    canonical_file = _admit_file(canonical_manifest, label="canonical raw manifest")
    proof_file = _admit_file(backup_proof, label="backup proof")
    object_file = _admit_file(object_manifest, label="object manifest")
    retention_file = _admit_file(retention_evidence, label="retention evidence")
    transcript_file = _admit_file(restore_transcript, label="restore transcript")
    _, canonical = _canonical_records(canonical_file, expected_count=expected_count)
    total_bytes = sum(int(row["size"]) for row in canonical.values())

    retention = _retention(
        retention_file,
        expected_count=len(canonical),
        now=current_time,
    )
    objects = _objects(object_file, canonical=canonical, retention=retention)
    _transcript(
        transcript_file,
        canonical_sha256=canonical_file.sha256,
        object_manifest_sha256=object_file.sha256,
        retention_evidence_sha256=retention_file.sha256,
        objects=objects,
        total_bytes=total_bytes,
        now=current_time,
    )
    proof = _proof(
        proof_file,
        canonical_sha256=canonical_file.sha256,
        object_manifest_sha256=object_file.sha256,
        retention_evidence_sha256=retention_file.sha256,
        restore_transcript_sha256=transcript_file.sha256,
        object_count=len(canonical),
        total_bytes=total_bytes,
        now=current_time,
    )

    restore_root = lexical_absolute(restore_root)
    snapshots = _directory_snapshot(restore_root)
    actual_paths = _restore_paths(restore_root)
    expected_paths = set(canonical)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise ArchivePreflightError(
            f"restore path set differs: missing={missing[:3]}, extra={extra[:3]}"
        )
    for relative, expected in canonical.items():
        size, digest = _hash_file(
            restore_root.joinpath(*PurePosixPath(relative).parts),
            label=f"restored file {relative}",
        )
        if size != expected["size"] or digest != expected["sha256"]:
            raise ArchivePreflightError(
                f"restored file size/hash differs for {relative}"
            )
    _verify_directories(snapshots)

    return {
        "archive_map_mutated": False,
        "backup_id": proof["backup_id"],
        "cutover_receipt_verified": False,
        "canonical_manifest": {
            "bytes": len(canonical_file.payload),
            "object_count": len(canonical),
            "path": str(canonical_file.path),
            "payload_bytes": total_bytes,
            "sha256": canonical_file.sha256,
        },
        "evidence": {
            "backup_proof_sha256": proof_file.sha256,
            "object_manifest_sha256": object_file.sha256,
            "restore_transcript_sha256": transcript_file.sha256,
            "retention_evidence_sha256": retention_file.sha256,
        },
        "evidence_bundle_structurally_valid": True,
        "externalization_authorized": False,
        "independent_approval_verified": False,
        "no_mutations": True,
        "preflight_passed": True,
        "preflight_scope": "structural_only",
        "provider_authenticity_verified": False,
        "restore_root": str(restore_root),
        "schema": REPORT_SCHEMA,
        "status": "evidence_bundle_structurally_valid",
        "validation_domain": "archive_evidence_structure",
    }


__all__ = [
    "ArchivePreflightError",
    "CANONICAL_RAW_COUNT",
    "CANONICAL_RAW_MANIFEST",
    "CANONICAL_RAW_SCHEMA",
    "OBJECT_SCHEMA",
    "PROOF_SCHEMA",
    "REPORT_SCHEMA",
    "RETENTION_SCHEMA",
    "TRANSCRIPT_SCHEMA",
    "validate_archive_preflight",
]
