"""Materialize audited experiment code into a human-readable working tree.

The sealed environment snapshots remain the canonical audit artifacts.  This
module creates byte-identical regular-file copies whose version is explicit in
their path: ``experiments4``, ``experiments5``, or ``experiments6``. Every
copied file is backed by an audited SHA-256 record.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping


class ReadableCodeError(RuntimeError):
    """Raised when the readable code view cannot be proved or synchronized."""


REGISTRY_PATH = "configs/environment/variant-registry-audited-v2.json"
OVERLAY_PATH = "environments/exp45-audited-overlay-v2-782df2d84077f93b"
BASE_PATH = "environments/exp6-base-v1-4f6a15ffa3db8dc3"
BASE_MANIFEST_PATH = f"{BASE_PATH}/.experiment-env/manifest.jsonl"
MANIFEST_PATH = "readable_code_manifest.json"

VERSION_LABELS = {
    "experiments4": "experiments4",
    "experiments5": "experiments5",
    "experiments6": "experiments6",
}
READABLE_VERSION_DIRECTORIES = tuple(VERSION_LABELS.values())
LEGACY_VERSION_LABELS = frozenset({"eval4", "eval5", "base6"})

METHOD_COMPONENTS = {
    "vanillallm": "vanilla_llm",
    "vanilla_llm": "vanilla_llm",
    "rag": "rag",
    "mem0": "mem0",
    "langmem": "langmem",
    "self_refine": "self_refine",
    "ours_memory": "our_memory",
    "our_memory": "our_memory",
    "preference_memory": "our_memory",
    "remem": "remem",
    "e_mem": "emem",
    "emem": "emem",
}

READABLE_VERSIONS = frozenset(READABLE_VERSION_DIRECTORIES)
MANAGED_VERSIONS = READABLE_VERSIONS | LEGACY_VERSION_LABELS
REQUIRED_DIRECTORIES = tuple(
    sorted(
        {
            *(f"data/sources/{version}" for version in READABLE_VERSIONS),
            *(f"experiments/{version}" for version in READABLE_VERSIONS),
            "experiments/variants",
        }
    )
)

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
_TEMP_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadableCodeError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReadableCodeError(f"JSON must contain an object: {path}")
    return value


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _identity(value: os.stat_result) -> tuple[int, int]:
    return value.st_dev, value.st_ino


def _relative_parts(relative_path: str) -> tuple[str, ...]:
    if not isinstance(relative_path, str) or not relative_path:
        raise ReadableCodeError("readable destination must be a non-empty string")
    if "\\" in relative_path or "//" in relative_path or relative_path.startswith("./"):
        raise ReadableCodeError(f"readable destination must use POSIX separators: {relative_path!r}")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise ReadableCodeError(f"unsafe readable destination: {relative_path!r}")
    return pure.parts


def _real_root(root: Path) -> Path:
    root_fd: int | None = None
    try:
        before = os.lstat(root)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
            raise ReadableCodeError(f"readable-code root is not a real directory: {root}")
        root_fd = os.open(root, _DIRECTORY_FLAGS)
        opened = os.fstat(root_fd)
        canonical = root.resolve(strict=True)
        after = os.lstat(root)
        canonical_stat = os.lstat(canonical)
        values = (before, opened, after, canonical_stat)
        if (
            any(stat.S_ISLNK(value.st_mode) for value in values)
            or any(not stat.S_ISDIR(value.st_mode) for value in values)
            or any(_identity(value) != _identity(opened) for value in values)
        ):
            raise ReadableCodeError(f"readable-code root binding changed: {root}")
        return canonical
    except ReadableCodeError:
        raise
    except OSError as exc:
        raise ReadableCodeError(f"readable-code root is not a real directory: {root}") from exc
    finally:
        if root_fd is not None:
            try:
                os.close(root_fd)
            except OSError:
                pass


def _open_parent(root: Path, relative_path: str, *, create: bool) -> tuple[int, str]:
    parts = _relative_parts(relative_path)
    current_fd: int | None = None
    try:
        current_fd = os.open(root, _DIRECTORY_FLAGS)
        for component in parts[:-1]:
            try:
                next_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    raise ReadableCodeError(
                        f"missing readable destination parent: {relative_path}"
                    ) from None
                os.mkdir(component, 0o755, dir_fd=current_fd)
                next_fd = os.open(component, _DIRECTORY_FLAGS, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        return current_fd, parts[-1]
    except ReadableCodeError:
        if current_fd is not None:
            try:
                os.close(current_fd)
            except OSError:
                pass
        raise
    except OSError as exc:
        if current_fd is not None:
            try:
                os.close(current_fd)
            except OSError:
                pass
        raise ReadableCodeError(f"unsafe readable destination parent: {relative_path}") from exc


def _destination_stat(parent_fd: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None


def _validate_unique_regular(value: os.stat_result | None, relative_path: str) -> None:
    if value is None:
        return
    if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
        raise ReadableCodeError(
            f"readable code copy is not a unique regular file: {relative_path}"
        )


def _ensure_directory(root: Path, relative_path: str, *, create: bool) -> None:
    parent_fd: int | None = None
    try:
        parent_fd, _ = _open_parent(
            root,
            f"{relative_path}/.directory-check",
            create=create,
        )
    finally:
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _normalized_component(value: str) -> str:
    return value.lower().replace("-", "_")


def _target_for(origin_root: str, relative_path: str) -> tuple[str, str, str | None]:
    version = VERSION_LABELS.get(origin_root)
    if version is None:
        raise ReadableCodeError(f"unsupported source origin: {origin_root}")
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise ReadableCodeError(f"unsafe audited source path: {relative_path!r}")

    normalized = [_normalized_component(part) for part in pure.parts]
    for index, component in enumerate(normalized):
        family = METHOD_COMPONENTS.get(component)
        if family is None:
            continue
        prefix = normalized[:index]
        context: list[str] = []
        if "extended_schema" in prefix:
            context.append("extended_schema")
        elif "dataset_tools" in prefix:
            context.append("dataset_tools")
        elif "memory_dev" in prefix:
            context.append("memory_dev")
        tail = list(pure.parts[index + 1 :])
        if not tail:
            tail = [pure.parts[index]]
        target = PurePosixPath("methods", family, version, *context, *tail).as_posix()
        return target, "method", family

    if "evaluation" in normalized:
        index = normalized.index("evaluation")
        tail = list(pure.parts[index + 1 :]) or [pure.parts[-1]]
        return (
            PurePosixPath("experiments", version, "evaluation", *tail).as_posix(),
            "experiment",
            None,
        )

    if normalized[0] in {"data", "config", "configs"}:
        return (
            PurePosixPath("data", "sources", version, *pure.parts).as_posix(),
            "data",
            None,
        )

    return (
        PurePosixPath("experiments", version, "shared", *pure.parts).as_posix(),
        "experiment",
        None,
    )


def _contained_regular_file(base: Path, relative_path: str) -> Path | None:
    candidate = base.joinpath(*PurePosixPath(relative_path).parts)
    try:
        base_real = base.resolve(strict=True)
        candidate_real = candidate.resolve(strict=True)
        candidate_real.relative_to(base_real)
        path_stat = os.lstat(candidate)
    except (OSError, ValueError):
        return None
    if stat.S_ISLNK(path_stat.st_mode) or not stat.S_ISREG(path_stat.st_mode):
        return None
    return candidate_real


def _source_candidates(root: Path, origin_root: str, relative_path: str) -> Iterable[tuple[str, Path]]:
    if origin_root in {"experiments4", "experiments5"}:
        yield (
            "sealed_overlay",
            root / OVERLAY_PATH / "origins" / origin_root / relative_path,
        )
    elif origin_root == "experiments6":
        yield "sealed_base", root / BASE_PATH / relative_path
    yield "original_source", root.parent / origin_root / relative_path


def _resolve_source(
    root: Path,
    origin_root: str,
    relative_path: str,
    expected_sha256: str,
) -> tuple[Path, str]:
    mismatches: list[str] = []
    for source_kind, candidate in _source_candidates(root, origin_root, relative_path):
        base = root / OVERLAY_PATH / "origins" / origin_root
        if source_kind == "sealed_base":
            base = root / BASE_PATH
        elif source_kind == "original_source":
            base = root.parent / origin_root
        resolved = _contained_regular_file(base, relative_path)
        if resolved is None:
            continue
        actual_sha256 = _sha256_file(resolved)
        if actual_sha256 == expected_sha256:
            return resolved, source_kind
        mismatches.append(f"{candidate}: {actual_sha256}")
    detail = f"; mismatches={mismatches!r}" if mismatches else ""
    raise ReadableCodeError(
        f"no audited source matches {origin_root}/{relative_path} {expected_sha256}{detail}"
    )


def build_readable_plan(root: Path) -> dict[str, Any]:
    """Return the deterministic copy plan derived from all audited variants."""

    root = root.resolve(strict=True)
    registry = _load_json(root / REGISTRY_PATH)
    variants = registry.get("variants")
    if not isinstance(variants, list) or registry.get("variant_count") != len(variants):
        raise ReadableCodeError("audited variant registry is malformed")

    evidence_by_source: dict[tuple[str, str], dict[str, Any]] = {}
    labels_by_source: dict[tuple[str, str], set[str]] = defaultdict(set)
    stages_by_source: dict[tuple[str, str], set[str]] = defaultdict(set)
    variant_evidence: dict[str, list[tuple[str, str]]] = {}

    for variant in variants:
        if not isinstance(variant, dict):
            raise ReadableCodeError("audited variant must be an object")
        variant_id = variant.get("variant_id")
        stage = variant.get("stage")
        evidence = variant.get("origin_evidence")
        if not isinstance(variant_id, str) or not variant_id:
            raise ReadableCodeError("audited variant_id must be non-empty")
        if not isinstance(stage, str) or not stage:
            raise ReadableCodeError(f"variant {variant_id} has no stage")
        if not isinstance(evidence, list) or not evidence:
            raise ReadableCodeError(f"variant {variant_id} has no origin evidence")
        keys: list[tuple[str, str]] = []
        for item in evidence:
            if not isinstance(item, dict):
                raise ReadableCodeError(f"variant {variant_id} has malformed evidence")
            origin_root = item.get("origin_root")
            relative_path = item.get("relative_path")
            audited_sha256 = item.get("sha256")
            if not all(isinstance(value, str) and value for value in (origin_root, relative_path, audited_sha256)):
                raise ReadableCodeError(f"variant {variant_id} has incomplete evidence")
            key = (origin_root, relative_path)
            previous = evidence_by_source.get(key)
            if previous is not None and previous["audited_sha256"] != audited_sha256:
                raise ReadableCodeError(f"conflicting audited hashes for {origin_root}/{relative_path}")
            evidence_by_source[key] = {
                "audited_sha256": audited_sha256,
                "evidence_id": item.get("evidence_id"),
                "source_manifest_record_id": item.get("source_manifest_record_id"),
            }
            labels_by_source[key].add(variant_id)
            stages_by_source[key].add(stage)
            keys.append(key)
        variant_evidence[variant_id] = keys

    base_regular_count = 0
    base_symlink_count = 0
    try:
        with (root / BASE_MANIFEST_PATH).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReadableCodeError(
                        f"invalid base manifest record at line {line_number}"
                    ) from exc
                if not isinstance(record, dict) or record.get("source_root_id") != "experiments6":
                    raise ReadableCodeError(
                        f"invalid base manifest provenance at line {line_number}"
                    )
                record_type = record.get("record_type")
                if record_type == "symlink":
                    base_symlink_count += 1
                    continue
                if record_type != "regular":
                    raise ReadableCodeError(
                        f"invalid base manifest record type at line {line_number}"
                    )
                relative_path = record.get("path")
                audited_sha256 = record.get("sha256")
                source_record_id = record.get("source_record_id")
                if not all(
                    isinstance(value, str) and value
                    for value in (relative_path, audited_sha256, source_record_id)
                ):
                    raise ReadableCodeError(
                        f"incomplete base manifest record at line {line_number}"
                    )
                key = ("experiments6", relative_path)
                previous = evidence_by_source.get(key)
                if previous is not None and previous["audited_sha256"] != audited_sha256:
                    raise ReadableCodeError(
                        f"base manifest hash conflicts with variant evidence: {relative_path}"
                    )
                if previous is None:
                    evidence_by_source[key] = {
                        "audited_sha256": audited_sha256,
                        "evidence_id": f"experiments6:{source_record_id}",
                        "source_manifest_record_id": source_record_id,
                    }
                labels_by_source[key].add("experiments6_inventory")
                stages_by_source[key].add("runtime_base")
                base_regular_count += 1
    except OSError as exc:
        raise ReadableCodeError(f"cannot read sealed base manifest: {exc}") from exc
    if base_regular_count != 288 or base_symlink_count != 26:
        raise ReadableCodeError(
            "sealed base inventory count mismatch: "
            f"regular={base_regular_count} symlink={base_symlink_count}"
        )

    entries: list[dict[str, Any]] = []
    target_owners: dict[str, tuple[str, str]] = {}
    target_by_source: dict[tuple[str, str], str] = {}
    for key in sorted(evidence_by_source):
        origin_root, relative_path = key
        evidence = evidence_by_source[key]
        source, source_kind = _resolve_source(
            root,
            origin_root,
            relative_path,
            evidence["audited_sha256"],
        )
        target_path, category, method_family = _target_for(origin_root, relative_path)
        owner = target_owners.get(target_path)
        if owner is not None and owner != key:
            raise ReadableCodeError(f"readable target collision: {target_path}: {owner!r} vs {key!r}")
        target_owners[target_path] = key
        target_by_source[key] = target_path
        try:
            source_path = source.relative_to(root).as_posix()
        except ValueError:
            source_path = source.as_posix()
        source_mode = stat.S_IMODE(os.stat(source, follow_symlinks=False).st_mode)
        entries.append(
            {
                "audited_sha256": evidence["audited_sha256"],
                "category": category,
                "evidence_id": evidence["evidence_id"],
                "method_family": method_family,
                "mode": "755" if source_mode & 0o111 else "644",
                "semantic_labels": sorted(labels_by_source[key]),
                "source_kind": source_kind,
                "source_manifest_record_id": evidence["source_manifest_record_id"],
                "source_origin": origin_root,
                "source_path": source_path,
                "source_relative_path": relative_path,
                "stages": sorted(stages_by_source[key]),
                "target_path": target_path,
                "version_label": VERSION_LABELS[origin_root],
            }
        )

    variant_documents: dict[str, dict[str, Any]] = {}
    for variant in sorted(variants, key=lambda value: value["variant_id"]):
        variant_id = variant["variant_id"]
        code_paths = sorted({target_by_source[key] for key in variant_evidence[variant_id]})
        variant_documents[f"experiments/variants/{variant_id}.json"] = {
            "affected_outputs": variant.get("affected_outputs", []),
            "code_paths": code_paths,
            "distinction": variant.get("distinction", ""),
            "schema": "experiments7-readable-variant/v1",
            "semantic_label": variant_id,
            "source_origins": sorted({key[0] for key in variant_evidence[variant_id]}),
            "stage": variant["stage"],
            "variant_id": variant_id,
        }

    origin_counts = Counter(entry["source_origin"] for entry in entries)
    category_counts = Counter(entry["category"] for entry in entries)
    source_kind_counts = Counter(entry["source_kind"] for entry in entries)
    return {
        "experiments6_inventory": {
            "regular_file_count": base_regular_count,
            "symlink_records_not_materialized": base_symlink_count,
        },
        "category_counts": dict(sorted(category_counts.items())),
        "entries": entries,
        "entry_count": len(entries),
        "policies": {
            "actual_regular_file_copies": True,
            "audited_sha256_required": True,
            "experiments6_full_regular_inventory": True,
            "existing_variant_runtime_unchanged": True,
            "sealed_sources_unchanged": True,
            "symlink_aliases": False,
            "version_labels": dict(VERSION_LABELS),
        },
        "schema": "experiments7-readable-code-manifest/v1",
        "source_kind_counts": dict(sorted(source_kind_counts.items())),
        "source_origin_counts": dict(sorted(origin_counts.items())),
        "variant_count": len(variant_documents),
        "variant_documents": variant_documents,
    }


def _managed_files(root: Path) -> set[str]:
    paths: set[str] = set()
    roots = [root / "data/sources"]
    roots.extend(root / "experiments" / version for version in MANAGED_VERSIONS)
    roots.append(root / "experiments/variants")
    methods = root / "methods"
    if methods.is_dir() and not methods.is_symlink():
        for family in methods.iterdir():
            if not family.is_dir() or family.is_symlink():
                continue
            roots.extend(family / version for version in MANAGED_VERSIONS)
    for managed_root in roots:
        try:
            value = os.lstat(managed_root)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ReadableCodeError(f"cannot inspect managed code root: {managed_root}") from exc
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
            raise ReadableCodeError(f"managed code root is unsafe: {managed_root}")
        for path in managed_root.rglob("*"):
            if path.is_symlink():
                raise ReadableCodeError(f"managed code contains a symlink: {path}")
            if path.is_file():
                paths.add(path.relative_to(root).as_posix())
            elif not path.is_dir():
                raise ReadableCodeError(f"managed code contains a special file: {path}")
    return paths


def _is_managed_payload(relative_path: str, *, allow_legacy: bool = False) -> bool:
    parts = _relative_parts(relative_path)
    versions = MANAGED_VERSIONS if allow_legacy else READABLE_VERSIONS
    method_families = frozenset(METHOD_COMPONENTS.values())
    return (
        len(parts) >= 4
        and parts[:2] == ("data", "sources")
        and parts[2] in versions
    ) or (
        len(parts) >= 3
        and parts[0] == "experiments"
        and parts[1] in versions
    ) or (
        len(parts) == 3
        and parts[:2] == ("experiments", "variants")
        and parts[2].endswith(".json")
    ) or (
        len(parts) >= 4
        and parts[0] == "methods"
        and parts[1] in method_families
        and parts[2] in versions
    )


def _read_regular_file(path: Path) -> bytes:
    file_fd: int | None = None
    try:
        file_fd = os.open(path, _READ_FLAGS)
        opened = os.fstat(file_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ReadableCodeError(f"audited source is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final = os.fstat(file_fd)
        if (
            _identity(final) != _identity(opened)
            or final.st_size != opened.st_size
            or final.st_mtime_ns != opened.st_mtime_ns
            or final.st_ctime_ns != opened.st_ctime_ns
        ):
            raise ReadableCodeError(f"audited source changed while reading: {path}")
        return b"".join(chunks)
    except ReadableCodeError:
        raise
    except OSError as exc:
        raise ReadableCodeError(f"cannot safely read audited source: {path}") from exc
    finally:
        if file_fd is not None:
            try:
                os.close(file_fd)
            except OSError:
                pass


def _verify_regular_copy(root: Path, relative_path: str, expected_sha256: str) -> None:
    parent_fd: int | None = None
    file_fd: int | None = None
    try:
        parent_fd, name = _open_parent(root, relative_path, create=False)
        current = _destination_stat(parent_fd, name)
        _validate_unique_regular(current, relative_path)
        if current is None:
            raise ReadableCodeError(f"missing readable code copy: {relative_path}")
        file_fd = os.open(name, _READ_FLAGS, dir_fd=parent_fd)
        opened = os.fstat(file_fd)
        if _identity(opened) != _identity(current):
            raise ReadableCodeError(f"readable code binding changed: {relative_path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        final = os.fstat(file_fd)
        if (
            _identity(final) != _identity(opened)
            or final.st_size != opened.st_size
            or final.st_mtime_ns != opened.st_mtime_ns
            or final.st_ctime_ns != opened.st_ctime_ns
        ):
            raise ReadableCodeError(f"readable code changed while verifying: {relative_path}")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ReadableCodeError(
                f"readable code copy hash mismatch: {relative_path}: "
                f"{actual_sha256} != {expected_sha256}"
            )
    except ReadableCodeError:
        raise
    except OSError as exc:
        raise ReadableCodeError(f"cannot safely verify readable code: {relative_path}") from exc
    finally:
        if file_fd is not None:
            try:
                os.close(file_fd)
            except OSError:
                pass
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _atomic_write(root: Path, relative_path: str, payload: bytes, mode: int) -> None:
    if relative_path != MANIFEST_PATH and not _is_managed_payload(relative_path):
        raise ReadableCodeError(f"refusing undeclared readable destination: {relative_path}")
    parent_fd: int | None = None
    file_fd: int | None = None
    temp_name: str | None = None
    try:
        parent_fd, name = _open_parent(root, relative_path, create=True)
        before = _destination_stat(parent_fd, name)
        _validate_unique_regular(before, relative_path)
        temp_name = f".{name}.tmp-{secrets.token_hex(12)}"
        file_fd = os.open(temp_name, _TEMP_FLAGS, mode, dir_fd=parent_fd)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(file_fd, remaining)
            if written <= 0:
                raise ReadableCodeError(f"short write for readable code: {relative_path}")
            remaining = remaining[written:]
        os.fchmod(file_fd, mode)
        os.fsync(file_fd)
        os.close(file_fd)
        file_fd = None
        current = _destination_stat(parent_fd, name)
        if (before is None) != (current is None) or (
            before is not None
            and current is not None
            and _identity(before) != _identity(current)
        ):
            raise ReadableCodeError(f"readable destination binding changed: {relative_path}")
        os.replace(
            temp_name,
            name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        temp_name = None
        os.fsync(parent_fd)
    except ReadableCodeError:
        raise
    except OSError as exc:
        raise ReadableCodeError(f"cannot safely write readable code: {relative_path}") from exc
    finally:
        if file_fd is not None:
            try:
                os.close(file_fd)
            except OSError:
                pass
        if parent_fd is not None and temp_name is not None:
            try:
                os.unlink(temp_name, dir_fd=parent_fd)
            except OSError:
                pass
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _unlink_stale(root: Path, relative_path: str) -> None:
    if not _is_managed_payload(relative_path, allow_legacy=True):
        raise ReadableCodeError(f"refusing undeclared stale destination: {relative_path}")
    parent_fd: int | None = None
    try:
        parent_fd, name = _open_parent(root, relative_path, create=False)
        current = _destination_stat(parent_fd, name)
        if current is None:
            return
        _validate_unique_regular(current, relative_path)
        os.unlink(name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    except ReadableCodeError:
        raise
    except OSError as exc:
        raise ReadableCodeError(f"cannot safely remove stale readable code: {relative_path}") from exc
    finally:
        if parent_fd is not None:
            try:
                os.close(parent_fd)
            except OSError:
                pass


def _legacy_directories(root: Path) -> list[Path]:
    paths = [root / "data/sources" / version for version in LEGACY_VERSION_LABELS]
    paths.extend(root / "experiments" / version for version in LEGACY_VERSION_LABELS)
    methods = root / "methods"
    if methods.is_dir() and not methods.is_symlink():
        for family in methods.iterdir():
            if family.is_dir() and not family.is_symlink():
                paths.extend(family / version for version in LEGACY_VERSION_LABELS)
    return sorted(paths, key=lambda path: path.as_posix())


def _remove_legacy_directories(root: Path) -> None:
    for legacy_root in _legacy_directories(root):
        if not legacy_root.exists() and not legacy_root.is_symlink():
            continue
        value = os.lstat(legacy_root)
        if stat.S_ISLNK(value.st_mode) or not stat.S_ISDIR(value.st_mode):
            raise ReadableCodeError(f"legacy version root is unsafe: {legacy_root}")
        descendants = sorted(
            legacy_root.rglob("*"),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for path in descendants:
            value = os.lstat(path)
            if stat.S_ISLNK(value.st_mode):
                raise ReadableCodeError(f"legacy version root contains a symlink: {path}")
            if stat.S_ISDIR(value.st_mode):
                try:
                    path.rmdir()
                except OSError:
                    pass
            elif not stat.S_ISREG(value.st_mode):
                raise ReadableCodeError(f"legacy version root contains a special file: {path}")
        try:
            legacy_root.rmdir()
        except OSError:
            pass
        if legacy_root.exists() or legacy_root.is_symlink():
            raise ReadableCodeError(
                f"legacy version directory is not empty after synchronization: {legacy_root}"
            )


def sync_readable_code(root: Path, *, check: bool = False) -> dict[str, Any]:
    """Create or verify the readable code copies and per-variant documents."""

    root = _real_root(root)
    plan = build_readable_plan(root)
    entries = plan["entries"]
    variant_documents = plan["variant_documents"]
    expected_files = {entry["target_path"] for entry in entries} | set(variant_documents)

    if check:
        if any(path.exists() or path.is_symlink() for path in _legacy_directories(root)):
            raise ReadableCodeError("legacy eval4/eval5/base6 directories are still present")
        for relative_path in REQUIRED_DIRECTORIES:
            _ensure_directory(root, relative_path, create=False)
        for entry in entries:
            _verify_regular_copy(root, entry["target_path"], entry["audited_sha256"])
        for relative_path, document in variant_documents.items():
            _verify_regular_copy(
                root,
                relative_path,
                _sha256_bytes(_json_bytes(document)),
            )
        manifest_document = dict(plan)
        manifest_document.pop("variant_documents")
        _verify_regular_copy(root, MANIFEST_PATH, _sha256_bytes(_json_bytes(manifest_document)))
        actual_files = _managed_files(root)
        if actual_files != expected_files:
            raise ReadableCodeError(
                f"managed readable code set mismatch: missing={sorted(expected_files - actual_files)!r} "
                f"extra={sorted(actual_files - expected_files)!r}"
            )
        return {
            "entry_count": len(entries),
            "status": "ok",
            "variant_count": len(variant_documents),
        }

    for relative_path in REQUIRED_DIRECTORIES:
        _ensure_directory(root, relative_path, create=True)

    previous_targets: set[str] = set()
    manifest_path = root / MANIFEST_PATH
    if manifest_path.is_file() and not manifest_path.is_symlink():
        previous = _load_json(manifest_path)
        previous_entries = previous.get("entries", [])
        if isinstance(previous_entries, list):
            previous_targets.update(
                item["target_path"]
                for item in previous_entries
                if isinstance(item, dict) and isinstance(item.get("target_path"), str)
            )
        previous_count = previous.get("variant_count")
        if isinstance(previous_count, int):
            variants_root = root / "experiments/variants"
            if variants_root.is_dir() and not variants_root.is_symlink():
                previous_targets.update(
                    path.relative_to(root).as_posix()
                    for path in variants_root.glob("*.json")
                    if path.is_file() and not path.is_symlink()
                )

    for entry in entries:
        source = Path(entry["source_path"])
        if not source.is_absolute():
            source = root / source
        payload = _read_regular_file(source)
        if _sha256_bytes(payload) != entry["audited_sha256"]:
            raise ReadableCodeError(f"source changed while materializing: {source}")
        _atomic_write(root, entry["target_path"], payload, int(entry["mode"], 8))

    for relative_path, document in variant_documents.items():
        _atomic_write(root, relative_path, _json_bytes(document), 0o644)

    for relative_path in sorted(previous_targets - expected_files):
        _unlink_stale(root, relative_path)
    _remove_legacy_directories(root)

    manifest_document = dict(plan)
    manifest_document.pop("variant_documents")
    _atomic_write(root, MANIFEST_PATH, _json_bytes(manifest_document), 0o644)
    actual_files = _managed_files(root)
    if actual_files != expected_files:
        raise ReadableCodeError(
            f"managed readable code set mismatch: missing={sorted(expected_files - actual_files)!r} "
            f"extra={sorted(actual_files - expected_files)!r}"
        )
    return {
        "entry_count": len(entries),
        "status": "ok",
        "variant_count": len(variant_documents),
    }
