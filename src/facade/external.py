"""Validated external subprocess contract for the experiments7 facade."""

from __future__ import annotations

from dataclasses import dataclass
import array
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
from typing import Any, Mapping

from .metadata import STAGES
from .registry import (
    Bundle,
    FacadeError,
    HEX64,
    SAFE_ID,
    SECRET_VALUE,
    require,
)
from .selection import canonical_bytes, sha256_bytes, sha256_file
from .sandbox import (
    DEFAULT_SANDBOX_POLICY,
    DeclaredOutputInode,
    RUNTIME_THREAT_MODEL_SHA256,
    SandboxUnavailable,
    current_subject_identity,
    make_preexec,
    parse_evidence,
    prepare_backend,
    require_strict_runtime_proof_environment,
    sandbox_policy_sha256,
    subject_identity_sha256,
)
from .snapshots import SnapshotPublication, validate_snapshot_publication


RUNTIME_SCHEMA = "experiments7-runtime-closure/v2"
RUNTIME_SEAL_SCHEMA = "experiments7-runtime-closure-seal/v2"
RUN_CONFIG_SCHEMA = "experiments7-external-run-config/v2"
ENV_NAME = re.compile(r"^[A-Z_][A-Z0-9_]*$")
PLACEHOLDER = re.compile(r"\{([^{}]+)\}")
RESERVED_OUTPUTS = frozenset({
    "selection.json", "blocked.json", "external-result.json", "steps", "home", "tmp",
})
KNOWN_PREREQUISITES = frozenset({
    "published_origin_snapshot",
    "published_exact_data_snapshot",
    "declared_runtime_dependencies",
    "credentials_or_local_service",
})
FS_IOC_GETFLAGS = 0x80086601
FS_IMMUTABLE_FL = 0x00000010


def _kernel_immutable_or_readonly_mount(path: Path) -> bool:
    try:
        if os.statvfs(path).f_flag & getattr(os, "ST_RDONLY", 1):
            return True
    except OSError as exc:
        raise FacadeError("FILESYSTEM_FLAGS_UNOBSERVABLE", str(path)) from exc
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return False
    try:
        value = array.array("L", [0])
        try:
            fcntl.ioctl(descriptor, FS_IOC_GETFLAGS, value, True)
        except OSError as exc:
            if exc.errno in {errno.ENOTTY, errno.EOPNOTSUPP, errno.ENOTSUP, errno.EPERM}:
                return False
            raise FacadeError("FILESYSTEM_IMMUTABILITY_UNOBSERVABLE", str(path)) from exc
        return bool(value[0] & FS_IMMUTABLE_FL)
    finally:
        os.close(descriptor)


def _require_effective_subject_immutable(path: Path, *, code: str) -> None:
    state = path.lstat()
    if _kernel_immutable_or_readonly_mount(path):
        return
    require(state.st_uid != os.geteuid(), code, str(path))
    require(not _subject_can_write(state.st_mode, state.st_uid, state.st_gid),
            code, str(path))


def _safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return (
        not path.is_absolute()
        and all(part not in {"", ".", ".."} for part in path.parts)
        and not any(character in value for character in "*?[")
    )


def _safe_owned_file(
    root: Path,
    relative: str,
    *,
    code: str,
    immutable: bool,
    effective_subject_immutable: bool = False,
) -> Path:
    require(_safe_relative(relative), code, relative)
    root = root.resolve(strict=True)
    path = root / relative
    cursor = root
    for part in PurePosixPath(relative).parts[:-1]:
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError as exc:
            raise FacadeError(code, str(cursor)) from exc
        require(stat.S_ISDIR(mode) and not cursor.is_symlink(), code, str(cursor))
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise FacadeError(code, str(path)) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(), code, str(path))
    if immutable:
        require(mode & 0o222 == 0, "RUNTIME_INPUT_NOT_IMMUTABLE", str(path))
    if effective_subject_immutable:
        _require_effective_subject_immutable(
            root, code="RUNTIME_INPUT_ANCESTOR_NOT_EFFECTIVELY_IMMUTABLE"
        )
        _require_effective_subject_immutable(
            path, code="RUNTIME_INPUT_NOT_EFFECTIVELY_IMMUTABLE"
        )
        cursor = root
        for part in PurePosixPath(relative).parts[:-1]:
            cursor = cursor / part
            _require_effective_subject_immutable(
                cursor, code="RUNTIME_INPUT_ANCESTOR_NOT_EFFECTIVELY_IMMUTABLE"
            )
    return path


def _safe_control(path: Path, *, root: Path, code: str) -> tuple[dict[str, Any], str, int]:
    root = root.resolve(strict=True)
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise FacadeError(code, str(path)) from exc
    require(_safe_relative(relative), code, relative)
    parts = PurePosixPath(relative).parts
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    descriptor = os.open(root, directory_flags)
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(part, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        file_flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        file_descriptor = os.open(parts[-1], file_flags, dir_fd=descriptor)
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise FacadeError(code, relative) from exc
    finally:
        os.close(descriptor)
    try:
        before = os.fstat(file_descriptor)
        require(stat.S_ISREG(before.st_mode), code, relative)
        require(before.st_mode & 0o222 == 0, "RUNTIME_CONTROL_NOT_IMMUTABLE", relative)
        chunks: list[bytes] = []
        while True:
            block = os.read(file_descriptor, 1024 * 1024)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(file_descriptor)
        require(
            (before.st_dev, before.st_ino, before.st_mode, before.st_size)
            == (after.st_dev, after.st_ino, after.st_mode, after.st_size),
            "RUNTIME_CONTROL_CHANGED_DURING_READ",
            relative,
        )
    finally:
        os.close(file_descriptor)
    payload = b"".join(chunks)
    digest = hashlib.sha256(payload).hexdigest()

    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            require(key not in value, "DUPLICATE_JSON_KEY", key)
            value[key] = child
        return value

    try:
        parsed = json.loads(payload.decode("utf-8"), object_pairs_hook=no_duplicates)
    except FacadeError:
        raise
    except Exception as exc:
        raise FacadeError("INVALID_JSON", relative) from exc
    require(isinstance(parsed, dict), "RUNTIME_CONTROL_NOT_OBJECT", relative)
    return parsed, digest, len(payload)


def _subject_can_write(mode: int, uid: int, gid: int) -> bool:
    if uid == os.geteuid() and mode & stat.S_IWUSR:
        return True
    if gid in {os.getegid(), *os.getgroups()} and mode & stat.S_IWGRP:
        return True
    return bool(mode & stat.S_IWOTH)


def _xattr_binding(path: Path) -> str:
    try:
        names = sorted(os.listxattr(path, follow_symlinks=False))
    except OSError as exc:
        if exc.errno in {errno.ENOTSUP, errno.EOPNOTSUPP}:
            names = []
        else:
            raise FacadeError(
                "RUNTIME_ARTIFACT_XATTR_UNOBSERVABLE",
                {"path": str(path), "errno_name": errno.errorcode.get(exc.errno, "UNKNOWN")},
            ) from exc
    records = []
    for name in names:
        try:
            value = os.getxattr(path, name, follow_symlinks=False)
        except OSError as exc:
            raise FacadeError(
                "RUNTIME_ARTIFACT_XATTR_UNOBSERVABLE",
                {"path": str(path), "name": name},
            ) from exc
        records.append({"name": name, "sha256": hashlib.sha256(value).hexdigest(), "bytes": len(value)})
    require(
        not any(row["name"] == "security.capability" for row in records),
        "RUNTIME_ARTIFACT_FILE_CAPABILITY",
        str(path),
    )
    require(
        not any(row["name"].startswith("system.posix_acl_") for row in records),
        "RUNTIME_ARTIFACT_ACL_UNSUPPORTED",
        str(path),
    )
    return sha256_bytes(canonical_bytes(records))


def _ancestor_binding(path: Path) -> str:
    records: list[dict[str, int | str]] = []
    cursor = Path(path.anchor)
    anchor_state = cursor.lstat()
    require(
        stat.S_ISDIR(anchor_state.st_mode) and not cursor.is_symlink(),
        "RUNTIME_ARTIFACT_ANCESTOR_UNSAFE",
        str(cursor),
    )
    _require_effective_subject_immutable(
        cursor, code="RUNTIME_ARTIFACT_ANCESTOR_NOT_EFFECTIVELY_IMMUTABLE"
    )
    records.append({
        "path": str(cursor),
        "device": anchor_state.st_dev,
        "inode": anchor_state.st_ino,
        "mode": stat.S_IMODE(anchor_state.st_mode),
        "uid": anchor_state.st_uid,
        "gid": anchor_state.st_gid,
    })
    for part in path.parts[1:-1]:
        cursor = cursor / part
        state = cursor.lstat()
        require(stat.S_ISDIR(state.st_mode) and not cursor.is_symlink(),
                "RUNTIME_ARTIFACT_ANCESTOR_UNSAFE", str(cursor))
        _require_effective_subject_immutable(
            cursor, code="RUNTIME_ARTIFACT_ANCESTOR_NOT_EFFECTIVELY_IMMUTABLE"
        )
        records.append({
            "path": str(cursor),
            "device": state.st_dev,
            "inode": state.st_ino,
            "mode": stat.S_IMODE(state.st_mode),
            "uid": state.st_uid,
            "gid": state.st_gid,
        })
    return sha256_bytes(canonical_bytes(records))


@dataclass(frozen=True)
class RuntimeArtifact:
    artifact_id: str
    path: Path
    sha256: str
    bytes: int
    device: int
    inode: int
    mode: int
    uid: int
    gid: int
    xattrs_sha256: str
    ancestors_sha256: str

    def binding(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "path": str(self.path),
            "sha256": self.sha256,
            "bytes": self.bytes,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "uid": self.uid,
            "gid": self.gid,
            "xattrs_sha256": self.xattrs_sha256,
            "ancestors_sha256": self.ancestors_sha256,
        }


def _safe_runtime_artifact(record: dict[str, Any]) -> RuntimeArtifact:
    value = record.get("path")
    require(isinstance(value, str) and value and "\x00" not in value,
            "INVALID_RUNTIME_ARTIFACT_PATH", value)
    declared = Path(value)
    require(declared.is_absolute(), "RUNTIME_ARTIFACT_NOT_ABSOLUTE", value)
    try:
        resolved = declared.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FacadeError("MISSING_RUNTIME_ARTIFACT", value) from exc
    require(declared == resolved, "RUNTIME_ARTIFACT_SYMLINK_OR_ALIAS", value)
    mode = resolved.lstat().st_mode
    require(stat.S_ISREG(mode) and not resolved.is_symlink(), "UNSAFE_RUNTIME_ARTIFACT", value)
    state_before = resolved.stat()
    _require_effective_subject_immutable(
        resolved, code="RUNTIME_ARTIFACT_WRITABLE_BY_SANDBOX_SUBJECT"
    )
    digest, size = sha256_file(resolved)
    state_after = resolved.stat()
    require(
        (state_before.st_dev, state_before.st_ino, state_before.st_mode, state_before.st_size)
        == (state_after.st_dev, state_after.st_ino, state_after.st_mode, state_after.st_size),
        "RUNTIME_ARTIFACT_CHANGED_DURING_VALIDATION",
        value,
    )
    require(HEX64.fullmatch(str(record.get("sha256"))) is not None,
            "INVALID_RUNTIME_ARTIFACT_HASH", value)
    require((record["sha256"], record.get("bytes")) == (digest, size),
            "RUNTIME_ARTIFACT_HASH_MISMATCH", value)
    return RuntimeArtifact(
        artifact_id=str(record["artifact_id"]),
        path=resolved,
        sha256=digest,
        bytes=size,
        device=state_after.st_dev,
        inode=state_after.st_ino,
        mode=stat.S_IMODE(state_after.st_mode),
        uid=state_after.st_uid,
        gid=state_after.st_gid,
        xattrs_sha256=_xattr_binding(resolved),
        ancestors_sha256=_ancestor_binding(resolved),
    )


@dataclass(frozen=True)
class RuntimeClosure:
    runtime_id: str
    manifest_sha256: str
    python: Path
    environment_allowlist: frozenset[str]
    environment_defaults: dict[str, str]
    timeout_seconds: int
    artifact_count: int
    artifacts: tuple[RuntimeArtifact, ...]
    artifacts_sha256: str
    sandbox_policy: dict[str, Any]
    sandbox_policy_sha256: str
    runtime_threat_model_sha256: str

    def summary(self) -> dict[str, Any]:
        return {
            "runtime_id": self.runtime_id,
            "manifest_sha256": self.manifest_sha256,
            "artifact_count": self.artifact_count,
            "artifacts_sha256": self.artifacts_sha256,
            "sandbox_policy_sha256": self.sandbox_policy_sha256,
            "runtime_threat_model_sha256": self.runtime_threat_model_sha256,
            "timeout_seconds": self.timeout_seconds,
            "state": "VERIFIED",
        }


def validate_runtime_closure(root: Path, runtime_id: str) -> RuntimeClosure:
    require(SAFE_ID.fullmatch(runtime_id or "") is not None, "INVALID_RUNTIME_ID", runtime_id)
    relative = f"runtime/closures/{runtime_id}"
    directory = root / relative
    # Reuse a synthetic child name to validate every directory component without
    # accepting a symlinked closure directory.
    root_resolved = root.resolve(strict=True)
    cursor = root_resolved
    for part in PurePosixPath(relative).parts:
        cursor = cursor / part
        try:
            mode = cursor.lstat().st_mode
        except FileNotFoundError as exc:
            raise FacadeError("MISSING_RUNTIME_CLOSURE", str(cursor)) from exc
        require(stat.S_ISDIR(mode) and not cursor.is_symlink(),
                "UNSAFE_RUNTIME_CLOSURE", str(cursor))
    require(directory.lstat().st_mode & 0o222 == 0,
            "RUNTIME_DIRECTORY_NOT_IMMUTABLE", str(directory))

    manifest, manifest_hash, manifest_size = _safe_control(
        directory / "manifest.json", root=root, code="MISSING_RUNTIME_MANIFEST"
    )
    seal, _, _ = _safe_control(
        directory / "seal.json", root=root, code="MISSING_RUNTIME_SEAL"
    )
    require(set(manifest) == {
        "schema", "runtime_id", "python_artifact_id", "artifact_count",
        "artifacts_sha256", "artifacts", "environment_allowlist",
        "environment_defaults", "timeout_seconds", "declared_runtime_dependencies",
        "sandbox_policy", "sandbox_policy_sha256", "runtime_threat_model_sha256",
    }, "RUNTIME_MANIFEST_FIELDS")
    require(set(seal) == {
        "schema", "runtime_id", "manifest_sha256", "manifest_bytes",
    }, "RUNTIME_SEAL_FIELDS")
    require(manifest.get("schema") == RUNTIME_SCHEMA, "RUNTIME_SCHEMA")
    require(seal.get("schema") == RUNTIME_SEAL_SCHEMA, "RUNTIME_SEAL_SCHEMA")
    require(
        manifest.get("runtime_id") == runtime_id and seal.get("runtime_id") == runtime_id,
        "RUNTIME_ID_MISMATCH",
    )
    require(
        seal.get("manifest_sha256") == manifest_hash
        and seal.get("manifest_bytes") == manifest_size,
        "RUNTIME_SEAL_MISMATCH",
    )

    artifacts = manifest.get("artifacts")
    require(isinstance(artifacts, list) and artifacts, "EMPTY_RUNTIME_CLOSURE")
    require(manifest.get("artifact_count") == len(artifacts), "RUNTIME_ARTIFACT_COUNT")
    require(
        manifest.get("artifacts_sha256") == sha256_bytes(canonical_bytes(artifacts)),
        "RUNTIME_ARTIFACTS_SEAL_MISMATCH",
    )
    artifacts_by_id: dict[str, RuntimeArtifact] = {}
    seen_paths: set[Path] = set()
    for record in artifacts:
        require(isinstance(record, dict), "INVALID_RUNTIME_ARTIFACT")
        require(set(record) == {"artifact_id", "path", "sha256", "bytes"},
                "RUNTIME_ARTIFACT_FIELDS")
        artifact_id = record.get("artifact_id")
        require(SAFE_ID.fullmatch(str(artifact_id)) is not None,
                "INVALID_RUNTIME_ARTIFACT_ID", artifact_id)
        require(artifact_id not in artifacts_by_id, "DUPLICATE_RUNTIME_ARTIFACT_ID", artifact_id)
        artifact = _safe_runtime_artifact(record)
        require(artifact.path not in seen_paths, "DUPLICATE_RUNTIME_ARTIFACT_PATH",
                str(artifact.path))
        artifacts_by_id[artifact_id] = artifact
        seen_paths.add(artifact.path)

    python_id = manifest.get("python_artifact_id")
    require(isinstance(python_id, str), "INVALID_PYTHON_RUNTIME_ID", python_id)
    require(python_id in artifacts_by_id, "MISSING_PYTHON_RUNTIME", python_id)
    dependencies = manifest.get("declared_runtime_dependencies")
    require(
        isinstance(dependencies, list)
        and len(dependencies) == len(set(dependencies))
        and all(isinstance(item, str) and SAFE_ID.fullmatch(item) for item in dependencies),
        "INVALID_RUNTIME_DEPENDENCIES",
    )
    require(set(dependencies) == set(artifacts_by_id), "RUNTIME_DEPENDENCY_CLOSURE_MISMATCH")
    sandbox_policy = manifest.get("sandbox_policy")
    require(sandbox_policy == DEFAULT_SANDBOX_POLICY, "RUNTIME_SANDBOX_POLICY_UNSUPPORTED")
    policy_hash = sandbox_policy_sha256(sandbox_policy)
    require(
        manifest.get("sandbox_policy_sha256") == policy_hash,
        "RUNTIME_SANDBOX_POLICY_HASH_MISMATCH",
    )
    require(
        manifest.get("runtime_threat_model_sha256") == RUNTIME_THREAT_MODEL_SHA256,
        "RUNTIME_THREAT_MODEL_HASH_MISMATCH",
    )
    allowlist = manifest.get("environment_allowlist")
    defaults = manifest.get("environment_defaults")
    require(isinstance(allowlist, list), "INVALID_RUNTIME_ENVIRONMENT_ALLOWLIST")
    require(len(allowlist) == len(set(allowlist)), "DUPLICATE_RUNTIME_ENVIRONMENT_KEY")
    require(all(isinstance(key, str) and ENV_NAME.fullmatch(key) for key in allowlist),
            "INVALID_RUNTIME_ENVIRONMENT_KEY")
    require(isinstance(defaults, dict), "INVALID_RUNTIME_ENVIRONMENT_DEFAULTS")
    require(all(isinstance(key, str) and ENV_NAME.fullmatch(key) for key in defaults),
            "INVALID_RUNTIME_DEFAULT_KEY")
    require(all(isinstance(value, str) and "\x00" not in value for value in defaults.values()),
            "INVALID_RUNTIME_DEFAULT_VALUE")
    require(not any(SECRET_VALUE.search(value) for value in defaults.values()),
            "SECRET_VALUE_IN_RUNTIME_CLOSURE")
    timeout = manifest.get("timeout_seconds")
    require(isinstance(timeout, int) and 1 <= timeout <= 86400, "INVALID_RUNTIME_TIMEOUT")
    return RuntimeClosure(
        runtime_id=runtime_id,
        manifest_sha256=manifest_hash,
        python=artifacts_by_id[python_id].path,
        environment_allowlist=frozenset(allowlist),
        environment_defaults=dict(defaults),
        timeout_seconds=timeout,
        artifact_count=len(artifacts),
        artifacts=tuple(artifacts_by_id[key] for key in sorted(artifacts_by_id)),
        artifacts_sha256=str(manifest["artifacts_sha256"]),
        sandbox_policy=dict(sandbox_policy),
        sandbox_policy_sha256=policy_hash,
        runtime_threat_model_sha256=RUNTIME_THREAT_MODEL_SHA256,
    )


@dataclass(frozen=True)
class BoundInput:
    path: Path
    sha256: str
    bytes: int
    device: int
    inode: int
    mode: int


@dataclass(frozen=True)
class PreparedStep:
    stage: str
    variant_id: str
    template: str
    argv: tuple[str, ...]
    argv_sha256: str
    inputs: tuple[BoundInput, ...]
    output_paths: tuple[Path, ...]
    expected_paths: tuple[Path, ...]
    expected_output_bindings: dict[str, tuple[str, int]]


@dataclass(frozen=True)
class ExternalContract:
    root: Path
    publication: SnapshotPublication
    runtime: RuntimeClosure
    run_config_id: str
    run_config_sha256: str
    environment: dict[str, str]
    steps: tuple[PreparedStep, ...]

    def summary(self) -> dict[str, Any]:
        return {
            "publication": self.publication.summary(),
            "runtime": self.runtime.summary(),
            "run_config_id": self.run_config_id,
            "run_config_sha256": self.run_config_sha256,
            "step_count": len(self.steps),
            "state": "VERIFIED",
        }


def _literal(value: Any, *, name: str) -> str:
    require(isinstance(value, (str, int, float, bool)) and not isinstance(value, type(None)),
            "INVALID_LITERAL_BINDING", name)
    if isinstance(value, float):
        require(math.isfinite(value), "INVALID_LITERAL_BINDING", name)
    rendered = str(value).lower() if isinstance(value, bool) else str(value)
    require("\x00" not in rendered and not SECRET_VALUE.search(rendered),
            "UNSAFE_LITERAL_BINDING", name)
    if "://" not in rendered:
        require(not Path(rendered).is_absolute() and ".." not in PurePosixPath(rendered).parts,
                "LITERAL_PATH_FORBIDDEN", name)
    return rendered


def _output_path(target: Path, relative: Any, *, name: str) -> Path:
    require(_safe_relative(relative), "INVALID_OUTPUT_BINDING", name)
    first = PurePosixPath(relative).parts[0]
    require(first not in RESERVED_OUTPUTS, "RESERVED_OUTPUT_BINDING", relative)
    path = target / relative
    try:
        path.relative_to(target)
    except ValueError as exc:
        raise FacadeError("OUTPUT_BINDING_ESCAPE", relative) from exc
    return path


def _binding_value(
    root: Path,
    target: Path,
    name: str,
    binding: Any,
) -> tuple[str, Path | None, BoundInput | None]:
    require(isinstance(binding, dict), "INVALID_RUN_BINDING", name)
    kind = binding.get("kind")
    if kind == "literal":
        require(set(binding) == {"kind", "value"}, "INVALID_LITERAL_BINDING_FIELDS", name)
        return _literal(binding["value"], name=name), None, None
    if kind == "input":
        require(set(binding) == {"kind", "path", "sha256", "bytes"},
                "INVALID_INPUT_BINDING_FIELDS", name)
        path = _safe_owned_file(
            root,
            binding["path"],
            code="UNSAFE_RUNTIME_INPUT",
            immutable=True,
            effective_subject_immutable=True,
        )
        digest, size = sha256_file(path)
        require(HEX64.fullmatch(str(binding.get("sha256"))) is not None,
                "INVALID_INPUT_BINDING_HASH", name)
        require((binding["sha256"], binding.get("bytes")) == (digest, size),
                "RUNTIME_INPUT_HASH_MISMATCH", name)
        state = path.stat()
        return str(path), None, BoundInput(
            path=path,
            sha256=digest,
            bytes=size,
            device=state.st_dev,
            inode=state.st_ino,
            mode=stat.S_IMODE(state.st_mode),
        )
    if kind == "output":
        require(set(binding) == {"kind", "path"}, "INVALID_OUTPUT_BINDING_FIELDS", name)
        path = _output_path(target, binding["path"], name=name)
        return str(path), path, None
    raise FacadeError("UNKNOWN_RUN_BINDING_KIND", {"name": name, "kind": kind})


def _replace_token(token: str, replacements: Mapping[str, str]) -> str:
    names = PLACEHOLDER.findall(token)
    for name in names:
        require(name in replacements, "UNBOUND_ARGV_PLACEHOLDER", name)
    rendered = PLACEHOLDER.sub(lambda match: replacements[match.group(1)], token)
    require("\x00" not in rendered and "{" not in rendered and "}" not in rendered,
            "INVALID_RENDERED_ARGV")
    return rendered


def validate_external_contract(
    bundle: Bundle,
    selection: dict[str, Any],
    target: Path,
    *,
    publication_id: str,
    runtime_id: str,
    run_config_id: str,
    source_environment: Mapping[str, str] | None = None,
    require_environment: bool = True,
) -> ExternalContract:
    publication = validate_snapshot_publication(bundle, publication_id)
    runtime = validate_runtime_closure(bundle.root, runtime_id)
    require(SAFE_ID.fullmatch(run_config_id or "") is not None,
            "INVALID_RUN_CONFIG_ID", run_config_id)
    config_path = bundle.root / "configs/external-runs" / f"{run_config_id}.json"
    config, config_hash, _ = _safe_control(
        config_path, root=bundle.root, code="MISSING_EXTERNAL_RUN_CONFIG"
    )
    require(config.get("schema") == RUN_CONFIG_SCHEMA, "RUN_CONFIG_SCHEMA")
    require(set(config) == {
        "schema", "run_config_id", "profile_id", "publication_id", "runtime_id",
        "environment_sources", "steps", "control_hashes",
        "publication_manifest_sha256", "runtime_manifest_sha256",
        "sandbox_policy_sha256", "runtime_threat_model_sha256", "output_dir",
        "dispatch_path", "transport",
    }, "RUN_CONFIG_FIELDS")
    require(config.get("run_config_id") == run_config_id, "RUN_CONFIG_ID_MISMATCH")
    require(config.get("profile_id") == selection["profile_id"], "RUN_CONFIG_PROFILE_MISMATCH")
    require(config.get("publication_id") == publication_id, "RUN_CONFIG_PUBLICATION_MISMATCH")
    require(config.get("runtime_id") == runtime_id, "RUN_CONFIG_RUNTIME_MISMATCH")
    require(config.get("control_hashes") == selection["control_hashes"],
            "RUN_CONFIG_CONTROL_HASH_MISMATCH")
    require(config.get("publication_manifest_sha256") == publication.manifest_sha256,
            "RUN_CONFIG_PUBLICATION_HASH_MISMATCH")
    require(config.get("runtime_manifest_sha256") == runtime.manifest_sha256,
            "RUN_CONFIG_RUNTIME_HASH_MISMATCH")
    require(config.get("sandbox_policy_sha256") == runtime.sandbox_policy_sha256,
            "RUN_CONFIG_SANDBOX_POLICY_MISMATCH")
    require(
        config.get("runtime_threat_model_sha256")
        == runtime.runtime_threat_model_sha256,
        "RUN_CONFIG_THREAT_MODEL_HASH_MISMATCH",
    )
    try:
        target_relative = target.relative_to(bundle.root).as_posix()
    except ValueError as exc:
        raise FacadeError("RUN_CONFIG_OUTPUT_ESCAPE", str(target)) from exc
    require(config.get("output_dir") == target_relative, "RUN_CONFIG_OUTPUT_DIR_MISMATCH")
    require(config.get("dispatch_path") == "production_external_contract",
            "RUN_CONFIG_DISPATCH_PATH_MISMATCH")

    environment_sources = config.get("environment_sources")
    require(isinstance(environment_sources, dict), "INVALID_RUN_ENVIRONMENT_SOURCES")
    require(all(
        isinstance(target_key, str)
        and isinstance(source_key, str)
        and ENV_NAME.fullmatch(target_key)
        and ENV_NAME.fullmatch(source_key)
        and target_key == source_key
        for target_key, source_key in environment_sources.items()
    ), "INVALID_RUN_ENVIRONMENT_SOURCE")
    require(set(environment_sources) <= runtime.environment_allowlist,
            "RUN_ENVIRONMENT_NOT_IN_RUNTIME_CLOSURE")

    if require_environment:
        try:
            require_strict_runtime_proof_environment((bundle.root,))
        except SandboxUnavailable as exc:
            raise FacadeError(exc.code, exc.detail) from exc

    selected_adapters = {row["stage"]: row for row in selection["adapters"]}
    expected = [
        (stage, selection["selected_stages"][stage])
        for stage in STAGES
        if stage in selected_adapters
        and selected_adapters[stage]["execution_kind"] == "subprocess"
    ]
    require(expected, "NO_SUBPROCESS_STEPS")
    steps = config.get("steps")
    require(isinstance(steps, list) and all(isinstance(row, dict) for row in steps),
            "INVALID_RUN_STEPS")
    require(
        [(row.get("stage"), row.get("variant_id")) for row in steps] == expected,
        "RUN_STEP_CLOSURE_MISMATCH",
        {"expected": expected},
    )
    required_environment = {
        key
        for stage, _ in expected
        for key in selected_adapters[stage]["environment_keys"]
    }
    require(set(environment_sources) == required_environment,
            "RUN_ENVIRONMENT_CLOSURE_MISMATCH",
            {"expected": sorted(required_environment)},)

    source_environment = source_environment or {}
    environment = dict(runtime.environment_defaults)
    for target_key, source_key in environment_sources.items():
        if require_environment:
            require(source_key in source_environment and source_environment[source_key] != "",
                    "MISSING_RUNTIME_ENVIRONMENT", source_key)
            require(isinstance(source_environment[source_key], str)
                    and "\x00" not in source_environment[source_key],
                    "INVALID_RUNTIME_ENVIRONMENT_VALUE", source_key)
            environment[target_key] = source_environment[source_key]
        else:
            environment[target_key] = f"<environment:{source_key}>"

    prepared: list[PreparedStep] = []
    all_output_paths: set[Path] = set()
    for row in steps:
        require(isinstance(row, dict) and set(row) == {
            "stage", "variant_id", "template", "bindings", "argv_sha256",
            "expected_outputs",
        },
                "INVALID_RUN_STEP_FIELDS", row.get("stage") if isinstance(row, dict) else None)
        stage = row["stage"]
        adapter = selected_adapters[stage]
        template_name = row["template"]
        require(SAFE_ID.fullmatch(str(template_name)) is not None,
                "INVALID_ARGV_TEMPLATE", template_name)
        templates = adapter["argv_template"]
        require(template_name in templates, "MISSING_ARGV_TEMPLATE",
                {"variant_id": row["variant_id"], "template": template_name})
        template = templates[template_name]
        require(isinstance(template, list) and len(template) >= 2,
                "INVALID_ARGV_TEMPLATE", row["variant_id"])
        require(template[:2] == ["{python}", "{entrypoint}"],
                "UNSAFE_ARGV_PREFIX", row["variant_id"])
        entrypoint = adapter["entrypoint_destination"]
        require(isinstance(entrypoint, str), "MISSING_ADAPTER_ENTRYPOINT", row["variant_id"])
        entrypoint_path = _safe_owned_file(
            bundle.root,
            entrypoint,
            code="UNSAFE_ADAPTER_ENTRYPOINT",
            immutable=True,
            effective_subject_immutable=True,
        )
        lineage_ids = {binding["lineage_id"] for binding in adapter["source_bindings"]}
        require(lineage_ids <= set(publication.records_by_lineage),
                "ENTRYPOINT_PUBLICATION_CLOSURE_MISMATCH", row["variant_id"])

        placeholder_names = {name for token in template for name in PLACEHOLDER.findall(token)}
        environment_names = {name[4:] for name in placeholder_names if name.startswith("env:")}
        require(environment_names <= set(environment_sources),
                "ARGV_ENVIRONMENT_NOT_DECLARED", sorted(environment_names))
        require(not environment_names, "ENVIRONMENT_VALUE_IN_ARGV_FORBIDDEN",
                sorted(environment_names))
        binding_names = placeholder_names - {"python", "entrypoint", "output_dir"} - {
            f"env:{name}" for name in environment_names
        }
        bindings = row["bindings"]
        require(isinstance(bindings, dict) and set(bindings) == binding_names,
                "RUN_BINDING_CLOSURE_MISMATCH",
                {"stage": stage, "expected": sorted(binding_names)})
        replacements = {
            "python": str(runtime.python),
            "entrypoint": str(entrypoint_path),
            "output_dir": str(target),
            **{f"env:{name}": environment[name] for name in environment_names},
        }
        output_paths: list[Path] = []
        input_files: list[BoundInput] = []
        for name, binding in bindings.items():
            rendered, output, input_file = _binding_value(bundle.root, target, name, binding)
            replacements[name] = rendered
            if output is not None:
                require(output not in all_output_paths, "DUPLICATE_RUN_OUTPUT", str(output))
                output_paths.append(output)
                all_output_paths.add(output)
            if input_file is not None:
                input_files.append(input_file)
        argv = tuple(_replace_token(token, replacements) for token in template)
        argv_hash = sha256_bytes(canonical_bytes(list(argv)))
        require(HEX64.fullmatch(str(row.get("argv_sha256"))) is not None,
                "INVALID_RUN_ARGV_HASH", stage)
        require(row["argv_sha256"] == argv_hash, "RUN_ARGV_HASH_MISMATCH", stage)

        expected_paths: list[Path] = []
        for expected_output in adapter["expected_outputs"]:
            prefix = "{output_dir}/"
            require(expected_output.startswith(prefix), "INVALID_EXPECTED_OUTPUT_TEMPLATE", expected_output)
            expected_path = _output_path(
                target, expected_output[len(prefix):], name=f"expected:{stage}"
            )
            require(expected_path in output_paths, "EXPECTED_OUTPUT_NOT_BOUND", expected_output)
            expected_paths.append(expected_path)
        expected_records = row["expected_outputs"]
        require(
            isinstance(expected_records, list)
            and all(isinstance(record, dict) and set(record) == {"path", "sha256", "bytes"}
                    for record in expected_records),
            "INVALID_EXPECTED_OUTPUT_BINDINGS",
            stage,
        )
        expected_bindings: dict[str, tuple[str, int]] = {}
        for record in expected_records:
            relative = record["path"]
            require(_safe_relative(relative), "INVALID_EXPECTED_OUTPUT_PATH", relative)
            require(relative not in expected_bindings, "DUPLICATE_EXPECTED_OUTPUT_PATH", relative)
            require(HEX64.fullmatch(str(record["sha256"])) is not None,
                    "INVALID_EXPECTED_OUTPUT_HASH", relative)
            require(isinstance(record["bytes"], int) and record["bytes"] >= 0,
                    "INVALID_EXPECTED_OUTPUT_SIZE", relative)
            expected_bindings[relative] = (record["sha256"], record["bytes"])
        bound_relatives = {path.relative_to(target).as_posix() for path in output_paths}
        require(set(expected_bindings) == bound_relatives,
                "EXPECTED_OUTPUT_CLOSURE_MISMATCH", stage)
        prepared.append(PreparedStep(
            stage=stage,
            variant_id=row["variant_id"],
            template=template_name,
            argv=argv,
            argv_sha256=argv_hash,
            inputs=tuple(input_files),
            output_paths=tuple(output_paths),
            expected_paths=tuple(expected_paths),
            expected_output_bindings=expected_bindings,
        ))

    transport = config.get("transport")
    require(isinstance(transport, dict) and isinstance(transport.get("mode"), str),
            "INVALID_TRANSPORT_POLICY")
    exchanges_external = any(
        step.stage in {"preprocessing", "inference"} for step in prepared
    )
    if exchanges_external:
        require(set(transport) == {"mode", "boundary", "fixture"},
                "INVALID_HERMETIC_TRANSPORT_POLICY")
        require(
            transport.get("mode") == "hermetic_replay"
            and transport.get("boundary") == "external_exchange_only",
            "HERMETIC_TRANSPORT_REQUIRED",
        )
        fixture = transport.get("fixture")
        require(isinstance(fixture, dict) and set(fixture) == {"path", "sha256", "bytes"},
                "INVALID_HERMETIC_TRANSPORT_FIXTURE")
        fixture_path = _safe_owned_file(
            bundle.root,
            fixture["path"],
            code="UNSAFE_TRANSPORT_FIXTURE",
            immutable=True,
            effective_subject_immutable=True,
        )
        fixture_digest, fixture_size = sha256_file(fixture_path)
        require((fixture.get("sha256"), fixture.get("bytes")) == (fixture_digest, fixture_size),
                "TRANSPORT_FIXTURE_HASH_MISMATCH")
        require(any(fixture_path == item.path for step in prepared for item in step.inputs),
                "TRANSPORT_FIXTURE_NOT_DISPATCH_BOUND")
    else:
        require(
            transport == {"mode": "not_applicable", "boundary": "no_external_exchange"},
            "INVALID_NON_NETWORK_TRANSPORT_POLICY",
        )

    prerequisites = set(selection["external_prerequisites"])
    require(prerequisites <= KNOWN_PREREQUISITES,
            "UNKNOWN_EXTERNAL_PREREQUISITE", sorted(prerequisites - KNOWN_PREREQUISITES))
    return ExternalContract(
        root=bundle.root,
        publication=publication,
        runtime=runtime,
        run_config_id=run_config_id,
        run_config_sha256=config_hash,
        environment=environment,
        steps=tuple(prepared),
    )


def _open_noreplace(path: Path) -> Any:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    return os.fdopen(descriptor, "wb")


def _write_noreplace(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_bytes(value)
    with _open_noreplace(path) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o444)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class OutputClaim:
    path: Path
    descriptor: int
    device: int
    inode: int
    mode: int
    ancestors: tuple[tuple[int, int], ...]

    def sandbox_binding(self) -> DeclaredOutputInode:
        return DeclaredOutputInode(
            path=self.path, device=self.device, inode=self.inode
        )


def _path_ancestor_identities(target: Path, path: Path) -> tuple[tuple[int, int], ...]:
    relative = path.relative_to(target)
    cursor = target
    identities: list[tuple[int, int]] = []
    target_state = cursor.lstat()
    require(stat.S_ISDIR(target_state.st_mode) and not cursor.is_symlink(),
            "UNSAFE_EXECUTION_OUTPUT_ROOT", str(cursor))
    identities.append((target_state.st_dev, target_state.st_ino))
    for part in relative.parts:
        cursor = cursor / part
        state = cursor.lstat()
        require(stat.S_ISDIR(state.st_mode) and not cursor.is_symlink(),
                "UNSAFE_EXECUTION_OUTPUT_PARENT", str(cursor))
        identities.append((state.st_dev, state.st_ino))
    return tuple(identities)


def _claim_output(path: Path, *, target: Path) -> OutputClaim:
    _ensure_output_parent(target, path)
    ancestors = _path_ancestor_identities(target, path.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise FacadeError("EXECUTION_OUTPUT_COLLISION", str(path)) from exc
    try:
        os.fsync(descriptor)
        state = os.fstat(descriptor)
    except Exception:
        os.close(descriptor)
        raise
    require(stat.S_ISREG(state.st_mode), "UNSAFE_EXECUTION_OUTPUT", str(path))
    return OutputClaim(
        path=path,
        descriptor=descriptor,
        device=state.st_dev,
        inode=state.st_ino,
        mode=stat.S_IMODE(state.st_mode),
        ancestors=ancestors,
    )


def _verify_bound_input(value: BoundInput) -> None:
    state = value.path.lstat()
    require(
        stat.S_ISREG(state.st_mode)
        and not value.path.is_symlink()
        and (state.st_dev, state.st_ino, stat.S_IMODE(state.st_mode))
        == (value.device, value.inode, value.mode),
        "RUNTIME_INPUT_IDENTITY_DRIFT",
        str(value.path),
    )
    digest, size = sha256_file(value.path)
    require((digest, size) == (value.sha256, value.bytes),
            "RUNTIME_INPUT_HASH_DRIFT", str(value.path))


def _verify_runtime_artifacts(runtime: RuntimeClosure) -> None:
    for expected in runtime.artifacts:
        actual = _safe_runtime_artifact({
            "artifact_id": expected.artifact_id,
            "path": str(expected.path),
            "sha256": expected.sha256,
            "bytes": expected.bytes,
        })
        require(actual.binding() == expected.binding(),
                "RUNTIME_ARTIFACT_IDENTITY_DRIFT", expected.artifact_id)


def _contains_secret(path: Path, values: tuple[bytes, ...]) -> bool:
    overlap = max([len(value) for value in values] + [128])
    tail = b""
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            payload = tail + block
            if any(value in payload for value in values):
                return True
            if SECRET_VALUE.search(payload.decode("utf-8", errors="ignore")):
                return True
            tail = payload[-overlap:]
    return False


def _seal_regular(path: Path) -> None:
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        state = os.fstat(descriptor)
        require(stat.S_ISREG(state.st_mode), "UNSAFE_EXECUTION_OUTPUT", str(path))
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _verified_output(
    path: Path, *, target: Path, claim: OutputClaim | None = None, seal: bool = False
) -> dict[str, Any]:
    try:
        relative = path.relative_to(target).as_posix()
    except ValueError as exc:
        raise FacadeError("EXECUTION_OUTPUT_ESCAPE", str(path)) from exc
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise FacadeError("MISSING_EXECUTION_OUTPUT", relative) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(), "UNSAFE_EXECUTION_OUTPUT", relative)
    if claim is not None:
        state = path.stat()
        descriptor_state = os.fstat(claim.descriptor)
        require(
            (state.st_dev, state.st_ino) == (claim.device, claim.inode)
            and (descriptor_state.st_dev, descriptor_state.st_ino)
            == (claim.device, claim.inode)
            and _path_ancestor_identities(target, path.parent) == claim.ancestors,
            "EXECUTION_OUTPUT_IDENTITY_DRIFT",
            relative,
        )
    if seal:
        _seal_regular(path)
    digest, size = sha256_file(path)
    final_mode = stat.S_IMODE(path.lstat().st_mode)
    if seal:
        require(final_mode & 0o222 == 0, "EXECUTION_OUTPUT_NOT_READONLY", relative)
        _require_effective_subject_immutable(
            path, code="EXECUTION_OUTPUT_NOT_EFFECTIVELY_IMMUTABLE"
        )
        cursor = path.parent
        while True:
            _require_effective_subject_immutable(
                cursor, code="EXECUTION_OUTPUT_ANCESTOR_NOT_EFFECTIVELY_IMMUTABLE"
            )
            if cursor == target:
                break
            cursor = cursor.parent
    return {"path": relative, "sha256": digest, "bytes": size, "mode": final_mode}


def _ensure_output_parent(target: Path, path: Path) -> None:
    relative = path.parent.relative_to(target)
    cursor = target
    for part in relative.parts:
        cursor = cursor / part
        if os.path.lexists(cursor):
            mode = cursor.lstat().st_mode
            require(stat.S_ISDIR(mode) and not cursor.is_symlink(),
                    "UNSAFE_EXECUTION_OUTPUT_PARENT", str(cursor))
        else:
            cursor.mkdir(mode=0o700)


def execute_external_contract(
    selection: dict[str, Any],
    target: Path,
    contract: ExternalContract,
) -> dict[str, Any]:
    protected_paths = [
        contract.root,
        *(artifact.path for artifact in contract.runtime.artifacts),
        *(bound.path for step in contract.steps for bound in step.inputs),
    ]
    try:
        require_strict_runtime_proof_environment(protected_paths)
    except SandboxUnavailable as exc:
        raise FacadeError(exc.code, exc.detail) from exc
    require(os.geteuid() != 0, "RUNTIME_HOST_ROOT_FORBIDDEN")
    try:
        backend = prepare_backend(contract.runtime.sandbox_policy)
    except SandboxUnavailable as exc:
        raise FacadeError(
            "RUNTIME_SANDBOX_UNAVAILABLE", {"sandbox_reason_code": exc.code}
        ) from exc
    require(
        backend.policy_sha256 == contract.runtime.sandbox_policy_sha256,
        "RUNTIME_SANDBOX_POLICY_MISMATCH",
    )
    expected_subject = current_subject_identity()
    require(
        all(
            expected_subject[key] == "0000000000000000"
            for key in (
                "capability_inheritable",
                "capability_permitted",
                "capability_effective",
                "capability_ambient",
            )
        ),
        "RUNTIME_SANDBOX_SUBJECT_CAPABILITIES_NONEMPTY",
    )
    expected_subject_hash = subject_identity_sha256(expected_subject)
    _verify_runtime_artifacts(contract.runtime)
    steps_directory = target / "steps"
    steps_directory.mkdir(mode=0o700)
    home = target / "home"
    temporary = target / "tmp"
    home.mkdir(mode=0o700)
    temporary.mkdir(mode=0o700)
    environment = {
        **contract.environment,
        "HOME": str(home),
        "TMPDIR": str(temporary),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
    }
    require(
        not ({"PYTHONPATH", "PYTHONHOME", "LD_PRELOAD", "LD_LIBRARY_PATH"} & set(environment)),
        "RUNTIME_INJECTION_ENVIRONMENT",
    )
    secret_values = tuple(
        value.encode("utf-8")
        for key, value in contract.environment.items()
        if re.search(r"(?:KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", key)
        and isinstance(value, str)
        and len(value) >= 4
    )
    step_results: list[dict[str, Any]] = []
    failure_code: str | None = None
    failure_detail: Any = None
    for index, step in enumerate(contract.steps, start=1):
        _verify_runtime_artifacts(contract.runtime)
        for input_file in step.inputs:
            _verify_bound_input(input_file)
        output_claims = tuple(_claim_output(output, target=target) for output in step.output_paths)
        stem = f"{index:02d}-{step.stage}-{step.variant_id}"
        stdout_path = steps_directory / f"{stem}.stdout.log"
        stderr_path = steps_directory / f"{stem}.stderr.log"
        canary_path = target / f".sandbox-denial-canary-{index:02d}"
        canary_claim = _claim_output(canary_path, target=target)
        canary_flags = os.O_WRONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            canary_flags |= os.O_NOFOLLOW
        canary_descriptor = os.open(canary_path, canary_flags)
        try:
            written = os.write(canary_descriptor, b"sandbox-denial-canary\n")
            require(written == len(b"sandbox-denial-canary\n"),
                    "SANDBOX_CANARY_PARENT_WRITE_FAILED")
            os.fsync(canary_descriptor)
            canary_state = os.fstat(canary_descriptor)
            require((canary_state.st_dev, canary_state.st_ino)
                    == (canary_claim.device, canary_claim.inode),
                    "SANDBOX_CANARY_IDENTITY_DRIFT")
            require(
                _subject_can_write(
                    canary_state.st_mode, canary_state.st_uid, canary_state.st_gid
                ),
                "SANDBOX_CANARY_PARENT_NOT_WRITABLE",
            )
        finally:
            os.close(canary_descriptor)
        canary_digest, canary_size = sha256_file(canary_path)
        returncode: int | None = None
        state = "PASS"
        sandbox_evidence: dict[str, Any] | None = None
        timed_out = False
        evidence_read, evidence_write = os.pipe2(os.O_CLOEXEC)
        try:
            with _open_noreplace(stdout_path) as stdout, _open_noreplace(stderr_path) as stderr:
                process = subprocess.Popen(
                    list(step.argv),
                    cwd=target,
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    close_fds=True,
                    pass_fds=(evidence_write,),
                    start_new_session=True,
                    preexec_fn=make_preexec(
                        backend,
                        tuple(claim.sandbox_binding() for claim in output_claims),
                        canary_path,
                        evidence_write,
                        expected_subject_hash,
                    ),
                )
                os.close(evidence_write)
                evidence_write = -1
                try:
                    returncode = process.wait(timeout=contract.runtime.timeout_seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            evidence_chunks: list[bytes] = []
            while True:
                block = os.read(evidence_read, 65536)
                if not block:
                    break
                evidence_chunks.append(block)
            try:
                sandbox_evidence = parse_evidence(
                    b"".join(evidence_chunks), backend, expected_subject_hash
                )
            except SandboxUnavailable as exc:
                state = "FAILED"
                failure_code = "RUNTIME_SANDBOX_UNOBSERVABLE"
                failure_detail = {"sandbox_reason_code": exc.code}
            if timed_out:
                state = "FAILED"
                failure_code = "EXTERNAL_STEP_TIMEOUT"
                failure_detail = {"stage": step.stage, "variant_id": step.variant_id}
            if returncode != 0:
                state = "FAILED"
                if failure_code is None:
                    failure_code = "EXTERNAL_STEP_NONZERO"
                    failure_detail = {"stage": step.stage, "variant_id": step.variant_id,
                                      "returncode": returncode}
        except subprocess.SubprocessError:
            state = "FAILED"
            failure_code = "RUNTIME_SANDBOX_ENFORCEMENT_FAILED"
            failure_detail = {"stage": step.stage, "variant_id": step.variant_id}
        except OSError as exc:
            state = "FAILED"
            failure_code = "EXTERNAL_STEP_LAUNCH_FAILED"
            failure_detail = {"stage": step.stage, "variant_id": step.variant_id,
                              "error": type(exc).__name__}
        finally:
            if evidence_write >= 0:
                os.close(evidence_write)
            os.close(evidence_read)

        current_canary = canary_path.lstat()
        require(
            stat.S_ISREG(current_canary.st_mode)
            and (current_canary.st_dev, current_canary.st_ino)
            == (canary_claim.device, canary_claim.inode)
            and sha256_file(canary_path) == (canary_digest, canary_size),
            "SANDBOX_CANARY_MUTATED",
        )
        os.close(canary_claim.descriptor)
        canary_path.unlink()

        outputs: list[dict[str, Any]] = []
        scan_paths = [*step.output_paths, stdout_path, stderr_path]
        secret_matches = [path for path in scan_paths if _contains_secret(path, secret_values)]
        if secret_matches:
            state = "FAILED"
            failure_code = "SECRET_VALUE_IN_EXECUTION_OUTPUT"
            failure_detail = {"matched_file_count": len(secret_matches)}
            for path in secret_matches:
                path.unlink()
        if not secret_matches:
            try:
                outputs = [
                    _verified_output(claim.path, target=target, claim=claim, seal=True)
                    for claim in output_claims
                ]
                observed = {
                    record["path"]: (record["sha256"], record["bytes"]) for record in outputs
                }
                require(
                    observed == step.expected_output_bindings,
                    "EXECUTION_GOLDEN_MISMATCH",
                    {"stage": step.stage, "variant_id": step.variant_id},
                )
                for path in step.expected_paths:
                    require(path in step.output_paths, "MISSING_EXPECTED_EXECUTION_OUTPUT", str(path))
                _verify_runtime_artifacts(contract.runtime)
                for input_file in step.inputs:
                    _verify_bound_input(input_file)
            except FacadeError as exc:
                state = "FAILED"
                failure_code = exc.code
                failure_detail = exc.detail
        for claim in output_claims:
            try:
                os.close(claim.descriptor)
            except OSError:
                pass
        stdout_record = (
            _verified_output(stdout_path, target=target, seal=True)
            if stdout_path.exists() else {"path": stdout_path.relative_to(target).as_posix(),
                                         "state": "REDACTED_REMOVED"}
        )
        stderr_record = (
            _verified_output(stderr_path, target=target, seal=True)
            if stderr_path.exists() else {"path": stderr_path.relative_to(target).as_posix(),
                                         "state": "REDACTED_REMOVED"}
        )
        step_results.append({
            "stage": step.stage,
            "variant_id": step.variant_id,
            "template": step.template,
            "argv_sha256": step.argv_sha256,
            "state": state,
            "returncode": returncode,
            "stdout": stdout_record,
            "stderr": stderr_record,
            "outputs": outputs,
            "sandbox": sandbox_evidence,
            "subprocess_output_secret_scan_performed": True,
            "secret_match_count": len(secret_matches),
        })
        if state != "PASS":
            break

    result = {
        "schema": "experiments7-external-result/v1",
        "state": "PASS" if failure_code is None else "FAILED",
        "profile_id": selection["profile_id"],
        "selection_sha256": selection["selection_sha256"],
        "contract": contract.summary(),
        "steps": step_results,
        "environment_keys": sorted(contract.environment),
        "secret_values_in_control_evidence": False,
        "subprocess_output_secret_scan_performed": True,
        "bounded_runnable_claim": "local hermetic exact-dispatch only; no live inference claim",
    }
    if failure_code is not None:
        result["reason_code"] = failure_code
        result["detail"] = failure_detail
    _write_noreplace(target / "external-result.json", result)
    if failure_code is not None:
        raise FacadeError(failure_code, failure_detail)
    return result
