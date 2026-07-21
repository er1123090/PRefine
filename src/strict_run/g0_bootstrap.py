"""Production-shaped V6 G0/CP0 bootstrap with no pre-envelope protected read.

The controller in this module never opens a protected experiment root or the
paper.  It freezes the protected child and its provider into the explicit run,
then receives canonical records over stdout from that child.  This keeps the
first protected enumeration/hash inside the one-way Landlock envelope while
still using the existing descriptor-bound StagePublisher for every run file.
"""
from __future__ import annotations

import errno
import ctypes
import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
import platform
import stat
import subprocess
import sys
import time
from typing import Iterable, Mapping

from provenance.strict_v6 import validate_source_pre_records

from .canonical import (
    StrictRunError,
    canonical_json,
    fail,
    sha256_bytes,
    validate_absolute_path_text,
    validate_sealed_run_id,
)
from .checkpoint import (
    CP0_DENIED_METADATA_SYSCALLS_X86_64,
    CP0_LANDLOCK_READ_EXEC_RIGHTS,
    CP0_TRUSTED_PROVIDER_SHA256,
    CheckpointResult,
    publish_checkpoint,
)
from .external import (
    ExternalOutputEvidence,
    ExternalOutputReservation,
    ExternalTranscriptRecord,
    ExternalTranscriptRegistry,
)
from .filesystem import (
    DIR_FLAGS,
    READ_FLAGS,
    RunDescriptorBinding,
    mount_identity,
    open_absolute_directory,
    open_bound_run_handle,
    open_parent_directory,
    stable_identity,
)
from .publication import Publication, StagePublisher
from .reservation import (
    ReservationInputs,
    accept_reservation,
    publish_owner_binding,
    reserve_strict_run,
)
from .writer_policy import WriterIdentity, default_writer_policy


G0_CONFIG_SCHEMA = "experiments7-g0-protected-config/v6"
G0_RESULT_SCHEMA = "experiments7-g0-protected-result/v6"
G0_BUNDLE_SCHEMA = "experiments7-g0-bundle-lock/v6"
G0_RUNTIME_SCHEMA = "experiments7-g0-runtime-identity/v6"
G0_ENVIRONMENT_SCHEMA = "experiments7-g0-environment-allowlist/v6"
G0_ARGUMENT_SCHEMA = "experiments7-g0-argv/v6"
G0_ENVELOPE_SCHEMA = "experiments7-cp0-envelope-evidence/v6"
G0_PRODUCER_BINDING_SCHEMA = "experiments7-cp0-producer-binding/v6"
G0_PROTECTED_READ_ATTESTATION_SCHEMA = (
    "experiments7-cp0-protected-read-attestation/v6"
)
G0_INVENTORY_SCHEMA = "experiments7-cp0-frozen-inventory/v6"
G0_CHILD_EXECUTION_SCHEMA = "experiments7-g0-child-execution/v6"
G0_DESCRIPTOR_TRANSPORT_SCHEMA = "experiments7-g0-descriptor-transport/v6"

_SYS_MEMFD_CREATE_X86_64 = 319
_MFD_CLOEXEC = 0x0001
_MFD_ALLOW_SEALING = 0x0002
_F_ADD_SEALS = 1033
_F_GET_SEALS = 1034
_F_SEAL_SEAL = 0x0001
_F_SEAL_SHRINK = 0x0002
_F_SEAL_GROW = 0x0004
_F_SEAL_WRITE = 0x0008
_REQUIRED_MEMFD_SEALS = (
    _F_SEAL_SEAL | _F_SEAL_SHRINK | _F_SEAL_GROW | _F_SEAL_WRITE
)


@dataclass(frozen=True)
class G0BootstrapResult:
    sealed_run_id: str
    strict_parent: str
    run_root: str
    checkpoint: CheckpointResult
    stage_publications: tuple[Publication, ...]
    external_paths: Mapping[str, str]


@dataclass(frozen=True)
class _ProtectedChildResult:
    result: dict[str, object]
    stdout: bytes
    stderr: bytes
    argv: tuple[str, ...]
    descriptor_transport: dict[str, object]
    environment: dict[str, str]
    launched_monotonic_ns: int
    completed_monotonic_ns: int
    exit_status: int


@dataclass(frozen=True)
class _MountInfoEntry:
    mount_id: int
    parent_mount_id: int
    device: tuple[int, int]
    mount_root: str
    mount_point: str


@dataclass(frozen=True)
class _MountBoundarySnapshot:
    mountinfo: bytes
    mount_namespace: tuple[int, int, str]
    mount_id: int
    paths: tuple[str, ...]
    writable_roots: tuple[str, ...]
    path_identities: tuple[tuple[str, int, int, int], ...]


def _canonical_json_loads(raw: bytes, label: str) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is not JSON") from exc
    if not isinstance(value, dict) or canonical_json(value) != raw:
        raise RuntimeError(f"{label} is not canonical JSON")
    return value


def _artifact_ref(publication: Publication) -> dict[str, str]:
    return {
        "relative_path": publication.evidence.relative_path,
        "artifact_evidence_sha256": sha256_bytes(canonical_json(publication.evidence.to_dict())),
    }


def _canonical_absolute_text(path: str, label: str) -> str:
    value = validate_absolute_path_text(path, label)
    if os.path.abspath(value) != value:
        fail("G0_PATH_INVALID", f"{label} must be lexically absolute and normalized")
    return value


def _read_absolute_regular_nofollow(
    path: str,
    label: str,
    *,
    limit: int = 256 * 1024 * 1024,
    expected_mount_id: int | None = None,
) -> bytes:
    path = _canonical_absolute_text(path, label)
    if os.path.realpath(path) != path:
        fail(
            "G0_TRUSTED_INPUT_INVALID",
            f"{label} is not a canonical real path",
        )
    parent, name = os.path.split(path)
    if not parent or not name:
        fail("G0_TRUSTED_INPUT_INVALID", f"{label} path is invalid")
    parent_fd = open_absolute_directory(parent)
    try:
        try:
            fd = os.open(name, READ_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise RuntimeError(
                f"{label} is missing, linked, or unreadable"
            ) from exc
        try:
            before = os.fstat(fd)
            before_identity = stable_identity(before)
            if not stat.S_ISREG(before.st_mode):
                fail("G0_TRUSTED_INPUT_INVALID", f"{label} is not regular")
            if (
                expected_mount_id is not None
                and _fd_mount_id(fd) != expected_mount_id
            ):
                fail(
                    "G0_TRUSTED_INPUT_MOUNT_INVALID",
                    f"{label} is not on the pinned repository mount",
                )
            if os.readlink(f"/proc/self/fd/{fd}") != path:
                fail(
                    "G0_TRUSTED_INPUT_INVALID",
                    f"{label} descriptor identity differs",
                )
            chunks: list[bytes] = []
            total = 0
            while True:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                total += len(block)
                if total > limit:
                    fail(
                        "G0_TRUSTED_INPUT_INVALID",
                        f"{label} exceeds bounded reader limit",
                    )
                chunks.append(block)
            after = os.fstat(fd)
            namespace = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                stable_identity(after) != before_identity
                or stable_identity(namespace) != before_identity
                or total != after.st_size
                or (
                    expected_mount_id is not None
                    and _fd_mount_id(fd) != expected_mount_id
                )
            ):
                fail(
                    "G0_TRUSTED_INPUT_INVALID",
                    f"{label} changed during descriptor-bound read",
                )
            return b"".join(chunks)
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def _read_proc_self_bytes(
    relative_path: str,
    *,
    limit: int = 16 * 1024 * 1024,
) -> bytes:
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or type(limit) is not int
        or limit <= 0
    ):
        fail("G0_PROCFS_INVALID", "procfs reader arguments are invalid")
    components = relative_path.split("/")
    if any(
        not component or component in {".", ".."} for component in components
    ):
        fail("G0_PROCFS_INVALID", "procfs relative path is invalid")
    path = f"/proc/self/{relative_path}"
    try:
        fd = os.open(path, READ_FLAGS)
    except OSError as exc:
        raise RuntimeError("required procfs evidence is unavailable") from exc
    try:
        before = os.fstat(fd)
        expected_path = f"/proc/{os.getpid()}/{relative_path}"
        if (
            not stat.S_ISREG(before.st_mode)
            or os.readlink(f"/proc/self/fd/{fd}") != expected_path
        ):
            fail("G0_PROCFS_INVALID", "procfs descriptor identity differs")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > limit:
                fail("G0_PROCFS_INVALID", "procfs evidence exceeds its limit")
            chunks.append(block)
        after = os.fstat(fd)
        namespace = os.stat(path)
        if (
            stable_identity(after) != stable_identity(before)
            or stable_identity(namespace) != stable_identity(before)
            or os.readlink(f"/proc/self/fd/{fd}") != expected_path
        ):
            fail("G0_PROCFS_INVALID", "procfs evidence changed during read")
        return b"".join(chunks)
    finally:
        os.close(fd)


def _fd_mount_id(fd: int) -> int:
    if type(fd) is not int or fd < 0:
        fail("G0_MOUNTINFO_INVALID", "file descriptor is invalid")
    raw = _read_proc_self_bytes(f"fdinfo/{fd}", limit=64 * 1024)
    rows = [
        line[len(b"mnt_id:\t") :]
        for line in raw.splitlines()
        if line.startswith(b"mnt_id:\t")
    ]
    if (
        len(rows) != 1
        or not rows[0]
        or any(byte < 48 or byte > 57 for byte in rows[0])
    ):
        fail("G0_MOUNTINFO_INVALID", "descriptor mount ID is unavailable")
    value = int(rows[0])
    if value <= 0:
        fail("G0_MOUNTINFO_INVALID", "descriptor mount ID is invalid")
    return value


def _mount_namespace_identity() -> tuple[int, int, str]:
    path = "/proc/self/ns/mnt"
    try:
        before = os.stat(path)
        target = os.readlink(path)
        after = os.stat(path)
    except OSError as exc:
        raise RuntimeError("mount namespace identity is unavailable") from exc
    if (
        before.st_dev != after.st_dev
        or before.st_ino != after.st_ino
        or not target.startswith("mnt:[")
        or not target.endswith("]")
        or not target[5:-1].isdigit()
    ):
        fail("G0_MOUNTINFO_UNSTABLE", "mount namespace identity changed")
    return before.st_dev, before.st_ino, target


_MOUNTINFO_ESCAPES = {
    b"040": 0x20,
    b"011": 0x09,
    b"012": 0x0A,
    b"134": 0x5C,
}


def _decode_mountinfo_path(value: bytes, label: str) -> str:
    if any(byte < 0x20 or byte == 0x7F for byte in value):
        fail("G0_MOUNTINFO_INVALID", f"{label} has a raw control byte")
    decoded = bytearray()
    index = 0
    while index < len(value):
        byte = value[index]
        if byte != 0x5C:
            decoded.append(byte)
            index += 1
            continue
        escape = value[index + 1 : index + 4]
        replacement = _MOUNTINFO_ESCAPES.get(escape)
        if len(escape) != 3 or replacement is None:
            fail("G0_MOUNTINFO_INVALID", f"{label} has an invalid escape")
        decoded.append(replacement)
        index += 4
    path = os.fsdecode(bytes(decoded))
    if (
        not path.startswith("/")
        or os.path.normpath(path) != path
        or os.path.abspath(path) != path
        or path.endswith(" (deleted)")
        or any(
            0xD800 <= ord(character) <= 0xDFFF
            for character in path
        )
    ):
        fail("G0_MOUNTINFO_INVALID", f"{label} is not a canonical path")
    return path


def _strict_decimal(value: bytes, label: str, *, allow_zero: bool) -> int:
    if not value or any(byte < 48 or byte > 57 for byte in value):
        fail("G0_MOUNTINFO_INVALID", f"{label} is not decimal")
    parsed = int(value)
    if parsed < 0 or (parsed == 0 and not allow_zero):
        fail("G0_MOUNTINFO_INVALID", f"{label} is out of range")
    return parsed


def _parse_mountinfo(raw: bytes) -> tuple[_MountInfoEntry, ...]:
    if not raw or not raw.endswith(b"\n"):
        fail("G0_MOUNTINFO_INVALID", "mountinfo is empty or unterminated")
    entries: list[_MountInfoEntry] = []
    mount_ids: set[int] = set()
    mount_points: set[str] = set()
    for line_number, line in enumerate(raw[:-1].split(b"\n"), start=1):
        if not line or line.count(b" - ") != 1:
            fail("G0_MOUNTINFO_INVALID", "mountinfo row structure differs")
        left, right = line.split(b" - ", 1)
        fields = left.split(b" ")
        post_fields = right.split(b" ")
        if len(fields) < 6 or len(post_fields) < 3 or any(not item for item in fields):
            fail("G0_MOUNTINFO_INVALID", "mountinfo row fields differ")
        mount_id = _strict_decimal(
            fields[0], f"mountinfo[{line_number}].mount_id", allow_zero=False
        )
        parent_mount_id = _strict_decimal(
            fields[1], f"mountinfo[{line_number}].parent_mount_id", allow_zero=True
        )
        device_parts = fields[2].split(b":")
        if len(device_parts) != 2:
            fail("G0_MOUNTINFO_INVALID", "mountinfo device differs")
        device = (
            _strict_decimal(
                device_parts[0],
                f"mountinfo[{line_number}].device_major",
                allow_zero=True,
            ),
            _strict_decimal(
                device_parts[1],
                f"mountinfo[{line_number}].device_minor",
                allow_zero=True,
            ),
        )
        mount_root = _decode_mountinfo_path(
            fields[3], f"mountinfo[{line_number}].root"
        )
        mount_point = _decode_mountinfo_path(
            fields[4], f"mountinfo[{line_number}].mount_point"
        )
        if mount_id in mount_ids or mount_point in mount_points:
            fail("G0_MOUNTINFO_INVALID", "mountinfo identifiers are ambiguous")
        mount_ids.add(mount_id)
        mount_points.add(mount_point)
        entries.append(
            _MountInfoEntry(
                mount_id=mount_id,
                parent_mount_id=parent_mount_id,
                device=device,
                mount_root=mount_root,
                mount_point=mount_point,
            )
        )
    return tuple(entries)


def _read_stable_mountinfo(
) -> tuple[bytes, tuple[int, int, str], tuple[_MountInfoEntry, ...]]:
    namespace_before = _mount_namespace_identity()
    first = _read_proc_self_bytes("mountinfo")
    namespace_middle = _mount_namespace_identity()
    second = _read_proc_self_bytes("mountinfo")
    namespace_after = _mount_namespace_identity()
    if not (
        namespace_before == namespace_middle == namespace_after
        and first == second
    ):
        fail("G0_MOUNTINFO_UNSTABLE", "mountinfo changed during capture")
    return first, namespace_before, _parse_mountinfo(first)


def _require_no_sys_admin() -> None:
    raw = _read_proc_self_bytes("status", limit=1024 * 1024)
    mask = 1 << 21
    for label in (b"CapInh", b"CapPrm", b"CapEff", b"CapAmb"):
        prefix = label + b":"
        rows = [line[len(prefix) :].strip() for line in raw.splitlines() if line.startswith(prefix)]
        if (
            len(rows) != 1
            or not rows[0]
            or any(
                not (48 <= byte <= 57 or 65 <= byte <= 70 or 97 <= byte <= 102)
                for byte in rows[0]
            )
        ):
            fail("G0_MOUNT_AUTHORITY_UNSAFE", "capability evidence differs")
        if int(rows[0], 16) & mask:
            fail(
                "G0_MOUNT_AUTHORITY_UNSAFE",
                "CAP_SYS_ADMIN is present in the bootstrap process",
            )


def _mount_entry_for_path(
    path: str, entries: tuple[_MountInfoEntry, ...]
) -> _MountInfoEntry:
    matches = [
        entry
        for entry in entries
        if entry.mount_point == "/"
        or path == entry.mount_point
        or path.startswith(f"{entry.mount_point}/")
    ]
    if not matches:
        fail("G0_MOUNT_BOUNDARY_INVALID", "path has no mountinfo mapping")
    longest = max(len(entry.mount_point) for entry in matches)
    winners = [entry for entry in matches if len(entry.mount_point) == longest]
    if len(winners) != 1:
        fail("G0_MOUNT_BOUNDARY_INVALID", "path mountinfo mapping is ambiguous")
    return winners[0]


def _directory_mount_evidence(
    path: str, label: str
) -> tuple[str, int, int, int]:
    path = _canonical_absolute_text(path, label)
    fd = open_absolute_directory(path)
    try:
        current = os.fstat(fd)
        return path, _fd_mount_id(fd), current.st_dev, current.st_ino
    finally:
        os.close(fd)


def _strict_descendant(path: str, parent: str) -> bool:
    prefix = "/" if parent == "/" else f"{parent}/"
    return path != parent and path.startswith(prefix)


def _capture_mount_boundary(
    paths: Iterable[str], writable_roots: Iterable[str]
) -> _MountBoundarySnapshot:
    canonical_paths = tuple(
        _canonical_absolute_text(path, f"mount_boundary[{index}]")
        for index, path in enumerate(paths)
    )
    canonical_writable_roots = tuple(
        _canonical_absolute_text(path, f"writable_root[{index}]")
        for index, path in enumerate(writable_roots)
    )
    if not canonical_paths or not canonical_writable_roots:
        fail("G0_MOUNT_BOUNDARY_INVALID", "mount boundary is incomplete")
    _require_no_sys_admin()
    raw, namespace, entries = _read_stable_mountinfo()
    identities = tuple(
        _directory_mount_evidence(path, f"mount_boundary[{index}]")
        for index, path in enumerate(canonical_paths)
    )
    mount_ids = {identity[1] for identity in identities}
    if len(mount_ids) != 1:
        fail(
            "G0_MOUNT_BOUNDARY_INVALID",
            "all protected, repository, and writable boundaries must share one mount ID",
        )
    mount_id = next(iter(mount_ids))
    mounted_rows = [entry for entry in entries if entry.mount_id == mount_id]
    if len(mounted_rows) != 1:
        fail("G0_MOUNT_BOUNDARY_INVALID", "boundary mount row is ambiguous")
    mounted_row = mounted_rows[0]
    for path, identity_mount_id, device, _inode in identities:
        mapped = _mount_entry_for_path(path, entries)
        if (
            identity_mount_id != mount_id
            or mapped.mount_id != mount_id
            or (os.major(device), os.minor(device)) != mounted_row.device
        ):
            fail(
                "G0_MOUNT_BOUNDARY_INVALID",
                "boundary descriptor and mountinfo evidence differ",
            )
    for entry in entries:
        if any(
            _strict_descendant(entry.mount_point, root)
            for root in canonical_writable_roots
        ):
            fail(
                "G0_WRITABLE_NESTED_MOUNT",
                "writable storage contains a nested mount",
            )
    return _MountBoundarySnapshot(
        mountinfo=raw,
        mount_namespace=namespace,
        mount_id=mount_id,
        paths=canonical_paths,
        writable_roots=canonical_writable_roots,
        path_identities=identities,
    )


def _revalidate_mount_boundary(
    expected: _MountBoundarySnapshot,
    required_directories: Iterable[str] = (),
) -> None:
    current = _capture_mount_boundary(expected.paths, expected.writable_roots)
    if current != expected:
        fail("G0_MOUNT_BOUNDARY_CHANGED", "mount boundary changed after preflight")
    mounted_row = next(
        entry
        for entry in _parse_mountinfo(expected.mountinfo)
        if entry.mount_id == expected.mount_id
    )
    for index, path in enumerate(required_directories):
        canonical_path = _canonical_absolute_text(
            path, f"required_directory[{index}]"
        )
        if not any(
            canonical_path == root or _strict_descendant(canonical_path, root)
            for root in expected.writable_roots
        ):
            fail(
                "G0_MOUNT_BOUNDARY_INVALID",
                "required directory escapes writable storage",
            )
        _path, mount_id, device, _inode = _directory_mount_evidence(
            canonical_path, f"required_directory[{index}]"
        )
        if (
            mount_id != expected.mount_id
            or (os.major(device), os.minor(device)) != mounted_row.device
        ):
            fail(
                "G0_MOUNT_BOUNDARY_CHANGED",
                "created directory is not on the pinned mount",
            )


def _validate_repository_tool_mount(
    path: str, label: str, expected_mount_id: int
) -> None:
    path = _canonical_absolute_text(path, label)
    if os.path.realpath(path) != path:
        fail("G0_TRUSTED_INPUT_INVALID", f"{label} is not canonical")
    parent, name = os.path.split(path)
    if not parent or not name:
        fail("G0_TRUSTED_INPUT_INVALID", f"{label} path is invalid")
    parent_fd = open_absolute_directory(parent)
    try:
        try:
            fd = os.open(name, READ_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise RuntimeError(f"{label} is missing, linked, or unreadable") from exc
        try:
            before = os.fstat(fd)
            if (
                not stat.S_ISREG(before.st_mode)
                or os.readlink(f"/proc/self/fd/{fd}") != path
                or _fd_mount_id(fd) != expected_mount_id
            ):
                fail(
                    "G0_TRUSTED_INPUT_MOUNT_INVALID",
                    f"{label} descriptor is not on the pinned repository mount",
                )
            namespace = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                stable_identity(namespace) != stable_identity(before)
                or _fd_mount_id(fd) != expected_mount_id
            ):
                fail(
                    "G0_TRUSTED_INPUT_MOUNT_INVALID",
                    f"{label} namespace identity differs",
                )
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def _repository_root() -> str:
    repository_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    )
    return _canonical_absolute_text(repository_root, "repository root")


def _repository_path(relative_path: str, label: str) -> str:
    return _canonical_absolute_text(
        os.path.join(_repository_root(), *relative_path.split("/")),
        label,
    )


def _repository_provider_path() -> str:
    return _repository_path(
        "scripts/g0/provider.py", "repository canonical provider"
    )


def _pin_repository_path(
    supplied_path: str, relative_path: str, label: str
) -> str:
    supplied = _canonical_absolute_text(supplied_path, label)
    expected = _repository_path(relative_path, f"repository canonical {label}")
    if supplied != expected:
        fail(
            "G0_TRUSTED_INPUT_PATH_INVALID",
            f"{label} must be the canonical repository path",
        )
    return expected


def _trusted_provider_bytes(
    provider_path: str, *, expected_mount_id: int
) -> bytes:
    canonical_path = _repository_provider_path()
    if _canonical_absolute_text(provider_path, "provider_path") != canonical_path:
        fail(
            "G0_TRUSTED_PROVIDER_INVALID",
            "provider path is not the canonical repository implementation",
        )
    canonical_bytes = _read_absolute_regular_nofollow(
        canonical_path,
        "repository canonical provider",
        expected_mount_id=expected_mount_id,
    )
    canonical_sha256 = sha256_bytes(canonical_bytes)
    if canonical_sha256 != CP0_TRUSTED_PROVIDER_SHA256:
        fail(
            "G0_TRUSTED_PROVIDER_INVALID",
            "repository provider is not the pinned implementation",
        )
    return canonical_bytes


def _validate_source_roots(value: Mapping[str, str]) -> list[dict[str, str]]:
    expected = {"experiments4", "experiments5", "experiments6"}
    if set(value) != expected:
        fail("G0_SOURCE_ROOTS_INVALID", "G0 requires exactly experiments4/5/6 root IDs")
    rows = []
    for root_id in sorted(value):
        path = value[root_id]
        if not isinstance(path, str):
            fail("G0_SOURCE_ROOTS_INVALID", "source root path must be text")
        rows.append({"root_id": root_id, "path": _canonical_absolute_text(path, root_id)})
    return rows


def _directory_inode_chain(path: str, label: str) -> tuple[tuple[int, int], ...]:
    path = _canonical_absolute_text(path, label)
    fd = os.open("/", DIR_FLAGS)
    expected = "/"
    chain: list[tuple[int, int]] = []
    try:
        root_stat = os.fstat(fd)
        chain.append((root_stat.st_dev, root_stat.st_ino))
        for component in path.split("/")[1:]:
            try:
                next_fd = os.open(component, DIR_FLAGS, dir_fd=fd)
            except OSError:
                fail(
                    "G0_PROTECTED_BOUNDARY_INVALID",
                    f"{label} is missing, linked, or not a directory",
                )
            os.close(fd)
            fd = next_fd
            expected = (
                f"/{component}" if expected == "/" else f"{expected}/{component}"
            )
            descriptor_path = os.readlink(f"/proc/self/fd/{fd}")
            if (
                descriptor_path != expected
                or descriptor_path.endswith(" (deleted)")
                or os.path.realpath(expected) != expected
            ):
                fail(
                    "G0_PROTECTED_BOUNDARY_INVALID",
                    f"{label} descriptor identity is not canonical",
                )
            current = os.fstat(fd)
            chain.append((current.st_dev, current.st_ino))
        if expected != path:
            fail("G0_PROTECTED_BOUNDARY_INVALID", f"{label} path differs")
        return tuple(chain)
    finally:
        os.close(fd)


def _paths_overlap(left: str, right: str) -> bool:
    try:
        common = os.path.commonpath((left, right))
    except ValueError:
        return True
    return common in {left, right}


def _require_run_root_absent(run_root: str) -> None:
    run_root = _canonical_absolute_text(run_root, "run_root")
    try:
        os.lstat(run_root)
    except FileNotFoundError:
        return
    except OSError:
        fail(
            "G0_RUN_ROOT_NOT_FRESH",
            "run_root freshness cannot be established",
        )
    fail("G0_RUN_ROOT_NOT_FRESH", "run_root must not already exist")


def _preflight_strict_protected_boundaries(
    *,
    strict_parent: str,
    run_root: str,
    roots: list[dict[str, str]],
    paper_path: str,
) -> None:
    protected_directories = [
        os.path.dirname(paper_path),
        *(row["path"] for row in roots),
    ]
    for storage in (strict_parent, run_root):
        for protected in protected_directories:
            if _paths_overlap(storage, protected):
                fail(
                    "G0_STRICT_PROTECTED_OVERLAP",
                    "strict storage overlaps a protected directory",
                )

    _require_run_root_absent(run_root)
    strict_chains = [_directory_inode_chain(strict_parent, "strict_parent")]

    for index, protected in enumerate(protected_directories):
        protected_chain = _directory_inode_chain(
            protected, f"protected_directory[{index}]"
        )
        for strict_chain in strict_chains:
            if (
                strict_chain[-1] in protected_chain
                or protected_chain[-1] in strict_chain
            ):
                fail(
                    "G0_STRICT_PROTECTED_ALIAS",
                    "strict storage aliases a protected directory ancestry",
                )


def _preflight_external_directory(
    external_dir: str,
    *,
    strict_parent: str,
    run_root: str,
    roots: list[dict[str, str]],
    paper_path: str,
) -> str:
    external_dir = _canonical_absolute_text(external_dir, "external_dir")
    protected_directories = [
        strict_parent,
        os.path.dirname(paper_path),
        *(row["path"] for row in roots),
    ]
    for protected in (run_root, *protected_directories):
        if _paths_overlap(external_dir, protected):
            fail(
                "G0_EXTERNAL_PROTECTED_OVERLAP",
                "external output directory overlaps protected or strict storage",
            )
    external_chain = _directory_inode_chain(external_dir, "external_dir")
    for index, protected in enumerate(protected_directories):
        protected_chain = _directory_inode_chain(
            protected, f"protected_directory[{index}]"
        )
        if (
            external_chain[-1] in protected_chain
            or protected_chain[-1] in external_chain
        ):
            fail(
                "G0_EXTERNAL_PROTECTED_ALIAS",
                "external output directory aliases a protected directory ancestry",
            )
    return external_dir


def _prepare_child_output_parents(
    external_dir: str, *, expected_mount_id: int
) -> dict[str, str]:
    names = {
        "source_stdout": "source-stdout",
        "source_execution": "source-execution",
        "paper_stdout": "paper-stdout",
        "paper_execution": "paper-execution",
    }
    root_fd = open_absolute_directory(external_dir)
    paths: dict[str, str] = {}
    try:
        if _fd_mount_id(root_fd) != expected_mount_id:
            fail(
                "G0_EXTERNAL_OUTPUT_PARENT_INVALID",
                "external root descriptor is not on the pinned mount",
            )
        for key, name in names.items():
            if _fd_mount_id(root_fd) != expected_mount_id:
                fail(
                    "G0_EXTERNAL_OUTPUT_PARENT_INVALID",
                    "external root mount changed before mkdirat",
                )
            try:
                os.mkdir(name, 0o700, dir_fd=root_fd)
                child_fd = os.open(name, DIR_FLAGS, dir_fd=root_fd)
            except OSError:
                fail(
                    "G0_EXTERNAL_OUTPUT_PARENT_INVALID",
                    "dedicated child evidence parent already exists or is unsafe",
                )
            try:
                path = f"{external_dir}/{name}"
                if (
                    os.readlink(f"/proc/self/fd/{child_fd}") != path
                    or os.path.realpath(path) != path
                    or _fd_mount_id(child_fd) != expected_mount_id
                ):
                    fail(
                        "G0_EXTERNAL_OUTPUT_PARENT_INVALID",
                        "dedicated child evidence parent identity differs",
                    )
                paths[key] = path
            finally:
                os.close(child_fd)
    finally:
        os.close(root_fd)
    return paths


def _reservation_argv(
    *,
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    external_dir: str,
    project_id: str,
    owner_seed: str,
    roots: list[dict[str, str]],
    paper_path: str,
    protected_executable: str,
    provider_path: str,
    comparison_tool: str,
    validator_path: str,
    timeout_seconds: int,
) -> tuple[str, ...]:
    argv = [
        sys.executable,
        "-B",
        "bootstrap_g0_cp0",
        "--strict-parent",
        strict_parent,
        "--sealed-run-id",
        sealed_run_id,
        "--run-root",
        run_root,
        "--external-dir",
        external_dir,
        "--project-id",
        project_id,
        "--owner-seed-sha256",
        sha256_bytes(owner_seed.encode("utf-8")),
    ]
    for row in roots:
        argv.extend((f"--source-root-{row['root_id']}", row["path"]))
    argv.extend(
        (
            "--paper-path",
            paper_path,
            "--protected-executable",
            protected_executable,
            "--provider-path",
            provider_path,
            "--comparison-tool",
            comparison_tool,
            "--validator-path",
            validator_path,
            "--timeout-seconds",
            str(timeout_seconds),
        )
    )
    return tuple(argv)


def _full_descriptor_identity(value: os.stat_result) -> dict[str, object]:
    identity = stable_identity(value)
    identity.update(
        {
            "nlink": value.st_nlink,
            "mtime_ns": value.st_mtime_ns,
            "ctime_ns": value.st_ctime_ns,
        }
    )
    return identity


def _assert_transport_mount(
    fd: int,
    run_binding: RunDescriptorBinding,
    label: str,
) -> int:
    current = mount_identity(fd)
    namespace = current.get("mount_namespace")
    expected = run_binding.root
    if (
        os.fstat(fd).st_dev != expected.dev
        or current.get("mount_id") != expected.mount_id
        or not isinstance(namespace, dict)
        or namespace.get("dev") != expected.mount_namespace_dev
        or namespace.get("inode") != expected.mount_namespace_inode
        or namespace.get("link") != expected.mount_namespace_link
    ):
        fail("MOUNT_SUBSTITUTION", f"{label} crossed the bound run mount")
    return expected.mount_id


def _read_frozen_publication_bytes(
    *,
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    run_binding: RunDescriptorBinding,
    publications: Mapping[str, Publication],
) -> dict[str, tuple[bytes, dict[str, object]]]:
    result: dict[str, tuple[bytes, dict[str, object]]] = {}
    with open_bound_run_handle(
        strict_parent, sealed_run_id, run_root, run_binding
    ) as handle:
        for role, publication in publications.items():
            evidence = publication.evidence
            if (
                evidence.artifact_type != "regular"
                or evidence.sha256 is None
                or evidence.size is None
            ):
                fail(
                    "G0_DESCRIPTOR_TRANSPORT_INVALID",
                    f"{role} publication is not a regular artifact",
                )
            parent_fd, name = open_parent_directory(
                handle.root_fd, evidence.relative_path
            )
            file_fd = -1
            try:
                _assert_transport_mount(parent_fd, run_binding, f"{role} parent")
                file_fd = os.open(name, READ_FLAGS, dir_fd=parent_fd)
                _assert_transport_mount(file_fd, run_binding, role)
                before = os.fstat(file_fd)
                before_full = _full_descriptor_identity(before)
                namespace = os.stat(
                    name, dir_fd=parent_fd, follow_symlinks=False
                )
                if (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_nlink != 1
                    or stable_identity(before) != evidence.identity
                    or stable_identity(namespace) != evidence.identity
                ):
                    fail(
                        "G0_DESCRIPTOR_TRANSPORT_INVALID",
                        f"{role} descriptor differs from publication evidence",
                    )
                digest = hashlib.sha256()
                chunks: list[bytes] = []
                total = 0
                while True:
                    block = os.read(file_fd, 1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > 128 * 1024 * 1024:
                        fail(
                            "G0_DESCRIPTOR_TRANSPORT_INVALID",
                            f"{role} exceeds the descriptor transport limit",
                        )
                    digest.update(block)
                    chunks.append(block)
                after = os.fstat(file_fd)
                namespace_after = os.stat(
                    name, dir_fd=parent_fd, follow_symlinks=False
                )
                if (
                    _full_descriptor_identity(after) != before_full
                    or stable_identity(namespace_after) != evidence.identity
                    or total != evidence.size
                    or digest.hexdigest() != evidence.sha256
                ):
                    fail(
                        "G0_DESCRIPTOR_TRANSPORT_INVALID",
                        f"{role} bytes differ from publication evidence",
                    )
                result[role] = (
                    b"".join(chunks),
                    {
                        "relative_path": evidence.relative_path,
                        "sha256": evidence.sha256,
                        "bytes": evidence.size,
                        "identity": dict(evidence.identity),
                        "mount_id": run_binding.root.mount_id,
                    },
                )
            finally:
                if file_fd >= 0:
                    os.close(file_fd)
                os.close(parent_fd)
        run_binding.assert_handle(handle)
    return result


def _read_running_python_bytes(expected_sha256: str) -> bytes:
    fd = os.open("/proc/self/exe", os.O_RDONLY | os.O_CLOEXEC)
    try:
        before = os.fstat(fd)
        before_full = _full_descriptor_identity(before)
        if not stat.S_ISREG(before.st_mode):
            fail(
                "G0_DESCRIPTOR_TRANSPORT_INVALID",
                "running Python executable is not regular",
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > 512 * 1024 * 1024:
                fail(
                    "G0_DESCRIPTOR_TRANSPORT_INVALID",
                    "running Python executable exceeds its limit",
                )
            digest.update(block)
            chunks.append(block)
        if (
            _full_descriptor_identity(os.fstat(fd)) != before_full
            or digest.hexdigest() != expected_sha256
        ):
            fail(
                "G0_DESCRIPTOR_TRANSPORT_INVALID",
                "running Python executable differs from runtime identity",
            )
        return b"".join(chunks)
    finally:
        os.close(fd)


def _sealed_memfd(label: str, payload: bytes, mode: int) -> int:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        fail(
            "G0_DESCRIPTOR_TRANSPORT_UNAVAILABLE",
            "sealed memfd transport requires Linux x86_64",
        )
    libc = ctypes.CDLL(None, use_errno=True)
    fd = int(
        libc.syscall(
            _SYS_MEMFD_CREATE_X86_64,
            f"experiments7-{label}".encode("ascii"),
            _MFD_CLOEXEC | _MFD_ALLOW_SEALING,
        )
    )
    if fd < 0:
        error = ctypes.get_errno()
        raise StrictRunError(
            "G0_DESCRIPTOR_TRANSPORT_UNAVAILABLE",
            f"sealed memfd creation failed with errno {error}",
        )
    try:
        os.fchmod(fd, mode)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                fail(
                    "G0_DESCRIPTOR_TRANSPORT_INVALID",
                    "sealed memfd write made no progress",
                )
            view = view[written:]
        os.lseek(fd, 0, os.SEEK_SET)
        fcntl.fcntl(fd, _F_ADD_SEALS, _REQUIRED_MEMFD_SEALS)
        if fcntl.fcntl(fd, _F_GET_SEALS) != _REQUIRED_MEMFD_SEALS:
            fail(
                "G0_DESCRIPTOR_TRANSPORT_INVALID",
                "sealed memfd seal set differs",
            )
        sealed_before = os.fstat(fd)
        sealed_identity = _full_descriptor_identity(sealed_before)
        if (
            not stat.S_ISREG(sealed_before.st_mode)
            or stat.S_IMODE(sealed_before.st_mode) != mode
            or sealed_before.st_size != len(payload)
        ):
            fail(
                "G0_DESCRIPTOR_TRANSPORT_INVALID",
                "sealed memfd metadata differs from its source payload",
            )
        digest = hashlib.sha256()
        total = 0
        offset = 0
        while True:
            block = os.pread(fd, 1024 * 1024, offset)
            if not block:
                break
            offset += len(block)
            total += len(block)
            digest.update(block)
        if (
            _full_descriptor_identity(os.fstat(fd)) != sealed_identity
            or total != len(payload)
            or digest.hexdigest() != sha256_bytes(payload)
        ):
            fail(
                "G0_DESCRIPTOR_TRANSPORT_INVALID",
                "sealed memfd bytes differ from their source payload",
            )
        os.lseek(fd, 0, os.SEEK_SET)
        return fd
    except Exception:
        os.close(fd)
        raise


def _protected_result(
    executable: str,
    config: str,
    mode: str,
    *,
    timeout_seconds: int,
    argv: list[str],
    environment: dict[str, str],
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    run_binding: RunDescriptorBinding,
    executable_publication: Publication,
    configuration_publication: Publication,
    provider_publication: Publication,
    runtime_executable_sha256: str,
    producer_binding: bytes | None = None,
) -> _ProtectedChildResult:
    if (
        len(argv) < 7
        or any(not isinstance(item, str) or not item for item in argv)
        or argv[2] != executable
        or argv[3:5] != ["--config", config]
    ):
        raise RuntimeError("protected G0 child argv is invalid")
    frozen = _read_frozen_publication_bytes(
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
        run_binding=run_binding,
        publications={
            "g0_executable": executable_publication,
            "configuration": configuration_publication,
            "provider": provider_publication,
        },
    )
    runtime_bytes = _read_running_python_bytes(runtime_executable_sha256)
    descriptors: dict[str, int] = {}
    try:
        descriptors["runtime"] = _sealed_memfd(
            "runtime", runtime_bytes, 0o500
        )
        for role in ("g0_executable", "configuration", "provider"):
            descriptors[role] = _sealed_memfd(role, frozen[role][0], 0o400)
        actual_argv = list(argv)
        actual_argv[2] = f"/proc/self/fd/{descriptors['g0_executable']}"
        actual_argv[4] = f"/proc/self/fd/{descriptors['configuration']}"
        actual_argv[5:5] = [
            "--provider-fd",
            str(descriptors["provider"]),
        ]
        transport = {
            "schema": G0_DESCRIPTOR_TRANSPORT_SCHEMA,
            "logical_argv": list(argv),
            "executed_argv": actual_argv,
            "executed_argv_sha256": sha256_bytes(canonical_json(actual_argv)),
            "runtime": {
                "fd": descriptors["runtime"],
                "proc_path": f"/proc/self/fd/{descriptors['runtime']}",
                "sha256": runtime_executable_sha256,
                "bytes": len(runtime_bytes),
                "seals": _REQUIRED_MEMFD_SEALS,
            },
            "artifacts": {
                role: {
                    **frozen[role][1],
                    "fd": descriptors[role],
                    "proc_path": f"/proc/self/fd/{descriptors[role]}",
                    "seals": _REQUIRED_MEMFD_SEALS,
                }
                for role in ("g0_executable", "configuration", "provider")
            },
            "run_root": {
                "identity": {
                    "type": "directory",
                    "dev": run_binding.root.dev,
                    "inode": run_binding.root.inode,
                    "mode": run_binding.root.mode,
                    "uid": run_binding.root.uid,
                    "gid": run_binding.root.gid,
                },
                "mount_id": run_binding.root.mount_id,
            },
        }
        launched = time.monotonic_ns()
        run_kwargs: dict[str, object] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "env": environment,
            "check": False,
            "text": False,
            "timeout": timeout_seconds,
            "executable": transport["runtime"]["proc_path"],
            "pass_fds": tuple(descriptors.values()),
        }
        if producer_binding is None:
            run_kwargs["stdin"] = subprocess.DEVNULL
        else:
            run_kwargs["input"] = producer_binding
        completed = subprocess.run(actual_argv, **run_kwargs)
    finally:
        for fd in descriptors.values():
            os.close(fd)
    completed_monotonic_ns = time.monotonic_ns()
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", "replace").strip()
        raise RuntimeError(f"protected G0 child failed in {mode}: {detail}")
    result = _canonical_json_loads(completed.stdout, f"protected G0 {mode} result")
    if result.get("schema") != G0_RESULT_SCHEMA or result.get("mode") != mode:
        raise RuntimeError("protected G0 child response schema differs")
    return _ProtectedChildResult(
        result,
        completed.stdout,
        completed.stderr,
        tuple(argv),
        transport,
        dict(environment),
        launched,
        completed_monotonic_ns,
        completed.returncode,
    )


def _publish(
    publisher: StagePublisher,
    relative: str,
    payload: bytes,
    stage: list[Publication],
    *,
    context: dict[str, object] | None = None,
) -> Publication:
    publications = publisher.publish_bytes(relative, payload, context=context)
    stage.extend(publications)
    return publications[-1]


def _validate_producer_attestation(
    value: object,
    operation: str,
    acceptance: Publication,
    envelope: Publication,
) -> dict[str, object]:
    required = {
        "schema",
        "sealed_run_id",
        "run_root",
        "operation",
        "started_monotonic_ns",
        "acceptance_transcript_sha256",
        "envelope_artifact_evidence_sha256",
        "envelope_transcript_sha256",
        "envelope_context",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise RuntimeError("protected read producer attestation schema is not exact")
    started = value["started_monotonic_ns"]
    acceptance_emitted = acceptance.transcript.get("emitted_monotonic_ns")
    envelope_emitted = envelope.transcript.get("emitted_monotonic_ns")
    if (
        value["schema"] != G0_PROTECTED_READ_ATTESTATION_SCHEMA
        or value["sealed_run_id"] != envelope.transcript.get("sealed_run_id")
        or value["run_root"] != envelope.transcript.get("run_root")
        or value["operation"] != operation
        or type(started) is not int
        or type(acceptance_emitted) is not int
        or type(envelope_emitted) is not int
        or not (acceptance_emitted < started and envelope_emitted < started)
        or value["acceptance_transcript_sha256"]
        != acceptance.evidence.publication_transcript_sha256
        or value["envelope_artifact_evidence_sha256"]
        != sha256_bytes(canonical_json(envelope.evidence.to_dict()))
        or value["envelope_transcript_sha256"]
        != envelope.evidence.publication_transcript_sha256
        or value["envelope_context"] != envelope.transcript.get("context")
    ):
        raise RuntimeError("protected read producer attestation binding differs")
    return value


def _external_evidence_payload(
    evidence: ExternalOutputEvidence,
) -> dict[str, object]:
    return {
        "path": evidence.path,
        "sha256": evidence.sha256,
        "bytes": evidence.size,
        "identity": evidence.identity,
        "parent_identity": evidence.parent_identity,
    }


def _run_binding_from_reservation(
    payload: Mapping[str, object],
) -> RunDescriptorBinding:
    parent = payload.get("parent")
    child = payload.get("child")
    if not isinstance(parent, Mapping) or not isinstance(child, Mapping):
        fail("RUN_BINDING_INVALID", "reservation namespace evidence is missing")
    parent_descriptor = parent.get("descriptor")
    child_descriptor = child.get("descriptor")
    if (
        not isinstance(parent_descriptor, Mapping)
        or not isinstance(child_descriptor, Mapping)
    ):
        fail("RUN_BINDING_INVALID", "reservation descriptor evidence is missing")
    return RunDescriptorBinding.from_evidence(
        parent_identity=parent_descriptor.get("identity"),
        parent_mount=parent.get("mount"),
        root_identity=child_descriptor.get("identity"),
        root_mount=child.get("mount"),
    )


def _assert_bound_run_current(
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    run_binding: RunDescriptorBinding,
) -> None:
    with open_bound_run_handle(
        strict_parent, sealed_run_id, run_root, run_binding
    ):
        pass


def _pinned_external_record(
    evidence: ExternalOutputEvidence,
    *,
    expected_mount_id: int,
    json_pointer: tuple[str | int, ...] = (),
) -> ExternalTranscriptRecord:
    return ExternalTranscriptRecord(
        evidence.path,
        json_pointer,
        expected_sha256=evidence.sha256,
        expected_size=evidence.size,
        expected_identity=evidence.identity,
        expected_parent_identity=evidence.parent_identity,
        expected_mount_id=expected_mount_id,
    )


def _publish_child_evidence(
    *,
    capture: _ProtectedChildResult,
    operation: str,
    expected_argv: list[str],
    expected_environment: dict[str, str],
    producer_binding_bytes: bytes,
    producer_attestation: dict[str, object],
    stdout_path: str,
    execution_path: str,
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    executable_sha256: str,
    runtime_executable_sha256: str,
    configuration_sha256: str,
    provider_sha256: str,
    expected_mount_id: int,
    run_binding: RunDescriptorBinding,
    evidence_sink: dict[str, ExternalOutputEvidence],
) -> dict[str, object]:
    started = producer_attestation.get("started_monotonic_ns")
    if (
        capture.argv != tuple(expected_argv)
        or capture.environment != expected_environment
        or capture.result.get("mode") != operation
        or capture.stdout != canonical_json(capture.result)
        or capture.exit_status != 0
        or capture.stderr != b""
        or type(capture.launched_monotonic_ns) is not int
        or type(capture.completed_monotonic_ns) is not int
        or type(started) is not int
        or not (
            capture.launched_monotonic_ns
            < started
            < capture.completed_monotonic_ns
        )
    ):
        raise RuntimeError("protected G0 child execution evidence differs")

    with ExternalOutputReservation.create_for_existing_run(
        stdout_path,
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
        expected_mount_id=expected_mount_id,
        expected_run_binding=run_binding,
    ) as stdout_output:
        stdout_evidence = stdout_output.publish_bytes(capture.stdout)
    evidence_sink[stdout_path] = stdout_evidence
    execution = {
        "schema": G0_CHILD_EXECUTION_SCHEMA,
        "sealed_run_id": sealed_run_id,
        "run_root": run_root,
        "mode": operation,
        "argv": list(capture.argv),
        "argv_sha256": sha256_bytes(canonical_json(list(capture.argv))),
        "descriptor_transport": capture.descriptor_transport,
        "runtime_executable_sha256": runtime_executable_sha256,
        "executable_sha256": executable_sha256,
        "configuration_sha256": configuration_sha256,
        "provider_sha256": provider_sha256,
        "producer_binding_sha256": sha256_bytes(producer_binding_bytes),
        "environment": dict(capture.environment),
        "environment_sha256": sha256_bytes(
            canonical_json(dict(capture.environment))
        ),
        "launched_monotonic_ns": capture.launched_monotonic_ns,
        "completed_monotonic_ns": capture.completed_monotonic_ns,
        "exit_status": capture.exit_status,
        "stderr_sha256": sha256_bytes(capture.stderr),
        "stderr_bytes": len(capture.stderr),
        "stdout_evidence": _external_evidence_payload(stdout_evidence),
    }
    with ExternalOutputReservation.create_for_existing_run(
        execution_path,
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
        expected_mount_id=expected_mount_id,
        expected_run_binding=run_binding,
    ) as execution_output:
        execution_evidence = execution_output.publish_json(execution)
    evidence_sink[execution_path] = execution_evidence
    context = dict(producer_attestation)
    context["child_stdout_sha256"] = stdout_evidence.sha256
    context["child_execution_sha256"] = execution_evidence.sha256
    context["child_execution_evidence"] = _external_evidence_payload(
        execution_evidence
    )
    return context


def _runtime_identity() -> dict[str, object]:
    executable = _canonical_absolute_text(
        os.path.realpath(sys.executable), "runtime python executable"
    )
    executable_bytes = _read_absolute_regular_nofollow(
        executable, "runtime python executable"
    )
    return {
        "schema": G0_RUNTIME_SCHEMA,
        "python": {
            "path": executable,
            "sha256": sha256_bytes(executable_bytes),
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
    }


def _validate_provider_evidence(
    value: object, runtime_platform: object
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "provider",
        "landlock_abi",
        "seccomp_mode",
        "denied_metadata_syscalls_x86_64",
        "kernel",
        "machine",
        "landlock_rules",
        "default_filesystem_rights",
        "writable_directories",
        "metadata_mutation_policy",
    }:
        raise RuntimeError("protected provider evidence schema differs")
    if not isinstance(runtime_platform, dict) or set(runtime_platform) != {
        "system",
        "release",
        "machine",
    }:
        raise RuntimeError("runtime platform evidence schema differs")
    rules = value["landlock_rules"]
    if not isinstance(rules, list) or len(rules) != 1:
        raise RuntimeError("protected Landlock rule set differs")
    rule = rules[0]
    if not isinstance(rule, dict) or set(rule) != {
        "path",
        "dev",
        "inode",
        "rights",
    }:
        raise RuntimeError("protected Landlock rule schema differs")
    if (
        value["provider"]
        != "landlock_path_beneath+seccomp_metadata_deny"
        or type(value["landlock_abi"]) is not int
        or value["landlock_abi"] < 4
        or value["seccomp_mode"] != "classic_bpf_errno_eperm"
        or value["denied_metadata_syscalls_x86_64"]
        != list(CP0_DENIED_METADATA_SYSCALLS_X86_64)
        or runtime_platform["system"] != "Linux"
        or value["kernel"] != runtime_platform["release"]
        or value["machine"] != runtime_platform["machine"]
        or value["machine"] != "x86_64"
        or rule["path"] != "/"
        or any(
            type(rule[field]) is not int or rule[field] < 0
            for field in ("dev", "inode")
        )
        or rule["rights"] != CP0_LANDLOCK_READ_EXEC_RIGHTS
        or value["default_filesystem_rights"] != "read_execute_only"
        or value["writable_directories"] != []
        or value["metadata_mutation_policy"]
        != "globally_denied_by_seccomp"
    ):
        raise RuntimeError("protected provider evidence policy differs")
    return value


def _validate_paper_write_probe(
    value: object, paper_path: str
) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "operation",
        "target_path",
        "denied",
        "errno",
    }:
        raise RuntimeError("protected paper write probe schema differs")
    if (
        value["operation"] != "open(O_WRONLY|O_NOFOLLOW)"
        or value["target_path"] != paper_path
        or value["denied"] is not True
        or type(value["errno"]) is not int
        or value["errno"] not in {errno.EACCES, errno.EPERM}
    ):
        raise RuntimeError("protected paper write probe policy differs")
    return value


def _require_runtime_executable_current(
    executable: str, expected_sha256: str
) -> None:
    actual_sha256 = sha256_bytes(
        _read_absolute_regular_nofollow(
            executable, "runtime python executable"
        )
    )
    if actual_sha256 != expected_sha256:
        fail(
            "G0_RUNTIME_IDENTITY_CHANGED",
            "runtime Python bytes changed before protected child launch",
        )


def _external_path(external_dir: str, name: str) -> str:
    root = _canonical_absolute_text(external_dir, "external_dir")
    return f"{root}/{name}"


def _stage_bundle(stage: Iterable[Publication]) -> dict[str, object]:
    return {
        "publications": [
            {"evidence": publication.evidence.to_dict(), "transcript": publication.transcript}
            for publication in stage
        ]
    }


def bootstrap_g0_cp0(
    *,
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    external_dir: str,
    project_id: str,
    owner_seed: str,
    source_roots: Mapping[str, str],
    paper_path: str,
    protected_executable: str,
    provider_path: str,
    comparison_tool: str,
    validator_path: str,
    timeout_seconds: int = 1800,
) -> G0BootstrapResult:
    """Create exactly one fresh V6 strict run and seal its CP0 stage.

    Callers must complete preflight and use a previously unused canonical run
    ID.  On any failure the created run remains a failed, non-retryable audit
    artifact; this function never adopts or overwrites it.
    """

    sealed_run_id = validate_sealed_run_id(sealed_run_id)
    strict_parent = _canonical_absolute_text(strict_parent, "strict_parent")
    run_root = _canonical_absolute_text(run_root, "run_root")
    if run_root != f"{strict_parent}/{sealed_run_id}":
        fail("RUN_BINDING_MISMATCH", "run root must be the explicit strict child")
    if not isinstance(project_id, str) or not project_id:
        fail("G0_PROJECT_INVALID", "project_id must be nonempty")
    if not isinstance(owner_seed, str) or not owner_seed:
        fail("G0_OWNER_INVALID", "owner_seed must be nonempty")
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        fail("G0_TIMEOUT_INVALID", "timeout must be a positive integer")

    _require_run_root_absent(run_root)
    protected_executable = _pin_repository_path(
        protected_executable,
        "scripts/strict_run/g0_protected.py",
        "protected_executable",
    )
    provider_path = _pin_repository_path(
        provider_path, "scripts/g0/provider.py", "provider_path"
    )
    comparison_tool = _pin_repository_path(
        comparison_tool,
        "scripts/g0/compare_manifests.py",
        "comparison_tool",
    )
    validator_path = _pin_repository_path(
        validator_path, "src/provenance/strict_v6.py", "validator_path"
    )
    roots = _validate_source_roots(source_roots)
    paper_path = _canonical_absolute_text(paper_path, "paper_path")
    _preflight_strict_protected_boundaries(
        strict_parent=strict_parent,
        run_root=run_root,
        roots=roots,
        paper_path=paper_path,
    )
    external_dir = _preflight_external_directory(
        external_dir,
        strict_parent=strict_parent,
        run_root=run_root,
        roots=roots,
        paper_path=paper_path,
    )
    mount_boundary = _capture_mount_boundary(
        (
            strict_parent,
            external_dir,
            *(row["path"] for row in roots),
            os.path.dirname(paper_path),
            _repository_root(),
        ),
        (strict_parent, external_dir),
    )
    repository_tools = (
        (protected_executable, "protected_executable"),
        (provider_path, "provider_path"),
        (comparison_tool, "comparison_tool"),
        (validator_path, "validator_path"),
    )
    for tool_path, tool_label in repository_tools:
        _validate_repository_tool_mount(
            tool_path, tool_label, mount_boundary.mount_id
        )
    executable_bytes = _read_absolute_regular_nofollow(
        protected_executable,
        "protected_executable",
        expected_mount_id=mount_boundary.mount_id,
    )
    provider_bytes = _trusted_provider_bytes(
        provider_path, expected_mount_id=mount_boundary.mount_id
    )
    comparison_bytes = _read_absolute_regular_nofollow(
        comparison_tool,
        "comparison_tool",
        expected_mount_id=mount_boundary.mount_id,
    )
    validator_bytes = _read_absolute_regular_nofollow(
        validator_path,
        "validator_path",
        expected_mount_id=mount_boundary.mount_id,
    )
    _revalidate_mount_boundary(mount_boundary)
    _require_run_root_absent(run_root)
    child_output_parents = _prepare_child_output_parents(
        external_dir, expected_mount_id=mount_boundary.mount_id
    )
    created_directories = tuple(child_output_parents.values())
    _revalidate_mount_boundary(mount_boundary, created_directories)
    executable_sha256 = sha256_bytes(executable_bytes)
    provider_sha256 = sha256_bytes(provider_bytes)
    policy = default_writer_policy()

    g0_config = {
        "schema": G0_CONFIG_SCHEMA,
        "sealed_run_id": sealed_run_id,
        "run_root": run_root,
        "provider_path": f"{run_root}/frozen/provider/provider.py",
        "provider_sha256": provider_sha256,
        "g0_executable_sha256": executable_sha256,
        "source_roots": roots,
        "paper_path": paper_path,
    }
    config_bytes = canonical_json(g0_config)
    runtime = _runtime_identity()
    runtime_python = runtime["python"]
    if (
        not isinstance(runtime_python, dict)
        or not isinstance(runtime_python.get("sha256"), str)
    ):
        raise RuntimeError("runtime identity is invalid")
    runtime_executable_sha256 = runtime_python["sha256"]
    runtime_executable = _canonical_absolute_text(
        os.path.realpath(sys.executable), "runtime python executable"
    )
    child_environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", "/tmp"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    environment_allowlist = {
        "schema": G0_ENVIRONMENT_SCHEMA,
        "environment": child_environment,
        "environment_sha256": sha256_bytes(canonical_json(child_environment)),
    }
    canonical_argv = _reservation_argv(
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
        external_dir=external_dir,
        project_id=project_id,
        owner_seed=owner_seed,
        roots=roots,
        paper_path=paper_path,
        protected_executable=protected_executable,
        provider_path=provider_path,
        comparison_tool=comparison_tool,
        validator_path=validator_path,
        timeout_seconds=timeout_seconds,
    )
    writers = {
        "reservation": WriterIdentity(
            "run_reservation", f"reservation:{sealed_run_id}", "run_reservation-writer"
        ),
        "acceptance": WriterIdentity(
            "reservation_acceptance", f"acceptance:{sealed_run_id}", "reservation_acceptance-writer"
        ),
        "g0": WriterIdentity("g0", f"g0:{sealed_run_id}", "g0-writer"),
        "controller": WriterIdentity(
            "checkpoint_controller", f"checkpoint:{sealed_run_id}", "checkpoint_controller-writer"
        ),
    }
    inputs = ReservationInputs(
        project_id=project_id,
        owner_seed=owner_seed,
        writer=writers["reservation"],
        tool_sha256=executable_sha256,
        configuration_sha256=sha256_bytes(config_bytes),
        runtime_sha256=sha256_bytes(canonical_json(runtime)),
        environment_path_policy_sha256=sha256_bytes(canonical_json(environment_allowlist)),
        canonical_argv=canonical_argv,
    )
    external_paths = {
        "reservation_transcript": _external_path(external_dir, "reservation-transcript.json"),
        "acceptance_transcript": _external_path(external_dir, "acceptance-transcript.json"),
        "cp0_stage_bundle": _external_path(external_dir, "cp0-stage-bundle.json"),
        "producer_binding": _external_path(external_dir, "producer-binding.json"),
        "source_stdout": f"{child_output_parents['source_stdout']}/result.json",
        "source_execution": f"{child_output_parents['source_execution']}/execution.json",
        "paper_stdout": f"{child_output_parents['paper_stdout']}/result.json",
        "paper_execution": f"{child_output_parents['paper_execution']}/execution.json",
        "cp0_transcript": _external_path(external_dir, "cp0-transcript.json"),
        "cp0_stage_transcripts": _external_path(external_dir, "cp0-stage-transcripts.json"),
    }

    _revalidate_mount_boundary(mount_boundary, created_directories)
    _require_run_root_absent(run_root)
    with ExternalOutputReservation.create(
        external_paths["reservation_transcript"],
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
        expected_mount_id=mount_boundary.mount_id,
    ) as reservation_output:
        _revalidate_mount_boundary(mount_boundary, created_directories)
        _require_run_root_absent(run_root)
        reservation = reserve_strict_run(
            strict_parent,
            sealed_run_id,
            run_root,
            inputs,
            policy=policy,
            expected_parent_mount_id=mount_boundary.mount_id,
        )
        run_binding = _run_binding_from_reservation(reservation.payload)
        active_directories = (*created_directories, run_root)
        _revalidate_mount_boundary(mount_boundary, active_directories)
        _assert_bound_run_current(
            strict_parent, sealed_run_id, run_root, run_binding
        )
        reservation_transcript_evidence = reservation_output.publish_json(
            reservation.transcript,
            expected_run_binding=run_binding,
        )

    _revalidate_mount_boundary(mount_boundary, active_directories)
    _assert_bound_run_current(
        strict_parent, sealed_run_id, run_root, run_binding
    )
    external_evidence: dict[str, ExternalOutputEvidence] = {
        external_paths["reservation_transcript"]: reservation_transcript_evidence
    }
    with (
        ExternalOutputReservation.create_for_existing_run(
            external_paths["acceptance_transcript"],
            strict_parent=strict_parent,
            sealed_run_id=sealed_run_id,
            run_root=run_root,
            expected_mount_id=mount_boundary.mount_id,
            expected_run_binding=run_binding,
        ) as acceptance_output,
        ExternalTranscriptRegistry(
            [
                _pinned_external_record(
                    reservation_transcript_evidence,
                    expected_mount_id=mount_boundary.mount_id,
                )
            ],
            strict_parent=strict_parent,
            sealed_run_id=sealed_run_id,
            run_root=run_root,
            run_binding=run_binding,
        ) as reservation_registry,
    ):
        acceptance = accept_reservation(
            strict_parent,
            sealed_run_id,
            run_root,
            reservation_registry,
            writers["acceptance"],
            reservation_transcript_sha256=reservation.publication.evidence.publication_transcript_sha256,
            policy=policy,
            run_binding=run_binding,
        )
        acceptance_transcript_evidence = acceptance_output.publish_json(
            acceptance.transcript
        )
    external_evidence[
        external_paths["acceptance_transcript"]
    ] = acceptance_transcript_evidence

    stage: list[Publication] = [reservation.publication, acceptance.publication]
    g0_publisher = StagePublisher(
        strict_parent,
        sealed_run_id,
        run_root,
        writers["g0"],
        policy,
        run_binding=run_binding,
    )
    stage.append(
        publish_owner_binding(
            strict_parent,
            sealed_run_id,
            run_root,
            owner_seed,
            project_id,
            writers["g0"],
            policy=policy,
            run_binding=run_binding,
        )
    )
    stage.extend(
        StagePublisher(
            strict_parent,
            sealed_run_id,
            run_root,
            writers["controller"],
            policy,
            allow_reserved_paths=True,
            run_binding=run_binding,
        ).ensure_directory("checkpoints")
    )

    roles: dict[str, list[dict[str, str]]] = {
        "g0_executable": [],
        "provider": [],
        "configuration": [],
        "pre_argv": [],
        "post_argv": [],
        "comparison_tool": [],
        "runtime_identity": [],
        "validators": [],
        "environment_allowlist": [],
        "bundle_lock": [],
        "envelope_evidence": [],
        "writer_policy": [],
    }
    frozen_regulars: list[Publication] = []

    def freeze(role: str, relative: str, payload: bytes, *, context: dict[str, object] | None = None) -> Publication:
        publication = _publish(g0_publisher, relative, payload, stage, context=context)
        roles[role].append(_artifact_ref(publication))
        frozen_regulars.append(publication)
        return publication

    frozen_executable = f"{run_root}/frozen/g0_executable/g0_protected.py"
    frozen_config = f"{run_root}/frozen/configuration/g0-config.json"
    source_argv = [
        runtime_executable,
        "-B",
        frozen_executable,
        "--config",
        frozen_config,
        "--mode",
        "source-pre",
        "--producer-binding-stdin",
    ]
    paper_argv = [
        runtime_executable,
        "-B",
        frozen_executable,
        "--config",
        frozen_config,
        "--mode",
        "paper-pre",
        "--producer-binding-stdin",
    ]
    audit_argv = [
        runtime_executable,
        "-B",
        frozen_executable,
        "--config",
        frozen_config,
        "--mode",
        "verify-envelope",
    ]
    frozen_executable_publication = freeze(
        "g0_executable",
        "frozen/g0_executable/g0_protected.py",
        executable_bytes,
    )
    producer_binding_path = external_paths["producer_binding"]
    frozen_provider_publication = freeze(
        "provider", "frozen/provider/provider.py", provider_bytes
    )
    frozen_config_publication = freeze(
        "configuration",
        "frozen/configuration/g0-config.json",
        config_bytes,
    )
    freeze(
        "pre_argv",
        "frozen/pre_argv/argv.json",
        canonical_json(
            {
                "schema": G0_ARGUMENT_SCHEMA,
                "mode": "source-pre",
                "argv": source_argv,
            }
        ),
    )
    freeze(
        "post_argv",
        "frozen/post_argv/argv.json",
        canonical_json(
            {
                "schema": G0_ARGUMENT_SCHEMA,
                "mode": "paper-pre",
                "argv": paper_argv,
            }
        ),
    )
    freeze("comparison_tool", "frozen/comparison_tool/compare_manifests.py", comparison_bytes)
    freeze("runtime_identity", "frozen/runtime_identity/runtime.json", canonical_json(runtime))
    freeze("validators", "frozen/validators/strict_v6.py", validator_bytes)
    freeze(
        "environment_allowlist",
        "frozen/environment_allowlist/environment.json",
        canonical_json(environment_allowlist),
    )
    freeze("writer_policy", "frozen/writer-policy.json", canonical_json(policy.to_dict()))

    _require_runtime_executable_current(
        runtime_executable, runtime_executable_sha256
    )
    _revalidate_mount_boundary(mount_boundary, active_directories)
    _assert_bound_run_current(
        strict_parent, sealed_run_id, run_root, run_binding
    )
    envelope_audit_capture = _protected_result(
        frozen_executable,
        frozen_config,
        "verify-envelope",
        timeout_seconds=timeout_seconds,
        argv=audit_argv,
        environment=child_environment,
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
        run_binding=run_binding,
        executable_publication=frozen_executable_publication,
        configuration_publication=frozen_config_publication,
        provider_publication=frozen_provider_publication,
        runtime_executable_sha256=runtime_executable_sha256,
    )
    envelope_audit = envelope_audit_capture.result
    required_audit = {
        "schema",
        "mode",
        "sealed_run_id",
        "run_root",
        "python_dont_write_bytecode",
        "pythonhashseed",
        "provider_evidence",
        "paper_write_probe",
    }
    if (
        set(envelope_audit) != required_audit
        or envelope_audit["schema"] != G0_RESULT_SCHEMA
        or envelope_audit["mode"] != "verify-envelope"
        or envelope_audit["sealed_run_id"] != sealed_run_id
        or envelope_audit["run_root"] != run_root
        or envelope_audit["python_dont_write_bytecode"] is not True
        or envelope_audit["pythonhashseed"] != "0"
        or envelope_audit_capture.argv != tuple(audit_argv)
        or envelope_audit_capture.environment != child_environment
        or envelope_audit_capture.stdout != canonical_json(envelope_audit)
        or envelope_audit_capture.stderr != b""
    ):
        raise RuntimeError("frozen provider envelope audit is incomplete")
    audit_provider = _validate_provider_evidence(
        envelope_audit["provider_evidence"], runtime["platform"]
    )
    audit_probe = _validate_paper_write_probe(
        envelope_audit["paper_write_probe"], paper_path
    )
    freeze(
        "runtime_identity",
        "frozen/runtime_identity/envelope-audit.json",
        canonical_json(envelope_audit),
    )

    bundle_components = [
        {
            "relative_path": publication.evidence.relative_path,
            "sha256": publication.evidence.sha256,
            "bytes": publication.evidence.size,
        }
        for publication in sorted(frozen_regulars, key=lambda item: item.evidence.relative_path)
    ]
    freeze(
        "bundle_lock",
        "frozen/bundle-lock.json",
        canonical_json(
            {
                "schema": G0_BUNDLE_SCHEMA,
                "sealed_run_id": sealed_run_id,
                "run_root": run_root,
                "components": bundle_components,
                "components_sha256": sha256_bytes(canonical_json(bundle_components)),
            }
        ),
    )
    acceptance_context = acceptance.publication.transcript["context"]
    if not isinstance(acceptance_context, dict):
        raise RuntimeError("acceptance publication context is invalid")
    envelope_payload = {
        "schema": G0_ENVELOPE_SCHEMA,
        "sealed_run_id": sealed_run_id,
        "run_root": run_root,
        "acceptance_transcript_sha256": acceptance.publication.evidence.publication_transcript_sha256,
    }
    envelope_publication = freeze(
        "envelope_evidence",
        "frozen/envelope_evidence/artifact.json",
        canonical_json(envelope_payload),
        context=acceptance_context,
    )
    producer_binding = {
        "schema": G0_PRODUCER_BINDING_SCHEMA,
        "acceptance_transcript": acceptance.publication.transcript,
        "envelope_artifact_evidence": envelope_publication.evidence.to_dict(),
        "envelope_transcript": envelope_publication.transcript,
        "envelope_payload": envelope_payload,
    }
    producer_binding_bytes = canonical_json(producer_binding)
    _revalidate_mount_boundary(mount_boundary, active_directories)
    _assert_bound_run_current(
        strict_parent, sealed_run_id, run_root, run_binding
    )
    with ExternalOutputReservation.create_for_existing_run(
        producer_binding_path,
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
        expected_mount_id=mount_boundary.mount_id,
        expected_run_binding=run_binding,
    ) as producer_binding_output:
        producer_binding_evidence = producer_binding_output.publish_bytes(
            producer_binding_bytes
        )
    external_evidence[producer_binding_path] = producer_binding_evidence

    _require_runtime_executable_current(
        runtime_executable, runtime_executable_sha256
    )
    _revalidate_mount_boundary(mount_boundary, active_directories)
    _assert_bound_run_current(
        strict_parent, sealed_run_id, run_root, run_binding
    )
    source_capture = _protected_result(
        frozen_executable,
        frozen_config,
        "source-pre",
        timeout_seconds=timeout_seconds,
        argv=source_argv,
        environment=child_environment,
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
        run_binding=run_binding,
        executable_publication=frozen_executable_publication,
        configuration_publication=frozen_config_publication,
        provider_publication=frozen_provider_publication,
        runtime_executable_sha256=runtime_executable_sha256,
        producer_binding=producer_binding_bytes,
    )
    source_result = source_capture.result
    if (
        set(source_result) != {"schema", "mode", "sealed_run_id", "run_root", "provider_evidence", "paper_write_probe", "producer_attestation", "records"}
        or source_result["sealed_run_id"] != sealed_run_id
        or source_result["run_root"] != run_root
        or not isinstance(source_result["records"], list)
        or source_result["provider_evidence"] != audit_provider
        or source_result["paper_write_probe"] != audit_probe
    ):
        raise RuntimeError("protected source result is invalid")
    source_records = validate_source_pre_records(source_result["records"])
    if source_records != source_result["records"]:
        raise RuntimeError("protected source result is not canonically normalized")
    source_context = _validate_producer_attestation(
        source_result["producer_attestation"],
        "source-pre",
        acceptance.publication,
        envelope_publication,
    )
    source_context = _publish_child_evidence(
        capture=source_capture,
        operation="source-pre",
        expected_argv=source_argv,
        expected_environment=child_environment,
        producer_binding_bytes=producer_binding_bytes,
        producer_attestation=source_context,
        stdout_path=external_paths["source_stdout"],
        execution_path=external_paths["source_execution"],
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
        executable_sha256=executable_sha256,
        runtime_executable_sha256=runtime_executable_sha256,
        configuration_sha256=sha256_bytes(config_bytes),
        provider_sha256=provider_sha256,
        expected_mount_id=mount_boundary.mount_id,
        run_binding=run_binding,
        evidence_sink=external_evidence,
    )
    source_publication = _publish(
        g0_publisher,
        "manifests/source-pre.jsonl",
        b"".join(canonical_json(record) for record in source_records),
        stage,
        context=source_context,
    )
    _require_runtime_executable_current(
        runtime_executable, runtime_executable_sha256
    )
    _revalidate_mount_boundary(mount_boundary, active_directories)
    _assert_bound_run_current(
        strict_parent, sealed_run_id, run_root, run_binding
    )
    paper_capture = _protected_result(
        frozen_executable,
        frozen_config,
        "paper-pre",
        timeout_seconds=timeout_seconds,
        argv=paper_argv,
        environment=child_environment,
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
        run_binding=run_binding,
        executable_publication=frozen_executable_publication,
        configuration_publication=frozen_config_publication,
        provider_publication=frozen_provider_publication,
        runtime_executable_sha256=runtime_executable_sha256,
        producer_binding=producer_binding_bytes,
    )
    paper_result = paper_capture.result
    if (
        set(paper_result) != {"schema", "mode", "sealed_run_id", "run_root", "provider_evidence", "paper_write_probe", "producer_attestation", "paper"}
        or paper_result["sealed_run_id"] != sealed_run_id
        or paper_result["run_root"] != run_root
        or not isinstance(paper_result["paper"], dict)
        or paper_result["provider_evidence"] != audit_provider
        or paper_result["paper_write_probe"] != audit_probe
        or paper_result["paper"].get("schema") != "experiments7-paper-pre/v6"
    ):
        raise RuntimeError("protected paper result is invalid")
    paper_payload = paper_result["paper"]
    required_paper = {
        "schema",
        "sealed_run_id",
        "run_root",
        "paper_path",
        "sha256",
        "size",
        "descriptor_identity",
        "parent_identity",
        "provider",
    }
    provider_identity_keys = (
        "provider",
        "landlock_abi",
        "seccomp_mode",
        "denied_metadata_syscalls_x86_64",
        "kernel",
        "machine",
    )
    expected_provider_identity = {
        key: audit_provider[key] for key in provider_identity_keys
    }
    if (
        set(paper_payload) != required_paper
        or paper_payload["sealed_run_id"] != sealed_run_id
        or paper_payload["run_root"] != run_root
        or paper_payload["paper_path"] != paper_path
        or paper_payload["provider"] != expected_provider_identity
    ):
        raise RuntimeError("protected paper payload binding differs")
    paper_context = _validate_producer_attestation(
        paper_result["producer_attestation"],
        "paper-pre",
        acceptance.publication,
        envelope_publication,
    )
    paper_context = _publish_child_evidence(
        capture=paper_capture,
        operation="paper-pre",
        expected_argv=paper_argv,
        expected_environment=child_environment,
        producer_binding_bytes=producer_binding_bytes,
        producer_attestation=paper_context,
        stdout_path=external_paths["paper_stdout"],
        execution_path=external_paths["paper_execution"],
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
        executable_sha256=executable_sha256,
        runtime_executable_sha256=runtime_executable_sha256,
        configuration_sha256=sha256_bytes(config_bytes),
        provider_sha256=provider_sha256,
        expected_mount_id=mount_boundary.mount_id,
        run_binding=run_binding,
        evidence_sink=external_evidence,
    )
    paper_publication = _publish(
        g0_publisher,
        "manifests/paper-pre.json",
        canonical_json(paper_result["paper"]),
        stage,
        context=paper_context,
    )

    if any(not references for references in roles.values()):
        raise RuntimeError("frozen role inventory is incomplete")
    canonical_roles = {
        role: sorted(references, key=lambda value: value["relative_path"])
        for role, references in roles.items()
    }
    all_refs = sorted(
        [reference for references in canonical_roles.values() for reference in references],
        key=lambda value: value["relative_path"],
    )
    inventory = {
        "schema": G0_INVENTORY_SCHEMA,
        "sealed_run_id": sealed_run_id,
        "run_root": run_root,
        "roles": canonical_roles,
        "frozen_artifact_count": len(all_refs),
        "frozen_artifacts_sha256": sha256_bytes(canonical_json(all_refs)),
        "source_pre": _artifact_ref(source_publication),
        "paper_pre": _artifact_ref(paper_publication),
    }
    _publish(g0_publisher, "frozen/inventory.json", canonical_json(inventory), stage)

    _revalidate_mount_boundary(mount_boundary, active_directories)
    _assert_bound_run_current(
        strict_parent, sealed_run_id, run_root, run_binding
    )
    with ExternalOutputReservation.create_for_existing_run(
        external_paths["cp0_stage_bundle"],
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
        expected_mount_id=mount_boundary.mount_id,
        expected_run_binding=run_binding,
    ) as stage_bundle_output:
        stage_bundle_evidence = stage_bundle_output.publish_json(
            _stage_bundle(stage)
        )
    external_evidence[
        external_paths["cp0_stage_bundle"]
    ] = stage_bundle_evidence
    with (
        ExternalOutputReservation.create_for_existing_run(
            external_paths["cp0_transcript"],
            strict_parent=strict_parent,
            sealed_run_id=sealed_run_id,
            run_root=run_root,
            expected_mount_id=mount_boundary.mount_id,
            expected_run_binding=run_binding,
        ) as checkpoint_output,
        ExternalOutputReservation.create_for_existing_run(
            external_paths["cp0_stage_transcripts"],
            strict_parent=strict_parent,
            sealed_run_id=sealed_run_id,
            run_root=run_root,
            expected_mount_id=mount_boundary.mount_id,
            expected_run_binding=run_binding,
        ) as stage_transcript_output,
        ExternalTranscriptRegistry(
            [
                _pinned_external_record(
                    stage_bundle_evidence,
                    expected_mount_id=mount_boundary.mount_id,
                    json_pointer=("publications", index, "transcript"),
                )
                for index in range(len(stage))
            ]
            + [
                _pinned_external_record(
                    external_evidence[external_paths[path]],
                    expected_mount_id=mount_boundary.mount_id,
                )
                for path in (
                    "source_stdout",
                    "source_execution",
                    "paper_stdout",
                    "paper_execution",
                )
            ],
            strict_parent=strict_parent,
            sealed_run_id=sealed_run_id,
            run_root=run_root,
            run_binding=run_binding,
        ) as registry,
    ):
        checkpoint = publish_checkpoint(
            strict_parent,
            sealed_run_id,
            run_root,
            0,
            stage,
            writers["controller"],
            policy,
            transcript_registry=registry,
            run_binding=run_binding,
        )
        checkpoint_output.publish_json(checkpoint.transcript)
        stage_transcript_output.publish_json(
            {"transcripts": [publication.transcript for publication in checkpoint.stage_publications]}
        )
    return G0BootstrapResult(
        sealed_run_id,
        strict_parent,
        run_root,
        checkpoint,
        tuple(stage),
        dict(external_paths),
    )
