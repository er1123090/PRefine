"""Descriptor-relative filesystem operations for strict-run code."""
from __future__ import annotations

import fcntl
import hashlib
import os
import stat
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Iterator

from .canonical import StrictRunError, fail, validate_absolute_path_text, validate_relative_path


DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC


@dataclass(frozen=True)
class RunHandle:
    sealed_run_id: str
    strict_parent: str
    run_root: str
    parent_fd: int
    root_fd: int

    def close(self) -> None:
        os.close(self.root_fd)
        os.close(self.parent_fd)

    def __enter__(self) -> "RunHandle":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


@dataclass(frozen=True)
class DirectoryDescriptorBinding:
    """Immutable directory identity captured before any later path reopen."""

    dev: int
    inode: int
    mode: int
    uid: int
    gid: int
    mount_id: int
    mount_namespace_dev: int
    mount_namespace_inode: int
    mount_namespace_link: str

    @classmethod
    def from_evidence(
        cls,
        identity: object,
        mount: object,
        label: str,
    ) -> "DirectoryDescriptorBinding":
        if not isinstance(identity, Mapping) or set(identity) != {
            "type",
            "dev",
            "inode",
            "mode",
            "uid",
            "gid",
        }:
            fail("RUN_BINDING_INVALID", f"{label} identity evidence is not exact")
        if identity["type"] != "directory":
            fail("RUN_BINDING_INVALID", f"{label} identity is not a directory")
        integer_fields = ("dev", "inode", "mode", "uid", "gid")
        if any(type(identity[field]) is not int for field in integer_fields):
            fail("RUN_BINDING_INVALID", f"{label} identity integers are invalid")
        if not isinstance(mount, Mapping):
            fail("RUN_BINDING_INVALID", f"{label} mount evidence is invalid")
        mount_id = mount.get("mount_id")
        namespace = mount.get("mount_namespace")
        if (
            type(mount_id) is not int
            or mount_id <= 0
            or not isinstance(namespace, Mapping)
            or set(namespace) != {"dev", "inode", "link"}
            or type(namespace["dev"]) is not int
            or type(namespace["inode"]) is not int
            or not isinstance(namespace["link"], str)
            or not namespace["link"]
        ):
            fail("RUN_BINDING_INVALID", f"{label} mount identity is invalid")
        return cls(
            dev=int(identity["dev"]),
            inode=int(identity["inode"]),
            mode=int(identity["mode"]),
            uid=int(identity["uid"]),
            gid=int(identity["gid"]),
            mount_id=mount_id,
            mount_namespace_dev=int(namespace["dev"]),
            mount_namespace_inode=int(namespace["inode"]),
            mount_namespace_link=str(namespace["link"]),
        )

    @classmethod
    def capture(cls, fd: int, label: str) -> "DirectoryDescriptorBinding":
        return cls.from_evidence(stable_identity(os.fstat(fd)), mount_identity(fd), label)

    def assert_fd(self, fd: int, label: str) -> None:
        current = stable_identity(os.fstat(fd))
        expected = {
            "type": "directory",
            "dev": self.dev,
            "inode": self.inode,
            "mode": self.mode,
            "uid": self.uid,
            "gid": self.gid,
        }
        if current != expected:
            fail("RUN_IDENTITY_SUBSTITUTION", f"{label} descriptor identity differs")
        current_mount = mount_identity(fd)
        namespace = current_mount.get("mount_namespace")
        if (
            current_mount.get("mount_id") != self.mount_id
            or not isinstance(namespace, Mapping)
            or namespace.get("dev") != self.mount_namespace_dev
            or namespace.get("inode") != self.mount_namespace_inode
            or namespace.get("link") != self.mount_namespace_link
        ):
            fail("MOUNT_SUBSTITUTION", f"{label} descriptor mount differs")


@dataclass(frozen=True)
class RunDescriptorBinding:
    """Exact reservation-created parent/root identity for all later operations."""

    parent: DirectoryDescriptorBinding
    root: DirectoryDescriptorBinding

    @classmethod
    def from_evidence(
        cls,
        *,
        parent_identity: object,
        parent_mount: object,
        root_identity: object,
        root_mount: object,
    ) -> "RunDescriptorBinding":
        binding = cls(
            DirectoryDescriptorBinding.from_evidence(
                parent_identity, parent_mount, "strict parent"
            ),
            DirectoryDescriptorBinding.from_evidence(
                root_identity, root_mount, "strict run root"
            ),
        )
        if (
            binding.parent.dev != binding.root.dev
            or binding.parent.mount_id != binding.root.mount_id
        ):
            fail("MOUNT_SUBSTITUTION", "bound strict parent and root mounts differ")
        return binding

    @classmethod
    def capture(cls, parent_fd: int, root_fd: int) -> "RunDescriptorBinding":
        binding = cls(
            DirectoryDescriptorBinding.capture(parent_fd, "strict parent"),
            DirectoryDescriptorBinding.capture(root_fd, "strict run root"),
        )
        if (
            binding.parent.dev != binding.root.dev
            or binding.parent.mount_id != binding.root.mount_id
        ):
            fail("MOUNT_SUBSTITUTION", "strict parent and root mounts differ")
        return binding

    def assert_fds(
        self,
        parent_fd: int,
        root_fd: int,
        *,
        strict_parent: str | None = None,
        run_root: str | None = None,
    ) -> None:
        self.parent.assert_fd(parent_fd, "strict parent")
        self.root.assert_fd(root_fd, "strict run root")
        if strict_parent is not None:
            assert_path_matches_fd(strict_parent, parent_fd)
        if run_root is not None:
            assert_path_matches_fd(run_root, root_fd)

    def assert_handle(self, handle: RunHandle) -> None:
        self.assert_fds(
            handle.parent_fd,
            handle.root_fd,
            strict_parent=handle.strict_parent,
            run_root=handle.run_root,
        )


def kind(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISDIR(mode):
        return "directory"
    if stat.S_ISLNK(mode):
        return "symlink"
    return "other"


def stable_identity(st: os.stat_result) -> dict[str, object]:
    result: dict[str, object] = {
        "type": kind(st.st_mode),
        "dev": st.st_dev,
        "inode": st.st_ino,
        "mode": stat.S_IMODE(st.st_mode),
        "uid": st.st_uid,
        "gid": st.st_gid,
    }
    if stat.S_ISREG(st.st_mode):
        result["size"] = st.st_size
    return result


def descriptor_evidence(fd: int) -> dict[str, object]:
    return {
        "identity": stable_identity(os.fstat(fd)),
        "status_flags": fcntl.fcntl(fd, fcntl.F_GETFL),
        "descriptor_flags": fcntl.fcntl(fd, fcntl.F_GETFD),
    }


def _canonical_fd_path(fd: int) -> str:
    value = os.readlink(f"/proc/self/fd/{fd}")
    if value.endswith(" (deleted)"):
        fail("NAMESPACE_SUBSTITUTION", "opened directory was removed")
    return value


def open_absolute_directory(path: str) -> int:
    path = validate_absolute_path_text(path, "directory")
    fd = os.open("/", DIR_FLAGS)
    try:
        for component in path.split("/")[1:]:
            next_fd = os.open(component, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        if _canonical_fd_path(fd) != path or os.path.realpath(path) != path:
            fail("NONCANONICAL_PATH", "directory descriptor does not match canonical path")
        return fd
    except Exception as exc:
        os.close(fd)
        if isinstance(exc, StrictRunError):
            raise
        raise StrictRunError(
            "UNSAFE_DIRECTORY", "directory traversal encountered a missing or unsafe component"
        ) from exc


def assert_path_matches_fd(path: str, fd: int) -> None:
    try:
        current = os.lstat(path)
    except OSError as exc:
        raise StrictRunError("NAMESPACE_SUBSTITUTION", "bound path disappeared") from exc
    opened = os.fstat(fd)
    if not stat.S_ISDIR(current.st_mode) or (current.st_dev, current.st_ino) != (
        opened.st_dev,
        opened.st_ino,
    ):
        fail("NAMESPACE_SUBSTITUTION", "path and opened descriptor identities differ")
    if _canonical_fd_path(fd) != path or os.path.realpath(path) != path:
        fail("NAMESPACE_SUBSTITUTION", "path canonical identity changed")


def validate_run_root_text(
    strict_parent: str, sealed_run_id: str, run_root: str
) -> tuple[str, str]:
    strict_parent = validate_absolute_path_text(strict_parent, "strict_parent")
    run_root = validate_absolute_path_text(run_root, "run_root")
    if run_root != f"{strict_parent}/{sealed_run_id}":
        fail("RUN_BINDING_MISMATCH", "run_root is not the exact explicit strict child")
    return strict_parent, run_root


def open_run_handle(strict_parent: str, sealed_run_id: str, run_root: str) -> RunHandle:
    strict_parent, run_root = validate_run_root_text(strict_parent, sealed_run_id, run_root)
    parent_fd = open_absolute_directory(strict_parent)
    try:
        root_fd = os.open(sealed_run_id, DIR_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        os.close(parent_fd)
        raise StrictRunError(
            "UNSAFE_RUN_ROOT", "strict run root is missing, symlinked, or not a directory"
        ) from exc
    try:
        assert_path_matches_fd(strict_parent, parent_fd)
        assert_path_matches_fd(run_root, root_fd)
        if os.fstat(parent_fd).st_dev != os.fstat(root_fd).st_dev:
            fail("MOUNT_SUBSTITUTION", "strict child changed filesystem device")
        return RunHandle(sealed_run_id, strict_parent, run_root, parent_fd, root_fd)
    except Exception:
        os.close(root_fd)
        os.close(parent_fd)
        raise


def open_bound_run_handle(
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    binding: RunDescriptorBinding,
) -> RunHandle:
    if not isinstance(binding, RunDescriptorBinding):
        fail("RUN_BINDING_INVALID", "run descriptor binding is not typed")
    handle = open_run_handle(strict_parent, sealed_run_id, run_root)
    try:
        binding.assert_handle(handle)
        return handle
    except Exception:
        handle.close()
        raise


def open_relative_directory(root_fd: int, relative: str) -> int:
    relative = validate_relative_path(relative)
    fd = os.dup(root_fd)
    try:
        for component in relative.split("/"):
            next_fd = os.open(component, DIR_FLAGS, dir_fd=fd)
            os.close(fd)
            fd = next_fd
        return fd
    except Exception:
        os.close(fd)
        raise


def open_parent_directory(root_fd: int, relative: str) -> tuple[int, str]:
    relative = validate_relative_path(relative)
    parts = relative.split("/")
    if len(parts) == 1:
        return os.dup(root_fd), parts[0]
    return open_relative_directory(root_fd, "/".join(parts[:-1])), parts[-1]


def lstat_relative(root_fd: int, relative: str) -> os.stat_result:
    parent_fd, name = open_parent_directory(root_fd, relative)
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    finally:
        os.close(parent_fd)


def exists_relative(root_fd: int, relative: str) -> bool:
    try:
        lstat_relative(root_fd, relative)
    except FileNotFoundError:
        return False
    return True


def read_regular_at(
    root_fd: int, relative: str, *, limit: int = 32 * 1024 * 1024
) -> tuple[bytes, os.stat_result]:
    parent_fd, name = open_parent_directory(root_fd, relative)
    try:
        fd = os.open(name, READ_FLAGS, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            fail("NOT_REGULAR", "artifact is not a regular file")
        chunks: list[bytes] = []
        total = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            total += len(block)
            if total > limit:
                fail("ARTIFACT_TOO_LARGE", "artifact exceeds bounded reader limit")
            chunks.append(block)
        after = os.fstat(fd)
        if stable_identity(before) != stable_identity(after) or total != after.st_size:
            fail("ARTIFACT_MUTATED", "artifact changed during read")
        return b"".join(chunks), after
    finally:
        os.close(fd)


def hash_regular_at(root_fd: int, relative: str) -> tuple[str, int, os.stat_result]:
    parent_fd, name = open_parent_directory(root_fd, relative)
    try:
        fd = os.open(name, READ_FLAGS, dir_fd=parent_fd)
    finally:
        os.close(parent_fd)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            fail("NOT_REGULAR", "artifact is not regular")
        digest = hashlib.sha256()
        total = 0
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            total += len(block)
            digest.update(block)
        after = os.fstat(fd)
        if stable_identity(before) != stable_identity(after) or total != after.st_size:
            fail("ARTIFACT_MUTATED", "artifact changed during hashing")
        return digest.hexdigest(), total, after
    finally:
        os.close(fd)


def mount_identity(fd: int) -> dict[str, object]:
    mount_id: int | None = None
    with open(f"/proc/self/fdinfo/{fd}", "r", encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("mnt_id:"):
                mount_id = int(line.split(":", 1)[1].strip())
                break
    if mount_id is None:
        fail("MOUNT_ID_UNAVAILABLE", "descriptor mount ID is unavailable")
    selected: dict[str, object] | None = None
    with open("/proc/self/mountinfo", "r", encoding="utf-8") as stream:
        for line in stream:
            left, separator, right = line.rstrip("\n").partition(" - ")
            fields = left.split()
            if not separator or len(fields) < 6 or int(fields[0]) != mount_id:
                continue
            right_fields = right.split()
            selected = {
                "mount_id": mount_id,
                "parent_mount_id": int(fields[1]),
                "major_minor": fields[2],
                "mount_root": fields[3],
                "mount_point": fields[4],
                "mount_options": fields[5].split(","),
                "filesystem_type": right_fields[0],
                "mount_source": right_fields[1] if len(right_fields) > 1 else "",
                "super_options": right_fields[2].split(",") if len(right_fields) > 2 else [],
            }
            break
    if selected is None:
        fail("MOUNT_ID_UNAVAILABLE", "descriptor mount entry is unavailable")
    namespace_st = os.stat("/proc/self/ns/mnt", follow_symlinks=True)
    selected["mount_namespace"] = {
        "dev": namespace_st.st_dev,
        "inode": namespace_st.st_ino,
        "link": os.readlink("/proc/self/ns/mnt"),
    }
    return selected


def process_identity() -> dict[str, object]:
    status: dict[str, str] = {}
    with open("/proc/self/status", "r", encoding="utf-8") as stream:
        for line in stream:
            key, separator, value = line.partition(":")
            if separator and key in {
                "CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb", "NoNewPrivs"
            }:
                status[key] = value.strip()
    return {
        "effective_uid": os.geteuid(),
        "effective_gid": os.getegid(),
        "supplementary_groups": sorted(os.getgroups()),
        "namespaces": {
            name: os.readlink(f"/proc/self/ns/{name}") for name in ("mnt", "user", "pid")
        },
        "capability_state": status,
    }


def iter_tree(root_fd: int) -> Iterator[tuple[str, os.stat_result]]:
    def visit(directory_fd: int, prefix: str) -> Iterator[tuple[str, os.stat_result]]:
        for name in sorted(os.listdir(directory_fd), key=os.fsencode):
            relative = name if not prefix else f"{prefix}/{name}"
            st = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            yield relative, st
            if stat.S_ISDIR(st.st_mode):
                child_fd = os.open(name, DIR_FLAGS, dir_fd=directory_fd)
                try:
                    yield from visit(child_fd, relative)
                finally:
                    os.close(child_fd)
    yield from visit(root_fd, "")


def assert_regular_or_directory(st: os.stat_result) -> None:
    if not (stat.S_ISREG(st.st_mode) or stat.S_ISDIR(st.st_mode)):
        fail("UNSAFE_ARTIFACT_TYPE", "strict tree contains a symlink or special file")
