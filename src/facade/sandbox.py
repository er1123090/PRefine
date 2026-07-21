"""Linux-only, fail-closed confinement for facade subprocesses.

The runner deliberately does not emulate a sandbox with path checks.  It loads
an in-kernel Landlock ruleset, a libseccomp filter, and ``no_new_privs`` in the
child immediately before ``execve``.  A child must also demonstrate that a
write outside its declared output inode set, a network socket, and a fork are
all denied.  Missing or unobservable enforcement is an execution blocker.
"""

from __future__ import annotations

import ctypes
import ctypes.util
from dataclasses import dataclass
import errno
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import resource
import stat
from typing import Any, Callable, Iterable

from .selection import canonical_bytes, sha256_bytes


SANDBOX_POLICY_SCHEMA = "experiments7-linux-sandbox-policy/v1"
SANDBOX_EVIDENCE_SCHEMA = "experiments7-linux-sandbox-evidence/v2"
RUNTIME_PROOF_ENVIRONMENT_SCHEMA = "experiments7-runtime-proof-environment/v2"
RUNTIME_THREAT_MODEL_SCHEMA = "experiments7-runtime-threat-model/v2"
RUNTIME_PROOF_ENVIRONMENT_UNAVAILABLE = "RUNTIME_PROOF_ENVIRONMENT_UNAVAILABLE"

RUNTIME_THREAT_MODEL: dict[str, Any] = {
    "schema": RUNTIME_THREAT_MODEL_SCHEMA,
    "protected_assets": [
        "runtime-closure",
        "published-snapshots",
        "bound-inputs",
        "golden-receipts",
    ],
    "adversary_classes": [
        "runtime-subprocess",
        "same-effective-uid-host-process",
        "inherited-descriptor-holder",
        "privileged-helper",
    ],
    "required_controls": {
        "active_capabilities": "empty",
        "cooperating_subjects": "externally-confined-or-absent",
        "descriptor_inheritance": "exact-allowlist",
        "mount_boundary": "protected-assets-read-only",
        "no_new_privs": True,
        "runtime_subject": "privilege-separated-from-controller",
        "user_and_mount_namespaces": "externally-bound",
    },
}
RUNTIME_THREAT_MODEL_SHA256 = sha256_bytes(canonical_bytes(RUNTIME_THREAT_MODEL))

_CAPABILITY_HEX = re.compile(r"^[0-9a-f]{16}$")
_NAMESPACE_ID = re.compile(r"^(?:user|mnt):\[[1-9][0-9]*\]$")
_MOUNT_ESCAPE = re.compile(r"\\([0-7]{3})")

DEFAULT_SANDBOX_POLICY: dict[str, Any] = {
    "schema": SANDBOX_POLICY_SCHEMA,
    "backend": "landlock+seccomp",
    "filesystem_write_policy": "declared-existing-output-inodes-only",
    "network_policy": "deny-all",
    "child_process_policy": "deny-all",
    "no_new_privs": True,
    "umask": "077",
    "locale": "C",
    "timezone": "UTC",
    "live_network": False,
}

PR_SET_NO_NEW_PRIVS = 38
PR_GET_NO_NEW_PRIVS = 39
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1

LANDLOCK_ACCESS_FS_WRITE_FILE = 1 << 1
LANDLOCK_ACCESS_FS_REMOVE_DIR = 1 << 4
LANDLOCK_ACCESS_FS_REMOVE_FILE = 1 << 5
LANDLOCK_ACCESS_FS_MAKE_CHAR = 1 << 6
LANDLOCK_ACCESS_FS_MAKE_DIR = 1 << 7
LANDLOCK_ACCESS_FS_MAKE_REG = 1 << 8
LANDLOCK_ACCESS_FS_MAKE_SOCK = 1 << 9
LANDLOCK_ACCESS_FS_MAKE_FIFO = 1 << 10
LANDLOCK_ACCESS_FS_MAKE_BLOCK = 1 << 11
LANDLOCK_ACCESS_FS_MAKE_SYM = 1 << 12
LANDLOCK_ACCESS_FS_REFER = 1 << 13
LANDLOCK_ACCESS_FS_TRUNCATE = 1 << 14

SCMP_ACT_ALLOW = 0x7FFF0000
SCMP_ACT_ERRNO = 0x00050000

_DENIED_SYSCALLS = (
    # Network, including Unix-domain escape/cooperating-helper channels.
    "socket", "socketpair", "connect", "bind", "listen", "accept", "accept4",
    "sendto", "sendmsg", "sendmmsg", "recvmsg", "recvmmsg", "shutdown",
    # New process/thread creation after the one exact exec target is launched.
    "clone", "clone3", "fork", "vfork",
    # Namespace, mount, tracing, module, keyring, and privileged-kernel surfaces.
    "unshare", "setns", "mount", "umount2", "pivot_root", "move_mount",
    "open_tree", "fsopen", "fsmount", "fspick", "mount_setattr", "ptrace",
    "process_vm_writev", "bpf", "perf_event_open", "keyctl", "add_key",
    "request_key", "open_by_handle_at", "name_to_handle_at", "init_module",
    "finit_module", "delete_module", "kexec_load", "kexec_file_load", "reboot",
    "swapon", "swapoff", "setuid", "setgid", "setreuid", "setregid",
    "setresuid", "setresgid", "setfsuid", "setfsgid", "capset",
    # Metadata and namespace mutation.  Landlock does not mediate all of these.
    "chmod", "fchmod", "fchmodat", "fchmodat2", "chown", "fchown", "fchownat",
    "lchown", "setxattr", "lsetxattr", "fsetxattr", "removexattr",
    "lremovexattr", "fremovexattr", "utime", "utimes", "futimesat", "utimensat",
    # Defense in depth for names: Landlock remains the path-aware primary gate.
    "rename", "renameat", "renameat2", "link", "linkat", "symlink", "symlinkat",
    "unlink", "unlinkat",
)


class SandboxUnavailable(RuntimeError):
    """Raised when a required kernel control is missing or not observable."""

    def __init__(self, code: str, detail: Any = None):
        super().__init__(code)
        self.code = code
        self.detail = detail


class _LandlockRulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _LandlockPathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


@dataclass(frozen=True)
class SandboxBackend:
    policy_sha256: str
    landlock_abi: int
    seccomp_library: str
    seccomp: ctypes.CDLL


@dataclass(frozen=True)
class DeclaredOutputInode:
    """Exact pre-created regular inode that the child may write."""

    path: Path
    device: int
    inode: int


def sandbox_policy_sha256(policy: dict[str, Any] | None = None) -> str:
    return sha256_bytes(canonical_bytes(policy or DEFAULT_SANDBOX_POLICY))


def _syscall_numbers() -> tuple[int, int, int]:
    machine = platform.machine().lower()
    if machine in {"x86_64", "amd64", "aarch64", "arm64", "riscv64"}:
        return 444, 445, 446
    raise SandboxUnavailable("SANDBOX_ARCHITECTURE_UNSUPPORTED", machine)


def _libc() -> ctypes.CDLL:
    library = ctypes.CDLL(None, use_errno=True)
    library.syscall.restype = ctypes.c_long
    library.prctl.restype = ctypes.c_int
    return library


def _landlock_abi(libc: ctypes.CDLL) -> int:
    create_ruleset, _, _ = _syscall_numbers()
    ctypes.set_errno(0)
    result = libc.syscall(
        create_ruleset,
        ctypes.c_void_p(),
        ctypes.c_size_t(0),
        ctypes.c_uint(LANDLOCK_CREATE_RULESET_VERSION),
    )
    if result < 1:
        raise SandboxUnavailable(
            "LANDLOCK_UNAVAILABLE", {"errno_name": errno.errorcode.get(ctypes.get_errno(), "UNKNOWN")}
        )
    return int(result)


def _load_seccomp() -> tuple[str, ctypes.CDLL]:
    name = ctypes.util.find_library("seccomp")
    if not name:
        raise SandboxUnavailable("SECCOMP_LIBRARY_UNAVAILABLE")
    library = ctypes.CDLL(name, use_errno=True)
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    library.seccomp_rule_add.argtypes = [
        ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint,
    ]
    library.seccomp_rule_add.restype = ctypes.c_int
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_load.restype = ctypes.c_int
    return name, library


def prepare_backend(policy: dict[str, Any] | None = None) -> SandboxBackend:
    selected = policy or DEFAULT_SANDBOX_POLICY
    if selected != DEFAULT_SANDBOX_POLICY:
        raise SandboxUnavailable("SANDBOX_POLICY_UNSUPPORTED")
    libc = _libc()
    abi = _landlock_abi(libc)
    name, seccomp = _load_seccomp()
    return SandboxBackend(
        policy_sha256=sandbox_policy_sha256(selected),
        landlock_abi=abi,
        seccomp_library=name,
        seccomp=seccomp,
    )


def _set_no_new_privs(libc: ctypes.CDLL) -> None:
    ctypes.set_errno(0)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise SandboxUnavailable(
            "NO_NEW_PRIVS_FAILED", {"errno_name": errno.errorcode.get(ctypes.get_errno(), "UNKNOWN")}
        )
    if libc.prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) != 1:
        raise SandboxUnavailable("NO_NEW_PRIVS_UNOBSERVABLE")


def _handled_landlock_rights(abi: int) -> int:
    rights = (
        LANDLOCK_ACCESS_FS_WRITE_FILE
        | LANDLOCK_ACCESS_FS_REMOVE_DIR
        | LANDLOCK_ACCESS_FS_REMOVE_FILE
        | LANDLOCK_ACCESS_FS_MAKE_CHAR
        | LANDLOCK_ACCESS_FS_MAKE_DIR
        | LANDLOCK_ACCESS_FS_MAKE_REG
        | LANDLOCK_ACCESS_FS_MAKE_SOCK
        | LANDLOCK_ACCESS_FS_MAKE_FIFO
        | LANDLOCK_ACCESS_FS_MAKE_BLOCK
        | LANDLOCK_ACCESS_FS_MAKE_SYM
    )
    if abi >= 2:
        rights |= LANDLOCK_ACCESS_FS_REFER
    if abi >= 3:
        rights |= LANDLOCK_ACCESS_FS_TRUNCATE
    return rights


def _apply_landlock(
    libc: ctypes.CDLL,
    abi: int,
    output_inodes: Iterable[DeclaredOutputInode],
) -> None:
    create_ruleset, add_rule, restrict_self = _syscall_numbers()
    handled = _handled_landlock_rights(abi)
    attribute = _LandlockRulesetAttr(handled_access_fs=handled)
    ctypes.set_errno(0)
    ruleset_fd = int(libc.syscall(
        create_ruleset, ctypes.byref(attribute), ctypes.sizeof(attribute), ctypes.c_uint(0)
    ))
    if ruleset_fd < 0:
        raise SandboxUnavailable(
            "LANDLOCK_RULESET_CREATE_FAILED",
            {"errno_name": errno.errorcode.get(ctypes.get_errno(), "UNKNOWN")},
        )
    try:
        allowed = LANDLOCK_ACCESS_FS_WRITE_FILE
        if abi >= 3:
            allowed |= LANDLOCK_ACCESS_FS_TRUNCATE
        seen: set[tuple[int, int]] = set()
        for binding in output_inodes:
            path = binding.path
            flags = os.O_CLOEXEC | getattr(os, "O_PATH", os.O_RDONLY)
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags)
            try:
                state = os.fstat(descriptor)
                identity = (state.st_dev, state.st_ino)
                if (
                    not stat.S_ISREG(state.st_mode)
                    or identity != (binding.device, binding.inode)
                    or identity in seen
                ):
                    raise SandboxUnavailable(
                        "SANDBOX_OUTPUT_IDENTITY_DRIFT", {"path": str(path)}
                    )
                seen.add(identity)
                rule = _LandlockPathBeneathAttr(
                    allowed_access=allowed, parent_fd=descriptor, reserved=0
                )
                ctypes.set_errno(0)
                result = libc.syscall(
                    add_rule,
                    ruleset_fd,
                    LANDLOCK_RULE_PATH_BENEATH,
                    ctypes.byref(rule),
                    ctypes.c_uint(0),
                )
                if result != 0:
                    raise SandboxUnavailable(
                        "LANDLOCK_RULE_ADD_FAILED",
                        {"errno_name": errno.errorcode.get(ctypes.get_errno(), "UNKNOWN")},
                    )
            finally:
                os.close(descriptor)
        ctypes.set_errno(0)
        if libc.syscall(restrict_self, ruleset_fd, ctypes.c_uint(0)) != 0:
            raise SandboxUnavailable(
                "LANDLOCK_RESTRICT_FAILED",
                {"errno_name": errno.errorcode.get(ctypes.get_errno(), "UNKNOWN")},
            )
    finally:
        os.close(ruleset_fd)


def _apply_seccomp(library: ctypes.CDLL) -> tuple[str, ...]:
    context = library.seccomp_init(SCMP_ACT_ALLOW)
    if not context:
        raise SandboxUnavailable("SECCOMP_INIT_FAILED")
    installed: list[str] = []
    action = SCMP_ACT_ERRNO | errno.EPERM
    try:
        for name in _DENIED_SYSCALLS:
            number = library.seccomp_syscall_resolve_name(name.encode("ascii"))
            if number < 0:
                continue
            result = library.seccomp_rule_add(context, action, number, 0)
            if result != 0:
                raise SandboxUnavailable(
                    "SECCOMP_RULE_FAILED", {"syscall": name, "result": result}
                )
            installed.append(name)
        for required in (
            "socket",
            "clone",
            "mount",
            "ptrace",
            "chmod",
            "chown",
            "setxattr",
            "utimensat",
            "rename",
            "link",
            "unlink",
        ):
            if required not in installed:
                raise SandboxUnavailable("SECCOMP_REQUIRED_RULE_MISSING", required)
        result = library.seccomp_load(context)
        if result != 0:
            raise SandboxUnavailable("SECCOMP_LOAD_FAILED", {"result": result})
    finally:
        library.seccomp_release(context)
    return tuple(installed)


def _status_fields() -> dict[str, str]:
    wanted = {
        "CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb", "NoNewPrivs", "Seccomp",
    }
    result: dict[str, str] = {}
    with open("/proc/self/status", "r", encoding="ascii") as handle:
        for line in handle:
            key, separator, value = line.partition(":")
            if separator and key in wanted:
                result[key] = value.strip()
    if set(result) != wanted:
        raise SandboxUnavailable("SANDBOX_STATUS_UNOBSERVABLE", sorted(wanted - set(result)))
    return result


def current_subject_identity(status: dict[str, str] | None = None) -> dict[str, Any]:
    """Return the exact identity fields bound across the fork boundary."""

    observed = status or _status_fields()
    return {
        "effective_uid": os.geteuid(),
        "effective_gid": os.getegid(),
        "supplementary_groups": sorted(os.getgroups()),
        "capability_inheritable": observed["CapInh"],
        "capability_permitted": observed["CapPrm"],
        "capability_effective": observed["CapEff"],
        "capability_ambient": observed["CapAmb"],
        "capability_bounding": observed["CapBnd"],
    }


def subject_identity_sha256(subject: dict[str, Any]) -> str:
    return sha256_bytes(canonical_bytes(subject))


def _exact_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_id_map(name: str) -> list[dict[str, int]]:
    try:
        payload = Path(f"/proc/self/{name}").read_text(encoding="ascii")
    except OSError as exc:
        raise SandboxUnavailable("RUNTIME_PROOF_ID_MAP_UNOBSERVABLE", name) from exc
    records: list[dict[str, int]] = []
    for line in payload.splitlines():
        fields = line.split()
        if len(fields) != 3 or any(not field.isascii() or not field.isdecimal() for field in fields):
            raise SandboxUnavailable("RUNTIME_PROOF_ID_MAP_INVALID", name)
        inside_id, outside_id, length = (int(field, 10) for field in fields)
        if length <= 0:
            raise SandboxUnavailable("RUNTIME_PROOF_ID_MAP_INVALID", name)
        records.append({
            "inside_id": inside_id,
            "outside_id": outside_id,
            "length": length,
        })
    if not records:
        raise SandboxUnavailable("RUNTIME_PROOF_ID_MAP_EMPTY", name)
    return records


def _namespace_id(name: str) -> str:
    try:
        value = os.readlink(f"/proc/self/ns/{name}")
    except OSError as exc:
        raise SandboxUnavailable("RUNTIME_PROOF_NAMESPACE_UNOBSERVABLE", name) from exc
    if _NAMESPACE_ID.fullmatch(value) is None:
        raise SandboxUnavailable("RUNTIME_PROOF_NAMESPACE_INVALID", {"name": name, "value": value})
    return value


def _descriptor_access_mode(flags: int) -> str:
    if hasattr(os, "O_PATH") and flags & os.O_PATH:
        return "path-only"
    access = flags & os.O_ACCMODE
    if access == os.O_RDONLY:
        return "read-only"
    if access == os.O_WRONLY:
        return "write-only"
    if access == os.O_RDWR:
        return "read-write"
    raise SandboxUnavailable("RUNTIME_PROOF_DESCRIPTOR_ACCESS_INVALID", access)


def _descriptor_table(allowed_inherited_fds: Iterable[int]) -> list[dict[str, Any]]:
    allowed = {int(value) for value in allowed_inherited_fds}
    records: list[dict[str, Any]] = []
    try:
        names = sorted(os.listdir("/proc/self/fd"), key=lambda value: int(value))
    except OSError as exc:
        raise SandboxUnavailable("RUNTIME_PROOF_DESCRIPTOR_TABLE_UNOBSERVABLE") from exc
    for name in names:
        if not name.isdecimal():
            raise SandboxUnavailable("RUNTIME_PROOF_DESCRIPTOR_TABLE_INVALID", name)
        descriptor = int(name, 10)
        try:
            status_flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
            descriptor_flags = fcntl.fcntl(descriptor, fcntl.F_GETFD)
            target = os.readlink(f"/proc/self/fd/{descriptor}")
        except OSError as exc:
            if exc.errno == errno.EBADF:
                continue
            raise SandboxUnavailable(
                "RUNTIME_PROOF_DESCRIPTOR_UNOBSERVABLE", descriptor
            ) from exc
        close_on_exec = bool(descriptor_flags & fcntl.FD_CLOEXEC)
        records.append({
            "fd": descriptor,
            "target": target,
            "access_mode": _descriptor_access_mode(status_flags),
            "close_on_exec": close_on_exec,
            "allowed_inherited": descriptor <= 2 or descriptor in allowed,
        })
    if not records:
        raise SandboxUnavailable("RUNTIME_PROOF_DESCRIPTOR_TABLE_EMPTY")
    return records


def _decode_mount_field(value: str) -> str:
    return _MOUNT_ESCAPE.sub(lambda match: chr(int(match.group(1), 8)), value)


def _mount_record(path: Path) -> dict[str, Any]:
    try:
        resolved = path.resolve(strict=True)
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SandboxUnavailable("RUNTIME_PROOF_MOUNT_UNOBSERVABLE", str(path)) from exc
    candidates: list[tuple[int, list[str], int]] = []
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            raise SandboxUnavailable("RUNTIME_PROOF_MOUNTINFO_INVALID")
        if separator < 6 or len(fields) <= separator + 3:
            raise SandboxUnavailable("RUNTIME_PROOF_MOUNTINFO_INVALID")
        mount_point = Path(_decode_mount_field(fields[4]))
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        candidates.append((len(mount_point.parts), fields, separator))
    if not candidates:
        raise SandboxUnavailable("RUNTIME_PROOF_MOUNT_NOT_FOUND", str(resolved))
    _, fields, separator = max(candidates, key=lambda row: row[0])
    mount_options = sorted(fields[5].split(","))
    super_options = sorted(fields[separator + 3].split(","))
    try:
        statvfs_read_only = bool(
            os.statvfs(resolved).f_flag & getattr(os, "ST_RDONLY", 1)
        )
    except OSError as exc:
        raise SandboxUnavailable("RUNTIME_PROOF_MOUNT_UNOBSERVABLE", str(resolved)) from exc
    mount_options_read_only = "ro" in mount_options
    return {
        "path": str(resolved),
        "mount_id": int(fields[0], 10),
        "parent_mount_id": int(fields[1], 10),
        "device": fields[2],
        "mount_root": _decode_mount_field(fields[3]),
        "mount_point": _decode_mount_field(fields[4]),
        "mount_options": mount_options,
        "super_options": super_options,
        "statvfs_read_only": statvfs_read_only,
        "mount_options_read_only": mount_options_read_only,
        "read_only": statvfs_read_only and mount_options_read_only,
    }


def runtime_proof_environment_sha256(environment: dict[str, Any]) -> str:
    payload = {
        key: value for key, value in environment.items()
        if key != "environment_sha256"
    }
    return sha256_bytes(canonical_bytes(payload))


def _runtime_environment_reason_codes(environment: dict[str, Any]) -> list[str]:
    reasons: set[str] = set()
    capabilities = environment["capability_closure"]
    if not capabilities["active_privileges_empty"]:
        reasons.add("ACTIVE_PRIVILEGE_NONEMPTY")
    if capabilities["no_new_privs"] is not True:
        reasons.add("NO_NEW_PRIVS_INACTIVE")
    if capabilities["seccomp_mode"] != 2:
        reasons.add("SECCOMP_FILTER_INACTIVE")
    if any(
        not record["close_on_exec"] and not record["allowed_inherited"]
        for record in environment["descriptor_table"]
    ):
        reasons.add("UNEXPECTED_INHERITED_DESCRIPTOR")
    if any(
        record["confinement"] != "externally-confined"
        for record in environment["cooperating_subjects"]
    ):
        reasons.add("UNCONFINED_SAME_UID_COOPERATING_SUBJECT")
    separation = environment["privilege_separation"]
    if (
        separation["trusted_supervisor"] != "externally-attested"
        or separation["distinct_runtime_subject"] is not True
    ):
        reasons.add("NO_PRIVILEGE_SEPARATED_SUPERVISOR")
    boundary = environment["mount_boundary"]
    if not boundary["protected_mounts"]:
        reasons.add("PROTECTED_MOUNT_SET_EMPTY")
    elif boundary["all_protected_mounts_read_only"] is not True:
        reasons.add("PROTECTED_MOUNT_WRITABLE")
    if capabilities["effective_uid"] == 0:
        reasons.add("HOST_ROOT_SUBJECT")
    return sorted(reasons)


def capture_runtime_proof_environment(
    protected_paths: Iterable[Path] = (),
    *,
    allowed_inherited_fds: Iterable[int] = (),
    status: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Capture the observable environment without upgrading it to a CP2 proof.

    The local runner has no privilege-separated supervisor, so this receipt is
    intentionally mechanism-only.  It can prove which controls were observed;
    it cannot authorize a paper-result execution.
    """

    observed = status or _status_fields()
    subject = current_subject_identity(observed)
    descriptors = _descriptor_table(allowed_inherited_fds)
    mounts = [_mount_record(Path(path)) for path in protected_paths]
    active_capabilities = tuple(
        subject[key]
        for key in (
            "capability_inheritable",
            "capability_permitted",
            "capability_effective",
            "capability_ambient",
        )
    )
    capabilities = {
        "effective_uid": subject["effective_uid"],
        "effective_gid": subject["effective_gid"],
        "supplementary_groups": subject["supplementary_groups"],
        "inheritable": subject["capability_inheritable"],
        "permitted": subject["capability_permitted"],
        "effective": subject["capability_effective"],
        "ambient": subject["capability_ambient"],
        "bounding": subject["capability_bounding"],
        "no_new_privs": observed["NoNewPrivs"] == "1",
        "seccomp_mode": int(observed["Seccomp"], 10),
        "active_privileges_empty": all(
            value == "0000000000000000" for value in active_capabilities
        ),
    }
    environment: dict[str, Any] = {
        "schema": RUNTIME_PROOF_ENVIRONMENT_SCHEMA,
        "threat_model": json.loads(canonical_bytes(RUNTIME_THREAT_MODEL)),
        "threat_model_sha256": RUNTIME_THREAT_MODEL_SHA256,
        "environment_capability": "mechanism-only",
        "reason_codes": [],
        "uid_map": _read_id_map("uid_map"),
        "gid_map": _read_id_map("gid_map"),
        "namespaces": {
            "user": _namespace_id("user"),
            "mount": _namespace_id("mnt"),
        },
        "descriptor_table": descriptors,
        "descriptor_table_sha256": sha256_bytes(canonical_bytes(descriptors)),
        "cooperating_subjects": [{
            "subject_class": "same-effective-uid-host-processes",
            "effective_uid": subject["effective_uid"],
            "confinement": "unconfined",
        }],
        "capability_closure": capabilities,
        "privilege_separation": {
            "trusted_supervisor": "absent",
            "controller_effective_uid": subject["effective_uid"],
            "runtime_effective_uid": subject["effective_uid"],
            "distinct_runtime_subject": False,
        },
        "mount_boundary": {
            "required": "read-only-protected-inputs-and-runtime",
            "protected_mounts": mounts,
            "all_protected_mounts_read_only": bool(mounts)
            and all(record["read_only"] for record in mounts),
        },
    }
    environment["reason_codes"] = _runtime_environment_reason_codes(environment)
    environment["environment_sha256"] = runtime_proof_environment_sha256(environment)
    return environment


def _validate_id_map(value: Any, *, name: str) -> None:
    if not isinstance(value, list) or not value:
        raise SandboxUnavailable("RUNTIME_PROOF_ID_MAP_INVALID", name)
    for record in value:
        if not isinstance(record, dict) or set(record) != {"inside_id", "outside_id", "length"}:
            raise SandboxUnavailable("RUNTIME_PROOF_ID_MAP_INVALID", name)
        if not all(_exact_int(record[key]) and record[key] >= 0 for key in record):
            raise SandboxUnavailable("RUNTIME_PROOF_ID_MAP_INVALID", name)
        if record["length"] <= 0:
            raise SandboxUnavailable("RUNTIME_PROOF_ID_MAP_INVALID", name)


def validate_runtime_proof_environment(
    environment: dict[str, Any],
    *,
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    required = {
        "schema", "threat_model", "threat_model_sha256", "environment_capability", "reason_codes",
        "uid_map", "gid_map", "namespaces", "descriptor_table",
        "descriptor_table_sha256", "cooperating_subjects", "capability_closure",
        "privilege_separation", "mount_boundary", "environment_sha256",
    }
    if not isinstance(environment, dict) or set(environment) != required:
        raise SandboxUnavailable("RUNTIME_PROOF_ENVIRONMENT_FIELDS")
    if environment.get("schema") != RUNTIME_PROOF_ENVIRONMENT_SCHEMA:
        raise SandboxUnavailable("RUNTIME_PROOF_ENVIRONMENT_SCHEMA")
    if (
        environment.get("threat_model") != RUNTIME_THREAT_MODEL
        or environment.get("threat_model_sha256") != RUNTIME_THREAT_MODEL_SHA256
        or sha256_bytes(canonical_bytes(environment["threat_model"]))
        != environment["threat_model_sha256"]
    ):
        raise SandboxUnavailable("RUNTIME_PROOF_THREAT_MODEL_MISMATCH")
    if environment.get("environment_capability") != "mechanism-only":
        raise SandboxUnavailable("RUNTIME_PROOF_CAPABILITY_UNSUPPORTED")
    if environment.get("environment_sha256") != runtime_proof_environment_sha256(environment):
        raise SandboxUnavailable("RUNTIME_PROOF_ENVIRONMENT_HASH_MISMATCH")

    _validate_id_map(environment.get("uid_map"), name="uid_map")
    _validate_id_map(environment.get("gid_map"), name="gid_map")
    namespaces = environment.get("namespaces")
    if (
        not isinstance(namespaces, dict)
        or set(namespaces) != {"user", "mount"}
        or _NAMESPACE_ID.fullmatch(str(namespaces.get("user"))) is None
        or _NAMESPACE_ID.fullmatch(str(namespaces.get("mount"))) is None
    ):
        raise SandboxUnavailable("RUNTIME_PROOF_NAMESPACE_INVALID")

    descriptors = environment.get("descriptor_table")
    if not isinstance(descriptors, list) or not descriptors:
        raise SandboxUnavailable("RUNTIME_PROOF_DESCRIPTOR_TABLE_INVALID")
    seen_descriptors: set[int] = set()
    for record in descriptors:
        if not isinstance(record, dict) or set(record) != {
            "fd", "target", "access_mode", "close_on_exec", "allowed_inherited",
        }:
            raise SandboxUnavailable("RUNTIME_PROOF_DESCRIPTOR_TABLE_INVALID")
        descriptor = record.get("fd")
        if not _exact_int(descriptor) or descriptor < 0 or descriptor in seen_descriptors:
            raise SandboxUnavailable("RUNTIME_PROOF_DESCRIPTOR_TABLE_INVALID")
        seen_descriptors.add(descriptor)
        if (
            not isinstance(record.get("target"), str)
            or record.get("access_mode") not in {
                "read-only", "write-only", "read-write", "path-only",
            }
            or not isinstance(record.get("close_on_exec"), bool)
            or not isinstance(record.get("allowed_inherited"), bool)
        ):
            raise SandboxUnavailable("RUNTIME_PROOF_DESCRIPTOR_TABLE_INVALID")
    if environment.get("descriptor_table_sha256") != sha256_bytes(canonical_bytes(descriptors)):
        raise SandboxUnavailable("RUNTIME_PROOF_DESCRIPTOR_TABLE_HASH_MISMATCH")

    subjects = environment.get("cooperating_subjects")
    if not isinstance(subjects, list) or not subjects:
        raise SandboxUnavailable("RUNTIME_PROOF_COOPERATING_SUBJECTS_INVALID")
    for record in subjects:
        if not isinstance(record, dict) or set(record) != {
            "subject_class", "effective_uid", "confinement",
        }:
            raise SandboxUnavailable("RUNTIME_PROOF_COOPERATING_SUBJECTS_INVALID")
        if (
            record.get("subject_class") != "same-effective-uid-host-processes"
            or not _exact_int(record.get("effective_uid"))
            or record.get("confinement") not in {"unconfined", "externally-confined"}
        ):
            raise SandboxUnavailable("RUNTIME_PROOF_COOPERATING_SUBJECTS_INVALID")

    capabilities = environment.get("capability_closure")
    capability_fields = {
        "effective_uid", "effective_gid", "supplementary_groups", "inheritable",
        "permitted", "effective", "ambient", "bounding", "no_new_privs",
        "seccomp_mode", "active_privileges_empty",
    }
    if not isinstance(capabilities, dict) or set(capabilities) != capability_fields:
        raise SandboxUnavailable("RUNTIME_PROOF_CAPABILITY_CLOSURE_INVALID")
    if (
        not _exact_int(capabilities.get("effective_uid"))
        or not _exact_int(capabilities.get("effective_gid"))
        or not isinstance(capabilities.get("supplementary_groups"), list)
        or not all(_exact_int(value) and value >= 0 for value in capabilities["supplementary_groups"])
        or capabilities["supplementary_groups"] != sorted(set(capabilities["supplementary_groups"]))
        or not all(
            _CAPABILITY_HEX.fullmatch(str(capabilities.get(key)))
            for key in ("inheritable", "permitted", "effective", "ambient", "bounding")
        )
        or not isinstance(capabilities.get("no_new_privs"), bool)
        or not _exact_int(capabilities.get("seccomp_mode"))
        or capabilities["seccomp_mode"] not in {0, 1, 2}
        or not isinstance(capabilities.get("active_privileges_empty"), bool)
    ):
        raise SandboxUnavailable("RUNTIME_PROOF_CAPABILITY_CLOSURE_INVALID")
    active_empty = all(
        capabilities[key] == "0000000000000000"
        for key in ("inheritable", "permitted", "effective", "ambient")
    )
    if capabilities["active_privileges_empty"] is not active_empty:
        raise SandboxUnavailable("RUNTIME_PROOF_PRIVILEGE_FLAG_MISMATCH")
    if not active_empty:
        raise SandboxUnavailable("RUNTIME_PROOF_PRIVILEGE_NONEMPTY")

    separation = environment.get("privilege_separation")
    if not isinstance(separation, dict) or set(separation) != {
        "trusted_supervisor", "controller_effective_uid", "runtime_effective_uid",
        "distinct_runtime_subject",
    }:
        raise SandboxUnavailable("RUNTIME_PROOF_PRIVILEGE_SEPARATION_INVALID")
    if (
        separation.get("trusted_supervisor") != "absent"
        or not _exact_int(separation.get("controller_effective_uid"))
        or not _exact_int(separation.get("runtime_effective_uid"))
        or separation.get("distinct_runtime_subject") is not False
    ):
        raise SandboxUnavailable("RUNTIME_PROOF_PRIVILEGE_SEPARATION_INVALID")

    boundary = environment.get("mount_boundary")
    if not isinstance(boundary, dict) or set(boundary) != {
        "required", "protected_mounts", "all_protected_mounts_read_only",
    } or boundary.get("required") != "read-only-protected-inputs-and-runtime":
        raise SandboxUnavailable("RUNTIME_PROOF_MOUNT_BOUNDARY_INVALID")
    mounts = boundary.get("protected_mounts")
    if not isinstance(mounts, list):
        raise SandboxUnavailable("RUNTIME_PROOF_MOUNT_BOUNDARY_INVALID")
    mount_fields = {
        "path", "mount_id", "parent_mount_id", "device", "mount_root",
        "mount_point", "mount_options", "super_options", "statvfs_read_only",
        "mount_options_read_only", "read_only",
    }
    for record in mounts:
        if not isinstance(record, dict) or set(record) != mount_fields:
            raise SandboxUnavailable("RUNTIME_PROOF_MOUNT_BOUNDARY_INVALID")
        if (
            not isinstance(record.get("path"), str)
            or not _exact_int(record.get("mount_id"))
            or not _exact_int(record.get("parent_mount_id"))
            or not all(isinstance(record.get(key), str) for key in (
                "device", "mount_root", "mount_point",
            ))
            or not all(
                isinstance(record.get(key), list)
                and all(isinstance(value, str) for value in record[key])
                for key in ("mount_options", "super_options")
            )
            or not all(isinstance(record.get(key), bool) for key in (
                "statvfs_read_only", "mount_options_read_only", "read_only",
            ))
            or record["mount_options_read_only"] is not ("ro" in record["mount_options"])
            or record["read_only"] is not (
                record["statvfs_read_only"] and record["mount_options_read_only"]
            )
        ):
            raise SandboxUnavailable("RUNTIME_PROOF_MOUNT_BOUNDARY_INVALID")
    observed_all_read_only = bool(mounts) and all(record["read_only"] for record in mounts)
    if boundary.get("all_protected_mounts_read_only") is not observed_all_read_only:
        raise SandboxUnavailable("RUNTIME_PROOF_MOUNT_BOUNDARY_INVALID")

    if expected is not None:
        comparisons = (
            ("uid_map", "RUNTIME_PROOF_UID_MAP_MISMATCH"),
            ("gid_map", "RUNTIME_PROOF_GID_MAP_MISMATCH"),
            ("namespaces", "RUNTIME_PROOF_NAMESPACE_MISMATCH"),
            ("descriptor_table", "RUNTIME_PROOF_DESCRIPTOR_TABLE_MISMATCH"),
            ("descriptor_table_sha256", "RUNTIME_PROOF_DESCRIPTOR_TABLE_MISMATCH"),
            ("cooperating_subjects", "RUNTIME_PROOF_COOPERATING_SUBJECTS_MISMATCH"),
            ("capability_closure", "RUNTIME_PROOF_CAPABILITY_CLOSURE_MISMATCH"),
            ("privilege_separation", "RUNTIME_PROOF_PRIVILEGE_SEPARATION_MISMATCH"),
            ("mount_boundary", "RUNTIME_PROOF_MOUNT_BOUNDARY_MISMATCH"),
        )
        for key, code in comparisons:
            if environment.get(key) != expected.get(key):
                raise SandboxUnavailable(code)

    reasons = environment.get("reason_codes")
    if (
        not isinstance(reasons, list)
        or reasons != sorted(set(reasons))
        or not all(isinstance(reason, str) and reason for reason in reasons)
        or reasons != _runtime_environment_reason_codes(environment)
    ):
        raise SandboxUnavailable("RUNTIME_PROOF_REASON_CODES_INVALID")
    return environment


def require_strict_runtime_proof_environment(
    protected_paths: Iterable[Path],
) -> dict[str, Any]:
    environment = capture_runtime_proof_environment(protected_paths)
    validate_runtime_proof_environment(environment)
    raise SandboxUnavailable(
        RUNTIME_PROOF_ENVIRONMENT_UNAVAILABLE,
        {
            "environment_capability": environment["environment_capability"],
            "environment_sha256": environment["environment_sha256"],
            "reason_codes": environment["reason_codes"],
            "runtime_proof_environment": environment,
        },
    )


def _denial_probe(denied_write_path: Path, libc: ctypes.CDLL) -> dict[str, bool]:
    probes = {
        "outside_write_denied": False,
        "chmod_denied": False,
        "chown_denied": False,
        "xattr_denied": False,
        "utime_denied": False,
        "rename_denied": False,
        "link_denied": False,
        "unlink_denied": False,
        "network_denied": False,
        "fork_denied": False,
    }
    try:
        descriptor = os.open(denied_write_path, os.O_WRONLY | os.O_CLOEXEC)
    except OSError as exc:
        probes["outside_write_denied"] = exc.errno in {errno.EACCES, errno.EPERM}
    else:
        os.close(descriptor)

    metadata_operations = {
        "chmod_denied": lambda: os.chmod(denied_write_path, 0o600),
        "chown_denied": lambda: os.chown(
            denied_write_path, os.geteuid(), os.getegid()
        ),
        "xattr_denied": lambda: os.setxattr(
            denied_write_path, "user.experiments7-sandbox-probe", b"denied"
        ),
        "utime_denied": lambda: os.utime(denied_write_path, None),
        "rename_denied": lambda: os.rename(denied_write_path, denied_write_path),
        "link_denied": lambda: os.link(
            denied_write_path, denied_write_path.with_name(".sandbox-probe-link")
        ),
        "unlink_denied": lambda: os.unlink(denied_write_path),
    }
    for name, operation in metadata_operations.items():
        try:
            operation()
        except OSError as exc:
            probes[name] = exc.errno == errno.EPERM

    ctypes.set_errno(0)
    socket_descriptor = libc.socket(2, 1, 0)
    if socket_descriptor < 0:
        probes["network_denied"] = ctypes.get_errno() == errno.EPERM
    else:
        os.close(socket_descriptor)

    try:
        child = os.fork()
    except OSError as exc:
        probes["fork_denied"] = exc.errno == errno.EPERM
    else:
        if child == 0:
            os._exit(91)
        os.waitpid(child, 0)
    return probes


def make_preexec(
    backend: SandboxBackend,
    output_inodes: tuple[DeclaredOutputInode, ...],
    denied_write_path: Path,
    evidence_fd: int,
    expected_subject_identity_sha256: str,
) -> Callable[[], None]:
    """Return the exact child setup hook used immediately before execve."""

    def apply() -> None:
        libc = _libc()
        os.umask(0o077)
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        _set_no_new_privs(libc)
        _apply_landlock(libc, backend.landlock_abi, output_inodes)
        installed = _apply_seccomp(backend.seccomp)
        status = _status_fields()
        probes = _denial_probe(denied_write_path, libc)
        if status["NoNewPrivs"] != "1" or status["Seccomp"] != "2":
            raise SandboxUnavailable("SANDBOX_STATUS_MISMATCH")
        if any(status[key] != "0000000000000000" for key in ("CapInh", "CapPrm", "CapEff", "CapAmb")):
            raise SandboxUnavailable("SANDBOX_CAPABILITIES_NONEMPTY")
        if not all(probes.values()):
            raise SandboxUnavailable("SANDBOX_DENIAL_PROBE_FAILED", probes)
        subject = current_subject_identity(status)
        subject_hash = subject_identity_sha256(subject)
        if subject_hash != expected_subject_identity_sha256:
            raise SandboxUnavailable("SANDBOX_SUBJECT_IDENTITY_DRIFT")
        runtime_environment = capture_runtime_proof_environment(
            (denied_write_path,),
            allowed_inherited_fds=(evidence_fd,),
            status=status,
        )
        validate_runtime_proof_environment(runtime_environment)
        evidence = {
            "schema": SANDBOX_EVIDENCE_SCHEMA,
            "policy_sha256": backend.policy_sha256,
            "backend": "landlock+seccomp",
            "landlock_abi": backend.landlock_abi,
            "seccomp_rule_count": len(installed),
            "no_new_privs": True,
            **subject,
            "subject_identity_sha256": subject_hash,
            "cooperating_subject_boundary": (
                "mechanism-only; unconfined same-effective-uid host subjects remain"
            ),
            "seccomp_mode": int(status["Seccomp"]),
            "denial_probes": probes,
            "runtime_proof_environment": runtime_environment,
        }
        payload = canonical_bytes(evidence)
        view = memoryview(payload)
        while view:
            written = os.write(evidence_fd, view)
            if written <= 0:
                raise SandboxUnavailable("SANDBOX_EVIDENCE_SHORT_WRITE")
            view = view[written:]
        os.close(evidence_fd)

    return apply


def parse_evidence(
    payload: bytes,
    backend: SandboxBackend,
    expected_subject_identity_sha256: str,
) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = item
        return value

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
        )
    except Exception as exc:
        raise SandboxUnavailable("SANDBOX_EVIDENCE_INVALID") from exc
    if payload != canonical_bytes(value):
        raise SandboxUnavailable("SANDBOX_EVIDENCE_INVALID")
    required = {
        "schema", "policy_sha256", "backend", "landlock_abi",
        "seccomp_rule_count", "no_new_privs", "effective_uid", "effective_gid",
        "supplementary_groups", "capability_inheritable", "capability_permitted",
        "capability_effective", "capability_ambient", "capability_bounding",
        "subject_identity_sha256", "cooperating_subject_boundary", "seccomp_mode",
        "denial_probes", "runtime_proof_environment",
    }
    if (
        not isinstance(value, dict)
        or set(value) != required
        or value.get("schema") != SANDBOX_EVIDENCE_SCHEMA
    ):
        raise SandboxUnavailable("SANDBOX_EVIDENCE_SCHEMA")
    if value.get("policy_sha256") != backend.policy_sha256:
        raise SandboxUnavailable("SANDBOX_POLICY_EVIDENCE_MISMATCH")
    if (
        value.get("backend") != "landlock+seccomp"
        or not _exact_int(value.get("landlock_abi"))
        or value["landlock_abi"] != backend.landlock_abi
        or not _exact_int(value.get("seccomp_rule_count"))
        or value["seccomp_rule_count"] <= 0
    ):
        raise SandboxUnavailable("SANDBOX_BACKEND_EVIDENCE_MISMATCH")
    probes = value.get("denial_probes")
    if (
        not isinstance(probes, dict)
        or set(probes) != {
            "outside_write_denied", "chmod_denied", "chown_denied",
            "xattr_denied", "utime_denied", "rename_denied", "link_denied",
            "unlink_denied", "network_denied", "fork_denied",
        }
        or not all(value is True for value in probes.values())
    ):
        raise SandboxUnavailable("SANDBOX_DENIAL_PROBE_FAILED")
    if value.get("no_new_privs") is not True or value.get("seccomp_mode") != 2:
        raise SandboxUnavailable("SANDBOX_STATUS_MISMATCH")
    subject_keys = (
        "effective_uid",
        "effective_gid",
        "supplementary_groups",
        "capability_inheritable",
        "capability_permitted",
        "capability_effective",
        "capability_ambient",
        "capability_bounding",
    )
    subject = {key: value.get(key) for key in subject_keys}
    observed_subject_hash = subject_identity_sha256(subject)
    if (
        value.get("subject_identity_sha256") != observed_subject_hash
        or observed_subject_hash != expected_subject_identity_sha256
    ):
        raise SandboxUnavailable("SANDBOX_SUBJECT_IDENTITY_MISMATCH")
    if value.get("cooperating_subject_boundary") != (
        "mechanism-only; unconfined same-effective-uid host subjects remain"
    ):
        raise SandboxUnavailable("SANDBOX_COOPERATING_SUBJECT_BOUNDARY_MISSING")
    runtime_environment = validate_runtime_proof_environment(
        value.get("runtime_proof_environment")
    )
    capabilities = runtime_environment["capability_closure"]
    if (
        capabilities["effective_uid"] != subject["effective_uid"]
        or capabilities["effective_gid"] != subject["effective_gid"]
        or capabilities["supplementary_groups"] != subject["supplementary_groups"]
        or capabilities["inheritable"] != subject["capability_inheritable"]
        or capabilities["permitted"] != subject["capability_permitted"]
        or capabilities["effective"] != subject["capability_effective"]
        or capabilities["ambient"] != subject["capability_ambient"]
        or capabilities["bounding"] != subject["capability_bounding"]
        or capabilities["no_new_privs"] is not True
        or capabilities["seccomp_mode"] != 2
    ):
        raise SandboxUnavailable("SANDBOX_RUNTIME_ENVIRONMENT_MISMATCH")
    return value
