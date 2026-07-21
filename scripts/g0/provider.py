#!/usr/bin/env python3
"""Unprivileged OS-enforced read-only envelope (Landlock ABI 4 + seccomp)."""
from __future__ import annotations

import ctypes
import errno
import os
import platform
import stat
from collections.abc import Iterable

SYS_LANDLOCK_CREATE_RULESET = 444
SYS_LANDLOCK_ADD_RULE = 445
SYS_LANDLOCK_RESTRICT_SELF = 446
LANDLOCK_CREATE_RULESET_VERSION = 1
LANDLOCK_RULE_PATH_BENEATH = 1

LL_EXECUTE = 1 << 0
LL_WRITE_FILE = 1 << 1
LL_READ_FILE = 1 << 2
LL_READ_DIR = 1 << 3
LL_REMOVE_DIR = 1 << 4
LL_REMOVE_FILE = 1 << 5
LL_MAKE_CHAR = 1 << 6
LL_MAKE_DIR = 1 << 7
LL_MAKE_REG = 1 << 8
LL_MAKE_SOCK = 1 << 9
LL_MAKE_FIFO = 1 << 10
LL_MAKE_BLOCK = 1 << 11
LL_MAKE_SYM = 1 << 12
LL_REFER = 1 << 13
LL_TRUNCATE = 1 << 14
LL_READ_EXEC = LL_EXECUTE | LL_READ_FILE | LL_READ_DIR
LL_ALL = (1 << 15) - 1

PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
AUDIT_ARCH_X86_64 = 0xC000003E
SECCOMP_DATA_NR_OFFSET = 0
SECCOMP_DATA_ARCH_OFFSET = 4
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ALLOW = 0x7FFF0000
SECCOMP_RET_ERRNO = 0x00050000
BPF_LD_W_ABS = 0x20
BPF_JMP_JEQ_K = 0x15
BPF_RET_K = 0x06

# x86_64 metadata mutators not mediated by Landlock ABI 4.  Denying them for
# the whole protected process is intentionally stronger than path checks.
DENIED_METADATA_SYSCALLS = (
    90,   # chmod
    91,   # fchmod
    92,   # chown
    93,   # fchown
    94,   # lchown
    132,  # utime
    188,  # setxattr
    189,  # lsetxattr
    190,  # fsetxattr
    197,  # removexattr
    198,  # lremovexattr
    199,  # fremovexattr
    235,  # utimes
    260,  # fchownat
    261,  # futimesat
    268,  # fchmodat
    280,  # utimensat
    452,  # fchmodat2
)


class RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]


LIBC = ctypes.CDLL(None, use_errno=True)


def _syscall(number: int, *args: object) -> int:
    result = int(LIBC.syscall(number, *args))
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def query_landlock_abi() -> int:
    return _syscall(
        SYS_LANDLOCK_CREATE_RULESET,
        0,
        0,
        LANDLOCK_CREATE_RULESET_VERSION,
    )


def provider_identity() -> dict[str, object]:
    return {
        "provider": "landlock_path_beneath+seccomp_metadata_deny",
        "landlock_abi": query_landlock_abi(),
        "seccomp_mode": "classic_bpf_errno_eperm",
        "denied_metadata_syscalls_x86_64": list(DENIED_METADATA_SYSCALLS),
        "kernel": platform.release(),
        "machine": platform.machine(),
    }


def _real_directory(path: str) -> os.stat_result:
    absolute = os.path.abspath(path)
    current = "/"
    for component in [part for part in absolute.split(os.sep) if part]:
        current = os.path.join(current, component)
        st = os.lstat(current)
        if stat.S_ISLNK(st.st_mode):
            raise RuntimeError(f"symlink component rejected: {current}")
    st = os.lstat(absolute)
    if not stat.S_ISDIR(st.st_mode) or os.path.realpath(absolute) != absolute:
        raise RuntimeError(f"not an exact real directory: {absolute}")
    return st


def _add_rule(ruleset_fd: int, path: str, rights: int) -> dict[str, object]:
    st = _real_directory(path)
    parent_fd = os.open(path, os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    try:
        attr = PathBeneathAttr(rights, parent_fd, 0)
        _syscall(
            SYS_LANDLOCK_ADD_RULE,
            ruleset_fd,
            LANDLOCK_RULE_PATH_BENEATH,
            ctypes.byref(attr),
            0,
        )
        check = os.fstat(parent_fd)
        if (check.st_dev, check.st_ino) != (st.st_dev, st.st_ino):
            raise RuntimeError(f"binding changed while adding Landlock rule: {path}")
        return {"path": path, "dev": st.st_dev, "inode": st.st_ino, "rights": rights}
    finally:
        os.close(parent_fd)


def _apply_seccomp_metadata_deny() -> None:
    if platform.machine() != "x86_64":
        raise RuntimeError(f"unsupported seccomp architecture: {platform.machine()}")
    instructions = [
        SockFilter(BPF_LD_W_ABS, 0, 0, SECCOMP_DATA_ARCH_OFFSET),
        SockFilter(BPF_JMP_JEQ_K, 1, 0, AUDIT_ARCH_X86_64),
        SockFilter(BPF_RET_K, 0, 0, SECCOMP_RET_KILL_PROCESS),
        SockFilter(BPF_LD_W_ABS, 0, 0, SECCOMP_DATA_NR_OFFSET),
    ]
    for syscall_number in DENIED_METADATA_SYSCALLS:
        instructions.append(SockFilter(BPF_JMP_JEQ_K, 0, 1, syscall_number))
        instructions.append(
            SockFilter(
                BPF_RET_K,
                0,
                0,
                SECCOMP_RET_ERRNO | errno.EPERM,
            )
        )
    instructions.append(SockFilter(BPF_RET_K, 0, 0, SECCOMP_RET_ALLOW))
    filters = (SockFilter * len(instructions))(*instructions)
    program = SockFprog(len(instructions), filters)
    if LIBC.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(program), 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def apply_readonly_envelope(writable_directories: Iterable[str]) -> dict[str, object]:
    """Restrict current process and future children; this operation is irreversible."""
    abi = query_landlock_abi()
    if abi < 4:
        raise RuntimeError(f"Landlock ABI {abi} lacks REFER/TRUNCATE mediation")
    writable = sorted({os.path.abspath(path) for path in writable_directories})
    attr = RulesetAttr(LL_ALL)
    ruleset_fd = _syscall(
        SYS_LANDLOCK_CREATE_RULESET,
        ctypes.byref(attr),
        ctypes.sizeof(attr),
        0,
    )
    rules: list[dict[str, object]] = []
    try:
        rules.append(_add_rule(ruleset_fd, "/", LL_READ_EXEC))
        for path in writable:
            rules.append(_add_rule(ruleset_fd, path, LL_ALL))
        if LIBC.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        _syscall(SYS_LANDLOCK_RESTRICT_SELF, ruleset_fd, 0)
    finally:
        os.close(ruleset_fd)
    _apply_seccomp_metadata_deny()
    return {
        **provider_identity(),
        "landlock_rules": rules,
        "default_filesystem_rights": "read_execute_only",
        "writable_directories": writable,
        "metadata_mutation_policy": "globally_denied_by_seccomp",
    }
