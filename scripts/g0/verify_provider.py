#!/usr/bin/env python3
"""Disposable canary verifier for the frozen OS read-only provider."""
from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import provider


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def denied(label: str, operation) -> dict[str, object]:
    try:
        operation()
    except OSError as exc:
        return {"operation": label, "denied": exc.errno in (errno.EACCES, errno.EPERM, errno.EXDEV), "errno": exc.errno}
    return {"operation": label, "denied": False, "errno": 0}


def child(protected: Path, writable: Path, result_fd: int) -> None:
    evidence = provider.apply_readonly_envelope([str(writable), "/tmp"])
    original = protected / "original.txt"
    before = digest(original)
    results = [
        denied("create", lambda: os.open(protected / "created.txt", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)),
        denied("content_write", lambda: os.open(original, os.O_WRONLY | os.O_NOFOLLOW)),
        denied("rename", lambda: os.rename(protected / "rename-src.txt", protected / "renamed.txt")),
        denied("unlink", lambda: os.unlink(protected / "unlink-src.txt")),
        denied("chmod", lambda: os.chmod(original, 0o600, follow_symlinks=False)),
        denied("timestamp", lambda: os.utime(original, ns=(1, 1), follow_symlinks=False)),
    ]
    fd = os.open(writable / "write-ok.txt", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    os.write(fd, b"ok\n")
    os.fsync(fd)
    os.close(fd)
    payload = {
        "provider": evidence,
        "operations": results,
        "protected_hash_before": before,
        "protected_hash_after": digest(original),
        "writable_child_ok": (writable / "write-ok.txt").read_bytes() == b"ok\n",
    }
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    os.write(result_fd, raw)
    os.close(result_fd)
    if not all(item["denied"] for item in results) or before != payload["protected_hash_after"] or not payload["writable_child_ok"]:
        os._exit(2)
    os._exit(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-parent", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parent = Path(args.fixture_parent)
    parent.mkdir(mode=0o755, parents=False, exist_ok=True)
    base = Path(tempfile.mkdtemp(prefix="provider-canary-", dir=parent))
    protected = base / "protected"
    writable = base / "writable"
    protected.mkdir()
    writable.mkdir()
    for name in ("original.txt", "rename-src.txt", "unlink-src.txt"):
        (protected / name).write_text(name, encoding="utf-8")
    read_fd, write_fd = os.pipe()
    try:
        pid = os.fork()
        if pid == 0:
            os.close(read_fd)
            try:
                child(protected, writable, write_fd)
            except BaseException as exc:
                os.write(write_fd, (json.dumps({"status": "BLOCKED", "reason": repr(exc)}, sort_keys=True) + "\n").encode())
                os._exit(3)
        os.close(write_fd)
        chunks = []
        while block := os.read(read_fd, 1 << 20):
            chunks.append(block)
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        exit_code = os.waitstatus_to_exitcode(status)
        payload = json.loads(b"".join(chunks))
        payload.update({"schema": "experiments7-provider-canary/v1", "exit_code": exit_code, "status": "PASS" if exit_code == 0 else "BLOCKED"})
        output = Path(args.output)
        fd = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
        os.write(fd, (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode())
        os.fsync(fd)
        os.close(fd)
        print(json.dumps(payload, sort_keys=True))
        if exit_code != 0:
            raise SystemExit(2)
    finally:
        try: os.close(read_fd)
        except OSError: pass
        try: os.close(write_fd)
        except OSError: pass
        shutil.rmtree(base)


if __name__ == "__main__":
    main()
