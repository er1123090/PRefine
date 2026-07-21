#!/usr/bin/env python3
"""Exercise the V6 Linux sandbox on a disposable synthetic fixture."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facade.sandbox import (  # noqa: E402
    DeclaredOutputInode,
    RUNTIME_PROOF_ENVIRONMENT_UNAVAILABLE,
    SandboxUnavailable,
    current_subject_identity,
    make_preexec,
    parse_evidence,
    prepare_backend,
    subject_identity_sha256,
)
from facade.selection import canonical_bytes, sha256_file  # noqa: E402


def _write_owned(path: Path, payload: bytes) -> tuple[int, int]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        if payload:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise SandboxUnavailable("SANDBOX_PROBE_SHORT_WRITE")
        os.fsync(descriptor)
        state = os.fstat(descriptor)
        return state.st_dev, state.st_ino
    finally:
        os.close(descriptor)


def validate_boundary() -> dict[str, object]:
    if os.geteuid() == 0:
        raise SandboxUnavailable("RUNTIME_HOST_ROOT_FORBIDDEN")
    backend = prepare_backend()
    with tempfile.TemporaryDirectory(prefix="experiments7-v6-sandbox-probe-") as raw:
        directory = Path(raw)
        allowed = directory / "declared-output"
        denied = directory / "parent-writable-denial-canary"
        allowed_identity = _write_owned(allowed, b"")
        denied_identity = _write_owned(denied, b"parent-write-proven\n")
        denied_hash = sha256_file(denied)
        expected_subject_hash = subject_identity_sha256(current_subject_identity())
        evidence_read, evidence_write = os.pipe2(os.O_CLOEXEC)
        try:
            process = subprocess.Popen(
                ["/usr/bin/true"],
                cwd=directory,
                env={
                    "PATH": "/usr/bin:/bin",
                    "LC_ALL": "C",
                    "LANG": "C",
                    "TZ": "UTC",
                },
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                pass_fds=(evidence_write,),
                start_new_session=True,
                preexec_fn=make_preexec(
                    backend,
                    (DeclaredOutputInode(allowed, *allowed_identity),),
                    denied,
                    evidence_write,
                    expected_subject_hash,
                ),
            )
            os.close(evidence_write)
            evidence_write = -1
            returncode = process.wait(timeout=10)
            chunks: list[bytes] = []
            while True:
                block = os.read(evidence_read, 65536)
                if not block:
                    break
                chunks.append(block)
            evidence = parse_evidence(
                b"".join(chunks), backend, expected_subject_hash
            )
        finally:
            if evidence_write >= 0:
                os.close(evidence_write)
            os.close(evidence_read)
        denied_state = denied.lstat()
        if (
            returncode != 0
            or (denied_state.st_dev, denied_state.st_ino) != denied_identity
            or sha256_file(denied) != denied_hash
        ):
            raise SandboxUnavailable("SANDBOX_BOUNDARY_PROBE_FAILED")
        return {
            "schema": "experiments7-v6-runtime-boundary-validation/v2",
            "state": "BLOCKED",
            "reason_codes": [RUNTIME_PROOF_ENVIRONMENT_UNAVAILABLE],
            "claim": "synthetic mechanism validation only; not CP2 execution proof",
            "mechanism_state": "PASS",
            "backend": "landlock+seccomp",
            "policy_sha256": backend.policy_sha256,
            "landlock_abi": backend.landlock_abi,
            "no_new_privs": evidence["no_new_privs"],
            "seccomp_mode": evidence["seccomp_mode"],
            "denial_probes": evidence["denial_probes"],
            "declared_output_inode_binding": {
                "device": allowed_identity[0],
                "inode": allowed_identity[1],
            },
            "parent_canary_write_proven": True,
            "subject_identity_sha256": evidence["subject_identity_sha256"],
            "cooperating_subject_boundary": evidence["cooperating_subject_boundary"],
            "runtime_proof_environment": evidence["runtime_proof_environment"],
            "environment_capability": evidence["runtime_proof_environment"][
                "environment_capability"
            ],
            "environment_reason_codes": evidence["runtime_proof_environment"][
                "reason_codes"
            ],
            "effective_uid": evidence["effective_uid"],
            "effective_gid": evidence["effective_gid"],
            "protected_reads": 0,
            "protected_writes": 0,
            "live_network_attempted": False,
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", required=True)
    parser.parse_args(argv)
    try:
        report = validate_boundary()
        returncode = 0 if report["state"] == "PASS" else 2
    except (SandboxUnavailable, OSError, subprocess.SubprocessError) as exc:
        reason = exc.code if isinstance(exc, SandboxUnavailable) else "SANDBOX_ENFORCEMENT_UNOBSERVABLE"
        report = {
            "schema": "experiments7-v6-runtime-boundary-validation/v2",
            "state": "BLOCKED",
            "reason_codes": [reason],
            "protected_reads": 0,
            "protected_writes": 0,
        }
        returncode = 2
    sys.stdout.buffer.write(canonical_bytes(report))
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
