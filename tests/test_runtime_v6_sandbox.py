from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facade.external import (  # noqa: E402
    _claim_output,
    _safe_runtime_artifact,
    _verified_output,
)
from facade.registry import FacadeError  # noqa: E402
from facade.sandbox import (  # noqa: E402
    DEFAULT_SANDBOX_POLICY,
    DeclaredOutputInode,
    RUNTIME_PROOF_ENVIRONMENT_UNAVAILABLE,
    SANDBOX_EVIDENCE_SCHEMA,
    SandboxBackend,
    SandboxUnavailable,
    _DENIED_SYSCALLS,
    capture_runtime_proof_environment,
    current_subject_identity,
    make_preexec,
    parse_evidence,
    prepare_backend,
    require_strict_runtime_proof_environment,
    runtime_proof_environment_sha256,
    sandbox_policy_sha256,
    subject_identity_sha256,
    validate_runtime_proof_environment,
)
from facade.selection import canonical_bytes, sha256_bytes, sha256_file  # noqa: E402


class RuntimeV6SandboxTests(unittest.TestCase):
    def _runtime_environment_with_seccomp_mode(self, mode: object) -> dict[str, Any]:
        environment = json.loads(
            canonical_bytes(capture_runtime_proof_environment((ROOT,)))
        )
        environment["capability_closure"]["seccomp_mode"] = mode
        reasons = set(environment["reason_codes"])
        reasons.discard("SECCOMP_FILTER_INACTIVE")
        if mode != 2:
            reasons.add("SECCOMP_FILTER_INACTIVE")
        environment["reason_codes"] = sorted(reasons)
        environment["environment_sha256"] = runtime_proof_environment_sha256(
            environment
        )
        return environment

    def _valid_sandbox_evidence(
        self,
    ) -> tuple[SandboxBackend, str, dict[str, Any]]:
        backend = prepare_backend()
        subject = current_subject_identity()
        subject_hash = subject_identity_sha256(subject)
        environment = self._runtime_environment_with_seccomp_mode(2)
        environment["capability_closure"]["no_new_privs"] = True
        environment["reason_codes"] = sorted(
            set(environment["reason_codes"]) - {"NO_NEW_PRIVS_INACTIVE"}
        )
        environment["environment_sha256"] = runtime_proof_environment_sha256(
            environment
        )
        evidence = {
            "schema": SANDBOX_EVIDENCE_SCHEMA,
            "policy_sha256": backend.policy_sha256,
            "backend": "landlock+seccomp",
            "landlock_abi": backend.landlock_abi,
            "seccomp_rule_count": 1,
            "no_new_privs": True,
            **subject,
            "subject_identity_sha256": subject_hash,
            "cooperating_subject_boundary": (
                "mechanism-only; unconfined same-effective-uid host subjects remain"
            ),
            "seccomp_mode": 2,
            "denial_probes": {
                "outside_write_denied": True,
                "chmod_denied": True,
                "chown_denied": True,
                "xattr_denied": True,
                "utime_denied": True,
                "rename_denied": True,
                "link_denied": True,
                "unlink_denied": True,
                "network_denied": True,
                "fork_denied": True,
            },
            "runtime_proof_environment": environment,
        }
        return backend, subject_hash, evidence

    def test_backend_is_landlock_seccomp_and_policy_hash_is_stable(self) -> None:
        backend = prepare_backend()
        self.assertGreaterEqual(backend.landlock_abi, 3)
        self.assertEqual(backend.policy_sha256, sandbox_policy_sha256())
        self.assertEqual(DEFAULT_SANDBOX_POLICY["backend"], "landlock+seccomp")

    def test_metadata_and_namespace_mutation_syscalls_are_denied(self) -> None:
        required = {
            "chmod", "fchmod", "fchmodat", "chown", "fchown", "fchownat",
            "setxattr", "fsetxattr", "removexattr", "utimensat",
            "rename", "renameat", "renameat2", "link", "linkat", "unlink", "unlinkat",
            "socket", "clone", "mount", "ptrace",
        }
        self.assertLessEqual(required, set(_DENIED_SYSCALLS))

    def test_same_effective_uid_writable_runtime_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="experiments7-writable-runtime-") as raw:
            directory = Path(raw)
            path = directory / "python"
            path.write_bytes(b"synthetic executable\n")
            path.chmod(0o555)
            directory.chmod(0o555)
            try:
                digest, size = sha256_file(path)
                with self.assertRaises(FacadeError) as caught:
                    _safe_runtime_artifact({
                        "artifact_id": "python",
                        "path": str(path),
                        "sha256": digest,
                        "bytes": size,
                    })
                self.assertEqual(
                    caught.exception.code,
                    "RUNTIME_ARTIFACT_WRITABLE_BY_SANDBOX_SUBJECT",
                )
            finally:
                directory.chmod(0o700)
                path.chmod(0o600)

    def test_same_uid_cooperating_helper_can_mutate_0555_tree(self) -> None:
        with tempfile.TemporaryDirectory(prefix="experiments7-same-uid-helper-") as raw:
            directory = Path(raw)
            path = directory / "sealed-looking"
            path.write_bytes(b"before\n")
            path.chmod(0o444)
            directory.chmod(0o555)
            try:
                completed = subprocess.run(
                    [
                        sys.executable,
                        "-B",
                        "-c",
                        (
                            "import os,sys; d,p=sys.argv[1:]; "
                            "os.chmod(d,0o700); os.chmod(p,0o600); "
                            "open(p,'wb').write(b'after\\n')"
                        ),
                        str(directory),
                        str(path),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(path.read_bytes(), b"after\n")
            finally:
                directory.chmod(0o700)
                path.chmod(0o600)

    def test_boundary_validator_runs_real_kernel_denial_probes(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "scripts/validate_runtime_boundary.py"),
                "--json",
            ],
            cwd=ROOT,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 2, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["state"], "BLOCKED")
        self.assertEqual(
            report["reason_codes"], [RUNTIME_PROOF_ENVIRONMENT_UNAVAILABLE]
        )
        self.assertEqual(report["mechanism_state"], "PASS")
        self.assertEqual(report["backend"], "landlock+seccomp")
        self.assertTrue(all(report["denial_probes"].values()))
        self.assertTrue(report["parent_canary_write_proven"])
        self.assertRegex(report["subject_identity_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(
            report["cooperating_subject_boundary"],
            "mechanism-only; unconfined same-effective-uid host subjects remain",
        )
        environment = report["runtime_proof_environment"]
        self.assertEqual(environment["environment_capability"], "mechanism-only")
        self.assertTrue(environment["uid_map"])
        self.assertTrue(environment["gid_map"])
        self.assertRegex(environment["namespaces"]["user"], r"^user:\[[0-9]+\]$")
        self.assertRegex(environment["namespaces"]["mount"], r"^mnt:\[[0-9]+\]$")
        self.assertTrue(environment["descriptor_table"])
        self.assertTrue(environment["capability_closure"]["no_new_privs"])
        self.assertFalse(
            environment["mount_boundary"]["all_protected_mounts_read_only"]
        )
        self.assertIn("PROTECTED_MOUNT_WRITABLE", environment["reason_codes"])
        self.assertEqual(report["protected_reads"], 0)

    def test_preexec_rejects_forged_declared_output_inode(self) -> None:
        backend = prepare_backend()
        with tempfile.TemporaryDirectory(prefix="experiments7-output-inode-") as raw:
            directory = Path(raw)
            output = directory / "output"
            canary = directory / "canary"
            output.write_bytes(b"")
            canary.write_bytes(b"parent-writable\n")
            state = output.stat()
            expected_subject_hash = subject_identity_sha256(
                current_subject_identity()
            )
            evidence_read, evidence_write = os.pipe2(os.O_CLOEXEC)
            try:
                with self.assertRaises(subprocess.SubprocessError):
                    subprocess.Popen(
                        ["/usr/bin/true"],
                        cwd=directory,
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        close_fds=True,
                        pass_fds=(evidence_write,),
                        preexec_fn=make_preexec(
                            backend,
                            (
                                DeclaredOutputInode(
                                    output, state.st_dev, state.st_ino + 1
                                ),
                            ),
                            canary,
                            evidence_write,
                            expected_subject_hash,
                        ),
                    )
            finally:
                os.close(evidence_read)
                os.close(evidence_write)

    def test_current_environment_is_explicitly_mechanism_only(self) -> None:
        environment = capture_runtime_proof_environment((ROOT,))
        self.assertIs(validate_runtime_proof_environment(environment), environment)
        self.assertEqual(environment["environment_capability"], "mechanism-only")
        self.assertIn(
            "NO_PRIVILEGE_SEPARATED_SUPERVISOR", environment["reason_codes"]
        )
        self.assertIn(
            "UNCONFINED_SAME_UID_COOPERATING_SUBJECT", environment["reason_codes"]
        )
        with self.assertRaises(SandboxUnavailable) as caught:
            require_strict_runtime_proof_environment((ROOT,))
        self.assertEqual(caught.exception.code, RUNTIME_PROOF_ENVIRONMENT_UNAVAILABLE)
        self.assertEqual(
            caught.exception.detail["environment_capability"], "mechanism-only"
        )

    def test_runtime_proof_rejects_forged_maps_namespaces_fds_and_subjects(self) -> None:
        observed = capture_runtime_proof_environment((ROOT,))

        def clone() -> dict[str, object]:
            return json.loads(canonical_bytes(observed))

        cases = []
        forged_uid = clone()
        forged_uid["uid_map"][0]["outside_id"] += 1
        cases.append((forged_uid, "RUNTIME_PROOF_UID_MAP_MISMATCH"))
        forged_gid = clone()
        forged_gid["gid_map"][0]["outside_id"] += 1
        cases.append((forged_gid, "RUNTIME_PROOF_GID_MAP_MISMATCH"))
        forged_namespace = clone()
        namespace_id = int(forged_namespace["namespaces"]["user"].split("[")[1][:-1])
        forged_namespace["namespaces"]["user"] = f"user:[{namespace_id + 1}]"
        cases.append((forged_namespace, "RUNTIME_PROOF_NAMESPACE_MISMATCH"))
        forged_descriptor = clone()
        forged_descriptor["descriptor_table"][0]["target"] += "-forged"
        forged_descriptor["descriptor_table_sha256"] = sha256_bytes(
            canonical_bytes(forged_descriptor["descriptor_table"])
        )
        cases.append((forged_descriptor, "RUNTIME_PROOF_DESCRIPTOR_TABLE_MISMATCH"))
        forged_subject = clone()
        forged_subject["cooperating_subjects"][0]["confinement"] = "externally-confined"
        cases.append((forged_subject, "RUNTIME_PROOF_COOPERATING_SUBJECTS_MISMATCH"))

        for forged, code in cases:
            with self.subTest(code=code):
                forged["environment_sha256"] = runtime_proof_environment_sha256(forged)
                with self.assertRaises(SandboxUnavailable) as caught:
                    validate_runtime_proof_environment(forged, expected=observed)
                self.assertEqual(caught.exception.code, code)

    def test_runtime_proof_rejects_nonempty_active_privilege(self) -> None:
        forged = json.loads(canonical_bytes(capture_runtime_proof_environment((ROOT,))))
        forged["capability_closure"]["effective"] = "0000000000000001"
        forged["capability_closure"]["active_privileges_empty"] = False
        forged["environment_sha256"] = runtime_proof_environment_sha256(forged)
        with self.assertRaises(SandboxUnavailable) as caught:
            validate_runtime_proof_environment(forged)
        self.assertEqual(caught.exception.code, "RUNTIME_PROOF_PRIVILEGE_NONEMPTY")

    def test_runtime_proof_accepts_inactive_seccomp_only_with_canonical_reason(self) -> None:
        for mode in (0, 1):
            with self.subTest(mode=mode):
                environment = self._runtime_environment_with_seccomp_mode(mode)
                self.assertIn("SECCOMP_FILTER_INACTIVE", environment["reason_codes"])
                self.assertIs(
                    validate_runtime_proof_environment(environment), environment
                )

    def test_runtime_proof_rejects_missing_inactive_seccomp_reason(self) -> None:
        for mode in (0, 1):
            with self.subTest(mode=mode):
                environment = self._runtime_environment_with_seccomp_mode(mode)
                environment["reason_codes"].remove("SECCOMP_FILTER_INACTIVE")
                environment["environment_sha256"] = runtime_proof_environment_sha256(
                    environment
                )
                with self.assertRaises(SandboxUnavailable) as caught:
                    validate_runtime_proof_environment(environment)
                self.assertEqual(
                    caught.exception.code,
                    "RUNTIME_PROOF_REASON_CODES_INVALID",
                )

    def test_runtime_proof_rejects_invalid_seccomp_modes(self) -> None:
        for mode in (-1, 3, False, True):
            with self.subTest(mode=mode):
                environment = self._runtime_environment_with_seccomp_mode(mode)
                with self.assertRaises(SandboxUnavailable) as caught:
                    validate_runtime_proof_environment(environment)
                self.assertEqual(
                    caught.exception.code,
                    "RUNTIME_PROOF_CAPABILITY_CLOSURE_INVALID",
                )

    def test_runtime_proof_rejects_expected_seccomp_mode_drift(self) -> None:
        observed = capture_runtime_proof_environment((ROOT,))
        drift_mode = 0 if observed["capability_closure"]["seccomp_mode"] != 0 else 1
        expected = self._runtime_environment_with_seccomp_mode(drift_mode)
        self.assertIs(validate_runtime_proof_environment(expected), expected)
        with self.assertRaises(SandboxUnavailable) as caught:
            validate_runtime_proof_environment(observed, expected=expected)
        self.assertEqual(
            caught.exception.code,
            "RUNTIME_PROOF_CAPABILITY_CLOSURE_MISMATCH",
        )

    def test_parse_evidence_requires_canonical_unique_key_json(self) -> None:
        backend, subject_hash, evidence = self._valid_sandbox_evidence()
        payload = canonical_bytes(evidence)
        self.assertEqual(
            parse_evidence(payload, backend, subject_hash),
            evidence,
        )

        reordered = json.loads(payload)
        first_key = next(iter(reordered))
        first_value = reordered.pop(first_key)
        reordered[first_key] = first_value
        reordered_payload = (
            json.dumps(
                reordered,
                ensure_ascii=False,
                sort_keys=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        duplicate_payload = (
            b'{"schema":"experiments7-linux-sandbox-evidence/v2",'
            + payload[1:]
        )
        cases = {
            "leading-whitespace": b" " + payload,
            "reordered-keys": reordered_payload,
            "duplicate-key": duplicate_payload,
        }
        for name, malformed in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(SandboxUnavailable) as caught:
                    parse_evidence(malformed, backend, subject_hash)
                self.assertEqual(caught.exception.code, "SANDBOX_EVIDENCE_INVALID")

    def test_runtime_proof_rejects_forged_readonly_mount_and_blocks_writable_mount(self) -> None:
        observed = capture_runtime_proof_environment((ROOT,))
        self.assertFalse(
            observed["mount_boundary"]["all_protected_mounts_read_only"]
        )
        self.assertIn("PROTECTED_MOUNT_WRITABLE", observed["reason_codes"])
        forged = json.loads(canonical_bytes(observed))
        forged["mount_boundary"]["protected_mounts"][0]["read_only"] = True
        forged["mount_boundary"]["all_protected_mounts_read_only"] = True
        forged["environment_sha256"] = runtime_proof_environment_sha256(forged)
        with self.assertRaises(SandboxUnavailable) as caught:
            validate_runtime_proof_environment(forged)
        self.assertEqual(caught.exception.code, "RUNTIME_PROOF_MOUNT_BOUNDARY_INVALID")

    def test_profile_gate_without_exact_index_is_blocked(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "scripts/validate_profile_dispatch.py"),
                "--json",
            ],
            cwd=ROOT,
            text=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        report = json.loads(completed.stdout)
        self.assertEqual(report["state"], "BLOCKED")
        self.assertEqual(report["reason_codes"], ["MISSING_PROFILE_DISPATCH_INDEX"])
        self.assertEqual(report["passed_profile_count"], 0)

    def test_output_claim_detects_inode_replacement(self) -> None:
        with tempfile.TemporaryDirectory(prefix="experiments7-output-claim-") as raw:
            target = Path(raw)
            output = target / "metrics.csv"
            claim = _claim_output(output, target=target)
            output.write_bytes(b"value\n7\n")
            record = _verified_output(output, target=target, claim=claim, seal=False)
            self.assertEqual(record["sha256"], sha256_file(output)[0])
            with self.assertRaises(FacadeError) as immutable:
                _verified_output(output, target=target, claim=claim, seal=True)
            self.assertEqual(
                immutable.exception.code,
                "EXECUTION_OUTPUT_NOT_EFFECTIVELY_IMMUTABLE",
            )

            output.chmod(0o600)
            output.unlink()
            output.write_bytes(b"value\n7\n")
            with self.assertRaises(FacadeError) as caught:
                _verified_output(output, target=target, claim=claim, seal=False)
            self.assertEqual(caught.exception.code, "EXECUTION_OUTPUT_IDENTITY_DRIFT")
            os.close(claim.descriptor)

    def test_output_claim_requires_absent_regular_destination(self) -> None:
        with tempfile.TemporaryDirectory(prefix="experiments7-output-collision-") as raw:
            target = Path(raw)
            output = target / "metrics.csv"
            output.write_bytes(b"occupied\n")
            with self.assertRaises(FacadeError) as caught:
                _claim_output(output, target=target)
            self.assertEqual(caught.exception.code, "EXECUTION_OUTPUT_COLLISION")


if __name__ == "__main__":
    unittest.main()
