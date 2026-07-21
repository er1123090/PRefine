from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import shutil
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provenance.strict_v6 import validate_source_pre_records  # noqa: E402
from strict_run import (  # noqa: E402
    StagePublisher,
    StrictRunError,
    WriterIdentity,
    default_writer_policy,
    validate_checkpoint_payload,
)
from strict_run.filesystem import RunDescriptorBinding, open_run_handle  # noqa: E402
import strict_run.g0_bootstrap as bootstrap_module  # noqa: E402
from strict_run.g0_bootstrap import bootstrap_g0_cp0  # noqa: E402


RUN_ID = "exp7-strict-v6-20260718T120000Z-" + "1" * 32


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            value.update(block)
    return value.hexdigest()


class StrictRunV6G0BootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="exp7-g0-bootstrap-v6-", dir=ROOT
        )
        self.base = Path(self.temporary.name)
        self.strict_parent = self.base / "paper_outputs" / "strict-runs"
        self.strict_parent.mkdir(parents=True)
        self.run_root = self.strict_parent / RUN_ID
        self.external = self.base / "external"
        self.external.mkdir()
        self.sources: dict[str, Path] = {}
        for root_id in ("experiments4", "experiments5", "experiments6"):
            root = self.base / root_id
            (root / "nested").mkdir(parents=True)
            (root / "nested" / "payload.jsonl").write_bytes(
                f"{root_id}-payload\n".encode("utf-8")
            )
            (root / "notes.txt").write_bytes(f"{root_id}-notes\n".encode("utf-8"))
            (root / "ignored-link").symlink_to(root / "notes.txt")
            self.sources[root_id] = root
        self.paper_dir = self.base / "_paper"
        self.paper_dir.mkdir()
        self.paper = self.paper_dir / "paper.pdf"
        self.paper.write_bytes(b"%PDF-synthetic-v6\n")
        self.source_digests = {
            path: digest(path)
            for root in self.sources.values()
            for path in root.rglob("*")
            if path.is_file() and not path.is_symlink()
        }
        self.paper_digest = digest(self.paper)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def bootstrap_kwargs(self, **overrides: object) -> dict[str, object]:
        values: dict[str, object] = {
            "strict_parent": str(self.strict_parent),
            "sealed_run_id": RUN_ID,
            "run_root": str(self.run_root),
            "external_dir": str(self.external),
            "project_id": "experiments7-g0-bootstrap-test",
            "owner_seed": "g0-bootstrap-owner",
            "source_roots": {
                key: str(value) for key, value in self.sources.items()
            },
            "paper_path": str(self.paper),
            "protected_executable": str(
                ROOT / "scripts" / "strict_run" / "g0_protected.py"
            ),
            "provider_path": str(ROOT / "scripts" / "g0" / "provider.py"),
            "comparison_tool": str(
                ROOT / "scripts" / "g0" / "compare_manifests.py"
            ),
            "validator_path": str(ROOT / "src" / "provenance" / "strict_v6.py"),
            "timeout_seconds": 60,
        }
        values.update(overrides)
        return values

    def descriptor_transport_fixture(
        self, suffix: str
    ) -> dict[str, object]:
        strict_parent = self.base / f"descriptor-{suffix}" / "strict-runs"
        strict_parent.mkdir(parents=True)
        run_root = strict_parent / RUN_ID
        run_root.mkdir()
        with open_run_handle(
            str(strict_parent), RUN_ID, str(run_root)
        ) as handle:
            run_binding = RunDescriptorBinding.capture(
                handle.parent_fd, handle.root_fd
            )
        publisher = StagePublisher(
            str(strict_parent),
            RUN_ID,
            str(run_root),
            WriterIdentity("g0", "g0:descriptor-test", "g0-writer"),
            default_writer_policy(),
            run_binding=run_binding,
        )
        payloads = {
            "g0_executable": b"raise SystemExit('canonical path reopened')\n",
            "configuration": bootstrap_module.canonical_json(
                {"schema": "descriptor-transport-test"}
            ),
            "provider": b"VALUE = 'held-provider'\n",
        }
        relative_paths = {
            "g0_executable": "frozen/g0_executable/g0_protected.py",
            "configuration": "frozen/configuration/g0-config.json",
            "provider": "frozen/provider/provider.py",
        }
        publications = {
            role: publisher.publish_bytes(relative_paths[role], payload)[-1]
            for role, payload in payloads.items()
        }
        runtime = bootstrap_module._runtime_identity()
        return {
            "strict_parent": str(strict_parent),
            "run_root": str(run_root),
            "run_binding": run_binding,
            "payloads": payloads,
            "relative_paths": relative_paths,
            "publications": publications,
            "runtime_sha256": runtime["python"]["sha256"],
        }

    def assert_canonical_swap_uses_sealed_descriptor(
        self, role: str, argv_index: int
    ) -> None:
        fixture = self.descriptor_transport_fixture(role)
        run_root = Path(fixture["run_root"])
        relative_paths = fixture["relative_paths"]
        payloads = fixture["payloads"]
        publications = fixture["publications"]
        canonical = run_root / relative_paths[role]
        replacement = f"replacement-{role}\n".encode("utf-8")
        logical_argv = [
            sys.executable,
            "-B",
            str(run_root / relative_paths["g0_executable"]),
            "--config",
            str(run_root / relative_paths["configuration"]),
            "--mode",
            "verify-envelope",
        ]

        def swapped_run(actual_argv: list[str], **kwargs: object) -> object:
            canonical.unlink()
            canonical.write_bytes(replacement)
            sealed_path = Path(actual_argv[argv_index])
            self.assertTrue(str(sealed_path).startswith("/proc/self/fd/"))
            self.assertEqual(sealed_path.read_bytes(), payloads[role])
            sealed_fd = int(sealed_path.name)
            self.assertEqual(
                bootstrap_module.fcntl.fcntl(
                    sealed_fd, bootstrap_module._F_GET_SEALS
                ),
                bootstrap_module._REQUIRED_MEMFD_SEALS,
            )
            self.assertIn(sealed_fd, kwargs["pass_fds"])
            return bootstrap_module.subprocess.CompletedProcess(
                actual_argv,
                0,
                stdout=bootstrap_module.canonical_json(
                    {
                        "schema": bootstrap_module.G0_RESULT_SCHEMA,
                        "mode": "verify-envelope",
                    }
                ),
                stderr=b"",
            )

        with patch.object(
            bootstrap_module.subprocess, "run", side_effect=swapped_run
        ):
            capture = bootstrap_module._protected_result(
                logical_argv[2],
                logical_argv[4],
                "verify-envelope",
                timeout_seconds=10,
                argv=logical_argv,
                environment={"PYTHONDONTWRITEBYTECODE": "1"},
                strict_parent=fixture["strict_parent"],
                sealed_run_id=RUN_ID,
                run_root=fixture["run_root"],
                run_binding=fixture["run_binding"],
                executable_publication=publications["g0_executable"],
                configuration_publication=publications["configuration"],
                provider_publication=publications["provider"],
                runtime_executable_sha256=fixture["runtime_sha256"],
            )
        self.assertEqual(canonical.read_bytes(), replacement)
        self.assertEqual(capture.argv, tuple(logical_argv))
        self.assertEqual(
            capture.descriptor_transport["logical_argv"], logical_argv
        )

    def test_script_swap_before_subprocess_uses_sealed_script(self) -> None:
        self.assert_canonical_swap_uses_sealed_descriptor("g0_executable", 2)

    def test_config_swap_before_subprocess_uses_sealed_config(self) -> None:
        self.assert_canonical_swap_uses_sealed_descriptor("configuration", 4)

    def test_sealed_memfd_rejects_mutation_immediately_before_seal(self) -> None:
        real_fcntl = bootstrap_module.fcntl.fcntl
        mutated_fds: list[int] = []

        def mutate_before_seal(
            fd: int, command: int, *args: object
        ) -> object:
            if command == bootstrap_module._F_ADD_SEALS:
                os.pwrite(fd, b"X", 0)
                mutated_fds.append(fd)
            return real_fcntl(fd, command, *args)

        with (
            patch.object(
                bootstrap_module.fcntl,
                "fcntl",
                side_effect=mutate_before_seal,
            ),
            self.assertRaises(StrictRunError) as caught,
        ):
            bootstrap_module._sealed_memfd("seal-race", b"trusted", 0o400)
        self.assertEqual(
            caught.exception.code, "G0_DESCRIPTOR_TRANSPORT_INVALID"
        )
        self.assertEqual(len(mutated_fds), 1)
        with self.assertRaises(OSError):
            os.fstat(mutated_fds[0])

    def test_sealed_memfd_rejects_unsupported_architecture_before_syscall(
        self,
    ) -> None:
        with (
            patch.object(
                bootstrap_module.platform, "machine", return_value="aarch64"
            ),
            patch.object(bootstrap_module.ctypes, "CDLL") as libc,
            self.assertRaises(StrictRunError) as caught,
        ):
            bootstrap_module._sealed_memfd("unsupported", b"payload", 0o400)
        self.assertEqual(
            caught.exception.code, "G0_DESCRIPTOR_TRANSPORT_UNAVAILABLE"
        )
        libc.assert_not_called()

    def test_bootstrap_seals_cp0_with_frozen_provider_only_protected_reads(self) -> None:
        result = bootstrap_g0_cp0(
            strict_parent=str(self.strict_parent),
            sealed_run_id=RUN_ID,
            run_root=str(self.run_root),
            external_dir=str(self.external),
            project_id="experiments7-g0-bootstrap-test",
            owner_seed="g0-bootstrap-owner",
            source_roots={key: str(value) for key, value in self.sources.items()},
            paper_path=str(self.paper),
            protected_executable=str(ROOT / "scripts" / "strict_run" / "g0_protected.py"),
            provider_path=str(ROOT / "scripts" / "g0" / "provider.py"),
            comparison_tool=str(ROOT / "scripts" / "g0" / "compare_manifests.py"),
            validator_path=str(ROOT / "src" / "provenance" / "strict_v6.py"),
            timeout_seconds=60,
        )

        self.assertEqual(result.sealed_run_id, RUN_ID)
        self.assertTrue((self.run_root / "checkpoints" / "cp0.json").is_file())
        payload = json.loads((self.run_root / "checkpoints" / "cp0.json").read_bytes())
        validated = validate_checkpoint_payload(
            payload,
            expected_sealed_run_id=RUN_ID,
            expected_run_root=str(self.run_root),
            expected_checkpoint=0,
            policy=default_writer_policy(),
        )
        self.assertEqual(validated["semantic_bindings"]["checkpoint"], 0)
        self.assertEqual(result.checkpoint.payload, payload)

        source_rows = [
            json.loads(line)
            for line in (self.run_root / "manifests" / "source-pre.jsonl").read_bytes().splitlines()
        ]
        normalized = validate_source_pre_records(source_rows)
        self.assertEqual(len(normalized), 6)
        self.assertEqual({row["root_id"] for row in normalized}, set(self.sources))
        self.assertTrue(all(row["type"] == "regular" for row in normalized))

        paper = json.loads((self.run_root / "manifests" / "paper-pre.json").read_bytes())
        self.assertEqual(paper["sha256"], self.paper_digest)
        self.assertEqual(paper["paper_path"], str(self.paper))
        self.assertEqual(paper["provider"]["landlock_abi"], 4)

        by_path = {
            publication.evidence.relative_path: publication
            for publication in result.stage_publications
        }
        for relative, operation in (
            ("manifests/source-pre.jsonl", "source-pre"),
            ("manifests/paper-pre.json", "paper-pre"),
        ):
            context = by_path[relative].transcript["context"]
            self.assertEqual(
                context["schema"],
                "experiments7-cp0-protected-read-attestation/v6",
            )
            self.assertEqual(context["operation"], operation)
            self.assertEqual(len(context), 12)
            prefix = "source" if operation == "source-pre" else "paper"
            stdout_path = Path(result.external_paths[f"{prefix}_stdout"])
            execution_path = Path(result.external_paths[f"{prefix}_execution"])
            self.assertEqual(context["child_stdout_sha256"], digest(stdout_path))
            self.assertEqual(
                context["child_execution_sha256"], digest(execution_path)
            )
            stdout = json.loads(stdout_path.read_bytes())
            execution = json.loads(execution_path.read_bytes())
            self.assertEqual(
                stdout["paper_write_probe"]["target_path"],
                str(self.paper),
            )
            self.assertEqual(stdout["producer_attestation"], {
                key: context[key]
                for key in context
                if key not in {
                    "child_stdout_sha256",
                    "child_execution_sha256",
                    "child_execution_evidence",
                }
            })
            self.assertEqual(execution["mode"], operation)
            self.assertEqual(execution["stdout_evidence"]["path"], str(stdout_path))
            self.assertEqual(execution["stdout_evidence"]["sha256"], digest(stdout_path))
            self.assertEqual(
                context["child_execution_evidence"]["path"],
                str(execution_path),
            )
            self.assertEqual(
                context["child_execution_evidence"]["sha256"],
                digest(execution_path),
            )
            frozen_argv = json.loads(
                (
                    self.run_root
                    / "frozen"
                    / ("pre_argv" if operation == "source-pre" else "post_argv")
                    / "argv.json"
                ).read_bytes()
            )["argv"]
            self.assertEqual(execution["argv"], frozen_argv)
            transport = execution["descriptor_transport"]
            self.assertEqual(
                transport["schema"],
                bootstrap_module.G0_DESCRIPTOR_TRANSPORT_SCHEMA,
            )
            self.assertEqual(transport["logical_argv"], frozen_argv)
            self.assertEqual(
                transport["executed_argv_sha256"],
                bootstrap_module.sha256_bytes(
                    bootstrap_module.canonical_json(transport["executed_argv"])
                ),
            )
            self.assertEqual(
                set(transport["artifacts"]),
                {"g0_executable", "configuration", "provider"},
            )
            self.assertTrue(
                all(
                    row["seals"] == bootstrap_module._REQUIRED_MEMFD_SEALS
                    for row in transport["artifacts"].values()
                )
            )
            runtime_identity = json.loads(
                (
                    self.run_root
                    / "frozen"
                    / "runtime_identity"
                    / "runtime.json"
                ).read_bytes()
            )
            self.assertEqual(
                execution["runtime_executable_sha256"],
                runtime_identity["python"]["sha256"],
            )
            self.assertEqual(frozen_argv[-1], "--producer-binding-stdin")

        producer_binding_path = Path(result.external_paths["producer_binding"])
        producer_binding_raw = producer_binding_path.read_bytes()
        producer_binding = json.loads(producer_binding_raw)
        self.assertEqual(
            producer_binding_raw,
            bootstrap_module.canonical_json(producer_binding),
        )
        self.assertEqual(
            producer_binding["acceptance_transcript"],
            by_path["reservation-acceptance.json"].transcript,
        )
        self.assertEqual(
            producer_binding["envelope_transcript"],
            by_path["frozen/envelope_evidence/artifact.json"].transcript,
        )
        for relative in (
            "frozen/pre_argv/argv.json",
            "frozen/post_argv/argv.json",
        ):
            argv = json.loads((self.run_root / relative).read_bytes())["argv"]
            self.assertEqual(argv[-1], "--producer-binding-stdin")
            self.assertNotIn(str(producer_binding_path), argv)

        reservation = json.loads((self.run_root / "reservation.json").read_bytes())
        reservation_argv = reservation["toolchain"]["canonical_argv"]
        for required in (
            str(self.strict_parent),
            str(self.run_root),
            str(self.external),
            str(self.paper),
            str(ROOT / "scripts" / "strict_run" / "g0_protected.py"),
            str(ROOT / "scripts" / "g0" / "provider.py"),
            str(ROOT / "scripts" / "g0" / "compare_manifests.py"),
            str(ROOT / "src" / "provenance" / "strict_v6.py"),
            "60",
        ):
            self.assertIn(required, reservation_argv)

        audit = json.loads(
            (self.run_root / "frozen" / "runtime_identity" / "envelope-audit.json").read_bytes()
        )
        self.assertTrue(audit["python_dont_write_bytecode"])
        self.assertEqual(audit["pythonhashseed"], "0")
        self.assertTrue(audit["paper_write_probe"]["denied"])
        self.assertEqual(
            audit["paper_write_probe"]["target_path"], str(self.paper)
        )
        self.assertEqual(audit["provider_evidence"]["landlock_abi"], 4)

        inventory = json.loads((self.run_root / "frozen" / "inventory.json").read_bytes())
        self.assertEqual(
            set(inventory["roles"]),
            {
                "g0_executable",
                "provider",
                "configuration",
                "pre_argv",
                "post_argv",
                "comparison_tool",
                "runtime_identity",
                "validators",
                "environment_allowlist",
                "bundle_lock",
                "envelope_evidence",
                "writer_policy",
            },
        )
        self.assertTrue(all(inventory["roles"].values()))

        self.assertEqual({publication.evidence.relative_path for publication in result.stage_publications} & {
            "reservation.json",
            "reservation-acceptance.json",
            "manifests/source-pre.jsonl",
            "manifests/paper-pre.json",
            "frozen/inventory.json",
        }, {
            "reservation.json",
            "reservation-acceptance.json",
            "manifests/source-pre.jsonl",
            "manifests/paper-pre.json",
            "frozen/inventory.json",
        })
        for path in result.external_paths.values():
            current = Path(path)
            self.assertTrue(current.is_file())
            self.assertEqual(stat.S_IMODE(current.stat().st_mode), 0o444)
            self.assertNotIn(str(self.run_root), str(current))

        self.assertEqual(digest(self.paper), self.paper_digest)
        self.assertEqual({path: digest(path) for path in self.source_digests}, self.source_digests)

    def test_mismatched_child_attestation_fails_before_manifest_publication(self) -> None:
        protected_result = bootstrap_module._protected_result

        def forged_result(
            executable: str,
            config: str,
            mode: str,
            *,
            timeout_seconds: int,
            argv: list[str],
            environment: dict[str, str],
            producer_binding: bytes | None = None,
            **descriptor_kwargs: object,
        ) -> object:
            capture = protected_result(
                executable,
                config,
                mode,
                timeout_seconds=timeout_seconds,
                argv=argv,
                environment=environment,
                producer_binding=producer_binding,
                **descriptor_kwargs,
            )
            if mode == "source-pre":
                result = dict(capture.result)
                attestation = dict(result["producer_attestation"])
                attestation["envelope_transcript_sha256"] = "0" * 64
                result["producer_attestation"] = attestation
                return replace(
                    capture,
                    result=result,
                    stdout=bootstrap_module.canonical_json(result),
                )
            return capture

        with patch.object(
            bootstrap_module,
            "_protected_result",
            side_effect=forged_result,
        ), self.assertRaisesRegex(
            RuntimeError,
            "producer attestation binding differs",
        ):
            bootstrap_g0_cp0(**self.bootstrap_kwargs())
        self.assertFalse((self.run_root / "manifests" / "source-pre.jsonl").exists())
        self.assertFalse((self.run_root / "manifests" / "paper-pre.json").exists())

    def test_probe_target_mismatch_fails_before_affected_manifest_publication(
        self,
    ) -> None:
        for forged_mode in ("source-pre", "paper-pre"):
            with self.subTest(mode=forged_mode), tempfile.TemporaryDirectory(
                prefix=f"exp7-g0-probe-{forged_mode}-",
                dir=self.base,
            ) as branch:
                branch_path = Path(branch)
                strict_parent = branch_path / "paper_outputs" / "strict-runs"
                strict_parent.mkdir(parents=True)
                run_root = strict_parent / RUN_ID
                external = branch_path / "external"
                external.mkdir()
                protected_result = bootstrap_module._protected_result

                def forged_result(
                    executable: str,
                    config: str,
                    mode: str,
                    *,
                    timeout_seconds: int,
                    argv: list[str],
                    environment: dict[str, str],
                    producer_binding: bytes | None = None,
                    **descriptor_kwargs: object,
                ) -> object:
                    capture = protected_result(
                        executable,
                        config,
                        mode,
                        timeout_seconds=timeout_seconds,
                        argv=argv,
                        environment=environment,
                        producer_binding=producer_binding,
                        **descriptor_kwargs,
                    )
                    if mode != forged_mode:
                        return capture
                    result = dict(capture.result)
                    probe = dict(result["paper_write_probe"])
                    probe["target_path"] = str(self.paper_dir / "other.pdf")
                    result["paper_write_probe"] = probe
                    return replace(
                        capture,
                        result=result,
                        stdout=bootstrap_module.canonical_json(result),
                    )

                with patch.object(
                    bootstrap_module,
                    "_protected_result",
                    side_effect=forged_result,
                ), self.assertRaisesRegex(
                    RuntimeError, "protected .* result is invalid"
                ):
                    bootstrap_g0_cp0(
                        **self.bootstrap_kwargs(
                            strict_parent=str(strict_parent),
                            run_root=str(run_root),
                            external_dir=str(external),
                        )
                    )
                affected_manifest = (
                    "source-pre.jsonl"
                    if forged_mode == "source-pre"
                    else "paper-pre.json"
                )
                self.assertFalse(
                    (run_root / "manifests" / affected_manifest).exists()
                )

    def test_actual_subprocess_uses_frozen_argv_environment_and_binding_stdin(
        self,
    ) -> None:
        real_run = bootstrap_module.subprocess.run
        with patch.object(
            bootstrap_module.subprocess, "run", wraps=real_run
        ) as runner:
            result = bootstrap_g0_cp0(**self.bootstrap_kwargs())
        calls: dict[str, object] = {}
        for call in runner.call_args_list:
            argv = call.args[0]
            mode = argv[argv.index("--mode") + 1]
            calls[mode] = call
        self.assertEqual(set(calls), {"verify-envelope", "source-pre", "paper-pre"})
        environment = json.loads(
            (
                self.run_root
                / "frozen"
                / "environment_allowlist"
                / "environment.json"
            ).read_bytes()
        )["environment"]
        binding = Path(result.external_paths["producer_binding"]).read_bytes()
        for mode, role in (("source-pre", "pre_argv"), ("paper-pre", "post_argv")):
            call = calls[mode]
            frozen = json.loads(
                (self.run_root / "frozen" / role / "argv.json").read_bytes()
            )["argv"]
            prefix = "source" if mode == "source-pre" else "paper"
            execution = json.loads(
                Path(result.external_paths[f"{prefix}_execution"]).read_bytes()
            )
            transport = execution["descriptor_transport"]
            self.assertEqual(transport["logical_argv"], frozen)
            self.assertEqual(call.args[0], transport["executed_argv"])
            self.assertEqual(
                call.kwargs["executable"], transport["runtime"]["proc_path"]
            )
            expected_fds = {
                transport["runtime"]["fd"],
                *(row["fd"] for row in transport["artifacts"].values()),
            }
            self.assertEqual(set(call.kwargs["pass_fds"]), expected_fds)
            self.assertEqual(call.kwargs["env"], environment)
            self.assertEqual(call.kwargs["input"], binding)
            self.assertNotIn("stdin", call.kwargs)

    def test_forged_child_argv_fails_before_any_manifest_or_child_evidence(
        self,
    ) -> None:
        protected_result = bootstrap_module._protected_result

        def forged_argv(
            executable: str,
            config: str,
            mode: str,
            *,
            timeout_seconds: int,
            argv: list[str],
            environment: dict[str, str],
            producer_binding: bytes | None = None,
            **descriptor_kwargs: object,
        ) -> object:
            capture = protected_result(
                executable,
                config,
                mode,
                timeout_seconds=timeout_seconds,
                argv=argv,
                environment=environment,
                producer_binding=producer_binding,
                **descriptor_kwargs,
            )
            if mode == "source-pre":
                return replace(capture, argv=capture.argv + ("--forged",))
            return capture

        with patch.object(
            bootstrap_module, "_protected_result", side_effect=forged_argv
        ), self.assertRaisesRegex(RuntimeError, "execution evidence differs"):
            bootstrap_g0_cp0(**self.bootstrap_kwargs())
        for relative in (
            "manifests/source-pre.jsonl",
            "manifests/paper-pre.json",
        ):
            self.assertFalse((self.run_root / relative).exists())
        self.assertFalse((self.external / "source-stdout" / "result.json").exists())
        self.assertFalse(
            (self.external / "source-execution" / "execution.json").exists()
        )

    def test_external_overlap_is_rejected_before_reservation_or_source_mutation(
        self,
    ) -> None:
        overlap = self.sources["experiments4"] / "external-output"
        overlap.mkdir()
        with self.assertRaisesRegex(
            Exception, "external output directory overlaps"
        ):
            bootstrap_g0_cp0(
                **self.bootstrap_kwargs(external_dir=str(overlap))
            )
        self.assertFalse(self.run_root.exists())
        self.assertEqual(
            {path: digest(path) for path in self.source_digests},
            self.source_digests,
        )
        self.assertEqual(digest(self.paper), self.paper_digest)

    def test_strict_storage_inside_each_protected_directory_is_rejected_before_mutation(
        self,
    ) -> None:
        for target in (
            "experiments4",
            "experiments5",
            "experiments6",
            "paper",
        ):
            with self.subTest(target=target), tempfile.TemporaryDirectory(
                prefix=f"strict-inside-{target}-",
                dir=self.base,
            ) as branch:
                branch_path = Path(branch)
                sources: dict[str, Path] = {}
                source_digests: dict[Path, str] = {}
                for root_id in (
                    "experiments4",
                    "experiments5",
                    "experiments6",
                ):
                    root = branch_path / root_id
                    root.mkdir()
                    payload = root / "payload.jsonl"
                    payload.write_bytes(f"{root_id}\n".encode("utf-8"))
                    sources[root_id] = root
                    source_digests[payload] = digest(payload)
                paper_dir = branch_path / "_paper"
                paper_dir.mkdir()
                paper = paper_dir / "paper.pdf"
                paper.write_bytes(b"%PDF-overlap-test\n")
                paper_digest = digest(paper)
                protected = (
                    paper_dir if target == "paper" else sources[target]
                )
                strict_parent = protected / "strict-runs"
                strict_parent.mkdir()
                run_root = strict_parent / RUN_ID
                external = branch_path / "external"
                external.mkdir()

                with (
                    patch.object(
                        bootstrap_module, "_trusted_provider_bytes"
                    ) as provider_reader,
                    patch.object(
                        bootstrap_module, "_prepare_child_output_parents"
                    ) as prepare_outputs,
                    patch.object(
                        bootstrap_module, "reserve_strict_run"
                    ) as reserve_run,
                    self.assertRaisesRegex(
                        Exception,
                        "strict storage overlaps a protected directory",
                    ),
                ):
                    bootstrap_g0_cp0(
                        **self.bootstrap_kwargs(
                            strict_parent=str(strict_parent),
                            run_root=str(run_root),
                            external_dir=str(external),
                            source_roots={
                                key: str(value)
                                for key, value in sources.items()
                            },
                            paper_path=str(paper),
                        )
                    )

                provider_reader.assert_not_called()
                prepare_outputs.assert_not_called()
                reserve_run.assert_not_called()
                self.assertFalse(run_root.exists())
                self.assertEqual(list(external.iterdir()), [])
                self.assertEqual(
                    {path: digest(path) for path in source_digests},
                    source_digests,
                )
                self.assertEqual(digest(paper), paper_digest)

    def test_each_protected_directory_inside_strict_storage_is_rejected_before_mutation(
        self,
    ) -> None:
        for target in (
            "experiments4",
            "experiments5",
            "experiments6",
            "paper",
        ):
            with self.subTest(target=target), tempfile.TemporaryDirectory(
                prefix=f"protected-inside-{target}-",
                dir=self.base,
            ) as branch:
                branch_path = Path(branch)
                strict_parent = branch_path / "strict-runs"
                strict_parent.mkdir()
                run_root = strict_parent / RUN_ID
                external = branch_path / "external"
                external.mkdir()
                sources: dict[str, Path] = {}
                source_digests: dict[Path, str] = {}
                for root_id in (
                    "experiments4",
                    "experiments5",
                    "experiments6",
                ):
                    root = (
                        strict_parent / f"protected-{root_id}"
                        if target == root_id
                        else branch_path / root_id
                    )
                    root.mkdir()
                    payload = root / "payload.jsonl"
                    payload.write_bytes(f"{root_id}\n".encode("utf-8"))
                    sources[root_id] = root
                    source_digests[payload] = digest(payload)
                paper_dir = (
                    strict_parent / "protected-paper"
                    if target == "paper"
                    else branch_path / "_paper"
                )
                paper_dir.mkdir()
                paper = paper_dir / "paper.pdf"
                paper.write_bytes(b"%PDF-overlap-test\n")
                paper_digest = digest(paper)

                with (
                    patch.object(
                        bootstrap_module, "_trusted_provider_bytes"
                    ) as provider_reader,
                    patch.object(
                        bootstrap_module, "_prepare_child_output_parents"
                    ) as prepare_outputs,
                    patch.object(
                        bootstrap_module, "reserve_strict_run"
                    ) as reserve_run,
                    self.assertRaisesRegex(
                        Exception,
                        "strict storage overlaps a protected directory",
                    ),
                ):
                    bootstrap_g0_cp0(
                        **self.bootstrap_kwargs(
                            strict_parent=str(strict_parent),
                            run_root=str(run_root),
                            external_dir=str(external),
                            source_roots={
                                key: str(value)
                                for key, value in sources.items()
                            },
                            paper_path=str(paper),
                        )
                    )

                provider_reader.assert_not_called()
                prepare_outputs.assert_not_called()
                reserve_run.assert_not_called()
                self.assertFalse(run_root.exists())
                self.assertEqual(list(external.iterdir()), [])
                self.assertEqual(
                    {path: digest(path) for path in source_digests},
                    source_digests,
                )
                self.assertEqual(digest(paper), paper_digest)

    def test_inode_ancestry_alias_is_rejected_before_mutation(self) -> None:
        directory_inode_chain = bootstrap_module._directory_inode_chain

        def aliased_chain(path: str, label: str) -> tuple[tuple[int, int], ...]:
            chain = directory_inode_chain(path, label)
            if label == "strict_parent":
                return (*chain[:-1], (987654, 123456))
            if label == "protected_directory[0]":
                return (*chain, (987654, 123456))
            return chain

        with (
            patch.object(
                bootstrap_module,
                "_directory_inode_chain",
                side_effect=aliased_chain,
            ),
            patch.object(
                bootstrap_module, "_trusted_provider_bytes"
            ) as provider_reader,
            patch.object(
                bootstrap_module, "_prepare_child_output_parents"
            ) as prepare_outputs,
            patch.object(
                bootstrap_module, "reserve_strict_run"
            ) as reserve_run,
            self.assertRaisesRegex(
                Exception,
                "strict storage aliases a protected directory ancestry",
            ),
        ):
            bootstrap_g0_cp0(**self.bootstrap_kwargs())

        provider_reader.assert_not_called()
        prepare_outputs.assert_not_called()
        reserve_run.assert_not_called()
        self.assertFalse(self.run_root.exists())
        self.assertEqual(list(self.external.iterdir()), [])
        self.assertEqual(
            {path: digest(path) for path in self.source_digests},
            self.source_digests,
        )
        self.assertEqual(digest(self.paper), self.paper_digest)

    def test_each_repository_input_substitution_is_rejected_before_read_or_mutation(
        self,
    ) -> None:
        protected_file = self.sources["experiments4"] / "notes.txt"
        for argument in (
            "protected_executable",
            "provider_path",
            "comparison_tool",
            "validator_path",
        ):
            with (
                self.subTest(argument=argument),
                patch.object(
                    bootstrap_module, "_read_absolute_regular_nofollow"
                ) as regular_reader,
                patch.object(
                    bootstrap_module, "_trusted_provider_bytes"
                ) as provider_reader,
                patch.object(
                    bootstrap_module, "_validate_repository_tool_mount"
                ) as tool_mount_validator,
                patch.object(
                    bootstrap_module, "_capture_mount_boundary"
                ) as mount_capture,
                patch.object(
                    bootstrap_module, "_prepare_child_output_parents"
                ) as prepare_outputs,
                patch.object(
                    bootstrap_module, "reserve_strict_run"
                ) as reserve_run,
                self.assertRaisesRegex(Exception, "canonical repository path"),
            ):
                bootstrap_g0_cp0(
                    **self.bootstrap_kwargs(**{argument: str(protected_file)})
                )

            regular_reader.assert_not_called()
            provider_reader.assert_not_called()
            tool_mount_validator.assert_not_called()
            mount_capture.assert_not_called()
            prepare_outputs.assert_not_called()
            reserve_run.assert_not_called()
            self.assertFalse(self.run_root.exists())
            self.assertEqual(list(self.external.iterdir()), [])
            self.assertEqual(
                {path: digest(path) for path in self.source_digests},
                self.source_digests,
            )
            self.assertEqual(digest(self.paper), self.paper_digest)

    def test_preexisting_run_root_is_rejected_before_read_or_mutation(self) -> None:
        self.run_root.mkdir()
        with (
            patch.object(
                bootstrap_module, "_read_absolute_regular_nofollow"
            ) as regular_reader,
            patch.object(
                bootstrap_module, "_trusted_provider_bytes"
            ) as provider_reader,
            patch.object(
                bootstrap_module, "_capture_mount_boundary"
            ) as mount_capture,
            patch.object(
                bootstrap_module, "_prepare_child_output_parents"
            ) as prepare_outputs,
            patch.object(
                bootstrap_module, "reserve_strict_run"
            ) as reserve_run,
            self.assertRaisesRegex(Exception, "run_root must not already exist"),
        ):
            bootstrap_g0_cp0(**self.bootstrap_kwargs())

        regular_reader.assert_not_called()
        provider_reader.assert_not_called()
        mount_capture.assert_not_called()
        prepare_outputs.assert_not_called()
        reserve_run.assert_not_called()
        self.assertEqual(list(self.run_root.iterdir()), [])
        self.assertEqual(list(self.external.iterdir()), [])
        self.assertEqual(
            {path: digest(path) for path in self.source_digests},
            self.source_digests,
        )
        self.assertEqual(digest(self.paper), self.paper_digest)

    def test_bind_mounted_protected_descendant_view_is_rejected_before_mutation(
        self,
    ) -> None:
        raw, namespace, entries = bootstrap_module._read_stable_mountinfo()
        original_evidence = bootstrap_module._directory_mount_evidence
        common_evidence = original_evidence(
            str(self.strict_parent), "strict_parent"
        )
        common_mount_id = common_evidence[1]
        common_entry = next(
            entry for entry in entries if entry.mount_id == common_mount_id
        )

        for offset, target in enumerate(
            (self.strict_parent, self.external), start=1
        ):
            hostile_mount_id = max(entry.mount_id for entry in entries) + offset
            hostile = bootstrap_module._MountInfoEntry(
                mount_id=hostile_mount_id,
                parent_mount_id=common_mount_id,
                device=common_entry.device,
                mount_root=str(self.sources["experiments4"] / "nested"),
                mount_point=str(target),
            )

            def hostile_evidence(
                path: str, label: str, *, _target: str = str(target)
            ) -> tuple[str, int, int, int]:
                evidence = original_evidence(path, label)
                if path == _target:
                    return (
                        evidence[0],
                        hostile_mount_id,
                        evidence[2],
                        evidence[3],
                    )
                return evidence

            with (
                self.subTest(target=str(target)),
                patch.object(
                    bootstrap_module,
                    "_read_stable_mountinfo",
                    return_value=(raw, namespace, (*entries, hostile)),
                ),
                patch.object(
                    bootstrap_module,
                    "_directory_mount_evidence",
                    side_effect=hostile_evidence,
                ),
                patch.object(
                    bootstrap_module, "_read_absolute_regular_nofollow"
                ) as regular_reader,
                patch.object(
                    bootstrap_module, "_trusted_provider_bytes"
                ) as provider_reader,
                patch.object(
                    bootstrap_module, "_validate_repository_tool_mount"
                ) as tool_mount_validator,
                patch.object(
                    bootstrap_module, "_prepare_child_output_parents"
                ) as prepare_outputs,
                patch.object(
                    bootstrap_module, "reserve_strict_run"
                ) as reserve_run,
                self.assertRaisesRegex(Exception, "share one mount ID"),
            ):
                bootstrap_g0_cp0(**self.bootstrap_kwargs())

            regular_reader.assert_not_called()
            provider_reader.assert_not_called()
            tool_mount_validator.assert_not_called()
            prepare_outputs.assert_not_called()
            reserve_run.assert_not_called()
            self.assertFalse(self.run_root.exists())
            self.assertEqual(list(self.external.iterdir()), [])
            self.assertEqual(
                {path: digest(path) for path in self.source_digests},
                self.source_digests,
            )
            self.assertEqual(digest(self.paper), self.paper_digest)

    def test_nested_mount_below_writable_root_is_rejected_before_mutation(
        self,
    ) -> None:
        raw, namespace, entries = bootstrap_module._read_stable_mountinfo()
        common_mount_id = bootstrap_module._directory_mount_evidence(
            str(self.external), "external"
        )[1]
        common_entry = next(
            entry for entry in entries if entry.mount_id == common_mount_id
        )
        hostile = bootstrap_module._MountInfoEntry(
            mount_id=max(entry.mount_id for entry in entries) + 1,
            parent_mount_id=common_mount_id,
            device=common_entry.device,
            mount_root=str(self.sources["experiments4"] / "nested"),
            mount_point=str(self.external / "hostile-nested"),
        )
        with (
            patch.object(
                bootstrap_module,
                "_read_stable_mountinfo",
                return_value=(raw, namespace, (*entries, hostile)),
            ),
            patch.object(
                bootstrap_module, "_read_absolute_regular_nofollow"
            ) as regular_reader,
            patch.object(
                bootstrap_module, "_trusted_provider_bytes"
            ) as provider_reader,
            patch.object(
                bootstrap_module, "_validate_repository_tool_mount"
            ) as tool_mount_validator,
            patch.object(
                bootstrap_module, "_prepare_child_output_parents"
            ) as prepare_outputs,
            patch.object(
                bootstrap_module, "reserve_strict_run"
            ) as reserve_run,
            self.assertRaisesRegex(Exception, "contains a nested mount"),
        ):
            bootstrap_g0_cp0(**self.bootstrap_kwargs())

        regular_reader.assert_not_called()
        provider_reader.assert_not_called()
        tool_mount_validator.assert_not_called()
        prepare_outputs.assert_not_called()
        reserve_run.assert_not_called()
        self.assertFalse(self.run_root.exists())
        self.assertEqual(list(self.external.iterdir()), [])
        self.assertEqual(
            {path: digest(path) for path in self.source_digests},
            self.source_digests,
        )
        self.assertEqual(digest(self.paper), self.paper_digest)

    def test_all_mount_capability_sets_must_exclude_sys_admin(self) -> None:
        labels = ("CapInh", "CapPrm", "CapEff", "CapAmb")

        def status(active: str | None) -> bytes:
            return b"".join(
                f"{label}:\t{'0000000000200000' if label == active else '0000000000000000'}\n".encode(
                    "ascii"
                )
                for label in labels
            )

        with patch.object(
            bootstrap_module, "_read_proc_self_bytes", return_value=status(None)
        ):
            bootstrap_module._require_no_sys_admin()
        for label in labels:
            with (
                self.subTest(label=label),
                patch.object(
                    bootstrap_module,
                    "_read_proc_self_bytes",
                    return_value=status(label),
                ),
                self.assertRaisesRegex(Exception, "CAP_SYS_ADMIN is present"),
            ):
                bootstrap_module._require_no_sys_admin()

    def test_repository_reader_checks_mount_id_before_payload_read(self) -> None:
        provider = str(ROOT / "scripts" / "g0" / "provider.py")
        mount_id = bootstrap_module._directory_mount_evidence(
            str(ROOT), "repository"
        )[1]
        with (
            patch.object(
                bootstrap_module, "_fd_mount_id", return_value=mount_id + 1
            ),
            patch.object(bootstrap_module.os, "read") as payload_reader,
            self.assertRaisesRegex(Exception, "pinned repository mount"),
        ):
            bootstrap_module._read_absolute_regular_nofollow(
                provider,
                "provider",
                expected_mount_id=mount_id,
            )
        payload_reader.assert_not_called()

    def test_child_output_parent_checks_mount_before_first_mkdir(self) -> None:
        mount_id = bootstrap_module._directory_mount_evidence(
            str(self.external), "external"
        )[1]
        with (
            patch.object(bootstrap_module.os, "mkdir") as mkdir,
            self.assertRaisesRegex(Exception, "not on the pinned mount"),
        ):
            bootstrap_module._prepare_child_output_parents(
                str(self.external), expected_mount_id=mount_id + 1
            )
        mkdir.assert_not_called()
        self.assertEqual(list(self.external.iterdir()), [])

    def test_mountinfo_parser_is_exact_and_decodes_only_standard_escapes(
        self,
    ) -> None:
        raw = (
            b"17 1 8:2 /root\\040space "
            b"/mnt\\011tab\\012line\\134slash rw - ext4 /dev/test rw\n"
        )
        parsed = bootstrap_module._parse_mountinfo(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0].mount_root, "/root space")
        self.assertEqual(parsed[0].mount_point, "/mnt\ttab\nline\\slash")
        malformed = (
            raw[:-1],
            raw.replace(b"\\040", b"\\777"),
            raw + b"17 1 8:2 / /other rw - ext4 /dev/test rw\n",
            raw + b"18 1 8:2 / /mnt\\011tab\\012line\\134slash rw - ext4 /dev/test rw\n",
        )
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(Exception):
                bootstrap_module._parse_mountinfo(value)

    def test_mountinfo_bytes_and_namespace_must_stay_stable(self) -> None:
        raw = b"1 0 8:1 / / rw - ext4 /dev/test rw\n"
        namespace = (4, 100, "mnt:[100]")
        with (
            patch.object(
                bootstrap_module,
                "_mount_namespace_identity",
                side_effect=(namespace, namespace, namespace),
            ),
            patch.object(
                bootstrap_module,
                "_read_proc_self_bytes",
                side_effect=(raw, raw.replace(b"rw\n", b"ro\n")),
            ),
            self.assertRaisesRegex(Exception, "mountinfo changed during capture"),
        ):
            bootstrap_module._read_stable_mountinfo()

        with (
            patch.object(
                bootstrap_module,
                "_mount_namespace_identity",
                side_effect=(
                    namespace,
                    (4, 101, "mnt:[101]"),
                    (4, 101, "mnt:[101]"),
                ),
            ),
            patch.object(
                bootstrap_module,
                "_read_proc_self_bytes",
                side_effect=(raw, raw),
            ),
            self.assertRaisesRegex(Exception, "mountinfo changed during capture"),
        ):
            bootstrap_module._read_stable_mountinfo()

    def test_reservation_argv_binds_timeout_and_all_bootstrap_inputs(self) -> None:
        roots = bootstrap_module._validate_source_roots(
            {key: str(value) for key, value in self.sources.items()}
        )
        common = {
            "strict_parent": str(self.strict_parent),
            "sealed_run_id": RUN_ID,
            "run_root": str(self.run_root),
            "external_dir": str(self.external),
            "project_id": "experiments7-g0-bootstrap-test",
            "owner_seed": "g0-bootstrap-owner",
            "roots": roots,
            "paper_path": str(self.paper),
            "protected_executable": str(
                ROOT / "scripts" / "strict_run" / "g0_protected.py"
            ),
            "provider_path": str(ROOT / "scripts" / "g0" / "provider.py"),
            "comparison_tool": str(
                ROOT / "scripts" / "g0" / "compare_manifests.py"
            ),
            "validator_path": str(ROOT / "src" / "provenance" / "strict_v6.py"),
        }
        first = bootstrap_module._reservation_argv(
            **common, timeout_seconds=60
        )
        second = bootstrap_module._reservation_argv(
            **common, timeout_seconds=61
        )
        self.assertNotEqual(
            bootstrap_module.sha256_bytes(
                bootstrap_module.canonical_json(list(first))
            ),
            bootstrap_module.sha256_bytes(
                bootstrap_module.canonical_json(list(second))
            ),
        )
        self.assertEqual(first[-2:], ("--timeout-seconds", "60"))

    def test_acceptance_then_same_mount_run_root_swap_fails_before_owner_freeze(
        self,
    ) -> None:
        original_publish_owner_binding = bootstrap_module.publish_owner_binding
        displaced = self.strict_parent / f"{RUN_ID}-reservation-created"
        protected_result_called = False

        def swap_before_owner(*args: object, **kwargs: object):
            os.rename(self.run_root, displaced)
            shutil.copytree(displaced, self.run_root)
            return original_publish_owner_binding(*args, **kwargs)

        def forbidden_protected_result(*args: object, **kwargs: object):
            nonlocal protected_result_called
            protected_result_called = True
            raise AssertionError("protected child launched after run-root substitution")

        with (
            patch.object(
                bootstrap_module,
                "publish_owner_binding",
                side_effect=swap_before_owner,
            ),
            patch.object(
                bootstrap_module,
                "_protected_result",
                side_effect=forbidden_protected_result,
            ),
            self.assertRaisesRegex(Exception, "descriptor identity differs"),
        ):
            bootstrap_g0_cp0(**self.bootstrap_kwargs())

        self.assertFalse(protected_result_called)
        self.assertFalse((self.run_root / "owner-binding.json").exists())
        self.assertFalse((displaced / "owner-binding.json").exists())
        self.assertEqual(
            {
                path: digest(path)
                for root in self.sources.values()
                for path in root.rglob("*")
                if path.is_file() and not path.is_symlink()
            },
            self.source_digests,
        )
        self.assertEqual(digest(self.paper), self.paper_digest)

    def test_protected_source_paths_are_not_lexically_normalized_through_symlinks(self) -> None:
        alias = self.base / "source-alias"
        alias.symlink_to(self.sources["experiments4"], target_is_directory=True)
        with self.assertRaises(Exception):
            bootstrap_g0_cp0(
                strict_parent=str(self.strict_parent),
                sealed_run_id=RUN_ID,
                run_root=str(self.run_root),
                external_dir=str(self.external),
                project_id="experiments7-g0-bootstrap-test",
                owner_seed="g0-bootstrap-owner",
                source_roots={
                    "experiments4": str(alias),
                    "experiments5": str(self.sources["experiments5"]),
                    "experiments6": str(self.sources["experiments6"]),
                },
                paper_path=str(self.paper),
                protected_executable=str(ROOT / "scripts" / "strict_run" / "g0_protected.py"),
                provider_path=str(ROOT / "scripts" / "g0" / "provider.py"),
                comparison_tool=str(ROOT / "scripts" / "g0" / "compare_manifests.py"),
                validator_path=str(ROOT / "src" / "provenance" / "strict_v6.py"),
                timeout_seconds=60,
            )
        self.assertFalse(self.run_root.exists())
        self.assertFalse((self.run_root / "checkpoints" / "cp0.json").exists())


if __name__ == "__main__":
    unittest.main()
