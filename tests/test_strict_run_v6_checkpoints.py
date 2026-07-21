from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strict_run import (  # noqa: E402
    ExternalTranscriptRecord,
    ExternalTranscriptRegistry,
    ReservationInputs,
    StagePublisher,
    StrictRunError,
    WriterIdentity,
    accept_reservation,
    canonical_json,
    default_writer_policy,
    publish_blocked,
    publish_checkpoint,
    publish_owner_binding,
    reserve_strict_run,
    sha256_bytes,
    validate_checkpoint_payload,
    validate_final_tree,
)
from strict_run.publication import Publication  # noqa: E402
from strict_run.checkpoint import (  # noqa: E402
    CP0_DENIED_METADATA_SYSCALLS_X86_64,
    CP0_LANDLOCK_READ_EXEC_RIGHTS,
    CP0_REQUIRED_MEMFD_SEALS,
    _SemanticReplayContext,
    _require_canonical_checkpoint_chain,
    _validate_cp0_child_evidence,
)
from strict_run.filesystem import stable_identity  # noqa: E402
from strict_run.semantics import CP0_ROLES  # noqa: E402


RUN_ID = "exp7-strict-v6-20260717T130000Z-" + "b" * 32


def make_writer(role: str, prefix: str) -> WriterIdentity:
    return WriterIdentity(role, f"{prefix}fixture", f"{role}-writer")


class SyntheticStrictRun:
    """Disposable honest run that can advance only through CP1 on this host."""

    def __init__(self, base: Path) -> None:
        self.base = base
        self.parent = base / "paper_outputs" / "strict-runs"
        self.parent.mkdir(parents=True)
        self.run_root = self.parent / RUN_ID
        self.external = base / "external-transcripts"
        self.external.mkdir(parents=True)
        self.policy = default_writer_policy()
        self.writers = {
            "reservation": make_writer("run_reservation", "reservation:"),
            "acceptance": make_writer("reservation_acceptance", "acceptance:"),
            "g0": make_writer("g0", "g0:"),
            "registry": make_writer("registry", "registry:"),
            "inventory": make_writer("inventory", "inventory:"),
            "controller": make_writer("checkpoint_controller", "checkpoint:"),
        }
        self.checkpoints: dict[int, object] = {}
        self.external_counter = 0
        self.child_records: list[ExternalTranscriptRecord] = []
        digest = sha256_bytes(b"strict-synthetic")
        self.reservation_inputs = ReservationInputs(
            project_id="experiments7-synthetic",
            owner_seed="strict-owner-seed",
            writer=self.writers["reservation"],
            tool_sha256=digest,
            configuration_sha256=digest,
            runtime_sha256=digest,
            environment_path_policy_sha256=digest,
            canonical_argv=("python", "-B", "strict-synthetic"),
        )

    def publisher(
        self, name: str, *, allow_reserved_paths: bool = False
    ) -> StagePublisher:
        return StagePublisher(
            str(self.parent),
            RUN_ID,
            str(self.run_root),
            self.writers[name],
            self.policy,
            allow_reserved_paths=allow_reserved_paths,
        )

    @staticmethod
    def ref(publication: Publication) -> dict[str, str]:
        evidence = publication.evidence
        return {
            "relative_path": evidence.relative_path,
            "artifact_evidence_sha256": sha256_bytes(
                canonical_json(evidence.to_dict())
            ),
        }

    def _document(self, value: object, stem: str) -> Path:
        self.external_counter += 1
        path = self.external / f"{stem}-{self.external_counter}.json"
        path.write_bytes(canonical_json(value))
        return path

    def _child_document(self, value: object, stem: str) -> Path:
        self.external_counter += 1
        parent = self.external / f"{stem}-{self.external_counter}"
        parent.mkdir()
        path = parent / "document.json"
        path.write_bytes(canonical_json(value))
        self.child_records.append(ExternalTranscriptRecord(str(path)))
        return path

    @staticmethod
    def _external_evidence(path: Path) -> dict[str, object]:
        def full_identity(st: os.stat_result) -> dict[str, object]:
            identity = stable_identity(st)
            identity.update(
                {
                    "nlink": st.st_nlink,
                    "mtime_ns": st.st_mtime_ns,
                    "ctime_ns": st.st_ctime_ns,
                }
            )
            return identity

        return {
            "path": str(path),
            "sha256": sha256_bytes(path.read_bytes()),
            "bytes": path.stat().st_size,
            "identity": full_identity(path.stat()),
            "parent_identity": full_identity(path.parent.stat()),
        }

    def transcript_registry(
        self,
        publications: list[Publication],
        predecessor: Publication | None = None,
    ) -> ExternalTranscriptRegistry:
        transcripts = [publication.transcript for publication in publications]
        if predecessor is not None:
            transcripts.append(predecessor.transcript)
        path = self._document({"transcripts": transcripts}, "registry")
        records = [
                ExternalTranscriptRecord(str(path), ("transcripts", index))
                for index in range(len(transcripts))
            ]
        records.extend(self.child_records)
        return ExternalTranscriptRegistry(
            records,
            strict_parent=str(self.parent),
            sealed_run_id=RUN_ID,
            run_root=str(self.run_root),
        )

    def accept(self, reservation_transcript: object):
        path = self._document(reservation_transcript, "reservation-transcript")
        payload = path.read_bytes()
        with ExternalTranscriptRegistry(
            [ExternalTranscriptRecord(str(path))],
            strict_parent=str(self.parent),
            sealed_run_id=RUN_ID,
            run_root=str(self.run_root),
        ) as registry:
            return accept_reservation(
                str(self.parent),
                RUN_ID,
                str(self.run_root),
                registry,
                self.writers["acceptance"],
                reservation_transcript_sha256=sha256_bytes(payload),
                policy=self.policy,
            )

    def bootstrap_pre_cp0(
        self,
        *,
        include_acceptance: bool = True,
        include_checkpoint_directory: bool = True,
        envelope_after_manifests: bool = False,
        envelope_binding_valid: bool = True,
        manifests_before_acceptance: bool = False,
        producer_attestation_mode: str = "valid",
        child_evidence_mode: str = "valid",
        child_evidence_operation: str = "source-pre",
    ) -> list[Publication]:
        self.child_records = []
        if child_evidence_operation not in {"source-pre", "paper-pre"}:
            raise RuntimeError("unknown child evidence operation")
        reservation = reserve_strict_run(
            str(self.parent),
            RUN_ID,
            str(self.run_root),
            self.reservation_inputs,
            policy=self.policy,
        )
        stage = [reservation.publication]
        if manifests_before_acceptance:
            stage.extend(
                self.publisher("g0").ensure_directory("manifests")
            )
        acceptance_publication: Publication | None = None
        if include_acceptance:
            acceptance = self.accept(reservation.transcript)
            acceptance_publication = acceptance.publication
            stage.append(acceptance_publication)
            stage.append(
                publish_owner_binding(
                    str(self.parent),
                    RUN_ID,
                    str(self.run_root),
                    "strict-owner-seed",
                    "experiments7-synthetic",
                    self.writers["g0"],
                    policy=self.policy,
                )
            )
        if include_checkpoint_directory:
            stage.extend(
                self.publisher(
                    "controller", allow_reserved_paths=True
                ).ensure_directory("checkpoints")
            )
        g0 = self.publisher("g0")
        roles: dict[str, list[dict[str, str]]] = {}
        role_publications: list[Publication] = []
        envelope_publication: Publication | None = None
        envelope_payload: dict[str, object] | None = None

        frozen_executable_path = "frozen/g0_executable/g0_protected.py"
        frozen_provider_path = "frozen/provider/provider.py"
        frozen_configuration_path = "frozen/configuration/g0-config.json"
        frozen_environment_path = "frozen/environment_allowlist/environment.json"
        executable_bytes = b"# synthetic frozen G0 child\n"
        provider_bytes = (ROOT / "scripts" / "g0" / "provider.py").read_bytes()
        paper_path = str(self.base / "protected" / "paper.pdf")
        configuration = {
            "schema": "experiments7-g0-protected-config/v6",
            "sealed_run_id": RUN_ID,
            "run_root": str(self.run_root),
            "provider_path": str(self.run_root / frozen_provider_path),
            "provider_sha256": sha256_bytes(provider_bytes),
            "g0_executable_sha256": sha256_bytes(executable_bytes),
            "source_roots": [
                {
                    "root_id": f"experiments{index}",
                    "path": str(self.base / f"protected-experiments{index}"),
                }
                for index in (4, 5, 6)
            ],
            "paper_path": paper_path,
        }
        environment = {
            "PATH": "/usr/bin:/bin",
            "HOME": "/nonexistent",
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }
        environment_payload = {
            "schema": "experiments7-g0-environment-allowlist/v6",
            "environment": environment,
            "environment_sha256": sha256_bytes(canonical_json(environment)),
        }
        runtime_path = os.path.realpath(sys.executable)
        runtime_sha256 = sha256_bytes(Path(runtime_path).read_bytes())
        runtime_identity = {
            "schema": "experiments7-g0-runtime-identity/v6",
            "python": {
                "path": runtime_path,
                "sha256": runtime_sha256,
                "version": sys.version,
                "implementation": "CPython",
            },
            "platform": {
                "system": "Linux",
                "release": "synthetic",
                "machine": "x86_64",
            },
        }
        provider_evidence = {
            "provider": "landlock_path_beneath+seccomp_metadata_deny",
            "landlock_abi": 4,
            "seccomp_mode": "classic_bpf_errno_eperm",
            "denied_metadata_syscalls_x86_64": list(
                CP0_DENIED_METADATA_SYSCALLS_X86_64
            ),
            "kernel": "synthetic",
            "machine": "x86_64",
            "landlock_rules": [
                {
                    "path": "/",
                    "dev": 1,
                    "inode": 2,
                    "rights": CP0_LANDLOCK_READ_EXEC_RIGHTS,
                }
            ],
            "default_filesystem_rights": "read_execute_only",
            "writable_directories": [],
            "metadata_mutation_policy": "globally_denied_by_seccomp",
        }
        paper_write_probe = {
            "operation": "open(O_WRONLY|O_NOFOLLOW)",
            "target_path": paper_path,
            "denied": True,
            "errno": 13,
        }
        if child_evidence_mode == "coordinated_provider_syscalls_forged":
            provider_evidence["denied_metadata_syscalls_x86_64"] = [90]
        elif child_evidence_mode == "coordinated_provider_rights_forged":
            provider_evidence["landlock_rules"][0]["rights"] = 5
        elif child_evidence_mode == "coordinated_provider_kernel_forged":
            provider_evidence["kernel"] = "forged-kernel"
        elif child_evidence_mode == "coordinated_provider_machine_forged":
            provider_evidence["machine"] = "forged-machine"
        elif child_evidence_mode == "coordinated_probe_target_forged":
            paper_write_probe["target_path"] = str(
                self.base / "protected" / "other.pdf"
            )
        elif child_evidence_mode == "coordinated_probe_errno_forged":
            paper_write_probe["errno"] = 30
        envelope_audit = {
            "schema": "experiments7-g0-protected-result/v6",
            "mode": "verify-envelope",
            "sealed_run_id": RUN_ID,
            "run_root": str(self.run_root),
            "python_dont_write_bytecode": True,
            "pythonhashseed": "0",
            "provider_evidence": provider_evidence,
            "paper_write_probe": paper_write_probe,
        }

        def argv_for(operation: str) -> list[str]:
            return [
                runtime_path,
                "-B",
                str(self.run_root / frozen_executable_path),
                "--config",
                str(self.run_root / frozen_configuration_path),
                "--mode",
                operation,
                "--producer-binding-stdin",
            ]

        source_argv = argv_for("source-pre")
        paper_argv = argv_for("paper-pre")
        if child_evidence_mode == "command_policy_forged":
            target_argv = (
                source_argv
                if child_evidence_operation == "source-pre"
                else paper_argv
            )
            target_argv[0] = "/forged/python"

        def publish_envelope() -> None:
            nonlocal envelope_publication, envelope_payload
            if acceptance_publication is None or not envelope_binding_valid:
                acceptance_transcript_sha256 = "0" * 64
                acceptance_context: dict[str, object] = {}
            else:
                acceptance_transcript_sha256 = (
                    acceptance_publication.evidence.publication_transcript_sha256
                )
                acceptance_context = copy.deepcopy(
                    acceptance_publication.transcript["context"]
                )
            envelope_payload = {
                "schema": "experiments7-cp0-envelope-evidence/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "acceptance_transcript_sha256": acceptance_transcript_sha256,
            }
            created = g0.publish_json(
                "frozen/envelope_evidence/artifact.json",
                envelope_payload,
                context=acceptance_context,
            )
            stage.extend(created)
            envelope_publication = created[-1]
            role_publications.append(envelope_publication)
            roles["envelope_evidence"] = [self.ref(envelope_publication)]

        if not envelope_after_manifests:
            publish_envelope()

        transport_publications: dict[str, Publication] = {}
        for role, created in (
            (
                "g0_executable",
                g0.publish_bytes(frozen_executable_path, executable_bytes),
            ),
            ("provider", g0.publish_bytes(frozen_provider_path, provider_bytes)),
            (
                "configuration",
                g0.publish_json(frozen_configuration_path, configuration),
            ),
        ):
            stage.extend(created)
            publication = created[-1]
            transport_publications[role] = publication
            role_publications.append(publication)
            roles[role] = [self.ref(publication)]

        def producer_attestation(operation: str) -> dict[str, object]:
            acceptance_emitted = 0
            acceptance_sha256 = "0" * 64
            envelope_emitted = 0
            envelope_evidence_sha256 = "0" * 64
            envelope_transcript_sha256 = "0" * 64
            envelope_context: dict[str, object] = {}
            if acceptance_publication is not None:
                acceptance_emitted = int(
                    acceptance_publication.transcript[
                        "emitted_monotonic_ns"
                    ]
                )
                acceptance_sha256 = (
                    acceptance_publication.evidence.publication_transcript_sha256
                )
                envelope_context = copy.deepcopy(
                    acceptance_publication.transcript["context"]
                )
            if envelope_publication is not None:
                envelope_emitted = int(
                    envelope_publication.transcript[
                        "emitted_monotonic_ns"
                    ]
                )
                envelope_evidence_sha256 = sha256_bytes(
                    canonical_json(envelope_publication.evidence.to_dict())
                )
                envelope_transcript_sha256 = (
                    envelope_publication.evidence.publication_transcript_sha256
                )
                envelope_context = copy.deepcopy(
                    envelope_publication.transcript["context"]
                )
            started = max(acceptance_emitted, envelope_emitted) + 2
            attestation: dict[str, object] = {
                "schema": "experiments7-cp0-protected-read-attestation/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "operation": operation,
                "started_monotonic_ns": started,
                "acceptance_transcript_sha256": acceptance_sha256,
                "envelope_artifact_evidence_sha256": envelope_evidence_sha256,
                "envelope_transcript_sha256": envelope_transcript_sha256,
                "envelope_context": envelope_context,
            }
            if producer_attestation_mode == "start_equal_envelope":
                attestation["started_monotonic_ns"] = (
                    envelope_publication.transcript["emitted_monotonic_ns"]
                )
            elif producer_attestation_mode == "hash_only":
                attestation["envelope_artifact_evidence_sha256"] = "0" * 64
            elif producer_attestation_mode == "context_only":
                attestation["envelope_context"] = {"forged": True}
            elif producer_attestation_mode not in {"valid", "publication_equal"}:
                raise RuntimeError("unknown producer attestation mode")
            return attestation

        def child_evidence(
            operation: str,
            attestation: dict[str, object],
            payload: object,
        ) -> tuple[dict[str, object], dict[str, object]]:
            child_payload = copy.deepcopy(payload)
            child_provider_evidence = copy.deepcopy(provider_evidence)
            if (
                child_evidence_mode == "provider_policy_forged"
                and operation == child_evidence_operation
            ):
                child_provider_evidence["writable_directories"] = [
                    str(self.run_root)
                ]
            child_paper_write_probe = copy.deepcopy(paper_write_probe)
            if (
                child_evidence_mode == "paper_probe_write_allowed"
                and operation == child_evidence_operation
            ):
                child_paper_write_probe["denied"] = False
                child_paper_write_probe["errno"] = 0
            child_environment = dict(environment)
            if (
                child_evidence_mode == "environment_policy_extra"
                and operation == child_evidence_operation
            ):
                child_environment["PYTHONPATH"] = "/forged"
            stdout_attestation = copy.deepcopy(attestation)
            if (
                child_evidence_mode == "stdout_attestation_mismatch"
                and operation == child_evidence_operation
            ):
                stdout_attestation["started_monotonic_ns"] = (
                    int(stdout_attestation["started_monotonic_ns"]) + 1
                )
            stdout: dict[str, object] = {
                "schema": "experiments7-g0-protected-result/v6",
                "mode": operation,
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "provider_evidence": child_provider_evidence,
                "paper_write_probe": child_paper_write_probe,
                "producer_attestation": stdout_attestation,
            }
            if operation == "source-pre":
                if child_evidence_mode == "coordinated_source_schema_forged":
                    child_payload[0]["unexpected"] = True
                elif child_evidence_mode == "coordinated_source_identity_forged":
                    child_payload[0]["descriptor_identity"]["file_type"] = (
                        "directory"
                    )
                stdout["records"] = child_payload
                argv = list(source_argv)
            else:
                if child_evidence_mode == "coordinated_paper_path_forged":
                    child_payload["paper_path"] = str(
                        self.base / "protected" / "other.pdf"
                    )
                elif child_evidence_mode == "coordinated_paper_provider_forged":
                    child_payload["provider"]["kernel"] = "forged-kernel"
                elif child_evidence_mode == "coordinated_paper_descriptor_forged":
                    child_payload["descriptor_identity"]["file_type"] = (
                        "directory"
                    )
                elif child_evidence_mode == "coordinated_paper_parent_forged":
                    child_payload["parent_identity"]["st_ino"] = False
                elif child_evidence_mode == "coordinated_paper_size_forged":
                    child_payload["size"] = (
                        int(child_payload["descriptor_identity"]["size"]) + 1
                    )
                stdout["paper"] = child_payload
                argv = list(paper_argv)
            stdout_path = self._child_document(stdout, f"{operation}-stdout")
            stdout_evidence = self._external_evidence(stdout_path)

            acceptance_emitted = (
                int(acceptance_publication.transcript["emitted_monotonic_ns"])
                if acceptance_publication is not None
                else 0
            )
            envelope_emitted = (
                int(envelope_publication.transcript["emitted_monotonic_ns"])
                if envelope_publication is not None
                else 0
            )
            launched = max(acceptance_emitted, envelope_emitted) + 1
            started = int(attestation["started_monotonic_ns"])
            completed = started + 1
            if child_evidence_mode == "execution_chronology_overlap" and operation == child_evidence_operation:
                launched = started
            if child_evidence_mode == "frozen_argv_divergence" and operation == child_evidence_operation:
                argv.insert(-1, "--forged")

            if (
                acceptance_publication is not None
                and envelope_publication is not None
                and envelope_payload is not None
            ):
                producer_binding = {
                    "schema": "experiments7-cp0-producer-binding/v6",
                    "acceptance_transcript": acceptance_publication.transcript,
                    "envelope_artifact_evidence": envelope_publication.evidence.to_dict(),
                    "envelope_transcript": envelope_publication.transcript,
                    "envelope_payload": envelope_payload,
                }
                producer_binding_sha256 = sha256_bytes(canonical_json(producer_binding))
            else:
                producer_binding_sha256 = "0" * 64
            if child_evidence_mode == "producer_binding_mismatch" and operation == child_evidence_operation:
                producer_binding_sha256 = "0" * 64

            reservation_payload = json.loads(
                (self.run_root / "reservation.json").read_text(encoding="utf-8")
            )
            reservation_child = reservation_payload["child"]
            runtime_fd = 10
            artifact_fds = {
                "g0_executable": 11,
                "configuration": 12,
                "provider": 13,
            }
            transport_artifacts = {}
            for role, fd in artifact_fds.items():
                evidence = transport_publications[role].evidence
                transport_artifacts[role] = {
                    "relative_path": evidence.relative_path,
                    "sha256": evidence.sha256,
                    "bytes": evidence.size,
                    "identity": evidence.identity,
                    "mount_id": reservation_child["mount"]["mount_id"],
                    "fd": fd,
                    "proc_path": f"/proc/self/fd/{fd}",
                    "seals": CP0_REQUIRED_MEMFD_SEALS,
                }
            executed_argv = list(argv)
            executed_argv[2] = transport_artifacts["g0_executable"]["proc_path"]
            executed_argv[4] = transport_artifacts["configuration"]["proc_path"]
            executed_argv[5:5] = [
                "--provider-fd",
                str(transport_artifacts["provider"]["fd"]),
            ]
            descriptor_transport = {
                "schema": "experiments7-g0-descriptor-transport/v6",
                "logical_argv": list(argv),
                "executed_argv": executed_argv,
                "executed_argv_sha256": sha256_bytes(canonical_json(executed_argv)),
                "runtime": {
                    "fd": runtime_fd,
                    "proc_path": f"/proc/self/fd/{runtime_fd}",
                    "sha256": runtime_sha256,
                    "bytes": Path(runtime_path).stat().st_size,
                    "seals": CP0_REQUIRED_MEMFD_SEALS,
                },
                "artifacts": transport_artifacts,
                "run_root": {
                    "identity": reservation_child["descriptor"]["identity"],
                    "mount_id": reservation_child["mount"]["mount_id"],
                },
            }

            if (
                child_evidence_mode == "runtime_transport_size_mismatch"
                and operation == child_evidence_operation
            ):
                descriptor_transport["runtime"]["bytes"] = (
                    int(descriptor_transport["runtime"]["bytes"]) + 1
                )
            if child_evidence_mode == "descriptor_transport_mismatch" and operation == child_evidence_operation:
                transport_artifacts["provider"]["fd"] = runtime_fd
            execution = {
                "schema": "experiments7-g0-child-execution/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "mode": operation,
                "argv": argv,
                "argv_sha256": sha256_bytes(canonical_json(argv)),
                "descriptor_transport": descriptor_transport,
                "runtime_executable_sha256": (
                    "0" * 64
                    if child_evidence_mode == "runtime_digest_mismatch"
                    and operation == child_evidence_operation
                    else runtime_sha256
                ),
                "executable_sha256": sha256_bytes(executable_bytes),
                "configuration_sha256": sha256_bytes(canonical_json(configuration)),
                "provider_sha256": sha256_bytes(provider_bytes),
                "producer_binding_sha256": producer_binding_sha256,
                "environment": child_environment,
                "environment_sha256": sha256_bytes(
                    canonical_json(child_environment)
                ),
                "launched_monotonic_ns": launched,
                "completed_monotonic_ns": completed,
                "exit_status": 0,
                "stderr_sha256": sha256_bytes(b""),
                "stderr_bytes": 0,
                "stdout_evidence": stdout_evidence,
            }
            execution_path = self._child_document(
                execution, f"{operation}-execution"
            )
            context = copy.deepcopy(attestation)
            context["child_stdout_sha256"] = stdout_evidence["sha256"]
            context["child_execution_sha256"] = sha256_bytes(
                execution_path.read_bytes()
            )
            context["child_execution_evidence"] = self._external_evidence(
                execution_path
            )
            if (
                child_evidence_mode == "execution_evidence_mismatch"
                and operation == child_evidence_operation
            ):
                context["child_execution_evidence"] = dict(
                    context["child_execution_evidence"]
                )
                context["child_execution_evidence"]["path"] = "/forged/execution.json"
            if child_evidence_mode == "stdout_hash_mismatch" and operation == child_evidence_operation:
                context["child_stdout_sha256"] = "0" * 64
            elif child_evidence_mode == "execution_unregistered" and operation == child_evidence_operation:
                context["child_execution_sha256"] = "0" * 64
            elif child_evidence_mode not in {
                "valid",
                "stdout_attestation_mismatch",
                "execution_chronology_overlap",
                "frozen_argv_divergence",
                "producer_binding_mismatch",
                "stdout_hash_mismatch",
                "execution_unregistered",
                "execution_evidence_mismatch",
                "environment_policy_extra",
                "command_policy_forged",
                "provider_policy_forged",
                "paper_probe_write_allowed",
                "runtime_digest_mismatch",
                "runtime_transport_size_mismatch",
                "coordinated_provider_syscalls_forged",
                "descriptor_transport_mismatch",
                "coordinated_provider_rights_forged",
                "coordinated_provider_kernel_forged",
                "coordinated_provider_machine_forged",
                "coordinated_probe_target_forged",
                "coordinated_probe_errno_forged",
                "coordinated_source_schema_forged",
                "coordinated_source_identity_forged",
                "coordinated_paper_path_forged",
                "coordinated_paper_provider_forged",
                "coordinated_paper_descriptor_forged",
                "coordinated_paper_parent_forged",
                "coordinated_paper_size_forged",
            }:
                raise RuntimeError("unknown child evidence mode")
            return stdout, context

        source_record = {
            "schema": "experiments7-source-pre/v6",
            "record_id": "source:" + "1" * 64,
            "root_id": "experiments4",
            "relative_path": "source.py",
            "type": "regular",
            "sha256": "1" * 64,
            "size": 1,
            "descriptor_identity": {
                "file_type": "regular",
                "st_dev": 1,
                "st_ino": 4,
                "mode": 0o444,
                "size": 1,
                "mtime_ns": 1,
            },
        }
        source_stdout, source_context = child_evidence(
            "source-pre", producer_attestation("source-pre"), [source_record]
        )
        stage.extend(
            g0.publish_bytes(
                "manifests/source-pre.jsonl",
                b"".join(canonical_json(record) for record in source_stdout["records"]),
                context=source_context,
            )
        )
        source_pre = stage[-1]
        paper_record = {
            "schema": "experiments7-paper-pre/v6",
            "sealed_run_id": RUN_ID,
            "run_root": str(self.run_root),
            "paper_path": paper_path,
            "sha256": "2" * 64,
            "size": 1,
            "descriptor_identity": {
                "file_type": "regular",
                "st_dev": 1,
                "st_ino": 3,
                "mode": 0o444,
                "size": 1,
                "mtime_ns": 1,
            },
            "parent_identity": {
                "file_type": "directory",
                "st_dev": 1,
                "st_ino": 2,
                "mode": 0o555,
                "size": 1,
                "mtime_ns": 1,
            },
            "provider": {
                key: copy.deepcopy(provider_evidence[key])
                for key in (
                    "provider",
                    "landlock_abi",
                    "seccomp_mode",
                    "denied_metadata_syscalls_x86_64",
                    "kernel",
                    "machine",
                )
            },
        }
        paper_stdout, paper_context = child_evidence(
            "paper-pre", producer_attestation("paper-pre"), paper_record
        )
        stage.extend(
            g0.publish_json(
                "manifests/paper-pre.json",
                paper_stdout["paper"],
                context=paper_context,
            )
        )
        paper_pre = stage[-1]
        if producer_attestation_mode == "publication_equal":
            def overlap(publication: Publication) -> Publication:
                transcript = copy.deepcopy(publication.transcript)
                transcript["context"]["started_monotonic_ns"] = transcript[
                    "events"
                ][0]["completed_monotonic_ns"]
                return Publication(
                    replace(
                        publication.evidence,
                        publication_transcript_sha256=sha256_bytes(
                            canonical_json(transcript)
                        ),
                    ),
                    transcript,
                )

            source_index = stage.index(source_pre)
            source_pre = overlap(source_pre)
            stage[source_index] = source_pre
            paper_index = stage.index(paper_pre)
            paper_pre = overlap(paper_pre)
            stage[paper_index] = paper_pre
        if envelope_after_manifests:
            publish_envelope()
        for role in CP0_ROLES:
            if role == "envelope_evidence" or role in transport_publications:
                continue
            if role == "writer_policy":
                created = g0.publish_json(
                    "frozen/writer-policy.json", self.policy.to_dict()
                )
            elif role == "g0_executable":
                created = g0.publish_bytes(frozen_executable_path, executable_bytes)
            elif role == "provider":
                created = g0.publish_bytes(frozen_provider_path, provider_bytes)
            elif role == "configuration":
                created = g0.publish_json(frozen_configuration_path, configuration)
            elif role == "pre_argv":
                created = g0.publish_json(
                    "frozen/pre_argv/argv.json",
                    {
                        "schema": "experiments7-g0-argv/v6",
                        "mode": "source-pre",
                        "argv": source_argv,
                    },
                )
            elif role == "post_argv":
                created = g0.publish_json(
                    "frozen/post_argv/argv.json",
                    {
                        "schema": "experiments7-g0-argv/v6",
                        "mode": "paper-pre",
                        "argv": paper_argv,
                    },
                )
            elif role == "environment_allowlist":
                created = g0.publish_json(
                    frozen_environment_path, environment_payload
                )
            elif role == "runtime_identity":
                created = []
                runtime_regulars: list[Publication] = []
                for path, payload in (
                    ("frozen/runtime_identity/runtime.json", runtime_identity),
                    (
                        "frozen/runtime_identity/envelope-audit.json",
                        envelope_audit,
                    ),
                ):
                    published = g0.publish_json(path, payload)
                    created.extend(published)
                    runtime_regulars.append(published[-1])
                stage.extend(created)
                role_publications.extend(runtime_regulars)
                roles[role] = sorted(
                    [self.ref(publication) for publication in runtime_regulars],
                    key=lambda item: item["relative_path"],
                )
                continue
            else:
                created = g0.publish_bytes(
                    f"frozen/{role}/artifact.bin", f"{role}\n".encode("utf-8")
                )
            stage.extend(created)
            role_publications.append(created[-1])
            roles[role] = [self.ref(created[-1])]
        canonical_refs = sorted(
            [self.ref(publication) for publication in role_publications],
            key=lambda item: item["relative_path"],
        )
        stage.extend(
            g0.publish_json(
                "frozen/inventory.json",
                {
                    "schema": "experiments7-cp0-frozen-inventory/v6",
                    "sealed_run_id": RUN_ID,
                    "run_root": str(self.run_root),
                    "roles": roles,
                    "frozen_artifact_count": len(role_publications),
                    "frozen_artifacts_sha256": sha256_bytes(
                        canonical_json(canonical_refs)
                    ),
                    "source_pre": self.ref(source_pre),
                    "paper_pre": self.ref(paper_pre),
                },
            )
        )
        return stage

    def cp1_stage(self) -> list[Publication]:
        registry = self.publisher("registry")
        inventory = self.publisher("inventory")
        stage: list[Publication] = []
        created = registry.publish_json(
            "registry/lineage-basis.json",
            {
                "schema": "experiments7-cp1-lineage-basis/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "lineage_ids": [f"lineage-{index:03d}" for index in range(129)],
            },
        )
        stage.extend(created)
        lineage = created[-1]
        created = registry.publish_json(
            "registry/profile-basis.json",
            {
                "schema": "experiments7-cp1-profile-basis/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "profile_ids": [f"profile-{index:02d}" for index in range(30)],
            },
        )
        stage.extend(created)
        profiles = created[-1]
        stage.extend(
            registry.publish_json(
                "registry/cp2-basis.json",
                {
                    "schema": "experiments7-cp1-cp2-basis/v6",
                    "sealed_run_id": RUN_ID,
                    "run_root": str(self.run_root),
                    "lineage_source": self.ref(lineage),
                    "profile_source": self.ref(profiles),
                },
            )
        )
        structures = [
            {"paper_section_id": f"section-{index:02d}", "ordinal": index}
            for index in range(51)
        ]
        records = [
            {
                "result_id": f"result:checkpoint:{index:04d}",
                "paper_section_id": f"section-{index % 51:02d}",
                "numeric_value": index,
                "display_value": str(index),
            }
            for index in range(1908)
        ]
        aliases = [
            {
                "alias_id": f"alias:checkpoint:{index:03d}",
                "target_result_id": f"result:checkpoint:{index:04d}",
            }
            for index in range(86)
        ]
        for path, rows in (
            ("inventory/structure.jsonl", structures),
            ("inventory/results.jsonl", records),
            ("inventory/aliases.jsonl", aliases),
        ):
            stage.extend(
                inventory.publish_bytes(
                    path,
                    b"".join(canonical_json(record) for record in rows),
                )
            )
        return stage

    def seal(self, checkpoint: int, stage: list[Publication]):
        predecessor = None if checkpoint == 0 else self.checkpoints[checkpoint - 1].publication
        with self.transcript_registry(stage, predecessor) as registry:
            result = publish_checkpoint(
                str(self.parent),
                RUN_ID,
                str(self.run_root),
                checkpoint,
                stage,
                self.writers["controller"],
                self.policy,
                transcript_registry=registry,
                previous_checkpoint_publication=predecessor,
            )
        self.checkpoints[checkpoint] = result
        return result

    def cp0(self):
        return self.seal(0, self.bootstrap_pre_cp0())

    def cp1(self):
        if 0 not in self.checkpoints:
            self.cp0()
        return self.seal(1, self.cp1_stage())


class StrictRunV6CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="exp7-strict-cp-v6-")
        self.fixture = SyntheticStrictRun(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_cp0_cp1_then_cp2_runtime_block_has_no_cp2_to_cp6(self) -> None:
        cp0 = self.fixture.cp0()
        cp1 = self.fixture.cp1()
        self.assertEqual(cp0.payload["semantic_bindings"]["checkpoint"], 0)
        self.assertEqual(cp1.payload["semantic_bindings"]["inventory_record_count"], 1908)
        with self.fixture.transcript_registry([], cp1.publication) as registry:
            with self.assertRaises(StrictRunError) as caught:
                publish_checkpoint(
                    str(self.fixture.parent),
                    RUN_ID,
                    str(self.fixture.run_root),
                    2,
                    [],
                    self.fixture.writers["controller"],
                    self.fixture.policy,
                    transcript_registry=registry,
                    previous_checkpoint_publication=cp1.publication,
                )
        self.assertEqual(caught.exception.code, "RUNTIME_PROOF_ENVIRONMENT_UNAVAILABLE")
        publish_blocked(
            str(self.fixture.parent),
            RUN_ID,
            str(self.fixture.run_root),
            [caught.exception.code],
            self.fixture.writers["controller"],
            self.fixture.policy,
        )
        self.assertTrue((self.fixture.run_root / "terminal" / "blocked.json").is_file())
        for index in range(2, 7):
            self.assertFalse((self.fixture.run_root / "checkpoints" / f"cp{index}.json").exists())

    def test_fresh_cp0_uses_public_controller_checkpoint_directory_bootstrap(self) -> None:
        from strict_run import prepare_checkpoint_directory

        stage = self.fixture.bootstrap_pre_cp0(include_checkpoint_directory=False)
        self.assertFalse((self.fixture.run_root / "checkpoints").exists())
        checkpoint_directory = prepare_checkpoint_directory(
            str(self.fixture.parent),
            RUN_ID,
            str(self.fixture.run_root),
            self.fixture.writers["controller"],
            self.fixture.policy,
        )
        self.assertEqual(checkpoint_directory.evidence.relative_path, "checkpoints")
        self.assertEqual(checkpoint_directory.evidence.artifact_type, "directory")
        with self.assertRaises(StrictRunError) as caught:
            prepare_checkpoint_directory(
                str(self.fixture.parent),
                RUN_ID,
                str(self.fixture.run_root),
                self.fixture.writers["controller"],
                self.fixture.policy,
            )
        self.assertEqual(caught.exception.code, "CHECKPOINT_DIRECTORY_EXISTS")
        stage.append(checkpoint_directory)
        with self.fixture.transcript_registry(stage) as registry:
            result = publish_checkpoint(
                str(self.fixture.parent),
                RUN_ID,
                str(self.fixture.run_root),
                0,
                stage,
                self.fixture.writers["controller"],
                self.fixture.policy,
                transcript_registry=registry,
            )
        self.assertEqual(result.publication.evidence.relative_path, "checkpoints/cp0.json")
        self.assertEqual(
            [item.evidence.relative_path for item in result.stage_publications].count(
                "checkpoints"
            ),
            1,
        )

    def test_cp0_requires_prepublished_controller_checkpoint_directory(self) -> None:
        stage = self.fixture.bootstrap_pre_cp0(include_checkpoint_directory=False)
        self.assertFalse((self.fixture.run_root / "checkpoints").exists())
        with self.fixture.transcript_registry(stage) as registry:
            with self.assertRaises(StrictRunError) as caught:
                publish_checkpoint(
                    str(self.fixture.parent),
                    RUN_ID,
                    str(self.fixture.run_root),
                    0,
                    stage,
                    self.fixture.writers["controller"],
                    self.fixture.policy,
                    transcript_registry=registry,
                )
        self.assertEqual(caught.exception.code, "CP0_ARTIFACT_SET_INVALID")
        self.assertFalse((self.fixture.run_root / "checkpoints").exists())

    def test_cp0_rejects_missing_reservation_acceptance(self) -> None:
        stage = self.fixture.bootstrap_pre_cp0(include_acceptance=False)
        with self.fixture.transcript_registry(stage) as registry:
            with self.assertRaises(StrictRunError) as caught:
                publish_checkpoint(
                    str(self.fixture.parent),
                    RUN_ID,
                    str(self.fixture.run_root),
                    0,
                    stage,
                    self.fixture.writers["controller"],
                    self.fixture.policy,
                    transcript_registry=registry,
                )
        self.assertEqual(caught.exception.code, "CP0_BINDING_MISSING")
        self.assertFalse((self.fixture.run_root / "checkpoints" / "cp0.json").exists())

    def _assert_initial_cp0_rejected(
        self,
        stage: list[Publication],
        expected_code: str,
        fixture: SyntheticStrictRun | None = None,
    ) -> None:
        fixture = fixture or self.fixture
        with fixture.transcript_registry(stage) as registry:
            with self.assertRaises(StrictRunError) as caught:
                publish_checkpoint(
                    str(fixture.parent),
                    RUN_ID,
                    str(fixture.run_root),
                    0,
                    stage,
                    fixture.writers["controller"],
                    fixture.policy,
                    transcript_registry=registry,
                )
        self.assertEqual(caught.exception.code, expected_code)
        self.assertFalse(
            (fixture.run_root / "checkpoints" / "cp0.json").exists()
        )

    def _assert_direct_child_evidence_rejected(
        self,
        fixture: SyntheticStrictRun,
        stage: list[Publication],
        operation: str,
    ) -> None:
        by_path = {
            publication.evidence.relative_path: publication
            for publication in stage
        }
        publication = by_path[
            "manifests/source-pre.jsonl"
            if operation == "source-pre"
            else "manifests/paper-pre.json"
        ]
        root_fd = os.open(
            fixture.run_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            with fixture.transcript_registry(stage) as registry:
                with self.assertRaises(StrictRunError) as caught:
                    _validate_cp0_child_evidence(
                        root_fd,
                        by_path["reservation-acceptance.json"],
                        by_path["frozen/envelope_evidence/artifact.json"],
                        publication,
                        operation,
                        RUN_ID,
                        str(fixture.run_root),
                        by_path,
                        registry,
                        publication.transcript["context"],
                    )
        finally:
            os.close(root_fd)
        self.assertEqual(caught.exception.code, "CP0_CHILD_EVIDENCE_INVALID")

    def test_cp0_rejects_coordinated_audit_stdout_forgeries_per_branch(
        self,
    ) -> None:
        modes = {
            "coordinated_provider_syscalls_forged": "provider_evidence",
            "coordinated_provider_rights_forged": "provider_evidence",
            "coordinated_provider_kernel_forged": "provider_evidence",
            "coordinated_provider_machine_forged": "provider_evidence",
            "coordinated_probe_target_forged": "paper_write_probe",
            "coordinated_probe_errno_forged": "paper_write_probe",
        }
        for operation in ("source-pre", "paper-pre"):
            for mode, evidence_field in modes.items():
                with self.subTest(
                    operation=operation, mode=mode
                ), tempfile.TemporaryDirectory(
                    prefix=f"exp7-coordinated-{operation}-{mode}-",
                    dir=self.temporary.name,
                ) as base:
                    fixture = SyntheticStrictRun(Path(base))
                    stage = fixture.bootstrap_pre_cp0(
                        child_evidence_mode=mode,
                        child_evidence_operation=operation,
                    )
                    audit = json.loads(
                        (
                            fixture.run_root
                            / "frozen/runtime_identity/envelope-audit.json"
                        ).read_bytes()
                    )
                    stdout_record = fixture.child_records[
                        0 if operation == "source-pre" else 2
                    ]
                    stdout = json.loads(Path(stdout_record.path).read_bytes())
                    self.assertEqual(
                        audit[evidence_field], stdout[evidence_field]
                    )
                    self._assert_direct_child_evidence_rejected(
                        fixture, stage, operation
                    )

    def test_cp0_rejects_coordinated_paper_payload_forgeries(self) -> None:
        for mode in (
            "coordinated_paper_path_forged",
            "coordinated_paper_provider_forged",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                prefix=f"exp7-paper-payload-{mode}-",
                dir=self.temporary.name,
            ) as base:
                fixture = SyntheticStrictRun(Path(base))
                stage = fixture.bootstrap_pre_cp0(child_evidence_mode=mode)
                with patch(
                    "strict_run.checkpoint._validate_cp0_child_evidence",
                    wraps=_validate_cp0_child_evidence,
                ) as validator:
                    self._assert_initial_cp0_rejected(
                        stage,
                        "CP0_CHILD_EVIDENCE_INVALID",
                        fixture=fixture,
                    )
                self.assertEqual(
                    [call.args[4] for call in validator.call_args_list],
                    ["source-pre", "paper-pre"],
                )

    def test_cp0_rejects_fully_resealed_false_source_records(self) -> None:
        for mode in (
            "coordinated_source_schema_forged",
            "coordinated_source_identity_forged",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                prefix=f"exp7-source-record-{mode}-",
                dir=self.temporary.name,
            ) as base:
                fixture = SyntheticStrictRun(Path(base))
                stage = fixture.bootstrap_pre_cp0(child_evidence_mode=mode)
                with patch(
                    "strict_run.checkpoint._validate_cp0_child_evidence",
                    wraps=_validate_cp0_child_evidence,
                ) as validator:
                    self._assert_initial_cp0_rejected(
                        stage,
                        "CP0_CHILD_EVIDENCE_INVALID",
                        fixture=fixture,
                    )
                self.assertEqual(
                    [call.args[4] for call in validator.call_args_list],
                    ["source-pre"],
                )

    def test_cp0_rejects_forged_paper_identities(self) -> None:
        for mode in (
            "coordinated_paper_descriptor_forged",
            "coordinated_paper_parent_forged",
            "coordinated_paper_size_forged",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                prefix=f"exp7-paper-identity-{mode}-",
                dir=self.temporary.name,
            ) as base:
                fixture = SyntheticStrictRun(Path(base))
                stage = fixture.bootstrap_pre_cp0(child_evidence_mode=mode)
                with patch(
                    "strict_run.checkpoint._validate_cp0_child_evidence",
                    wraps=_validate_cp0_child_evidence,
                ) as validator:
                    self._assert_initial_cp0_rejected(
                        stage,
                        "CP0_CHILD_EVIDENCE_INVALID",
                        fixture=fixture,
                    )
                self.assertEqual(
                    [call.args[4] for call in validator.call_args_list],
                    ["source-pre", "paper-pre"],
                )

    def test_cp0_rejects_manifests_directory_before_acceptance(self) -> None:
        self._assert_initial_cp0_rejected(
            self.fixture.bootstrap_pre_cp0(
                manifests_before_acceptance=True
            ),
            "CP0_CAUSAL_ORDER_INVALID",
        )

    def test_cp0_rejects_producer_start_equal_to_envelope_gate(self) -> None:
        self._assert_initial_cp0_rejected(
            self.fixture.bootstrap_pre_cp0(
                producer_attestation_mode="start_equal_envelope"
            ),
            "CP0_CAUSAL_BINDING_INVALID",
        )

    def test_cp0_rejects_producer_start_overlapping_publication(self) -> None:
        self._assert_initial_cp0_rejected(
            self.fixture.bootstrap_pre_cp0(
                producer_attestation_mode="publication_equal"
            ),
            "CP0_CAUSAL_BINDING_INVALID",
        )

    def test_cp0_rejects_producer_hash_only_substitution(self) -> None:
        self._assert_initial_cp0_rejected(
            self.fixture.bootstrap_pre_cp0(
                producer_attestation_mode="hash_only"
            ),
            "CP0_CAUSAL_BINDING_INVALID",
        )

    def test_cp0_rejects_producer_context_only_substitution(self) -> None:
        self._assert_initial_cp0_rejected(
            self.fixture.bootstrap_pre_cp0(
                producer_attestation_mode="context_only"
            ),
            "CP0_CAUSAL_BINDING_INVALID",
        )

    def test_cp0_rejects_all_unbound_child_execution_variants(self) -> None:
        cases = {
            "stdout_hash_mismatch": "TRANSCRIPT_NOT_REGISTERED",
            "execution_unregistered": "TRANSCRIPT_NOT_REGISTERED",
            "stdout_attestation_mismatch": "CP0_CHILD_EVIDENCE_INVALID",
            "execution_chronology_overlap": "CP0_CHILD_EVIDENCE_INVALID",
            "frozen_argv_divergence": "CP0_CHILD_EVIDENCE_INVALID",
            "producer_binding_mismatch": "CP0_CHILD_EVIDENCE_INVALID",
            "execution_evidence_mismatch": "CP0_CHILD_EVIDENCE_INVALID",
            "environment_policy_extra": "CP0_CHILD_EVIDENCE_INVALID",
            "command_policy_forged": "CP0_CHILD_EVIDENCE_INVALID",
            "provider_policy_forged": "CP0_CHILD_EVIDENCE_INVALID",
            "paper_probe_write_allowed": "CP0_CHILD_EVIDENCE_INVALID",
            "runtime_digest_mismatch": "CP0_CHILD_EVIDENCE_INVALID",
            "runtime_transport_size_mismatch": "CP0_CHILD_EVIDENCE_INVALID",
            "descriptor_transport_mismatch": "CP0_CHILD_EVIDENCE_INVALID",
        }
        for operation in ("source-pre", "paper-pre"):
            for mode, expected_code in cases.items():
                with self.subTest(
                    operation=operation, mode=mode
                ), tempfile.TemporaryDirectory(
                    prefix=f"exp7-{operation}-{mode}-", dir=self.temporary.name
                ) as base:
                    fixture = SyntheticStrictRun(Path(base))
                    stage = fixture.bootstrap_pre_cp0(
                        child_evidence_mode=mode,
                        child_evidence_operation=operation,
                    )
                    with patch(
                        "strict_run.checkpoint._validate_cp0_child_evidence",
                        wraps=_validate_cp0_child_evidence,
                    ) as validator:
                        self._assert_initial_cp0_rejected(
                            stage, expected_code, fixture=fixture
                        )
                    self.assertEqual(
                        [call.args[4] for call in validator.call_args_list],
                        ["source-pre"]
                        if operation == "source-pre"
                        else ["source-pre", "paper-pre"],
                    )

    def _late_acceptance_stage(self) -> list[Publication]:
        stage = self.fixture.bootstrap_pre_cp0()
        acceptance_index = next(
            index
            for index, publication in enumerate(stage)
            if publication.evidence.relative_path == "reservation-acceptance.json"
        )
        acceptance = stage[acceptance_index]
        transcript = copy.deepcopy(acceptance.transcript)
        transcript["emitted_monotonic_ns"] = max(
            publication.transcript["events"][0]["completed_monotonic_ns"]
            for publication in stage
            if publication.evidence.relative_path
            not in {"reservation.json", "reservation-acceptance.json"}
        )
        stage[acceptance_index] = Publication(
            replace(
                acceptance.evidence,
                publication_transcript_sha256=sha256_bytes(
                    canonical_json(transcript)
                ),
            ),
            transcript,
        )
        return stage

    def test_cp0_rejects_fully_resealed_late_acceptance_without_mutation(self) -> None:
        stage = self._late_acceptance_stage()
        by_path = {
            publication.evidence.relative_path: publication
            for publication in stage
        }
        acceptance_emitted = by_path["reservation-acceptance.json"].transcript[
            "emitted_monotonic_ns"
        ]
        hostile_paths = {
            "owner-binding.json",
            "checkpoints",
            "manifests/source-pre.jsonl",
            "manifests/paper-pre.json",
            "frozen/envelope_evidence/artifact.json",
            "frozen/g0_executable/g0_protected.py",
            "frozen/inventory.json",
        }
        first_completed = {
            path: by_path[path].transcript["events"][0]["completed_monotonic_ns"]
            for path in hostile_paths
        }
        for path, completed in first_completed.items():
            with self.subTest(path=path):
                self.assertGreaterEqual(acceptance_emitted, completed)
        self.assertIn(acceptance_emitted, first_completed.values())
        with self.fixture.transcript_registry(stage) as registry:
            with self.assertRaises(StrictRunError) as caught:
                publish_checkpoint(
                    str(self.fixture.parent),
                    RUN_ID,
                    str(self.fixture.run_root),
                    0,
                    stage,
                    self.fixture.writers["controller"],
                    self.fixture.policy,
                    transcript_registry=registry,
                )
        self.assertEqual(caught.exception.code, "CP0_CAUSAL_ORDER_INVALID")
        self.assertFalse(
            (self.fixture.run_root / "checkpoints" / "cp0.json").exists()
        )

    def test_cp0_rejects_reservation_acceptance_equal_boundary(self) -> None:
        stage = self.fixture.bootstrap_pre_cp0()
        by_path = {
            publication.evidence.relative_path: publication
            for publication in stage
        }
        acceptance = by_path["reservation-acceptance.json"]
        transcript = copy.deepcopy(acceptance.transcript)
        transcript["events"][0]["completed_monotonic_ns"] = by_path[
            "reservation.json"
        ].transcript["emitted_monotonic_ns"]
        acceptance_index = stage.index(acceptance)
        stage[acceptance_index] = Publication(
            replace(
                acceptance.evidence,
                publication_transcript_sha256=sha256_bytes(
                    canonical_json(transcript)
                ),
            ),
            transcript,
        )
        with self.fixture.transcript_registry(stage) as registry:
            with self.assertRaises(StrictRunError) as caught:
                publish_checkpoint(
                    str(self.fixture.parent),
                    RUN_ID,
                    str(self.fixture.run_root),
                    0,
                    stage,
                    self.fixture.writers["controller"],
                    self.fixture.policy,
                    transcript_registry=registry,
                )
        self.assertEqual(caught.exception.code, "CP0_CAUSAL_ORDER_INVALID")
        self.assertFalse(
            (self.fixture.run_root / "checkpoints" / "cp0.json").exists()
        )

    def _assert_canonical_cp0_replay_rejected(
        self,
        stage: list[Publication],
        expected_code: str,
        fixture: SyntheticStrictRun | None = None,
    ) -> None:
        fixture = fixture or self.fixture
        with patch(
            "strict_run.checkpoint._validate_cp0_causal_order",
            return_value=None,
            create=True,
        ):
            cp0 = fixture.seal(0, stage)
        cp1 = fixture.cp1()
        context = _SemanticReplayContext(
            checkpoint_publications={0: cp0.publication, 1: cp1.publication},
            stage_publications={
                0: cp0.stage_publications,
                1: cp1.stage_publications,
            },
        )
        registered = (
            list(cp0.stage_publications)
            + list(cp1.stage_publications)
            + [cp0.publication, cp1.publication]
        )
        root_fd = os.open(
            fixture.run_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            with fixture.transcript_registry(registered) as registry:
                with self.assertRaises(StrictRunError) as caught:
                    _require_canonical_checkpoint_chain(
                        root_fd,
                        RUN_ID,
                        str(fixture.run_root),
                        1,
                        registry,
                        context,
                    )
        finally:
            os.close(root_fd)
        self.assertEqual(caught.exception.code, expected_code)

    def test_canonical_chain_replay_rejects_fully_resealed_late_acceptance(self) -> None:
        self._assert_canonical_cp0_replay_rejected(
            self._late_acceptance_stage(),
            "CP0_CAUSAL_ORDER_INVALID",
        )

    def test_canonical_chain_replay_rejects_invalid_envelope_binding(self) -> None:
        self._assert_canonical_cp0_replay_rejected(
            self.fixture.bootstrap_pre_cp0(envelope_binding_valid=False),
            "CP0_CAUSAL_BINDING_INVALID",
        )

    def test_canonical_chain_replay_rejects_manifests_before_acceptance(
        self,
    ) -> None:
        self._assert_canonical_cp0_replay_rejected(
            self.fixture.bootstrap_pre_cp0(
                manifests_before_acceptance=True
            ),
            "CP0_CAUSAL_ORDER_INVALID",
        )

    def test_canonical_chain_replay_rejects_producer_start_at_gate(
        self,
    ) -> None:
        self._assert_canonical_cp0_replay_rejected(
            self.fixture.bootstrap_pre_cp0(
                producer_attestation_mode="start_equal_envelope"
            ),
            "CP0_CAUSAL_BINDING_INVALID",
        )

    def test_canonical_chain_replay_rejects_producer_publication_overlap(
        self,
    ) -> None:
        self._assert_canonical_cp0_replay_rejected(
            self.fixture.bootstrap_pre_cp0(
                producer_attestation_mode="publication_equal"
            ),
            "CP0_CAUSAL_BINDING_INVALID",
        )

    def test_canonical_chain_replay_rejects_producer_hash_only_substitution(
        self,
    ) -> None:
        self._assert_canonical_cp0_replay_rejected(
            self.fixture.bootstrap_pre_cp0(
                producer_attestation_mode="hash_only"
            ),
            "CP0_CAUSAL_BINDING_INVALID",
        )

    def test_canonical_chain_replay_rejects_producer_context_only_substitution(
        self,
    ) -> None:
        self._assert_canonical_cp0_replay_rejected(
            self.fixture.bootstrap_pre_cp0(
                producer_attestation_mode="context_only"
            ),
            "CP0_CAUSAL_BINDING_INVALID",
        )

    def test_canonical_replay_rejects_all_unbound_child_execution_variants(
        self,
    ) -> None:
        cases = {
            "stdout_hash_mismatch": "TRANSCRIPT_NOT_REGISTERED",
            "execution_unregistered": "TRANSCRIPT_NOT_REGISTERED",
            "stdout_attestation_mismatch": "CP0_CHILD_EVIDENCE_INVALID",
            "execution_chronology_overlap": "CP0_CHILD_EVIDENCE_INVALID",
            "frozen_argv_divergence": "CP0_CHILD_EVIDENCE_INVALID",
            "producer_binding_mismatch": "CP0_CHILD_EVIDENCE_INVALID",
            "execution_evidence_mismatch": "CP0_CHILD_EVIDENCE_INVALID",
            "environment_policy_extra": "CP0_CHILD_EVIDENCE_INVALID",
            "command_policy_forged": "CP0_CHILD_EVIDENCE_INVALID",
            "provider_policy_forged": "CP0_CHILD_EVIDENCE_INVALID",
            "paper_probe_write_allowed": "CP0_CHILD_EVIDENCE_INVALID",
            "runtime_digest_mismatch": "CP0_CHILD_EVIDENCE_INVALID",
            "runtime_transport_size_mismatch": "CP0_CHILD_EVIDENCE_INVALID",
            "descriptor_transport_mismatch": "CP0_CHILD_EVIDENCE_INVALID",
        }
        for operation in ("source-pre", "paper-pre"):
            for mode, expected_code in cases.items():
                with self.subTest(
                    operation=operation, mode=mode
                ), tempfile.TemporaryDirectory(
                    prefix=f"exp7-replay-{operation}-{mode}-",
                    dir=self.temporary.name,
                ) as base:
                    fixture = SyntheticStrictRun(Path(base))
                    stage = fixture.bootstrap_pre_cp0(
                        child_evidence_mode=mode,
                        child_evidence_operation=operation,
                    )
                    with patch(
                        "strict_run.checkpoint._validate_cp0_child_evidence",
                        wraps=_validate_cp0_child_evidence,
                    ) as validator:
                        self._assert_canonical_cp0_replay_rejected(
                            stage, expected_code, fixture=fixture
                        )
                    self.assertEqual(
                        [call.args[4] for call in validator.call_args_list],
                        ["source-pre"]
                        if operation == "source-pre"
                        else ["source-pre", "paper-pre"],
                    )

    def test_canonical_replay_rejects_coordinated_protection_forgeries(
        self,
    ) -> None:
        for mode in (
            "coordinated_provider_syscalls_forged",
            "coordinated_provider_rights_forged",
            "coordinated_provider_kernel_forged",
            "coordinated_provider_machine_forged",
            "coordinated_probe_target_forged",
            "coordinated_probe_errno_forged",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                prefix=f"exp7-replay-coordinated-{mode}-",
                dir=self.temporary.name,
            ) as base:
                fixture = SyntheticStrictRun(Path(base))
                stage = fixture.bootstrap_pre_cp0(child_evidence_mode=mode)
                with patch(
                    "strict_run.checkpoint._validate_cp0_child_evidence",
                    wraps=_validate_cp0_child_evidence,
                ) as validator:
                    self._assert_canonical_cp0_replay_rejected(
                        stage,
                        "CP0_CHILD_EVIDENCE_INVALID",
                        fixture=fixture,
                    )
                self.assertEqual(
                    [call.args[4] for call in validator.call_args_list],
                    ["source-pre"],
                )

    def test_canonical_replay_rejects_coordinated_paper_payload_forgeries(
        self,
    ) -> None:
        for mode in (
            "coordinated_paper_path_forged",
            "coordinated_paper_provider_forged",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                prefix=f"exp7-replay-paper-payload-{mode}-",
                dir=self.temporary.name,
            ) as base:
                fixture = SyntheticStrictRun(Path(base))
                stage = fixture.bootstrap_pre_cp0(child_evidence_mode=mode)
                with patch(
                    "strict_run.checkpoint._validate_cp0_child_evidence",
                    wraps=_validate_cp0_child_evidence,
                ) as validator:
                    self._assert_canonical_cp0_replay_rejected(
                        stage,
                        "CP0_CHILD_EVIDENCE_INVALID",
                        fixture=fixture,
                    )
                self.assertEqual(
                    [call.args[4] for call in validator.call_args_list],
                    ["source-pre", "paper-pre"],
                )

    def test_canonical_replay_rejects_fully_resealed_false_source_records(
        self,
    ) -> None:
        for mode in (
            "coordinated_source_schema_forged",
            "coordinated_source_identity_forged",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                prefix=f"exp7-replay-source-record-{mode}-",
                dir=self.temporary.name,
            ) as base:
                fixture = SyntheticStrictRun(Path(base))
                stage = fixture.bootstrap_pre_cp0(child_evidence_mode=mode)
                with patch(
                    "strict_run.checkpoint._validate_cp0_child_evidence",
                    wraps=_validate_cp0_child_evidence,
                ) as validator:
                    self._assert_canonical_cp0_replay_rejected(
                        stage,
                        "CP0_CHILD_EVIDENCE_INVALID",
                        fixture=fixture,
                    )
                self.assertEqual(
                    [call.args[4] for call in validator.call_args_list],
                    ["source-pre"],
                )

    def test_canonical_replay_rejects_forged_paper_identities(self) -> None:
        for mode in (
            "coordinated_paper_descriptor_forged",
            "coordinated_paper_parent_forged",
            "coordinated_paper_size_forged",
        ):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(
                prefix=f"exp7-replay-paper-identity-{mode}-",
                dir=self.temporary.name,
            ) as base:
                fixture = SyntheticStrictRun(Path(base))
                stage = fixture.bootstrap_pre_cp0(child_evidence_mode=mode)
                with patch(
                    "strict_run.checkpoint._validate_cp0_child_evidence",
                    wraps=_validate_cp0_child_evidence,
                ) as validator:
                    self._assert_canonical_cp0_replay_rejected(
                        stage,
                        "CP0_CHILD_EVIDENCE_INVALID",
                        fixture=fixture,
                    )
                self.assertEqual(
                    [call.args[4] for call in validator.call_args_list],
                    ["source-pre", "paper-pre"],
                )

    def test_cp0_rejects_source_and_paper_pre_before_envelope(self) -> None:
        stage = self.fixture.bootstrap_pre_cp0(envelope_after_manifests=True)
        with self.fixture.transcript_registry(stage) as registry:
            with self.assertRaises(StrictRunError) as caught:
                publish_checkpoint(
                    str(self.fixture.parent),
                    RUN_ID,
                    str(self.fixture.run_root),
                    0,
                    stage,
                    self.fixture.writers["controller"],
                    self.fixture.policy,
                    transcript_registry=registry,
                )
        self.assertEqual(caught.exception.code, "CP0_CAUSAL_ORDER_INVALID")
        self.assertFalse(
            (self.fixture.run_root / "checkpoints" / "cp0.json").exists()
        )

    def test_cp0_rejects_envelope_without_acceptance_hash_context_binding(self) -> None:
        stage = self.fixture.bootstrap_pre_cp0(envelope_binding_valid=False)
        with self.fixture.transcript_registry(stage) as registry:
            with self.assertRaises(StrictRunError) as caught:
                publish_checkpoint(
                    str(self.fixture.parent),
                    RUN_ID,
                    str(self.fixture.run_root),
                    0,
                    stage,
                    self.fixture.writers["controller"],
                    self.fixture.policy,
                    transcript_registry=registry,
                )
        self.assertEqual(caught.exception.code, "CP0_CAUSAL_BINDING_INVALID")
        self.assertFalse(
            (self.fixture.run_root / "checkpoints" / "cp0.json").exists()
        )

    def test_stage_and_predecessor_must_match_typed_registry_exactly(self) -> None:
        cp0 = self.fixture.cp0()
        stage = self.fixture.cp1_stage()
        forged_transcript = copy.deepcopy(stage[-1].transcript)
        forged_transcript["context"] = {"forged": True}
        forged = Publication(stage[-1].evidence, forged_transcript)
        forged_stage = [*stage[:-1], forged]
        with self.fixture.transcript_registry(stage, cp0.publication) as registry:
            with self.assertRaises(StrictRunError) as caught:
                publish_checkpoint(
                    str(self.fixture.parent),
                    RUN_ID,
                    str(self.fixture.run_root),
                    1,
                    forged_stage,
                    self.fixture.writers["controller"],
                    self.fixture.policy,
                    transcript_registry=registry,
                    previous_checkpoint_publication=cp0.publication,
                )
        self.assertEqual(caught.exception.code, "TRANSCRIPT_REGISTRY_MISMATCH")

    def test_checkpoint_payload_binds_semantics_digest_and_rejects_unknown_fields(self) -> None:
        payload = copy.deepcopy(self.fixture.cp0().payload)
        payload["semantic_bindings"]["role_count"] = 11
        with self.assertRaises(StrictRunError) as caught:
            validate_checkpoint_payload(
                payload,
                expected_sealed_run_id=RUN_ID,
                expected_run_root=str(self.fixture.run_root),
            )
        self.assertEqual(caught.exception.code, "CHECKPOINT_SEMANTICS_INVALID")

        payload = copy.deepcopy(self.fixture.checkpoints[0].payload)
        payload["semantic_bindings"]["schema"] = "experiments7-cp1-semantics/v6"
        payload["semantic_bindings_sha256"] = sha256_bytes(
            canonical_json(payload["semantic_bindings"])
        )
        with self.assertRaises(StrictRunError) as caught:
            validate_checkpoint_payload(
                payload,
                expected_sealed_run_id=RUN_ID,
                expected_run_root=str(self.fixture.run_root),
            )
        self.assertEqual(caught.exception.code, "CHECKPOINT_SEMANTICS_INVALID")

        payload = copy.deepcopy(self.fixture.checkpoints[0].payload)
        payload["undeclared"] = True
        with self.assertRaises(StrictRunError) as caught:
            validate_checkpoint_payload(
                payload,
                expected_sealed_run_id=RUN_ID,
                expected_run_root=str(self.fixture.run_root),
            )
        self.assertEqual(caught.exception.code, "SCHEMA_INVALID")

    def test_forged_stage_evidence_and_bare_final_registry_are_rejected(self) -> None:
        cp0 = self.fixture.cp0()
        stage = self.fixture.cp1_stage()
        forged = Publication(
            replace(stage[-1].evidence, writer=self.fixture.writers["registry"]),
            stage[-1].transcript,
        )
        with self.fixture.transcript_registry(stage, cp0.publication) as registry:
            with self.assertRaises(StrictRunError) as caught:
                publish_checkpoint(
                    str(self.fixture.parent),
                    RUN_ID,
                    str(self.fixture.run_root),
                    1,
                    [*stage[:-1], forged],
                    self.fixture.writers["controller"],
                    self.fixture.policy,
                    transcript_registry=registry,
                    previous_checkpoint_publication=cp0.publication,
                )
        self.assertIn(caught.exception.code, {"TRANSCRIPT_MISMATCH", "WRITER_MISMATCH"})

        with self.assertRaises(StrictRunError) as caught:
            validate_final_tree(
                str(self.fixture.parent),
                RUN_ID,
                str(self.fixture.run_root),
                {},  # type: ignore[arg-type]
                cp0.publication,
            )
        self.assertEqual(caught.exception.code, "EXTERNAL_REGISTRY_REQUIRED")

    def test_checkpoint_booleans_fail_before_controller_mutation(self) -> None:
        stage = self.fixture.bootstrap_pre_cp0()
        checkpoint_directory = self.fixture.run_root / "checkpoints"
        before_stat = os.stat(checkpoint_directory, follow_symlinks=False)
        before_identity = stable_identity(before_stat)
        with self.fixture.transcript_registry(stage) as registry:
            for checkpoint in (False, True):
                with self.assertRaises(StrictRunError) as caught:
                    publish_checkpoint(
                        str(self.fixture.parent),
                        RUN_ID,
                        str(self.fixture.run_root),
                        checkpoint,
                        stage,
                        self.fixture.writers["controller"],
                        self.fixture.policy,
                        transcript_registry=registry,
                    )
                self.assertEqual(caught.exception.code, "CHECKPOINT_SEQUENCE_INVALID")
        after_stat = os.stat(checkpoint_directory, follow_symlinks=False)
        self.assertEqual(stable_identity(after_stat), before_identity)
        self.assertEqual(
            (
                after_stat.st_nlink,
                after_stat.st_size,
                after_stat.st_mtime_ns,
                after_stat.st_ctime_ns,
            ),
            (
                before_stat.st_nlink,
                before_stat.st_size,
                before_stat.st_mtime_ns,
                before_stat.st_ctime_ns,
            ),
        )
        self.assertFalse((self.fixture.run_root / "checkpoints" / "cp0.json").exists())


if __name__ == "__main__":
    unittest.main()
