"""Causal CP0..CP6 controller, failure terminal, and read-only tree closure."""
from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import os
import sys
import time
from typing import Iterable

from provenance.strict_v6 import V6ContractError, validate_source_pre_records

from .canonical import (
    StrictRunError,
    canonical_json,
    fail,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
    strict_json_loads,
    validate_absolute_path_text,
    validate_sealed_run_id,
)
from .filesystem import (
    READ_FLAGS,
    RunDescriptorBinding,
    assert_regular_or_directory,
    exists_relative,
    iter_tree,
    open_absolute_directory,
    open_bound_run_handle,
    open_run_handle,
    read_regular_at,
    stable_identity,
)
from .external import ExternalTranscriptRegistry
from .publication import (
    ArtifactEvidence,
    Publication,
    StagePublisher,
    transcript_hash,
    validate_publication_transcript,
)
from .reservation import (
    validate_acceptance_payload,
    validate_owner_binding_payload,
    validate_reservation_payload,
)
from .semantics import (
    CP0_SEMANTICS_SCHEMA,
    CP1_SEMANTICS_SCHEMA,
    CP2_SEMANTICS_SCHEMA,
    CP6_RESULT_SCHEMA,
    validate_stage_semantics,
)
from .writer_policy import WriterIdentity, WriterPolicy


CHECKPOINT_SCHEMA = "experiments7-strict-checkpoint/v6"
BLOCKED_SCHEMA = "experiments7-strict-blocked/v6"
CP0_ENVELOPE_EVIDENCE_SCHEMA = "experiments7-cp0-envelope-evidence/v6"
CP0_PROTECTED_READ_ATTESTATION_SCHEMA = (
    "experiments7-cp0-protected-read-attestation/v6"
)
CP0_PROTECTED_RESULT_SCHEMA = "experiments7-g0-protected-result/v6"
CP0_CHILD_EXECUTION_SCHEMA = "experiments7-g0-child-execution/v6"
CP0_DESCRIPTOR_TRANSPORT_SCHEMA = "experiments7-g0-descriptor-transport/v6"
CP0_ARGV_SCHEMA = "experiments7-g0-argv/v6"
CP0_REQUIRED_MEMFD_SEALS = 15
CP0_ENVIRONMENT_SCHEMA = "experiments7-g0-environment-allowlist/v6"
CP0_RUNTIME_SCHEMA = "experiments7-g0-runtime-identity/v6"
CP0_CONFIGURATION_SCHEMA = "experiments7-g0-protected-config/v6"
CP0_PRODUCER_BINDING_SCHEMA = "experiments7-cp0-producer-binding/v6"
CP0_TRUSTED_PROVIDER_SHA256 = (
    "11924a7f0458a56cc1a90c9f25bc1d47555063762415727e8d3977c1d7e1f045"
)
CP0_PRODUCER_IDENTITY_KEYS = frozenset(
    {"file_type", "st_dev", "st_ino", "mode", "size", "mtime_ns"}
)
CP0_DENIED_METADATA_SYSCALLS_X86_64 = (
    90,
    91,
    92,
    93,
    94,
    132,
    188,
    189,
    190,
    197,
    198,
    199,
    235,
    260,
    261,
    268,
    280,
    452,
)
CP0_LANDLOCK_READ_EXEC_RIGHTS = 13
CP0_PROTECTED_READ_ATTESTATION_KEYS = frozenset(
    {
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
)
CHECKPOINT_PATHS = tuple(f"checkpoints/cp{index}.json" for index in range(7))
SEMANTIC_SCHEMA_BY_CHECKPOINT = {
    0: CP0_SEMANTICS_SCHEMA,
    1: CP1_SEMANTICS_SCHEMA,
    2: CP2_SEMANTICS_SCHEMA,
    3: "experiments7-cp3-semantic-binding/v6",
    4: "experiments7-cp4-semantic-binding/v6",
    5: "experiments7-cp5-semantic-binding/v6",
    6: CP6_RESULT_SCHEMA,
}


@dataclass(frozen=True)
class CheckpointResult:
    checkpoint: int
    payload: dict[str, object]
    publication: Publication
    stage_publications: tuple[Publication, ...]

    @property
    def sha256(self) -> str:
        return self.publication.evidence.sha256 or ""

    @property
    def transcript(self) -> dict[str, object]:
        return self.publication.transcript


@dataclass(frozen=True)
class _SemanticReplayContext:
    checkpoint_publications: dict[int, Publication]
    stage_publications: dict[int, tuple[Publication, ...]]


def _load_json(root_fd: int, relative: str) -> tuple[dict[str, object], bytes]:
    raw, _ = read_regular_at(root_fd, relative)
    try:
        value = strict_json_loads(raw, relative)
    except ValueError as exc:
        raise ValueError(f"invalid JSON at {relative}") from exc
    if not isinstance(value, dict) or canonical_json(value) != raw:
        fail("NONCANONICAL_JSON", f"{relative} is not canonical JSON")
    return value, raw


def _stage_allows(checkpoint: int, path: str) -> bool:
    if checkpoint == 0:
        return (
            path in {
                "reservation.json", "reservation-acceptance.json", "owner-binding.json",
                "checkpoints", "manifests", "manifests/source-pre.jsonl",
                "manifests/paper-pre.json",
            }
            or path == "frozen"
            or path.startswith("frozen/")
        )
    if checkpoint == 1:
        return path in {"registry", "inventory"} or path.startswith(("registry/", "inventory/"))
    if checkpoint == 2:
        prefixes = ("adapters", "snapshots", "runtime", "run-configs", "golden")
        return path in prefixes or path.startswith(tuple(f"{prefix}/" for prefix in prefixes))
    if checkpoint == 3:
        return (
            path == "provenance" or path.startswith("provenance/")
            or path in {"admission", "admission/precopy"}
            or path.startswith("admission/precopy/")
        )
    if checkpoint == 4:
        return path in {"raw", "copies.jsonl"} or path.startswith("raw/")
    if checkpoint == 5:
        return (
            path == "admission/final" or path.startswith("admission/final/")
            or path in {"manifests/source-post.jsonl", "manifests/paper-post.json"}
        )
    if checkpoint == 6:
        return path == "verification" or path.startswith("verification/")
    return False


def _cp0_bindings(value: object) -> dict[str, object]:
    row = require_exact_keys(
        value,
        {
            "reservation_sha256", "reservation_acceptance_sha256", "owner_binding_sha256",
            "reservation_transcript_sha256", "acceptance_transcript_sha256",
            "writer_policy_sha256",
        },
        "cp0-bindings",
    )
    for key in row:
        require_sha256(row[key], key)
    return row


def validate_checkpoint_payload(
    value: object,
    *,
    expected_sealed_run_id: str,
    expected_run_root: str,
    expected_checkpoint: int | None = None,
    policy: WriterPolicy | None = None,
) -> dict[str, object]:
    expected_sealed_run_id = validate_sealed_run_id(expected_sealed_run_id)
    expected_run_root = validate_absolute_path_text(expected_run_root, "expected_run_root")
    if expected_checkpoint is not None and (
        type(expected_checkpoint) is not int or expected_checkpoint not in range(7)
    ):
        fail("CHECKPOINT_SEQUENCE_INVALID", "expected checkpoint index is invalid")
    row = require_exact_keys(
        value,
        {
            "schema", "sealed_run_id", "run_root", "checkpoint",
            "previous_checkpoint_sha256", "previous_checkpoint_transcript_sha256",
            "stage_actual_paths", "stage_actual_paths_sha256", "bindings",
            "semantic_bindings", "semantic_bindings_sha256",
        },
        CHECKPOINT_SCHEMA,
    )
    if row["schema"] != CHECKPOINT_SCHEMA:
        fail("CHECKPOINT_SCHEMA_INVALID", "checkpoint schema differs")
    if (
        row["sealed_run_id"] != expected_sealed_run_id
        or row["run_root"] != expected_run_root
    ):
        fail("CHECKPOINT_RUN_BINDING_MISMATCH", "checkpoint run ID/root differs")
    checkpoint = row["checkpoint"]
    if type(checkpoint) is not int or checkpoint not in range(7):
        fail("CHECKPOINT_SCHEMA_INVALID", "checkpoint index is invalid")
    if expected_checkpoint is not None and checkpoint != expected_checkpoint:
        fail("CHECKPOINT_SEQUENCE_INVALID", "checkpoint index differs")
    if checkpoint == 0:
        if (
            row["previous_checkpoint_sha256"] is not None
            or row["previous_checkpoint_transcript_sha256"] is not None
        ):
            fail("CHECKPOINT_SELF_OR_FUTURE", "CP0 cannot name a predecessor")
        _cp0_bindings(row["bindings"])
    else:
        require_sha256(row["previous_checkpoint_sha256"], "previous checkpoint hash")
        require_sha256(
            row["previous_checkpoint_transcript_sha256"], "previous checkpoint transcript"
        )
        require_exact_keys(row["bindings"], set(), "non-cp0-bindings")
    if not isinstance(row["stage_actual_paths"], list) or not row["stage_actual_paths"]:
        fail("CHECKPOINT_STAGE_INVALID", "stage path set must be nonempty")
    evidence = [ArtifactEvidence.from_dict(item) for item in row["stage_actual_paths"]]
    paths = [item.relative_path for item in evidence]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        fail("CHECKPOINT_STAGE_INVALID", "stage paths must be sorted and unique")
    if any(path in CHECKPOINT_PATHS or not _stage_allows(checkpoint, path) for path in paths):
        fail("CHECKPOINT_SELF_OR_FUTURE", "stage contains a self/future/wrong-stage path")
    if row["stage_actual_paths_sha256"] != sha256_bytes(canonical_json(row["stage_actual_paths"])):
        fail("CHECKPOINT_STAGE_INVALID", "stage path-set digest differs")
    if not isinstance(row["semantic_bindings"], dict):
        fail("CHECKPOINT_SEMANTICS_INVALID", "semantic bindings must be an object")
    if row["semantic_bindings_sha256"] != sha256_bytes(
        canonical_json(row["semantic_bindings"])
    ):
        fail("CHECKPOINT_SEMANTICS_INVALID", "semantic bindings digest differs")
    expected_semantic_schema = SEMANTIC_SCHEMA_BY_CHECKPOINT.get(checkpoint)
    if expected_semantic_schema is None:
        fail(
            "SEMANTICS_UNSUPPORTED_CHECKPOINT",
            "checkpoint semantics are not integrated for this stage",
        )
    if row["semantic_bindings"].get("schema") != expected_semantic_schema:
        fail("CHECKPOINT_SEMANTICS_INVALID", "stage semantic schema differs")
    if (
        row["semantic_bindings"].get("sealed_run_id") != expected_sealed_run_id
        or row["semantic_bindings"].get("run_root") != expected_run_root
        or row["semantic_bindings"].get("checkpoint") != checkpoint
    ):
        fail("CHECKPOINT_SEMANTICS_INVALID", "semantic bindings run/index differs")
    if policy is not None:
        for item in evidence:
            rule = policy.authorize(item.relative_path, item.writer)
            if rule.rule_id != item.writer_rule_id:
                fail("WRITER_MISMATCH", "stage writer rule evidence differs")
    return row


def _validate_publication(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    publication: Publication,
    policy: WriterPolicy,
    transcript_registry: ExternalTranscriptRegistry,
) -> None:
    evidence = publication.evidence
    registered = transcript_registry.get_exact(
        evidence.publication_transcript_sha256
    )
    if not isinstance(registered, dict) or registered != publication.transcript:
        fail(
            "TRANSCRIPT_REGISTRY_MISMATCH",
            "publication transcript is not the exact externally registered object",
        )
    transcript = validate_publication_transcript(
        root_fd,
        sealed_run_id,
        run_root,
        registered,
        expected_relative_path=evidence.relative_path,
        expected_writer=evidence.writer,
    )
    if transcript_hash(transcript) != evidence.publication_transcript_sha256:
        fail("TRANSCRIPT_MISMATCH", "artifact evidence transcript digest differs")
    rule = policy.authorize(evidence.relative_path, evidence.writer)
    if rule.rule_id != evidence.writer_rule_id:
        fail("WRITER_MISMATCH", "artifact writer rule differs")
    if (
        transcript["artifact_type"] != evidence.artifact_type
        or transcript["sha256"] != evidence.sha256
        or transcript["bytes"] != evidence.size
        or transcript["identity"] != evidence.identity
    ):
        fail("EVIDENCE_MISMATCH", "artifact evidence and transcript differ")


def _publication_monotonic_bounds(
    publication: Publication,
    field: str,
) -> tuple[int, int]:
    transcript = publication.transcript
    events = transcript.get("events")
    if not isinstance(events, list) or not events or not isinstance(events[0], dict):
        fail("CHECKPOINT_CHRONOLOGY_INVALID", f"{field} transcript events are invalid")
    first_completed = events[0].get("completed_monotonic_ns")
    emitted = transcript.get("emitted_monotonic_ns")
    if (
        type(first_completed) is not int
        or type(emitted) is not int
        or first_completed < 0
        or emitted < first_completed
    ):
        fail("CHECKPOINT_CHRONOLOGY_INVALID", f"{field} transcript times are invalid")
    return first_completed, emitted


def _cp0_causal_precedes(
    earlier: Publication,
    later: Publication,
    relation: str,
) -> None:
    _, earlier_emitted = _publication_monotonic_bounds(earlier, relation)
    later_first_completed, _ = _publication_monotonic_bounds(later, relation)
    if earlier_emitted >= later_first_completed:
        fail(
            "CP0_CAUSAL_ORDER_INVALID",
            f"CP0 causal predecessor does not strictly precede {relation}",
        )


def _validate_cp0_protected_read_attestation(
    acceptance: Publication,
    envelope: Publication,
    publication: Publication,
    operation: str,
    sealed_run_id: str,
    run_root: str,
) -> dict[str, object]:
    context = publication.transcript.get("context")
    required = set(CP0_PROTECTED_READ_ATTESTATION_KEYS) | {
        "child_stdout_sha256",
        "child_execution_sha256",
        "child_execution_evidence",
    }
    if not isinstance(context, dict) or set(context) != required:
        fail(
            "CP0_CAUSAL_BINDING_INVALID",
            f"{operation} producer attestation schema is not exact",
        )
    started = context["started_monotonic_ns"]
    expected_envelope_evidence_sha256 = sha256_bytes(
        canonical_json(envelope.evidence.to_dict())
    )
    if (
        context["schema"] != CP0_PROTECTED_READ_ATTESTATION_SCHEMA
        or context["sealed_run_id"] != sealed_run_id
        or context["run_root"] != run_root
        or context["operation"] != operation
        or type(started) is not int
        or started < 0
        or context["acceptance_transcript_sha256"]
        != acceptance.evidence.publication_transcript_sha256
        or context["envelope_artifact_evidence_sha256"]
        != expected_envelope_evidence_sha256
        or context["envelope_transcript_sha256"]
        != envelope.evidence.publication_transcript_sha256
        or context["envelope_context"] != envelope.transcript.get("context")
    ):
        fail(
            "CP0_CAUSAL_BINDING_INVALID",
            f"{operation} producer attestation binding differs",
        )
    _, acceptance_emitted = _publication_monotonic_bounds(
        acceptance, f"{operation} acceptance gate"
    )
    _, envelope_emitted = _publication_monotonic_bounds(
        envelope, f"{operation} envelope gate"
    )
    publication_first, _ = _publication_monotonic_bounds(
        publication, f"{operation} publication"
    )
    if not (
        acceptance_emitted < started
        and envelope_emitted < started
        and started < publication_first
    ):
        fail(
            "CP0_CAUSAL_BINDING_INVALID",
            f"{operation} protected read interval is not strictly gated",
        )
    require_sha256(context["child_stdout_sha256"], "child stdout sha256")
    require_sha256(context["child_execution_sha256"], "child execution sha256")
    return context


def _cp0_whole_external_object(
    transcript_registry: ExternalTranscriptRegistry,
    digest: object,
    field: str,
) -> tuple[dict[str, object], object]:
    exact_digest = require_sha256(digest, field)
    value = transcript_registry.get_exact(exact_digest)
    documents = [
        document
        for document in transcript_registry.documents.values()
        if document.sha256 == exact_digest
    ]
    if len(documents) != 1 or not isinstance(value, dict):
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{field} must resolve to one whole external JSON object",
        )
    document = documents[0]
    if document.value != value or document.raw_bytes != canonical_json(value):
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{field} external document bytes differ",
        )
    return value, document


def _cp0_exact_nonnegative_int(value: object, field: str) -> int:
    if type(value) is not int or value < 0:
        fail("CP0_CHILD_EVIDENCE_INVALID", f"{field} must be an exact nonnegative integer")
    return value


def _validate_cp0_producer_identity(
    value: object,
    expected_file_type: str,
    field: str,
) -> dict[str, object]:
    identity = require_exact_keys(
        value,
        set(CP0_PRODUCER_IDENTITY_KEYS),
        field,
    )
    if identity["file_type"] != expected_file_type:
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{field} file type differs from the protected producer schema",
        )
    for name in CP0_PRODUCER_IDENTITY_KEYS - {"file_type"}:
        _cp0_exact_nonnegative_int(identity[name], f"{field}.{name}")
    return identity


def _descriptor_bound_absolute_sha256(
    path: str, label: str
) -> tuple[str, int]:
    path = validate_absolute_path_text(path, label)
    if os.path.abspath(path) != path or os.path.realpath(path) != path:
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{label} is not a canonical real path",
        )
    parent, name = os.path.split(path)
    if not parent or not name:
        fail("CP0_CHILD_EVIDENCE_INVALID", f"{label} path is invalid")
    try:
        parent_fd = open_absolute_directory(parent)
    except (OSError, StrictRunError) as exc:
        raise StrictRunError(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{label} parent is missing, linked, or noncanonical",
        ) from exc
    try:
        try:
            fd = os.open(name, READ_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise StrictRunError(
                "CP0_CHILD_EVIDENCE_INVALID",
                f"{label} is missing, linked, or unreadable",
            ) from exc
        try:
            before = os.fstat(fd)
            before_identity = stable_identity(before)
            if before_identity["type"] != "regular":
                fail("CP0_CHILD_EVIDENCE_INVALID", f"{label} is not regular")
            descriptor_path = os.readlink(f"/proc/self/fd/{fd}")
            if descriptor_path != path or descriptor_path.endswith(" (deleted)"):
                fail(
                    "CP0_CHILD_EVIDENCE_INVALID",
                    f"{label} descriptor identity differs",
                )
            digest = hashlib.sha256()
            total = 0
            while True:
                block = os.read(fd, 1024 * 1024)
                if not block:
                    break
                total += len(block)
                digest.update(block)
            after = os.fstat(fd)
            namespace = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                stable_identity(after) != before_identity
                or stable_identity(namespace) != before_identity
                or total != after.st_size
            ):
                fail(
                    "CP0_CHILD_EVIDENCE_INVALID",
                    f"{label} changed during descriptor-bound hashing",
                )
            return digest.hexdigest(), total
        finally:
            os.close(fd)
    finally:
        os.close(parent_fd)


def _validate_cp0_runtime_executable(
    python_path: str,
    frozen_sha256: object,
    execution_sha256: object,
) -> tuple[str, int]:
    python_path = validate_absolute_path_text(
        python_path, "frozen runtime python path"
    )
    trusted_path = validate_absolute_path_text(
        os.path.realpath(os.path.abspath(sys.executable)),
        "checkpoint controller python path",
    )
    if python_path != trusted_path:
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            "frozen runtime path is not the trusted controller interpreter",
        )
    frozen_digest = require_sha256(
        frozen_sha256, "frozen runtime python sha256"
    )
    execution_digest = require_sha256(
        execution_sha256, "child runtime executable sha256"
    )
    actual_digest, actual_size = _descriptor_bound_absolute_sha256(
        python_path, "frozen runtime python"
    )
    if frozen_digest != actual_digest or execution_digest != actual_digest:
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            "child runtime digest differs from descriptor-bound interpreter bytes",
        )
    return actual_digest, actual_size


def _validate_cp0_protection_evidence(
    root_fd: int,
    stdout: dict[str, object],
    operation: str,
    sealed_run_id: str,
    run_root: str,
) -> tuple[str, dict[str, object]]:
    configuration, _ = _load_json(
        root_fd, "frozen/configuration/g0-config.json"
    )
    require_exact_keys(
        configuration,
        {
            "schema",
            "sealed_run_id",
            "run_root",
            "provider_path",
            "provider_sha256",
            "g0_executable_sha256",
            "source_roots",
            "paper_path",
        },
        "frozen G0 configuration",
    )
    paper_path = validate_absolute_path_text(
        configuration["paper_path"], "frozen configuration paper path"
    )
    runtime, _ = _load_json(
        root_fd, "frozen/runtime_identity/runtime.json"
    )
    require_exact_keys(runtime, {"schema", "python", "platform"}, "frozen runtime")
    runtime_platform = require_exact_keys(
        runtime["platform"],
        {"system", "release", "machine"},
        "frozen runtime platform",
    )
    audit, _ = _load_json(
        root_fd, "frozen/runtime_identity/envelope-audit.json"
    )
    require_exact_keys(
        audit,
        {
            "schema",
            "mode",
            "sealed_run_id",
            "run_root",
            "python_dont_write_bytecode",
            "pythonhashseed",
            "provider_evidence",
            "paper_write_probe",
        },
        "frozen envelope audit",
    )
    provider = require_exact_keys(
        audit["provider_evidence"],
        {
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
        },
        "frozen provider evidence",
    )
    denied_syscalls = provider["denied_metadata_syscalls_x86_64"]
    rules = provider["landlock_rules"]
    if not isinstance(rules, list) or len(rules) != 1:
        fail("CP0_CHILD_EVIDENCE_INVALID", "frozen Landlock rule set is invalid")
    rule = require_exact_keys(
        rules[0], {"path", "dev", "inode", "rights"}, "frozen Landlock rule"
    )
    if (
        audit["schema"] != CP0_PROTECTED_RESULT_SCHEMA
        or audit["mode"] != "verify-envelope"
        or audit["sealed_run_id"] != sealed_run_id
        or audit["run_root"] != run_root
        or configuration["schema"] != CP0_CONFIGURATION_SCHEMA
        or configuration["sealed_run_id"] != sealed_run_id
        or configuration["run_root"] != run_root
        or runtime["schema"] != CP0_RUNTIME_SCHEMA
        or runtime_platform["system"] != "Linux"
        or not isinstance(runtime_platform["release"], str)
        or not runtime_platform["release"]
        or runtime_platform["machine"] != "x86_64"
        or audit["python_dont_write_bytecode"] is not True
        or audit["pythonhashseed"] != "0"
        or provider["provider"]
        != "landlock_path_beneath+seccomp_metadata_deny"
        or type(provider["landlock_abi"]) is not int
        or provider["landlock_abi"] < 4
        or provider["seccomp_mode"] != "classic_bpf_errno_eperm"
        or denied_syscalls != list(CP0_DENIED_METADATA_SYSCALLS_X86_64)
        or provider["kernel"] != runtime_platform["release"]
        or provider["machine"] != runtime_platform["machine"]
        or rule["path"] != "/"
        or any(type(rule[field]) is not int or rule[field] < 0 for field in ("dev", "inode"))
        or rule["rights"] != CP0_LANDLOCK_READ_EXEC_RIGHTS
        or provider["default_filesystem_rights"] != "read_execute_only"
        or provider["writable_directories"] != []
        or provider["metadata_mutation_policy"]
        != "globally_denied_by_seccomp"
    ):
        fail("CP0_CHILD_EVIDENCE_INVALID", "frozen protection evidence is invalid")
    probe = require_exact_keys(
        audit["paper_write_probe"],
        {"operation", "target_path", "denied", "errno"},
        "frozen paper write probe",
    )
    if (
        probe["operation"] != "open(O_WRONLY|O_NOFOLLOW)"
        or probe["target_path"] != paper_path
        or probe["denied"] is not True
        or type(probe["errno"]) is not int
        or probe["errno"] not in {errno.EACCES, errno.EPERM}
        or stdout["provider_evidence"] != provider
        or stdout["paper_write_probe"] != probe
    ):
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{operation} protection evidence differs from the frozen audit",
        )
    provider_identity_keys = (
        "provider",
        "landlock_abi",
        "seccomp_mode",
        "denied_metadata_syscalls_x86_64",
        "kernel",
        "machine",
    )
    return paper_path, {
        key: provider[key] for key in provider_identity_keys
    }


def _validate_cp0_child_evidence(
    root_fd: int,
    acceptance: Publication,
    envelope: Publication,
    publication: Publication,
    operation: str,
    sealed_run_id: str,
    run_root: str,
    by_path: dict[str, Publication],
    transcript_registry: ExternalTranscriptRegistry,
    context: dict[str, object],
) -> None:
    stdout, stdout_document = _cp0_whole_external_object(
        transcript_registry,
        context["child_stdout_sha256"],
        f"{operation} child stdout sha256",
    )
    execution, execution_document = _cp0_whole_external_object(
        transcript_registry,
        context["child_execution_sha256"],
        f"{operation} child execution sha256",
    )

    stdout_keys = {
        "schema",
        "mode",
        "sealed_run_id",
        "run_root",
        "provider_evidence",
        "paper_write_probe",
        "producer_attestation",
        "records" if operation == "source-pre" else "paper",
    }
    require_exact_keys(stdout, stdout_keys, f"{operation} child stdout")
    child_attestation = {
        key: context[key] for key in CP0_PROTECTED_READ_ATTESTATION_KEYS
    }
    if (
        stdout["schema"] != CP0_PROTECTED_RESULT_SCHEMA
        or stdout["mode"] != operation
        or stdout["sealed_run_id"] != sealed_run_id
        or stdout["run_root"] != run_root
        or stdout["producer_attestation"] != child_attestation
    ):
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{operation} child stdout binding differs",
        )
    paper_path, provider_identity = _validate_cp0_protection_evidence(
        root_fd, stdout, operation, sealed_run_id, run_root
    )
    if operation == "source-pre":
        records = stdout["records"]
        try:
            normalized_records = validate_source_pre_records(records)
        except V6ContractError as exc:
            raise StrictRunError(
                "CP0_CHILD_EVIDENCE_INVALID",
                "source-pre stdout records fail canonical validation",
            ) from exc
        if normalized_records != records:
            fail(
                "CP0_CHILD_EVIDENCE_INVALID",
                "source-pre stdout records differ from canonical normalization",
            )
        expected_manifest = b"".join(canonical_json(record) for record in records)
    else:
        paper = require_exact_keys(
            stdout["paper"],
            {
                "schema",
                "sealed_run_id",
                "run_root",
                "paper_path",
                "sha256",
                "size",
                "descriptor_identity",
                "parent_identity",
                "provider",
            },
            "paper-pre stdout record",
        )
        paper_provider = require_exact_keys(
            paper["provider"],
            set(provider_identity),
            "paper-pre nested provider identity",
        )
        if (
            paper["schema"] != "experiments7-paper-pre/v6"
            or paper["sealed_run_id"] != sealed_run_id
            or paper["run_root"] != run_root
            or paper["paper_path"] != paper_path
            or paper_provider != provider_identity
        ):
            fail("CP0_CHILD_EVIDENCE_INVALID", "paper-pre stdout binding differs")
        require_sha256(paper["sha256"], "paper-pre sha256")
        paper_size = _cp0_exact_nonnegative_int(
            paper["size"], "paper-pre size"
        )
        descriptor_identity = _validate_cp0_producer_identity(
            paper["descriptor_identity"],
            "regular",
            "paper-pre descriptor identity",
        )
        _validate_cp0_producer_identity(
            paper["parent_identity"],
            "directory",
            "paper-pre parent identity",
        )
        if descriptor_identity["size"] != paper_size:
            fail(
                "CP0_CHILD_EVIDENCE_INVALID",
                "paper-pre size differs from descriptor identity",
            )
        expected_manifest = canonical_json(paper)
    manifest_raw, _ = read_regular_at(root_fd, publication.evidence.relative_path)
    if manifest_raw != expected_manifest:
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{operation} manifest is not the exact child stdout payload",
        )

    execution_keys = {
        "schema",
        "sealed_run_id",
        "run_root",
        "mode",
        "argv",
        "argv_sha256",
        "descriptor_transport",
        "runtime_executable_sha256",
        "executable_sha256",
        "configuration_sha256",
        "provider_sha256",
        "producer_binding_sha256",
        "environment",
        "environment_sha256",
        "launched_monotonic_ns",
        "completed_monotonic_ns",
        "exit_status",
        "stderr_sha256",
        "stderr_bytes",
        "stdout_evidence",
    }
    require_exact_keys(execution, execution_keys, f"{operation} child execution")
    argv = execution["argv"]
    if (
        execution["schema"] != CP0_CHILD_EXECUTION_SCHEMA
        or execution["sealed_run_id"] != sealed_run_id
        or execution["run_root"] != run_root
        or execution["mode"] != operation
        or not isinstance(argv, list)
        or not argv
        or any(not isinstance(arg, str) or not arg for arg in argv)
        or argv[-1] != "--producer-binding-stdin"
        or execution["argv_sha256"] != sha256_bytes(canonical_json(argv))
    ):
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{operation} child execution argv binding differs",
        )

    argv_path = (
        "frozen/pre_argv/argv.json"
        if operation == "source-pre"
        else "frozen/post_argv/argv.json"
    )
    frozen_argv, _ = _load_json(root_fd, argv_path)
    require_exact_keys(frozen_argv, {"schema", "mode", "argv"}, argv_path)
    runtime, _ = _load_json(root_fd, "frozen/runtime_identity/runtime.json")
    require_exact_keys(runtime, {"schema", "python", "platform"}, "frozen runtime")
    runtime_python = require_exact_keys(
        runtime["python"],
        {"path", "sha256", "version", "implementation"},
        "frozen runtime python",
    )
    runtime_platform = require_exact_keys(
        runtime["platform"],
        {"system", "release", "machine"},
        "frozen runtime platform",
    )
    python_path = validate_absolute_path_text(
        runtime_python["path"], "frozen runtime python path"
    )
    require_sha256(runtime_python["sha256"], "frozen runtime python sha256")
    if (
        runtime["schema"] != CP0_RUNTIME_SCHEMA
        or any(
            not isinstance(runtime_python[field], str)
            or not runtime_python[field]
            for field in ("version", "implementation")
        )
        or runtime_platform["system"] != "Linux"
        or runtime_platform["machine"] != "x86_64"
        or not isinstance(runtime_platform["release"], str)
        or not runtime_platform["release"]
    ):
        fail("CP0_CHILD_EVIDENCE_INVALID", "frozen runtime policy is invalid")
    _, actual_runtime_size = _validate_cp0_runtime_executable(
        python_path,
        runtime_python["sha256"],
        execution["runtime_executable_sha256"],
    )
    expected_argv = [
        python_path,
        "-B",
        f"{run_root}/frozen/g0_executable/g0_protected.py",
        "--config",
        f"{run_root}/frozen/configuration/g0-config.json",
        "--mode",
        operation,
        "--producer-binding-stdin",
    ]
    if (
        frozen_argv["schema"] != CP0_ARGV_SCHEMA
        or frozen_argv["mode"] != operation
        or frozen_argv["argv"] != argv
        or argv != expected_argv
    ):
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{operation} execution differs from frozen argv",
        )

    transport = require_exact_keys(
        execution["descriptor_transport"],
        {
            "schema",
            "logical_argv",
            "executed_argv",
            "executed_argv_sha256",
            "runtime",
            "artifacts",
            "run_root",
        },
        f"{operation} descriptor transport",
    )
    runtime_transport = require_exact_keys(
        transport["runtime"],
        {"fd", "proc_path", "sha256", "bytes", "seals"},
        f"{operation} runtime descriptor",
    )
    run_transport = require_exact_keys(
        transport["run_root"],
        {"identity", "mount_id"},
        f"{operation} run descriptor",
    )
    reservation, _ = _load_json(root_fd, "reservation.json")
    validate_reservation_payload(
        reservation,
        sealed_run_id,
        os.path.dirname(run_root),
        run_root,
    )
    reservation_child = require_exact_keys(
        reservation["child"],
        {"name", "path", "descriptor", "mount"},
        "reservation child",
    )
    reservation_child_descriptor = require_exact_keys(
        reservation_child["descriptor"],
        {"identity", "status_flags", "descriptor_flags"},
        "reservation child descriptor",
    )
    reservation_child_mount = require_exact_keys(
        reservation_child["mount"],
        {
            "mount_id",
            "parent_mount_id",
            "major_minor",
            "mount_root",
            "mount_point",
            "mount_options",
            "filesystem_type",
            "mount_source",
            "super_options",
            "mount_namespace",
        },
        "reservation child mount",
    )
    logical_argv = transport["logical_argv"]
    executed_argv = transport["executed_argv"]
    if (
        transport["schema"] != CP0_DESCRIPTOR_TRANSPORT_SCHEMA
        or logical_argv != argv
        or not isinstance(executed_argv, list)
        or transport["executed_argv_sha256"]
        != sha256_bytes(canonical_json(executed_argv))
        or type(runtime_transport["fd"]) is not int
        or runtime_transport["fd"] < 0
        or runtime_transport["proc_path"]
        != f"/proc/self/fd/{runtime_transport['fd']}"
        or runtime_transport["sha256"]
        != execution["runtime_executable_sha256"]
        or type(runtime_transport["bytes"]) is not int
        or runtime_transport["bytes"] != actual_runtime_size
        or runtime_transport["seals"] != CP0_REQUIRED_MEMFD_SEALS
        or run_transport["identity"]
        != reservation_child_descriptor["identity"]
        or run_transport["mount_id"] != reservation_child_mount["mount_id"]
    ):
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{operation} descriptor transport binding differs",
        )
    artifact_rows = transport["artifacts"]
    if not isinstance(artifact_rows, dict) or set(artifact_rows) != {
        "g0_executable",
        "configuration",
        "provider",
    }:
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{operation} descriptor artifact roles differ",
        )
    expected_transport_paths = {
        "g0_executable": "frozen/g0_executable/g0_protected.py",
        "configuration": "frozen/configuration/g0-config.json",
        "provider": "frozen/provider/provider.py",
    }
    descriptor_fds = {runtime_transport["fd"]}
    normalized_rows: dict[str, dict[str, object]] = {}
    for role, relative_path in expected_transport_paths.items():
        row = require_exact_keys(
            artifact_rows[role],
            {
                "relative_path",
                "sha256",
                "bytes",
                "identity",
                "mount_id",
                "fd",
                "proc_path",
                "seals",
            },
            f"{operation} {role} descriptor",
        )
        publication_row = by_path.get(relative_path)
        if (
            publication_row is None
            or type(row["fd"]) is not int
            or row["fd"] < 0
            or row["fd"] in descriptor_fds
            or row["proc_path"] != f"/proc/self/fd/{row['fd']}"
            or row["relative_path"] != relative_path
            or row["sha256"] != publication_row.evidence.sha256
            or row["bytes"] != publication_row.evidence.size
            or row["identity"] != publication_row.evidence.identity
            or row["mount_id"] != run_transport["mount_id"]
            or row["seals"] != CP0_REQUIRED_MEMFD_SEALS
        ):
            fail(
                "CP0_CHILD_EVIDENCE_INVALID",
                f"{operation} {role} descriptor binding differs",
            )
        descriptor_fds.add(row["fd"])
        normalized_rows[role] = row
    expected_executed_argv = list(argv)
    expected_executed_argv[2] = normalized_rows["g0_executable"]["proc_path"]
    expected_executed_argv[4] = normalized_rows["configuration"]["proc_path"]
    expected_executed_argv[5:5] = [
        "--provider-fd",
        str(normalized_rows["provider"]["fd"]),
    ]
    if executed_argv != expected_executed_argv:
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{operation} executed descriptor argv differs",
        )

    frozen_hash_paths = {
        "executable_sha256": "frozen/g0_executable/g0_protected.py",
        "configuration_sha256": "frozen/configuration/g0-config.json",
        "provider_sha256": "frozen/provider/provider.py",
    }
    for field, path in frozen_hash_paths.items():
        expected = by_path.get(path)
        if (
            expected is None
            or expected.evidence.artifact_type != "regular"
            or execution[field] != expected.evidence.sha256
        ):
            fail(
                "CP0_CHILD_EVIDENCE_INVALID",
                f"{operation} {field} differs from frozen evidence",
            )

    provider_raw, _ = read_regular_at(root_fd, "frozen/provider/provider.py")
    provider_publication = by_path.get("frozen/provider/provider.py")
    if (
        sha256_bytes(provider_raw) != CP0_TRUSTED_PROVIDER_SHA256
        or provider_publication is None
        or provider_publication.evidence.sha256
        != CP0_TRUSTED_PROVIDER_SHA256
        or execution["provider_sha256"] != CP0_TRUSTED_PROVIDER_SHA256
    ):
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            "frozen provider is not the trusted canonical implementation",
        )

    envelope_payload, _ = _load_json(root_fd, envelope.evidence.relative_path)
    producer_binding = {
        "schema": CP0_PRODUCER_BINDING_SCHEMA,
        "acceptance_transcript": acceptance.transcript,
        "envelope_artifact_evidence": envelope.evidence.to_dict(),
        "envelope_transcript": envelope.transcript,
        "envelope_payload": envelope_payload,
    }
    if execution["producer_binding_sha256"] != sha256_bytes(
        canonical_json(producer_binding)
    ):
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{operation} producer binding digest differs",
        )

    environment_payload, _ = _load_json(
        root_fd, "frozen/environment_allowlist/environment.json"
    )
    require_exact_keys(
        environment_payload,
        {"schema", "environment", "environment_sha256"},
        "frozen environment allowlist",
    )
    environment = environment_payload["environment"]
    environment_keys = {
        "PATH",
        "HOME",
        "LANG",
        "LC_ALL",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHASHSEED",
    }
    if (
        environment_payload["schema"] != CP0_ENVIRONMENT_SCHEMA
        or not isinstance(environment, dict)
        or set(environment) != environment_keys
        or any(
            not isinstance(key, str)
            or not isinstance(value, str)
            or "\x00" in value
            for key, value in environment.items()
        )
        or environment["PYTHONDONTWRITEBYTECODE"] != "1"
        or environment["PYTHONHASHSEED"] != "0"
        or environment_payload["environment_sha256"]
        != sha256_bytes(canonical_json(environment))
        or execution["environment"] != environment
        or execution["environment_sha256"]
        != environment_payload["environment_sha256"]
    ):
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{operation} child environment binding differs",
        )

    stdout_evidence = require_exact_keys(
        execution["stdout_evidence"],
        {"path", "sha256", "bytes", "identity", "parent_identity"},
        f"{operation} stdout evidence",
    )
    if (
        stdout_evidence != stdout_document.evidence()
        or stdout_evidence["sha256"] != context["child_stdout_sha256"]
    ):
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{operation} stdout external evidence differs",
        )
    execution_evidence = require_exact_keys(
        context["child_execution_evidence"],
        {"path", "sha256", "bytes", "identity", "parent_identity"},
        f"{operation} execution evidence",
    )
    if (
        execution_evidence != execution_document.evidence()
        or execution_evidence["sha256"] != context["child_execution_sha256"]
    ):
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{operation} execution external evidence differs",
        )

    launched = _cp0_exact_nonnegative_int(
        execution["launched_monotonic_ns"], "child launch time"
    )
    completed = _cp0_exact_nonnegative_int(
        execution["completed_monotonic_ns"], "child completion time"
    )
    exit_status = _cp0_exact_nonnegative_int(
        execution["exit_status"], "child exit status"
    )
    stderr_bytes = _cp0_exact_nonnegative_int(
        execution["stderr_bytes"], "child stderr bytes"
    )
    started = context["started_monotonic_ns"]
    _, acceptance_emitted = _publication_monotonic_bounds(
        acceptance, f"{operation} child acceptance gate"
    )
    _, envelope_emitted = _publication_monotonic_bounds(
        envelope, f"{operation} child envelope gate"
    )
    publication_first, _ = _publication_monotonic_bounds(
        publication, f"{operation} child publication"
    )
    if (
        exit_status != 0
        or stderr_bytes != 0
        or execution["stderr_sha256"] != sha256_bytes(b"")
        or not (
            acceptance_emitted < launched
            and envelope_emitted < launched
            and launched < started
            and started < completed
            and completed < publication_first
        )
    ):
        fail(
            "CP0_CHILD_EVIDENCE_INVALID",
            f"{operation} child execution interval is invalid",
        )


def _validate_cp0_causal_order(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    stage_publications: tuple[Publication, ...],
    transcript_registry: ExternalTranscriptRegistry,
) -> None:
    by_path = {
        publication.evidence.relative_path: publication
        for publication in stage_publications
    }
    if len(by_path) != len(stage_publications):
        fail("CP0_CAUSAL_ORDER_INVALID", "CP0 causal stage contains duplicate paths")
    required = {
        "reservation.json",
        "reservation-acceptance.json",
        "owner-binding.json",
        "checkpoints",
        "manifests",
        "manifests/source-pre.jsonl",
        "manifests/paper-pre.json",
        "frozen/inventory.json",
    }
    if not required.issubset(by_path):
        fail("CP0_CAUSAL_ORDER_INVALID", "CP0 causal predecessor evidence is missing")
    reservation = by_path["reservation.json"]
    acceptance = by_path["reservation-acceptance.json"]
    _cp0_causal_precedes(
        reservation,
        acceptance,
        "reservation acceptance",
    )
    after_acceptance = {
        "owner-binding.json",
        "checkpoints",
        "manifests",
        "manifests/source-pre.jsonl",
        "manifests/paper-pre.json",
    } | {
        path
        for path in by_path
        if path == "frozen" or path.startswith("frozen/")
    }
    for path in sorted(after_acceptance):
        _cp0_causal_precedes(
            acceptance,
            by_path[path],
            path,
        )

    inventory, _ = _load_json(root_fd, "frozen/inventory.json")
    roles = inventory.get("roles")
    envelope_refs = (
        roles.get("envelope_evidence") if isinstance(roles, dict) else None
    )
    if (
        not isinstance(envelope_refs, list)
        or len(envelope_refs) != 1
        or not isinstance(envelope_refs[0], dict)
        or set(envelope_refs[0])
        != {"relative_path", "artifact_evidence_sha256"}
    ):
        fail(
            "CP0_CAUSAL_BINDING_INVALID",
            "CP0 envelope evidence reference is not exact",
        )
    envelope_ref = envelope_refs[0]
    envelope_path = envelope_ref["relative_path"]
    envelope = by_path.get(envelope_path) if isinstance(envelope_path, str) else None
    if (
        not isinstance(envelope, Publication)
        or envelope.evidence.artifact_type != "regular"
        or envelope_ref["artifact_evidence_sha256"]
        != sha256_bytes(canonical_json(envelope.evidence.to_dict()))
    ):
        fail(
            "CP0_CAUSAL_BINDING_INVALID",
            "CP0 envelope evidence reference does not resolve",
        )
    for path in (
        "manifests",
        "manifests/source-pre.jsonl",
        "manifests/paper-pre.json",
    ):
        _cp0_causal_precedes(envelope, by_path[path], path)

    envelope_payload, _ = _load_json(root_fd, envelope.evidence.relative_path)
    if set(envelope_payload) != {
        "schema",
        "sealed_run_id",
        "run_root",
        "acceptance_transcript_sha256",
    }:
        fail(
            "CP0_CAUSAL_BINDING_INVALID",
            "CP0 envelope evidence schema is not exact",
        )
    acceptance_transcript_sha256 = require_sha256(
        envelope_payload["acceptance_transcript_sha256"],
        "envelope acceptance transcript sha256",
    )
    if (
        envelope_payload["schema"] != CP0_ENVELOPE_EVIDENCE_SCHEMA
        or envelope_payload["sealed_run_id"] != sealed_run_id
        or envelope_payload["run_root"] != run_root
        or acceptance_transcript_sha256
        != acceptance.evidence.publication_transcript_sha256
        or envelope.transcript.get("context")
        != acceptance.transcript.get("context")
    ):
        fail(
            "CP0_CAUSAL_BINDING_INVALID",
            "CP0 envelope does not bind acceptance transcript and context",
        )
    for path, operation in (
        ("manifests/source-pre.jsonl", "source-pre"),
        ("manifests/paper-pre.json", "paper-pre"),
    ):
        context = _validate_cp0_protected_read_attestation(
            acceptance,
            envelope,
            by_path[path],
            operation,
            sealed_run_id,
            run_root,
        )
        _validate_cp0_child_evidence(
            root_fd,
            acceptance,
            envelope,
            by_path[path],
            operation,
            sealed_run_id,
            run_root,
            by_path,
            transcript_registry,
            context,
        )


def _validate_stage_chronology(
    stage_publications: tuple[Publication, ...],
    *,
    previous_checkpoint_publication: Publication | None,
    current_checkpoint_publication: Publication | None = None,
    upper_barrier_monotonic_ns: int | None = None,
) -> None:
    if not stage_publications:
        fail("CHECKPOINT_CHRONOLOGY_INVALID", "checkpoint stage is empty")
    lower: int | None = None
    if previous_checkpoint_publication is not None:
        _, lower = _publication_monotonic_bounds(
            previous_checkpoint_publication,
            "previous checkpoint",
        )
    upper: int | None = None
    if current_checkpoint_publication is not None:
        upper, _ = _publication_monotonic_bounds(
            current_checkpoint_publication,
            "current checkpoint",
        )
    if upper_barrier_monotonic_ns is not None:
        if type(upper_barrier_monotonic_ns) is not int or upper_barrier_monotonic_ns < 0:
            fail("CHECKPOINT_CHRONOLOGY_INVALID", "checkpoint barrier is invalid")
        if upper is not None and upper < upper_barrier_monotonic_ns:
            fail(
                "CHECKPOINT_CHRONOLOGY_INVALID",
                "checkpoint publication predates its prepublication barrier",
            )
        upper = upper_barrier_monotonic_ns if upper is None else upper
    for publication in stage_publications:
        first_completed, emitted = _publication_monotonic_bounds(
            publication,
            publication.evidence.relative_path,
        )
        if lower is not None and first_completed <= lower:
            fail(
                "CHECKPOINT_CHRONOLOGY_INVALID",
                "stage publication does not follow its predecessor checkpoint",
            )
        if upper is not None and emitted >= upper:
            fail(
                "CHECKPOINT_CHRONOLOGY_INVALID",
                "stage publication does not precede its checkpoint",
            )


def _registered_stage_publication(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    evidence: ArtifactEvidence,
    policy: WriterPolicy,
    transcript_registry: ExternalTranscriptRegistry,
) -> Publication:
    transcript = transcript_registry.get_exact(
        evidence.publication_transcript_sha256
    )
    if not isinstance(transcript, dict):
        fail("TRANSCRIPT_MISSING", "stage publication transcript is not an object")
    publication = Publication(evidence, transcript)
    _validate_publication(
        root_fd,
        sealed_run_id,
        run_root,
        publication,
        policy,
        transcript_registry,
    )
    return publication


def _registered_checkpoint_publication(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    checkpoint: int,
    transcript_sha256: object,
    policy: WriterPolicy,
    transcript_registry: ExternalTranscriptRegistry,
) -> Publication:
    digest = require_sha256(
        transcript_sha256,
        f"CP{checkpoint} publication transcript sha256",
    )
    transcript = transcript_registry.get_exact(digest)
    if not isinstance(transcript, dict):
        fail(
            "PREDECESSOR_TRANSCRIPT_MISSING",
            f"CP{checkpoint} publication transcript is not an object",
        )
    path = CHECKPOINT_PATHS[checkpoint]
    validated = validate_publication_transcript(
        root_fd,
        sealed_run_id,
        run_root,
        transcript,
        expected_relative_path=path,
    )
    if transcript_hash(validated) != digest:
        fail("CHECKPOINT_CHAIN_MISMATCH", "checkpoint transcript digest differs")
    writer = WriterIdentity.from_dict(validated["writer"])
    publication = Publication(
        ArtifactEvidence(
            path,
            str(validated["artifact_type"]),
            validated["sha256"],
            validated["bytes"],
            dict(validated["identity"]),
            writer,
            policy.authorize(path, writer).rule_id,
            digest,
        ),
        validated,
    )
    _validate_publication(
        root_fd,
        sealed_run_id,
        run_root,
        publication,
        policy,
        transcript_registry,
    )
    return publication


def _semantic_context_for_publish(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    checkpoint: int,
    current_stage: tuple[Publication, ...],
    previous_checkpoint_publication: Publication,
    policy: WriterPolicy,
    transcript_registry: ExternalTranscriptRegistry,
) -> _SemanticReplayContext:
    payloads: dict[int, dict[str, object]] = {}
    stages: dict[int, tuple[Publication, ...]] = {checkpoint: current_stage}
    for index in range(checkpoint):
        payload, _ = _load_json(root_fd, CHECKPOINT_PATHS[index])
        validate_checkpoint_payload(
            payload,
            expected_sealed_run_id=sealed_run_id,
            expected_run_root=run_root,
            expected_checkpoint=index,
            policy=policy,
        )
        payloads[index] = payload
        stages[index] = tuple(
            _registered_stage_publication(
                root_fd,
                sealed_run_id,
                run_root,
                ArtifactEvidence.from_dict(item),
                policy,
                transcript_registry,
            )
            for item in payload["stage_actual_paths"]
        )
    checkpoints: dict[int, Publication] = {}
    for index in range(checkpoint):
        if index == checkpoint - 1:
            checkpoints[index] = previous_checkpoint_publication
            continue
        checkpoints[index] = _registered_checkpoint_publication(
            root_fd,
            sealed_run_id,
            run_root,
            index,
            payloads[index + 1]["previous_checkpoint_transcript_sha256"],
            policy,
            transcript_registry,
        )
    return _SemanticReplayContext(checkpoints, stages)


def _stage_publication(
    context: _SemanticReplayContext,
    checkpoint: int,
    relative_path: str,
) -> Publication:
    matches = tuple(
        publication
        for publication in context.stage_publications.get(checkpoint, ())
        if publication.evidence.relative_path == relative_path
    )
    if len(matches) != 1:
        fail(
            "SEMANTICS_CONTEXT_INVALID",
            f"CP{checkpoint} must contain exactly one {relative_path} Publication",
        )
    return matches[0]


def _cp3_semantic_inputs(context: _SemanticReplayContext):
    from .cp45_semantic import CP3SemanticPublications

    cp3_stage = context.stage_publications.get(3, ())
    buckets = tuple(
        sorted(
            (
                publication
                for publication in cp3_stage
                if publication.evidence.artifact_type == "regular"
                and publication.evidence.relative_path.startswith(
                    "admission/precopy/buckets/"
                )
            ),
            key=lambda publication: publication.evidence.relative_path,
        )
    )
    return CP3SemanticPublications(
        checkpoint=context.checkpoint_publications[3],
        cp1_checkpoint=context.checkpoint_publications[1],
        cp2_checkpoint=context.checkpoint_publications[2],
        source_pre=_stage_publication(
            context, 0, "manifests/source-pre.jsonl"
        ),
        paper_pre=_stage_publication(context, 0, "manifests/paper-pre.json"),
        inventory=_stage_publication(context, 1, "inventory/results.jsonl"),
        raw_nodes=_stage_publication(context, 3, "provenance/nodes.jsonl"),
        provenance_edges=_stage_publication(
            context, 3, "provenance/edges.jsonl"
        ),
        provenance_recomputations=_stage_publication(
            context, 3, "provenance/recomputations.jsonl"
        ),
        provenance_rounding_proofs=_stage_publication(
            context, 3, "provenance/rounding-proofs.jsonl"
        ),
        reconciliation=_stage_publication(
            context, 3, "admission/precopy/reconciliation.json"
        ),
        buckets=buckets,
    )


def _replay_checkpoint_semantics(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    checkpoint: int,
    publications: tuple[Publication, ...],
    previous_checkpoint_sha256: str | None,
    previous_checkpoint_transcript_sha256: str | None,
    transcript_registry: ExternalTranscriptRegistry,
    context: _SemanticReplayContext | None,
) -> dict[str, object]:
    if checkpoint in {0, 1, 2}:
        return validate_stage_semantics(
            root_fd,
            sealed_run_id,
            run_root,
            checkpoint,
            publications,
            previous_checkpoint_sha256,
            previous_checkpoint_transcript_sha256,
        )
    if context is None:
        fail("SEMANTICS_CONTEXT_INVALID", "typed semantic replay context is missing")
    if checkpoint == 3:
        from .cp3_semantic import validate_cp3_semantics

        _require_canonical_checkpoint_chain(
            root_fd,
            sealed_run_id,
            run_root,
            2,
            transcript_registry,
            context,
        )
        return validate_cp3_semantics(
            root_fd,
            sealed_run_id,
            run_root,
            publications,
            context.checkpoint_publications[1],
            context.checkpoint_publications[2],
            _stage_publication(context, 0, "manifests/source-pre.jsonl"),
            transcript_registry,
            require_sha256(
                previous_checkpoint_sha256, "previous checkpoint sha256"
            ),
            require_sha256(
                previous_checkpoint_transcript_sha256,
                "previous checkpoint transcript sha256",
            ),
        )
    if checkpoint == 4:
        from .cp45_semantic import validate_cp4_semantics

        _require_canonical_checkpoint_chain(
            root_fd,
            sealed_run_id,
            run_root,
            3,
            transcript_registry,
            context,
        )
        _require_stored_semantic_replay(
            root_fd,
            sealed_run_id,
            run_root,
            3,
            transcript_registry,
            context,
        )
        return validate_cp4_semantics(
            root_fd,
            sealed_run_id,
            run_root,
            publications,
            _cp3_semantic_inputs(context),
            transcript_registry,
            require_sha256(
                previous_checkpoint_sha256, "previous checkpoint sha256"
            ),
            require_sha256(
                previous_checkpoint_transcript_sha256,
                "previous checkpoint transcript sha256",
            ),
        )
    if checkpoint == 5:
        from .cp45_semantic import CP4SemanticPublications, validate_cp5_semantics

        _require_canonical_checkpoint_chain(
            root_fd,
            sealed_run_id,
            run_root,
            4,
            transcript_registry,
            context,
        )
        _require_stored_semantic_replay(
            root_fd,
            sealed_run_id,
            run_root,
            3,
            transcript_registry,
            context,
        )
        return validate_cp5_semantics(
            root_fd,
            sealed_run_id,
            run_root,
            publications,
            _cp3_semantic_inputs(context),
            CP4SemanticPublications(
                checkpoint=context.checkpoint_publications[4],
                stage=context.stage_publications[4],
            ),
            transcript_registry,
            require_sha256(
                previous_checkpoint_sha256, "previous checkpoint sha256"
            ),
            require_sha256(
                previous_checkpoint_transcript_sha256,
                "previous checkpoint transcript sha256",
            ),
        )
    if checkpoint == 6:
        _require_canonical_checkpoint_chain(
            root_fd,
            sealed_run_id,
            run_root,
            5,
            transcript_registry,
            context,
        )
        prior_publications = tuple(
            publication
            for index in range(6)
            for publication in context.stage_publications[index]
        ) + tuple(context.checkpoint_publications[index] for index in range(6))
        return validate_stage_semantics(
            root_fd,
            sealed_run_id,
            run_root,
            checkpoint,
            publications,
            previous_checkpoint_sha256,
            previous_checkpoint_transcript_sha256,
            prior_publications=prior_publications,
        )
    fail("SEMANTICS_NOT_IMPLEMENTED", "requested semantic checkpoint is not implemented")


def _require_stored_semantic_replay(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    checkpoint: int,
    transcript_registry: ExternalTranscriptRegistry,
    context: _SemanticReplayContext,
) -> None:
    payload, _ = _load_json(root_fd, CHECKPOINT_PATHS[checkpoint])
    replay = _replay_checkpoint_semantics(
        root_fd,
        sealed_run_id,
        run_root,
        checkpoint,
        context.stage_publications[checkpoint],
        payload["previous_checkpoint_sha256"],
        payload["previous_checkpoint_transcript_sha256"],
        transcript_registry,
        context,
    )
    if (
        payload["semantic_bindings"] != replay
        or payload["semantic_bindings_sha256"]
        != sha256_bytes(canonical_json(replay))
    ):
        fail(
            "CHECKPOINT_SEMANTICS_INVALID",
            f"stored CP{checkpoint} semantic replay differs",
        )


def _strict_parent_from_run_root(sealed_run_id: str, run_root: str) -> str:
    suffix = f"/{sealed_run_id}"
    if not run_root.endswith(suffix):
        fail("CHECKPOINT_CHAIN_MISMATCH", "run root is not the sealed strict child")
    return validate_absolute_path_text(run_root[: -len(suffix)], "strict_parent")


def _recompute_cp0_bindings(
    root_fd: int,
    sealed_run_id: str,
    strict_parent: str,
    run_root: str,
    stage_publications: tuple[Publication, ...],
    policy: WriterPolicy,
) -> dict[str, object]:
    by_path = {
        publication.evidence.relative_path: publication
        for publication in stage_publications
    }
    if len(by_path) != len(stage_publications):
        fail("CP0_BINDING_MISMATCH", "CP0 stage contains duplicate paths")
    required = {
        "reservation.json",
        "reservation-acceptance.json",
        "owner-binding.json",
        "frozen/writer-policy.json",
    }
    if not required.issubset(by_path):
        fail("CP0_BINDING_MISSING", "CP0 core reservation/policy artifacts are missing")

    reservation, reservation_raw = _load_json(root_fd, "reservation.json")
    acceptance, acceptance_raw = _load_json(
        root_fd, "reservation-acceptance.json"
    )
    owner, owner_raw = _load_json(root_fd, "owner-binding.json")
    validate_reservation_payload(
        reservation, sealed_run_id, strict_parent, run_root
    )
    validate_acceptance_payload(acceptance, sealed_run_id, run_root)
    validate_owner_binding_payload(owner, sealed_run_id, run_root)

    reservation_publication = by_path["reservation.json"]
    acceptance_publication = by_path["reservation-acceptance.json"]
    owner_publication = by_path["owner-binding.json"]
    reservation_writer = reservation_publication.evidence.writer.to_dict()
    acceptance_writer = acceptance_publication.evidence.writer.to_dict()
    owner_writer = owner_publication.evidence.writer.to_dict()
    expected_context = {
        "parent_identity": reservation["parent"]["descriptor"]["identity"],
        "child_identity": reservation["child"]["descriptor"]["identity"],
        "parent_mount": reservation["parent"]["mount"],
        "child_mount": reservation["child"]["mount"],
    }
    if (
        acceptance["reservation_sha256"] != sha256_bytes(reservation_raw)
        or acceptance["reservation_transcript_sha256"]
        != reservation_publication.evidence.publication_transcript_sha256
        or acceptance["reservation_writer"] != reservation_writer
        or acceptance["acceptance_writer"] != acceptance_writer
        or acceptance["validated_identity"]["parent"]
        != expected_context["parent_identity"]
        or acceptance["validated_identity"]["child"]
        != expected_context["child_identity"]
        or acceptance["validated_identity"]["reservation"]
        != reservation_publication.evidence.identity
        or reservation["producer"]["writer"] != reservation_writer
        or reservation_publication.transcript["context"] != expected_context
        or acceptance_publication.transcript["context"] != expected_context
        or owner["reservation_sha256"] != sha256_bytes(reservation_raw)
        or owner["reservation_acceptance_sha256"]
        != sha256_bytes(acceptance_raw)
        or owner["writer"] != owner_writer
        or {
            "project_id": owner["project_id"],
            "owner_seed": owner["owner_seed"],
        }
        != reservation["owner"]
    ):
        fail("CP0_BINDING_MISMATCH", "reservation/acceptance/owner bindings differ")

    policy_value, policy_raw = _load_json(root_fd, "frozen/writer-policy.json")
    stored_policy = WriterPolicy.from_dict(policy_value)
    if canonical_json(stored_policy.to_dict()) != canonical_json(policy.to_dict()):
        fail("WRITER_POLICY_MISMATCH", "stored writer policy differs from controller")
    return {
        "reservation_sha256": sha256_bytes(reservation_raw),
        "reservation_acceptance_sha256": sha256_bytes(acceptance_raw),
        "owner_binding_sha256": sha256_bytes(owner_raw),
        "reservation_transcript_sha256": (
            reservation_publication.evidence.publication_transcript_sha256
        ),
        "acceptance_transcript_sha256": (
            acceptance_publication.evidence.publication_transcript_sha256
        ),
        "writer_policy_sha256": sha256_bytes(policy_raw),
    }


def _require_canonical_checkpoint_chain(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    through_checkpoint: int,
    transcript_registry: ExternalTranscriptRegistry,
    context: _SemanticReplayContext,
) -> None:
    if type(through_checkpoint) is not int or through_checkpoint not in range(7):
        fail("CHECKPOINT_CHAIN_MISMATCH", "checkpoint chain endpoint is invalid")
    try:
        policy_value, _ = _load_json(root_fd, "frozen/writer-policy.json")
    except OSError:
        fail(
            "CHECKPOINT_CHAIN_MISMATCH",
            "frozen writer policy is missing from the canonical checkpoint chain",
        )
    policy = WriterPolicy.from_dict(policy_value)
    strict_parent = _strict_parent_from_run_root(sealed_run_id, run_root)
    controller: WriterIdentity | None = None
    previous: Publication | None = None
    cp0_payload: dict[str, object] | None = None
    for index in range(through_checkpoint + 1):
        payload, raw = _load_json(root_fd, CHECKPOINT_PATHS[index])
        validate_checkpoint_payload(
            payload,
            expected_sealed_run_id=sealed_run_id,
            expected_run_root=run_root,
            expected_checkpoint=index,
            policy=policy,
        )
        publication = context.checkpoint_publications.get(index)
        if not isinstance(publication, Publication):
            fail(
                "CHECKPOINT_CHAIN_MISMATCH",
                f"typed CP{index} Publication is missing",
            )
        _validate_publication(
            root_fd,
            sealed_run_id,
            run_root,
            publication,
            policy,
            transcript_registry,
        )
        evidence = publication.evidence
        if (
            evidence.relative_path != CHECKPOINT_PATHS[index]
            or evidence.artifact_type != "regular"
            or evidence.sha256 != sha256_bytes(raw)
            or evidence.writer.role != "checkpoint_controller"
        ):
            fail(
                "CHECKPOINT_CHAIN_MISMATCH",
                f"CP{index} Publication differs from current checkpoint bytes",
            )
        if controller is None:
            controller = evidence.writer
        elif evidence.writer != controller:
            fail(
                "CONTROLLER_WRITER_MISMATCH",
                "checkpoint controller identity changed within the chain",
            )
        if previous is None:
            if (
                payload["previous_checkpoint_sha256"] is not None
                or payload["previous_checkpoint_transcript_sha256"] is not None
            ):
                fail("CHECKPOINT_CHAIN_MISMATCH", "CP0 names a predecessor")
        elif (
            payload["previous_checkpoint_sha256"] != previous.evidence.sha256
            or payload["previous_checkpoint_transcript_sha256"]
            != previous.evidence.publication_transcript_sha256
        ):
            fail(
                "CHECKPOINT_CHAIN_MISMATCH",
                f"CP{index} does not bind the current CP{index - 1} Publication",
            )
        stage_publications = context.stage_publications.get(index)
        if not isinstance(stage_publications, tuple) or not stage_publications:
            fail(
                "CHECKPOINT_CHAIN_MISMATCH",
                f"typed CP{index} stage Publications are missing",
            )
        for stage_publication in stage_publications:
            _validate_publication(
                root_fd,
                sealed_run_id,
                run_root,
                stage_publication,
                policy,
                transcript_registry,
            )
        current_stage = [
            publication.evidence.to_dict()
            for publication in sorted(
                stage_publications,
                key=lambda item: item.evidence.relative_path,
            )
        ]
        if current_stage != payload["stage_actual_paths"]:
            fail(
                "CHECKPOINT_CHAIN_MISMATCH",
                f"typed CP{index} stage differs from current checkpoint bytes",
            )
        _validate_stage_chronology(
            stage_publications,
            previous_checkpoint_publication=previous,
            current_checkpoint_publication=publication,
        )
        if index == 0:
            cp0_payload = payload
        previous = publication
    assert cp0_payload is not None
    recomputed_cp0_bindings = _recompute_cp0_bindings(
        root_fd,
        sealed_run_id,
        strict_parent,
        run_root,
        context.stage_publications[0],
        policy,
    )
    if cp0_payload["bindings"] != recomputed_cp0_bindings:
        fail(
            "CP0_BINDING_MISMATCH",
            "stored CP0 bindings differ from current reservation/policy evidence",
        )
    _validate_cp0_causal_order(
        root_fd,
        sealed_run_id,
        run_root,
        context.stage_publications[0],
        transcript_registry,
    )
    _require_stored_semantic_replay(
        root_fd,
        sealed_run_id,
        run_root,
        0,
        transcript_registry,
        context,
    )


def _prior_stage_paths(
    root_fd: int,
    count: int,
    policy: WriterPolicy,
    sealed_run_id: str,
    run_root: str,
) -> set[str]:
    paths: set[str] = set()
    for checkpoint in range(count):
        payload, _ = _load_json(root_fd, CHECKPOINT_PATHS[checkpoint])
        validate_checkpoint_payload(
            payload,
            expected_sealed_run_id=sealed_run_id,
            expected_run_root=run_root,
            expected_checkpoint=checkpoint,
            policy=policy,
        )
        for item in payload["stage_actual_paths"]:
            relative = ArtifactEvidence.from_dict(item).relative_path
            if relative in paths:
                fail("CHECKPOINT_STAGE_OVERLAP", "path appears in more than one stage")
            paths.add(relative)
    return paths


def _current_noncheckpoint_paths(root_fd: int) -> set[str]:
    result: set[str] = set()
    for relative, st in iter_tree(root_fd):
        assert_regular_or_directory(st)
        if relative not in CHECKPOINT_PATHS:
            result.add(relative)
    return result


def prepare_checkpoint_directory(
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    controller: WriterIdentity,
    policy: WriterPolicy,
    *,
    run_binding: RunDescriptorBinding | None = None,
) -> Publication:
    """Exclusively publish the controller-owned CP0 checkpoint directory."""

    sealed_run_id = validate_sealed_run_id(sealed_run_id)
    if controller.role != "checkpoint_controller":
        fail(
            "CONTROLLER_WRITER_INVALID",
            "checkpoint directory requires controller role",
        )
    rule = policy.authorize("checkpoints", controller)
    published = StagePublisher(
        strict_parent,
        sealed_run_id,
        run_root,
        controller,
        policy,
        allow_reserved_paths=True,
        run_binding=run_binding,
    ).ensure_directory("checkpoints")
    if not published:
        fail(
            "CHECKPOINT_DIRECTORY_EXISTS",
            "checkpoint directory already exists and cannot be replaced",
        )
    if len(published) != 1:
        fail(
            "CHECKPOINT_PUBLICATION_INVALID",
            "checkpoint directory created undeclared paths",
        )
    publication = published[0]
    evidence = publication.evidence
    if (
        evidence.relative_path != "checkpoints"
        or evidence.artifact_type != "directory"
        or evidence.sha256 is not None
        or evidence.size is not None
        or evidence.writer != controller
        or evidence.writer_rule_id != rule.rule_id
    ):
        fail(
            "CHECKPOINT_PUBLICATION_INVALID",
            "checkpoint directory publication evidence differs",
        )
    return publication


def publish_checkpoint(
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    checkpoint: int,
    stage_publications: Iterable[Publication],
    controller: WriterIdentity,
    policy: WriterPolicy,
    *,
    transcript_registry: ExternalTranscriptRegistry,
    previous_checkpoint_publication: Publication | None = None,
    run_binding: RunDescriptorBinding | None = None,
) -> CheckpointResult:
    sealed_run_id = validate_sealed_run_id(sealed_run_id)
    if type(checkpoint) is not int or checkpoint not in range(7):
        fail("CHECKPOINT_SEQUENCE_INVALID", "checkpoint index is invalid")
    if controller.role != "checkpoint_controller":
        fail("CONTROLLER_WRITER_INVALID", "checkpoint requires controller role")
    if not isinstance(transcript_registry, ExternalTranscriptRegistry):
        fail("EXTERNAL_REGISTRY_REQUIRED", "checkpoint requires a typed transcript registry")
    if run_binding is not None and not isinstance(run_binding, RunDescriptorBinding):
        fail("RUN_BINDING_INVALID", "checkpoint run binding is not typed")
    transcript_registry.verify_current()
    policy.authorize(CHECKPOINT_PATHS[checkpoint], controller)
    publications = list(stage_publications)
    first_handle = (
        open_run_handle(strict_parent, sealed_run_id, run_root)
        if run_binding is None
        else open_bound_run_handle(
            strict_parent, sealed_run_id, run_root, run_binding
        )
    )
    with first_handle as handle:
        if exists_relative(handle.root_fd, "terminal/blocked.json"):
            fail("RUN_TERMINAL_BLOCKED", "blocked run cannot publish a checkpoint")
        if any(
            exists_relative(handle.root_fd, CHECKPOINT_PATHS[index])
            for index in range(checkpoint, 7)
        ):
            fail("CHECKPOINT_EXISTS_OR_FUTURE", "current/future checkpoint path already exists")
        for index in range(checkpoint):
            if not exists_relative(handle.root_fd, CHECKPOINT_PATHS[index]):
                fail("CHECKPOINT_SEQUENCE_INVALID", "predecessor checkpoint is missing")
        if checkpoint == 0:
            if previous_checkpoint_publication is not None:
                fail("CHECKPOINT_SEQUENCE_INVALID", "CP0 cannot accept a prior checkpoint")
        else:
            previous_payload, previous_raw = _load_json(
                handle.root_fd, CHECKPOINT_PATHS[checkpoint - 1]
            )
            validate_checkpoint_payload(
                previous_payload,
                expected_sealed_run_id=sealed_run_id,
                expected_run_root=run_root,
                expected_checkpoint=checkpoint - 1,
                policy=policy,
            )
            if not isinstance(previous_checkpoint_publication, Publication):
                fail("PREDECESSOR_TRANSCRIPT_MISSING", "typed predecessor publication is required")
            _validate_publication(
                handle.root_fd,
                sealed_run_id,
                run_root,
                previous_checkpoint_publication,
                policy,
                transcript_registry,
            )
            previous_evidence = previous_checkpoint_publication.evidence
            if (
                previous_evidence.relative_path != CHECKPOINT_PATHS[checkpoint - 1]
                or previous_evidence.writer != controller
                or previous_evidence.sha256 != sha256_bytes(previous_raw)
            ):
                fail("CHECKPOINT_CHAIN_MISMATCH", "predecessor publication differs")
            previous_sha = sha256_bytes(previous_raw)
            previous_transcript_sha = previous_evidence.publication_transcript_sha256
    if checkpoint == 0:
        previous_sha = None
        previous_transcript_sha = None
    validation_handle = (
        open_run_handle(strict_parent, sealed_run_id, run_root)
        if run_binding is None
        else open_bound_run_handle(
            strict_parent, sealed_run_id, run_root, run_binding
        )
    )
    with validation_handle as handle:
        for publication in publications:
            _validate_publication(
                handle.root_fd,
                sealed_run_id,
                run_root,
                publication,
                policy,
                transcript_registry,
            )
        evidence = [publication.evidence for publication in publications]
        if len({item.relative_path for item in evidence}) != len(evidence):
            fail("CHECKPOINT_STAGE_OVERLAP", "current stage contains duplicate paths")
        prior = _prior_stage_paths(
            handle.root_fd,
            checkpoint,
            policy,
            sealed_run_id,
            run_root,
        )
        if prior.intersection(item.relative_path for item in evidence):
            fail("CHECKPOINT_STAGE_OVERLAP", "current stage overlaps a prior stage")
        for item in evidence:
            if not _stage_allows(checkpoint, item.relative_path):
                fail("CHECKPOINT_SELF_OR_FUTURE", "artifact belongs to another stage")
        expected_now = prior.union(item.relative_path for item in evidence)
        if _current_noncheckpoint_paths(handle.root_fd) != expected_now:
            fail("CHECKPOINT_TREE_GAP", "current tree differs from declared stage union")
        if checkpoint == 0:
            bindings: dict[str, object] = _recompute_cp0_bindings(
                handle.root_fd,
                sealed_run_id,
                strict_parent,
                run_root,
                tuple(publications),
                policy,
            )
        else:
            bindings = {}
        semantic_context = None
        if checkpoint >= 3:
            if not isinstance(previous_checkpoint_publication, Publication):
                fail(
                    "SEMANTICS_CONTEXT_INVALID",
                    "CP3 through CP6 require a typed predecessor Publication",
                )
            semantic_context = _semantic_context_for_publish(
                handle.root_fd,
                sealed_run_id,
                run_root,
                checkpoint,
                tuple(publications),
                previous_checkpoint_publication,
                policy,
                transcript_registry,
            )
            if checkpoint == 6:
                _require_stored_semantic_replay(
                    handle.root_fd,
                    sealed_run_id,
                    run_root,
                    5,
                    transcript_registry,
                    semantic_context,
                )
        semantic_bindings = _replay_checkpoint_semantics(
            handle.root_fd,
            sealed_run_id,
            run_root,
            checkpoint,
            tuple(publications),
            previous_sha,
            previous_transcript_sha,
            transcript_registry,
            semantic_context,
        )
        if checkpoint == 0:
            _validate_cp0_causal_order(
                handle.root_fd,
                sealed_run_id,
                run_root,
                tuple(publications),
                transcript_registry,
            )
        checkpoint_barrier_monotonic_ns = time.monotonic_ns()
        _validate_stage_chronology(
            tuple(publications),
            previous_checkpoint_publication=previous_checkpoint_publication,
            upper_barrier_monotonic_ns=checkpoint_barrier_monotonic_ns,
        )
        rows = [item.to_dict() for item in sorted(evidence, key=lambda item: item.relative_path)]
        payload: dict[str, object] = {
            "schema": CHECKPOINT_SCHEMA,
            "sealed_run_id": sealed_run_id,
            "run_root": run_root,
            "checkpoint": checkpoint,
            "previous_checkpoint_sha256": previous_sha,
            "previous_checkpoint_transcript_sha256": previous_transcript_sha,
            "stage_actual_paths": rows,
            "stage_actual_paths_sha256": sha256_bytes(canonical_json(rows)),
            "bindings": bindings,
            "semantic_bindings": semantic_bindings,
            "semantic_bindings_sha256": sha256_bytes(canonical_json(semantic_bindings)),
        }
        validate_checkpoint_payload(
            payload,
            expected_sealed_run_id=sealed_run_id,
            expected_run_root=run_root,
            expected_checkpoint=checkpoint,
            policy=policy,
        )
        transcript_registry.verify_current()
    published = StagePublisher(
        strict_parent,
        sealed_run_id,
        run_root,
        controller,
        policy,
        allow_reserved_paths=True,
        run_binding=run_binding,
    ).publish_json(CHECKPOINT_PATHS[checkpoint], payload)
    if len(published) != 1:
        fail("CHECKPOINT_PUBLICATION_INVALID", "checkpoint created undeclared paths")
    _validate_stage_chronology(
        tuple(publications),
        previous_checkpoint_publication=previous_checkpoint_publication,
        current_checkpoint_publication=published[0],
        upper_barrier_monotonic_ns=checkpoint_barrier_monotonic_ns,
    )
    return CheckpointResult(checkpoint, payload, published[0], tuple(publications))


def publish_blocked(
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    reason_codes: Iterable[str],
    controller: WriterIdentity,
    policy: WriterPolicy,
) -> tuple[Publication, ...]:
    sealed_run_id = validate_sealed_run_id(sealed_run_id)
    reasons = sorted(set(reason_codes))
    if not reasons or not all(isinstance(reason, str) and reason for reason in reasons):
        fail("BLOCKED_REASON_INVALID", "blocked terminal needs nonempty reason codes")
    if controller.role != "checkpoint_controller":
        fail("CONTROLLER_WRITER_INVALID", "blocked terminal requires controller role")
    with open_run_handle(strict_parent, sealed_run_id, run_root) as handle:
        if exists_relative(handle.root_fd, CHECKPOINT_PATHS[6]):
            fail("RUN_SUCCESS_TERMINAL", "successful CP6 run cannot become blocked")
        if exists_relative(handle.root_fd, "terminal/blocked.json"):
            fail("RUN_TERMINAL_BLOCKED", "blocked terminal is no-replace")
        last_hash: str | None = None
        for path in CHECKPOINT_PATHS:
            if not exists_relative(handle.root_fd, path):
                break
            _, raw = _load_json(handle.root_fd, path)
            last_hash = sha256_bytes(raw)
    payload = {
        "schema": BLOCKED_SCHEMA,
        "sealed_run_id": sealed_run_id,
        "run_root": run_root,
        "state": "BLOCKED",
        "reason_codes": reasons,
        "last_checkpoint_sha256": last_hash,
        "writer": controller.to_dict(),
    }
    return tuple(
        StagePublisher(
            strict_parent,
            sealed_run_id,
            run_root,
            controller,
            policy,
            allow_reserved_paths=True,
        ).publish_json("terminal/blocked.json", payload)
    )


def validate_final_tree(
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    transcript_registry: ExternalTranscriptRegistry,
    cp6_publication: Publication,
) -> dict[str, object]:
    """Read-only final validation.  Its report is returned out of tree."""

    sealed_run_id = validate_sealed_run_id(sealed_run_id)
    if not isinstance(transcript_registry, ExternalTranscriptRegistry):
        fail("EXTERNAL_REGISTRY_REQUIRED", "final validation requires a typed registry")
    if not isinstance(cp6_publication, Publication):
        fail("CP6_PUBLICATION_REQUIRED", "final validation requires typed CP6 publication")
    transcript_registry.verify_current()
    with open_run_handle(strict_parent, sealed_run_id, run_root) as handle:
        if exists_relative(handle.root_fd, "terminal/blocked.json"):
            fail("RUN_TERMINAL_BLOCKED", "blocked terminal cannot be success")
        policy_value, _ = _load_json(handle.root_fd, "frozen/writer-policy.json")
        policy = WriterPolicy.from_dict(policy_value)
        payloads: list[dict[str, object]] = []
        raws: list[bytes] = []
        checkpoint_writers: list[WriterIdentity] = []
        checkpoint_publications: list[Publication] = []
        for index, path in enumerate(CHECKPOINT_PATHS):
            payload, raw = _load_json(handle.root_fd, path)
            validate_checkpoint_payload(
                payload,
                expected_sealed_run_id=sealed_run_id,
                expected_run_root=run_root,
                expected_checkpoint=index,
                policy=policy,
            )
            payloads.append(payload)
            raws.append(raw)
        for index in range(1, 7):
            if payloads[index]["previous_checkpoint_sha256"] != sha256_bytes(raws[index - 1]):
                fail("CHECKPOINT_CHAIN_MISMATCH", "previous checkpoint hash differs")
            expected_transcript_hash = payloads[index][
                "previous_checkpoint_transcript_sha256"
            ]
            transcript = transcript_registry.get_exact(str(expected_transcript_hash))
            if not isinstance(transcript, dict):
                fail("PREDECESSOR_TRANSCRIPT_MISSING", "checkpoint transcript is not an object")
            validated = validate_publication_transcript(
                handle.root_fd,
                sealed_run_id,
                run_root,
                transcript,
                expected_relative_path=CHECKPOINT_PATHS[index - 1],
            )
            if transcript_hash(validated) != expected_transcript_hash:
                fail("CHECKPOINT_CHAIN_MISMATCH", "checkpoint transcript digest differs")
            writer = WriterIdentity.from_dict(validated["writer"])
            checkpoint_writers.append(writer)
            checkpoint_publications.append(
                Publication(
                    ArtifactEvidence(
                        CHECKPOINT_PATHS[index - 1],
                        str(validated["artifact_type"]),
                        validated["sha256"],
                        validated["bytes"],
                        dict(validated["identity"]),
                        writer,
                        policy.authorize(CHECKPOINT_PATHS[index - 1], writer).rule_id,
                        str(expected_transcript_hash),
                    ),
                    validated,
                )
            )
        _validate_publication(
            handle.root_fd,
            sealed_run_id,
            run_root,
            cp6_publication,
            policy,
            transcript_registry,
        )
        if cp6_publication.evidence.relative_path != CHECKPOINT_PATHS[6]:
            fail("CP6_PUBLICATION_REQUIRED", "typed CP6 publication names the wrong path")
        checkpoint_publications.append(cp6_publication)
        checkpoint_writers.append(cp6_publication.evidence.writer)
        if len(set(checkpoint_writers)) != 1:
            fail("CONTROLLER_WRITER_MISMATCH", "checkpoint controller identity changed")
        expected_paths: set[str] = set(CHECKPOINT_PATHS)
        artifact_count = 0
        stage_publications: list[list[Publication]] = []
        for payload in payloads:
            current_stage: list[Publication] = []
            for raw_evidence in payload["stage_actual_paths"]:
                evidence = ArtifactEvidence.from_dict(raw_evidence)
                if evidence.relative_path in expected_paths:
                    fail("CHECKPOINT_STAGE_OVERLAP", "stage path is duplicate/schema-implied")
                expected_paths.add(evidence.relative_path)
                artifact_count += 1
                transcript = transcript_registry.get_exact(
                    evidence.publication_transcript_sha256
                )
                if not isinstance(transcript, dict):
                    fail("TRANSCRIPT_MISSING", "stage publication transcript is not an object")
                validated = validate_publication_transcript(
                    handle.root_fd,
                    sealed_run_id,
                    run_root,
                    transcript,
                    expected_relative_path=evidence.relative_path,
                    expected_writer=evidence.writer,
                )
                if transcript_hash(validated) != evidence.publication_transcript_sha256:
                    fail("TRANSCRIPT_MISMATCH", "stage transcript digest differs")
                rule = policy.authorize(evidence.relative_path, evidence.writer)
                if rule.rule_id != evidence.writer_rule_id:
                    fail("WRITER_MISMATCH", "stage evidence rule differs")
                if (
                    validated["identity"] != evidence.identity
                    or validated["sha256"] != evidence.sha256
                    or validated["bytes"] != evidence.size
                ):
                    fail("EVIDENCE_MISMATCH", "stage evidence changed")
                current_stage.append(Publication(evidence, validated))
            stage_publications.append(current_stage)
        semantic_context = _SemanticReplayContext(
            {
                index: publication
                for index, publication in enumerate(checkpoint_publications)
            },
            {
                index: tuple(publications)
                for index, publications in enumerate(stage_publications)
            },
        )
        for index in range(7):
            _validate_stage_chronology(
                tuple(stage_publications[index]),
                previous_checkpoint_publication=(
                    None if index == 0 else checkpoint_publications[index - 1]
                ),
                current_checkpoint_publication=checkpoint_publications[index],
            )
            semantic_bindings = _replay_checkpoint_semantics(
                handle.root_fd,
                sealed_run_id,
                run_root,
                index,
                tuple(stage_publications[index]),
                payloads[index]["previous_checkpoint_sha256"],
                payloads[index]["previous_checkpoint_transcript_sha256"],
                transcript_registry,
                semantic_context,
            )
            if (
                payloads[index]["semantic_bindings"] != semantic_bindings
                or payloads[index]["semantic_bindings_sha256"]
                != sha256_bytes(canonical_json(semantic_bindings))
            ):
                fail("CHECKPOINT_SEMANTICS_INVALID", "stored semantic replay differs")
        actual: set[str] = set()
        for relative, st in iter_tree(handle.root_fd):
            assert_regular_or_directory(st)
            actual.add(relative)
        if actual != expected_paths:
            fail(
                "FINAL_TREE_CLOSURE",
                f"final tree differs; missing={sorted(expected_paths - actual)!r} "
                f"extra={sorted(actual - expected_paths)!r}",
            )
        for index, path in enumerate(CHECKPOINT_PATHS):
            policy.authorize(path, checkpoint_publications[index].evidence.writer)
        transcript_registry.verify_current()
        return {
            "schema": "experiments7-final-tree-validation/v6",
            "sealed_run_id": sealed_run_id,
            "run_root": run_root,
            "terminal": "CP6",
            "cp6_sha256": sha256_bytes(raws[6]),
            "actual_path_count": len(actual),
            "stage_artifact_count": artifact_count,
            "checkpoint_path_count": 7,
            "unowned_path_count": 0,
            "overlap_count": 0,
            "post_cp6_path_count": 0,
        }
