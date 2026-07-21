"""Strict child reservation, causal acceptance, and owner binding."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Callable

from .canonical import (
    canonical_json,
    fail,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
    validate_sealed_run_id,
)
from .filesystem import (
    DIR_FLAGS,
    RunDescriptorBinding,
    assert_path_matches_fd,
    descriptor_evidence,
    mount_identity,
    open_absolute_directory,
    open_bound_run_handle,
    open_run_handle,
    process_identity,
    read_regular_at,
    stable_identity,
    validate_run_root_text,
)
from .external import ExternalTranscriptRegistry
from .publication import (
    ArtifactEvidence,
    Publication,
    StagePublisher,
    transcript_hash,
    validate_publication_transcript,
)
from .writer_policy import WriterIdentity, WriterPolicy, default_writer_policy


RESERVATION_SCHEMA = "experiments7-strict-reservation/v6"
ACCEPTANCE_SCHEMA = "experiments7-reservation-acceptance/v6"
OWNER_SCHEMA = "experiments7-owner-binding/v6"


@dataclass(frozen=True)
class ReservationInputs:
    project_id: str
    owner_seed: str
    writer: WriterIdentity
    tool_sha256: str
    configuration_sha256: str
    runtime_sha256: str
    environment_path_policy_sha256: str
    canonical_argv: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.project_id or not self.owner_seed:
            fail("RESERVATION_INPUT_INVALID", "project ID and owner seed must be nonempty")
        if self.writer.role != "run_reservation":
            fail("RESERVATION_WRITER_INVALID", "reservation writer role is invalid")
        for field, value in (
            ("tool_sha256", self.tool_sha256),
            ("configuration_sha256", self.configuration_sha256),
            ("runtime_sha256", self.runtime_sha256),
            ("environment_path_policy_sha256", self.environment_path_policy_sha256),
        ):
            require_sha256(value, field)
        if not self.canonical_argv or not all(
            isinstance(value, str) and "\x00" not in value for value in self.canonical_argv
        ):
            fail("RESERVATION_INPUT_INVALID", "canonical argv must be a nonempty NUL-free array")


@dataclass(frozen=True)
class ReservationResult:
    sealed_run_id: str
    strict_parent: str
    run_root: str
    payload: dict[str, object]
    publication: Publication

    @property
    def evidence(self) -> ArtifactEvidence:
        return self.publication.evidence

    @property
    def transcript(self) -> dict[str, object]:
        return self.publication.transcript


@dataclass(frozen=True)
class AcceptanceResult:
    payload: dict[str, object]
    publication: Publication

    @property
    def evidence(self) -> ArtifactEvidence:
        return self.publication.evidence

    @property
    def transcript(self) -> dict[str, object]:
        return self.publication.transcript


def _load_canonical_json(root_fd: int, relative: str) -> tuple[dict[str, object], bytes]:
    raw, _ = read_regular_at(root_fd, relative)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid canonical JSON at {relative}") from exc
    if not isinstance(value, dict) or canonical_json(value) != raw:
        fail("NONCANONICAL_JSON", f"{relative} is not canonical JSON")
    return value, raw


def _validate_identity(value: object, expected_type: str, schema: str) -> dict[str, object]:
    keys = {"type", "dev", "inode", "mode", "uid", "gid"}
    if expected_type == "regular":
        keys.add("size")
    row = require_exact_keys(value, keys, schema)
    if row["type"] != expected_type:
        fail("RESERVATION_SCHEMA_INVALID", f"{schema} type differs")
    numeric = keys - {"type"}
    if not all(isinstance(row[key], int) and not isinstance(row[key], bool) for key in numeric):
        fail("RESERVATION_SCHEMA_INVALID", f"{schema} numeric identity fields differ")
    return row


def _validate_descriptor(value: object, schema: str) -> dict[str, object]:
    row = require_exact_keys(
        value, {"identity", "status_flags", "descriptor_flags"}, schema
    )
    _validate_identity(row["identity"], "directory", f"{schema}-identity")
    if not isinstance(row["status_flags"], int) or not isinstance(row["descriptor_flags"], int):
        fail("RESERVATION_SCHEMA_INVALID", f"{schema} flags differ")
    return row


def _validate_mount(value: object, schema: str) -> dict[str, object]:
    row = require_exact_keys(
        value,
        {
            "mount_id", "parent_mount_id", "major_minor", "mount_root", "mount_point",
            "mount_options", "filesystem_type", "mount_source", "super_options",
            "mount_namespace",
        },
        schema,
    )
    namespace = require_exact_keys(
        row["mount_namespace"], {"dev", "inode", "link"}, f"{schema}-namespace"
    )
    if not all(isinstance(row[key], int) for key in ("mount_id", "parent_mount_id")):
        fail("RESERVATION_SCHEMA_INVALID", f"{schema} mount IDs differ")
    if not all(isinstance(namespace[key], int) for key in ("dev", "inode")):
        fail("RESERVATION_SCHEMA_INVALID", f"{schema} namespace identity differs")
    if not isinstance(namespace["link"], str):
        fail("RESERVATION_SCHEMA_INVALID", f"{schema} namespace link differs")
    for key in ("major_minor", "mount_root", "mount_point", "filesystem_type", "mount_source"):
        if not isinstance(row[key], str):
            fail("RESERVATION_SCHEMA_INVALID", f"{schema} text field differs")
    for key in ("mount_options", "super_options"):
        if not isinstance(row[key], list) or not all(isinstance(item, str) for item in row[key]):
            fail("RESERVATION_SCHEMA_INVALID", f"{schema} options differ")
    return row


def _validate_process_identity(value: object) -> dict[str, object]:
    row = require_exact_keys(
        value,
        {
            "effective_uid", "effective_gid", "supplementary_groups", "namespaces",
            "capability_state",
        },
        "reservation-process-identity",
    )
    if not all(isinstance(row[key], int) for key in ("effective_uid", "effective_gid")):
        fail("RESERVATION_SCHEMA_INVALID", "effective process IDs differ")
    if (
        not isinstance(row["supplementary_groups"], list)
        or not all(isinstance(item, int) for item in row["supplementary_groups"])
        or row["supplementary_groups"] != sorted(row["supplementary_groups"])
    ):
        fail("RESERVATION_SCHEMA_INVALID", "supplementary groups differ")
    namespaces = require_exact_keys(
        row["namespaces"], {"mnt", "user", "pid"}, "reservation-process-namespaces"
    )
    capabilities = require_exact_keys(
        row["capability_state"],
        {"CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb", "NoNewPrivs"},
        "reservation-process-capabilities",
    )
    if not all(isinstance(item, str) for item in (*namespaces.values(), *capabilities.values())):
        fail("RESERVATION_SCHEMA_INVALID", "namespace/capability values differ")
    return row


def validate_reservation_payload(
    value: object, sealed_run_id: str, strict_parent: str, run_root: str
) -> dict[str, object]:
    row = require_exact_keys(
        value,
        {
            "schema", "sealed_run_id", "strict_parent", "run_root", "parent", "child",
            "owner", "producer", "toolchain", "process_identity", "creation",
            "publication_policy",
        },
        RESERVATION_SCHEMA,
    )
    if (
        row["schema"] != RESERVATION_SCHEMA
        or row["sealed_run_id"] != sealed_run_id
        or row["strict_parent"] != strict_parent
        or row["run_root"] != run_root
    ):
        fail("RESERVATION_MISMATCH", "reservation run binding differs")
    parent = require_exact_keys(
        row["parent"], {"path", "descriptor", "mount"}, "reservation-parent"
    )
    child = require_exact_keys(
        row["child"], {"name", "path", "descriptor", "mount"}, "reservation-child"
    )
    _validate_descriptor(parent["descriptor"], "reservation-parent-descriptor")
    _validate_descriptor(child["descriptor"], "reservation-child-descriptor")
    _validate_mount(parent["mount"], "reservation-parent-mount")
    _validate_mount(child["mount"], "reservation-child-mount")
    owner = require_exact_keys(
        row["owner"], {"project_id", "owner_seed"}, "reservation-owner"
    )
    if not all(isinstance(owner[key], str) and owner[key] for key in owner):
        fail("RESERVATION_SCHEMA_INVALID", "owner seed fields differ")
    producer = require_exact_keys(row["producer"], {"writer"}, "reservation-producer")
    reservation_writer = WriterIdentity.from_dict(producer["writer"])
    if reservation_writer.role != "run_reservation":
        fail("RESERVATION_WRITER_INVALID", "stored reservation writer role differs")
    toolchain = require_exact_keys(
        row["toolchain"],
        {
            "tool_sha256", "configuration_sha256", "runtime_sha256",
            "environment_path_policy_sha256", "canonical_argv", "canonical_argv_sha256",
        },
        "reservation-toolchain",
    )
    for key in (
        "tool_sha256", "configuration_sha256", "runtime_sha256",
        "environment_path_policy_sha256", "canonical_argv_sha256",
    ):
        require_sha256(toolchain[key], key)
    if (
        not isinstance(toolchain["canonical_argv"], list)
        or not toolchain["canonical_argv"]
        or not all(
            isinstance(item, str) and "\x00" not in item
            for item in toolchain["canonical_argv"]
        )
        or sha256_bytes(canonical_json(toolchain["canonical_argv"]))
        != toolchain["canonical_argv_sha256"]
    ):
        fail("RESERVATION_SCHEMA_INVALID", "canonical argv closure differs")
    creation = require_exact_keys(
        row["creation"],
        {
            "mkdir_method", "mkdir_flags", "child_open_flags", "absent_child_verified",
            "parent_synced_after_child_creation", "descriptor_containment_verified",
        },
        "reservation-creation",
    )
    if (
        creation["mkdir_method"] != "mkdirat"
        or creation["mkdir_flags"]
        != ["exclusive-absent-child", "relative-trusted-parent-dirfd"]
        or creation["child_open_flags"]
        != ["O_RDONLY", "O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC"]
        or any(
            creation[key] is not True
            for key in (
                "absent_child_verified", "parent_synced_after_child_creation",
                "descriptor_containment_verified",
            )
        )
    ):
        fail("RESERVATION_SCHEMA_INVALID", "child creation facts differ")
    publication_policy = require_exact_keys(
        row["publication_policy"],
        {
            "final_path_method", "exclusive_flags", "file_sync_required",
            "containing_directory_sync_required", "descriptor_readback_required",
        },
        "reservation-publication-policy",
    )
    if (
        publication_policy["final_path_method"] != "direct-openat-exclusive"
        or publication_policy["exclusive_flags"]
        != ["O_CREAT", "O_EXCL", "O_NOFOLLOW", "O_CLOEXEC"]
        or any(
            publication_policy[key] is not True
            for key in (
                "file_sync_required", "containing_directory_sync_required",
                "descriptor_readback_required",
            )
        )
    ):
        fail("RESERVATION_SCHEMA_INVALID", "intended publication policy differs")
    _validate_process_identity(row["process_identity"])
    if (
        parent["path"] != strict_parent
        or child["path"] != run_root
        or child["name"] != sealed_run_id
    ):
        fail("RESERVATION_MISMATCH", "reservation parent/child path differs")
    return row


def validate_acceptance_payload(
    value: object, sealed_run_id: str, run_root: str
) -> dict[str, object]:
    row = require_exact_keys(
        value,
        {
            "schema", "sealed_run_id", "run_root", "reservation_sha256",
            "reservation_transcript_sha256", "reservation_writer", "acceptance_writer",
            "validated_identity",
        },
        ACCEPTANCE_SCHEMA,
    )
    if (
        row["schema"] != ACCEPTANCE_SCHEMA
        or row["sealed_run_id"] != sealed_run_id
        or row["run_root"] != run_root
    ):
        fail("ACCEPTANCE_MISMATCH", "acceptance run binding differs")
    require_sha256(row["reservation_sha256"], "reservation hash")
    require_sha256(row["reservation_transcript_sha256"], "reservation transcript hash")
    reservation_writer = WriterIdentity.from_dict(row["reservation_writer"])
    acceptance_writer = WriterIdentity.from_dict(row["acceptance_writer"])
    if (
        reservation_writer.role != "run_reservation"
        or acceptance_writer.role != "reservation_acceptance"
    ):
        fail("ACCEPTANCE_WRITER_NOT_DISTINCT", "reservation/acceptance roles differ")
    if (
        reservation_writer.writer_id == acceptance_writer.writer_id
        or reservation_writer.task_id == acceptance_writer.task_id
        or reservation_writer.role == acceptance_writer.role
    ):
        fail("ACCEPTANCE_WRITER_NOT_DISTINCT", "reservation acceptance writer is not distinct")
    identities = require_exact_keys(
        row["validated_identity"],
        {"parent", "child", "reservation"},
        "acceptance-validated-identity",
    )
    _validate_identity(identities["parent"], "directory", "acceptance-parent-identity")
    _validate_identity(identities["child"], "directory", "acceptance-child-identity")
    _validate_identity(
        identities["reservation"], "regular", "acceptance-reservation-identity"
    )
    return row


def validate_owner_binding_payload(
    value: object, sealed_run_id: str, run_root: str
) -> dict[str, object]:
    row = require_exact_keys(
        value,
        {
            "schema", "sealed_run_id", "run_root", "project_id", "owner_seed",
            "reservation_sha256", "reservation_acceptance_sha256", "writer",
        },
        OWNER_SCHEMA,
    )
    if (
        row["schema"] != OWNER_SCHEMA
        or row["sealed_run_id"] != sealed_run_id
        or row["run_root"] != run_root
    ):
        fail("OWNER_BINDING_MISMATCH", "owner binding run differs")
    require_sha256(row["reservation_sha256"], "reservation hash")
    require_sha256(row["reservation_acceptance_sha256"], "acceptance hash")
    if not all(isinstance(row[key], str) and row[key] for key in ("project_id", "owner_seed")):
        fail("OWNER_BINDING_MISMATCH", "owner/project fields differ")
    if WriterIdentity.from_dict(row["writer"]).role != "g0":
        fail("OWNER_BINDING_MISMATCH", "owner binding writer role differs")
    return row


def reserve_strict_run(
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    inputs: ReservationInputs,
    *,
    policy: WriterPolicy | None = None,
    fail_after: str | None = None,
    race_hook: Callable[[], None] | None = None,
    expected_parent_mount_id: int | None = None,
) -> ReservationResult:
    sealed_run_id = validate_sealed_run_id(sealed_run_id)
    strict_parent, run_root = validate_run_root_text(strict_parent, sealed_run_id, run_root)
    policy = default_writer_policy() if policy is None else policy
    policy.authorize("reservation.json", inputs.writer)
    if expected_parent_mount_id is not None and (
        type(expected_parent_mount_id) is not int
        or expected_parent_mount_id <= 0
    ):
        fail("MOUNT_SUBSTITUTION", "expected parent mount ID is invalid")
    parent_fd = open_absolute_directory(strict_parent)
    child_fd = -1
    try:
        if (
            expected_parent_mount_id is not None
            and mount_identity(parent_fd)["mount_id"]
            != expected_parent_mount_id
        ):
            fail(
                "MOUNT_SUBSTITUTION",
                "strict parent descriptor mount differs from preflight",
            )
        try:
            os.stat(sealed_run_id, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            fail("RUN_ROOT_EXISTS", "strict child already exists and cannot be adopted")
        if race_hook is not None:
            race_hook()
        assert_path_matches_fd(strict_parent, parent_fd)
        if (
            expected_parent_mount_id is not None
            and mount_identity(parent_fd)["mount_id"]
            != expected_parent_mount_id
        ):
            fail(
                "MOUNT_SUBSTITUTION",
                "strict parent descriptor mount changed before mkdirat",
            )
        os.mkdir(sealed_run_id, 0o755, dir_fd=parent_fd)
        os.fsync(parent_fd)
        child_fd = os.open(sealed_run_id, DIR_FLAGS, dir_fd=parent_fd)
        if (
            expected_parent_mount_id is not None
            and mount_identity(child_fd)["mount_id"]
            != expected_parent_mount_id
        ):
            fail(
                "MOUNT_SUBSTITUTION",
                "strict child descriptor mount differs before publication",
            )
        assert_path_matches_fd(strict_parent, parent_fd)
        assert_path_matches_fd(run_root, child_fd)
        if os.fstat(parent_fd).st_dev != os.fstat(child_fd).st_dev:
            fail("MOUNT_SUBSTITUTION", "child and parent filesystem devices differ")
        parent_descriptor = descriptor_evidence(parent_fd)
        child_descriptor = descriptor_evidence(child_fd)
        parent_mount = mount_identity(parent_fd)
        child_mount = mount_identity(child_fd)
        payload: dict[str, object] = {
            "schema": RESERVATION_SCHEMA,
            "sealed_run_id": sealed_run_id,
            "strict_parent": strict_parent,
            "run_root": run_root,
            "parent": {
                "path": strict_parent, "descriptor": parent_descriptor, "mount": parent_mount,
            },
            "child": {
                "name": sealed_run_id, "path": run_root, "descriptor": child_descriptor,
                "mount": child_mount,
            },
            "owner": {"project_id": inputs.project_id, "owner_seed": inputs.owner_seed},
            "producer": {"writer": inputs.writer.to_dict()},
            "toolchain": {
                "tool_sha256": inputs.tool_sha256,
                "configuration_sha256": inputs.configuration_sha256,
                "runtime_sha256": inputs.runtime_sha256,
                "environment_path_policy_sha256": inputs.environment_path_policy_sha256,
                "canonical_argv": list(inputs.canonical_argv),
                "canonical_argv_sha256": sha256_bytes(canonical_json(list(inputs.canonical_argv))),
            },
            "process_identity": process_identity(),
            "creation": {
                "mkdir_method": "mkdirat",
                "mkdir_flags": ["exclusive-absent-child", "relative-trusted-parent-dirfd"],
                "child_open_flags": ["O_RDONLY", "O_DIRECTORY", "O_NOFOLLOW", "O_CLOEXEC"],
                "absent_child_verified": True,
                "parent_synced_after_child_creation": True,
                "descriptor_containment_verified": True,
            },
            "publication_policy": {
                "final_path_method": "direct-openat-exclusive",
                "exclusive_flags": ["O_CREAT", "O_EXCL", "O_NOFOLLOW", "O_CLOEXEC"],
                "file_sync_required": True,
                "containing_directory_sync_required": True,
                "descriptor_readback_required": True,
            },
        }
        validate_reservation_payload(payload, sealed_run_id, strict_parent, run_root)
        context = {
            "parent_identity": stable_identity(os.fstat(parent_fd)),
            "child_identity": stable_identity(os.fstat(child_fd)),
            "parent_mount": parent_mount,
            "child_mount": child_mount,
        }
        if fail_after == "child_created":
            fail("INJECTED_CRASH", "injected after child creation")
        publisher = StagePublisher(
            strict_parent, sealed_run_id, run_root, inputs.writer, policy
        )
        publications = publisher.publish_json_on_reserved_descriptors(
            parent_fd,
            child_fd,
            "reservation.json",
            payload,
            context=context,
            _failure_after=fail_after,
        )
        if len(publications) != 1:
            fail(
                "RESERVATION_PUBLICATION_INVALID",
                "reservation created unexpected directories",
            )
        return ReservationResult(
            sealed_run_id, strict_parent, run_root, payload, publications[0]
        )
    finally:
        if child_fd >= 0:
            os.close(child_fd)
        os.close(parent_fd)


def accept_reservation(
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    transcript_registry: ExternalTranscriptRegistry,
    writer: WriterIdentity,
    *,
    reservation_transcript_sha256: str,
    policy: WriterPolicy | None = None,
    run_binding: RunDescriptorBinding | None = None,
) -> AcceptanceResult:
    sealed_run_id = validate_sealed_run_id(sealed_run_id)
    policy = default_writer_policy() if policy is None else policy
    if writer.role != "reservation_acceptance":
        fail("ACCEPTANCE_WRITER_INVALID", "reservation acceptance role differs")
    if not isinstance(transcript_registry, ExternalTranscriptRegistry):
        fail(
            "EXTERNAL_REGISTRY_INVALID",
            "reservation acceptance requires a typed external transcript registry",
        )
    reservation_transcript_sha256 = require_sha256(
        reservation_transcript_sha256,
        "reservation transcript sha256",
    )
    reservation_transcript = transcript_registry.get_exact(
        reservation_transcript_sha256
    )
    if run_binding is not None and not isinstance(run_binding, RunDescriptorBinding):
        fail("RUN_BINDING_INVALID", "acceptance run binding is not typed")
    handle_context = (
        open_run_handle(strict_parent, sealed_run_id, run_root)
        if run_binding is None
        else open_bound_run_handle(
            strict_parent, sealed_run_id, run_root, run_binding
        )
    )
    with handle_context as handle:
        reservation, reservation_raw = _load_canonical_json(handle.root_fd, "reservation.json")
        validate_reservation_payload(reservation, sealed_run_id, strict_parent, run_root)
        reservation_writer = WriterIdentity.from_dict(reservation["producer"]["writer"])
        if (
            writer.writer_id == reservation_writer.writer_id
            or writer.task_id == reservation_writer.task_id
            or writer.role == reservation_writer.role
        ):
            fail("ACCEPTANCE_WRITER_NOT_DISTINCT", "acceptance writer is not distinct")
        policy.authorize("reservation-acceptance.json", writer)
        transcript = validate_publication_transcript(
            handle.root_fd,
            sealed_run_id,
            run_root,
            reservation_transcript,
            expected_relative_path="reservation.json",
            expected_writer=reservation_writer,
        )
        expected_context = {
            "parent_identity": stable_identity(os.fstat(handle.parent_fd)),
            "child_identity": stable_identity(os.fstat(handle.root_fd)),
            "parent_mount": mount_identity(handle.parent_fd),
            "child_mount": mount_identity(handle.root_fd),
        }
        if transcript["context"] != expected_context:
            fail("TRANSCRIPT_MISMATCH", "reservation transcript namespace context differs")
        transcript_registry.verify_current()
        payload: dict[str, object] = {
            "schema": ACCEPTANCE_SCHEMA,
            "sealed_run_id": sealed_run_id,
            "run_root": run_root,
            "reservation_sha256": sha256_bytes(reservation_raw),
            "reservation_transcript_sha256": transcript_hash(transcript),
            "reservation_writer": reservation_writer.to_dict(),
            "acceptance_writer": writer.to_dict(),
            "validated_identity": {
                "parent": expected_context["parent_identity"],
                "child": expected_context["child_identity"],
                "reservation": transcript["identity"],
            },
        }
        validate_acceptance_payload(payload, sealed_run_id, run_root)
        transcript_registry.verify_current()
        publications = StagePublisher(
            strict_parent,
            sealed_run_id,
            run_root,
            writer,
            policy,
            run_binding=run_binding,
        ).publish_json_on_reserved_descriptors(
            handle.parent_fd,
            handle.root_fd,
            "reservation-acceptance.json",
            payload,
            context=expected_context,
        )
        if len(publications) != 1:
            fail(
                "ACCEPTANCE_PUBLICATION_INVALID",
                "acceptance created unexpected directories",
            )
        for document in transcript_registry.documents.values():
            document.assert_current()
        return AcceptanceResult(payload, publications[0])


def publish_owner_binding(
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    owner_seed: str,
    project_id: str,
    writer: WriterIdentity,
    *,
    policy: WriterPolicy | None = None,
    run_binding: RunDescriptorBinding | None = None,
) -> Publication:
    sealed_run_id = validate_sealed_run_id(sealed_run_id)
    policy = default_writer_policy() if policy is None else policy
    if writer.role != "g0":
        fail("OWNER_WRITER_INVALID", "owner binding requires the G0 writer")
    if run_binding is not None and not isinstance(run_binding, RunDescriptorBinding):
        fail("RUN_BINDING_INVALID", "owner run binding is not typed")
    handle_context = (
        open_run_handle(strict_parent, sealed_run_id, run_root)
        if run_binding is None
        else open_bound_run_handle(
            strict_parent, sealed_run_id, run_root, run_binding
        )
    )
    with handle_context as handle:
        reservation, reservation_raw = _load_canonical_json(handle.root_fd, "reservation.json")
        acceptance, acceptance_raw = _load_canonical_json(
            handle.root_fd, "reservation-acceptance.json"
        )
        validate_reservation_payload(reservation, sealed_run_id, strict_parent, run_root)
        validate_acceptance_payload(acceptance, sealed_run_id, run_root)
        if reservation["owner"] != {"project_id": project_id, "owner_seed": owner_seed}:
            fail("OWNER_BINDING_MISMATCH", "owner/project seed differs from reservation")
        if acceptance["reservation_sha256"] != sha256_bytes(reservation_raw):
            fail("OWNER_BINDING_MISMATCH", "acceptance does not bind reservation bytes")
        payload: dict[str, object] = {
            "schema": OWNER_SCHEMA,
            "sealed_run_id": sealed_run_id,
            "run_root": run_root,
            "project_id": project_id,
            "owner_seed": owner_seed,
            "reservation_sha256": sha256_bytes(reservation_raw),
            "reservation_acceptance_sha256": sha256_bytes(acceptance_raw),
            "writer": writer.to_dict(),
        }
        validate_owner_binding_payload(payload, sealed_run_id, run_root)
        publications = StagePublisher(
            strict_parent,
            sealed_run_id,
            run_root,
            writer,
            policy,
            run_binding=run_binding,
        ).publish_json_on_reserved_descriptors(
            handle.parent_fd,
            handle.root_fd,
            "owner-binding.json",
            payload,
        )
        if len(publications) != 1:
            fail(
                "OWNER_PUBLICATION_INVALID",
                "owner binding created unexpected directories",
            )
        return publications[0]
