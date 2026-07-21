"""Descriptor-bound semantic reconciliation for strict V6 CP4 and CP5.

The bridge is deliberately synthetic-fixture-safe.  It consumes only typed,
already-published run artifacts and never enumerates protected experiment
roots.  The CP4 copy plan is recomputed in memory and is never published.
"""
from __future__ import annotations

import os
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from provenance.strict_v6 import (
    V6ContractError,
    digest_json as provenance_digest_json,
    digest_records as provenance_digest_records,
    plan_source_pre_copies,
    reconcile_cp3_buckets,
    reconcile_final_admission,
    validate_copy_ledger,
    validate_source_pre_records,
)

from .canonical import (
    canonical_json,
    fail,
    require_sha256,
    sha256_bytes,
    strict_json_loads,
    validate_absolute_path_text,
    validate_sealed_run_id,
)
from .external import ExternalTranscriptRegistry
from .filesystem import (
    DIR_FLAGS,
    exists_relative,
    open_relative_directory,
    read_regular_at,
    stable_identity,
)
from .publication import (
    ArtifactEvidence,
    Publication,
    transcript_hash,
    validate_publication_transcript,
)


CHECKPOINT_SCHEMA = "experiments7-strict-checkpoint/v6"
CP4_SEMANTIC_SCHEMA = "experiments7-cp4-semantic-binding/v6"
CP5_SEMANTIC_SCHEMA = "experiments7-cp5-semantic-binding/v6"

CP3_CHECKPOINT_PATH = "checkpoints/cp3.json"
CP4_CHECKPOINT_PATH = "checkpoints/cp4.json"
SOURCE_PRE_PATH = "manifests/source-pre.jsonl"
PAPER_PRE_PATH = "manifests/paper-pre.json"
SOURCE_POST_PATH = "manifests/source-post.jsonl"
PAPER_POST_PATH = "manifests/paper-post.json"
INVENTORY_PATH = "inventory/results.jsonl"
RAW_NODES_PATH = "provenance/nodes.jsonl"
PROVENANCE_EDGES_PATH = "provenance/edges.jsonl"
PROVENANCE_RECOMPUTATIONS_PATH = "provenance/recomputations.jsonl"
PROVENANCE_ROUNDING_PATH = "provenance/rounding-proofs.jsonl"
PROVENANCE_PATHS = (
    RAW_NODES_PATH,
    PROVENANCE_EDGES_PATH,
    PROVENANCE_RECOMPUTATIONS_PATH,
    PROVENANCE_ROUNDING_PATH,
)
CP3_RECONCILIATION_PATH = "admission/precopy/reconciliation.json"
BUCKET_ROOT = "admission/precopy/buckets"
COPIES_PATH = "copies.jsonl"
FINAL_ROOT = "admission/final"
FINAL_RESULTS_PATH = "admission/final/results.jsonl"
FINAL_RECONCILIATION_PATH = "admission/final/reconciliation.json"
FORBIDDEN_COPY_PLAN_PATHS = (
    "copy-plan.json",
    "copy-plan.jsonl",
    "manifests/copy-plan.json",
    "manifests/copy-plan.jsonl",
)

_CHECKPOINT_KEYS = {
    "schema",
    "sealed_run_id",
    "run_root",
    "checkpoint",
    "previous_checkpoint_sha256",
    "previous_checkpoint_transcript_sha256",
    "stage_actual_paths",
    "stage_actual_paths_sha256",
    "bindings",
}
_SEMANTIC_CHECKPOINT_KEYS = {"semantic_bindings", "semantic_bindings_sha256"}
_CP3_SEMANTIC_KEYS = {
    "schema",
    "sealed_run_id",
    "run_root",
    "checkpoint",
    "predecessor",
    "cp1_inventory",
    "source_pre",
    "stage_publications",
    "stage_publication_count",
    "provenance",
    "bucket_publications",
    "bucket_publication_count",
    "reconciliation",
    "semantic_sha256",
}
_BINDING_FIELDS = (
    "relative_path",
    "artifact_sha256",
    "bytes",
    "identity",
    "writer",
    "writer_rule_id",
    "publication_transcript_sha256",
    "physical_evidence_sha256",
)
_STAGE_BINDING_FIELDS = (
    "artifact_type",
    "publication_first_completed_monotonic_ns",
    "publication_emitted_monotonic_ns",
)


@dataclass(frozen=True)
class CP3SemanticPublications:
    """Typed prior artifacts needed to replay CP3 and derive CP4."""

    checkpoint: Publication
    cp1_checkpoint: Publication
    cp2_checkpoint: Publication
    source_pre: Publication
    paper_pre: Publication
    inventory: Publication
    raw_nodes: Publication
    provenance_edges: Publication
    provenance_recomputations: Publication
    provenance_rounding_proofs: Publication
    reconciliation: Publication
    buckets: tuple[Publication, ...]


@dataclass(frozen=True)
class CP4SemanticPublications:
    """Typed CP4 checkpoint and the exact stage it sealed."""

    checkpoint: Publication
    stage: tuple[Publication, ...]


def _validate_root(root_fd: int, sealed_run_id: str, run_root: str) -> tuple[str, str]:
    if type(root_fd) is not int or root_fd < 0:
        fail("CP45_ROOT_INVALID", "root_fd must be an open directory descriptor")
    try:
        current = os.fstat(root_fd)
    except OSError:
        fail("CP45_ROOT_INVALID", "root descriptor is unavailable")
    if not stat.S_ISDIR(current.st_mode):
        fail("CP45_ROOT_INVALID", "root_fd is not a directory descriptor")
    sealed_run_id = validate_sealed_run_id(sealed_run_id)
    run_root = validate_absolute_path_text(run_root, "run_root")
    if run_root.rsplit("/", 1)[-1] != sealed_run_id:
        fail("CP45_ROOT_INVALID", "run_root is not the exact sealed_run_id child")
    descriptor_path = os.readlink(f"/proc/self/fd/{root_fd}")
    if (
        descriptor_path.endswith(" (deleted)")
        or descriptor_path != run_root
        or os.path.realpath(run_root) != run_root
    ):
        fail("CP45_ROOT_INVALID", "root descriptor does not match canonical run_root")
    return sealed_run_id, run_root


def _publications(value: Iterable[Publication], field: str) -> tuple[Publication, ...]:
    try:
        rows = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be an iterable of Publication values") from exc
    if not rows or any(not isinstance(row, Publication) for row in rows):
        fail("CP45_PUBLICATION_INVALID", f"{field} must contain typed Publications")
    return rows


def _load_object(root_fd: int, relative: str) -> tuple[dict[str, object], bytes, os.stat_result]:
    raw, current = read_regular_at(root_fd, relative)
    value = strict_json_loads(raw, relative)
    if not isinstance(value, dict) or canonical_json(value) != raw:
        fail("CP45_NONCANONICAL_JSON", f"{relative} is not canonical JSON")
    return value, raw, current


def _load_jsonl(
    root_fd: int,
    relative: str,
    *,
    require_nonempty: bool,
) -> tuple[list[dict[str, object]], bytes, os.stat_result]:
    raw, current = read_regular_at(root_fd, relative)
    if require_nonempty and not raw:
        fail("CP45_JSONL_INVALID", f"{relative} must not be empty")
    if raw and not raw.endswith(b"\n"):
        fail("CP45_JSONL_INVALID", f"{relative} lacks a final newline")
    rows: list[dict[str, object]] = []
    for number, line in enumerate(raw.splitlines(keepends=True), start=1):
        if line == b"\n" or not line.endswith(b"\n"):
            fail("CP45_JSONL_INVALID", f"{relative} line {number} is empty or unterminated")
        value = strict_json_loads(line, f"{relative}:{number}")
        if not isinstance(value, dict) or canonical_json(value) != line:
            fail("CP45_JSONL_INVALID", f"{relative} line {number} is not canonical JSON")
        rows.append(value)
    return rows, raw, current


def _writer_contract(relative: str) -> tuple[str, str, str, str]:
    if relative.startswith("checkpoints/"):
        return (
            "checkpoint_controller",
            "checkpoint:",
            "checkpoint_controller-writer",
            "checkpoints",
        )
    if relative.startswith("manifests/"):
        return ("g0", "g0:", "g0-writer", "manifests")
    if relative == INVENTORY_PATH:
        return ("inventory", "inventory:", "inventory-writer", "inventory")
    if relative == "provenance" or relative in PROVENANCE_PATHS:
        return ("provenance", "provenance:", "provenance-writer", "provenance")
    if relative in {"admission", "admission/precopy", BUCKET_ROOT}:
        return ("admission", "admission:", "admission-writer", "admission")
    if relative == CP3_RECONCILIATION_PATH or relative.startswith(f"{BUCKET_ROOT}/"):
        return ("admission", "admission:", "admission-writer", "admission")
    if relative == FINAL_ROOT or relative.startswith(f"{FINAL_ROOT}/"):
        return ("admission", "admission:", "admission-writer", "admission")
    if relative == COPIES_PATH or relative == "raw" or relative.startswith("raw/"):
        rule = "copies" if relative == COPIES_PATH else "raw"
        return ("copy", "copy:", "copy-writer", rule)
    fail("CP45_STAGE_PATH_INVALID", f"no strict writer contract exists for {relative}")


def _require_physical_match(
    relative: str,
    evidence: ArtifactEvidence,
    current: os.stat_result,
    raw: bytes | None,
) -> None:
    identity = stable_identity(current)
    if evidence.identity != identity:
        fail("CP45_EVIDENCE_MISMATCH", f"{relative} descriptor identity is stale")
    if evidence.artifact_type == "regular":
        assert raw is not None
        if (
            identity.get("mode") != 0o444
            or evidence.sha256 != sha256_bytes(raw)
            or evidence.size != len(raw)
        ):
            fail("CP45_EVIDENCE_MISMATCH", f"{relative} bytes or immutable mode differ")
    elif raw is not None or evidence.sha256 is not None or evidence.size is not None:
        fail("CP45_EVIDENCE_MISMATCH", f"{relative} directory evidence has byte fields")


def _validate_publication(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    publication: Publication,
    transcript_registry: ExternalTranscriptRegistry,
    *,
    expected_path: str,
    expected_type: str,
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(publication, Publication):
        fail("CP45_PUBLICATION_INVALID", f"{expected_path} is not a typed Publication")
    evidence = publication.evidence
    if ArtifactEvidence.from_dict(evidence.to_dict()) != evidence:
        fail("CP45_PUBLICATION_INVALID", f"{expected_path} evidence does not round-trip")
    role, task_prefix, writer_id, rule_id = _writer_contract(expected_path)
    writer = evidence.writer
    if (
        evidence.relative_path != expected_path
        or evidence.artifact_type != expected_type
        or writer.role != role
        or not writer.task_id.startswith(task_prefix)
        or writer.writer_id != writer_id
        or evidence.writer_rule_id != rule_id
    ):
        fail("CP45_WRITER_INVALID", f"writer/path evidence differs for {expected_path}")
    external = transcript_registry.get_exact(evidence.publication_transcript_sha256)
    if external != publication.transcript:
        fail("CP45_TRANSCRIPT_MISMATCH", f"{expected_path} transcript is not registered exactly")
    transcript = validate_publication_transcript(
        root_fd,
        sealed_run_id,
        run_root,
        external,
        expected_relative_path=expected_path,
        expected_writer=writer,
    )
    if (
        transcript_hash(transcript) != evidence.publication_transcript_sha256
        or transcript["artifact_type"] != evidence.artifact_type
        or transcript["sha256"] != evidence.sha256
        or transcript["bytes"] != evidence.size
        or transcript["identity"] != evidence.identity
    ):
        fail("CP45_TRANSCRIPT_MISMATCH", f"{expected_path} transcript/evidence differs")
    if expected_type == "regular":
        raw, current = read_regular_at(root_fd, expected_path)
    else:
        directory_fd = open_relative_directory(root_fd, expected_path)
        try:
            current = os.fstat(directory_fd)
        finally:
            os.close(directory_fd)
        raw = None
    _require_physical_match(expected_path, evidence, current, raw)
    first = transcript["events"][0]
    assert isinstance(first, dict)
    binding = {
        "relative_path": expected_path,
        "artifact_type": expected_type,
        "artifact_sha256": evidence.sha256,
        "bytes": evidence.size,
        "identity": evidence.identity,
        "writer": writer.to_dict(),
        "writer_rule_id": evidence.writer_rule_id,
        "publication_transcript_sha256": evidence.publication_transcript_sha256,
        "physical_evidence_sha256": sha256_bytes(canonical_json(evidence.to_dict())),
        "publication_first_completed_monotonic_ns": first["completed_monotonic_ns"],
        "publication_emitted_monotonic_ns": transcript["emitted_monotonic_ns"],
    }
    return binding, transcript


def _validate_checkpoint(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    publication: Publication,
    transcript_registry: ExternalTranscriptRegistry,
    checkpoint: int,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    relative = f"checkpoints/cp{checkpoint}.json"
    binding, transcript = _validate_publication(
        root_fd,
        sealed_run_id,
        run_root,
        publication,
        transcript_registry,
        expected_path=relative,
        expected_type="regular",
    )
    payload, raw, _ = _load_object(root_fd, relative)
    keys = set(payload)
    if keys != _CHECKPOINT_KEYS | _SEMANTIC_CHECKPOINT_KEYS:
        fail("CP45_CHECKPOINT_INVALID", f"CP{checkpoint} keys differ from the V6 contract")
    if (
        payload.get("schema") != CHECKPOINT_SCHEMA
        or payload.get("sealed_run_id") != sealed_run_id
        or payload.get("run_root") != run_root
        or type(payload.get("checkpoint")) is not int
        or payload["checkpoint"] != checkpoint
        or publication.evidence.sha256 != sha256_bytes(raw)
    ):
        fail("CP45_CHECKPOINT_INVALID", f"CP{checkpoint} run binding or bytes differ")
    require_sha256(payload.get("previous_checkpoint_sha256"), "previous checkpoint sha256")
    require_sha256(
        payload.get("previous_checkpoint_transcript_sha256"),
        "previous checkpoint transcript sha256",
    )
    if not isinstance(payload.get("bindings"), dict):
        fail("CP45_CHECKPOINT_INVALID", f"CP{checkpoint} bindings must be an object")
    stage = payload.get("stage_actual_paths")
    if not isinstance(stage, list) or not stage:
        fail("CP45_CHECKPOINT_INVALID", f"CP{checkpoint} stage evidence is empty")
    evidences = [ArtifactEvidence.from_dict(item) for item in stage]
    paths = [item.relative_path for item in evidences]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        fail("CP45_CHECKPOINT_INVALID", f"CP{checkpoint} stage paths are not sorted/unique")
    if payload.get("stage_actual_paths_sha256") != sha256_bytes(canonical_json(stage)):
        fail("CP45_CHECKPOINT_INVALID", f"CP{checkpoint} stage digest differs")
    semantic = payload.get("semantic_bindings")
    expected_semantic_schema = {
        3: "experiments7-cp3-semantic-binding/v6",
        4: CP4_SEMANTIC_SCHEMA,
    }.get(checkpoint)
    if (
        not isinstance(semantic, dict)
        or expected_semantic_schema is None
        or semantic.get("schema") != expected_semantic_schema
        or semantic.get("sealed_run_id") != sealed_run_id
        or semantic.get("run_root") != run_root
        or type(semantic.get("checkpoint")) is not int
        or semantic["checkpoint"] != checkpoint
        or payload.get("semantic_bindings_sha256")
        != sha256_bytes(canonical_json(semantic))
    ):
        fail("CP45_CHECKPOINT_INVALID", f"CP{checkpoint} semantic envelope differs")
    semantic_sha256 = semantic.get("semantic_sha256")
    semantic_base = {
        key: value for key, value in semantic.items() if key != "semantic_sha256"
    }
    if semantic_sha256 != sha256_bytes(canonical_json(semantic_base)):
        fail("CP45_CHECKPOINT_INVALID", f"CP{checkpoint} inner semantic digest differs")
    return payload, binding, transcript


def _require_binding(
    candidate: object,
    current: Mapping[str, object],
    field: str,
    *,
    extra_fields: Iterable[str] = (),
    stage_binding: bool = False,
) -> Mapping[str, object]:
    expected_keys = set(_BINDING_FIELDS) | set(extra_fields)
    if stage_binding:
        expected_keys.update(_STAGE_BINDING_FIELDS)
    comparable = expected_keys & set(current)
    if (
        not isinstance(candidate, Mapping)
        or set(candidate) != expected_keys
        or canonical_json({key: candidate[key] for key in comparable})
        != canonical_json({key: current[key] for key in comparable})
    ):
        fail("CP45_CHECKPOINT_INVALID", f"{field} is not the current Publication binding")
    return candidate


def _require_evidence_binding(
    candidate: object,
    evidence: ArtifactEvidence,
    field: str,
) -> Mapping[str, object]:
    current = {
        "relative_path": evidence.relative_path,
        "artifact_sha256": evidence.sha256,
        "bytes": evidence.size,
        "identity": evidence.identity,
        "writer": evidence.writer.to_dict(),
        "writer_rule_id": evidence.writer_rule_id,
        "publication_transcript_sha256": evidence.publication_transcript_sha256,
        "physical_evidence_sha256": sha256_bytes(canonical_json(evidence.to_dict())),
    }
    return _require_binding(candidate, current, field, stage_binding=True)


def _validate_cp3_checkpoint_semantics(
    checkpoint: Mapping[str, object],
    *,
    sealed_run_id: str,
    run_root: str,
    current_bindings: Mapping[str, Mapping[str, object]],
    source_pre_records: Sequence[Mapping[str, object]],
    inventory: Sequence[Mapping[str, object]],
    buckets: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    graph_rows: Mapping[str, Sequence[Mapping[str, object]]],
    cp1_checkpoint: Mapping[str, object],
    cp2_checkpoint: Mapping[str, object],
) -> None:
    semantic = checkpoint.get("semantic_bindings")
    if not isinstance(semantic, Mapping) or set(semantic) != _CP3_SEMANTIC_KEYS:
        fail("CP45_CHECKPOINT_INVALID", "CP3 semantic binding keys differ")
    if (
        semantic.get("schema") != "experiments7-cp3-semantic-binding/v6"
        or semantic.get("sealed_run_id") != sealed_run_id
        or semantic.get("run_root") != run_root
        or semantic.get("checkpoint") != 3
    ):
        fail("CP45_CHECKPOINT_INVALID", "CP3 semantic run/index binding differs")
    predecessor = semantic.get("predecessor")
    if (
        not isinstance(predecessor, Mapping)
        or set(predecessor)
        != {
            "checkpoint",
            "checkpoint_sha256",
            "checkpoint_transcript_sha256",
            "cp1_publication",
            "cp2_publication",
            "semantics",
        }
        or type(predecessor.get("checkpoint")) is not int
        or predecessor.get("checkpoint") != 2
        or predecessor.get("checkpoint_sha256")
        != checkpoint.get("previous_checkpoint_sha256")
        or predecessor.get("checkpoint_transcript_sha256")
        != checkpoint.get("previous_checkpoint_transcript_sha256")
    ):
        fail("CP45_CHECKPOINT_INVALID", "CP3 semantic predecessor differs")
    require_sha256(predecessor["checkpoint_sha256"], "CP3 predecessor sha256")
    require_sha256(
        predecessor["checkpoint_transcript_sha256"],
        "CP3 predecessor transcript sha256",
    )
    _require_binding(
        predecessor.get("cp1_publication"),
        current_bindings["checkpoints/cp1.json"],
        "CP3 predecessor CP1 binding",
    )
    _require_binding(
        predecessor.get("cp2_publication"),
        current_bindings["checkpoints/cp2.json"],
        "CP3 predecessor CP2 binding",
    )
    cp1_semantic = cp1_checkpoint.get("semantic_bindings")
    cp2_semantic = cp2_checkpoint.get("semantic_bindings")
    expected_predecessor_semantics = {
        "cp1": cp1_semantic,
        "cp1_semantics_sha256": sha256_bytes(canonical_json(cp1_semantic)),
        "cp2": cp2_semantic,
        "cp2_semantics_sha256": sha256_bytes(canonical_json(cp2_semantic)),
    }
    if canonical_json(predecessor.get("semantics")) != canonical_json(
        expected_predecessor_semantics
    ):
        fail("CP45_CHECKPOINT_INVALID", "CP3 predecessor semantics differ")

    source_pre = _require_binding(
        semantic.get("source_pre"),
        current_bindings[SOURCE_PRE_PATH],
        "CP3 source-pre semantic binding",
        extra_fields=(
            "record_count",
            "records_sha256",
            "publication_first_completed_monotonic_ns",
            "publication_emitted_monotonic_ns",
        ),
    )
    if (
        type(source_pre.get("record_count")) is not int
        or source_pre.get("record_count") != len(source_pre_records)
        or source_pre.get("records_sha256")
        != provenance_digest_records(source_pre_records)
    ):
        fail("CP45_CHECKPOINT_INVALID", "CP3 source-pre semantic record binding differs")
    cp1_inventory = _require_binding(
        semantic.get("cp1_inventory"),
        current_bindings[INVENTORY_PATH],
        "CP3 inventory semantic binding",
        extra_fields=(
            "checkpoint_sha256",
            "checkpoint_transcript_sha256",
            "inventory_sha256",
            "result_count",
        ),
    )
    if (
        cp1_inventory.get("checkpoint_sha256")
        != current_bindings["checkpoints/cp1.json"]["artifact_sha256"]
        or cp1_inventory.get("checkpoint_transcript_sha256")
        != current_bindings["checkpoints/cp1.json"][
            "publication_transcript_sha256"
        ]
        or type(cp1_inventory.get("result_count")) is not int
        or cp1_inventory.get("result_count") != len(inventory)
        or cp1_inventory.get("inventory_sha256") != summary.get("inventory_sha256")
    ):
        fail("CP45_CHECKPOINT_INVALID", "CP3 inventory semantic binding differs")

    stage_rows = semantic.get("stage_publications")
    checkpoint_stage = checkpoint.get("stage_actual_paths")
    if not isinstance(stage_rows, list) or not isinstance(checkpoint_stage, list):
        fail("CP45_CHECKPOINT_INVALID", "CP3 semantic stage binding is invalid")
    if (
        type(semantic.get("stage_publication_count")) is not int
        or semantic.get("stage_publication_count") != len(stage_rows)
    ):
        fail("CP45_CHECKPOINT_INVALID", "CP3 semantic stage count differs")
    stage_by_path: dict[str, Mapping[str, object]] = {}
    for row in stage_rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("relative_path"), str):
            fail("CP45_CHECKPOINT_INVALID", "CP3 semantic stage row is invalid")
        relative = str(row["relative_path"])
        if relative in stage_by_path:
            fail("CP45_CHECKPOINT_INVALID", "CP3 semantic stage paths repeat")
        stage_by_path[relative] = row
    evidence_rows = [ArtifactEvidence.from_dict(row) for row in checkpoint_stage]
    evidence_paths = [evidence.relative_path for evidence in evidence_rows]
    semantic_stage_paths = [str(row["relative_path"]) for row in stage_rows]
    if semantic_stage_paths != evidence_paths or set(stage_by_path) != set(evidence_paths):
        fail("CP45_CHECKPOINT_INVALID", "CP3 semantic/checkpoint stage paths differ")
    for evidence in evidence_rows:
        _require_binding(
            stage_by_path[evidence.relative_path],
            current_bindings[evidence.relative_path],
            f"CP3 stage {evidence.relative_path}",
            stage_binding=True,
        )

    provenance = semantic.get("provenance")
    if (
        not isinstance(provenance, Mapping)
        or set(provenance) != {"publications", "reconciliation"}
        or not isinstance(provenance.get("publications"), list)
        or not isinstance(provenance.get("reconciliation"), Mapping)
    ):
        fail("CP45_CHECKPOINT_INVALID", "CP3 provenance semantic binding is invalid")
    provenance_publications = provenance["publications"]
    assert isinstance(provenance_publications, list)
    if (
        len(provenance_publications) != len(PROVENANCE_PATHS)
        or any(not isinstance(row, Mapping) for row in provenance_publications)
    ):
        fail("CP45_CHECKPOINT_INVALID", "CP3 provenance publication rows differ")
    provenance_by_path = {
        row.get("relative_path"): row
        for row in provenance_publications
    }
    if (
        [row.get("relative_path") for row in provenance_publications]
        != list(PROVENANCE_PATHS)
        or
        len(provenance_by_path) != len(provenance_publications)
        or set(provenance_by_path) != set(PROVENANCE_PATHS)
    ):
        fail("CP45_CHECKPOINT_INVALID", "CP3 provenance publication paths differ")
    id_fields = {
        RAW_NODES_PATH: "node_id",
        PROVENANCE_EDGES_PATH: "edge_id",
        PROVENANCE_RECOMPUTATIONS_PATH: "recomputation_id",
        PROVENANCE_ROUNDING_PATH: "rounding_proof_id",
    }
    expected_provenance = {
        "schema": "experiments7-cp3-provenance-reconciliation/v6",
    }
    reconciliation_fields = {
        RAW_NODES_PATH: ("node_count", "nodes_sha256"),
        PROVENANCE_EDGES_PATH: ("edge_count", "edges_sha256"),
        PROVENANCE_RECOMPUTATIONS_PATH: (
            "recomputation_count",
            "recomputations_sha256",
        ),
        PROVENANCE_ROUNDING_PATH: (
            "rounding_proof_count",
            "rounding_proofs_sha256",
        ),
    }
    for relative in PROVENANCE_PATHS:
        rows = list(graph_rows[relative])
        id_field = id_fields[relative]
        identifiers = [row.get(id_field) for row in rows]
        if (
            any(not isinstance(identifier, str) or not identifier for identifier in identifiers)
            or len(identifiers) != len(set(identifiers))
        ):
            fail("CP45_CHECKPOINT_INVALID", f"CP3 graph IDs differ at {relative}")
        ordered = [row for _, row in sorted(zip(identifiers, rows))]
        count_field, digest_field = reconciliation_fields[relative]
        expected_provenance[count_field] = len(rows)
        expected_provenance[digest_field] = provenance_digest_records(ordered)
        graph_binding = _require_binding(
            provenance_by_path[relative],
            current_bindings[relative],
            f"CP3 graph {relative}",
            extra_fields=("canonical_file_sha256", "record_count"),
            stage_binding=True,
        )
        if (
            graph_binding.get("canonical_file_sha256")
            != current_bindings[relative]["artifact_sha256"]
            or type(graph_binding.get("record_count")) is not int
            or graph_binding.get("record_count") != len(rows)
        ):
            fail("CP45_CHECKPOINT_INVALID", f"CP3 graph content differs at {relative}")
    if canonical_json(provenance.get("reconciliation")) != canonical_json(
        expected_provenance
    ):
        fail("CP45_CHECKPOINT_INVALID", "CP3 provenance reconciliation differs")

    bucket_rows = semantic.get("bucket_publications")
    if (
        not isinstance(bucket_rows, list)
        or any(not isinstance(row, Mapping) for row in bucket_rows)
        or type(semantic.get("bucket_publication_count")) is not int
        or semantic.get("bucket_publication_count") != len(bucket_rows)
    ):
        fail("CP45_CHECKPOINT_INVALID", "CP3 bucket semantic count differs")
    bucket_by_path = {
        row.get("relative_path"): row
        for row in bucket_rows
    }
    expected_bucket_paths = {
        f"{BUCKET_ROOT}/{bucket['paper_section_id']}.json" for bucket in buckets
    }
    if (
        [row.get("relative_path") for row in bucket_rows]
        != sorted(expected_bucket_paths)
        or
        len(bucket_by_path) != len(bucket_rows)
        or set(bucket_by_path) != expected_bucket_paths
    ):
        fail("CP45_CHECKPOINT_INVALID", "CP3 bucket semantic paths differ")
    for bucket in buckets:
        relative = f"{BUCKET_ROOT}/{bucket['paper_section_id']}.json"
        row = _require_binding(
            bucket_by_path[relative],
            current_bindings[relative],
            f"CP3 bucket {relative}",
            extra_fields=(
                "paper_section_id",
                "canonical_bucket_sha256",
                "canonical_file_sha256",
            ),
            stage_binding=True,
        )
        if (
            row.get("paper_section_id") != bucket["paper_section_id"]
            or row.get("canonical_bucket_sha256") != provenance_digest_json(bucket)
            or row.get("canonical_file_sha256")
            != current_bindings[relative]["artifact_sha256"]
        ):
            fail("CP45_CHECKPOINT_INVALID", f"CP3 bucket content binding differs at {relative}")

    reconciliation = _require_binding(
        semantic.get("reconciliation"),
        current_bindings[CP3_RECONCILIATION_PATH],
        "CP3 reconciliation semantic binding",
        extra_fields=(
            "canonical_file_sha256",
            "recomputed_sha256",
            "total_result_count",
            "precopy_admitted_count",
            "precopy_unresolved_count",
            "copy_plan_input_sha256",
        ),
        stage_binding=True,
    )
    expected_reconciliation = {
        "canonical_file_sha256": current_bindings[CP3_RECONCILIATION_PATH][
            "artifact_sha256"
        ],
        "recomputed_sha256": sha256_bytes(canonical_json(summary)),
        "total_result_count": summary.get("total_result_count"),
        "precopy_admitted_count": summary.get("precopy_admitted_count"),
        "precopy_unresolved_count": summary.get("precopy_unresolved_count"),
        "copy_plan_input_sha256": summary.get("copy_plan_input_sha256"),
    }
    for field in (
        "total_result_count",
        "precopy_admitted_count",
        "precopy_unresolved_count",
    ):
        if type(reconciliation.get(field)) is not int:
            fail("CP45_CHECKPOINT_INVALID", "CP3 reconciliation count type differs")
    if canonical_json(
        {key: reconciliation.get(key) for key in expected_reconciliation}
    ) != canonical_json(expected_reconciliation):
        fail("CP45_CHECKPOINT_INVALID", "CP3 reconciliation semantic content differs")


def _checkpoint_contains(
    checkpoint: Mapping[str, object], publications: Iterable[Publication]
) -> None:
    stage = checkpoint["stage_actual_paths"]
    assert isinstance(stage, list)
    by_path = {ArtifactEvidence.from_dict(row).relative_path: row for row in stage}
    for publication in publications:
        evidence = publication.evidence
        if by_path.get(evidence.relative_path) != evidence.to_dict():
            fail(
                "CP45_CHECKPOINT_INVALID",
                f"checkpoint does not bind current {evidence.relative_path} evidence",
            )


def _checkpoint_exact_stage(
    checkpoint: Mapping[str, object], publications: Iterable[Publication]
) -> None:
    expected = [
        publication.evidence.to_dict()
        for publication in sorted(publications, key=lambda row: row.evidence.relative_path)
    ]
    if checkpoint.get("stage_actual_paths") != expected:
        fail("CP45_CHECKPOINT_INVALID", "checkpoint stage is not the exact typed stage")


def _walk_subtree(root_fd: int, relative: str) -> dict[str, str]:
    directory_fd = open_relative_directory(root_fd, relative)
    result: dict[str, str] = {}

    def walk(fd: int, prefix: str) -> None:
        before = stable_identity(os.fstat(fd))
        result[prefix] = "directory"
        for name in sorted(os.listdir(fd), key=os.fsencode):
            if not isinstance(name, str) or not name or "\x00" in name:
                fail("CP45_TREE_INVALID", f"{prefix} contains an invalid entry")
            current = os.stat(name, dir_fd=fd, follow_symlinks=False)
            child = f"{prefix}/{name}"
            if stat.S_ISDIR(current.st_mode):
                child_fd = os.open(name, DIR_FLAGS, dir_fd=fd)
                try:
                    if stable_identity(os.fstat(child_fd)) != stable_identity(current):
                        fail("CP45_TREE_MUTATED", f"{child} changed while opening")
                    walk(child_fd, child)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(current.st_mode):
                result[child] = "regular"
            else:
                fail("CP45_TREE_INVALID", f"{child} is linked or special")
        if stable_identity(os.fstat(fd)) != before:
            fail("CP45_TREE_MUTATED", f"{prefix} changed while enumerating")

    try:
        walk(directory_fd, relative)
        return result
    finally:
        os.close(directory_fd)


def _directory_paths(destinations: Sequence[str]) -> set[str]:
    directories: set[str] = set()
    for destination in destinations:
        parts = destination.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add("/".join(parts[:index]))
    return directories


def _stage_map(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    publications: Iterable[Publication],
    transcript_registry: ExternalTranscriptRegistry,
    expected: Mapping[str, str],
    *,
    predecessor_emitted: int,
) -> tuple[dict[str, Publication], dict[str, dict[str, object]]]:
    stage = _publications(publications, "stage publications")
    by_path: dict[str, Publication] = {}
    for publication in stage:
        relative = publication.evidence.relative_path
        if relative in by_path:
            fail("CP45_STAGE_INVALID", "stage contains a duplicate publication path")
        by_path[relative] = publication
    if set(by_path) != set(expected):
        fail("CP45_STAGE_PATH_INVALID", "stage paths are not the exact required set")
    bindings: dict[str, dict[str, object]] = {}
    for relative in sorted(by_path):
        binding, _ = _validate_publication(
            root_fd,
            sealed_run_id,
            run_root,
            by_path[relative],
            transcript_registry,
            expected_path=relative,
            expected_type=expected[relative],
        )
        if binding["publication_first_completed_monotonic_ns"] <= predecessor_emitted:
            fail("CP45_CHRONOLOGY_INVALID", f"{relative} predates its predecessor checkpoint")
        bindings[relative] = binding
    return by_path, bindings


def _load_cp3_inputs(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    inputs: CP3SemanticPublications,
    transcript_registry: ExternalTranscriptRegistry,
    *,
    expected_result_count: int,
) -> dict[str, object]:
    if not isinstance(inputs, CP3SemanticPublications):
        fail("CP45_INPUT_INVALID", "CP3 inputs must be CP3SemanticPublications")
    buckets = _publications(inputs.buckets, "CP3 bucket publications")
    checkpoint, checkpoint_binding, checkpoint_transcript = _validate_checkpoint(
        root_fd,
        sealed_run_id,
        run_root,
        inputs.checkpoint,
        transcript_registry,
        3,
    )
    named = (
        (inputs.cp1_checkpoint, "checkpoints/cp1.json"),
        (inputs.cp2_checkpoint, "checkpoints/cp2.json"),
        (inputs.source_pre, SOURCE_PRE_PATH),
        (inputs.paper_pre, PAPER_PRE_PATH),
        (inputs.inventory, INVENTORY_PATH),
        (inputs.raw_nodes, RAW_NODES_PATH),
        (inputs.provenance_edges, PROVENANCE_EDGES_PATH),
        (inputs.provenance_recomputations, PROVENANCE_RECOMPUTATIONS_PATH),
        (inputs.provenance_rounding_proofs, PROVENANCE_ROUNDING_PATH),
        (inputs.reconciliation, CP3_RECONCILIATION_PATH),
    )
    bindings: dict[str, dict[str, object]] = {}
    transcripts: dict[str, dict[str, object]] = {}
    for publication, relative in named:
        binding, transcript = _validate_publication(
            root_fd,
            sealed_run_id,
            run_root,
            publication,
            transcript_registry,
            expected_path=relative,
            expected_type="regular",
        )
        bindings[relative] = binding
        transcripts[relative] = transcript
    bucket_values: list[dict[str, object]] = []
    for publication in buckets:
        relative = publication.evidence.relative_path
        if not relative.startswith(f"{BUCKET_ROOT}/") or not relative.endswith(".json"):
            fail("CP45_CP3_INVALID", "CP3 bucket Publication path is not canonical")
        if relative in bindings:
            fail("CP45_CP3_INVALID", "CP3 input paths overlap")
        binding, transcript = _validate_publication(
            root_fd,
            sealed_run_id,
            run_root,
            publication,
            transcript_registry,
            expected_path=relative,
            expected_type="regular",
        )
        bucket, _, _ = _load_object(root_fd, relative)
        section = bucket.get("paper_section_id")
        if not isinstance(section, str) or relative != f"{BUCKET_ROOT}/{section}.json":
            fail("CP45_CP3_INVALID", "CP3 bucket filename/section binding differs")
        bindings[relative] = binding
        transcripts[relative] = transcript
        bucket_values.append(bucket)

    checkpoint_stage = checkpoint.get("stage_actual_paths")
    assert isinstance(checkpoint_stage, list)
    expected_cp3_stage = {
        "provenance",
        "admission",
        "admission/precopy",
        BUCKET_ROOT,
        *PROVENANCE_PATHS,
        CP3_RECONCILIATION_PATH,
        *(publication.evidence.relative_path for publication in buckets),
    }
    stage_evidences = [ArtifactEvidence.from_dict(row) for row in checkpoint_stage]
    stage_paths = [evidence.relative_path for evidence in stage_evidences]
    if (
        len(stage_paths) != len(set(stage_paths))
        or set(stage_paths) != expected_cp3_stage
    ):
        fail("CP45_CHECKPOINT_INVALID", "CP3 checkpoint stage is not the exact CP3 tree")
    for evidence in stage_evidences:
        external = transcript_registry.get_exact(
            evidence.publication_transcript_sha256
        )
        if not isinstance(external, dict):
            fail("CP45_TRANSCRIPT_MISMATCH", "CP3 stage transcript is not an object")
        binding, transcript = _validate_publication(
            root_fd,
            sealed_run_id,
            run_root,
            Publication(evidence, external),
            transcript_registry,
            expected_path=evidence.relative_path,
            expected_type=evidence.artifact_type,
        )
        existing = bindings.get(evidence.relative_path)
        if existing is not None and canonical_json(existing) != canonical_json(binding):
            fail("CP45_CHECKPOINT_INVALID", "typed CP3 input differs from checkpoint stage")
        bindings[evidence.relative_path] = binding
        transcripts[evidence.relative_path] = transcript

    expected_bucket_tree = {BUCKET_ROOT: "directory"}
    expected_bucket_tree.update(
        {publication.evidence.relative_path: "regular" for publication in buckets}
    )
    if _walk_subtree(root_fd, BUCKET_ROOT) != expected_bucket_tree:
        fail("CP45_CP3_INVALID", "CP3 bucket tree has extra or missing paths")
    expected_provenance_tree = {"provenance": "directory"}
    expected_provenance_tree.update(
        {relative: "regular" for relative in PROVENANCE_PATHS}
    )
    if _walk_subtree(root_fd, "provenance") != expected_provenance_tree:
        fail("CP45_CP3_INVALID", "CP3 provenance tree has extra or missing paths")

    _checkpoint_contains(
        checkpoint,
        (
            inputs.raw_nodes,
            inputs.provenance_edges,
            inputs.provenance_recomputations,
            inputs.provenance_rounding_proofs,
            inputs.reconciliation,
            *buckets,
        ),
    )
    cp1, cp1_raw, _ = _load_object(root_fd, "checkpoints/cp1.json")
    cp2, cp2_raw, _ = _load_object(root_fd, "checkpoints/cp2.json")
    for index, value, raw, publication in (
        (1, cp1, cp1_raw, inputs.cp1_checkpoint),
        (2, cp2, cp2_raw, inputs.cp2_checkpoint),
    ):
        semantic = value.get("semantic_bindings")
        stage = value.get("stage_actual_paths")
        if (
            set(value) != _CHECKPOINT_KEYS | _SEMANTIC_CHECKPOINT_KEYS
            or value.get("schema") != CHECKPOINT_SCHEMA
            or value.get("sealed_run_id") != sealed_run_id
            or value.get("run_root") != run_root
            or type(value.get("checkpoint")) is not int
            or value.get("checkpoint") != index
            or publication.evidence.sha256 != sha256_bytes(raw)
            or not isinstance(stage, list)
            or not stage
            or value.get("stage_actual_paths_sha256")
            != sha256_bytes(canonical_json(stage))
            or not isinstance(semantic, dict)
            or value.get("semantic_bindings_sha256")
            != sha256_bytes(canonical_json(semantic))
        ):
            fail("CP45_CHECKPOINT_INVALID", f"CP{index} current run binding differs")
    _checkpoint_contains(cp1, (inputs.inventory,))
    if (
        cp2.get("previous_checkpoint_sha256") != inputs.cp1_checkpoint.evidence.sha256
        or cp2.get("previous_checkpoint_transcript_sha256")
        != inputs.cp1_checkpoint.evidence.publication_transcript_sha256
        or checkpoint.get("previous_checkpoint_sha256")
        != inputs.cp2_checkpoint.evidence.sha256
        or checkpoint.get("previous_checkpoint_transcript_sha256")
        != inputs.cp2_checkpoint.evidence.publication_transcript_sha256
    ):
        fail("CP45_CHECKPOINT_INVALID", "CP1/CP2/CP3 checkpoint chain differs")
    checkpoint_first = checkpoint_transcript["events"][0]
    assert isinstance(checkpoint_first, dict)
    for relative, transcript in transcripts.items():
        if transcript["emitted_monotonic_ns"] >= checkpoint_first["completed_monotonic_ns"]:
            fail("CP45_CHRONOLOGY_INVALID", f"{relative} does not predate CP3")
    cp1_transcript = transcripts["checkpoints/cp1.json"]
    cp2_transcript = transcripts["checkpoints/cp2.json"]
    cp1_first = cp1_transcript["events"][0]
    cp2_first = cp2_transcript["events"][0]
    assert isinstance(cp1_first, dict) and isinstance(cp2_first, dict)
    if (
        transcripts[SOURCE_PRE_PATH]["emitted_monotonic_ns"]
        >= cp1_first["completed_monotonic_ns"]
        or transcripts[INVENTORY_PATH]["emitted_monotonic_ns"]
        >= cp1_first["completed_monotonic_ns"]
        or cp1_transcript["emitted_monotonic_ns"]
        >= cp2_first["completed_monotonic_ns"]
    ):
        fail("CP45_CHRONOLOGY_INVALID", "CP0/CP1/CP2 publication chronology differs")

    source_pre, source_pre_raw, _ = _load_jsonl(
        root_fd, SOURCE_PRE_PATH, require_nonempty=True
    )
    try:
        source_pre = validate_source_pre_records(source_pre)
    except V6ContractError as exc:
        fail("CP45_SOURCE_PRE_INVALID", str(exc))
    paper_pre, paper_pre_raw, _ = _load_object(root_fd, PAPER_PRE_PATH)
    inventory, _, _ = _load_jsonl(root_fd, INVENTORY_PATH, require_nonempty=True)
    raw_nodes, _, _ = _load_jsonl(root_fd, RAW_NODES_PATH, require_nonempty=True)
    graph_rows = {
        RAW_NODES_PATH: raw_nodes,
    }
    for relative in PROVENANCE_PATHS[1:]:
        rows, _, _ = _load_jsonl(root_fd, relative, require_nonempty=False)
        graph_rows[relative] = rows
    summary, summary_raw, _ = _load_object(root_fd, CP3_RECONCILIATION_PATH)
    try:
        recomputed = reconcile_cp3_buckets(
            inventory,
            bucket_values,
            expected_result_count=expected_result_count,
        )
    except V6ContractError as exc:
        fail("CP45_CP3_INVALID", str(exc))
    if summary_raw != canonical_json(recomputed):
        fail("CP45_CP3_INVALID", "CP3 reconciliation differs from current buckets/inventory")
    _validate_cp3_checkpoint_semantics(
        checkpoint,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
        current_bindings=bindings,
        source_pre_records=source_pre,
        inventory=inventory,
        buckets=bucket_values,
        summary=summary,
        graph_rows=graph_rows,
        cp1_checkpoint=cp1,
        cp2_checkpoint=cp2,
    )
    raw_only = [row for row in raw_nodes if row.get("node_type") == "raw_artifact"]
    return {
        "checkpoint": checkpoint,
        "checkpoint_binding": checkpoint_binding,
        "checkpoint_transcript": checkpoint_transcript,
        "bindings": bindings,
        "source_pre": source_pre,
        "source_pre_raw": source_pre_raw,
        "paper_pre": paper_pre,
        "paper_pre_raw": paper_pre_raw,
        "inventory": inventory,
        "raw_nodes": raw_only,
        "buckets": bucket_values,
        "summary": summary,
    }


def _provenance_identity(current: os.stat_result) -> dict[str, object]:
    return {
        "file_type": "regular",
        "st_dev": current.st_dev,
        "st_ino": current.st_ino,
        "mode": current.st_mode,
        "size": current.st_size,
        "mtime_ns": current.st_mtime_ns,
    }


def _validate_cp4_internal(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    publications: Iterable[Publication],
    cp3_inputs: CP3SemanticPublications,
    transcript_registry: ExternalTranscriptRegistry,
    previous_checkpoint_sha256: str,
    previous_checkpoint_transcript_sha256: str,
    *,
    expected_result_count: int,
) -> dict[str, object]:
    cp3 = _load_cp3_inputs(
        root_fd,
        sealed_run_id,
        run_root,
        cp3_inputs,
        transcript_registry,
        expected_result_count=expected_result_count,
    )
    previous_checkpoint_sha256 = require_sha256(
        previous_checkpoint_sha256, "previous checkpoint sha256"
    )
    previous_checkpoint_transcript_sha256 = require_sha256(
        previous_checkpoint_transcript_sha256,
        "previous checkpoint transcript sha256",
    )
    checkpoint_publication = cp3_inputs.checkpoint
    if (
        previous_checkpoint_sha256 != checkpoint_publication.evidence.sha256
        or previous_checkpoint_transcript_sha256
        != checkpoint_publication.evidence.publication_transcript_sha256
    ):
        fail("CP4_PREDECESSOR_INVALID", "CP4 predecessor is not the current typed CP3")
    try:
        plan = plan_source_pre_copies(
            cp3["summary"],
            cp3["raw_nodes"],
            cp3["source_pre"],
            expected_result_count=expected_result_count,
        )
    except V6ContractError as exc:
        fail("CP4_COPY_PLAN_INVALID", str(exc))
    if any(exists_relative(root_fd, relative) for relative in FORBIDDEN_COPY_PLAN_PATHS):
        fail("CP4_COPY_PLAN_INVALID", "copy-plan artifacts must not be published")
    entries = plan["entries"]
    assert isinstance(entries, list)
    destinations = [str(entry["destination_relative_path"]) for entry in entries]
    directories = _directory_paths(destinations)
    expected_stage = {COPIES_PATH: "regular"}
    expected_stage.update({relative: "directory" for relative in directories})
    expected_stage.update({relative: "regular" for relative in destinations})
    predecessor_transcript = cp3["checkpoint_transcript"]
    assert isinstance(predecessor_transcript, dict)
    predecessor_emitted = predecessor_transcript["emitted_monotonic_ns"]
    assert type(predecessor_emitted) is int
    stage_by_path, stage_bindings = _stage_map(
        root_fd,
        sealed_run_id,
        run_root,
        publications,
        transcript_registry,
        expected_stage,
        predecessor_emitted=predecessor_emitted,
    )
    expected_tree = {
        relative: ("directory" if relative in directories else "regular")
        for relative in directories | set(destinations)
    }
    if expected_tree:
        if _walk_subtree(root_fd, "raw") != expected_tree:
            fail("CP4_TREE_INVALID", "raw tree has extra, missing, or mistyped paths")
    elif exists_relative(root_fd, "raw"):
        fail("CP4_TREE_INVALID", "raw tree exists for an empty copy plan")

    ledger_rows, _, _ = _load_jsonl(root_fd, COPIES_PATH, require_nonempty=bool(entries))
    ordered_raw = [row.get("raw_artifact_id") for row in ledger_rows]
    if ordered_raw != plan["raw_artifact_ids"]:
        fail("CP4_LEDGER_INVALID", "copies.jsonl rows are not in deterministic plan order")
    copied_payloads: dict[str, bytes] = {}
    ledger_by_raw = {row.get("raw_artifact_id"): row for row in ledger_rows}
    for entry in entries:
        raw_id = entry["raw_artifact_id"]
        destination = str(entry["destination_relative_path"])
        payload, current = read_regular_at(root_fd, destination)
        row = ledger_by_raw.get(raw_id)
        if not isinstance(row, dict):
            fail("CP4_LEDGER_INVALID", f"copy ledger row is absent for {raw_id}")
        if row.get("destination_identity") != _provenance_identity(current):
            fail("CP4_LEDGER_INVALID", f"destination descriptor evidence differs for {raw_id}")
        copied_payloads[str(entry["copied_artifact_id"])] = payload
        if stage_by_path[destination].evidence.sha256 != sha256_bytes(payload):
            fail("CP4_EVIDENCE_MISMATCH", f"copied Publication bytes differ for {raw_id}")
    try:
        copy_reconciliation = validate_copy_ledger(
            cp3["summary"],
            plan,
            ledger_rows,
            copied_payloads,
            cp3["source_pre"],
            expected_result_count=expected_result_count,
        )
    except V6ContractError as exc:
        fail("CP4_LEDGER_INVALID", str(exc))
    base = {
        "schema": CP4_SEMANTIC_SCHEMA,
        "sealed_run_id": sealed_run_id,
        "run_root": run_root,
        "checkpoint": 4,
        "predecessor": {
            "checkpoint": 3,
            "checkpoint_sha256": previous_checkpoint_sha256,
            "checkpoint_transcript_sha256": previous_checkpoint_transcript_sha256,
            "publication": cp3["checkpoint_binding"],
        },
        "cp3_inputs": [cp3["bindings"][path] for path in sorted(cp3["bindings"])],
        "stage_publications": [stage_bindings[path] for path in sorted(stage_bindings)],
        "stage_publication_count": len(stage_bindings),
        "copy_plan_sha256": provenance_digest_json(plan),
        "copy_plan_published": False,
        "planned_destinations": destinations,
        "copy_reconciliation": copy_reconciliation,
    }
    semantic = {**base, "semantic_sha256": sha256_bytes(canonical_json(base))}
    return {
        "semantic": semantic,
        "cp3": cp3,
        "plan": plan,
        "ledger_rows": ledger_rows,
        "copied_payloads": copied_payloads,
        "stage_publications": tuple(stage_by_path.values()),
        "stage_bindings": stage_bindings,
    }


def validate_cp4_semantics(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    publications: Iterable[Publication],
    cp3_inputs: CP3SemanticPublications,
    transcript_registry: ExternalTranscriptRegistry,
    previous_checkpoint_sha256: str,
    previous_checkpoint_transcript_sha256: str,
    *,
    expected_result_count: int = 1908,
) -> dict[str, object]:
    """Replay CP3, derive the unpersisted plan, and validate the exact CP4 tree."""

    sealed_run_id, run_root = _validate_root(root_fd, sealed_run_id, run_root)
    if not isinstance(transcript_registry, ExternalTranscriptRegistry):
        fail("EXTERNAL_REGISTRY_INVALID", "CP4 requires a typed transcript registry")
    if type(expected_result_count) is not int or expected_result_count <= 0:
        fail("CP45_COUNT_INVALID", "expected_result_count must be a positive exact integer")
    transcript_registry.verify_current()
    result = _validate_cp4_internal(
        root_fd,
        sealed_run_id,
        run_root,
        publications,
        cp3_inputs,
        transcript_registry,
        previous_checkpoint_sha256,
        previous_checkpoint_transcript_sha256,
        expected_result_count=expected_result_count,
    )
    transcript_registry.verify_current()
    semantic = result["semantic"]
    assert isinstance(semantic, dict)
    return semantic


def validate_cp5_semantics(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    publications: Iterable[Publication],
    cp3_inputs: CP3SemanticPublications,
    cp4_inputs: CP4SemanticPublications,
    transcript_registry: ExternalTranscriptRegistry,
    previous_checkpoint_sha256: str,
    previous_checkpoint_transcript_sha256: str,
    *,
    expected_result_count: int = 1908,
) -> dict[str, object]:
    """Replay CP3/CP4 and prove exact final admission plus post manifests."""

    sealed_run_id, run_root = _validate_root(root_fd, sealed_run_id, run_root)
    if not isinstance(transcript_registry, ExternalTranscriptRegistry):
        fail("EXTERNAL_REGISTRY_INVALID", "CP5 requires a typed transcript registry")
    if not isinstance(cp4_inputs, CP4SemanticPublications):
        fail("CP45_INPUT_INVALID", "CP4 inputs must be CP4SemanticPublications")
    if type(expected_result_count) is not int or expected_result_count <= 0:
        fail("CP45_COUNT_INVALID", "expected_result_count must be a positive exact integer")
    transcript_registry.verify_current()
    cp4_stage = _publications(cp4_inputs.stage, "CP4 stage publications")
    replay = _validate_cp4_internal(
        root_fd,
        sealed_run_id,
        run_root,
        cp4_stage,
        cp3_inputs,
        transcript_registry,
        cp3_inputs.checkpoint.evidence.sha256 or "",
        cp3_inputs.checkpoint.evidence.publication_transcript_sha256,
        expected_result_count=expected_result_count,
    )
    cp4_checkpoint, cp4_binding, cp4_transcript = _validate_checkpoint(
        root_fd,
        sealed_run_id,
        run_root,
        cp4_inputs.checkpoint,
        transcript_registry,
        4,
    )
    if (
        cp4_checkpoint.get("previous_checkpoint_sha256")
        != cp3_inputs.checkpoint.evidence.sha256
        or cp4_checkpoint.get("previous_checkpoint_transcript_sha256")
        != cp3_inputs.checkpoint.evidence.publication_transcript_sha256
    ):
        fail("CP5_PREDECESSOR_INVALID", "CP4 checkpoint does not bind current CP3")
    _checkpoint_exact_stage(cp4_checkpoint, cp4_stage)
    if cp4_checkpoint.get("semantic_bindings") != replay["semantic"]:
        fail("CP5_PREDECESSOR_INVALID", "CP4 checkpoint semantics differ from current replay")
    cp4_first = cp4_transcript["events"][0]
    assert isinstance(cp4_first, dict)
    stage_bindings = replay["stage_bindings"]
    assert isinstance(stage_bindings, dict)
    if any(
        binding["publication_emitted_monotonic_ns"]
        >= cp4_first["completed_monotonic_ns"]
        for binding in stage_bindings.values()
    ):
        fail("CP5_PREDECESSOR_INVALID", "CP4 checkpoint predates its sealed stage")
    previous_checkpoint_sha256 = require_sha256(
        previous_checkpoint_sha256, "previous checkpoint sha256"
    )
    previous_checkpoint_transcript_sha256 = require_sha256(
        previous_checkpoint_transcript_sha256,
        "previous checkpoint transcript sha256",
    )
    if (
        previous_checkpoint_sha256 != cp4_inputs.checkpoint.evidence.sha256
        or previous_checkpoint_transcript_sha256
        != cp4_inputs.checkpoint.evidence.publication_transcript_sha256
    ):
        fail("CP5_PREDECESSOR_INVALID", "CP5 predecessor is not the current typed CP4")

    expected_stage = {
        FINAL_ROOT: "directory",
        FINAL_RESULTS_PATH: "regular",
        FINAL_RECONCILIATION_PATH: "regular",
        SOURCE_POST_PATH: "regular",
        PAPER_POST_PATH: "regular",
    }
    cp4_emitted = cp4_transcript["emitted_monotonic_ns"]
    assert type(cp4_emitted) is int
    _, cp5_bindings = _stage_map(
        root_fd,
        sealed_run_id,
        run_root,
        publications,
        transcript_registry,
        expected_stage,
        predecessor_emitted=cp4_emitted,
    )
    expected_final_tree = {
        FINAL_ROOT: "directory",
        FINAL_RESULTS_PATH: "regular",
        FINAL_RECONCILIATION_PATH: "regular",
    }
    if _walk_subtree(root_fd, FINAL_ROOT) != expected_final_tree:
        fail("CP5_TREE_INVALID", "final admission tree has extra/missing paths")

    final_records, _, _ = _load_jsonl(
        root_fd, FINAL_RESULTS_PATH, require_nonempty=True
    )
    result_ids = [row.get("result_id") for row in final_records]
    if (
        any(not isinstance(result_id, str) or not result_id for result_id in result_ids)
        or len(result_ids) != len(set(result_ids))
        or result_ids != sorted(result_ids)
    ):
        fail("CP5_FINAL_INVALID", "final results are not in canonical result_id order")
    cp3 = replay["cp3"]
    assert isinstance(cp3, dict)
    try:
        recomputed = reconcile_final_admission(
            cp3["inventory"],
            cp3["buckets"],
            cp3["summary"],
            final_records,
            replay["plan"],
            replay["ledger_rows"],
            replay["copied_payloads"],
            cp3["source_pre"],
            expected_result_count=expected_result_count,
        )
    except V6ContractError as exc:
        fail("CP5_FINAL_INVALID", str(exc))
    published_reconciliation, reconciliation_raw, _ = _load_object(
        root_fd, FINAL_RECONCILIATION_PATH
    )
    if reconciliation_raw != canonical_json(recomputed):
        fail("CP5_FINAL_INVALID", "published final reconciliation differs from recomputation")

    source_post, source_post_raw, _ = _load_jsonl(
        root_fd, SOURCE_POST_PATH, require_nonempty=True
    )
    try:
        validate_source_pre_records(source_post)
    except V6ContractError as exc:
        fail("CP5_MIGRATION_INVALID", f"source-post record set is invalid: {exc}")
    _, paper_post_raw, _ = _load_object(root_fd, PAPER_POST_PATH)
    if source_post_raw != cp3["source_pre_raw"] or paper_post_raw != cp3["paper_pre_raw"]:
        fail("CP5_MIGRATION_INVALID", "post manifests are not byte-exact CP0 baselines")
    source_pre_identity = cp3_inputs.source_pre.evidence.identity
    paper_pre_identity = cp3_inputs.paper_pre.evidence.identity
    source_post_identity = cp5_bindings[SOURCE_POST_PATH]["identity"]
    paper_post_identity = cp5_bindings[PAPER_POST_PATH]["identity"]
    if (
        (source_pre_identity.get("dev"), source_pre_identity.get("inode"))
        == (source_post_identity.get("dev"), source_post_identity.get("inode"))
        or (paper_pre_identity.get("dev"), paper_pre_identity.get("inode"))
        == (paper_post_identity.get("dev"), paper_post_identity.get("inode"))
    ):
        fail("CP5_MIGRATION_INVALID", "post manifests are not independently published")
    if published_reconciliation != recomputed:
        fail("CP5_FINAL_INVALID", "final reconciliation object differs")
    cp4_semantic = replay["semantic"]
    assert isinstance(cp4_semantic, dict)
    base = {
        "schema": CP5_SEMANTIC_SCHEMA,
        "sealed_run_id": sealed_run_id,
        "run_root": run_root,
        "checkpoint": 5,
        "predecessor": {
            "checkpoint": 4,
            "checkpoint_sha256": previous_checkpoint_sha256,
            "checkpoint_transcript_sha256": previous_checkpoint_transcript_sha256,
            "publication": cp4_binding,
        },
        "cp4_semantic_sha256": cp4_semantic["semantic_sha256"],
        "stage_publications": [cp5_bindings[path] for path in sorted(cp5_bindings)],
        "stage_publication_count": len(cp5_bindings),
        "migration": {
            "source_pre_sha256": cp3_inputs.source_pre.evidence.sha256,
            "source_post_sha256": cp5_bindings[SOURCE_POST_PATH]["artifact_sha256"],
            "paper_pre_sha256": cp3_inputs.paper_pre.evidence.sha256,
            "paper_post_sha256": cp5_bindings[PAPER_POST_PATH]["artifact_sha256"],
            "source_post_independent_publication": True,
            "paper_post_independent_publication": True,
        },
        "final_reconciliation": recomputed,
    }
    transcript_registry.verify_current()
    return {**base, "semantic_sha256": sha256_bytes(canonical_json(base))}


__all__ = [
    "CP3SemanticPublications",
    "CP4SemanticPublications",
    "CP4_SEMANTIC_SCHEMA",
    "CP5_SEMANTIC_SCHEMA",
    "validate_cp4_semantics",
    "validate_cp5_semantics",
]
