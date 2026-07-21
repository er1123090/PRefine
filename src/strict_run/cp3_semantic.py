"""Descriptor-bound semantic reconciliation for the strict V6 CP3 gate."""
from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterable, Mapping

from provenance.strict_v6 import (
    V6ContractError,
    digest_json as provenance_digest_json,
    digest_records as provenance_digest_records,
    result_inventory_digest,
    reconcile_cp3_buckets,
    validate_cp3_provenance,
    validate_source_pre_records,
)

from .canonical import (
    canonical_json,
    fail,
    require_sha256,
    sha256_bytes,
    validate_absolute_path_text,
    validate_sealed_run_id,
)
from .checkpoint import validate_checkpoint_payload
from .filesystem import open_relative_directory, read_regular_at, stable_identity
from .external import ExternalTranscriptRegistry
from .publication import (
    ArtifactEvidence,
    Publication,
    transcript_hash,
    validate_publication_transcript,
)
from .semantics import validate_cp2_artifact_structure, validate_stage_semantics
from .writer_policy import default_writer_policy


SEMANTIC_SCHEMA = "experiments7-cp3-semantic-binding/v6"
INVENTORY_PATH = "inventory/results.jsonl"
CP1_PATH = "checkpoints/cp1.json"
CP2_PATH = "checkpoints/cp2.json"
SOURCE_PRE_PATH = "manifests/source-pre.jsonl"
BUCKET_ROOT = "admission/precopy/buckets"
RECONCILIATION_PATH = "admission/precopy/reconciliation.json"
PROVENANCE_ROOT = "provenance"
PROVENANCE_NODES_PATH = "provenance/nodes.jsonl"
PROVENANCE_EDGES_PATH = "provenance/edges.jsonl"
PROVENANCE_RECOMPUTATIONS_PATH = "provenance/recomputations.jsonl"
PROVENANCE_ROUNDING_PATH = "provenance/rounding-proofs.jsonl"
PROVENANCE_PATHS = (
    PROVENANCE_NODES_PATH,
    PROVENANCE_EDGES_PATH,
    PROVENANCE_RECOMPUTATIONS_PATH,
    PROVENANCE_ROUNDING_PATH,
)
REQUIRED_STAGE_DIRECTORIES = frozenset(
    {PROVENANCE_ROOT, "admission", "admission/precopy", BUCKET_ROOT}
)
EXPECTED_RESULT_COUNT = 1908

_BUCKET_KEYS = {
    "schema",
    "phase",
    "paper_section_id",
    "inventory_sha256",
    "result_ids",
    "result_records",
    "result_records_sha256",
    "copy_required_raw_artifact_ids",
}
_RECONCILIATION_KEYS = {
    "schema",
    "phase",
    "checkpoint_state",
    "inventory_sha256",
    "total_result_count",
    "precopy_admitted_count",
    "precopy_unresolved_count",
    "paper_section_bucket_count",
    "bucket_subseal_sha256s",
    "copy_required_raw_artifact_ids",
    "copy_plan_input_sha256",
}


class _DuplicateJsonKey(ValueError):
    pass


def _object_from_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKey(key)
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON number: {value}")


def _decode_json(raw: bytes, relative: str) -> object:
    try:
        text = raw.decode("utf-8")
        return json.loads(
            text,
            object_pairs_hook=_object_from_pairs,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKey, ValueError) as exc:
        fail("CP3_JSON_INVALID", f"{relative} is not strict JSON: {exc}")


def _load_canonical_object(root_fd: int, relative: str) -> tuple[dict[str, object], bytes, os.stat_result]:
    raw, current = read_regular_at(root_fd, relative)
    value = _decode_json(raw, relative)
    if not isinstance(value, dict) or canonical_json(value) != raw:
        fail("CP3_NONCANONICAL_JSON", f"{relative} is not canonical JSON")
    return value, raw, current


def _load_canonical_jsonl_rows(
    root_fd: int,
    relative: str,
    *,
    require_nonempty: bool = True,
) -> tuple[list[dict[str, object]], bytes, os.stat_result]:
    raw, current = read_regular_at(root_fd, relative)
    if (require_nonempty and not raw) or (raw and not raw.endswith(b"\n")):
        fail("CP3_JSONL_INVALID", f"{relative} is empty or lacks final newline")
    records: list[dict[str, object]] = []
    for number, line in enumerate(raw.splitlines(keepends=True), start=1):
        if line == b"\n" or not line.endswith(b"\n"):
            fail("CP3_JSONL_INVALID", f"{relative} line {number} is empty or unterminated")
        value = _decode_json(line, f"{relative}:{number}")
        if not isinstance(value, dict) or canonical_json(value) != line:
            fail("CP3_JSONL_INVALID", f"{relative} line {number} is not canonical JSON")
        records.append(value)
    return records, raw, current


def _load_canonical_jsonl(root_fd: int, relative: str) -> tuple[list[dict[str, object]], bytes, os.stat_result]:
    records, raw, current = _load_canonical_jsonl_rows(root_fd, relative)
    if len(records) != EXPECTED_RESULT_COUNT:
        fail("CP3_INVENTORY_COUNT", "CP1 inventory must contain exactly 1908 records")
    try:
        result_inventory_digest(records)
    except V6ContractError as exc:
        fail("CP3_INVENTORY_INVALID", str(exc))
    return records, raw, current


def _require_regular_matches(
    relative: str,
    raw: bytes,
    current: os.stat_result,
    evidence: ArtifactEvidence,
    *,
    code: str = "CP3_EVIDENCE_MISMATCH",
) -> None:
    identity = stable_identity(current)
    if (
        evidence.relative_path != relative
        or evidence.artifact_type != "regular"
        or evidence.sha256 != sha256_bytes(raw)
        or evidence.size != len(raw)
        or evidence.identity != identity
        or identity.get("mode") != 0o444
    ):
        detail = (
            "CP1 inventory physical evidence is stale"
            if code == "CP3_INVENTORY_EVIDENCE"
            else f"{relative} no longer matches its descriptor-bound publication"
        )
        fail(code, detail)


def _validate_checkpoint_core(
    value: object,
    *,
    checkpoint: int,
    sealed_run_id: str,
    run_root: str,
) -> dict[str, object]:
    return validate_checkpoint_payload(
        value,
        expected_sealed_run_id=sealed_run_id,
        expected_run_root=run_root,
        expected_checkpoint=checkpoint,
        policy=default_writer_policy(),
    )


def _require_writer(evidence: ArtifactEvidence) -> None:
    path = evidence.relative_path
    if path == "provenance" or path.startswith("provenance/"):
        expected = ("provenance", "provenance:", "provenance-writer", "provenance")
    elif path in {"admission", "admission/precopy", BUCKET_ROOT} or path.startswith(
        "admission/precopy/"
    ):
        expected = ("admission", "admission:", "admission-writer", "admission")
    elif path == INVENTORY_PATH:
        expected = ("inventory", "inventory:", "inventory-writer", "inventory")
    else:
        fail("CP3_STAGE_PATH_INVALID", "CP3 publication is outside provenance/precopy scope")
    role, task_prefix, writer_id, rule_id = expected
    if (
        evidence.writer.role != role
        or not evidence.writer.task_id.startswith(task_prefix)
        or evidence.writer.writer_id != writer_id
        or evidence.writer_rule_id != rule_id
    ):
        fail("CP3_WRITER_INVALID", f"writer evidence differs for {path}")


def _require_checkpoint_writer(evidence: ArtifactEvidence, relative: str) -> None:
    writer = evidence.writer
    if (
        evidence.relative_path != relative
        or evidence.artifact_type != "regular"
        or writer.role != "checkpoint_controller"
        or not writer.task_id.startswith("checkpoint:")
        or writer.writer_id != "checkpoint_controller-writer"
        or evidence.writer_rule_id != "checkpoints"
    ):
        fail("CP3_PREDECESSOR_WRITER_INVALID", f"writer evidence differs for {relative}")


def _validate_checkpoint_publication(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    relative: str,
    publication: Publication,
    transcript_registry: ExternalTranscriptRegistry,
    raw: bytes,
    current: os.stat_result,
) -> tuple[dict[str, object], dict[str, object]]:
    if not isinstance(publication, Publication):
        fail("CP3_PREDECESSOR_INVALID", f"{relative} is not a typed Publication")
    evidence = publication.evidence
    if ArtifactEvidence.from_dict(evidence.to_dict()) != evidence:
        fail("CP3_PREDECESSOR_INVALID", f"{relative} evidence does not round-trip")
    _require_checkpoint_writer(evidence, relative)
    external = transcript_registry.get_exact(evidence.publication_transcript_sha256)
    if external != publication.transcript:
        fail("CP3_PREDECESSOR_INVALID", f"{relative} transcript is not the registered transcript")
    transcript = validate_publication_transcript(
        root_fd,
        sealed_run_id,
        run_root,
        external,
        expected_relative_path=relative,
        expected_writer=evidence.writer,
    )
    if transcript_hash(transcript) != evidence.publication_transcript_sha256:
        fail("CP3_PREDECESSOR_INVALID", f"{relative} transcript digest differs")
    _require_regular_matches(
        relative,
        raw,
        current,
        evidence,
        code="CP3_PREDECESSOR_INVALID",
    )
    first_event = transcript["events"][0]
    assert isinstance(first_event, dict)
    binding = {
        "relative_path": relative,
        "artifact_sha256": evidence.sha256,
        "bytes": evidence.size,
        "identity": evidence.identity,
        "writer": evidence.writer.to_dict(),
        "writer_rule_id": evidence.writer_rule_id,
        "publication_transcript_sha256": evidence.publication_transcript_sha256,
        "physical_evidence_sha256": sha256_bytes(canonical_json(evidence.to_dict())),
    }
    return binding, transcript


def _reconstruct_checkpoint_stage_publications(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    checkpoint: Mapping[str, object],
    checkpoint_transcript: Mapping[str, object],
    transcript_registry: ExternalTranscriptRegistry,
    *,
    predecessor_emitted_monotonic_ns: int | None = None,
) -> list[Publication]:
    stage = checkpoint["stage_actual_paths"]
    assert isinstance(stage, list)
    checkpoint_events = checkpoint_transcript.get("events")
    if not isinstance(checkpoint_events, list) or not checkpoint_events:
        fail("CP3_PREDECESSOR_STAGE_INVALID", "checkpoint transcript events are empty")
    first_checkpoint_event = checkpoint_events[0]
    if not isinstance(first_checkpoint_event, dict) or type(
        first_checkpoint_event.get("completed_monotonic_ns")
    ) is not int:
        fail("CP3_PREDECESSOR_STAGE_INVALID", "checkpoint chronology is invalid")
    checkpoint_started = int(first_checkpoint_event["completed_monotonic_ns"])
    if (
        predecessor_emitted_monotonic_ns is not None
        and type(predecessor_emitted_monotonic_ns) is not int
    ):
        fail(
            "CP3_PREDECESSOR_STAGE_INVALID",
            "predecessor publication chronology is invalid",
        )
    publications: list[Publication] = []
    for raw_evidence in stage:
        evidence = ArtifactEvidence.from_dict(raw_evidence)
        external = transcript_registry.get_exact(
            evidence.publication_transcript_sha256
        )
        if not isinstance(external, dict):
            fail(
                "CP3_PREDECESSOR_STAGE_INVALID",
                "registered stage transcript is not an object",
            )
        transcript = validate_publication_transcript(
            root_fd,
            sealed_run_id,
            run_root,
            external,
            expected_relative_path=evidence.relative_path,
            expected_writer=evidence.writer,
        )
        events = transcript.get("events")
        if not isinstance(events, list) or not events or not isinstance(events[0], dict):
            fail(
                "CP3_PREDECESSOR_STAGE_INVALID",
                "predecessor stage transcript events are invalid",
            )
        first_stage_event = events[0]
        first_stage_completed = first_stage_event.get("completed_monotonic_ns")
        if (
            transcript_hash(transcript)
            != evidence.publication_transcript_sha256
            or transcript.get("artifact_type") != evidence.artifact_type
            or transcript.get("sha256") != evidence.sha256
            or transcript.get("bytes") != evidence.size
            or transcript.get("identity") != evidence.identity
            or type(transcript.get("emitted_monotonic_ns")) is not int
            or type(first_stage_completed) is not int
            or int(transcript["emitted_monotonic_ns"]) >= checkpoint_started
            or (
                predecessor_emitted_monotonic_ns is not None
                and int(first_stage_completed)
                <= predecessor_emitted_monotonic_ns
            )
        ):
            fail(
                "CP3_PREDECESSOR_STAGE_INVALID",
                "predecessor stage transcript/evidence/chronology differs",
            )
        publications.append(Publication(evidence, transcript))
    return publications


def _validate_predecessor_semantics(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    cp1: Mapping[str, object],
    cp1_transcript: Mapping[str, object],
    cp2: Mapping[str, object],
    cp2_transcript: Mapping[str, object],
    transcript_registry: ExternalTranscriptRegistry,
    cp1_stage_predecessor_emitted_monotonic_ns: int,
    cp2_stage_predecessor_emitted_monotonic_ns: int,
) -> dict[str, object]:
    cp1_stage = _reconstruct_checkpoint_stage_publications(
        root_fd,
        sealed_run_id,
        run_root,
        cp1,
        cp1_transcript,
        transcript_registry,
        predecessor_emitted_monotonic_ns=(
            cp1_stage_predecessor_emitted_monotonic_ns
        ),
    )
    cp1_semantics = validate_stage_semantics(
        root_fd,
        sealed_run_id,
        run_root,
        1,
        cp1_stage,
        cp1["previous_checkpoint_sha256"],
        cp1["previous_checkpoint_transcript_sha256"],
    )
    if (
        cp1.get("semantic_bindings") != cp1_semantics
        or cp1.get("semantic_bindings_sha256")
        != sha256_bytes(canonical_json(cp1_semantics))
    ):
        fail("CP3_PREDECESSOR_SEMANTICS_INVALID", "CP1 semantics differ")
    if (
        cp1_semantics.get("lineage_count") != 129
        or cp1_semantics.get("profile_count") != 30
        or cp1_semantics.get("inventory_record_count") != EXPECTED_RESULT_COUNT
    ):
        fail("CP3_PREDECESSOR_SEMANTICS_INVALID", "CP1 exact counts differ")

    cp2_stage = _reconstruct_checkpoint_stage_publications(
        root_fd,
        sealed_run_id,
        run_root,
        cp2,
        cp2_transcript,
        transcript_registry,
        predecessor_emitted_monotonic_ns=(
            cp2_stage_predecessor_emitted_monotonic_ns
        ),
    )
    cp2_semantics = validate_cp2_artifact_structure(
        root_fd,
        sealed_run_id,
        run_root,
        cp2_stage,
        cp2["previous_checkpoint_sha256"],
        cp2["previous_checkpoint_transcript_sha256"],
    )
    if (
        cp2.get("semantic_bindings") != cp2_semantics
        or cp2.get("semantic_bindings_sha256")
        != sha256_bytes(canonical_json(cp2_semantics))
    ):
        fail("CP3_PREDECESSOR_SEMANTICS_INVALID", "CP2 semantics differ")
    seals = cp2_semantics.get("seals")
    if (
        cp2_semantics.get("lineage_count") != 129
        or cp2_semantics.get("profile_count") != 30
        or not isinstance(seals, list)
        or len(seals) != 3
    ):
        fail("CP3_PREDECESSOR_SEMANTICS_INVALID", "CP2 exact counts/seals differ")
    return {
        "cp1": cp1_semantics,
        "cp1_semantics_sha256": sha256_bytes(canonical_json(cp1_semantics)),
        "cp2": cp2_semantics,
        "cp2_semantics_sha256": sha256_bytes(canonical_json(cp2_semantics)),
    }


def _validate_source_pre_publication(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    publication: Publication,
    transcript_registry: ExternalTranscriptRegistry,
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    if not isinstance(publication, Publication):
        fail("CP3_SOURCE_PRE_INVALID", "source-pre is not a typed Publication")
    evidence = publication.evidence
    if ArtifactEvidence.from_dict(evidence.to_dict()) != evidence:
        fail("CP3_SOURCE_PRE_INVALID", "source-pre evidence does not round-trip")
    writer = evidence.writer
    if (
        evidence.relative_path != SOURCE_PRE_PATH
        or evidence.artifact_type != "regular"
        or writer.role != "g0"
        or not writer.task_id.startswith("g0:")
        or writer.writer_id != "g0-writer"
        or evidence.writer_rule_id != "manifests"
    ):
        fail("CP3_SOURCE_PRE_INVALID", "source-pre Publication writer/path differs")
    external = transcript_registry.get_exact(evidence.publication_transcript_sha256)
    if external != publication.transcript:
        fail(
            "CP3_SOURCE_PRE_INVALID",
            "source-pre transcript is not the exact registered transcript",
        )
    transcript = validate_publication_transcript(
        root_fd,
        sealed_run_id,
        run_root,
        external,
        expected_relative_path=SOURCE_PRE_PATH,
        expected_writer=writer,
    )
    if transcript_hash(transcript) != evidence.publication_transcript_sha256:
        fail("CP3_SOURCE_PRE_INVALID", "source-pre transcript digest differs")
    rows, raw, current = _load_canonical_jsonl_rows(root_fd, SOURCE_PRE_PATH)
    _require_regular_matches(
        SOURCE_PRE_PATH,
        raw,
        current,
        evidence,
        code="CP3_SOURCE_PRE_INVALID",
    )
    try:
        records = validate_source_pre_records(rows)
    except V6ContractError as exc:
        fail("CP3_SOURCE_PRE_INVALID", str(exc))
    first_event = transcript["events"][0]
    assert isinstance(first_event, dict)
    binding = {
        "relative_path": SOURCE_PRE_PATH,
        "artifact_sha256": evidence.sha256,
        "bytes": evidence.size,
        "identity": evidence.identity,
        "writer": writer.to_dict(),
        "writer_rule_id": evidence.writer_rule_id,
        "publication_transcript_sha256": evidence.publication_transcript_sha256,
        "physical_evidence_sha256": sha256_bytes(canonical_json(evidence.to_dict())),
        "record_count": len(records),
        "records_sha256": provenance_digest_records(records),
        "publication_first_completed_monotonic_ns": first_event[
            "completed_monotonic_ns"
        ],
        "publication_emitted_monotonic_ns": transcript["emitted_monotonic_ns"],
    }
    return records, binding, transcript


def _validate_current_publication(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    publication: Publication,
    transcript_registry: ExternalTranscriptRegistry,
) -> dict[str, object]:
    if not isinstance(publication, Publication):
        fail("CP3_PUBLICATION_INVALID", "CP3 stage contains a non-Publication value")
    evidence = publication.evidence
    if ArtifactEvidence.from_dict(evidence.to_dict()) != evidence:
        fail("CP3_PUBLICATION_INVALID", "publication evidence does not round-trip")
    external = transcript_registry.get_exact(evidence.publication_transcript_sha256)
    if external != publication.transcript:
        fail(
            "CP3_TRANSCRIPT_MISMATCH",
            "CP3 publication transcript is not the exact registered transcript",
        )
    _require_writer(evidence)
    transcript = validate_publication_transcript(
        root_fd,
        sealed_run_id,
        run_root,
        external,
        expected_relative_path=evidence.relative_path,
        expected_writer=evidence.writer,
    )
    if transcript_hash(transcript) != evidence.publication_transcript_sha256:
        fail("CP3_TRANSCRIPT_MISMATCH", "publication transcript digest differs")
    if (
        transcript["artifact_type"] != evidence.artifact_type
        or transcript["sha256"] != evidence.sha256
        or transcript["bytes"] != evidence.size
        or transcript["identity"] != evidence.identity
    ):
        fail("CP3_EVIDENCE_MISMATCH", "publication transcript and evidence differ")
    first_event = transcript["events"][0]
    assert isinstance(first_event, dict)
    return {
        "relative_path": evidence.relative_path,
        "artifact_type": evidence.artifact_type,
        "artifact_sha256": evidence.sha256,
        "bytes": evidence.size,
        "identity": evidence.identity,
        "writer": evidence.writer.to_dict(),
        "writer_rule_id": evidence.writer_rule_id,
        "publication_transcript_sha256": evidence.publication_transcript_sha256,
        "physical_evidence_sha256": sha256_bytes(canonical_json(evidence.to_dict())),
        "publication_first_completed_monotonic_ns": first_event["completed_monotonic_ns"],
        "publication_emitted_monotonic_ns": transcript["emitted_monotonic_ns"],
    }


def _validate_inventory_evidence(
    root_fd: int,
    cp1: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, object], str]:
    stage = cp1["stage_actual_paths"]
    assert isinstance(stage, list)
    evidence_rows = [ArtifactEvidence.from_dict(item) for item in stage]
    matches = [item for item in evidence_rows if item.relative_path == INVENTORY_PATH]
    if len(matches) != 1:
        fail("CP3_INVENTORY_EVIDENCE", "CP1 must bind inventory/results.jsonl exactly once")
    evidence = matches[0]
    _require_writer(evidence)
    if evidence.artifact_type != "regular":
        fail("CP3_INVENTORY_EVIDENCE", "CP1 inventory evidence is not regular")
    records, raw, current = _load_canonical_jsonl(root_fd, INVENTORY_PATH)
    _require_regular_matches(
        INVENTORY_PATH,
        raw,
        current,
        evidence,
        code="CP3_INVENTORY_EVIDENCE",
    )
    inv_digest = result_inventory_digest(records)
    binding = {
        "relative_path": INVENTORY_PATH,
        "artifact_sha256": evidence.sha256,
        "bytes": evidence.size,
        "identity": evidence.identity,
        "writer": evidence.writer.to_dict(),
        "writer_rule_id": evidence.writer_rule_id,
        "publication_transcript_sha256": evidence.publication_transcript_sha256,
        "physical_evidence_sha256": sha256_bytes(canonical_json(evidence.to_dict())),
        "inventory_sha256": inv_digest,
        "result_count": len(records),
    }
    return records, binding, inv_digest


def _checkpoint_artifact_bindings(
    root_fd: int,
    checkpoints: Iterable[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    evidence_by_path: dict[str, ArtifactEvidence] = {}
    for checkpoint in checkpoints:
        stage = checkpoint["stage_actual_paths"]
        assert isinstance(stage, list)
        for item in stage:
            evidence = ArtifactEvidence.from_dict(item)
            if evidence.relative_path in evidence_by_path:
                fail(
                    "CP3_PREDECESSOR_INVALID",
                    "CP1/CP2 stage paths overlap or repeat",
                )
            evidence_by_path[evidence.relative_path] = evidence

    bindings: dict[str, dict[str, object]] = {}
    for relative, evidence in sorted(evidence_by_path.items()):
        if evidence.artifact_type != "regular":
            continue
        raw, current = read_regular_at(root_fd, relative)
        _require_regular_matches(
            relative,
            raw,
            current,
            evidence,
            code="CP3_PREDECESSOR_INVALID",
        )
        json_value: object | None = None
        try:
            decoded = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_object_from_pairs,
                parse_constant=_reject_constant,
            )
            if canonical_json(decoded) == raw:
                json_value = decoded
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            _DuplicateJsonKey,
            ValueError,
        ):
            pass
        bindings[relative] = {
            "evidence_sha256": sha256_bytes(canonical_json(evidence.to_dict())),
            "json_value": json_value,
        }
    return bindings


def _actual_bucket_paths(root_fd: int) -> list[str]:
    try:
        directory_fd = open_relative_directory(root_fd, BUCKET_ROOT)
    except OSError as exc:
        fail("CP3_BUCKET_SET_INVALID", f"bucket directory is unavailable: {exc}")
    try:
        paths: list[str] = []
        for name in sorted(os.listdir(directory_fd), key=os.fsencode):
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if (
                not isinstance(name, str)
                or not name
                or not name.endswith(".json")
                or not stat.S_ISREG(current.st_mode)
            ):
                fail("CP3_BUCKET_SET_INVALID", "bucket directory contains a non-JSON regular entry")
            paths.append(f"{BUCKET_ROOT}/{name}")
        if not paths:
            fail("CP3_BUCKET_SET_INVALID", "CP3 has no admission buckets")
        return paths
    finally:
        os.close(directory_fd)


def _require_exact_directory_entries(
    root_fd: int,
    relative: str,
    expected: Mapping[str, str],
) -> None:
    try:
        directory_fd = open_relative_directory(root_fd, relative)
    except OSError as exc:
        fail("CP3_STAGE_PATH_INVALID", f"{relative} directory is unavailable: {exc}")
    try:
        actual: dict[str, str] = {}
        for name in os.listdir(directory_fd):
            if not isinstance(name, str) or not name or "\x00" in name:
                fail("CP3_STAGE_PATH_INVALID", f"{relative} contains an invalid entry")
            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(current.st_mode):
                entry_type = "directory"
            elif stat.S_ISREG(current.st_mode):
                entry_type = "regular"
            else:
                fail(
                    "CP3_STAGE_PATH_INVALID",
                    f"{relative}/{name} is not a regular file or directory",
                )
            actual[name] = entry_type
        if actual != dict(expected):
            fail(
                "CP3_STAGE_PATH_INVALID",
                f"{relative} entries are not the exact CP3 set",
            )
    finally:
        os.close(directory_fd)


def _validate_reconciliation_shape(value: Mapping[str, object]) -> None:
    if set(value) != _RECONCILIATION_KEYS:
        fail("CP3_RECONCILIATION_INVALID", "reconciliation keys differ")
    for field in (
        "total_result_count",
        "precopy_admitted_count",
        "precopy_unresolved_count",
        "paper_section_bucket_count",
    ):
        if type(value.get(field)) is not int or int(value[field]) < 0:
            fail("CP3_RECONCILIATION_INVALID", f"{field} must be an exact nonnegative integer")


def validate_cp3_semantics(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    publications: Iterable[Publication],
    cp1_publication: Publication,
    cp2_publication: Publication,
    source_pre_publication: Publication,
    transcript_registry: ExternalTranscriptRegistry,
    previous_checkpoint_sha256: str,
    previous_checkpoint_transcript_sha256: str,
) -> dict[str, object]:
    """Recompute and bind CP3 from current CP1 bytes and current CP3 publications."""

    if type(root_fd) is not int or root_fd < 0 or not stat.S_ISDIR(os.fstat(root_fd).st_mode):
        fail("CP3_ROOT_INVALID", "root_fd must be an open directory descriptor")
    sealed_run_id = validate_sealed_run_id(sealed_run_id)
    run_root = validate_absolute_path_text(run_root, "run_root")
    if run_root.rsplit("/", 1)[-1] != sealed_run_id:
        fail("CP3_ROOT_INVALID", "run_root is not the exact sealed_run_id child")
    descriptor_path = os.readlink(f"/proc/self/fd/{root_fd}")
    if descriptor_path.endswith(" (deleted)") or descriptor_path != run_root or os.path.realpath(run_root) != run_root:
        fail("CP3_ROOT_INVALID", "root descriptor does not match canonical run_root")
    previous_checkpoint_sha256 = require_sha256(
        previous_checkpoint_sha256, "previous checkpoint sha256"
    )
    previous_checkpoint_transcript_sha256 = require_sha256(
        previous_checkpoint_transcript_sha256, "previous checkpoint transcript sha256"
    )
    if not isinstance(transcript_registry, ExternalTranscriptRegistry):
        fail(
            "EXTERNAL_REGISTRY_INVALID",
            "CP3 requires a typed external transcript registry",
        )
    transcript_registry.verify_current()

    cp1, cp1_raw, cp1_current = _load_canonical_object(root_fd, CP1_PATH)
    if stable_identity(cp1_current).get("mode") != 0o444:
        fail("CP3_CHECKPOINT_INVALID", "current CP1 checkpoint is not immutable mode 0444")
    cp1 = _validate_checkpoint_core(
        cp1, checkpoint=1, sealed_run_id=sealed_run_id, run_root=run_root
    )
    cp2, cp2_raw, cp2_current = _load_canonical_object(root_fd, CP2_PATH)
    if stable_identity(cp2_current).get("mode") != 0o444:
        fail("CP3_CHECKPOINT_INVALID", "current CP2 checkpoint is not immutable mode 0444")
    cp2 = _validate_checkpoint_core(
        cp2, checkpoint=2, sealed_run_id=sealed_run_id, run_root=run_root
    )
    cp1_physical, cp1_transcript = _validate_checkpoint_publication(
        root_fd,
        sealed_run_id,
        run_root,
        CP1_PATH,
        cp1_publication,
        transcript_registry,
        cp1_raw,
        cp1_current,
    )
    cp2_physical, cp2_transcript = _validate_checkpoint_publication(
        root_fd,
        sealed_run_id,
        run_root,
        CP2_PATH,
        cp2_publication,
        transcript_registry,
        cp2_raw,
        cp2_current,
    )
    source_pre_records, source_pre_binding, source_pre_transcript = (
        _validate_source_pre_publication(
            root_fd,
            sealed_run_id,
            run_root,
            source_pre_publication,
            transcript_registry,
        )
    )
    cp1_sha256 = sha256_bytes(cp1_raw)
    if cp2.get("previous_checkpoint_sha256") != cp1_sha256:
        fail("CP3_PREDECESSOR_INVALID", "CP2 does not bind the current CP1 bytes")
    cp1_transcript_sha256 = require_sha256(
        cp2.get("previous_checkpoint_transcript_sha256"),
        "CP1 checkpoint transcript sha256",
    )
    if cp1_transcript_sha256 != cp1_publication.evidence.publication_transcript_sha256:
        fail("CP3_PREDECESSOR_INVALID", "CP2 does not bind the registered CP1 transcript")
    if sha256_bytes(cp2_raw) != previous_checkpoint_sha256:
        fail("CP3_PREDECESSOR_INVALID", "CP3 predecessor hash is not current CP2")
    if previous_checkpoint_sha256 != cp2_publication.evidence.sha256:
        fail("CP3_PREDECESSOR_INVALID", "CP3 predecessor Publication hash differs")
    if (
        previous_checkpoint_transcript_sha256
        != cp2_publication.evidence.publication_transcript_sha256
    ):
        fail("CP3_PREDECESSOR_INVALID", "CP3 predecessor transcript is not current CP2")
    cp1_emitted = cp1_transcript["emitted_monotonic_ns"]
    cp2_first_event = cp2_transcript["events"][0]
    cp1_first_event = cp1_transcript["events"][0]
    source_pre_emitted = source_pre_transcript["emitted_monotonic_ns"]
    assert type(cp1_emitted) is int and isinstance(cp2_first_event, dict)
    assert isinstance(cp1_first_event, dict) and type(source_pre_emitted) is int
    if source_pre_emitted >= cp1_first_event["completed_monotonic_ns"]:
        fail("CP3_SOURCE_PRE_INVALID", "source-pre publication does not predate CP1")
    if cp1_emitted >= cp2_first_event["completed_monotonic_ns"]:
        fail("CP3_PREDECESSOR_INVALID", "CP1/CP2 publication chronology is invalid")
    predecessor_semantics = _validate_predecessor_semantics(
        root_fd,
        sealed_run_id,
        run_root,
        cp1,
        cp1_transcript,
        cp2,
        cp2_transcript,
        transcript_registry,
        source_pre_emitted,
        cp1_emitted,
    )

    inventory, inventory_binding, inv_digest = _validate_inventory_evidence(root_fd, cp1)
    artifact_bindings = _checkpoint_artifact_bindings(root_fd, (cp1, cp2))

    stage = list(publications)
    if not stage:
        fail("CP3_STAGE_INVALID", "CP3 stage publications are empty")
    physical_by_path: dict[str, dict[str, object]] = {}
    publication_by_path: dict[str, Publication] = {}
    for publication in stage:
        physical = _validate_current_publication(
            root_fd,
            sealed_run_id,
            run_root,
            publication,
            transcript_registry,
        )
        relative = str(physical["relative_path"])
        if relative in physical_by_path:
            fail("CP3_STAGE_INVALID", "CP3 stage contains a duplicate publication path")
        physical_by_path[relative] = physical
        publication_by_path[relative] = publication

    _require_exact_directory_entries(
        root_fd,
        PROVENANCE_ROOT,
        {path.rsplit("/", 1)[-1]: "regular" for path in PROVENANCE_PATHS},
    )
    _require_exact_directory_entries(root_fd, "admission", {"precopy": "directory"})
    _require_exact_directory_entries(
        root_fd,
        "admission/precopy",
        {"buckets": "directory", "reconciliation.json": "regular"},
    )
    current_bucket_paths = _actual_bucket_paths(root_fd)
    expected_stage_paths = (
        set(REQUIRED_STAGE_DIRECTORIES)
        | set(PROVENANCE_PATHS)
        | {RECONCILIATION_PATH}
        | set(current_bucket_paths)
    )
    if set(publication_by_path) != expected_stage_paths:
        fail(
            "CP3_STAGE_PATH_INVALID",
            "CP3 stage paths are not exactly the required directories, provenance, buckets, and reconciliation",
        )
    cp2_emitted = cp2_transcript["emitted_monotonic_ns"]
    assert type(cp2_emitted) is int
    for relative in sorted(expected_stage_paths):
        publication = publication_by_path[relative]
        expected_type = "directory" if relative in REQUIRED_STAGE_DIRECTORIES else "regular"
        if publication.evidence.artifact_type != expected_type:
            fail("CP3_STAGE_PATH_INVALID", f"CP3 stage artifact type differs at {relative}")
        if physical_by_path[relative]["publication_first_completed_monotonic_ns"] <= cp2_emitted:
            fail("CP3_PREDECESSOR_INVALID", "a CP3 publication predates completed CP2")

    published_bucket_paths = sorted(
        path for path in publication_by_path if path.startswith(f"{BUCKET_ROOT}/")
    )
    if published_bucket_paths != current_bucket_paths:
        fail("CP3_BUCKET_SET_INVALID", "published bucket paths differ from the exact current set")

    graph_rows: dict[str, list[dict[str, object]]] = {}
    graph_bindings: list[dict[str, object]] = []
    for relative in PROVENANCE_PATHS:
        rows, raw, current = _load_canonical_jsonl_rows(
            root_fd, relative, require_nonempty=False
        )
        _require_regular_matches(
            relative,
            raw,
            current,
            publication_by_path[relative].evidence,
        )
        graph_rows[relative] = rows
        graph_bindings.append(
            {
                **physical_by_path[relative],
                "canonical_file_sha256": sha256_bytes(raw),
                "record_count": len(rows),
            }
        )

    buckets: list[dict[str, object]] = []
    admission_records: list[dict[str, object]] = []
    bucket_bindings: list[dict[str, object]] = []
    for relative in published_bucket_paths:
        bucket, raw, current = _load_canonical_object(root_fd, relative)
        _require_regular_matches(
            relative,
            raw,
            current,
            publication_by_path[relative].evidence,
        )
        if set(bucket) != _BUCKET_KEYS:
            fail("CP3_BUCKET_SCHEMA_INVALID", f"bucket keys differ at {relative}")
        section = bucket.get("paper_section_id")
        if not isinstance(section, str) or not section:
            fail("CP3_BUCKET_SCHEMA_INVALID", "bucket paper_section_id is invalid")
        if relative != f"{BUCKET_ROOT}/{section}.json":
            fail(
                "CP3_BUCKET_SET_INVALID",
                "bucket filename is not the exact paper_section_id",
            )
        rows = bucket.get("result_records")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            fail("CP3_BUCKET_SCHEMA_INVALID", "bucket result_records are invalid")
        admission_records.extend(rows)
        buckets.append(bucket)
        bucket_bindings.append(
            {
                **physical_by_path[relative],
                "paper_section_id": section,
                "canonical_bucket_sha256": provenance_digest_json(bucket),
                "canonical_file_sha256": sha256_bytes(raw),
            }
        )

    try:
        recomputed = reconcile_cp3_buckets(
            inventory,
            buckets,
            expected_result_count=EXPECTED_RESULT_COUNT,
        )
    except V6ContractError as exc:
        fail("CP3_RECONCILIATION_INVALID", str(exc))
    if recomputed["inventory_sha256"] != inv_digest:
        fail("CP3_RECONCILIATION_INVALID", "recomputed inventory digest differs")

    try:
        provenance_reconciliation = validate_cp3_provenance(
            inventory,
            admission_records,
            graph_rows[PROVENANCE_NODES_PATH],
            graph_rows[PROVENANCE_EDGES_PATH],
            graph_rows[PROVENANCE_RECOMPUTATIONS_PATH],
            graph_rows[PROVENANCE_ROUNDING_PATH],
            source_pre_records,
            artifact_bindings,
        )
    except V6ContractError as exc:
        fail("CP3_PROVENANCE_INVALID", str(exc))

    published_reconciliation, reconciliation_raw, reconciliation_current = _load_canonical_object(
        root_fd, RECONCILIATION_PATH
    )
    _require_regular_matches(
        RECONCILIATION_PATH,
        reconciliation_raw,
        reconciliation_current,
        publication_by_path[RECONCILIATION_PATH].evidence,
    )
    _validate_reconciliation_shape(published_reconciliation)
    if reconciliation_raw != canonical_json(recomputed):
        fail("CP3_RECONCILIATION_INVALID", "published reconciliation differs from recomputation")
    transcript_registry.verify_current()

    base: dict[str, object] = {
        "schema": SEMANTIC_SCHEMA,
        "sealed_run_id": sealed_run_id,
        "run_root": run_root,
        "checkpoint": 3,
        "predecessor": {
            "checkpoint": 2,
            "checkpoint_sha256": previous_checkpoint_sha256,
            "checkpoint_transcript_sha256": previous_checkpoint_transcript_sha256,
            "cp1_publication": cp1_physical,
            "cp2_publication": cp2_physical,
            "semantics": predecessor_semantics,
        },
        "cp1_inventory": {
            "checkpoint_sha256": cp1_sha256,
            "checkpoint_transcript_sha256": cp1_transcript_sha256,
            **inventory_binding,
        },
        "source_pre": source_pre_binding,
        "stage_publications": [
            physical_by_path[path] for path in sorted(physical_by_path)
        ],
        "stage_publication_count": len(physical_by_path),
        "provenance": {
            "publications": graph_bindings,
            "reconciliation": provenance_reconciliation,
        },
        "bucket_publications": bucket_bindings,
        "bucket_publication_count": len(bucket_bindings),
        "reconciliation": {
            **physical_by_path[RECONCILIATION_PATH],
            "canonical_file_sha256": sha256_bytes(reconciliation_raw),
            "recomputed_sha256": sha256_bytes(canonical_json(recomputed)),
            "total_result_count": recomputed["total_result_count"],
            "precopy_admitted_count": recomputed["precopy_admitted_count"],
            "precopy_unresolved_count": recomputed["precopy_unresolved_count"],
            "copy_plan_input_sha256": recomputed["copy_plan_input_sha256"],
        },
    }
    return {**base, "semantic_sha256": sha256_bytes(canonical_json(base))}
