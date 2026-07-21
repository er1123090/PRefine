"""Fail-closed semantic validation for strict-run stage publications."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from facade.sandbox import (
    SandboxUnavailable,
    require_strict_runtime_proof_environment,
)
from provenance.strict_v6 import (
    CP1_ALIAS_COUNT,
    CP1_RESULT_COUNT,
    CP1_STRUCTURE_COUNT,
    V6ContractError,
    inventory_digest,
)

from .canonical import (
    canonical_json,
    fail,
    require_exact_keys,
    require_sha256,
    sha256_bytes,
    strict_json_loads,
    validate_absolute_path_text,
    validate_relative_path,
    validate_sealed_run_id,
)
from .filesystem import hash_regular_at, read_regular_at, stable_identity
from .publication import (
    ArtifactEvidence,
    Publication,
    transcript_hash,
    validate_publication_transcript,
)
from .writer_policy import WriterIdentity


EVIDENCE_REF_KEYS = {"relative_path", "artifact_evidence_sha256"}
CP6_SCHEMA = "experiments7-cp6-assurance-report/v6"
CP6_RESULT_SCHEMA = "experiments7-cp6-semantics/v6"


@dataclass(frozen=True)
class _ValidatedPublications:
    by_path: dict[str, ArtifactEvidence]
    publications: dict[str, Publication]

    @property
    def regular_paths(self) -> set[str]:
        return {
            path
            for path, evidence in self.by_path.items()
            if evidence.artifact_type == "regular"
        }


def _exact_int(value: object, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        fail("SEMANTICS_SCHEMA_INVALID", f"{field} must be an exact integer")
    return value


def _evidence_digest(evidence: ArtifactEvidence) -> str:
    return sha256_bytes(canonical_json(evidence.to_dict()))


def _evidence_ref(evidence: ArtifactEvidence) -> dict[str, str]:
    return {
        "relative_path": evidence.relative_path,
        "artifact_evidence_sha256": _evidence_digest(evidence),
    }


def _validate_publications(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    publications: Iterable[Publication],
) -> _ValidatedPublications:
    by_path: dict[str, ArtifactEvidence] = {}
    publication_by_path: dict[str, Publication] = {}
    for publication in publications:
        if not isinstance(publication, Publication):
            fail("SEMANTICS_PUBLICATION_INVALID", "stage publication has invalid type")
        evidence = ArtifactEvidence.from_dict(publication.evidence.to_dict())
        if evidence != publication.evidence:
            fail("SEMANTICS_PUBLICATION_INVALID", "artifact evidence did not round-trip")
        path = evidence.relative_path
        if path in by_path:
            fail("SEMANTICS_DUPLICATE_PATH", "stage publication paths must be unique")
        transcript = validate_publication_transcript(
            root_fd,
            sealed_run_id,
            run_root,
            publication.transcript,
            expected_relative_path=path,
            expected_writer=evidence.writer,
        )
        if transcript_hash(transcript) != evidence.publication_transcript_sha256:
            fail("SEMANTICS_TRANSCRIPT_MISMATCH", "publication transcript hash differs")
        if (
            transcript["artifact_type"] != evidence.artifact_type
            or transcript["sha256"] != evidence.sha256
            or transcript["bytes"] != evidence.size
            or transcript["identity"] != evidence.identity
        ):
            fail("SEMANTICS_EVIDENCE_MISMATCH", "publication evidence differs from transcript")
        by_path[path] = evidence
        publication_by_path[path] = publication
    return _ValidatedPublications(by_path, publication_by_path)


def _resolve_ref(
    value: object,
    evidence_by_path: dict[str, ArtifactEvidence],
    field: str,
) -> ArtifactEvidence:
    row = require_exact_keys(value, EVIDENCE_REF_KEYS, field)
    path = validate_relative_path(row["relative_path"])
    digest = require_sha256(
        row["artifact_evidence_sha256"], f"{field}.artifact_evidence_sha256"
    )
    evidence = evidence_by_path.get(path)
    if evidence is None or evidence.artifact_type != "regular":
        fail("SEMANTICS_EVIDENCE_REF_INVALID", f"{field} is not a regular publication")
    if _evidence_digest(evidence) != digest:
        fail("SEMANTICS_EVIDENCE_REF_INVALID", f"{field} evidence digest differs")
    return evidence


def _read_canonical_json(root_fd: int, evidence: ArtifactEvidence, schema: str) -> dict[str, object]:
    if evidence.artifact_type != "regular":
        fail("SEMANTICS_SCHEMA_INVALID", f"{schema} must be a regular artifact")
    raw, current = read_regular_at(root_fd, evidence.relative_path)
    if len(raw) != evidence.size or sha256_bytes(raw) != evidence.sha256:
        fail("SEMANTICS_EVIDENCE_STALE", f"{schema} bytes differ from evidence")
    if stable_identity(current) != evidence.identity:
        fail("SEMANTICS_EVIDENCE_STALE", f"{schema} identity differs from evidence")
    try:
        value = strict_json_loads(raw, schema)
    except ValueError as exc:
        fail("SEMANTICS_SCHEMA_INVALID", f"{schema} is not canonical JSON: {exc}")
    if not isinstance(value, dict) or canonical_json(value) != raw:
        fail("SEMANTICS_SCHEMA_INVALID", f"{schema} must be a canonical JSON object")
    return value


def _validate_current_evidence(root_fd: int, evidence: ArtifactEvidence, field: str) -> None:
    digest, size, current = hash_regular_at(root_fd, evidence.relative_path)
    if digest != evidence.sha256 or size != evidence.size or stable_identity(current) != evidence.identity:
        fail("SEMANTICS_EVIDENCE_STALE", f"{field} no longer matches its publication")


def _validate_cp6(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    stage: _ValidatedPublications,
    prior: _ValidatedPublications,
    previous_checkpoint_sha256: object,
    previous_checkpoint_transcript_sha256: object,
) -> dict[str, object]:
    cp5_sha256 = require_sha256(previous_checkpoint_sha256, "previous checkpoint sha256")
    cp5_transcript_sha256 = require_sha256(
        previous_checkpoint_transcript_sha256, "previous checkpoint transcript sha256"
    )
    expected = {
        "verification/verifier.json": (
            "verifier",
            "verification:verifier:",
            "verification-verifier-writer",
        ),
        "verification/code-reviewer.json": (
            "code-reviewer",
            "verification:code-reviewer:",
            "verification-code-reviewer-writer",
        ),
        "verification/adversarial-qa.json": (
            "adversarial-qa",
            "verification:adversarial-qa:",
            "verification-adversarial-qa-writer",
        ),
    }
    if stage.regular_paths != set(expected):
        fail("CP6_REPORT_SET_INVALID", "CP6 must publish exactly the three assurance reports")
    if set(stage.by_path) - set(expected) - {"verification"}:
        fail("CP6_REPORT_SET_INVALID", "CP6 contains an unexpected publication")
    prior_evidence_by_sha256: dict[str, ArtifactEvidence] = {}
    for evidence in prior.by_path.values():
        if evidence.artifact_type != "regular":
            continue
        digest = _evidence_digest(evidence)
        if digest in prior_evidence_by_sha256:
            fail(
                "CP6_PRIOR_EVIDENCE_INVALID",
                "pre-CP6 ArtifactEvidence digest is not unique",
            )
        prior_evidence_by_sha256[digest] = evidence
    if not prior_evidence_by_sha256:
        fail(
            "CP6_PRIOR_EVIDENCE_MISSING",
            "CP6 requires current regular ArtifactEvidence from CP0 through CP5",
        )

    writers: set[WriterIdentity] = set()
    report_rows: list[dict[str, object]] = []
    report_keys = {
        "schema",
        "sealed_run_id",
        "run_root",
        "reviewer",
        "cp5_sha256",
        "cp5_transcript_sha256",
        "tool_evidence_sha256",
        "runtime_evidence_sha256",
        "test_evidence_sha256",
        "verdict",
    }
    for path, (reviewer, task_prefix, writer_id) in expected.items():
        evidence = stage.by_path[path]
        writer = evidence.writer
        if (
            writer.role != "verification"
            or not writer.task_id.startswith(task_prefix)
            or writer.writer_id != writer_id
        ):
            fail("CP6_WRITER_INVALID", f"{path} has the wrong exact writer")
        writers.add(writer)
        row = require_exact_keys(
            _read_canonical_json(root_fd, evidence, CP6_SCHEMA), report_keys, CP6_SCHEMA
        )
        if (
            row["schema"] != CP6_SCHEMA
            or row["sealed_run_id"] != sealed_run_id
            or row["run_root"] != run_root
            or row["reviewer"] != reviewer
            or row["cp5_sha256"] != cp5_sha256
            or row["cp5_transcript_sha256"] != cp5_transcript_sha256
            or row["verdict"] != "PASS"
        ):
            fail("CP6_REPORT_INVALID", f"{path} semantic binding or verdict differs")
        evidence_fields = {
            "tool_evidence_sha256": "frozen/",
            "runtime_evidence_sha256": "runtime/",
            "test_evidence_sha256": "admission/final/",
        }
        digests = {
            field: require_sha256(row[field], f"{path}.{field}")
            for field in evidence_fields
        }
        if len(set(digests.values())) != len(digests):
            fail(
                "CP6_PRIOR_EVIDENCE_NOT_DISTINCT",
                f"{path} must bind three distinct evidence artifacts",
            )
        resolved: dict[str, dict[str, str]] = {}
        for field, required_prefix in evidence_fields.items():
            digest = digests[field]
            prior_evidence = prior_evidence_by_sha256.get(digest)
            if prior_evidence is None:
                fail(
                    "CP6_PRIOR_EVIDENCE_UNRESOLVED",
                    f"{path}.{field} does not resolve to current pre-CP6 ArtifactEvidence",
                )
            if not prior_evidence.relative_path.startswith(required_prefix):
                fail(
                    "CP6_PRIOR_EVIDENCE_CATEGORY_INVALID",
                    f"{path}.{field} resolves outside {required_prefix}",
                )
            resolved[field.removesuffix("_sha256")] = _evidence_ref(prior_evidence)
        report_rows.append(
            {
                "relative_path": path,
                "artifact_sha256": str(evidence.sha256),
                "artifact_evidence_sha256": _evidence_digest(evidence),
                "resolved_evidence": resolved,
            }
        )
    if len(writers) != 3:
        fail("CP6_WRITER_INVALID", "CP6 reports require three distinct exact writers")
    return {
        "schema": CP6_RESULT_SCHEMA,
        "sealed_run_id": sealed_run_id,
        "run_root": run_root,
        "checkpoint": 6,
        "cp5_sha256": cp5_sha256,
        "cp5_transcript_sha256": cp5_transcript_sha256,
        "report_count": 3,
        "reports_sha256": sha256_bytes(canonical_json(report_rows)),
        "reports": report_rows,
    }



CP1_CHECKPOINT_SCHEMA = "experiments7-strict-checkpoint/v6"
CP1_LINEAGE_SCHEMA = "experiments7-cp1-lineage-basis/v6"
CP1_PROFILE_SCHEMA = "experiments7-cp1-profile-basis/v6"
CP1_BASIS_SCHEMA = "experiments7-cp1-cp2-basis/v6"
CP1_SEMANTICS_SCHEMA = "experiments7-cp1-semantics/v6"
CP2_SNAPSHOT_REPORT_SCHEMA = "experiments7-cp2-snapshot-producer-results/v6"
CP2_RUNTIME_REPORT_SCHEMA = "experiments7-cp2-runtime-closure-validation/v6"
CP2_REPLAY_REPORT_SCHEMA = "experiments7-cp2-hermetic-profile-replay-results/v6"
CP2_CONFIG_SCHEMA = "experiments7-cp2-run-config/v6"
CP2_DENIAL_SCHEMA = "experiments7-cp2-denial-evidence/v6"
CP2_RESULT_SCHEMA = "experiments7-cp2-profile-result/v6"
CP2_SEMANTICS_SCHEMA = "experiments7-cp2-semantics/v6"
DENIAL_PROBES = (
    "outside_write_denied",
    "chmod_denied",
    "chown_denied",
    "xattr_denied",
    "utime_denied",
    "rename_denied",
    "link_denied",
    "unlink_denied",
    "network_denied",
    "fork_denied",
)
DISPATCH_STAGES = (
    "profile_expansion",
    "argv_construction",
    "dispatch",
    "transport_serialization",
    "transport_replay",
    "response_ingestion",
    "parser",
    "normalization",
    "evaluation",
    "aggregation",
    "reporting",
)
CP2_SEALS = {
    "snapshots/snapshot-producer-seal.json": (
        "experiments7-cp2-snapshot-producer-seal/v6",
        "snapshots/snapshot-producer-results.json",
    ),
    "runtime/runtime-sandbox-seal.json": (
        "experiments7-cp2-runtime-sandbox-seal/v6",
        "runtime/runtime-closure-validation.json",
    ),
    "runtime/hermetic-profile-replay-seal.json": (
        "experiments7-cp2-hermetic-profile-replay-seal/v6",
        "runtime/hermetic-profile-replay-results.json",
    ),
}


def _read_canonical_json_path(root_fd: int, relative_path: str, schema: str) -> tuple[dict[str, object], bytes]:
    raw, _ = read_regular_at(root_fd, relative_path)
    try:
        value = strict_json_loads(raw, schema)
    except ValueError as exc:
        fail("SEMANTICS_SCHEMA_INVALID", f"{schema} is not JSON: {exc}")
    if not isinstance(value, dict) or canonical_json(value) != raw:
        fail("SEMANTICS_SCHEMA_INVALID", f"{schema} must be a canonical JSON object")
    return value, raw


def _sorted_ids(value: object, field: str, expected_count: int) -> list[str]:
    if not isinstance(value, list):
        fail("SEMANTICS_SCHEMA_INVALID", f"{field} must be a list")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item or "\x00" in item or len(item) > 256:
            fail("SEMANTICS_SCHEMA_INVALID", f"{field} contains an invalid ID")
        result.append(item)
    if len(result) != expected_count or result != sorted(result) or len(set(result)) != len(result):
        fail("SEMANTICS_SCHEMA_INVALID", f"{field} must be sorted, unique, and exhaustive")
    return result


def _resolve_ref_list(
    value: object,
    evidence_by_path: dict[str, ArtifactEvidence],
    field: str,
    *,
    nonempty: bool = True,
) -> list[ArtifactEvidence]:
    if not isinstance(value, list) or (nonempty and not value):
        fail("SEMANTICS_EVIDENCE_REF_INVALID", f"{field} must be a nonempty list")
    resolved = [
        _resolve_ref(item, evidence_by_path, f"{field}[{index}]")
        for index, item in enumerate(value)
    ]
    paths = [item.relative_path for item in resolved]
    if paths != sorted(paths) or len(set(paths)) != len(paths):
        fail("SEMANTICS_EVIDENCE_REF_INVALID", f"{field} must be sorted and unique")
    return resolved


def _load_cp1_basis(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    cp1_sha256: str,
) -> tuple[list[str], list[str], ArtifactEvidence, dict[str, ArtifactEvidence]]:
    checkpoint, raw = _read_canonical_json_path(
        root_fd, "checkpoints/cp1.json", CP1_CHECKPOINT_SCHEMA
    )
    row = require_exact_keys(
        checkpoint,
        {
            "schema",
            "sealed_run_id",
            "run_root",
            "checkpoint",
            "previous_checkpoint_sha256",
            "previous_checkpoint_transcript_sha256",
            "stage_actual_paths",
            "stage_actual_paths_sha256",
            "bindings",
            "semantic_bindings",
            "semantic_bindings_sha256",
        },
        CP1_CHECKPOINT_SCHEMA,
    )
    if (
        sha256_bytes(raw) != cp1_sha256
        or row["schema"] != CP1_CHECKPOINT_SCHEMA
        or row["sealed_run_id"] != sealed_run_id
        or row["run_root"] != run_root
        or type(row["checkpoint"]) is not int
        or row["checkpoint"] != 1
        or not isinstance(row["stage_actual_paths"], list)
        or not isinstance(row["bindings"], dict)
        or not isinstance(row["semantic_bindings"], dict)
    ):
        fail("CP2_CP1_INVALID", "CP1 checkpoint binding differs")
    require_sha256(row["stage_actual_paths_sha256"], "CP1 stage paths sha256")
    if row["stage_actual_paths_sha256"] != sha256_bytes(canonical_json(row["stage_actual_paths"])):
        fail("CP2_CP1_INVALID", "CP1 stage path digest differs")

    cp1_evidence: dict[str, ArtifactEvidence] = {}
    for raw_evidence in row["stage_actual_paths"]:
        evidence = ArtifactEvidence.from_dict(raw_evidence)
        if evidence.relative_path in cp1_evidence:
            fail("CP2_CP1_INVALID", "CP1 stage paths contain a duplicate")
        cp1_evidence[evidence.relative_path] = evidence
        if evidence.artifact_type == "regular":
            _validate_current_evidence(root_fd, evidence, "CP1 evidence")

    semantic_bindings = _validate_cp1(
        root_fd,
        sealed_run_id,
        run_root,
        _ValidatedPublications(cp1_evidence, {}),
        row["previous_checkpoint_sha256"],
        row["previous_checkpoint_transcript_sha256"],
    )
    if (
        row["semantic_bindings"] != semantic_bindings
        or row["semantic_bindings_sha256"]
        != sha256_bytes(canonical_json(semantic_bindings))
    ):
        fail("CP2_CP1_INVALID", "CP1 semantic binding differs")

    basis_evidence = cp1_evidence.get("registry/cp2-basis.json")
    lineage_evidence = cp1_evidence.get("registry/lineage-basis.json")
    profile_evidence = cp1_evidence.get("registry/profile-basis.json")
    if basis_evidence is None or lineage_evidence is None or profile_evidence is None:
        fail("CP2_CP1_INVALID", "CP1 basis evidence is incomplete")
    basis = require_exact_keys(
        _read_canonical_json(root_fd, basis_evidence, CP1_BASIS_SCHEMA),
        {"schema", "sealed_run_id", "run_root", "lineage_source", "profile_source"},
        CP1_BASIS_SCHEMA,
    )
    if (
        basis["schema"] != CP1_BASIS_SCHEMA
        or basis["sealed_run_id"] != sealed_run_id
        or basis["run_root"] != run_root
        or _resolve_ref(basis["lineage_source"], cp1_evidence, "lineage_source") != lineage_evidence
        or _resolve_ref(basis["profile_source"], cp1_evidence, "profile_source") != profile_evidence
    ):
        fail("CP2_CP1_INVALID", "CP1 basis semantic binding differs")
    lineage = require_exact_keys(
        _read_canonical_json(root_fd, lineage_evidence, CP1_LINEAGE_SCHEMA),
        {"schema", "sealed_run_id", "run_root", "lineage_ids"},
        CP1_LINEAGE_SCHEMA,
    )
    profiles = require_exact_keys(
        _read_canonical_json(root_fd, profile_evidence, CP1_PROFILE_SCHEMA),
        {"schema", "sealed_run_id", "run_root", "profile_ids"},
        CP1_PROFILE_SCHEMA,
    )
    if (
        lineage["schema"] != CP1_LINEAGE_SCHEMA
        or profiles["schema"] != CP1_PROFILE_SCHEMA
        or lineage["sealed_run_id"] != sealed_run_id
        or profiles["sealed_run_id"] != sealed_run_id
        or lineage["run_root"] != run_root
        or profiles["run_root"] != run_root
    ):
        fail("CP2_CP1_INVALID", "CP1 lineage/profile binding differs")
    return (
        _sorted_ids(lineage["lineage_ids"], "lineage_ids", 129),
        _sorted_ids(profiles["profile_ids"], "profile_ids", 30),
        basis_evidence,
        cp1_evidence,
    )


def _cp2_header(
    row: dict[str, object],
    schema: str,
    sealed_run_id: str,
    run_root: str,
    cp1_sha256: str,
    cp1_transcript_sha256: str,
    basis_ref: dict[str, str],
) -> None:
    if (
        row["schema"] != schema
        or row["sealed_run_id"] != sealed_run_id
        or row["run_root"] != run_root
        or row["cp1_sha256"] != cp1_sha256
        or row["cp1_transcript_sha256"] != cp1_transcript_sha256
        or row["cp1_basis"] != basis_ref
    ):
        fail("CP2_REPORT_INVALID", f"{schema} header binding differs")


def _validate_snapshot_report(
    root_fd: int,
    stage: _ValidatedPublications,
    lineage_ids: list[str],
    header: tuple[str, str, str, str, dict[str, str]],
) -> tuple[dict[str, object], set[str]]:
    path = "snapshots/snapshot-producer-results.json"
    evidence = stage.by_path.get(path)
    if evidence is None:
        fail("CP2_REPORT_INVALID", "snapshot report is missing")
    row = require_exact_keys(
        _read_canonical_json(root_fd, evidence, CP2_SNAPSHOT_REPORT_SCHEMA),
        {
            "schema", "sealed_run_id", "run_root", "cp1_sha256",
            "cp1_transcript_sha256", "cp1_basis", "lineage_count",
            "records_sha256", "records",
        },
        CP2_SNAPSHOT_REPORT_SCHEMA,
    )
    _cp2_header(row, CP2_SNAPSHOT_REPORT_SCHEMA, *header)
    if _exact_int(row["lineage_count"], "lineage_count") != len(lineage_ids):
        fail("CP2_REPORT_INVALID", "snapshot lineage count differs")
    if not isinstance(row["records"], list):
        fail("CP2_REPORT_INVALID", "snapshot records must be a list")
    require_sha256(row["records_sha256"], "snapshot records sha256")
    if row["records_sha256"] != sha256_bytes(canonical_json(row["records"])):
        fail("CP2_REPORT_INVALID", "snapshot records digest differs")
    seen_ids: list[str] = []
    output_paths: list[str] = []
    for index, value in enumerate(row["records"]):
        item = require_exact_keys(value, {"lineage_id", "output"}, "snapshot-record")
        if not isinstance(item["lineage_id"], str):
            fail("CP2_REPORT_INVALID", "snapshot lineage ID is invalid")
        seen_ids.append(item["lineage_id"])
        output_paths.append(
            _resolve_ref(item["output"], stage.by_path, f"snapshot.records[{index}].output").relative_path
        )
    if seen_ids != lineage_ids or len(output_paths) != len(set(output_paths)):
        fail("CP2_REPORT_INVALID", "snapshot records do not exactly cover CP1 lineage")
    summary = {
        "verdict": "PASS",
        "expected_lineage_count": len(lineage_ids),
        "passed_lineage_count": len(output_paths),
        "lineage_ids_sha256": sha256_bytes(canonical_json(lineage_ids)),
    }
    return summary, {path, *output_paths}


def _validate_runtime_report(
    root_fd: int,
    stage: _ValidatedPublications,
    profile_ids: list[str],
    header: tuple[str, str, str, str, dict[str, str]],
) -> tuple[dict[str, object], set[str], dict[str, ArtifactEvidence]]:
    path = "runtime/runtime-closure-validation.json"
    evidence = stage.by_path.get(path)
    if evidence is None:
        fail("CP2_REPORT_INVALID", "runtime report is missing")
    row = require_exact_keys(
        _read_canonical_json(root_fd, evidence, CP2_RUNTIME_REPORT_SCHEMA),
        {
            "schema", "sealed_run_id", "run_root", "cp1_sha256",
            "cp1_transcript_sha256", "cp1_basis", "dependency_count",
            "config_count", "denial_probe_count", "dependencies", "configs",
            "denial_probes", "subject",
        },
        CP2_RUNTIME_REPORT_SCHEMA,
    )
    _cp2_header(row, CP2_RUNTIME_REPORT_SCHEMA, *header)
    dependencies = row["dependencies"]
    configs = row["configs"]
    denials = row["denial_probes"]
    if not isinstance(dependencies, list) or not dependencies or not isinstance(configs, list) or not isinstance(denials, list):
        fail("CP2_REPORT_INVALID", "runtime evidence lists are invalid")

    dependency_ids: list[str] = []
    consumed = {path}
    for index, value in enumerate(dependencies):
        item = require_exact_keys(value, {"dependency_id", "artifact"}, "runtime-dependency")
        if not isinstance(item["dependency_id"], str) or not item["dependency_id"]:
            fail("CP2_REPORT_INVALID", "dependency ID is invalid")
        dependency_ids.append(item["dependency_id"])
        consumed.add(_resolve_ref(item["artifact"], stage.by_path, f"dependencies[{index}]").relative_path)
    if dependency_ids != sorted(set(dependency_ids)):
        fail("CP2_REPORT_INVALID", "dependency IDs must be sorted and unique")

    config_map: dict[str, ArtifactEvidence] = {}
    config_ids: list[str] = []
    for index, value in enumerate(configs):
        item = require_exact_keys(value, {"profile_id", "config"}, "runtime-config-ref")
        if not isinstance(item["profile_id"], str):
            fail("CP2_REPORT_INVALID", "config profile ID is invalid")
        config_ids.append(item["profile_id"])
        config_evidence = _resolve_ref(item["config"], stage.by_path, f"configs[{index}]")
        config = require_exact_keys(
            _read_canonical_json(root_fd, config_evidence, CP2_CONFIG_SCHEMA),
            {"schema", "profile_id", "dependency_ids", "dispatch_path", "transport"},
            CP2_CONFIG_SCHEMA,
        )
        if (
            config["schema"] != CP2_CONFIG_SCHEMA
            or config["profile_id"] != item["profile_id"]
            or config["dependency_ids"] != dependency_ids
            or config["dispatch_path"] != "production_external_contract"
            or config["transport"] != "hermetic_replay"
        ):
            fail("CP2_REPORT_INVALID", "runtime config closure differs")
        config_map[str(item["profile_id"])] = config_evidence
        consumed.add(config_evidence.relative_path)
    if config_ids != profile_ids or len(config_map) != len(profile_ids):
        fail("CP2_REPORT_INVALID", "runtime configs do not exactly cover CP1 profiles")

    denial_ids: list[str] = []
    for index, value in enumerate(denials):
        item = require_exact_keys(
            value, {"probe_id", "evidence", "observed_errno", "state"}, "denial-probe"
        )
        if not isinstance(item["probe_id"], str):
            fail("CP2_REPORT_INVALID", "denial probe ID is invalid")
        observed_errno = _exact_int(item["observed_errno"], "observed_errno", minimum=1)
        if item["state"] != "DENIED" or observed_errno not in {1, 13}:
            fail("CP2_REPORT_INVALID", "denial probe result is invalid")
        denial_ids.append(item["probe_id"])
        denial_evidence = _resolve_ref(item["evidence"], stage.by_path, f"denial_probes[{index}]")
        denial = require_exact_keys(
            _read_canonical_json(root_fd, denial_evidence, CP2_DENIAL_SCHEMA),
            {"schema", "probe_id", "observed_errno", "result"},
            CP2_DENIAL_SCHEMA,
        )
        if (
            denial["schema"] != CP2_DENIAL_SCHEMA
            or denial["probe_id"] != item["probe_id"]
            or _exact_int(denial["observed_errno"], "denial observed_errno", minimum=1) != observed_errno
            or denial["result"] != "DENIED"
        ):
            fail("CP2_REPORT_INVALID", "denial evidence differs from report")
        consumed.add(denial_evidence.relative_path)
    if denial_ids != list(DENIAL_PROBES):
        fail("CP2_REPORT_INVALID", "denial probes are not the exact required set")

    subject = require_exact_keys(
        row["subject"],
        {
            "effective_uid", "no_new_privs", "seccomp_mode", "capability_inheritable",
            "capability_permitted", "capability_effective", "capability_ambient",
            "cooperating_subjects",
        },
        "runtime-subject",
    )
    if (
        _exact_int(subject["effective_uid"], "effective_uid") == 0
        or subject["no_new_privs"] != "ENFORCED"
        or _exact_int(subject["seccomp_mode"], "seccomp_mode") != 2
        or any(subject[field] != "0000000000000000" for field in (
            "capability_inheritable", "capability_permitted", "capability_effective", "capability_ambient"
        ))
        or subject["cooperating_subjects"] != "DENIED"
    ):
        fail("CP2_REPORT_INVALID", "runtime subject closure differs")
    if (
        _exact_int(row["dependency_count"], "dependency_count") != len(dependency_ids)
        or _exact_int(row["config_count"], "config_count") != len(config_ids)
        or _exact_int(row["denial_probe_count"], "denial_probe_count") != len(denial_ids)
    ):
        fail("CP2_REPORT_INVALID", "runtime exact counts differ")
    summary = {
        "verdict": "PASS",
        "expected_profile_count": len(profile_ids),
        "config_closed_count": len(config_ids),
        "dependency_count": len(dependency_ids),
        "denial_probe_count": len(denial_ids),
    }
    return summary, consumed, config_map


def _validate_replay_report(
    root_fd: int,
    stage: _ValidatedPublications,
    profile_ids: list[str],
    configs: dict[str, ArtifactEvidence],
    header: tuple[str, str, str, str, dict[str, str]],
) -> tuple[dict[str, object], set[str]]:
    path = "runtime/hermetic-profile-replay-results.json"
    evidence = stage.by_path.get(path)
    if evidence is None:
        fail("CP2_REPORT_INVALID", "profile replay report is missing")
    row = require_exact_keys(
        _read_canonical_json(root_fd, evidence, CP2_REPLAY_REPORT_SCHEMA),
        {
            "schema", "sealed_run_id", "run_root", "cp1_sha256",
            "cp1_transcript_sha256", "cp1_basis", "profile_count",
            "records_sha256", "records",
        },
        CP2_REPLAY_REPORT_SCHEMA,
    )
    _cp2_header(row, CP2_REPLAY_REPORT_SCHEMA, *header)
    if _exact_int(row["profile_count"], "profile_count") != len(profile_ids) or not isinstance(row["records"], list):
        fail("CP2_REPORT_INVALID", "replay profile count or records differ")
    require_sha256(row["records_sha256"], "replay records sha256")
    if row["records_sha256"] != sha256_bytes(canonical_json(row["records"])):
        fail("CP2_REPORT_INVALID", "replay records digest differs")
    snapshot_ref = _evidence_ref(stage.by_path["snapshots/snapshot-producer-results.json"])
    runtime_ref = _evidence_ref(stage.by_path["runtime/runtime-closure-validation.json"])
    seen_ids: list[str] = []
    result_paths: list[str] = []
    for index, value in enumerate(row["records"]):
        item = require_exact_keys(value, {"profile_id", "config", "result", "state"}, "replay-record")
        profile_id = item["profile_id"]
        if not isinstance(profile_id, str) or item["state"] != "PASS":
            fail("CP2_REPORT_INVALID", "replay record identity or state differs")
        seen_ids.append(profile_id)
        config_evidence = _resolve_ref(item["config"], stage.by_path, f"replay.records[{index}].config")
        if configs.get(profile_id) != config_evidence:
            fail("CP2_REPORT_INVALID", "replay record config differs from runtime closure")
        result_evidence = _resolve_ref(item["result"], stage.by_path, f"replay.records[{index}].result")
        result = require_exact_keys(
            _read_canonical_json(root_fd, result_evidence, CP2_RESULT_SCHEMA),
            {
                "schema", "profile_id", "config_sha256", "snapshot_report",
                "runtime_report", "dispatch_stages", "golden_result",
            },
            CP2_RESULT_SCHEMA,
        )
        if (
            result["schema"] != CP2_RESULT_SCHEMA
            or result["profile_id"] != profile_id
            or result["config_sha256"] != config_evidence.sha256
            or result["snapshot_report"] != snapshot_ref
            or result["runtime_report"] != runtime_ref
            or result["dispatch_stages"] != list(DISPATCH_STAGES)
            or result["golden_result"] != "EXACT"
        ):
            fail("CP2_REPORT_INVALID", "replay result evidence differs")
        result_paths.append(result_evidence.relative_path)
    if seen_ids != profile_ids or len(result_paths) != len(set(result_paths)):
        fail("CP2_REPORT_INVALID", "replay records do not exactly cover CP1 profiles")
    summary = {
        "verdict": "PASS",
        "expected_profile_count": len(profile_ids),
        "passed_profile_count": len(result_paths),
        "profile_ids_sha256": sha256_bytes(canonical_json(profile_ids)),
        "dispatch_path": "production_external_contract",
        "golden_mode": "exact_output_hashes",
    }
    return summary, {path, *result_paths}


def _validate_cp2_seal(
    root_fd: int,
    stage: _ValidatedPublications,
    seal_path: str,
    expected_owned: set[str],
    expected_summary: dict[str, object],
    header: tuple[str, str, str, str, dict[str, str]],
) -> set[str]:
    seal_schema, report_path = CP2_SEALS[seal_path]
    evidence = stage.by_path.get(seal_path)
    if evidence is None:
        fail("CP2_SEAL_INVALID", "required CP2 seal is missing")
    row = require_exact_keys(
        _read_canonical_json(root_fd, evidence, seal_schema),
        {
            "schema", "sealed_run_id", "run_root", "cp1_sha256",
            "cp1_transcript_sha256", "cp1_basis", "report", "owned_artifacts",
            "owned_artifacts_sha256", "summary",
        },
        seal_schema,
    )
    _cp2_header(row, seal_schema, *header)
    if row["report"] != _evidence_ref(stage.by_path[report_path]):
        fail("CP2_SEAL_INVALID", "seal report EvidenceRef differs")
    owned = _resolve_ref_list(row["owned_artifacts"], stage.by_path, "owned_artifacts")
    owned_paths = {item.relative_path for item in owned}
    require_sha256(row["owned_artifacts_sha256"], "owned artifacts sha256")
    if (
        row["owned_artifacts_sha256"] != sha256_bytes(canonical_json(row["owned_artifacts"]))
        or owned_paths != expected_owned
        or row["summary"] != expected_summary
        or report_path not in owned_paths
        or owned_paths & set(CP2_SEALS)
    ):
        fail("CP2_SEAL_INVALID", "seal ownership, digest, or derived summary differs")
    return owned_paths


def _validate_cp2_structure(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    stage: _ValidatedPublications,
    previous_checkpoint_sha256: object,
    previous_checkpoint_transcript_sha256: object,
) -> dict[str, object]:
    cp1_sha256 = require_sha256(previous_checkpoint_sha256, "previous checkpoint sha256")
    cp1_transcript_sha256 = require_sha256(
        previous_checkpoint_transcript_sha256, "previous checkpoint transcript sha256"
    )
    lineage_ids, profile_ids, basis_evidence, _ = _load_cp1_basis(
        root_fd, sealed_run_id, run_root, cp1_sha256
    )
    basis_ref = _evidence_ref(basis_evidence)
    header = (sealed_run_id, run_root, cp1_sha256, cp1_transcript_sha256, basis_ref)

    for path, evidence in stage.by_path.items():
        if not (path == "snapshots" or path.startswith("snapshots/") or path == "runtime" or path.startswith("runtime/")):
            fail("CP2_ARTIFACT_SET_INVALID", "CP2 contains an artifact outside its exact roots")
        writer = evidence.writer
        if (
            writer.role != "adapter_snapshot"
            or not writer.task_id.startswith("snapshot:")
            or writer.writer_id != "adapter_snapshot-writer"
        ):
            fail("CP2_WRITER_INVALID", "CP2 artifact has the wrong exact writer")

    snapshot_summary, snapshot_owned = _validate_snapshot_report(
        root_fd, stage, lineage_ids, header
    )
    runtime_summary, runtime_owned, configs = _validate_runtime_report(
        root_fd, stage, profile_ids, header
    )
    replay_summary, replay_owned = _validate_replay_report(
        root_fd, stage, profile_ids, configs, header
    )
    owned_sets = (
        _validate_cp2_seal(
            root_fd, stage, "snapshots/snapshot-producer-seal.json",
            snapshot_owned, snapshot_summary, header,
        ),
        _validate_cp2_seal(
            root_fd, stage, "runtime/runtime-sandbox-seal.json",
            runtime_owned, runtime_summary, header,
        ),
        _validate_cp2_seal(
            root_fd, stage, "runtime/hermetic-profile-replay-seal.json",
            replay_owned, replay_summary, header,
        ),
    )
    if any(owned_sets[left] & owned_sets[right] for left in range(3) for right in range(left + 1, 3)):
        fail("CP2_OWNERSHIP_INVALID", "CP2 seal ownership sets overlap")
    owned_union = set().union(*owned_sets)
    if owned_union != stage.regular_paths - set(CP2_SEALS):
        fail("CP2_OWNERSHIP_INVALID", "CP2 ownership is not the exact regular-artifact union")
    seal_rows = [
        {
            "relative_path": path,
            "artifact_sha256": str(stage.by_path[path].sha256),
            "artifact_evidence_sha256": _evidence_digest(stage.by_path[path]),
        }
        for path in sorted(CP2_SEALS)
    ]
    return {
        "schema": CP2_SEMANTICS_SCHEMA,
        "sealed_run_id": sealed_run_id,
        "run_root": run_root,
        "checkpoint": 2,
        "cp1_sha256": cp1_sha256,
        "cp1_transcript_sha256": cp1_transcript_sha256,
        "cp1_basis_artifact_evidence_sha256": _evidence_digest(basis_evidence),
        "lineage_count": len(lineage_ids),
        "profile_count": len(profile_ids),
        "owned_artifact_count": len(owned_union),
        "owned_artifacts_sha256": sha256_bytes(canonical_json(sorted(owned_union))),
        "seals_sha256": sha256_bytes(canonical_json(seal_rows)),
        "seals": seal_rows,
    }


def validate_cp2_artifact_structure(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    publications: Iterable[Publication],
    previous_checkpoint_sha256: str,
    previous_checkpoint_transcript_sha256: str,
) -> dict[str, object]:
    """Validate CP2 artifact structure only; this is never acceptance evidence."""

    sealed_run_id = validate_sealed_run_id(sealed_run_id)
    run_root = validate_absolute_path_text(run_root, "run_root")
    stage = _validate_publications(root_fd, sealed_run_id, run_root, publications)
    return _validate_cp2_structure(
        root_fd,
        sealed_run_id,
        run_root,
        stage,
        previous_checkpoint_sha256,
        previous_checkpoint_transcript_sha256,
    )


CP0_INVENTORY_SCHEMA = "experiments7-cp0-frozen-inventory/v6"
CP0_SEMANTICS_SCHEMA = "experiments7-cp0-semantics/v6"
CP0_ROLES = (
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
)
CP0_FIXED_REGULAR_PATHS = {
    "reservation.json",
    "reservation-acceptance.json",
    "owner-binding.json",
    "manifests/source-pre.jsonl",
    "manifests/paper-pre.json",
    "frozen/inventory.json",
}


def _required_ancestor_directories(paths: Iterable[str]) -> set[str]:
    directories: set[str] = set()
    for path in paths:
        parts = path.split("/")[:-1]
        for index in range(1, len(parts) + 1):
            directories.add("/".join(parts[:index]))
    return directories


def _cp0_writer_valid(path: str, writer: WriterIdentity) -> bool:
    if path == "checkpoints":
        return (
            writer.role == "checkpoint_controller"
            and writer.task_id.startswith("checkpoint:")
            and writer.writer_id == "checkpoint_controller-writer"
        )
    if path == "reservation.json":
        return (
            writer.role == "run_reservation"
            and writer.task_id.startswith("reservation:")
            and writer.writer_id == "run_reservation-writer"
        )
    if path == "reservation-acceptance.json":
        return (
            writer.role == "reservation_acceptance"
            and writer.task_id.startswith("acceptance:")
            and writer.writer_id == "reservation_acceptance-writer"
        )
    return (
        writer.role == "g0"
        and writer.task_id.startswith("g0:")
        and writer.writer_id == "g0-writer"
    )


def _validate_source_pre(root_fd: int, evidence: ArtifactEvidence) -> int:
    raw, current = read_regular_at(root_fd, evidence.relative_path)
    if (
        not raw
        or not raw.endswith(b"\n")
        or sha256_bytes(raw) != evidence.sha256
        or len(raw) != evidence.size
        or stable_identity(current) != evidence.identity
    ):
        fail("CP0_SOURCE_PRE_INVALID", "source-pre bytes or evidence differ")
    lines = raw.splitlines(keepends=True)
    if not lines:
        fail("CP0_SOURCE_PRE_INVALID", "source-pre must contain a record")
    for line in lines:
        if not line.strip():
            fail("CP0_SOURCE_PRE_INVALID", "source-pre contains an empty record")
        try:
            value = strict_json_loads(line, "source-pre record")
        except ValueError as exc:
            fail("CP0_SOURCE_PRE_INVALID", f"source-pre record is invalid: {exc}")
        if not isinstance(value, dict) or canonical_json(value) != line:
            fail("CP0_SOURCE_PRE_INVALID", "source-pre records must be canonical JSON objects")
    return len(lines)


def _validate_cp0(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    stage: _ValidatedPublications,
    previous_checkpoint_sha256: object,
    previous_checkpoint_transcript_sha256: object,
) -> dict[str, object]:
    if previous_checkpoint_sha256 is not None or previous_checkpoint_transcript_sha256 is not None:
        fail("CP0_PREDECESSOR_INVALID", "CP0 cannot have a predecessor checkpoint")
    if not CP0_FIXED_REGULAR_PATHS <= stage.regular_paths:
        fail("CP0_ARTIFACT_SET_INVALID", "CP0 fixed artifacts are incomplete")
    for path, evidence in stage.by_path.items():
        if not _cp0_writer_valid(path, evidence.writer):
            fail("CP0_WRITER_INVALID", "CP0 artifact has the wrong exact writer")
        if evidence.artifact_type == "directory" and not (
            path == "frozen"
            or path.startswith("frozen/")
            or path == "manifests"
            or path.startswith("manifests/")
            or path == "checkpoints"
        ):
            fail("CP0_ARTIFACT_SET_INVALID", "CP0 contains an unexpected directory")

    inventory_evidence = stage.by_path["frozen/inventory.json"]
    inventory = require_exact_keys(
        _read_canonical_json(root_fd, inventory_evidence, CP0_INVENTORY_SCHEMA),
        {
            "schema",
            "sealed_run_id",
            "run_root",
            "roles",
            "frozen_artifact_count",
            "frozen_artifacts_sha256",
            "source_pre",
            "paper_pre",
        },
        CP0_INVENTORY_SCHEMA,
    )
    if (
        inventory["schema"] != CP0_INVENTORY_SCHEMA
        or inventory["sealed_run_id"] != sealed_run_id
        or inventory["run_root"] != run_root
    ):
        fail("CP0_INVENTORY_INVALID", "CP0 inventory run binding differs")
    roles = require_exact_keys(inventory["roles"], set(CP0_ROLES), "CP0 roles")
    role_paths: dict[str, list[str]] = {}
    all_paths: list[str] = []
    all_refs: list[dict[str, str]] = []
    for role in CP0_ROLES:
        resolved = _resolve_ref_list(roles[role], stage.by_path, f"roles.{role}")
        paths = [evidence.relative_path for evidence in resolved]
        if any(
            not path.startswith("frozen/") or path == "frozen/inventory.json"
            for path in paths
        ):
            fail("CP0_INVENTORY_INVALID", "CP0 role references a non-frozen artifact")
        role_paths[role] = paths
        all_paths.extend(paths)
        all_refs.extend(_evidence_ref(evidence) for evidence in resolved)
    if len(all_paths) != len(set(all_paths)):
        fail("CP0_INVENTORY_INVALID", "CP0 role ownership overlaps")
    frozen_paths = sorted(
        path
        for path in stage.regular_paths
        if path.startswith("frozen/") and path != "frozen/inventory.json"
    )
    if sorted(all_paths) != frozen_paths:
        fail("CP0_INVENTORY_INVALID", "CP0 roles are not the exhaustive frozen inventory")
    canonical_refs = sorted(all_refs, key=lambda item: item["relative_path"])
    if (
        _exact_int(inventory["frozen_artifact_count"], "frozen_artifact_count")
        != len(frozen_paths)
    ):
        fail("CP0_INVENTORY_INVALID", "CP0 frozen artifact count differs")
    require_sha256(inventory["frozen_artifacts_sha256"], "frozen artifacts sha256")
    if inventory["frozen_artifacts_sha256"] != sha256_bytes(canonical_json(canonical_refs)):
        fail("CP0_INVENTORY_INVALID", "CP0 frozen artifact digest differs")

    source_evidence = _resolve_ref(inventory["source_pre"], stage.by_path, "source_pre")
    paper_evidence = _resolve_ref(inventory["paper_pre"], stage.by_path, "paper_pre")
    if (
        source_evidence.relative_path != "manifests/source-pre.jsonl"
        or paper_evidence.relative_path != "manifests/paper-pre.json"
    ):
        fail("CP0_INVENTORY_INVALID", "CP0 source/paper references use the wrong paths")
    source_record_count = _validate_source_pre(root_fd, source_evidence)
    paper = _read_canonical_json(root_fd, paper_evidence, "paper-pre")
    if not paper:
        fail("CP0_PAPER_PRE_INVALID", "paper-pre must be a nonempty canonical object")
    expected_regular = CP0_FIXED_REGULAR_PATHS | set(frozen_paths)
    if stage.regular_paths != expected_regular:
        fail("CP0_ARTIFACT_SET_INVALID", "CP0 contains an unaccounted regular artifact")
    expected_paths = (
        expected_regular
        | _required_ancestor_directories(expected_regular)
        | {"checkpoints"}
    )
    if set(stage.by_path) != expected_paths:
        fail("CP0_ARTIFACT_SET_INVALID", "CP0 directory ownership is not exhaustive")
    return {
        "schema": CP0_SEMANTICS_SCHEMA,
        "sealed_run_id": sealed_run_id,
        "run_root": run_root,
        "checkpoint": 0,
        "role_count": len(CP0_ROLES),
        "frozen_artifact_count": len(frozen_paths),
        "frozen_artifacts_sha256": sha256_bytes(canonical_json(canonical_refs)),
        "role_paths_sha256": sha256_bytes(canonical_json(role_paths)),
        "source_pre_artifact_evidence_sha256": _evidence_digest(source_evidence),
        "source_pre_record_count": source_record_count,
        "paper_pre_artifact_evidence_sha256": _evidence_digest(paper_evidence),
        "inventory_artifact_evidence_sha256": _evidence_digest(inventory_evidence),
    }


def _read_canonical_jsonl_inventory(
    root_fd: int,
    evidence: ArtifactEvidence,
    *,
    expected_path: str,
    expected_count: int,
) -> tuple[list[dict[str, object]], bytes]:
    if (
        evidence.relative_path != expected_path
        or evidence.artifact_type != "regular"
    ):
        fail("CP1_INVENTORY_INVALID", f"{expected_path} must be regular")
    raw, current = read_regular_at(root_fd, expected_path)
    if (
        not raw
        or not raw.endswith(b"\n")
        or len(raw) != evidence.size
        or sha256_bytes(raw) != evidence.sha256
        or stable_identity(current) != evidence.identity
    ):
        fail(
            "CP1_INVENTORY_INVALID",
            f"{expected_path} bytes or publication evidence differ",
        )
    records: list[dict[str, object]] = []
    for number, line in enumerate(raw.splitlines(keepends=True), start=1):
        if line == b"\n" or not line.endswith(b"\n"):
            fail(
                "CP1_INVENTORY_INVALID",
                f"{expected_path} line {number} is empty or unterminated",
            )
        try:
            value = strict_json_loads(line, f"{expected_path} line {number}")
        except ValueError as exc:
            fail(
                "CP1_INVENTORY_INVALID",
                f"{expected_path} line {number} is invalid JSON: {exc}",
            )
        if not isinstance(value, dict) or canonical_json(value) != line:
            fail(
                "CP1_INVENTORY_INVALID",
                f"{expected_path} line {number} is not canonical JSON",
            )
        records.append(value)
    if len(records) != expected_count:
        fail(
            "CP1_INVENTORY_INVALID",
            f"{expected_path} must contain exactly {expected_count} records",
        )
    return records, raw


def _cp1_writer_valid(path: str, writer: WriterIdentity) -> bool:
    if path == "registry" or path.startswith("registry/"):
        return (
            writer.role == "registry"
            and writer.task_id.startswith("registry:")
            and writer.writer_id == "registry-writer"
        )
    if path == "inventory" or path.startswith("inventory/"):
        return (
            writer.role == "inventory"
            and writer.task_id.startswith("inventory:")
            and writer.writer_id == "inventory-writer"
        )
    return False


def _validate_cp1(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    stage: _ValidatedPublications,
    previous_checkpoint_sha256: object,
    previous_checkpoint_transcript_sha256: object,
) -> dict[str, object]:
    cp0_sha256 = require_sha256(previous_checkpoint_sha256, "previous checkpoint sha256")
    cp0_transcript_sha256 = require_sha256(
        previous_checkpoint_transcript_sha256,
        "previous checkpoint transcript sha256",
    )
    regular_paths = {
        "registry/lineage-basis.json",
        "registry/profile-basis.json",
        "registry/cp2-basis.json",
        "inventory/structure.jsonl",
        "inventory/results.jsonl",
        "inventory/aliases.jsonl",
    }
    expected_paths = regular_paths | {"registry", "inventory"}
    if set(stage.by_path) != expected_paths or stage.regular_paths != regular_paths:
        fail("CP1_ARTIFACT_SET_INVALID", "CP1 artifact ownership is not exact")
    for path, evidence in stage.by_path.items():
        if not _cp1_writer_valid(path, evidence.writer):
            fail("CP1_WRITER_INVALID", f"{path} has the wrong exact writer")

    lineage_evidence = stage.by_path["registry/lineage-basis.json"]
    profile_evidence = stage.by_path["registry/profile-basis.json"]
    basis_evidence = stage.by_path["registry/cp2-basis.json"]
    structure_evidence = stage.by_path["inventory/structure.jsonl"]
    inventory_evidence = stage.by_path["inventory/results.jsonl"]
    aliases_evidence = stage.by_path["inventory/aliases.jsonl"]
    lineage = require_exact_keys(
        _read_canonical_json(root_fd, lineage_evidence, CP1_LINEAGE_SCHEMA),
        {"schema", "sealed_run_id", "run_root", "lineage_ids"},
        CP1_LINEAGE_SCHEMA,
    )
    profiles = require_exact_keys(
        _read_canonical_json(root_fd, profile_evidence, CP1_PROFILE_SCHEMA),
        {"schema", "sealed_run_id", "run_root", "profile_ids"},
        CP1_PROFILE_SCHEMA,
    )
    basis = require_exact_keys(
        _read_canonical_json(root_fd, basis_evidence, CP1_BASIS_SCHEMA),
        {"schema", "sealed_run_id", "run_root", "lineage_source", "profile_source"},
        CP1_BASIS_SCHEMA,
    )
    if any(
        row["sealed_run_id"] != sealed_run_id or row["run_root"] != run_root
        for row in (lineage, profiles, basis)
    ) or (
        lineage["schema"] != CP1_LINEAGE_SCHEMA
        or profiles["schema"] != CP1_PROFILE_SCHEMA
        or basis["schema"] != CP1_BASIS_SCHEMA
    ):
        fail("CP1_BASIS_INVALID", "CP1 registry basis run binding differs")
    lineage_ids = _sorted_ids(lineage["lineage_ids"], "lineage_ids", 129)
    profile_ids = _sorted_ids(profiles["profile_ids"], "profile_ids", 30)
    if (
        _resolve_ref(basis["lineage_source"], stage.by_path, "lineage_source")
        != lineage_evidence
        or _resolve_ref(basis["profile_source"], stage.by_path, "profile_source")
        != profile_evidence
    ):
        fail("CP1_BASIS_INVALID", "CP1 cp2-basis references differ")
    structure, structure_raw = _read_canonical_jsonl_inventory(
        root_fd,
        structure_evidence,
        expected_path="inventory/structure.jsonl",
        expected_count=CP1_STRUCTURE_COUNT,
    )
    inventory, raw = _read_canonical_jsonl_inventory(
        root_fd,
        inventory_evidence,
        expected_path="inventory/results.jsonl",
        expected_count=CP1_RESULT_COUNT,
    )
    aliases, aliases_raw = _read_canonical_jsonl_inventory(
        root_fd,
        aliases_evidence,
        expected_path="inventory/aliases.jsonl",
        expected_count=CP1_ALIAS_COUNT,
    )
    try:
        inventory_sha256 = inventory_digest(structure, inventory, aliases)
    except V6ContractError as exc:
        fail("CP1_INVENTORY_INVALID", str(exc))
    return {
        "schema": CP1_SEMANTICS_SCHEMA,
        "sealed_run_id": sealed_run_id,
        "run_root": run_root,
        "checkpoint": 1,
        "cp0_sha256": cp0_sha256,
        "cp0_transcript_sha256": cp0_transcript_sha256,
        "lineage_count": len(lineage_ids),
        "lineage_ids_sha256": sha256_bytes(canonical_json(lineage_ids)),
        "profile_count": len(profile_ids),
        "profile_ids_sha256": sha256_bytes(canonical_json(profile_ids)),
        "inventory_structure_count": len(structure),
        "inventory_record_count": len(inventory),
        "inventory_alias_count": len(aliases),
        "inventory_sha256": inventory_sha256,
        "inventory_structure_bytes_sha256": sha256_bytes(structure_raw),
        "inventory_bytes_sha256": sha256_bytes(raw),
        "inventory_alias_bytes_sha256": sha256_bytes(aliases_raw),
        "basis_artifact_evidence_sha256": _evidence_digest(basis_evidence),
        "inventory_structure_artifact_evidence_sha256": _evidence_digest(
            structure_evidence
        ),
        "inventory_artifact_evidence_sha256": _evidence_digest(inventory_evidence),
        "inventory_alias_artifact_evidence_sha256": _evidence_digest(
            aliases_evidence
        ),
    }

def validate_stage_semantics(
    root_fd: int,
    sealed_run_id: str,
    run_root: str,
    checkpoint: int,
    publications: Iterable[Publication],
    previous_checkpoint_sha256: str | None,
    previous_checkpoint_transcript_sha256: str | None,
    *,
    prior_publications: Iterable[Publication] | None = None,
) -> dict[str, object]:
    """Validate one stage from descriptor-relative bytes and publication bindings."""

    sealed_run_id = validate_sealed_run_id(sealed_run_id)
    run_root = validate_absolute_path_text(run_root, "run_root")
    if type(checkpoint) is not int or checkpoint not in {0, 1, 2, 6}:
        fail("SEMANTICS_UNSUPPORTED_CHECKPOINT", "semantic validation supports CP0, CP1, CP2, and CP6")
    if checkpoint == 2:
        try:
            require_strict_runtime_proof_environment((Path(run_root),))
        except SandboxUnavailable as exc:
            fail(exc.code, "runtime V2 positive proof environment is unavailable")
        return validate_cp2_artifact_structure(
            root_fd,
            sealed_run_id,
            run_root,
            publications,
            previous_checkpoint_sha256,
            previous_checkpoint_transcript_sha256,
        )
    stage = _validate_publications(root_fd, sealed_run_id, run_root, publications)
    if checkpoint == 6:
        if prior_publications is None:
            fail(
                "CP6_PRIOR_EVIDENCE_MISSING",
                "CP6 requires typed pre-CP6 publications",
            )
        prior = _validate_publications(
            root_fd,
            sealed_run_id,
            run_root,
            prior_publications,
        )
        if set(stage.by_path).intersection(prior.by_path):
            fail(
                "CP6_PRIOR_EVIDENCE_INVALID",
                "CP6 stage overlaps the pre-CP6 evidence set",
            )
        return _validate_cp6(
            root_fd,
            sealed_run_id,
            run_root,
            stage,
            prior,
            previous_checkpoint_sha256,
            previous_checkpoint_transcript_sha256,
        )
    if checkpoint == 0:
        return _validate_cp0(
            root_fd,
            sealed_run_id,
            run_root,
            stage,
            previous_checkpoint_sha256,
            previous_checkpoint_transcript_sha256,
        )
    if checkpoint == 1:
        return _validate_cp1(
            root_fd,
            sealed_run_id,
            run_root,
            stage,
            previous_checkpoint_sha256,
            previous_checkpoint_transcript_sha256,
        )
    fail("SEMANTICS_NOT_IMPLEMENTED", "requested semantic checkpoint is not implemented")
