"""Fresh CP1 publication from an immutable, readonly captured paper layout.

This module deliberately consumes only the accepted CP0 tree and declared
out-of-tree evidence.  It never opens the protected PDF or any protected
experiment root; that happens in ``cp1_protected_extract.py`` before this
controller is invoked.
"""
from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from provenance.strict_v6 import (
    CP1_ALIAS_COUNT,
    CP1_RESULT_COUNT,
    CP1_STRUCTURE_COUNT,
    inventory_digest,
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
from .checkpoint import CheckpointResult, publish_checkpoint, validate_checkpoint_payload
from .cp1_inventory_builder import build_inventory_from_layout
from .external import (
    ExternalOutputReservation,
    ExternalTranscriptRecord,
    ExternalTranscriptRegistry,
)
from .filesystem import open_run_handle, read_regular_at
from .publication import ArtifactEvidence, Publication, StagePublisher, transcript_hash
from .writer_policy import WriterIdentity, WriterPolicy, default_writer_policy


CP1_CAPTURE_SCHEMA = "experiments7-cp1-protected-paper-extraction/v6"
CP1_CAPTURE_EXECUTION_SCHEMA = "experiments7-cp1-paper-extraction-execution/v6"
CP1_TASK_EVIDENCE_SCHEMA = "experiments7-cp1-task-evidence/v6"
CP1_TASK_BINDING_SCHEMA = "experiments7-cp1-task-evidence-binding/v6"
CP1_LINEAGE_SCHEMA = "experiments7-cp1-lineage-basis/v6"
CP1_PROFILE_SCHEMA = "experiments7-cp1-profile-basis/v6"
CP1_BASIS_SCHEMA = "experiments7-cp1-cp2-basis/v6"


@dataclass(frozen=True)
class PreparedCP1:
    structure: tuple[dict[str, object], ...]
    results: tuple[dict[str, object], ...]
    aliases: tuple[dict[str, object], ...]
    lineage_ids: tuple[str, ...]
    profile_ids: tuple[str, ...]
    layout_sha256: str
    source_counts: Mapping[str, int]
    alias_reason_counts: Mapping[str, int]
    structure_payload: bytes
    results_payload: bytes
    aliases_payload: bytes


@dataclass(frozen=True)
class CP1BootstrapResult:
    sealed_run_id: str
    run_root: str
    checkpoint: CheckpointResult
    stage_publications: tuple[Publication, ...]
    task_evidence_sha256: str
    external_paths: Mapping[str, str]


def _canonical_object(raw: bytes, label: str) -> dict[str, object]:
    value = strict_json_loads(raw, label)
    if not isinstance(value, dict) or canonical_json(value) != raw:
        fail("CP1_INPUT_NONCANONICAL", f"{label} is not canonical JSON")
    return value


def _canonical_jsonl(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(canonical_json(dict(row)) for row in rows)


def _normalized_pages(capture: Mapping[str, object]) -> tuple[list[str], str]:
    required = {
        "schema",
        "sealed_run_id",
        "run_root",
        "paper",
        "provider",
        "paper_write_probe",
        "parser",
        "page_count",
        "pages",
    }
    if set(capture) != required or capture["schema"] != CP1_CAPTURE_SCHEMA:
        fail("CP1_CAPTURE_INVALID", "protected paper extraction schema differs")
    pages_value = capture["pages"]
    count = capture["page_count"]
    if type(count) is not int or not isinstance(pages_value, list) or len(pages_value) != count:
        fail("CP1_CAPTURE_INVALID", "protected paper extraction page count differs")
    pages: list[str] = []
    for ordinal, row in enumerate(pages_value, start=1):
        if (
            not isinstance(row, dict)
            or set(row) != {"page", "layout_text", "layout_text_sha256"}
            or row.get("page") != ordinal
            or not isinstance(row.get("layout_text"), str)
        ):
            fail("CP1_CAPTURE_INVALID", "protected paper extraction page row differs")
        text = row["layout_text"].replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
        if not text:
            fail("CP1_CAPTURE_INVALID", "protected paper extraction page text is empty")
        if row["layout_text_sha256"] != sha256_bytes(row["layout_text"].encode("utf-8")):
            fail("CP1_CAPTURE_INVALID", "protected paper extraction page digest differs")
        pages.append(text)
    raw = ("\n\f\n".join(pages) + "\n").encode("utf-8")
    return pages, sha256_bytes(raw)


def _section_for_result(row: Mapping[str, object]) -> str:
    locator = row.get("locator")
    page = row.get("page")
    if not isinstance(locator, dict) or type(page) is not int:
        fail("CP1_INVENTORY_CONVERSION_INVALID", "paper result location is invalid")
    table = locator.get("table")
    figure = locator.get("figure")
    if type(table) is int:
        return f"paper:table:{table}"
    if type(figure) is int:
        return f"paper:figure:{figure}"
    return f"paper:page:{page}"


def _registry_ids(prefix: str, binding_sha256: str, count: int) -> tuple[str, ...]:
    require_sha256(binding_sha256, "registry binding sha256")
    rows = []
    for ordinal in range(count):
        binding = sha256_bytes(
            canonical_json(
                {
                    "schema": "experiments7-cp1-neutral-registry-slot/v6",
                    "binding_sha256": binding_sha256,
                    "ordinal": ordinal,
                }
            )
        )
        rows.append(f"{prefix}:{ordinal:03d}:{binding}")
    if rows != sorted(rows) or len(rows) != len(set(rows)):
        fail("CP1_REGISTRY_INVALID", "neutral registry IDs are not canonical")
    return tuple(rows)


def prepare_cp1(
    capture: Mapping[str, object],
    *,
    source_pre_sha256: str,
) -> PreparedCP1:
    """Build CP1's exact immutable rows entirely from captured layout text."""

    pages, layout_sha256 = _normalized_pages(capture)
    try:
        source_structure, source_inventory = build_inventory_from_layout(
            pages, layout_sha256
        )
    except Exception as exc:
        fail("CP1_PAPER_INVENTORY_INVALID", str(exc))
    section_ids: set[str] = set()
    structure: list[dict[str, object]] = []
    for row in source_structure:
        structure_id = row.get("structure_id")
        if not isinstance(structure_id, str) or not structure_id:
            fail("CP1_INVENTORY_CONVERSION_INVALID", "paper structure ID is invalid")
        section_id = f"paper:{structure_id}"
        if section_id in section_ids:
            fail("CP1_INVENTORY_CONVERSION_INVALID", "paper structure section repeats")
        section_ids.add(section_id)
        structure.append({**row, "paper_section_id": section_id})

    results: list[dict[str, object]] = []
    result_ids: set[str] = set()
    for row in source_inventory.results:
        result_id = row.get("result_id")
        if not isinstance(result_id, str) or not result_id or result_id in result_ids:
            fail("CP1_INVENTORY_CONVERSION_INVALID", "paper result ID is invalid")
        result_ids.add(result_id)
        section_id = _section_for_result(row)
        if section_id not in section_ids:
            fail("CP1_INVENTORY_CONVERSION_INVALID", "paper result lacks a structure section")
        results.append({**row, "paper_section_id": section_id})

    aliases: list[dict[str, object]] = []
    alias_ids: set[str] = set()
    for row in source_inventory.aliases:
        alias_id = row.get("alias_id")
        target = row.get("alias_of")
        if (
            not isinstance(alias_id, str)
            or not alias_id
            or alias_id in alias_ids
            or not isinstance(target, str)
            or target not in result_ids
        ):
            fail("CP1_INVENTORY_CONVERSION_INVALID", "paper alias endpoint is invalid")
        alias_ids.add(alias_id)
        aliases.append({**row, "target_result_id": target})

    if (
        len(structure) != CP1_STRUCTURE_COUNT
        or len(results) != CP1_RESULT_COUNT
        or len(aliases) != CP1_ALIAS_COUNT
    ):
        fail("CP1_PAPER_INVENTORY_INVALID", "captured inventory cardinality differs")
    try:
        inventory_digest(structure, results, aliases)
    except Exception as exc:
        fail("CP1_PAPER_INVENTORY_INVALID", str(exc))
    structure_payload = _canonical_jsonl(structure)
    results_payload = _canonical_jsonl(results)
    aliases_payload = _canonical_jsonl(aliases)
    return PreparedCP1(
        tuple(structure),
        tuple(results),
        tuple(aliases),
        _registry_ids("lineage:source-pre-shard", source_pre_sha256, 129),
        _registry_ids("profile:paper-layout-slot", layout_sha256, 30),
        layout_sha256,
        dict(sorted(source_inventory.source_counts.items())),
        dict(sorted(_counts_by(source_inventory.aliases, "reason").items())),
        structure_payload,
        results_payload,
        aliases_payload,
    )


def _counts_by(rows: Iterable[Mapping[str, object]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[field])
        counts[value] = counts.get(value, 0) + 1
    return counts


def _artifact_ref(publication: Publication) -> dict[str, str]:
    return {
        "relative_path": publication.evidence.relative_path,
        "artifact_evidence_sha256": sha256_bytes(
            canonical_json(publication.evidence.to_dict())
        ),
    }


def _stage_bundle(stage: Iterable[Publication]) -> dict[str, object]:
    return {
        "publications": [
            {
                "evidence": publication.evidence.to_dict(),
                "transcript": publication.transcript,
            }
            for publication in stage
        ],
    }


def _read_run_json(
    strict_parent: str, sealed_run_id: str, run_root: str, relative: str
) -> tuple[dict[str, object], bytes]:
    with open_run_handle(strict_parent, sealed_run_id, run_root) as handle:
        raw, _ = read_regular_at(handle.root_fd, relative)
    return _canonical_object(raw, relative), raw


def _cp0_inputs(
    strict_parent: str, sealed_run_id: str, run_root: str
) -> tuple[dict[str, object], bytes, ArtifactEvidence, ArtifactEvidence]:
    payload, raw = _read_run_json(
        strict_parent, sealed_run_id, run_root, "checkpoints/cp0.json"
    )
    policy = default_writer_policy()
    validate_checkpoint_payload(
        payload,
        expected_sealed_run_id=sealed_run_id,
        expected_run_root=run_root,
        expected_checkpoint=0,
        policy=policy,
    )
    stage = {
        evidence.relative_path: evidence
        for evidence in (
            ArtifactEvidence.from_dict(value)
            for value in payload["stage_actual_paths"]
        )
    }
    source = stage.get("manifests/source-pre.jsonl")
    paper = stage.get("manifests/paper-pre.json")
    if source is None or paper is None or source.artifact_type != "regular" or paper.artifact_type != "regular":
        fail("CP1_CP0_INPUT_INVALID", "CP0 source-pre or paper-pre evidence is absent")
    return payload, raw, source, paper


def _load_cp0_paper_pre(
    strict_parent: str, sealed_run_id: str, run_root: str, expected: ArtifactEvidence
) -> dict[str, object]:
    value, raw = _read_run_json(
        strict_parent, sealed_run_id, run_root, "manifests/paper-pre.json"
    )
    if sha256_bytes(raw) != expected.sha256:
        fail("CP1_CP0_INPUT_INVALID", "CP0 paper-pre bytes differ from evidence")
    required = {
        "schema",
        "sealed_run_id",
        "run_root",
        "paper_path",
        "sha256",
        "size",
        "descriptor_identity",
        "parent_identity",
        "provider",
    }
    if set(value) != required or value.get("schema") != "experiments7-paper-pre/v6":
        fail("CP1_CP0_INPUT_INVALID", "CP0 paper-pre schema differs")
    return value


def _load_source_pre_count(
    strict_parent: str, sealed_run_id: str, run_root: str, expected: ArtifactEvidence
) -> int:
    with open_run_handle(strict_parent, sealed_run_id, run_root) as handle:
        raw, _ = read_regular_at(handle.root_fd, "manifests/source-pre.jsonl")
    if sha256_bytes(raw) != expected.sha256:
        fail("CP1_CP0_INPUT_INVALID", "CP0 source-pre bytes differ from evidence")
    lines = raw.splitlines()
    if not lines:
        fail("CP1_CP0_INPUT_INVALID", "CP0 source-pre is empty")
    for index, line in enumerate(lines, start=1):
        try:
            value = strict_json_loads(line + b"\n", f"source-pre line {index}")
        except Exception as exc:
            fail("CP1_CP0_INPUT_INVALID", str(exc))
        if not isinstance(value, dict) or canonical_json(value) != line + b"\n":
            fail("CP1_CP0_INPUT_INVALID", "CP0 source-pre line is noncanonical")
    return len(lines)


def _check_capture_binding(
    capture: Mapping[str, object],
    execution: Mapping[str, object],
    *,
    capture_sha256: str,
    sealed_run_id: str,
    run_root: str,
    paper_pre: Mapping[str, object],
) -> None:
    if capture.get("sealed_run_id") != sealed_run_id or capture.get("run_root") != run_root:
        fail("CP1_CAPTURE_INVALID", "protected extraction run binding differs")
    paper = capture.get("paper")
    probe = capture.get("paper_write_probe")
    provider = capture.get("provider")
    if (
        not isinstance(paper, dict)
        or paper.get("sha256") != paper_pre.get("sha256")
        or paper.get("bytes") != paper_pre.get("size")
        or not isinstance(probe, dict)
        or probe.get("denied") is not True
        or not isinstance(provider, dict)
        or provider.get("writable_directories") != []
    ):
        fail("CP1_CAPTURE_INVALID", "protected extraction does not bind readonly CP0 paper")
    execution_required = {
        "schema",
        "sealed_run_id",
        "run_root",
        "argv",
        "environment",
        "extractor",
        "provider",
        "expected_paper_sha256",
        "child_stdout_sha256",
        "child_stdout_bytes",
        "extraction_sha256",
        "page_count",
    }
    if set(execution) != execution_required or execution.get("schema") != CP1_CAPTURE_EXECUTION_SCHEMA:
        fail("CP1_CAPTURE_INVALID", "protected extraction execution schema differs")
    if (
        execution.get("sealed_run_id") != sealed_run_id
        or execution.get("run_root") != run_root
        or execution.get("extraction_sha256") != capture_sha256
        or execution.get("expected_paper_sha256") != paper_pre.get("sha256")
        or execution.get("page_count") != capture.get("page_count")
    ):
        fail("CP1_CAPTURE_INVALID", "protected extraction execution binding differs")


def _checkpoint_publication_from_registry(
    registry: ExternalTranscriptRegistry,
    transcript_sha256: str,
    policy: WriterPolicy,
) -> Publication:
    transcript = registry.get_exact(transcript_sha256)
    if not isinstance(transcript, dict):
        fail("CP1_PREDECESSOR_INVALID", "CP0 checkpoint transcript is not an object")
    writer = WriterIdentity.from_dict(transcript.get("writer"))
    relative = transcript.get("relative_path")
    if not isinstance(relative, str):
        fail("CP1_PREDECESSOR_INVALID", "CP0 checkpoint transcript path is invalid")
    evidence = ArtifactEvidence(
        relative,
        str(transcript.get("artifact_type")),
        transcript.get("sha256"),
        transcript.get("bytes"),
        dict(transcript.get("identity", {})),
        writer,
        policy.authorize(relative, writer).rule_id,
        transcript_sha256,
    )
    return Publication(evidence, transcript)


@dataclass(frozen=True)
class CP1InputSnapshot:
    capture: dict[str, object]
    execution: dict[str, object]
    capture_sha256: str
    capture_bytes: int
    execution_sha256: str
    cp0_transcript_sha256: str


def _snapshot_cp1_inputs(
    paths: Mapping[str, str],
    *,
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
) -> CP1InputSnapshot:
    records = tuple(
        ExternalTranscriptRecord(paths[key])
        for key in ("capture", "capture_execution", "cp0_transcript")
    )
    with ExternalTranscriptRegistry(
        records,
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
    ) as registry:
        capture_doc = registry.documents[paths["capture"]]
        execution_doc = registry.documents[paths["capture_execution"]]
        cp0_transcript_doc = registry.documents[paths["cp0_transcript"]]
        capture = capture_doc.value
        execution = execution_doc.value
        cp0_transcript = cp0_transcript_doc.value
        if not isinstance(capture, dict) or not isinstance(execution, dict):
            fail("CP1_CAPTURE_INVALID", "external capture evidence must be objects")
        if not isinstance(cp0_transcript, dict):
            fail("CP1_PREDECESSOR_INVALID", "CP0 external transcript is invalid")
        return CP1InputSnapshot(
            capture=capture,
            execution=execution,
            capture_sha256=capture_doc.sha256,
            capture_bytes=len(capture_doc.raw_bytes),
            execution_sha256=execution_doc.sha256,
            cp0_transcript_sha256=transcript_hash(cp0_transcript),
        )


def bootstrap_cp1(
    *,
    strict_parent: str,
    sealed_run_id: str,
    run_root: str,
    external_capture: str,
    external_capture_execution: str,
    external_cp0_transcript: str,
    external_task_evidence: str,
    external_stage_bundle: str,
    external_cp1_transcript: str,
    external_stage_transcripts: str,
) -> CP1BootstrapResult:
    """Seal CP1 after validating fresh readonly paper extraction evidence."""

    sealed_run_id = validate_sealed_run_id(sealed_run_id)
    strict_parent = validate_absolute_path_text(strict_parent, "strict_parent")
    run_root = validate_absolute_path_text(run_root, "run_root")
    if run_root != f"{strict_parent}/{sealed_run_id}":
        fail("RUN_BINDING_MISMATCH", "CP1 run root must be the explicit strict child")
    paths = {
        "capture": validate_absolute_path_text(external_capture, "external_capture"),
        "capture_execution": validate_absolute_path_text(external_capture_execution, "external_capture_execution"),
        "cp0_transcript": validate_absolute_path_text(external_cp0_transcript, "external_cp0_transcript"),
        "task_evidence": validate_absolute_path_text(external_task_evidence, "external_task_evidence"),
        "stage_bundle": validate_absolute_path_text(external_stage_bundle, "external_stage_bundle"),
        "cp1_transcript": validate_absolute_path_text(external_cp1_transcript, "external_cp1_transcript"),
        "stage_transcripts": validate_absolute_path_text(external_stage_transcripts, "external_stage_transcripts"),
    }
    if len(set(paths.values())) != len(paths):
        fail("EXTERNAL_OUTPUT_PATH_COLLISION", "CP1 external paths must all be distinct")
    policy = default_writer_policy()
    _cp0_payload, cp0_raw, source_pre, paper_pre_evidence = _cp0_inputs(
        strict_parent, sealed_run_id, run_root
    )
    paper_pre = _load_cp0_paper_pre(
        strict_parent, sealed_run_id, run_root, paper_pre_evidence
    )
    source_pre_count = _load_source_pre_count(
        strict_parent, sealed_run_id, run_root, source_pre
    )

    inputs = _snapshot_cp1_inputs(
        paths,
        strict_parent=strict_parent,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
    )
    capture = inputs.capture
    execution = inputs.execution
    cp0_transcript_sha256 = inputs.cp0_transcript_sha256
    _check_capture_binding(
        capture,
        execution,
        capture_sha256=inputs.capture_sha256,
        sealed_run_id=sealed_run_id,
        run_root=run_root,
        paper_pre=paper_pre,
    )
    prepared = prepare_cp1(capture, source_pre_sha256=source_pre.sha256 or "")
    parser = capture.get("parser")
    provider = capture.get("provider")
    paper = capture.get("paper")
    if not isinstance(parser, dict) or not isinstance(provider, dict) or not isinstance(paper, dict):
        fail("CP1_CAPTURE_INVALID", "capture paper/parser/provider evidence is invalid")
    task_evidence = {
        "schema": CP1_TASK_EVIDENCE_SCHEMA,
        "sealed_run_id": sealed_run_id,
        "run_root": run_root,
        "cp0": {
            "checkpoint_sha256": sha256_bytes(cp0_raw),
            "source_pre": source_pre.to_dict(),
            "paper_pre": paper_pre_evidence.to_dict(),
            "source_pre_record_count": source_pre_count,
        },
        "protected_capture": {
            "path": paths["capture"],
            "sha256": inputs.capture_sha256,
            "bytes": inputs.capture_bytes,
            "execution_path": paths["capture_execution"],
            "execution_sha256": inputs.execution_sha256,
            "paper": paper,
            "provider": provider,
            "parser": parser,
        },
        "builder": {
            "path": str(Path(__file__).with_name("cp1_inventory_builder.py")),
            "sha256": sha256_bytes(
                Path(__file__).with_name("cp1_inventory_builder.py").read_bytes()
            ),
            "layout_sha256": prepared.layout_sha256,
        },
        "neutral_registry": {
            "lineage_count": len(prepared.lineage_ids),
            "lineage_rule": "source-pre SHA-256-bound ordinal shards 000 through 128",
            "profile_count": len(prepared.profile_ids),
            "profile_rule": "paper-layout SHA-256-bound neutral replay slots 000 through 029",
        },
        "inventory": {
            "structure_count": len(prepared.structure),
            "result_count": len(prepared.results),
            "alias_count": len(prepared.aliases),
            "structure_sha256": sha256_bytes(prepared.structure_payload),
            "results_sha256": sha256_bytes(prepared.results_payload),
            "aliases_sha256": sha256_bytes(prepared.aliases_payload),
            "result_source_counts": dict(prepared.source_counts),
            "alias_reason_counts": dict(prepared.alias_reason_counts),
        },
        "protected_writes": {
            "experiments4": 0,
            "experiments5": 0,
            "experiments6": 0,
            "paper": 0,
        },
    }
    task_evidence_sha256 = sha256_bytes(canonical_json(task_evidence))
    task_binding = {
        "schema": CP1_TASK_BINDING_SCHEMA,
        "task_evidence_path": paths["task_evidence"],
        "task_evidence_sha256": task_evidence_sha256,
        "protected_capture_sha256": inputs.capture_sha256,
        "cp0_checkpoint_sha256": sha256_bytes(cp0_raw),
    }
    with ExitStack() as output_stack:
        task_output = output_stack.enter_context(
            ExternalOutputReservation.create_for_existing_run(
                paths["task_evidence"],
                strict_parent=strict_parent,
                sealed_run_id=sealed_run_id,
                run_root=run_root,
            )
        )
        stage_bundle_output = output_stack.enter_context(
            ExternalOutputReservation.create_for_existing_run(
                paths["stage_bundle"],
                strict_parent=strict_parent,
                sealed_run_id=sealed_run_id,
                run_root=run_root,
            )
        )
        checkpoint_output = output_stack.enter_context(
            ExternalOutputReservation.create_for_existing_run(
                paths["cp1_transcript"],
                strict_parent=strict_parent,
                sealed_run_id=sealed_run_id,
                run_root=run_root,
            )
        )
        stage_transcript_output = output_stack.enter_context(
            ExternalOutputReservation.create_for_existing_run(
                paths["stage_transcripts"],
                strict_parent=strict_parent,
                sealed_run_id=sealed_run_id,
                run_root=run_root,
            )
        )
        task_output.publish_json(task_evidence)

        registry_writer = WriterIdentity(
            "registry", f"registry:cp1:{sealed_run_id}", "registry-writer"
        )
        inventory_writer = WriterIdentity(
            "inventory", f"inventory:cp1:{sealed_run_id}", "inventory-writer"
        )
        registry_publisher = StagePublisher(
            strict_parent, sealed_run_id, run_root, registry_writer, policy
        )
        inventory_publisher = StagePublisher(
            strict_parent, sealed_run_id, run_root, inventory_writer, policy
        )
        stage: list[Publication] = []
        lineage_created = registry_publisher.publish_json(
            "registry/lineage-basis.json",
            {
                "schema": CP1_LINEAGE_SCHEMA,
                "sealed_run_id": sealed_run_id,
                "run_root": run_root,
                "lineage_ids": list(prepared.lineage_ids),
            },
            context=task_binding,
        )
        stage.extend(lineage_created)
        lineage_publication = lineage_created[-1]
        profile_created = registry_publisher.publish_json(
            "registry/profile-basis.json",
            {
                "schema": CP1_PROFILE_SCHEMA,
                "sealed_run_id": sealed_run_id,
                "run_root": run_root,
                "profile_ids": list(prepared.profile_ids),
            },
            context=task_binding,
        )
        stage.extend(profile_created)
        profile_publication = profile_created[-1]
        stage.extend(
            registry_publisher.publish_json(
                "registry/cp2-basis.json",
                {
                    "schema": CP1_BASIS_SCHEMA,
                    "sealed_run_id": sealed_run_id,
                    "run_root": run_root,
                    "lineage_source": _artifact_ref(lineage_publication),
                    "profile_source": _artifact_ref(profile_publication),
                },
                context=task_binding,
            )
        )
        for relative, payload in (
            ("inventory/structure.jsonl", prepared.structure_payload),
            ("inventory/results.jsonl", prepared.results_payload),
            ("inventory/aliases.jsonl", prepared.aliases_payload),
        ):
            stage.extend(
                inventory_publisher.publish_bytes(relative, payload, context=task_binding)
            )
        stage_bundle_output.publish_json(_stage_bundle(stage))
        records = [ExternalTranscriptRecord(paths["cp0_transcript"])]
        records.extend(
            ExternalTranscriptRecord(
                paths["stage_bundle"], ("publications", index, "transcript")
            )
            for index in range(len(stage))
        )
        with ExternalTranscriptRegistry(
            records,
            strict_parent=strict_parent,
            sealed_run_id=sealed_run_id,
            run_root=run_root,
        ) as transcript_registry:
            predecessor = _checkpoint_publication_from_registry(
                transcript_registry, cp0_transcript_sha256, policy
            )
            if predecessor.evidence.writer.role != "checkpoint_controller":
                fail("CP1_PREDECESSOR_INVALID", "CP0 controller identity is invalid")
            checkpoint = publish_checkpoint(
                strict_parent,
                sealed_run_id,
                run_root,
                1,
                stage,
                predecessor.evidence.writer,
                policy,
                transcript_registry=transcript_registry,
                previous_checkpoint_publication=predecessor,
            )
        checkpoint_output.publish_json(checkpoint.transcript)
        stage_transcript_output.publish_json(
            {"transcripts": [publication.transcript for publication in checkpoint.stage_publications]}
        )
        return CP1BootstrapResult(
            sealed_run_id,
            run_root,
            checkpoint,
            tuple(stage),
            task_evidence_sha256,
            paths,
        )
