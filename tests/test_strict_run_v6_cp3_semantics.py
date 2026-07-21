from __future__ import annotations

import copy
from dataclasses import replace
from fractions import Fraction
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from provenance.strict_v6 import (  # noqa: E402
    CLOSURE_FIELDS,
    V6ContractError,
    digest_records,
    result_inventory_digest,
    make_bucket_subseal,
    make_provenance_edge,
    make_provenance_node,
    make_provenance_recomputation,
    make_provenance_rounding_proof,
    make_raw_provenance_node,
    reconcile_cp3_buckets,
)
from strict_run import (  # noqa: E402
    ExternalTranscriptRecord,
    ExternalTranscriptRegistry,
    StagePublisher,
    StrictRunError,
    WriterIdentity,
    canonical_json,
    default_writer_policy,
    sha256_bytes,
)
from strict_run.cp3_semantic import validate_cp3_semantics  # noqa: E402
from strict_run.publication import Publication  # noqa: E402
from strict_run.semantics import (  # noqa: E402
    DENIAL_PROBES,
    DISPATCH_STAGES,
    validate_cp2_artifact_structure,
    validate_stage_semantics,
)


RUN_ID = "exp7-strict-v6-20260717T150000Z-" + "3" * 32
RESULT_COUNT = 1908
SECTION_COUNT = 8
_DELETE = object()


def _positive_record(
    number: int,
    section: str,
    closure: dict[str, list[str]],
    *,
    raw_ids: tuple[str, ...] = (),
) -> dict[str, object]:
    contributors = sorted(
        {*raw_ids, *(value for values in closure.values() for value in values)}
    )
    return {
        "schema": "experiments7-admission-result/v6",
        "result_id": f"result:synthetic:{number:04d}",
        "paper_section_id": section,
        "phase": "precopy",
        "state": "ADMITTED_FOR_COPY",
        "result_kind": "inference" if raw_ids else "non_inference",
        "copy_required": bool(raw_ids),
        "raw_artifact_ids": list(raw_ids),
        "candidate_raw_artifact_ids": [],
        "required_parent_result_ids": [],
        "closed_parent_result_ids": [],
        "required_contributor_node_ids": contributors,
        "sealed_contributor_node_ids": contributors,
        "positive_evidence_origin": "run_local_strict",
        "source_pre_bound": bool(raw_ids),
        "declared_drift": False,
        "ambiguous": False,
        "support_only": False,
        "audit_positive_evidence": False,
        "reason_codes": [],
        "copied_artifact_ids": [],
        "copied_to_edge_ids": [],
        **closure,
    }


class CP3Fixture:
    def __init__(
        self,
        base: Path,
        *,
        result_count: int = RESULT_COUNT,
        section_count: int = SECTION_COUNT,
        swap_sections: bool = False,
        omit_last: bool = False,
        duplicate_record: bool = False,
        extra_record: bool = False,
        bool_reconciliation_count: bool = False,
        stale_inventory_evidence: bool = False,
        bucket_schema: str | None = None,
        omit_stage_directories: bool = False,
        extra_provenance: bool = False,
        hide_extra_provenance_publication: bool = False,
        bucket_filename_mismatch: bool = False,
        empty_graph_with_opaque_ids: bool = False,
        invalid_reason_codes: bool = False,
        extra_result_key: bool = False,
        bad_recomputation: bool = False,
        bad_rounding: bool = False,
        legacy_checkpoint: int | None = None,
        raw_mutation: tuple[str, object] | None = None,
        raw_from_snapshot: bool = False,
        rational_rounding: bool = False,
        cp1_directory_before_source_pre: bool = False,
        cp2_directory_before_cp1: bool = False,
        checkpoint_mutation: tuple[int, str, object] | None = None,
        semantic_mutation: tuple[int, str, object] | None = None,
        stage_evidence_mutation: tuple[int, str, str, object] | None = None,
    ) -> None:
        self.base = base
        self.parent = base / "strict-runs"
        self.run_root = self.parent / RUN_ID
        self.run_root.mkdir(parents=True)
        self.policy = default_writer_policy()
        self.inventory_writer = WriterIdentity(
            "inventory", "inventory:cp3-fixture", "inventory-writer"
        )
        self.admission_writer = WriterIdentity(
            "admission", "admission:cp3-fixture", "admission-writer"
        )
        self.provenance_writer = WriterIdentity(
            "provenance", "provenance:cp3-fixture", "provenance-writer"
        )
        self.snapshot_writer = WriterIdentity(
            "adapter_snapshot", "snapshot:cp3-fixture", "adapter_snapshot-writer"
        )
        self.controller = WriterIdentity(
            "checkpoint_controller", "checkpoint:cp3-fixture", "checkpoint_controller-writer"
        )
        self.g0_writer = WriterIdentity("g0", "g0:cp3-fixture", "g0-writer")
        self.registry_writer = WriterIdentity(
            "registry", "registry:cp3-fixture", "registry-writer"
        )
        self.external_counter = 0
        cp1_prepublished = (
            self.publisher(self.registry_writer).ensure_directory("registry")
            if cp1_directory_before_source_pre
            else []
        )
        raw_payload = b"synthetic-raw-inference"
        self.source_pre_record = {
            "schema": "experiments7-source-pre/v6",
            "record_id": "source:synthetic-raw-inference",
            "root_id": "experiments4",
            "relative_path": "synthetic/output.json",
            "type": "regular",
            "sha256": hashlib.sha256(raw_payload).hexdigest(),
            "size": len(raw_payload),
            "descriptor_identity": {
                "file_type": "regular",
                "st_dev": 100,
                "st_ino": 200,
                "mode": 0o100444,
                "size": len(raw_payload),
                "mtime_ns": 300,
            },
        }
        source_pre_publications = self.publisher(self.g0_writer).publish_bytes(
            "manifests/source-pre.jsonl",
            canonical_json(self.source_pre_record),
        )
        self.source_pre_publication = source_pre_publications[-1]
        cp2_prepublished = (
            self.publisher(self.snapshot_writer).ensure_directory("snapshots")
            if cp2_directory_before_cp1
            else []
        )
        inventory = [
            {
                "result_id": f"result:synthetic:{number:04d}",
                "paper_section_id": f"section-{number % section_count:02d}",
                "numeric_value": 129,
                "display_value": "129",
            }
            for number in range(result_count)
        ]
        if rational_rounding:
            inventory[0]["numeric_value"] = "0.33"
            inventory[0]["display_value"] = "0.33"
        (
            self.cp1_stage_publications,
            self.cp1_publication,
            cp1_basis_publication,
        ) = self.publish_cp1(
            inventory,
            prepublished_directories=cp1_prepublished,
            checkpoint_mutation=checkpoint_mutation,
            semantic_mutation=semantic_mutation,
            stage_evidence_mutation=stage_evidence_mutation,
            legacy_checkpoint=legacy_checkpoint,
            stale_inventory_evidence=stale_inventory_evidence,
        )
        (
            self.cp2_stage_publications,
            self.cp2_publication,
            snapshot_report_publication,
        ) = self.publish_cp2(
            self.cp1_publication,
            cp1_basis_publication,
            prepublished_directories=cp2_prepublished,
            checkpoint_mutation=checkpoint_mutation,
            semantic_mutation=semantic_mutation,
            stage_evidence_mutation=stage_evidence_mutation,
            legacy_checkpoint=legacy_checkpoint,
        )
        snapshot_evidence = snapshot_report_publication.evidence

        evidence_sha256 = sha256_bytes(canonical_json(snapshot_evidence.to_dict()))
        common_nodes = {
            field: make_provenance_node(
                node_type,
                snapshot_evidence.relative_path,
                evidence_sha256,
            )
            for field, node_type in {
                "producer_node_ids": "producer",
                "config_node_ids": "config",
                "script_node_ids": "script",
                "runtime_node_ids": "runtime",
            }.items()
        }
        nodes: dict[str, dict[str, object]] = {
            str(node["node_id"]): node for node in common_nodes.values()
        }
        recomputations: dict[str, dict[str, object]] = {}
        rounding_proofs: dict[str, dict[str, object]] = {}
        edges: dict[str, dict[str, object]] = {}
        records: list[dict[str, object]] = []
        relation_by_field = {
            "producer_node_ids": "producer_supports_result",
            "config_node_ids": "config_supports_result",
            "script_node_ids": "script_supports_result",
            "runtime_node_ids": "runtime_supports_result",
            "input_node_ids": "input_supports_result",
            "recomputation_node_ids": "recomputation_supports_result",
            "rounding_proof_node_ids": "rounding_proof_supports_result",
        }
        for number, inventory_row in enumerate(inventory):
            section = str(inventory_row["paper_section_id"])
            if empty_graph_with_opaque_ids:
                closure = {
                    field: [f"{field}:{number}"] for field in CLOSURE_FIELDS
                }
            else:
                input_numeric = 129
                input_node = make_provenance_node(
                    "input",
                    snapshot_evidence.relative_path,
                    evidence_sha256,
                    json_pointer=["lineage_count"],
                    numeric_value=str(input_numeric),
                )
                nodes[str(input_node["node_id"])] = input_node
                coefficient = "2" if bad_recomputation and number == 1 else "1"
                denominator = "387" if rational_rounding and number == 0 else "1"
                recomputation = make_provenance_recomputation(
                    [
                        {
                            "input_node_id": input_node["node_id"],
                            "coefficient": coefficient,
                        }
                    ],
                    denominator,
                )
                recomputations[str(recomputation["recomputation_id"])] = recomputation
                unrounded = Fraction(
                    input_numeric * int(coefficient), int(denominator)
                )
                places = (
                    2
                    if rational_rounding and number == 0
                    else 1
                    if bad_rounding and number == 1
                    else 0
                )
                rounding = make_provenance_rounding_proof(
                    str(recomputation["recomputation_id"]),
                    unrounded.numerator,
                    unrounded.denominator,
                    places,
                )
                rounding_proofs[str(rounding["rounding_proof_id"])] = rounding
                closure = {
                    "producer_node_ids": [str(common_nodes["producer_node_ids"]["node_id"])],
                    "config_node_ids": [str(common_nodes["config_node_ids"]["node_id"])],
                    "script_node_ids": [str(common_nodes["script_node_ids"]["node_id"])],
                    "runtime_node_ids": [str(common_nodes["runtime_node_ids"]["node_id"])],
                    "input_node_ids": [str(input_node["node_id"])],
                    "recomputation_node_ids": [str(recomputation["recomputation_id"])],
                    "rounding_proof_node_ids": [str(rounding["rounding_proof_id"])],
                }
            raw_ids: tuple[str, ...] = ()
            if not empty_graph_with_opaque_ids and number == 0:
                if raw_from_snapshot:
                    raw_node = make_provenance_node(
                        "config",
                        snapshot_evidence.relative_path,
                        evidence_sha256,
                    )
                    raw_node["node_type"] = "raw_artifact"
                else:
                    raw_node = make_raw_provenance_node(self.source_pre_record)
                if raw_mutation is not None:
                    field, value = raw_mutation
                    raw_node[field] = value
                nodes[str(raw_node["node_id"])] = raw_node
                raw_ids = (str(raw_node["node_id"]),)
            record = _positive_record(number, section, closure, raw_ids=raw_ids)
            records.append(record)
            if not empty_graph_with_opaque_ids:
                for field, relation in relation_by_field.items():
                    for source_id in record[field]:
                        edge = make_provenance_edge(
                            str(source_id), str(record["result_id"]), relation
                        )
                        edges[str(edge["edge_id"])] = edge
                for source_id in record["raw_artifact_ids"]:
                    edge = make_provenance_edge(
                        str(source_id),
                        str(record["result_id"]),
                        "raw_artifact_supports_result",
                    )
                    edges[str(edge["edge_id"])] = edge

        if swap_sections and len(records) >= 2:
            records[0]["paper_section_id"], records[1]["paper_section_id"] = (
                records[1]["paper_section_id"],
                records[0]["paper_section_id"],
            )
        selected = records[:-1] if omit_last else records
        if extra_record:
            extra_closure = {
                field: [f"extra-{field}"] for field in CLOSURE_FIELDS
            }
            selected = [
                *selected,
                _positive_record(result_count, "section-00", extra_closure),
            ]

        inv_hash = result_inventory_digest(inventory)
        buckets = [
            make_bucket_subseal(
                inv_hash,
                section,
                [row for row in selected if row["paper_section_id"] == section],
            )
            for section in sorted({str(row["paper_section_id"]) for row in selected})
        ]
        if duplicate_record:
            bucket = buckets[0]
            rows = bucket["result_records"]
            assert isinstance(rows, list)
            rows.append(copy.deepcopy(rows[0]))
            rows.sort(key=lambda row: row["result_id"])
            bucket["result_ids"] = [row["result_id"] for row in rows]
            bucket["result_records_sha256"] = digest_records(rows)
        if invalid_reason_codes:
            buckets[0]["result_records"][0]["reason_codes"] = "not-a-list"
            buckets[0]["result_records_sha256"] = digest_records(
                buckets[0]["result_records"]
            )
        if extra_result_key:
            buckets[0]["result_records"][0]["undeclared"] = True
            buckets[0]["result_records_sha256"] = digest_records(
                buckets[0]["result_records"]
            )
        if bucket_schema is not None:
            buckets[0]["schema"] = bucket_schema
        self.stage: list[Publication] = []
        graph_payloads = {
            "provenance/nodes.jsonl": []
            if empty_graph_with_opaque_ids
            else [nodes[key] for key in sorted(nodes)],
            "provenance/edges.jsonl": []
            if empty_graph_with_opaque_ids
            else [edges[key] for key in sorted(edges)],
            "provenance/recomputations.jsonl": []
            if empty_graph_with_opaque_ids
            else [recomputations[key] for key in sorted(recomputations)],
            "provenance/rounding-proofs.jsonl": []
            if empty_graph_with_opaque_ids
            else [rounding_proofs[key] for key in sorted(rounding_proofs)],
        }
        for relative, rows in graph_payloads.items():
            payload = b"".join(canonical_json(row) for row in rows)
            self.stage += self.publisher(self.provenance_writer).publish_bytes(
                relative, payload
            )
        if extra_provenance:
            self.stage += self.publisher(self.provenance_writer).publish_json(
                "provenance/unchecked.json", {"unchecked": True}
            )
            if hide_extra_provenance_publication:
                self.stage = [
                    publication
                    for publication in self.stage
                    if publication.evidence.relative_path != "provenance/unchecked.json"
                ]
        for bucket in buckets:
            relative = f"admission/precopy/buckets/{bucket['paper_section_id']}.json"
            if bucket_filename_mismatch and bucket is buckets[0]:
                relative = "admission/precopy/buckets/not-the-section.json"
            self.stage += self.publisher(self.admission_writer).publish_json(
                relative, bucket
            )
        try:
            reconciliation = reconcile_cp3_buckets(
                inventory, buckets, expected_result_count=RESULT_COUNT
            )
        except V6ContractError:
            # Hostile fixtures still need a published reconciliation so the bridge
            # reaches and rejects the malformed bucket union itself.
            reconciliation = {
                "schema": "experiments7-cp3-admission-reconciliation/v6",
                "phase": "precopy",
                "checkpoint_state": "READY_FOR_CP3",
                "inventory_sha256": inv_hash,
                "total_result_count": RESULT_COUNT,
                "precopy_admitted_count": RESULT_COUNT,
                "precopy_unresolved_count": 0,
                "paper_section_bucket_count": len(buckets),
                "bucket_subseal_sha256s": {
                    str(bucket["paper_section_id"]): sha256_bytes(canonical_json(bucket))
                    for bucket in buckets
                },
                "copy_required_raw_artifact_ids": [],
                "copy_plan_input_sha256": sha256_bytes(canonical_json([])),
            }
        if bool_reconciliation_count:
            reconciliation["total_result_count"] = True
        self.stage += self.publisher(self.admission_writer).publish_json(
            "admission/precopy/reconciliation.json", reconciliation
        )
        if omit_stage_directories:
            self.stage = [
                publication
                for publication in self.stage
                if publication.evidence.artifact_type != "directory"
            ]

    def publisher(
        self, writer: WriterIdentity, *, allow_reserved_paths: bool = False
    ) -> StagePublisher:
        return StagePublisher(
            str(self.parent),
            RUN_ID,
            str(self.run_root),
            writer,
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

    def checkpoint_payload(
        self,
        checkpoint: int,
        publications: list[Publication],
        *,
        previous_sha256: str,
        previous_transcript_sha256: str,
        semantic_bindings: dict[str, object],
        checkpoint_mutation: tuple[int, str, object] | None,
        semantic_mutation: tuple[int, str, object] | None,
        stage_evidence_mutation: tuple[int, str, str, object] | None,
        legacy_checkpoint: int | None,
        stale_inventory_evidence: bool = False,
    ) -> dict[str, object]:
        semantic = copy.deepcopy(semantic_bindings)
        if semantic_mutation is not None and semantic_mutation[0] == checkpoint:
            _, field, value = semantic_mutation
            parts = field.split(".")
            target: object = semantic
            for part in parts[:-1]:
                if isinstance(target, list):
                    target = target[int(part)]
                else:
                    assert isinstance(target, dict)
                    target = target[part]
            last = parts[-1]
            if isinstance(target, list):
                index = int(last)
                if value is _DELETE:
                    target.pop(index)
                else:
                    target[index] = value
            else:
                assert isinstance(target, dict)
                if value is _DELETE:
                    target.pop(last, None)
                else:
                    target[last] = value
        stage = [
            publication.evidence.to_dict()
            for publication in sorted(
                publications, key=lambda item: item.evidence.relative_path
            )
        ]
        if stale_inventory_evidence and checkpoint == 1:
            inventory = next(
                row for row in stage if row["relative_path"] == "inventory/results.jsonl"
            )
            inventory["sha256"] = "f" * 64
        if (
            stage_evidence_mutation is not None
            and stage_evidence_mutation[0] == checkpoint
        ):
            _, relative_path, field, value = stage_evidence_mutation
            target = next(row for row in stage if row["relative_path"] == relative_path)
            if value == "SOURCE_PRE_TRANSCRIPT":
                value = (
                    self.source_pre_publication.evidence.publication_transcript_sha256
                )
            if value is _DELETE:
                target.pop(field, None)
            else:
                target[field] = value
        payload: dict[str, object] = {
            "schema": "experiments7-strict-checkpoint/v6",
            "sealed_run_id": RUN_ID,
            "run_root": str(self.run_root),
            "checkpoint": checkpoint,
            "previous_checkpoint_sha256": previous_sha256,
            "previous_checkpoint_transcript_sha256": previous_transcript_sha256,
            "stage_actual_paths": stage,
            "stage_actual_paths_sha256": sha256_bytes(canonical_json(stage)),
            "bindings": {},
            "semantic_bindings": semantic,
            "semantic_bindings_sha256": sha256_bytes(
                canonical_json(semantic)
            ),
        }
        if checkpoint_mutation is not None and checkpoint_mutation[0] == checkpoint:
            _, field, value = checkpoint_mutation
            if value is _DELETE:
                payload.pop(field, None)
            else:
                payload[field] = value
            if field == "semantic_bindings" and value is not _DELETE:
                payload["semantic_bindings_sha256"] = sha256_bytes(
                    canonical_json(value)
                )
        if legacy_checkpoint == checkpoint:
            payload.pop("semantic_bindings", None)
            payload.pop("semantic_bindings_sha256", None)
        return payload

    def publish_cp1(
        self,
        inventory: list[dict[str, object]],
        *,
        prepublished_directories: list[Publication],
        checkpoint_mutation: tuple[int, str, object] | None,
        semantic_mutation: tuple[int, str, object] | None,
        stage_evidence_mutation: tuple[int, str, str, object] | None,
        legacy_checkpoint: int | None,
        stale_inventory_evidence: bool,
    ) -> tuple[list[Publication], Publication, Publication]:
        registry = self.publisher(self.registry_writer)
        stage = list(prepublished_directories)
        lineage_ids = [f"lineage-{index:03d}" for index in range(129)]
        profile_ids = [f"profile-{index:02d}" for index in range(30)]
        created = registry.publish_json(
            "registry/lineage-basis.json",
            {
                "schema": "experiments7-cp1-lineage-basis/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "lineage_ids": lineage_ids,
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
                "profile_ids": profile_ids,
            },
        )
        stage.extend(created)
        profiles = created[-1]
        created = registry.publish_json(
            "registry/cp2-basis.json",
            {
                "schema": "experiments7-cp1-cp2-basis/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "lineage_source": self.ref(lineage),
                "profile_source": self.ref(profiles),
            },
        )
        stage.extend(created)
        basis = created[-1]
        structure = [
            {"paper_section_id": f"section-{index:02d}", "ordinal": index}
            for index in range(51)
        ]
        aliases = [
            {
                "alias_id": f"alias:cp3:{index:03d}",
                "target_result_id": f"result:synthetic:{index:04d}",
            }
            for index in range(86)
        ]
        inventory_publisher = self.publisher(self.inventory_writer)
        inventory_publication: Publication | None = None
        for path, rows in (
            ("inventory/structure.jsonl", structure),
            ("inventory/results.jsonl", inventory),
            ("inventory/aliases.jsonl", aliases),
        ):
            created = inventory_publisher.publish_bytes(
                path,
                b"".join(canonical_json(row) for row in rows),
            )
            stage.extend(created)
            if path == "inventory/results.jsonl":
                inventory_publication = created[-1]
        assert inventory_publication is not None
        root_fd = os.open(
            self.run_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            semantics = validate_stage_semantics(
                root_fd,
                RUN_ID,
                str(self.run_root),
                1,
                stage,
                "1" * 64,
                "2" * 64,
            )
        except StrictRunError:
            semantics = {
                "schema": "experiments7-cp1-semantics/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "checkpoint": 1,
                "lineage_count": 129,
                "profile_count": 30,
                "inventory_record_count": len(inventory),
            }
        finally:
            os.close(root_fd)
        payload = self.checkpoint_payload(
            1,
            stage,
            previous_sha256="1" * 64,
            previous_transcript_sha256="2" * 64,
            semantic_bindings=semantics,
            checkpoint_mutation=checkpoint_mutation,
            semantic_mutation=semantic_mutation,
            stage_evidence_mutation=stage_evidence_mutation,
            legacy_checkpoint=legacy_checkpoint,
            stale_inventory_evidence=stale_inventory_evidence,
        )
        checkpoint = self.publisher(
            self.controller, allow_reserved_paths=True
        ).publish_json("checkpoints/cp1.json", payload)[-1]
        return stage, checkpoint, basis

    def publish_cp2(
        self,
        cp1: Publication,
        basis: Publication,
        *,
        prepublished_directories: list[Publication],
        checkpoint_mutation: tuple[int, str, object] | None,
        semantic_mutation: tuple[int, str, object] | None,
        stage_evidence_mutation: tuple[int, str, str, object] | None,
        legacy_checkpoint: int | None,
    ) -> tuple[list[Publication], Publication, Publication]:
        cp1_sha256 = str(cp1.evidence.sha256)
        cp1_transcript_sha256 = cp1.evidence.publication_transcript_sha256
        basis_ref = self.ref(basis)
        publisher = self.publisher(self.snapshot_writer)
        stage = list(prepublished_directories)
        files: dict[str, Publication] = {}

        def publish_bytes(path: str, payload: bytes) -> Publication:
            created = publisher.publish_bytes(path, payload)
            stage.extend(created)
            files[path] = created[-1]
            return created[-1]

        def publish_json(path: str, value: object) -> Publication:
            created = publisher.publish_json(path, value)
            stage.extend(created)
            files[path] = created[-1]
            return created[-1]

        lineage_ids = [f"lineage-{index:03d}" for index in range(129)]
        profile_ids = [f"profile-{index:02d}" for index in range(30)]
        snapshot_records = []
        for lineage_id in lineage_ids:
            output = publish_bytes(
                f"snapshots/lineage/{lineage_id}.bin",
                f"snapshot:{lineage_id}\n".encode(),
            )
            snapshot_records.append(
                {"lineage_id": lineage_id, "output": self.ref(output)}
            )
        snapshot_report = publish_json(
            "snapshots/snapshot-producer-results.json",
            {
                "schema": "experiments7-cp2-snapshot-producer-results/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "cp1_sha256": cp1_sha256,
                "cp1_transcript_sha256": cp1_transcript_sha256,
                "cp1_basis": basis_ref,
                "lineage_count": len(lineage_ids),
                "records_sha256": sha256_bytes(canonical_json(snapshot_records)),
                "records": snapshot_records,
            },
        )
        dependency = publish_bytes(
            "runtime/dependencies/runtime.bin", b"pinned-runtime-dependency\n"
        )
        dependency_rows = [
            {"dependency_id": "runtime", "artifact": self.ref(dependency)}
        ]
        config_rows = []
        configs: dict[str, Publication] = {}
        for profile_id in profile_ids:
            config = publish_json(
                f"runtime/configs/{profile_id}.json",
                {
                    "schema": "experiments7-cp2-run-config/v6",
                    "profile_id": profile_id,
                    "dependency_ids": ["runtime"],
                    "dispatch_path": "production_external_contract",
                    "transport": "hermetic_replay",
                },
            )
            configs[profile_id] = config
            config_rows.append({"profile_id": profile_id, "config": self.ref(config)})
        denial_rows = []
        for probe_id in DENIAL_PROBES:
            denial = publish_json(
                f"runtime/denials/{probe_id}.json",
                {
                    "schema": "experiments7-cp2-denial-evidence/v6",
                    "probe_id": probe_id,
                    "observed_errno": 1,
                    "result": "DENIED",
                },
            )
            denial_rows.append(
                {
                    "probe_id": probe_id,
                    "evidence": self.ref(denial),
                    "observed_errno": 1,
                    "state": "DENIED",
                }
            )
        runtime_report = publish_json(
            "runtime/runtime-closure-validation.json",
            {
                "schema": "experiments7-cp2-runtime-closure-validation/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "cp1_sha256": cp1_sha256,
                "cp1_transcript_sha256": cp1_transcript_sha256,
                "cp1_basis": basis_ref,
                "dependency_count": 1,
                "config_count": len(profile_ids),
                "denial_probe_count": len(DENIAL_PROBES),
                "dependencies": dependency_rows,
                "configs": config_rows,
                "denial_probes": denial_rows,
                "subject": {
                    "effective_uid": 1000,
                    "no_new_privs": "ENFORCED",
                    "seccomp_mode": 2,
                    "capability_inheritable": "0000000000000000",
                    "capability_permitted": "0000000000000000",
                    "capability_effective": "0000000000000000",
                    "capability_ambient": "0000000000000000",
                    "cooperating_subjects": "DENIED",
                },
            },
        )
        replay_records = []
        for profile_id in profile_ids:
            config = configs[profile_id]
            result = publish_json(
                f"runtime/results/{profile_id}.json",
                {
                    "schema": "experiments7-cp2-profile-result/v6",
                    "profile_id": profile_id,
                    "config_sha256": config.evidence.sha256,
                    "snapshot_report": self.ref(snapshot_report),
                    "runtime_report": self.ref(runtime_report),
                    "dispatch_stages": list(DISPATCH_STAGES),
                    "golden_result": "EXACT",
                },
            )
            replay_records.append(
                {
                    "profile_id": profile_id,
                    "config": self.ref(config),
                    "result": self.ref(result),
                    "state": "PASS",
                }
            )
        replay_report = publish_json(
            "runtime/hermetic-profile-replay-results.json",
            {
                "schema": "experiments7-cp2-hermetic-profile-replay-results/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "cp1_sha256": cp1_sha256,
                "cp1_transcript_sha256": cp1_transcript_sha256,
                "cp1_basis": basis_ref,
                "profile_count": len(profile_ids),
                "records_sha256": sha256_bytes(canonical_json(replay_records)),
                "records": replay_records,
            },
        )
        snapshot_owned_paths = {
            "snapshots/snapshot-producer-results.json",
            *(f"snapshots/lineage/{lineage_id}.bin" for lineage_id in lineage_ids),
        }
        runtime_owned_paths = {
            "runtime/runtime-closure-validation.json",
            "runtime/dependencies/runtime.bin",
            *(f"runtime/configs/{profile_id}.json" for profile_id in profile_ids),
            *(f"runtime/denials/{probe_id}.json" for probe_id in DENIAL_PROBES),
        }
        replay_owned_paths = {
            "runtime/hermetic-profile-replay-results.json",
            *(f"runtime/results/{profile_id}.json" for profile_id in profile_ids),
        }

        def refs(paths: set[str]) -> list[dict[str, str]]:
            return [self.ref(files[path]) for path in sorted(paths)]

        seals = (
            (
                "snapshots/snapshot-producer-seal.json",
                "experiments7-cp2-snapshot-producer-seal/v6",
                snapshot_report,
                refs(snapshot_owned_paths),
                {
                    "verdict": "PASS",
                    "expected_lineage_count": 129,
                    "passed_lineage_count": 129,
                    "lineage_ids_sha256": sha256_bytes(canonical_json(lineage_ids)),
                },
            ),
            (
                "runtime/runtime-sandbox-seal.json",
                "experiments7-cp2-runtime-sandbox-seal/v6",
                runtime_report,
                refs(runtime_owned_paths),
                {
                    "verdict": "PASS",
                    "expected_profile_count": 30,
                    "config_closed_count": 30,
                    "dependency_count": 1,
                    "denial_probe_count": len(DENIAL_PROBES),
                },
            ),
            (
                "runtime/hermetic-profile-replay-seal.json",
                "experiments7-cp2-hermetic-profile-replay-seal/v6",
                replay_report,
                refs(replay_owned_paths),
                {
                    "verdict": "PASS",
                    "expected_profile_count": 30,
                    "passed_profile_count": 30,
                    "profile_ids_sha256": sha256_bytes(canonical_json(profile_ids)),
                    "dispatch_path": "production_external_contract",
                    "golden_mode": "exact_output_hashes",
                },
            ),
        )
        for path, schema, report, owned, summary in seals:
            publish_json(
                path,
                {
                    "schema": schema,
                    "sealed_run_id": RUN_ID,
                    "run_root": str(self.run_root),
                    "cp1_sha256": cp1_sha256,
                    "cp1_transcript_sha256": cp1_transcript_sha256,
                    "cp1_basis": basis_ref,
                    "report": self.ref(report),
                    "owned_artifacts": owned,
                    "owned_artifacts_sha256": sha256_bytes(canonical_json(owned)),
                    "summary": summary,
                },
            )
        root_fd = os.open(
            self.run_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            semantics = validate_cp2_artifact_structure(
                root_fd,
                RUN_ID,
                str(self.run_root),
                stage,
                cp1_sha256,
                cp1_transcript_sha256,
            )
        except StrictRunError:
            semantics = {
                "schema": "experiments7-cp2-semantics/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "checkpoint": 2,
                "lineage_count": 129,
                "profile_count": 30,
                "seals": [],
            }
        finally:
            os.close(root_fd)
        payload = self.checkpoint_payload(
            2,
            stage,
            previous_sha256=cp1_sha256,
            previous_transcript_sha256=cp1_transcript_sha256,
            semantic_bindings=semantics,
            checkpoint_mutation=checkpoint_mutation,
            semantic_mutation=semantic_mutation,
            stage_evidence_mutation=stage_evidence_mutation,
            legacy_checkpoint=legacy_checkpoint,
        )
        checkpoint = self.publisher(
            self.controller, allow_reserved_paths=True
        ).publish_json("checkpoints/cp2.json", payload)[-1]
        return stage, checkpoint, snapshot_report

    def validate(
        self,
        stage: list[Publication] | None = None,
        *,
        previous_sha256: str | None = None,
        previous_transcript_sha256: str | None = None,
        registry_stage: list[Publication] | None = None,
        omit_registry_path: str | None = None,
        remove_registry_document: bool = False,
        noncanonical_registry: bool = False,
    ) -> dict[str, object]:
        root_fd = os.open(
            self.run_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            selected_stage = self.stage if stage is None else stage
            registered_stage = self.stage if registry_stage is None else registry_stage
            self.external_counter += 1
            external_path = self.base / f"cp3-transcripts-{self.external_counter}.json"
            transcript_rows = {
                "cp1": self.cp1_publication.transcript,
                "cp2": self.cp2_publication.transcript,
                "source_pre": self.source_pre_publication.transcript,
                **{
                    f"cp1_stage_{index:04d}": publication.transcript
                    for index, publication in enumerate(
                        self.cp1_stage_publications
                    )
                },
                **{
                    f"cp2_stage_{index:04d}": publication.transcript
                    for index, publication in enumerate(
                        self.cp2_stage_publications
                    )
                },
                **{
                    f"stage_{index:04d}": publication.transcript
                    for index, publication in enumerate(
                        registered_stage
                    )
                },
            }
            if omit_registry_path is not None:
                publications_by_key = {
                    "cp1": self.cp1_publication,
                    "cp2": self.cp2_publication,
                    "source_pre": self.source_pre_publication,
                    **{
                        f"cp1_stage_{index:04d}": publication
                        for index, publication in enumerate(
                            self.cp1_stage_publications
                        )
                    },
                    **{
                        f"cp2_stage_{index:04d}": publication
                        for index, publication in enumerate(
                            self.cp2_stage_publications
                        )
                    },
                    **{
                        f"stage_{index:04d}": publication
                        for index, publication in enumerate(registered_stage)
                    },
                }
                transcript_rows = {
                    key: transcript
                    for key, transcript in transcript_rows.items()
                    if publications_by_key[key].evidence.relative_path
                    != omit_registry_path
                }
            external_payload = canonical_json(transcript_rows)
            external_path.write_bytes(
                external_payload.rstrip(b"\n")
                if noncanonical_registry
                else external_payload
            )
            with ExternalTranscriptRegistry(
                [
                    ExternalTranscriptRecord(str(external_path), (key,))
                    for key in transcript_rows
                ],
                strict_parent=str(self.parent),
                sealed_run_id=RUN_ID,
                run_root=str(self.run_root),
            ) as registry:
                if remove_registry_document:
                    external_path.unlink()
                return validate_cp3_semantics(
                    root_fd,
                    RUN_ID,
                    str(self.run_root),
                    selected_stage,
                    self.cp1_publication,
                    self.cp2_publication,
                    self.source_pre_publication,
                    registry,
                    previous_sha256 or str(self.cp2_publication.evidence.sha256),
                    previous_transcript_sha256
                    or self.cp2_publication.evidence.publication_transcript_sha256,
                )
        finally:
            os.close(root_fd)


class StrictRunV6CP3SemanticTests(unittest.TestCase):
    def make_fixture(self, **kwargs: object) -> CP3Fixture:
        temporary = tempfile.TemporaryDirectory(prefix="exp7-cp3-semantic-v6-")
        self.addCleanup(temporary.cleanup)
        fixture = CP3Fixture(Path(temporary.name), **kwargs)
        return fixture

    def test_valid_cp3_binds_current_cp1_cp2_buckets_and_reconciliation(self) -> None:
        fixture = self.make_fixture()
        result = fixture.validate()
        self.assertEqual(result["schema"], "experiments7-cp3-semantic-binding/v6")
        self.assertEqual(result["checkpoint"], 3)
        self.assertEqual(
            result["predecessor"]["checkpoint_sha256"],
            fixture.cp2_publication.evidence.sha256,
        )
        semantics = result["predecessor"]["semantics"]
        self.assertEqual(
            set(semantics),
            {
                "cp1",
                "cp1_semantics_sha256",
                "cp2",
                "cp2_semantics_sha256",
            },
        )
        self.assertEqual(semantics["cp1"]["lineage_count"], 129)
        self.assertEqual(semantics["cp1"]["profile_count"], 30)
        self.assertEqual(semantics["cp1"]["inventory_record_count"], 1908)
        self.assertEqual(semantics["cp2"]["lineage_count"], 129)
        self.assertEqual(semantics["cp2"]["profile_count"], 30)
        self.assertEqual(len(semantics["cp2"]["seals"]), 3)
        self.assertEqual(result["cp1_inventory"]["result_count"], RESULT_COUNT)
        self.assertEqual(result["bucket_publication_count"], SECTION_COUNT)
        self.assertEqual(
            result["reconciliation"]["precopy_admitted_count"], RESULT_COUNT
        )
        base = dict(result)
        semantic_sha256 = base.pop("semantic_sha256")
        self.assertEqual(semantic_sha256, sha256_bytes(canonical_json(base)))
        self.assertEqual(result["stage_publication_count"], len(fixture.stage))
        self.assertEqual(result["provenance"]["reconciliation"]["node_count"], 6)
        self.assertEqual(result["source_pre"]["record_count"], 1)

    def test_cp1_and_cp2_legacy_checkpoint_schemas_are_rejected(self) -> None:
        for checkpoint in (1, 2):
            with self.subTest(checkpoint=checkpoint):
                fixture = self.make_fixture(legacy_checkpoint=checkpoint)
                with self.assertRaisesRegex(
                    StrictRunError, "SCHEMA_INVALID.*keys differ"
                ):
                    fixture.validate()

    def test_fully_resealed_cp1_cp2_wrong_schema_run_root_or_index_is_rejected(
        self,
    ) -> None:
        foreign_run_id = "exp7-strict-v6-20260717T150000Z-" + "9" * 32
        mutations = (
            ("schema", "experiments7-strict-checkpoint/v5", "CHECKPOINT_SCHEMA_INVALID"),
            ("sealed_run_id", foreign_run_id, "CHECKPOINT_RUN_BINDING_MISMATCH"),
            ("run_root", "/tmp/foreign-experiments7-run", "CHECKPOINT_RUN_BINDING_MISMATCH"),
            ("checkpoint", None, "CHECKPOINT_SEQUENCE_INVALID"),
        )
        for checkpoint in (1, 2):
            for field, configured_value, code in mutations:
                value = 2 if checkpoint == 1 else 1
                if configured_value is not None:
                    value = configured_value
                with self.subTest(
                    checkpoint=checkpoint,
                    field=field,
                ):
                    fixture = self.make_fixture(
                        checkpoint_mutation=(checkpoint, field, value)
                    )
                    with self.assertRaisesRegex(StrictRunError, code):
                        fixture.validate()

    def test_fully_resealed_missing_empty_stale_or_extra_semantics_are_rejected(
        self,
    ) -> None:
        for checkpoint in (1, 2):
            with self.subTest(checkpoint=checkpoint, mutation="empty"):
                empty = self.make_fixture(
                    checkpoint_mutation=(checkpoint, "semantic_bindings", {})
                )
                with self.assertRaisesRegex(
                    StrictRunError, "CHECKPOINT_SEMANTICS_INVALID"
                ):
                    empty.validate()

            with self.subTest(checkpoint=checkpoint, mutation="stale-digest"):
                stale = self.make_fixture(
                    checkpoint_mutation=(
                        checkpoint,
                        "semantic_bindings_sha256",
                        "f" * 64,
                    )
                )
                with self.assertRaisesRegex(
                    StrictRunError, "CHECKPOINT_SEMANTICS_INVALID"
                ):
                    stale.validate()

            with self.subTest(checkpoint=checkpoint, mutation="extra-key"):
                extra = self.make_fixture(
                    semantic_mutation=(checkpoint, "unexpected", True)
                )
                with self.assertRaisesRegex(
                    StrictRunError, "CP3_PREDECESSOR_SEMANTICS_INVALID"
                ):
                    extra.validate()

    def test_fully_resealed_cp1_individual_counts_and_digests_are_rejected(
        self,
    ) -> None:
        mutations: tuple[tuple[str, object], ...] = (
            ("lineage_count", 128),
            ("profile_count", 29),
            ("inventory_record_count", RESULT_COUNT - 1),
            ("lineage_ids_sha256", "f" * 64),
            ("profile_ids_sha256", "f" * 64),
            ("inventory_sha256", "f" * 64),
            ("inventory_bytes_sha256", "f" * 64),
            ("basis_artifact_evidence_sha256", "f" * 64),
            ("inventory_artifact_evidence_sha256", "f" * 64),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                fixture = self.make_fixture(
                    semantic_mutation=(1, field, value)
                )
                with self.assertRaisesRegex(
                    StrictRunError, "CP3_PREDECESSOR_SEMANTICS_INVALID"
                ):
                    fixture.validate()

    def test_fully_resealed_cp2_counts_ownership_and_each_seal_hash_are_rejected(
        self,
    ) -> None:
        mutations: tuple[tuple[str, object], ...] = (
            ("lineage_count", 128),
            ("profile_count", 29),
            ("owned_artifact_count", 0),
            ("owned_artifacts_sha256", "f" * 64),
            ("seals_sha256", "f" * 64),
            ("seals.0.artifact_sha256", "f" * 64),
            ("seals.0.artifact_evidence_sha256", "f" * 64),
            ("seals.1.artifact_sha256", "f" * 64),
            ("seals.1.artifact_evidence_sha256", "f" * 64),
            ("seals.2.artifact_sha256", "f" * 64),
            ("seals.2.artifact_evidence_sha256", "f" * 64),
            ("seals.2", _DELETE),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                fixture = self.make_fixture(
                    semantic_mutation=(2, field, value)
                )
                with self.assertRaisesRegex(
                    StrictRunError, "CP3_PREDECESSOR_SEMANTICS_INVALID"
                ):
                    fixture.validate()

    def test_cp1_cp2_stage_transcript_must_be_registered_and_descriptor_bound(
        self,
    ) -> None:
        for relative_path in (
            "registry/lineage-basis.json",
            "snapshots/snapshot-producer-results.json",
        ):
            with self.subTest(relative_path=relative_path, mutation="missing"):
                missing = self.make_fixture()
                with self.assertRaisesRegex(StrictRunError, "digest is absent"):
                    missing.validate(omit_registry_path=relative_path)

        for checkpoint, relative_path in (
            (1, "registry/lineage-basis.json"),
            (2, "snapshots/snapshot-producer-results.json"),
        ):
            with self.subTest(checkpoint=checkpoint, mutation="foreign"):
                foreign = self.make_fixture(
                    stage_evidence_mutation=(
                        checkpoint,
                        relative_path,
                        "publication_transcript_sha256",
                        "SOURCE_PRE_TRANSCRIPT",
                    )
                )
                with self.assertRaises(StrictRunError):
                    foreign.validate()

    def test_cp1_cp2_stage_chronology_is_bounded_by_its_predecessor(
        self,
    ) -> None:
        for kwargs in (
            {"cp1_directory_before_source_pre": True},
            {"cp2_directory_before_cp1": True},
        ):
            with self.subTest(kwargs=kwargs):
                fixture = self.make_fixture(**kwargs)
                with self.assertRaisesRegex(
                    StrictRunError, "CP3_PREDECESSOR_STAGE_INVALID"
                ):
                    fixture.validate()

    def test_raw_node_is_exact_source_pre_identity_not_snapshot_or_substitution(self) -> None:
        substituted = self.make_fixture(raw_mutation=("sha256", "f" * 64))
        with self.assertRaisesRegex(
            StrictRunError, "raw provenance node is not content-addressed exactly"
        ):
            substituted.validate()

        snapshot = self.make_fixture(raw_from_snapshot=True)
        with self.assertRaisesRegex(StrictRunError, "raw provenance node schema keys"):
            snapshot.validate()

    def test_source_pre_publication_is_current_and_descriptor_bound(self) -> None:
        fixture = self.make_fixture()
        target = fixture.run_root / "manifests/source-pre.jsonl"
        raw = target.read_bytes()
        replacement = target.with_name(target.name + ".replacement")
        replacement.write_bytes(raw)
        replacement.chmod(0o444)
        target.unlink()
        replacement.rename(target)
        with self.assertRaisesRegex(StrictRunError, "TRANSCRIPT_STALE"):
            fixture.validate()

    def test_every_stage_transcript_requires_exact_current_registry_entry(self) -> None:
        missing = self.make_fixture()
        with self.assertRaisesRegex(StrictRunError, "digest is absent"):
            missing.validate(
                omit_registry_path="admission/precopy/reconciliation.json"
            )

        removed = self.make_fixture()
        with self.assertRaises(StrictRunError):
            removed.validate(remove_registry_document=True)

        replaced = self.make_fixture()
        reconciliation = next(
            publication
            for publication in replaced.stage
            if publication.evidence.relative_path
            == "admission/precopy/reconciliation.json"
        )
        replacement = Publication(
            reconciliation.evidence,
            replaced.cp1_publication.transcript,
        )
        supplied = [
            replacement if publication is reconciliation else publication
            for publication in replaced.stage
        ]
        with self.assertRaisesRegex(StrictRunError, "exact registered transcript"):
            replaced.validate(supplied, registry_stage=replaced.stage)

        noncanonical = self.make_fixture()
        with self.assertRaisesRegex(StrictRunError, "NONCANONICAL_JSON"):
            noncanonical.validate(noncanonical_registry=True)

    def test_unregistered_or_wrong_cp2_transcript_digest_is_rejected(self) -> None:
        fixture = self.make_fixture()
        with self.assertRaisesRegex(
            StrictRunError, "predecessor transcript is not current CP2"
        ):
            fixture.validate(previous_transcript_sha256="f" * 64)

    def test_required_cp3_directory_publications_cannot_be_omitted(self) -> None:
        fixture = self.make_fixture(omit_stage_directories=True)
        with self.assertRaisesRegex(StrictRunError, "stage paths are not exactly"):
            fixture.validate()

    def test_undeclared_provenance_publication_is_rejected(self) -> None:
        baseline = self.make_fixture()
        self.assertIn("semantic_sha256", baseline.validate())
        hostile = self.make_fixture(extra_provenance=True)
        with self.assertRaisesRegex(StrictRunError, "exact CP3 set|stage paths are not exactly"):
            hostile.validate()
        hidden = self.make_fixture(
            extra_provenance=True, hide_extra_provenance_publication=True
        )
        with self.assertRaisesRegex(StrictRunError, "exact CP3 set"):
            hidden.validate()

    def test_same_byte_new_inode_cp1_or_cp2_is_rejected(self) -> None:
        for relative in ("checkpoints/cp1.json", "checkpoints/cp2.json"):
            with self.subTest(relative=relative):
                fixture = self.make_fixture()
                target = fixture.run_root / relative
                raw = target.read_bytes()
                replacement = target.with_name(target.name + ".replacement")
                replacement.write_bytes(raw)
                replacement.chmod(0o444)
                target.unlink()
                replacement.rename(target)
                with self.assertRaises(StrictRunError):
                    fixture.validate()

    def test_bucket_filename_must_equal_internal_paper_section_id(self) -> None:
        fixture = self.make_fixture(bucket_filename_mismatch=True)
        with self.assertRaisesRegex(StrictRunError, "exact paper_section_id"):
            fixture.validate()

    def test_zero_node_graph_cannot_close_opaque_contributor_ids(self) -> None:
        fixture = self.make_fixture(empty_graph_with_opaque_ids=True)
        with self.assertRaisesRegex(StrictRunError, "typed contributor is absent"):
            fixture.validate()

    def test_positive_result_schema_and_reason_codes_are_exact(self) -> None:
        wrong_reason = self.make_fixture(invalid_reason_codes=True)
        with self.assertRaisesRegex(StrictRunError, "reason_codes"):
            wrong_reason.validate()
        extra_key = self.make_fixture(extra_result_key=True)
        with self.assertRaisesRegex(StrictRunError, "admission keys differ"):
            extra_key.validate()

    def test_numeric_recomputation_and_paper_rounding_are_exact(self) -> None:
        nonterminating = self.make_fixture(rational_rounding=True)
        self.assertIn("semantic_sha256", nonterminating.validate())

        wrong_math = self.make_fixture(bad_recomputation=True)
        with self.assertRaisesRegex(StrictRunError, "exact paper rounding"):
            wrong_math.validate()
        wrong_display = self.make_fixture(bad_rounding=True)
        with self.assertRaisesRegex(StrictRunError, "exact paper rounding"):
            wrong_display.validate()

    def test_multi_megabyte_bucket_publications_validate(self) -> None:
        fixture = self.make_fixture(section_count=2)
        result = fixture.validate()
        self.assertEqual(result["bucket_publication_count"], 2)
        self.assertTrue(
            any(
                int(binding["bytes"]) > 1024 * 1024
                for binding in result["bucket_publications"]
            )
        )

    def test_fully_resealed_inventory_section_swap_is_rejected(self) -> None:
        fixture = self.make_fixture(swap_sections=True)
        with self.assertRaisesRegex(StrictRunError, "inventory result section mismatch"):
            fixture.validate()

    def test_missing_record_and_non_1908_inventory_are_rejected(self) -> None:
        missing = self.make_fixture(omit_last=True)
        with self.assertRaisesRegex(StrictRunError, "exact inventory"):
            missing.validate()

        short = self.make_fixture(result_count=RESULT_COUNT - 1)
        with self.assertRaisesRegex(StrictRunError, "exactly 1908"):
            short.validate()

    def test_fully_resealed_duplicate_and_extra_records_are_rejected(self) -> None:
        duplicate = self.make_fixture(duplicate_record=True)
        with self.assertRaisesRegex(StrictRunError, "duplicate result in bucket"):
            duplicate.validate()

        extra = self.make_fixture(extra_record=True)
        with self.assertRaisesRegex(StrictRunError, "inventory result section mismatch"):
            extra.validate()

    def test_stale_cp1_inventory_evidence_and_wrong_cp2_hash_are_rejected(self) -> None:
        stale = self.make_fixture(stale_inventory_evidence=True)
        with self.assertRaisesRegex(
            StrictRunError, "CP3_PREDECESSOR_STAGE_INVALID"
        ):
            stale.validate()

        fixture = self.make_fixture()
        with self.assertRaisesRegex(
            StrictRunError, "predecessor hash is not current CP2"
        ):
            fixture.validate(previous_sha256="0" * 64)

    def test_boolean_reconciliation_count_is_rejected_as_non_integer(self) -> None:
        fixture = self.make_fixture(bool_reconciliation_count=True)
        with self.assertRaisesRegex(StrictRunError, "exact nonnegative integer"):
            fixture.validate()

    def test_bucket_schema_wrong_writer_and_wrong_path_are_rejected(self) -> None:
        schema = self.make_fixture(
            bucket_schema="experiments7-admission-bucket-subseal/v5"
        )
        with self.assertRaisesRegex(StrictRunError, "bucket schema mismatch"):
            schema.validate()

        fixture = self.make_fixture()
        reconciliation = next(
            publication
            for publication in fixture.stage
            if publication.evidence.relative_path
            == "admission/precopy/reconciliation.json"
        )
        wrong_writer = Publication(
            replace(reconciliation.evidence, writer=fixture.inventory_writer),
            reconciliation.transcript,
        )
        writer_stage = [
            wrong_writer if publication is reconciliation else publication
            for publication in fixture.stage
        ]
        with self.assertRaisesRegex(StrictRunError, "CP3_WRITER_INVALID"):
            fixture.validate(writer_stage)

        wrong_path = Publication(
            replace(
                reconciliation.evidence,
                relative_path="admission/final/reconciliation.json",
            ),
            reconciliation.transcript,
        )
        path_stage = [
            wrong_path if publication is reconciliation else publication
            for publication in fixture.stage
        ]
        with self.assertRaisesRegex(StrictRunError, "CP3_STAGE_PATH_INVALID"):
            fixture.validate(path_stage)

    def test_current_publication_rewrite_is_rejected_as_stale(self) -> None:
        fixture = self.make_fixture()
        target = fixture.run_root / "admission/precopy/reconciliation.json"
        target.chmod(0o644)
        target.write_bytes(canonical_json({"rewritten": True}))
        with self.assertRaises(StrictRunError):
            fixture.validate()


if __name__ == "__main__":
    unittest.main()
