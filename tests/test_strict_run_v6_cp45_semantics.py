from __future__ import annotations

import copy
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
    digest_json,
    digest_records,
    result_inventory_digest,
    make_bucket_subseal,
    make_raw_provenance_node,
    plan_source_pre_copies,
    reconcile_cp3_buckets,
    reconcile_final_admission,
)
from strict_run import (  # noqa: E402
    CP3SemanticPublications,
    CP4SemanticPublications,
    ExternalTranscriptRecord,
    ExternalTranscriptRegistry,
    StagePublisher,
    StrictRunError,
    WriterIdentity,
    canonical_json,
    default_writer_policy,
    sha256_bytes,
    validate_cp4_semantics,
    validate_cp5_semantics,
)
from strict_run.publication import Publication  # noqa: E402


RUN_ID = "exp7-strict-v6-20260717T160000Z-" + "4" * 32


def _positive_record(raw_id: str) -> dict[str, object]:
    closure = {field: [f"{field}:0"] for field in CLOSURE_FIELDS}
    contributors = {raw_id}
    for values in closure.values():
        contributors.update(values)
    return {
        "schema": "experiments7-admission-result/v6",
        "result_id": "result:synthetic:0000",
        "paper_section_id": "main",
        "phase": "precopy",
        "state": "ADMITTED_FOR_COPY",
        "result_kind": "inference",
        "copy_required": True,
        "raw_artifact_ids": [raw_id],
        "candidate_raw_artifact_ids": [],
        "required_parent_result_ids": [],
        "closed_parent_result_ids": [],
        "required_contributor_node_ids": sorted(contributors),
        "sealed_contributor_node_ids": sorted(contributors),
        "positive_evidence_origin": "run_local_strict",
        "source_pre_bound": True,
        "declared_drift": False,
        "ambiguous": False,
        "support_only": False,
        "audit_positive_evidence": False,
        "reason_codes": [],
        "copied_artifact_ids": [],
        "copied_to_edge_ids": [],
        **closure,
    }


class CP45Fixture:
    def __init__(
        self,
        base: Path,
        *,
        extra_raw: bool = False,
        extra_final: bool = False,
        source_post_mismatch: bool = False,
        final_reconciliation_mismatch: bool = False,
        cp4_checkpoint_omits_copy: bool = False,
        destination_identity_mismatch: bool = False,
        published_copy_plan: bool = False,
        extra_bucket: bool = False,
        extra_cp3_stage: bool = False,
        cp3_semantic_mode: str = "valid",
        cp4_semantic_mode: str = "valid",
    ) -> None:
        self.base = base
        self.strict_parent = base / "strict-runs"
        self.strict_parent.mkdir()
        self.run_root = self.strict_parent / RUN_ID
        self.run_root.mkdir(mode=0o755)
        self.policy = default_writer_policy()
        self.g0 = WriterIdentity("g0", "g0:cp45-fixture", "g0-writer")
        self.inventory_writer = WriterIdentity(
            "inventory", "inventory:cp45-fixture", "inventory-writer"
        )
        self.provenance_writer = WriterIdentity(
            "provenance", "provenance:cp45-fixture", "provenance-writer"
        )
        self.admission_writer = WriterIdentity(
            "admission", "admission:cp45-fixture", "admission-writer"
        )
        self.copy_writer = WriterIdentity("copy", "copy:cp45-fixture", "copy-writer")
        self.controller = WriterIdentity(
            "checkpoint_controller",
            "checkpoint:cp45-fixture",
            "checkpoint_controller-writer",
        )

        self.payload = b"synthetic raw inference\n"
        source_identity = {
            "file_type": "regular",
            "st_dev": 10,
            "st_ino": 20,
            "mode": 0o100444,
            "size": len(self.payload),
            "mtime_ns": 123456,
        }
        self.source_record = {
            "schema": "experiments7-source-pre/v6",
            "record_id": "source:synthetic-0",
            "root_id": "experiments4",
            "relative_path": "synthetic/output.jsonl",
            "type": "regular",
            "sha256": hashlib.sha256(self.payload).hexdigest(),
            "size": len(self.payload),
            "descriptor_identity": source_identity,
        }
        self.source_pre_raw = canonical_json(self.source_record)
        self.paper_pre_value = {
            "schema": "experiments7-paper-pre/v6",
            "paper_sha256": "9" * 64,
            "bytes": 123,
        }
        self.paper_pre_raw = canonical_json(self.paper_pre_value)
        source_created = self.publisher(self.g0).publish_bytes(
            "manifests/source-pre.jsonl", self.source_pre_raw
        )
        self.source_pre = source_created[-1]
        paper_created = self.publisher(self.g0).publish_bytes(
            "manifests/paper-pre.json", self.paper_pre_raw
        )
        self.paper_pre = paper_created[-1]

        self.raw_node = make_raw_provenance_node(self.source_record)
        self.inventory = [
            {
                "result_id": "result:synthetic:0000",
                "paper_section_id": "main",
                "numeric_value": 0,
                "display_value": "0.0",
            }
        ]
        self.precopy = _positive_record(str(self.raw_node["node_id"]))
        self.bucket = make_bucket_subseal(
            result_inventory_digest(self.inventory), "main", [self.precopy]
        )
        self.cp3_summary = reconcile_cp3_buckets(
            self.inventory, [self.bucket], expected_result_count=1
        )
        inventory_created = self.publisher(self.inventory_writer).publish_bytes(
            "inventory/results.jsonl", canonical_json(self.inventory[0])
        )
        self.inventory_publication = inventory_created[-1]
        self.cp1_semantic = {
            "schema": "experiments7-synthetic-cp1-semantic/v6",
            "checkpoint": 1,
        }
        self.cp1_checkpoint = self.publish_checkpoint(
            1,
            [self.inventory_publication],
            "a" * 64,
            "b" * 64,
            semantic_bindings=self.cp1_semantic,
        )
        self.cp2_semantic = {
            "schema": "experiments7-synthetic-cp2-semantic/v6",
            "checkpoint": 2,
        }
        self.cp2_checkpoint = self.publish_checkpoint(
            2,
            [self.paper_pre],
            self.cp1_checkpoint.evidence.sha256 or "",
            self.cp1_checkpoint.evidence.publication_transcript_sha256,
            semantic_bindings=self.cp2_semantic,
        )
        cp3_stage: list[Publication] = []
        created = self.publisher(self.provenance_writer).publish_bytes(
            "provenance/nodes.jsonl", canonical_json(self.raw_node)
        )
        cp3_stage.extend(created)
        self.raw_nodes_publication = created[-1]
        created = self.publisher(self.provenance_writer).publish_bytes(
            "provenance/edges.jsonl", b""
        )
        cp3_stage.extend(created)
        self.provenance_edges_publication = created[-1]
        created = self.publisher(self.provenance_writer).publish_bytes(
            "provenance/recomputations.jsonl", b""
        )
        cp3_stage.extend(created)
        self.provenance_recomputations_publication = created[-1]
        created = self.publisher(self.provenance_writer).publish_bytes(
            "provenance/rounding-proofs.jsonl", b""
        )
        cp3_stage.extend(created)
        self.provenance_rounding_publication = created[-1]
        created = self.publisher(self.admission_writer).publish_bytes(
            "admission/precopy/buckets/main.json", canonical_json(self.bucket)
        )
        cp3_stage.extend(created)
        self.bucket_publication = created[-1]
        created = self.publisher(self.admission_writer).publish_bytes(
            "admission/precopy/reconciliation.json", canonical_json(self.cp3_summary)
        )
        cp3_stage.extend(created)
        self.cp3_reconciliation_publication = created[-1]
        if extra_cp3_stage:
            cp3_stage.extend(
                self.publisher(self.admission_writer).publish_json(
                    "admission/precopy/buckets/extra.json", {"extra": True}
                )
            )
        self.cp3_stage = tuple(cp3_stage)
        cp3_semantic = self.make_cp3_semantic(cp3_stage)
        if cp3_semantic_mode == "missing":
            cp3_semantic = None
        elif cp3_semantic_mode == "wrong":
            assert cp3_semantic is not None
            cp3_semantic["schema"] = "experiments7-cp3-semantic-binding/v5"
            self.reseal_semantic(cp3_semantic)
        elif cp3_semantic_mode == "resealed":
            assert cp3_semantic is not None
            cp3_semantic["reconciliation"]["copy_plan_input_sha256"] = "f" * 64
            self.reseal_semantic(cp3_semantic)
        elif cp3_semantic_mode == "resealed_predecessor":
            assert cp3_semantic is not None
            cp3_semantic["predecessor"]["checkpoint_sha256"] = "f" * 64
            self.reseal_semantic(cp3_semantic)
        elif cp3_semantic_mode == "resealed_provenance":
            assert cp3_semantic is not None
            cp3_semantic["provenance"]["reconciliation"]["nodes_sha256"] = "f" * 64
            self.reseal_semantic(cp3_semantic)
        elif cp3_semantic_mode == "duplicate_provenance":
            assert cp3_semantic is not None
            cp3_semantic["provenance"]["publications"].append(
                copy.deepcopy(cp3_semantic["provenance"]["publications"][0])
            )
            self.reseal_semantic(cp3_semantic)
        elif cp3_semantic_mode == "nonmapping_provenance":
            assert cp3_semantic is not None
            cp3_semantic["provenance"]["publications"].append("forged")
            self.reseal_semantic(cp3_semantic)
        elif cp3_semantic_mode == "extra_provenance_key":
            assert cp3_semantic is not None
            cp3_semantic["provenance"]["extra"] = True
            self.reseal_semantic(cp3_semantic)
        elif cp3_semantic_mode == "duplicate_bucket":
            assert cp3_semantic is not None
            cp3_semantic["bucket_publications"].append(
                copy.deepcopy(cp3_semantic["bucket_publications"][0])
            )
            cp3_semantic["bucket_publication_count"] = 2
            self.reseal_semantic(cp3_semantic)
        elif cp3_semantic_mode == "bool_count":
            assert cp3_semantic is not None
            cp3_semantic["source_pre"]["record_count"] = True
            self.reseal_semantic(cp3_semantic)
        elif cp3_semantic_mode == "reordered_stage":
            assert cp3_semantic is not None
            cp3_semantic["stage_publications"].reverse()
            self.reseal_semantic(cp3_semantic)
        elif cp3_semantic_mode == "reordered_provenance":
            assert cp3_semantic is not None
            cp3_semantic["provenance"]["publications"].reverse()
            self.reseal_semantic(cp3_semantic)
        elif cp3_semantic_mode != "valid":
            raise ValueError(f"unknown CP3 semantic mode: {cp3_semantic_mode}")
        self.cp3_checkpoint = self.publish_checkpoint(
            3,
            cp3_stage,
            self.cp2_checkpoint.evidence.sha256 or "",
            self.cp2_checkpoint.evidence.publication_transcript_sha256,
            semantic_bindings=cp3_semantic,
        )
        if extra_bucket:
            self.publisher(self.admission_writer).publish_json(
                "admission/precopy/buckets/extra.json", {"extra": True}
            )
        self.cp3_inputs = CP3SemanticPublications(
            checkpoint=self.cp3_checkpoint,
            cp1_checkpoint=self.cp1_checkpoint,
            cp2_checkpoint=self.cp2_checkpoint,
            source_pre=self.source_pre,
            paper_pre=self.paper_pre,
            inventory=self.inventory_publication,
            raw_nodes=self.raw_nodes_publication,
            provenance_edges=self.provenance_edges_publication,
            provenance_recomputations=self.provenance_recomputations_publication,
            provenance_rounding_proofs=self.provenance_rounding_publication,
            reconciliation=self.cp3_reconciliation_publication,
            buckets=(self.bucket_publication,),
        )

        self.plan = plan_source_pre_copies(
            self.cp3_summary,
            [self.raw_node],
            [self.source_record],
            expected_result_count=1,
        )
        entry = self.plan["entries"][0]
        self.destination = str(entry["destination_relative_path"])
        cp4_stage: list[Publication] = []
        created = self.publisher(self.copy_writer).publish_bytes(
            self.destination, self.payload
        )
        cp4_stage.extend(created)
        self.destination_publication = created[-1]
        destination_stat = os.stat(self.run_root / self.destination, follow_symlinks=False)
        destination_identity = {
            "file_type": "regular",
            "st_dev": destination_stat.st_dev,
            "st_ino": destination_stat.st_ino,
            "mode": destination_stat.st_mode,
            "size": destination_stat.st_size,
            "mtime_ns": destination_stat.st_mtime_ns,
        }
        if destination_identity_mismatch:
            destination_identity["mtime_ns"] += 1
        self.ledger = [
            {
                "schema": "experiments7-copy-ledger-row/v6",
                "copy_plan_entry_sha256": digest_json(entry),
                **{
                    key: entry[key]
                    for key in (
                        "raw_artifact_id",
                        "source_manifest_record_id",
                        "source_root_id",
                        "source_relative_path",
                        "source_pre_record_sha256",
                        "source_sha256",
                        "size",
                        "destination_relative_path",
                        "copied_artifact_id",
                        "copied_to_edge_id",
                    )
                },
                "source_before": copy.deepcopy(source_identity),
                "source_after": copy.deepcopy(source_identity),
                "source_sha256_before": self.source_record["sha256"],
                "source_sha256_after": self.source_record["sha256"],
                "destination_identity": destination_identity,
                "publication_method": "byte_stream_no_replace",
                "hardlink_used": False,
                "reflink_used": False,
                "clone_used": False,
                "cache_used": False,
                "destination_sha256": self.source_record["sha256"],
                "destination_size": len(self.payload),
            }
        ]
        created = self.publisher(self.copy_writer).publish_bytes(
            "copies.jsonl", canonical_json(self.ledger[0])
        )
        cp4_stage.extend(created)
        self.copies_publication = created[-1]
        self.cp4_stage = tuple(cp4_stage)
        if published_copy_plan:
            self.publisher(self.g0).publish_json(
                "manifests/copy-plan.json", self.plan
            )
        if extra_raw:
            self.publisher(self.copy_writer).publish_bytes("raw/extra.bin", b"extra\n")
        checkpoint_stage = list(self.cp4_stage)
        if cp4_checkpoint_omits_copy:
            checkpoint_stage = [
                publication
                for publication in checkpoint_stage
                if publication.evidence.relative_path != self.destination
            ]
        cp4_validation_is_expected_to_pass = not (
            extra_raw
            or destination_identity_mismatch
            or published_copy_plan
            or extra_bucket
            or extra_cp3_stage
            or cp3_semantic_mode != "valid"
        )
        if cp4_validation_is_expected_to_pass:
            with self.registry_for(
                [
                    self.cp3_checkpoint,
                    self.cp1_checkpoint,
                    self.cp2_checkpoint,
                    self.source_pre,
                    self.paper_pre,
                    self.inventory_publication,
                    *self.cp3_stage,
                    *self.cp4_stage,
                ],
                "pre-cp4-checkpoint",
            ) as registry:
                root_fd = os.open(
                    self.run_root,
                    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                )
                try:
                    cp4_semantic = validate_cp4_semantics(
                        root_fd,
                        RUN_ID,
                        str(self.run_root),
                        self.cp4_stage,
                        self.cp3_inputs,
                        registry,
                        self.cp3_checkpoint.evidence.sha256 or "",
                        self.cp3_checkpoint.evidence.publication_transcript_sha256,
                        expected_result_count=1,
                    )
                finally:
                    os.close(root_fd)
        else:
            cp4_base = {
                "schema": "experiments7-cp4-semantic-binding/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "checkpoint": 4,
                "fixture_placeholder": True,
            }
            cp4_semantic = {
                **cp4_base,
                "semantic_sha256": sha256_bytes(canonical_json(cp4_base)),
            }
        if cp4_semantic_mode == "missing":
            cp4_semantic = None
        elif cp4_semantic_mode == "wrong":
            assert cp4_semantic is not None
            cp4_semantic["schema"] = "experiments7-cp4-semantic-binding/v5"
            self.reseal_semantic(cp4_semantic)
        elif cp4_semantic_mode == "resealed":
            assert cp4_semantic is not None
            cp4_semantic["planned_destinations"] = ["raw/verified/forged.json"]
            self.reseal_semantic(cp4_semantic)
        elif cp4_semantic_mode != "valid":
            raise ValueError(f"unknown CP4 semantic mode: {cp4_semantic_mode}")
        self.cp4_checkpoint = self.publish_checkpoint(
            4,
            checkpoint_stage,
            self.cp3_checkpoint.evidence.sha256 or "",
            self.cp3_checkpoint.evidence.publication_transcript_sha256,
            semantic_bindings=cp4_semantic,
        )
        self.cp4_inputs = CP4SemanticPublications(
            checkpoint=self.cp4_checkpoint,
            stage=self.cp4_stage,
        )

        final = copy.deepcopy(self.precopy)
        final["phase"] = "final"
        final["state"] = "VERIFIED"
        final["precopy_record_sha256"] = digest_json(self.precopy)
        final["copied_artifact_ids"] = [entry["copied_artifact_id"]]
        final["copied_to_edge_ids"] = [entry["copied_to_edge_id"]]
        self.final_records = [final]
        final_summary = reconcile_final_admission(
            self.inventory,
            [self.bucket],
            self.cp3_summary,
            self.final_records,
            self.plan,
            self.ledger,
            {entry["copied_artifact_id"]: self.payload},
            [self.source_record],
            expected_result_count=1,
        )
        cp5_stage: list[Publication] = []
        created = self.publisher(self.admission_writer).publish_bytes(
            "admission/final/results.jsonl", canonical_json(final)
        )
        cp5_stage.extend(created)
        reconciliation_value: object = final_summary
        if final_reconciliation_mismatch:
            reconciliation_value = {"schema": "hostile-final-reconciliation"}
        created = self.publisher(self.admission_writer).publish_bytes(
            "admission/final/reconciliation.json", canonical_json(reconciliation_value)
        )
        cp5_stage.extend(created)
        if extra_final:
            self.publisher(self.admission_writer).publish_bytes(
                "admission/final/extra.json", canonical_json({"extra": True})
            )
        post_source_raw = self.source_pre_raw
        if source_post_mismatch:
            hostile_source = copy.deepcopy(self.source_record)
            hostile_source["sha256"] = "f" * 64
            post_source_raw = canonical_json(hostile_source)
        created = self.publisher(self.g0).publish_bytes(
            "manifests/source-post.jsonl", post_source_raw
        )
        cp5_stage.extend(created)
        self.source_post = created[-1]
        created = self.publisher(self.g0).publish_bytes(
            "manifests/paper-post.json", self.paper_pre_raw
        )
        cp5_stage.extend(created)
        self.paper_post = created[-1]
        self.cp5_stage = tuple(cp5_stage)

    @staticmethod
    def reseal_semantic(semantic: dict[str, object]) -> None:
        base = {key: value for key, value in semantic.items() if key != "semantic_sha256"}
        semantic["semantic_sha256"] = sha256_bytes(canonical_json(base))

    @staticmethod
    def publication_binding(publication: Publication) -> dict[str, object]:
        evidence = publication.evidence
        first = publication.transcript["events"][0]
        assert isinstance(first, dict)
        return {
            "relative_path": evidence.relative_path,
            "artifact_type": evidence.artifact_type,
            "artifact_sha256": evidence.sha256,
            "bytes": evidence.size,
            "identity": evidence.identity,
            "writer": evidence.writer.to_dict(),
            "writer_rule_id": evidence.writer_rule_id,
            "publication_transcript_sha256": evidence.publication_transcript_sha256,
            "physical_evidence_sha256": sha256_bytes(
                canonical_json(evidence.to_dict())
            ),
            "publication_first_completed_monotonic_ns": first[
                "completed_monotonic_ns"
            ],
            "publication_emitted_monotonic_ns": publication.transcript[
                "emitted_monotonic_ns"
            ],
        }

    @classmethod
    def common_binding(cls, publication: Publication) -> dict[str, object]:
        binding = cls.publication_binding(publication)
        for field in (
            "artifact_type",
            "publication_first_completed_monotonic_ns",
            "publication_emitted_monotonic_ns",
        ):
            binding.pop(field)
        return binding

    def make_cp3_semantic(
        self, cp3_stage: list[Publication]
    ) -> dict[str, object]:
        stage_bindings = [
            self.publication_binding(publication)
            for publication in sorted(
                cp3_stage, key=lambda row: row.evidence.relative_path
            )
        ]
        source_pre_binding = self.publication_binding(self.source_pre)
        source_pre_binding.pop("artifact_type")
        source_pre = {
            **source_pre_binding,
            "record_count": 1,
            "records_sha256": digest_records([self.source_record]),
        }
        cp1_inventory = {
            **self.common_binding(self.inventory_publication),
            "checkpoint_sha256": self.cp1_checkpoint.evidence.sha256,
            "checkpoint_transcript_sha256": self.cp1_checkpoint.evidence.publication_transcript_sha256,
            "inventory_sha256": self.cp3_summary["inventory_sha256"],
            "result_count": 1,
        }
        graph_publications = (
            (self.raw_nodes_publication, [self.raw_node]),
            (self.provenance_edges_publication, []),
            (self.provenance_recomputations_publication, []),
            (self.provenance_rounding_publication, []),
        )
        graph_bindings = [
            {
                **self.publication_binding(publication),
                "canonical_file_sha256": publication.evidence.sha256,
                "record_count": len(rows),
            }
            for publication, rows in graph_publications
        ]
        bucket_binding = {
            **self.publication_binding(self.bucket_publication),
            "paper_section_id": "main",
            "canonical_bucket_sha256": digest_json(self.bucket),
            "canonical_file_sha256": self.bucket_publication.evidence.sha256,
        }
        reconciliation = {
            **self.publication_binding(self.cp3_reconciliation_publication),
            "canonical_file_sha256": self.cp3_reconciliation_publication.evidence.sha256,
            "recomputed_sha256": sha256_bytes(canonical_json(self.cp3_summary)),
            "total_result_count": self.cp3_summary["total_result_count"],
            "precopy_admitted_count": self.cp3_summary["precopy_admitted_count"],
            "precopy_unresolved_count": self.cp3_summary["precopy_unresolved_count"],
            "copy_plan_input_sha256": self.cp3_summary["copy_plan_input_sha256"],
        }
        base = {
            "schema": "experiments7-cp3-semantic-binding/v6",
            "sealed_run_id": RUN_ID,
            "run_root": str(self.run_root),
            "checkpoint": 3,
            "predecessor": {
                "checkpoint": 2,
                "checkpoint_sha256": self.cp2_checkpoint.evidence.sha256,
                "checkpoint_transcript_sha256": self.cp2_checkpoint.evidence.publication_transcript_sha256,
                "cp1_publication": self.common_binding(self.cp1_checkpoint),
                "cp2_publication": self.common_binding(self.cp2_checkpoint),
                "semantics": {
                    "cp1": self.cp1_semantic,
                    "cp1_semantics_sha256": sha256_bytes(
                        canonical_json(self.cp1_semantic)
                    ),
                    "cp2": self.cp2_semantic,
                    "cp2_semantics_sha256": sha256_bytes(
                        canonical_json(self.cp2_semantic)
                    ),
                },
            },
            "cp1_inventory": cp1_inventory,
            "source_pre": source_pre,
            "stage_publications": stage_bindings,
            "stage_publication_count": len(stage_bindings),
            "provenance": {
                "publications": graph_bindings,
                "reconciliation": {
                    "schema": "experiments7-cp3-provenance-reconciliation/v6",
                    "node_count": 1,
                    "edge_count": 0,
                    "recomputation_count": 0,
                    "rounding_proof_count": 0,
                    "nodes_sha256": digest_records([self.raw_node]),
                    "edges_sha256": digest_records([]),
                    "recomputations_sha256": digest_records([]),
                    "rounding_proofs_sha256": digest_records([]),
                },
            },
            "bucket_publications": [bucket_binding],
            "bucket_publication_count": 1,
            "reconciliation": reconciliation,
        }
        return {**base, "semantic_sha256": sha256_bytes(canonical_json(base))}

    def publisher(
        self, writer: WriterIdentity, *, allow_reserved_paths: bool = False
    ) -> StagePublisher:
        return StagePublisher(
            str(self.strict_parent),
            RUN_ID,
            str(self.run_root),
            writer,
            self.policy,
            allow_reserved_paths=allow_reserved_paths,
        )

    def publish_checkpoint(
        self,
        checkpoint: int,
        stage: list[Publication],
        previous_sha256: str,
        previous_transcript_sha256: str,
        *,
        semantic_bindings: dict[str, object] | None,
    ) -> Publication:
        evidence = [
            publication.evidence.to_dict()
            for publication in sorted(stage, key=lambda row: row.evidence.relative_path)
        ]
        payload = {
            "schema": "experiments7-strict-checkpoint/v6",
            "sealed_run_id": RUN_ID,
            "run_root": str(self.run_root),
            "checkpoint": checkpoint,
            "previous_checkpoint_sha256": previous_sha256,
            "previous_checkpoint_transcript_sha256": previous_transcript_sha256,
            "stage_actual_paths": evidence,
            "stage_actual_paths_sha256": sha256_bytes(canonical_json(evidence)),
            "bindings": {},
        }
        if semantic_bindings is not None:
            payload["semantic_bindings"] = semantic_bindings
            payload["semantic_bindings_sha256"] = sha256_bytes(
                canonical_json(semantic_bindings)
            )
        created = self.publisher(
            self.controller, allow_reserved_paths=True
        ).publish_json(f"checkpoints/cp{checkpoint}.json", payload)
        return created[-1]

    def all_registered_publications(self) -> list[Publication]:
        rows = [
            self.cp1_checkpoint,
            self.cp2_checkpoint,
            self.cp3_checkpoint,
            self.source_pre,
            self.paper_pre,
            self.inventory_publication,
            *self.cp3_stage,
            *self.cp4_stage,
            self.cp4_checkpoint,
            *self.cp5_stage,
        ]
        by_digest: dict[str, Publication] = {}
        for publication in rows:
            by_digest[publication.evidence.publication_transcript_sha256] = publication
        return list(by_digest.values())

    def registry(
        self, *, omit_path: str | None = None
    ) -> ExternalTranscriptRegistry:
        publications = [
            publication
            for publication in self.all_registered_publications()
            if publication.evidence.relative_path != omit_path
        ]
        return self.registry_for(publications, f"final-{omit_path or 'complete'}")

    def registry_for(
        self, publications: list[Publication], label: str
    ) -> ExternalTranscriptRegistry:
        document = {
            f"transcript_{index:04d}": publication.transcript
            for index, publication in enumerate(publications)
        }
        external = self.base / f"transcripts-{label.replace('/', '_')}.json"
        external.write_bytes(canonical_json(document))
        records = [
            ExternalTranscriptRecord(str(external), (key,))
            for key in document
        ]
        return ExternalTranscriptRegistry(
            records,
            strict_parent=str(self.strict_parent),
            sealed_run_id=RUN_ID,
            run_root=str(self.run_root),
        )

    def validate_cp4(
        self,
        registry: ExternalTranscriptRegistry,
        *,
        stage: tuple[Publication, ...] | None = None,
    ) -> dict[str, object]:
        root_fd = os.open(
            self.run_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            return validate_cp4_semantics(
                root_fd,
                RUN_ID,
                str(self.run_root),
                self.cp4_stage if stage is None else stage,
                self.cp3_inputs,
                registry,
                self.cp3_checkpoint.evidence.sha256 or "",
                self.cp3_checkpoint.evidence.publication_transcript_sha256,
                expected_result_count=1,
            )
        finally:
            os.close(root_fd)

    def validate_cp5(
        self, registry: ExternalTranscriptRegistry
    ) -> dict[str, object]:
        root_fd = os.open(
            self.run_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            return validate_cp5_semantics(
                root_fd,
                RUN_ID,
                str(self.run_root),
                self.cp5_stage,
                self.cp3_inputs,
                self.cp4_inputs,
                registry,
                self.cp4_checkpoint.evidence.sha256 or "",
                self.cp4_checkpoint.evidence.publication_transcript_sha256,
                expected_result_count=1,
            )
        finally:
            os.close(root_fd)


class CP45SemanticTests(unittest.TestCase):
    def make_fixture(self, **kwargs: object) -> CP45Fixture:
        temporary = tempfile.TemporaryDirectory(prefix="exp7-cp45-semantic-v6-")
        self.addCleanup(temporary.cleanup)
        return CP45Fixture(Path(temporary.name), **kwargs)

    def test_cp4_derives_unpublished_plan_and_verifies_exact_copies(self) -> None:
        fixture = self.make_fixture()
        with fixture.registry() as registry:
            semantic = fixture.validate_cp4(registry)
        self.assertEqual(semantic["schema"], "experiments7-cp4-semantic-binding/v6")
        self.assertEqual(semantic["checkpoint"], 4)
        self.assertEqual(
            semantic["copy_reconciliation"]["copy_state"], "COPIED_BYTES_VERIFIED"
        )
        self.assertFalse(semantic["copy_plan_published"])
        self.assertFalse((fixture.run_root / "copy-plan.json").exists())

    def test_cp4_fails_closed_on_extra_or_missing_paths(self) -> None:
        extra = self.make_fixture(extra_raw=True)
        with extra.registry() as registry:
            with self.assertRaisesRegex(StrictRunError, "CP4_TREE_INVALID"):
                extra.validate_cp4(registry)
        missing = self.make_fixture()
        stage = tuple(
            publication
            for publication in missing.cp4_stage
            if publication.evidence.relative_path != missing.destination
        )
        with missing.registry() as registry:
            with self.assertRaisesRegex(StrictRunError, "CP45_STAGE_PATH_INVALID"):
                missing.validate_cp4(registry, stage=stage)

    def test_cp4_rejects_unregistered_transcript(self) -> None:
        fixture = self.make_fixture()
        with fixture.registry(omit_path="copies.jsonl") as registry:
            with self.assertRaisesRegex(StrictRunError, "TRANSCRIPT_NOT_REGISTERED"):
                fixture.validate_cp4(registry)

    def test_cp4_rejects_ledger_not_bound_to_current_destination_descriptor(self) -> None:
        fixture = self.make_fixture(destination_identity_mismatch=True)
        with fixture.registry() as registry:
            with self.assertRaisesRegex(StrictRunError, "CP4_LEDGER_INVALID"):
                fixture.validate_cp4(registry)

    def test_cp4_rejects_any_published_copy_plan_artifact(self) -> None:
        fixture = self.make_fixture(published_copy_plan=True)
        with fixture.registry() as registry:
            with self.assertRaisesRegex(StrictRunError, "CP4_COPY_PLAN_INVALID"):
                fixture.validate_cp4(registry)

    def test_cp4_rejects_missing_wrong_and_resealed_cp3_semantics(self) -> None:
        for mode in (
            "missing",
            "wrong",
            "resealed",
            "resealed_predecessor",
            "resealed_provenance",
            "duplicate_provenance",
            "nonmapping_provenance",
            "extra_provenance_key",
            "duplicate_bucket",
            "bool_count",
            "reordered_stage",
            "reordered_provenance",
        ):
            with self.subTest(mode=mode):
                fixture = self.make_fixture(cp3_semantic_mode=mode)
                with fixture.registry() as registry:
                    with self.assertRaisesRegex(
                        StrictRunError, "CP45_CHECKPOINT_INVALID"
                    ):
                        fixture.validate_cp4(registry)

    def test_cp4_rejects_extra_cp3_bucket_and_checkpoint_stage(self) -> None:
        for kwargs in ({"extra_bucket": True}, {"extra_cp3_stage": True}):
            with self.subTest(**kwargs):
                fixture = self.make_fixture(**kwargs)
                with fixture.registry() as registry:
                    with self.assertRaisesRegex(
                        StrictRunError, "CP45_(?:CP3|CHECKPOINT)_INVALID"
                    ):
                        fixture.validate_cp4(registry)

    def test_cp5_replays_cp3_cp4_and_verifies_final_admission(self) -> None:
        fixture = self.make_fixture()
        with fixture.registry() as registry:
            semantic = fixture.validate_cp5(registry)
        self.assertEqual(semantic["schema"], "experiments7-cp5-semantic-binding/v6")
        self.assertEqual(semantic["checkpoint"], 5)
        self.assertEqual(semantic["final_reconciliation"]["final_state"], "VERIFIED")
        self.assertTrue(semantic["migration"]["source_post_independent_publication"])

    def test_cp5_rejects_post_manifest_drift(self) -> None:
        fixture = self.make_fixture(source_post_mismatch=True)
        with fixture.registry() as registry:
            with self.assertRaisesRegex(StrictRunError, "CP5_MIGRATION_INVALID"):
                fixture.validate_cp5(registry)

    def test_cp5_rejects_nonrecomputed_reconciliation_and_extra_path(self) -> None:
        mismatch = self.make_fixture(final_reconciliation_mismatch=True)
        with mismatch.registry() as registry:
            with self.assertRaisesRegex(StrictRunError, "CP5_FINAL_INVALID"):
                mismatch.validate_cp5(registry)
        extra = self.make_fixture(extra_final=True)
        with extra.registry() as registry:
            with self.assertRaisesRegex(StrictRunError, "CP5_TREE_INVALID"):
                extra.validate_cp5(registry)

    def test_cp5_rejects_cp4_checkpoint_stage_misbinding(self) -> None:
        fixture = self.make_fixture(cp4_checkpoint_omits_copy=True)
        with fixture.registry() as registry:
            with self.assertRaisesRegex(StrictRunError, "CP45_CHECKPOINT_INVALID"):
                fixture.validate_cp5(registry)

    def test_cp5_rejects_missing_wrong_and_fully_resealed_cp4_semantics(self) -> None:
        expected = {
            "missing": "CP45_CHECKPOINT_INVALID",
            "wrong": "CP45_CHECKPOINT_INVALID",
            "resealed": "CP5_PREDECESSOR_INVALID",
        }
        for mode, code in expected.items():
            with self.subTest(mode=mode):
                fixture = self.make_fixture(cp4_semantic_mode=mode)
                with fixture.registry() as registry:
                    with self.assertRaisesRegex(StrictRunError, code):
                        fixture.validate_cp5(registry)


if __name__ == "__main__":
    unittest.main()
