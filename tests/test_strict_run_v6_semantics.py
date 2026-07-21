from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strict_run.canonical import StrictRunError, canonical_json, sha256_bytes  # noqa: E402
from strict_run.publication import Publication, StagePublisher  # noqa: E402
from strict_run.semantics import (  # noqa: E402
    CP0_ROLES,
    DENIAL_PROBES,
    DISPATCH_STAGES,
    validate_cp2_artifact_structure,
    validate_stage_semantics,
)
from strict_run.writer_policy import (  # noqa: E402
    WriterIdentity,
    default_writer_policy,
)


RUN_ID = "exp7-strict-v6-20260717T120000Z-" + "a" * 32
CP5_SHA256 = "5" * 64
CP5_TRANSCRIPT_SHA256 = "6" * 64


class StrictRunV6SemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="exp7-semantics-v6-")
        self.base = Path(self.temporary.name)
        self.parent = self.base / "strict-runs"
        self.parent.mkdir()
        self.run_root = self.parent / RUN_ID
        self.run_root.mkdir()
        self.root_fd = os.open(
            self.run_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        self.prior_publications: list[Publication] = []
        self.prior_evidence: list[Publication] = []
        for path, payload, writer in (
            (
                "frozen/tool-evidence.json",
                {"kind": "tool"},
                WriterIdentity("g0", "g0:cp6-prior-fixture", "g0-writer"),
            ),
            (
                "runtime/runtime-evidence.json",
                {"kind": "runtime"},
                WriterIdentity(
                    "adapter_snapshot",
                    "snapshot:cp6-prior-fixture",
                    "adapter_snapshot-writer",
                ),
            ),
            (
                "admission/final/test-evidence.json",
                {"kind": "test"},
                WriterIdentity(
                    "admission",
                    "admission:cp6-prior-fixture",
                    "admission-writer",
                ),
            ),
        ):
            created = self.publisher(writer).publish_json(path, payload)
            self.prior_publications.extend(created)
            self.prior_evidence.append(created[-1])
        self.prior_digests = [
            sha256_bytes(canonical_json(item.evidence.to_dict()))
            for item in self.prior_evidence
        ]

    def tearDown(self) -> None:
        os.close(self.root_fd)
        self.temporary.cleanup()

    def publisher(self, writer: WriterIdentity) -> StagePublisher:
        return StagePublisher(
            str(self.parent),
            RUN_ID,
            str(self.run_root),
            writer,
            default_writer_policy(),
        )

    def cp6_publications(
        self,
        *,
        verdict: str = "PASS",
        omit: str | None = None,
        add_extra: bool = False,
        tool_evidence_sha256: str | None = None,
        runtime_evidence_sha256: str | None = None,
        test_evidence_sha256: str | None = None,
    ) -> list[Publication]:
        publications: list[Publication] = []
        cases = (
            (
                "verification/verifier.json",
                "verifier",
                WriterIdentity(
                    "verification",
                    "verification:verifier:fixture",
                    "verification-verifier-writer",
                ),
            ),
            (
                "verification/code-reviewer.json",
                "code-reviewer",
                WriterIdentity(
                    "verification",
                    "verification:code-reviewer:fixture",
                    "verification-code-reviewer-writer",
                ),
            ),
            (
                "verification/adversarial-qa.json",
                "adversarial-qa",
                WriterIdentity(
                    "verification",
                    "verification:adversarial-qa:fixture",
                    "verification-adversarial-qa-writer",
                ),
            ),
        )
        for path, reviewer, writer in cases:
            if reviewer == omit:
                continue
            publications.extend(
                self.publisher(writer).publish_json(
                    path,
                    {
                        "schema": "experiments7-cp6-assurance-report/v6",
                        "sealed_run_id": RUN_ID,
                        "run_root": str(self.run_root),
                        "reviewer": reviewer,
                        "cp5_sha256": CP5_SHA256,
                        "cp5_transcript_sha256": CP5_TRANSCRIPT_SHA256,
                        "tool_evidence_sha256": (
                            self.prior_digests[0]
                            if tool_evidence_sha256 is None
                            else tool_evidence_sha256
                        ),
                        "runtime_evidence_sha256": (
                            self.prior_digests[1]
                            if runtime_evidence_sha256 is None
                            else runtime_evidence_sha256
                        ),
                        "test_evidence_sha256": (
                            self.prior_digests[2]
                            if test_evidence_sha256 is None
                            else test_evidence_sha256
                        ),
                        "verdict": verdict if reviewer == "verifier" else "PASS",
                    },
                )
            )
        if add_extra:
            publications.extend(
                self.publisher(
                    WriterIdentity("g0", "g0:fixture", "g0-writer")
                ).publish_json(
                    "frozen/extra.json", {"unexpected": "artifact"}
                )
            )
        return publications

    def validate_cp6(
        self,
        publications: list[Publication],
        checkpoint: object = 6,
        *,
        prior_publications: list[Publication] | None = None,
    ) -> dict[str, object]:
        return validate_stage_semantics(
            self.root_fd,
            RUN_ID,
            str(self.run_root),
            checkpoint,  # type: ignore[arg-type]
            publications,
            CP5_SHA256,
            CP5_TRANSCRIPT_SHA256,
            prior_publications=(
                self.prior_publications
                if prior_publications is None
                else prior_publications
            ),
        )

    def test_cp6_accepts_only_resolved_current_prior_evidence(self) -> None:
        publications = self.cp6_publications()
        result = self.validate_cp6(publications)
        self.assertEqual(result["checkpoint"], 6)
        self.assertEqual(result["report_count"], 3)

        with self.assertRaises(StrictRunError) as caught:
            self.validate_cp6(publications, prior_publications=[])
        self.assertEqual(caught.exception.code, "CP6_PRIOR_EVIDENCE_MISSING")

    def test_cp6_rejects_arbitrary_resealed_prior_evidence_digest(self) -> None:
        with self.assertRaises(StrictRunError) as caught:
            self.validate_cp6(
                self.cp6_publications(tool_evidence_sha256="f" * 64)
            )
        self.assertEqual(caught.exception.code, "CP6_PRIOR_EVIDENCE_UNRESOLVED")

    def test_cp6_rejects_wrong_evidence_category_and_digest_reuse(self) -> None:
        with self.assertRaises(StrictRunError) as caught:
            self.validate_cp6(
                self.cp6_publications(
                    tool_evidence_sha256=self.prior_digests[2],
                    test_evidence_sha256=self.prior_digests[0],
                )
            )
        self.assertEqual(
            caught.exception.code,
            "CP6_PRIOR_EVIDENCE_CATEGORY_INVALID",
        )

        duplicate_fixture = StrictRunV6SemanticsTests("runTest")
        duplicate_fixture.setUp()
        try:
            digest = duplicate_fixture.prior_digests[0]
            with self.assertRaises(StrictRunError) as caught:
                duplicate_fixture.validate_cp6(
                    duplicate_fixture.cp6_publications(
                        tool_evidence_sha256=digest,
                        runtime_evidence_sha256=digest,
                        test_evidence_sha256=digest,
                    )
                )
            self.assertEqual(
                caught.exception.code,
                "CP6_PRIOR_EVIDENCE_NOT_DISTINCT",
            )
        finally:
            duplicate_fixture.tearDown()

    def test_cp6_rejects_missing_empty_and_extra_report_sets(self) -> None:
        cases = (
            self.cp6_publications(omit="adversarial-qa"),
            [],
        )
        for publications in cases:
            with self.subTest(count=len(publications)), self.assertRaises(StrictRunError):
                self.validate_cp6(publications)

        extra_fixture = StrictRunV6SemanticsTests("runTest")
        extra_fixture.setUp()
        try:
            with self.assertRaises(StrictRunError):
                extra_fixture.validate_cp6(extra_fixture.cp6_publications(add_extra=True))
        finally:
            extra_fixture.tearDown()

    def test_cp6_rejects_non_pass_verdict(self) -> None:
        with self.assertRaises(StrictRunError) as caught:
            self.validate_cp6(self.cp6_publications(verdict="FAIL"))
        self.assertEqual(caught.exception.code, "CP6_REPORT_INVALID")

    def test_common_gate_rejects_boolean_checkpoint_and_duplicate_publication(self) -> None:
        publications = self.cp6_publications()
        with self.assertRaises(StrictRunError) as caught:
            self.validate_cp6(publications, True)
        self.assertEqual(caught.exception.code, "SEMANTICS_UNSUPPORTED_CHECKPOINT")
        with self.assertRaises(StrictRunError) as caught:
            self.validate_cp6(publications + [publications[-1]])
        self.assertEqual(caught.exception.code, "SEMANTICS_DUPLICATE_PATH")



class StrictRunV6CP2SemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="exp7-cp2-semantics-v6-")
        self.base = Path(self.temporary.name)
        self.parent = self.base / "strict-runs"
        self.parent.mkdir()
        self.run_root = self.parent / RUN_ID
        self.run_root.mkdir()
        self.root_fd = os.open(
            self.run_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        self.policy = default_writer_policy()

    def tearDown(self) -> None:
        os.close(self.root_fd)
        self.temporary.cleanup()

    @staticmethod
    def ref(publication: Publication) -> dict[str, str]:
        evidence = publication.evidence
        return {
            "relative_path": evidence.relative_path,
            "artifact_evidence_sha256": sha256_bytes(canonical_json(evidence.to_dict())),
        }

    def publisher(
        self,
        writer: WriterIdentity,
        *,
        allow_reserved_paths: bool = False,
    ) -> StagePublisher:
        return StagePublisher(
            str(self.parent),
            RUN_ID,
            str(self.run_root),
            writer,
            self.policy,
            allow_reserved_paths=allow_reserved_paths,
        )

    def cp1_basis(self) -> tuple[str, str, Publication]:
        registry = self.publisher(
            WriterIdentity("registry", "registry:cp2-fixture", "registry-writer")
        )
        cp1_publications: list[Publication] = []
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
        cp1_publications.extend(created)
        lineage_publication = created[-1]
        created = registry.publish_json(
            "registry/profile-basis.json",
            {
                "schema": "experiments7-cp1-profile-basis/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "profile_ids": profile_ids,
            },
        )
        cp1_publications.extend(created)
        profile_publication = created[-1]
        created = registry.publish_json(
            "registry/cp2-basis.json",
            {
                "schema": "experiments7-cp1-cp2-basis/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "lineage_source": self.ref(lineage_publication),
                "profile_source": self.ref(profile_publication),
            },
        )
        cp1_publications.extend(created)
        basis_publication = created[-1]
        structure_rows = [
            {"paper_section_id": f"section-{index:02d}", "ordinal": index}
            for index in range(51)
        ]
        inventory_rows = [
            {
                "result_id": f"result:cp2-structure:{index:04d}",
                "paper_section_id": f"section-{index % 51:02d}",
                "numeric_value": index,
                "display_value": str(index),
            }
            for index in range(1908)
        ]
        alias_rows = [
            {
                "alias_id": f"alias:cp2-structure:{index:03d}",
                "target_result_id": f"result:cp2-structure:{index:04d}",
            }
            for index in range(86)
        ]
        inventory_publisher = self.publisher(
            WriterIdentity(
                "inventory", "inventory:cp2-fixture", "inventory-writer"
            )
        )
        for path, rows in (
            ("inventory/structure.jsonl", structure_rows),
            ("inventory/results.jsonl", inventory_rows),
            ("inventory/aliases.jsonl", alias_rows),
        ):
            cp1_publications.extend(
                inventory_publisher.publish_bytes(
                    path,
                    b"".join(canonical_json(row) for row in rows),
                )
            )
        semantic_bindings = validate_stage_semantics(
            self.root_fd,
            RUN_ID,
            str(self.run_root),
            1,
            cp1_publications,
            "0" * 64,
            "1" * 64,
        )
        actual_paths = [
            publication.evidence.to_dict()
            for publication in sorted(
                cp1_publications, key=lambda item: item.evidence.relative_path
            )
        ]
        checkpoint = {
            "schema": "experiments7-strict-checkpoint/v6",
            "sealed_run_id": RUN_ID,
            "run_root": str(self.run_root),
            "checkpoint": 1,
            "previous_checkpoint_sha256": "0" * 64,
            "previous_checkpoint_transcript_sha256": "1" * 64,
            "stage_actual_paths": actual_paths,
            "stage_actual_paths_sha256": sha256_bytes(canonical_json(actual_paths)),
            "bindings": {},
            "semantic_bindings": semantic_bindings,
            "semantic_bindings_sha256": sha256_bytes(
                canonical_json(semantic_bindings)
            ),
        }
        controller = self.publisher(
            WriterIdentity(
                "checkpoint_controller",
                "checkpoint:cp1-fixture",
                "checkpoint_controller-writer",
            ),
            allow_reserved_paths=True,
        )
        checkpoint_publication = controller.publish_json(
            "checkpoints/cp1.json", checkpoint
        )[-1]
        return (
            str(checkpoint_publication.evidence.sha256),
            checkpoint_publication.evidence.publication_transcript_sha256,
            basis_publication,
        )

    def cp2_publications(
        self,
        *,
        boolean_lineage_count: bool = False,
        extra_unowned: bool = False,
    ) -> tuple[list[Publication], str, str]:
        cp1_sha256, cp1_transcript_sha256, basis_publication = self.cp1_basis()
        basis_ref = self.ref(basis_publication)
        writer = WriterIdentity(
            "adapter_snapshot", "snapshot:cp2-fixture", "adapter_snapshot-writer"
        )
        publisher = self.publisher(writer)
        publications: list[Publication] = []
        files: dict[str, Publication] = {}

        def publish_bytes(path: str, payload: bytes) -> Publication:
            created = publisher.publish_bytes(path, payload)
            publications.extend(created)
            files[path] = created[-1]
            return created[-1]

        def publish_json(path: str, value: object) -> Publication:
            created = publisher.publish_json(path, value)
            publications.extend(created)
            files[path] = created[-1]
            return created[-1]

        lineage_ids = [f"lineage-{index:03d}" for index in range(129)]
        profile_ids = [f"profile-{index:02d}" for index in range(30)]
        snapshot_records = []
        for lineage_id in lineage_ids:
            output = publish_bytes(
                f"snapshots/lineage/{lineage_id}.bin",
                f"snapshot:{lineage_id}\n".encode("utf-8"),
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
                "lineage_count": True if boolean_lineage_count else len(lineage_ids),
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
        config_publications: dict[str, Publication] = {}
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
            config_publications[profile_id] = config
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
            config = config_publications[profile_id]
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
        if extra_unowned:
            publish_bytes("runtime/extra.bin", b"unowned\n")

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

        snapshot_owned = refs(snapshot_owned_paths)
        runtime_owned = refs(runtime_owned_paths)
        replay_owned = refs(replay_owned_paths)
        seals = (
            (
                "snapshots/snapshot-producer-seal.json",
                "experiments7-cp2-snapshot-producer-seal/v6",
                snapshot_report,
                snapshot_owned,
                {
                    "verdict": "PASS",
                    "expected_lineage_count": len(lineage_ids),
                    "passed_lineage_count": len(lineage_ids),
                    "lineage_ids_sha256": sha256_bytes(canonical_json(lineage_ids)),
                },
            ),
            (
                "runtime/runtime-sandbox-seal.json",
                "experiments7-cp2-runtime-sandbox-seal/v6",
                runtime_report,
                runtime_owned,
                {
                    "verdict": "PASS",
                    "expected_profile_count": len(profile_ids),
                    "config_closed_count": len(profile_ids),
                    "dependency_count": 1,
                    "denial_probe_count": len(DENIAL_PROBES),
                },
            ),
            (
                "runtime/hermetic-profile-replay-seal.json",
                "experiments7-cp2-hermetic-profile-replay-seal/v6",
                replay_report,
                replay_owned,
                {
                    "verdict": "PASS",
                    "expected_profile_count": len(profile_ids),
                    "passed_profile_count": len(profile_ids),
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
        return publications, cp1_sha256, cp1_transcript_sha256

    def validate_cp2(
        self,
        publications: list[Publication],
        cp1_sha256: str,
        cp1_transcript_sha256: str,
    ) -> dict[str, object]:
        return validate_cp2_artifact_structure(
            self.root_fd,
            RUN_ID,
            str(self.run_root),
            publications,
            cp1_sha256,
            cp1_transcript_sha256,
        )

    def test_cp2_production_dispatch_requires_runtime_v2_positive_environment(self) -> None:
        with self.assertRaises(StrictRunError) as caught:
            validate_stage_semantics(
                self.root_fd,
                RUN_ID,
                str(self.run_root),
                2,
                [],
                "0" * 64,
                "1" * 64,
            )
        self.assertEqual(
            caught.exception.code, "RUNTIME_PROOF_ENVIRONMENT_UNAVAILABLE"
        )

    def test_cp2_structure_fixture_round_trips_without_granting_acceptance(self) -> None:
        publications, cp1_sha256, cp1_transcript_sha256 = self.cp2_publications()
        result = self.validate_cp2(
            publications, cp1_sha256, cp1_transcript_sha256
        )
        self.assertEqual(result["schema"], "experiments7-cp2-semantics/v6")
        self.assertEqual(result["lineage_count"], 129)
        self.assertEqual(result["profile_count"], 30)
        self.assertEqual(len(result["seals"]), 3)

    def test_cp2_rejects_boolean_count(self) -> None:
        publications, cp1_sha256, cp1_transcript_sha256 = self.cp2_publications(
            boolean_lineage_count=True
        )
        with self.assertRaises(StrictRunError) as caught:
            self.validate_cp2(publications, cp1_sha256, cp1_transcript_sha256)
        self.assertEqual(caught.exception.code, "SEMANTICS_SCHEMA_INVALID")

    def test_cp2_rejects_unowned_extra_artifact(self) -> None:
        publications, cp1_sha256, cp1_transcript_sha256 = self.cp2_publications(
            extra_unowned=True
        )
        with self.assertRaises(StrictRunError) as caught:
            self.validate_cp2(publications, cp1_sha256, cp1_transcript_sha256)
        self.assertEqual(caught.exception.code, "CP2_OWNERSHIP_INVALID")

    def test_cp2_rejects_empty_stage(self) -> None:
        cp1_sha256, cp1_transcript_sha256, _ = self.cp1_basis()
        with self.assertRaises(StrictRunError) as caught:
            self.validate_cp2([], cp1_sha256, cp1_transcript_sha256)
        self.assertEqual(caught.exception.code, "CP2_REPORT_INVALID")


class StrictRunV6CP1SemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="exp7-cp1-semantics-v6-")
        self.base = Path(self.temporary.name)
        self.parent = self.base / "strict-runs"
        self.parent.mkdir()
        self.run_root = self.parent / RUN_ID
        self.run_root.mkdir()
        self.root_fd = os.open(
            self.run_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        self.policy = default_writer_policy()

    def tearDown(self) -> None:
        os.close(self.root_fd)
        self.temporary.cleanup()

    @staticmethod
    def ref(publication: Publication) -> dict[str, str]:
        evidence = publication.evidence
        return {
            "relative_path": evidence.relative_path,
            "artifact_evidence_sha256": sha256_bytes(
                canonical_json(evidence.to_dict())
            ),
        }

    def publisher(self, writer: WriterIdentity) -> StagePublisher:
        return StagePublisher(
            str(self.parent), RUN_ID, str(self.run_root), writer, self.policy
        )

    def cp1_publications(
        self,
        *,
        structure_count: int = 51,
        inventory_count: int = 1908,
        alias_count: int = 86,
        extra_registry: bool = False,
        omit_inventory_path: str | None = None,
        hostile_inventory: str | None = None,
    ) -> list[Publication]:
        registry = self.publisher(
            WriterIdentity("registry", "registry:cp1-fixture", "registry-writer")
        )
        inventory = self.publisher(
            WriterIdentity("inventory", "inventory:cp1-fixture", "inventory-writer")
        )
        publications: list[Publication] = []
        created = registry.publish_json(
            "registry/lineage-basis.json",
            {
                "schema": "experiments7-cp1-lineage-basis/v6",
                "sealed_run_id": RUN_ID,
                "run_root": str(self.run_root),
                "lineage_ids": [f"lineage-{index:03d}" for index in range(129)],
            },
        )
        publications.extend(created)
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
        publications.extend(created)
        profiles = created[-1]
        publications.extend(
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
            {
                "paper_section_id": f"section-{index:02d}",
                "ordinal": index,
            }
            for index in range(structure_count)
        ]
        results = [
            {
                "result_id": f"result:cp1:{index:04d}",
                "paper_section_id": f"section-{index % max(structure_count, 1):02d}",
                "numeric_value": index,
                "display_value": str(index),
            }
            for index in range(inventory_count)
        ]
        aliases = [
            {
                "alias_id": f"alias:cp1:{index:03d}",
                "target_result_id": f"result:cp1:{index % max(inventory_count, 1):04d}",
            }
            for index in range(alias_count)
        ]
        if hostile_inventory == "duplicate-structure-id" and len(structures) > 1:
            structures[1]["paper_section_id"] = structures[0]["paper_section_id"]
        elif hostile_inventory == "duplicate-result-id" and len(results) > 1:
            results[1]["result_id"] = results[0]["result_id"]
        elif hostile_inventory == "duplicate-alias-id" and len(aliases) > 1:
            aliases[1]["alias_id"] = aliases[0]["alias_id"]
        elif hostile_inventory == "missing-section-reference" and results:
            results[0]["paper_section_id"] = "section-missing"
        elif hostile_inventory == "missing-alias-target" and aliases:
            aliases[0]["target_result_id"] = "result:missing"
        elif hostile_inventory == "empty-stable-id" and aliases:
            aliases[0]["alias_id"] = ""

        structure_bytes = b"".join(canonical_json(record) for record in structures)
        inventory_bytes = b"".join(canonical_json(record) for record in results)
        alias_bytes = b"".join(canonical_json(record) for record in aliases)
        if hostile_inventory == "nonfinite":
            first, separator, rest = inventory_bytes.partition(b"\n")
            first = first.replace(b'"numeric_value":0', b'"numeric_value":NaN')
            inventory_bytes = first + separator + rest
        elif hostile_inventory == "duplicate":
            first, separator, rest = inventory_bytes.partition(b"\n")
            first = first.replace(
                b'"result_id":"result:cp1:0000"',
                b'"result_id":"result:cp1:0000","result_id":"result:cp1:0000"',
            )
            inventory_bytes = first + separator + rest
        for path, payload in (
            ("inventory/structure.jsonl", structure_bytes),
            ("inventory/results.jsonl", inventory_bytes),
            ("inventory/aliases.jsonl", alias_bytes),
        ):
            if omit_inventory_path != path:
                publications.extend(inventory.publish_bytes(path, payload))
        if extra_registry:
            publications.extend(registry.publish_json("registry/extra.json", {"extra": True}))
        return publications

    def validate_cp1(self, publications: list[Publication]) -> dict[str, object]:
        return validate_stage_semantics(
            self.root_fd,
            RUN_ID,
            str(self.run_root),
            1,
            publications,
            "0" * 64,
            "1" * 64,
        )

    def test_cp1_exact_basis_and_51_1908_86_inventory_round_trip(self) -> None:
        result = self.validate_cp1(self.cp1_publications())
        self.assertEqual(result["schema"], "experiments7-cp1-semantics/v6")
        self.assertEqual(result["lineage_count"], 129)
        self.assertEqual(result["profile_count"], 30)
        self.assertEqual(result["inventory_structure_count"], 51)
        self.assertEqual(result["inventory_record_count"], 1908)
        self.assertEqual(result["inventory_alias_count"], 86)
        self.assertEqual(len(result["inventory_sha256"]), 64)

    def test_cp1_rejects_wrong_inventory_count_and_unowned_extra(self) -> None:
        with self.assertRaises(StrictRunError) as caught:
            self.validate_cp1(self.cp1_publications(inventory_count=1907))
        self.assertEqual(caught.exception.code, "CP1_INVENTORY_INVALID")

        second = StrictRunV6CP1SemanticsTests("runTest")
        second.setUp()
        try:
            with self.assertRaises(StrictRunError) as caught:
                second.validate_cp1(second.cp1_publications(extra_registry=True))
            self.assertEqual(caught.exception.code, "CP1_ARTIFACT_SET_INVALID")
        finally:
            second.tearDown()

    def test_cp1_rejects_missing_inventory_file_and_each_wrong_count(self) -> None:
        cases = (
            {"omit_inventory_path": "inventory/structure.jsonl"},
            {"omit_inventory_path": "inventory/results.jsonl"},
            {"omit_inventory_path": "inventory/aliases.jsonl"},
            {"structure_count": 50},
            {"inventory_count": 1907},
            {"alias_count": 85},
        )
        for kwargs in cases:
            fixture = StrictRunV6CP1SemanticsTests("runTest")
            fixture.setUp()
            try:
                with self.assertRaises(StrictRunError) as caught:
                    fixture.validate_cp1(fixture.cp1_publications(**kwargs))
                self.assertIn(
                    caught.exception.code,
                    {"CP1_ARTIFACT_SET_INVALID", "CP1_INVENTORY_INVALID"},
                )
            finally:
                fixture.tearDown()

    def test_cp1_rejects_duplicate_empty_and_dangling_stable_ids(self) -> None:
        for hostile in (
            "duplicate-structure-id",
            "duplicate-result-id",
            "duplicate-alias-id",
            "missing-section-reference",
            "missing-alias-target",
            "empty-stable-id",
        ):
            fixture = StrictRunV6CP1SemanticsTests("runTest")
            fixture.setUp()
            try:
                with self.assertRaises(StrictRunError) as caught:
                    fixture.validate_cp1(
                        fixture.cp1_publications(hostile_inventory=hostile)
                    )
                self.assertEqual(caught.exception.code, "CP1_INVENTORY_INVALID")
            finally:
                fixture.tearDown()

    def test_cp1_rejects_nonfinite_and_duplicate_key_inventory_json(self) -> None:
        for hostile in ("nonfinite", "duplicate"):
            fixture = StrictRunV6CP1SemanticsTests("runTest")
            fixture.setUp()
            try:
                with self.assertRaises(StrictRunError) as caught:
                    fixture.validate_cp1(
                        fixture.cp1_publications(hostile_inventory=hostile)
                    )
                self.assertIn(
                    caught.exception.code,
                    {"NONFINITE_JSON_NUMBER", "DUPLICATE_JSON_KEY"},
                )
            finally:
                fixture.tearDown()


class StrictRunV6CP0SemanticsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="exp7-cp0-semantics-v6-")
        self.base = Path(self.temporary.name)
        self.parent = self.base / "strict-runs"
        self.parent.mkdir()
        self.run_root = self.parent / RUN_ID
        self.run_root.mkdir()
        self.root_fd = os.open(
            self.run_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        self.policy = default_writer_policy()

    def tearDown(self) -> None:
        os.close(self.root_fd)
        self.temporary.cleanup()

    @staticmethod
    def ref(publication: Publication) -> dict[str, str]:
        evidence = publication.evidence
        return {
            "relative_path": evidence.relative_path,
            "artifact_evidence_sha256": sha256_bytes(canonical_json(evidence.to_dict())),
        }

    def publisher(self, writer: WriterIdentity) -> StagePublisher:
        return StagePublisher(
            str(self.parent), RUN_ID, str(self.run_root), writer, self.policy
        )

    def cp0_publications(
        self,
        *,
        empty_role: bool = False,
        omit_role: bool = False,
        duplicate_role: bool = False,
        boolean_count: bool = False,
        extra_frozen: bool = False,
        hostile_paper: str | None = None,
    ) -> list[Publication]:
        publications: list[Publication] = []
        publications.extend(
            StagePublisher(
                str(self.parent),
                RUN_ID,
                str(self.run_root),
                WriterIdentity(
                    "checkpoint_controller",
                    "checkpoint:cp0-fixture",
                    "checkpoint_controller-writer",
                ),
                self.policy,
                allow_reserved_paths=True,
            ).ensure_directory("checkpoints")
        )
        reservation = self.publisher(
            WriterIdentity(
                "run_reservation", "reservation:cp0-fixture", "run_reservation-writer"
            )
        )
        publications.extend(
            reservation.publish_json("reservation.json", {"reservation": "sealed"})
        )
        acceptance = self.publisher(
            WriterIdentity(
                "reservation_acceptance",
                "acceptance:cp0-fixture",
                "reservation_acceptance-writer",
            )
        )
        publications.extend(
            acceptance.publish_json(
                "reservation-acceptance.json", {"acceptance": "bound"}
            )
        )
        g0 = self.publisher(WriterIdentity("g0", "g0:cp0-fixture", "g0-writer"))
        publications.extend(
            g0.publish_json("owner-binding.json", {"owner": "g0"})
        )
        created = g0.publish_bytes(
            "manifests/source-pre.jsonl",
            canonical_json({"relative_path": "source.py", "sha256": "1" * 64}),
        )
        publications.extend(created)
        source_pre = created[-1]
        if hostile_paper == "nonfinite":
            created = g0.publish_bytes(
                "manifests/paper-pre.json", b'{"metric":NaN}\n'
            )
        elif hostile_paper == "duplicate":
            created = g0.publish_bytes(
                "manifests/paper-pre.json", b'{"metric":1,"metric":1}\n'
            )
        else:
            created = g0.publish_json(
                "manifests/paper-pre.json",
                {"relative_path": "paper.pdf", "sha256": "2" * 64},
            )
        publications.extend(created)
        paper_pre = created[-1]

        roles: dict[str, list[dict[str, str]]] = {}
        role_publications: list[Publication] = []
        for role in CP0_ROLES:
            created = g0.publish_bytes(
                f"frozen/{role}/artifact.bin", f"{role}\n".encode("utf-8")
            )
            publications.extend(created)
            role_publications.append(created[-1])
            roles[role] = [self.ref(created[-1])]
        if empty_role:
            roles[CP0_ROLES[0]] = []
        if omit_role:
            roles.pop(CP0_ROLES[-1])
        if duplicate_role:
            roles[CP0_ROLES[1]] = sorted(
                [roles[CP0_ROLES[1]][0], roles[CP0_ROLES[0]][0]],
                key=lambda item: item["relative_path"],
            )
        if extra_frozen:
            publications.extend(g0.publish_bytes("frozen/unowned.bin", b"extra\n"))
        canonical_refs = sorted(
            [self.ref(publication) for publication in role_publications],
            key=lambda item: item["relative_path"],
        )
        publications.extend(
            g0.publish_json(
                "frozen/inventory.json",
                {
                    "schema": "experiments7-cp0-frozen-inventory/v6",
                    "sealed_run_id": RUN_ID,
                    "run_root": str(self.run_root),
                    "roles": roles,
                    "frozen_artifact_count": True if boolean_count else len(role_publications),
                    "frozen_artifacts_sha256": sha256_bytes(canonical_json(canonical_refs)),
                    "source_pre": self.ref(source_pre),
                    "paper_pre": self.ref(paper_pre),
                },
            )
        )
        return publications

    def validate_cp0(self, publications: list[Publication]) -> dict[str, object]:
        return validate_stage_semantics(
            self.root_fd,
            RUN_ID,
            str(self.run_root),
            0,
            publications,
            None,
            None,
        )

    def test_cp0_exact_12_role_exhaustive_inventory_round_trips(self) -> None:
        result = self.validate_cp0(self.cp0_publications())
        self.assertEqual(result["schema"], "experiments7-cp0-semantics/v6")
        self.assertEqual(result["role_count"], 12)
        self.assertEqual(result["frozen_artifact_count"], 12)
        self.assertEqual(result["source_pre_record_count"], 1)

    def test_cp0_rejects_empty_role(self) -> None:
        with self.assertRaises(StrictRunError):
            self.validate_cp0(self.cp0_publications(empty_role=True))

    def test_cp0_rejects_missing_role(self) -> None:
        with self.assertRaises(StrictRunError):
            self.validate_cp0(self.cp0_publications(omit_role=True))

    def test_cp0_rejects_boolean_count(self) -> None:
        with self.assertRaises(StrictRunError) as caught:
            self.validate_cp0(self.cp0_publications(boolean_count=True))
        self.assertEqual(caught.exception.code, "SEMANTICS_SCHEMA_INVALID")

    def test_cp0_rejects_cross_role_duplicate(self) -> None:
        with self.assertRaises(StrictRunError) as caught:
            self.validate_cp0(self.cp0_publications(duplicate_role=True))
        self.assertEqual(caught.exception.code, "CP0_INVENTORY_INVALID")

    def test_cp0_rejects_unaccounted_frozen_extra(self) -> None:
        with self.assertRaises(StrictRunError) as caught:
            self.validate_cp0(self.cp0_publications(extra_frozen=True))
        self.assertEqual(caught.exception.code, "CP0_INVENTORY_INVALID")

    def test_cp0_rejects_nonfinite_and_duplicate_key_paper_json(self) -> None:
        for hostile in ("nonfinite", "duplicate"):
            fixture = StrictRunV6CP0SemanticsTests("runTest")
            fixture.setUp()
            try:
                with self.assertRaises(StrictRunError) as caught:
                    fixture.validate_cp0(
                        fixture.cp0_publications(hostile_paper=hostile)
                    )
                self.assertIn(
                    caught.exception.code,
                    {"NONFINITE_JSON_NUMBER", "DUPLICATE_JSON_KEY"},
                )
            finally:
                fixture.tearDown()

if __name__ == "__main__":
    unittest.main()
