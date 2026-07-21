from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/provenance/strict_v6.py"
SPEC = importlib.util.spec_from_file_location("strict_v6", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
V6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V6)


def positive_record(
    number: int,
    *,
    result_id: str | None = None,
    section: str = "main",
    result_kind: str = "non_inference",
    raw_ids: tuple[str, ...] = (),
    phase: str = "precopy",
    copied_ids: tuple[str, ...] = (),
    copied_edges: tuple[str, ...] = (),
) -> dict[str, object]:
    closure = {
        field: [f"{field}:{number}"]
        for field in V6.CLOSURE_FIELDS
    }
    parents = [f"result:parent:{number}"] if result_kind == "derived_inference" else []
    contributors = set(raw_ids) | set(parents)
    for values in closure.values():
        contributors.update(values)
    return {
        "schema": "experiments7-admission-result/v6",
        "result_id": result_id or f"result:synthetic:{number:04d}",
        "paper_section_id": section,
        "phase": phase,
        "state": "ADMITTED_FOR_COPY" if phase == "precopy" else "VERIFIED",
        "result_kind": result_kind,
        "copy_required": bool(raw_ids),
        "raw_artifact_ids": list(raw_ids),
        "candidate_raw_artifact_ids": [],
        "required_parent_result_ids": parents,
        "closed_parent_result_ids": list(parents),
        "required_contributor_node_ids": sorted(contributors),
        "sealed_contributor_node_ids": sorted(contributors),
        "positive_evidence_origin": "run_local_strict",
        "source_pre_bound": bool(raw_ids),
        "declared_drift": False,
        "ambiguous": False,
        "support_only": False,
        "audit_positive_evidence": False,
        "reason_codes": [],
        "copied_artifact_ids": list(copied_ids),
        "copied_to_edge_ids": list(copied_edges),
        **closure,
    }


def unresolved_record(result_id: str, *, candidates: tuple[str, ...] = ()) -> dict[str, object]:
    return {
        "schema": "experiments7-admission-result/v6",
        "result_id": result_id,
        "paper_section_id": "audit",
        "phase": "precopy",
        "state": "UNRESOLVED",
        "result_kind": "derived",
        "copy_required": False,
        "raw_artifact_ids": [],
        "candidate_raw_artifact_ids": list(candidates),
        "required_parent_result_ids": [],
        "closed_parent_result_ids": [],
        "required_contributor_node_ids": [],
        "sealed_contributor_node_ids": [],
        "positive_evidence_origin": "legacy_audit",
        "source_pre_bound": False,
        "declared_drift": False,
        "ambiguous": False,
        "support_only": False,
        "audit_positive_evidence": False,
        "reason_codes": ["NEW_RUN_TYPED_EVIDENCE_REQUIRED"],
        "copied_artifact_ids": [],
        "copied_to_edge_ids": [],
        **{field: [] for field in V6.CLOSURE_FIELDS},
    }


def inventory_and_buckets(count: int = 3, *, raw_id: str = "raw:synthetic-0"):
    inventory: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for number in range(count):
        section = "main" if number % 2 == 0 else "appendix"
        inventory.append(
            {
                "result_id": f"result:synthetic:{number:04d}",
                "paper_section_id": section,
                "numeric_value": number / 10,
                "display_value": f"{number / 10:.1f}",
            }
        )
        if number == 0:
            record = positive_record(
                number,
                section=section,
                result_kind="inference",
                raw_ids=(raw_id,),
            )
        else:
            record = positive_record(number, section=section)
        records.append(record)
    inv_hash = V6.result_inventory_digest(inventory)
    buckets = [
        V6.make_bucket_subseal(
            inv_hash,
            section,
            [record for record in records if record["paper_section_id"] == section],
        )
        for section in sorted({row["paper_section_id"] for row in inventory})
    ]
    cp3 = V6.reconcile_cp3_buckets(
        inventory, buckets, expected_result_count=count
    )
    return inventory, records, buckets, cp3


def copy_fixture(count: int = 3):
    payload = b"synthetic-raw"
    digest = hashlib.sha256(payload).hexdigest()
    source_identity = {
        "file_type": "regular",
        "st_dev": 10,
        "st_ino": 20,
        "mode": 0o100444,
        "size": len(payload),
        "mtime_ns": 123456,
    }
    source = {
        "schema": "experiments7-source-pre/v6",
        "record_id": "source:synthetic-0",
        "root_id": "experiments4",
        "relative_path": "synthetic/output.json",
        "type": "regular",
        "sha256": digest,
        "size": len(payload),
        "descriptor_identity": source_identity,
    }
    raw_node = V6.make_raw_provenance_node(source)
    inventory, records, buckets, cp3 = inventory_and_buckets(
        count, raw_id=raw_node["node_id"]
    )
    plan = V6.plan_source_pre_copies(
        cp3, [raw_node], [source], expected_result_count=count
    )
    entry = plan["entries"][0]
    destination_identity = {
        "file_type": "regular",
        "st_dev": 11,
        "st_ino": 21,
        "mode": 0o100444,
        "size": len(payload),
        "mtime_ns": 123457,
    }
    ledger = [
        {
            "schema": "experiments7-copy-ledger-row/v6",
            "copy_plan_entry_sha256": V6.digest_json(entry),
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
            "source_sha256_before": digest,
            "source_sha256_after": digest,
            "destination_identity": destination_identity,
            "publication_method": "byte_stream_no_replace",
            "hardlink_used": False,
            "reflink_used": False,
            "clone_used": False,
            "cache_used": False,
            "destination_sha256": digest,
            "destination_size": len(payload),
        }
    ]
    payloads = {entry["copied_artifact_id"]: payload}
    return (
        inventory,
        records,
        buckets,
        cp3,
        source,
        raw_node,
        plan,
        ledger,
        payloads,
    )


def final_records(buckets, plan):
    raw_to_entry = {row["raw_artifact_id"]: row for row in plan["entries"]}
    rows = []
    for bucket in buckets:
        for precopy in bucket["result_records"]:
            row = copy.deepcopy(precopy)
            row["phase"] = "final"
            row["state"] = "VERIFIED"
            row["precopy_record_sha256"] = V6.digest_json(precopy)
            entries = [raw_to_entry[raw_id] for raw_id in row["raw_artifact_ids"]]
            row["copied_artifact_ids"] = [entry["copied_artifact_id"] for entry in entries]
            row["copied_to_edge_ids"] = [entry["copied_to_edge_id"] for entry in entries]
            rows.append(row)
    return rows


class AdmissionStateTests(unittest.TestCase):
    def test_only_three_states_and_phase_legality(self) -> None:
        self.assertEqual(
            V6.LEGAL_STATES,
            {"UNRESOLVED", "ADMITTED_FOR_COPY", "VERIFIED"},
        )
        row = positive_record(1)
        row["state"] = "DERIVED_FROM_ADMITTED_RAW"
        with self.assertRaisesRegex(V6.V6ContractError, "illegal state"):
            V6.validate_admission_record(row, "precopy")
        row = positive_record(1)
        row["state"] = "VERIFIED"
        with self.assertRaisesRegex(V6.V6ContractError, "final-only"):
            V6.validate_admission_record(row, "precopy")
        row = positive_record(1)
        row["schema"] = "experiments7-admission-result/v5"
        with self.assertRaisesRegex(V6.V6ContractError, "schema mismatch"):
            V6.validate_admission_record(row, "precopy")

    def test_inference_zero_raw_and_unsupported_derived_only_are_unresolved(self) -> None:
        inference = positive_record(1, result_kind="inference")
        self.assertEqual(V6.derive_admission_state(inference, "precopy"), "UNRESOLVED")
        with self.assertRaisesRegex(V6.V6ContractError, "state forgery"):
            V6.validate_admission_record(inference, "precopy")
        derived = positive_record(2)
        derived["result_kind"] = "derived"
        self.assertEqual(V6.derive_admission_state(derived, "precopy"), "UNRESOLVED")

    def test_partial_parent_union_and_declared_drift_fail_closed(self) -> None:
        row = positive_record(
            1, result_kind="derived_inference", raw_ids=("raw:a",)
        )
        row["closed_parent_result_ids"] = []
        self.assertEqual(V6.derive_admission_state(row, "precopy"), "UNRESOLVED")
        row = positive_record(2, result_kind="inference", raw_ids=("raw:b",))
        row["declared_drift"] = True
        self.assertEqual(V6.derive_admission_state(row, "precopy"), "UNRESOLVED")

    def test_exclusion_and_binding_booleans_are_explicit_and_typed(self) -> None:
        for field in V6.ADMISSION_BOOLEAN_FIELDS:
            for mutation in ("missing", "false"):
                with self.subTest(field=field, mutation=mutation):
                    row = positive_record(1)
                    if mutation == "missing":
                        del row[field]
                    else:
                        row[field] = "false"
                    self.assertEqual(
                        V6.derive_admission_state(row, "precopy"), "UNRESOLVED"
                    )
                    with self.assertRaisesRegex(V6.V6ContractError, "explicit boolean"):
                        V6.validate_admission_record(row, "precopy")

    def test_closed_zero_raw_non_inference_has_no_copy_nodes(self) -> None:
        row = positive_record(1)
        validated = V6.validate_admission_record(row, "precopy")
        self.assertEqual(validated["state"], "ADMITTED_FOR_COPY")
        self.assertFalse(validated["copy_required"])
        self.assertEqual(validated["raw_artifact_ids"], [])
        self.assertEqual(validated["copied_artifact_ids"], [])
        self.assertEqual(validated["copied_to_edge_ids"], [])

    def test_candidates_are_diagnostics_only(self) -> None:
        row = positive_record(1)
        row["candidate_raw_artifact_ids"] = ["raw:candidate"]
        validated = V6.validate_admission_record(row, "precopy")
        self.assertEqual(validated["raw_artifact_ids"], [])
        self.assertFalse(validated["copy_required"])
        row["raw_artifact_ids"] = ["raw:candidate"]
        row["copy_required"] = True
        self.assertEqual(V6.derive_admission_state(row, "precopy"), "UNRESOLVED")


class BucketTests(unittest.TestCase):
    def test_exact_1908_disjoint_union(self) -> None:
        inventory, _, buckets, cp3 = inventory_and_buckets(1908)
        self.assertEqual(len(inventory), 1908)
        self.assertEqual(cp3["precopy_admitted_count"], 1908)
        self.assertEqual(cp3["precopy_unresolved_count"], 0)
        self.assertEqual(cp3["copy_required_raw_artifact_ids"], ["raw:synthetic-0"])

    def test_result_inventory_digest_binds_full_canonical_row_content(self) -> None:
        inventory, _, buckets, _ = inventory_and_buckets()
        original = V6.result_inventory_digest(inventory)
        reversed_inventory = list(reversed(inventory))
        self.assertNotEqual(original, V6.result_inventory_digest(reversed_inventory))
        with self.assertRaisesRegex(V6.V6ContractError, "stale or foreign bucket"):
            V6.reconcile_cp3_buckets(
                reversed_inventory, buckets, expected_result_count=3
            )
        mutated = copy.deepcopy(inventory)
        mutated[0]["numeric_value"] = 999.0
        self.assertNotEqual(original, V6.result_inventory_digest(mutated))
        with self.assertRaisesRegex(V6.V6ContractError, "stale or foreign bucket"):
            V6.reconcile_cp3_buckets(mutated, buckets, expected_result_count=3)

    def test_partial_duplicate_cross_bucket_and_raw_union_fail(self) -> None:
        inventory, records, buckets, _ = inventory_and_buckets()
        inv_hash = V6.result_inventory_digest(inventory)
        partial = [
            V6.make_bucket_subseal(
                inv_hash,
                section,
                [
                    row
                    for row in records[:-1]
                    if row["paper_section_id"] == section
                ],
            )
            for section in ("appendix", "main")
        ]
        with self.assertRaisesRegex(V6.V6ContractError, "exact inventory"):
            V6.reconcile_cp3_buckets(inventory, partial, expected_result_count=3)
        with self.assertRaisesRegex(V6.V6ContractError, "duplicate bucket"):
            V6.reconcile_cp3_buckets(
                inventory, buckets + [copy.deepcopy(buckets[0])], expected_result_count=3
            )
        tampered = copy.deepcopy(buckets)
        tampered[0]["copy_required_raw_artifact_ids"].append("raw:orphan")
        with self.assertRaisesRegex(V6.V6ContractError, "raw union mismatch"):
            V6.reconcile_cp3_buckets(inventory, tampered, expected_result_count=3)
        cross = copy.deepcopy(records[0])
        cross["paper_section_id"] = "appendix"
        with self.assertRaisesRegex(V6.V6ContractError, "cross-bucket"):
            V6.make_bucket_subseal(inv_hash, "main", [cross])

    def test_bucket_record_mutation_is_detected(self) -> None:
        inventory, _, buckets, _ = inventory_and_buckets()
        tampered = copy.deepcopy(buckets)
        tampered[0]["result_records"][0]["producer_node_ids"] = ["producer:changed"]
        with self.assertRaises(V6.V6ContractError):
            V6.reconcile_cp3_buckets(inventory, tampered, expected_result_count=3)

    def test_bucket_local_duplicate_rejects_even_when_resealed(self) -> None:
        inventory, _, buckets, _ = inventory_and_buckets()
        tampered = copy.deepcopy(buckets)
        bucket = tampered[0]
        bucket["result_records"].append(copy.deepcopy(bucket["result_records"][0]))
        bucket["result_records"].sort(key=lambda row: row["result_id"])
        bucket["result_ids"] = [row["result_id"] for row in bucket["result_records"]]
        bucket["result_records_sha256"] = V6.digest_records(bucket["result_records"])
        with self.assertRaisesRegex(V6.V6ContractError, "duplicate result in bucket"):
            V6.reconcile_cp3_buckets(inventory, tampered, expected_result_count=3)

    def test_inventory_section_swap_rejects_fully_resealed_buckets(self) -> None:
        inventory, records, _, _ = inventory_and_buckets(4)
        inventory_sha256 = V6.result_inventory_digest(inventory)
        swapped = copy.deepcopy(records)
        main = next(row for row in swapped if row["paper_section_id"] == "main")
        appendix = next(
            row for row in swapped if row["paper_section_id"] == "appendix"
        )
        main["paper_section_id"] = "appendix"
        appendix["paper_section_id"] = "main"
        resealed = [
            V6.make_bucket_subseal(
                inventory_sha256,
                section,
                [row for row in swapped if row["paper_section_id"] == section],
            )
            for section in ("appendix", "main")
        ]

        with self.assertRaisesRegex(
            V6.V6ContractError, "inventory result section mismatch"
        ):
            V6.reconcile_cp3_buckets(
                inventory, resealed, expected_result_count=4
            )

    def test_expected_result_count_rejects_booleans(self) -> None:
        inventory, _, buckets, _ = inventory_and_buckets()
        for hostile_count in (False, True):
            with self.subTest(hostile_count=hostile_count):
                with self.assertRaisesRegex(
                    V6.V6ContractError, "positive integer"
                ):
                    V6.reconcile_cp3_buckets(
                        inventory,
                        buckets,
                        expected_result_count=hostile_count,
                    )


class ExactRationalRoundingTests(unittest.TestCase):
    def test_exact_integer_remainder_round_half_up(self) -> None:
        cases = (
            (1, 3, 2, "0.33"),
            (2, 3, 2, "0.67"),
            (1, 8, 2, "0.13"),
            (-1, 8, 2, "-0.13"),
            (-1, 1000, 2, "-0.00"),
        )
        for numerator, denominator, places, expected in cases:
            with self.subTest(
                numerator=numerator, denominator=denominator, places=places
            ):
                proof = V6.make_provenance_rounding_proof(
                    "recomputation:exact",
                    numerator,
                    denominator,
                    places,
                )
                self.assertEqual(proof["unrounded_numerator"], numerator)
                self.assertEqual(proof["unrounded_denominator"], denominator)
                self.assertEqual(proof["rounded_decimal"], expected)
                self.assertEqual(proof["display_value"], expected)

    def test_malformed_or_unreduced_rationals_are_rejected(self) -> None:
        hostile = (
            (True, 3, 2),
            (1, True, 2),
            (1, 0, 2),
            (1, -3, 2),
            (2, 6, 2),
            (0, 2, 2),
            (1, 3, True),
        )
        for numerator, denominator, places in hostile:
            with self.subTest(
                numerator=numerator, denominator=denominator, places=places
            ):
                with self.assertRaises(V6.V6ContractError):
                    V6.make_provenance_rounding_proof(
                        "recomputation:exact", numerator, denominator, places
                    )


class CopyAndFinalTests(unittest.TestCase):
    def test_source_pre_plan_and_copy_ledger_pass(self) -> None:
        fixture = copy_fixture()
        cp3, plan, ledger, payloads = fixture[3], fixture[6], fixture[7], fixture[8]
        summary = V6.validate_copy_ledger(
            cp3, plan, ledger, payloads, [fixture[4]], expected_result_count=3
        )
        self.assertEqual(summary["copy_state"], "COPIED_BYTES_VERIFIED")
        self.assertEqual(summary["raw_artifact_count"], 1)

    def test_copy_ledger_rejects_undeclared_fields(self) -> None:
        fixture = copy_fixture()
        cp3, plan, ledger, payloads = fixture[3], fixture[6], fixture[7], fixture[8]
        hostile = copy.deepcopy(ledger)
        hostile[0]["forged_but_resealed"] = True
        with self.assertRaisesRegex(V6.V6ContractError, "row keys differ"):
            V6.validate_copy_ledger(
                cp3,
                plan,
                hostile,
                payloads,
                [fixture[4]],
                expected_result_count=3,
            )

    def test_plan_requires_complete_typed_cp3_envelope(self) -> None:
        fixture = copy_fixture()
        cp3, source, raw_node = fixture[3], fixture[4], fixture[5]
        mutations = (
            ("schema", "experiments7-cp3-admission-reconciliation/v5"),
            ("phase", "final"),
            ("total_result_count", 2),
            ("paper_section_bucket_count", 99),
            ("inventory_sha256", "not-a-sha256"),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                hostile = copy.deepcopy(cp3)
                hostile[field] = value
                with self.assertRaises(V6.V6ContractError):
                    V6.plan_source_pre_copies(
                        hostile, [raw_node], [source], expected_result_count=3
                    )

    def test_copy_ledger_requires_plan_raw_list_and_cp3_digest_binding(self) -> None:
        fixture = copy_fixture()
        cp3, plan, ledger, payloads = fixture[3], fixture[6], fixture[7], fixture[8]
        raw_list_drift = copy.deepcopy(plan)
        raw_list_drift["raw_artifact_ids"] = []
        with self.assertRaisesRegex(V6.V6ContractError, "raw list"):
            V6.validate_copy_ledger(
                cp3,
                raw_list_drift,
                ledger,
                payloads,
                [fixture[4]],
                expected_result_count=3,
            )

        digest_drift = copy.deepcopy(plan)
        digest_drift["copy_plan_input_sha256"] = "0" * 64
        with self.assertRaisesRegex(V6.V6ContractError, "raw-union digest"):
            V6.validate_copy_ledger(
                cp3,
                digest_drift,
                ledger,
                payloads,
                [fixture[4]],
                expected_result_count=3,
            )

    def test_copy_ledger_recomputes_plan_entry_security_identities(self) -> None:
        fixture = copy_fixture()
        cp3, plan, ledger, payloads = fixture[3], fixture[6], fixture[7], fixture[8]
        mutations = (
            ("source_root_id", "experiments7"),
            ("destination_relative_path", "raw/verified/experiments4/forged.json"),
            ("copied_artifact_id", "copied:" + "0" * 64),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                hostile = copy.deepcopy(plan)
                hostile["entries"][0][field] = value
                hostile["entries_sha256"] = V6.digest_records(hostile["entries"])
                with self.assertRaises(V6.V6ContractError):
                    V6.validate_copy_ledger(
                        cp3,
                        hostile,
                        ledger,
                        payloads,
                        [fixture[4]],
                        expected_result_count=3,
                    )

    def test_copy_ledger_rejects_rehashed_forged_source_pre_bindings(self) -> None:
        fixture = copy_fixture()
        cp3, source, plan = fixture[3], fixture[4], fixture[6]
        ledger, payloads = fixture[7], fixture[8]
        mutations = (
            {
                "source_manifest_record_id": "source:forged",
                "source_pre_record_sha256": "0" * 64,
            },
            {"source_pre_record_sha256": "f" * 64},
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                hostile_plan = copy.deepcopy(plan)
                hostile_plan["entries"][0].update(mutation)
                hostile_plan["entries_sha256"] = V6.digest_records(
                    hostile_plan["entries"]
                )
                hostile_ledger = copy.deepcopy(ledger)
                hostile_ledger[0].update(mutation)
                hostile_ledger[0]["copy_plan_entry_sha256"] = V6.digest_json(
                    hostile_plan["entries"][0]
                )
                with self.assertRaises(V6.V6ContractError):
                    V6.validate_copy_ledger(
                        cp3,
                        hostile_plan,
                        hostile_ledger,
                        payloads,
                        [source],
                        expected_result_count=3,
                    )

    def test_plan_rejects_legacy_missing_extra_and_drift_raw(self) -> None:
        fixture = copy_fixture()
        cp3, source, raw_node = fixture[3], fixture[4], fixture[5]
        legacy_source = copy.deepcopy(source)
        legacy_source["root_id"] = "experiments7"
        legacy_node = copy.deepcopy(raw_node)
        legacy_node["root_id"] = "experiments7"
        with self.assertRaisesRegex(V6.V6ContractError, "legacy"):
            V6.plan_source_pre_copies(
                cp3, [legacy_node], [legacy_source], expected_result_count=3
            )
        with self.assertRaisesRegex(V6.V6ContractError, "absent|must not be empty"):
            V6.plan_source_pre_copies(cp3, [raw_node], [], expected_result_count=3)
        with self.assertRaisesRegex(
            V6.V6ContractError, "raw provenance node is not content-addressed exactly"
        ):
            V6.plan_source_pre_copies(
                cp3, [raw_node, {**raw_node, "node_id": "raw:extra"}], [source],
                expected_result_count=3,
            )
        drift_cp3 = copy.deepcopy(cp3)
        drift_cp3["copy_required_raw_artifact_ids"] = [V6.DRIFT_ONLY_RAW_ID]
        drift_cp3["copy_plan_input_sha256"] = V6.digest_json(
            drift_cp3["copy_required_raw_artifact_ids"]
        )
        drift_node = {**raw_node, "node_id": V6.DRIFT_ONLY_RAW_ID}
        with self.assertRaises(V6.V6ContractError):
            V6.plan_source_pre_copies(
                drift_cp3, [drift_node], [source], expected_result_count=3
            )

    def test_plan_requires_explicit_false_raw_exclusion_flags(self) -> None:
        fixture = copy_fixture()
        cp3, source, raw_node = fixture[3], fixture[4], fixture[5]
        for field in ("declared_drift", "unrelated", "support_only", "derived_only"):
            for mutation in ("missing", "false"):
                with self.subTest(field=field, mutation=mutation):
                    hostile = copy.deepcopy(raw_node)
                    if mutation == "missing":
                        del hostile[field]
                    else:
                        hostile[field] = "false"
                    with self.assertRaises(V6.V6ContractError):
                        V6.plan_source_pre_copies(
                            cp3, [hostile], [source], expected_result_count=3
                        )

    def test_plan_rejects_every_one_field_raw_identity_substitution(self) -> None:
        fixture = copy_fixture()
        cp3, source, raw_node = fixture[3], fixture[4], fixture[5]
        substitutions = {
            "source_manifest_record_id": "source:substituted",
            "source_pre_record_sha256": "0" * 64,
            "root_id": "experiments5",
            "relative_path": "synthetic/substituted.json",
            "sha256": "f" * 64,
            "size": int(raw_node["size"]) + 1,
            "descriptor_identity": {
                **raw_node["descriptor_identity"],
                "st_ino": int(raw_node["descriptor_identity"]["st_ino"]) + 1,
            },
            "raw_kind": "snapshot",
            "copy_required": False,
            "declared_drift": True,
            "unrelated": True,
            "support_only": True,
            "derived_only": True,
            "node_type": "config",
        }
        for field, value in substitutions.items():
            with self.subTest(field=field):
                hostile = copy.deepcopy(raw_node)
                hostile[field] = value
                with self.assertRaises(V6.V6ContractError):
                    V6.plan_source_pre_copies(
                        cp3, [hostile], [source], expected_result_count=3
                    )

        snapshot = V6.make_provenance_node(
            "config", "snapshots/config.json", "a" * 64
        )
        snapshot["node_type"] = "raw_artifact"
        with self.assertRaises(V6.V6ContractError):
            V6.plan_source_pre_copies(
                cp3, [snapshot], [source], expected_result_count=3
            )

        extra = {**raw_node, "post_hoc": "not-cp3-bytes"}
        with self.assertRaisesRegex(V6.V6ContractError, "schema keys"):
            V6.plan_source_pre_copies(
                cp3, [extra], [source], expected_result_count=3
            )

    def test_hostile_ledger_missing_extra_duplicate_drift_and_mutation(self) -> None:
        fixture = copy_fixture()
        cp3, plan, ledger, payloads = fixture[3], fixture[6], fixture[7], fixture[8]
        cases = []
        missing_edge = copy.deepcopy(ledger)
        missing_edge[0]["copied_to_edge_id"] = ""
        cases.append(missing_edge)
        source_drift = copy.deepcopy(ledger)
        source_drift[0]["source_after"]["mtime_ns"] += 1
        cases.append(source_drift)
        hardlink = copy.deepcopy(ledger)
        hardlink[0]["destination_identity"]["st_dev"] = hardlink[0]["source_before"]["st_dev"]
        hardlink[0]["destination_identity"]["st_ino"] = hardlink[0]["source_before"]["st_ino"]
        cases.append(hardlink)
        clone = copy.deepcopy(ledger)
        clone[0]["reflink_used"] = True
        cases.append(clone)
        for hostile in cases:
            with self.subTest(hostile=hostile):
                with self.assertRaises(V6.V6ContractError):
                    V6.validate_copy_ledger(
                        cp3,
                        plan,
                        hostile,
                        payloads,
                        [fixture[4]],
                        expected_result_count=3,
                    )
        with self.assertRaisesRegex(V6.V6ContractError, "missing, extra, or orphan"):
            V6.validate_copy_ledger(
                cp3, plan, [], payloads, [fixture[4]], expected_result_count=3
            )
        with self.assertRaises(V6.V6ContractError):
            V6.validate_copy_ledger(
                cp3,
                plan,
                ledger + ledger,
                payloads,
                [fixture[4]],
                expected_result_count=3,
            )
        copied_id = next(iter(payloads))
        with self.assertRaisesRegex(V6.V6ContractError, "mutated"):
            V6.validate_copy_ledger(
                cp3,
                plan,
                ledger,
                {copied_id: b"tampered"},
                [fixture[4]],
                expected_result_count=3,
            )
        with self.assertRaisesRegex(V6.V6ContractError, "extra or missing"):
            V6.validate_copy_ledger(
                cp3,
                plan,
                ledger,
                {**payloads, "copied:orphan": b"orphan"},
                [fixture[4]],
                expected_result_count=3,
            )

    def test_final_verified_requires_copied_bytes_and_copied_to_closure(self) -> None:
        fixture = copy_fixture()
        inventory, buckets, cp3 = fixture[0], fixture[2], fixture[3]
        plan, ledger, payloads = fixture[6], fixture[7], fixture[8]
        final = final_records(buckets, plan)
        summary = V6.reconcile_final_admission(
            inventory, buckets, cp3, final, plan, ledger, payloads, [fixture[4]],
            expected_result_count=3,
        )
        self.assertEqual(summary["final_state"], "VERIFIED")
        self.assertEqual(summary["final_verified_count"], 3)
        self.assertEqual(summary["final_unresolved_count"], 0)

        missing_edge = copy.deepcopy(final)
        raw_final = next(row for row in missing_edge if row["raw_artifact_ids"])
        raw_final["copied_to_edge_ids"] = []
        with self.assertRaises(V6.V6ContractError):
            V6.reconcile_final_admission(
                inventory, buckets, cp3, missing_edge, plan, ledger, payloads,
                [fixture[4]],
                expected_result_count=3,
            )
        copied_id = next(iter(payloads))
        with self.assertRaises(V6.V6ContractError):
            V6.reconcile_final_admission(
                inventory, buckets, cp3, final, plan, ledger,
                {copied_id: b"mutated"}, [fixture[4]], expected_result_count=3,
            )
        mutated_source_pre = copy.deepcopy(fixture[4])
        mutated_source_pre["sealed_extra_field"] = "mutated"
        with self.assertRaisesRegex(
            V6.V6ContractError, "source-pre record schema keys|exact-record mismatch"
        ):
            V6.reconcile_final_admission(
                inventory,
                buckets,
                cp3,
                final,
                plan,
                ledger,
                payloads,
                [mutated_source_pre],
                expected_result_count=3,
            )

    def test_final_rejects_pre_copy_provenance_mutation(self) -> None:
        fixture = copy_fixture()
        inventory, buckets, cp3 = fixture[0], fixture[2], fixture[3]
        plan, ledger, payloads = fixture[6], fixture[7], fixture[8]
        final = final_records(buckets, plan)
        final[0]["producer_node_ids"] = ["producer:replacement"]
        contributors = set(final[0]["raw_artifact_ids"])
        contributors.update(final[0]["closed_parent_result_ids"])
        for field in V6.CLOSURE_FIELDS:
            contributors.update(final[0][field])
        final[0]["required_contributor_node_ids"] = sorted(contributors)
        final[0]["sealed_contributor_node_ids"] = sorted(contributors)
        with self.assertRaises(V6.V6ContractError):
            V6.reconcile_final_admission(
                inventory, buckets, cp3, final, plan, ledger, payloads,
                [fixture[4]],
                expected_result_count=3,
            )


class ImmutableAuditTests(unittest.TestCase):
    def test_exact_pinned_audits_are_negative_only(self) -> None:
        oracle = V6.validate_immutable_audit_oracles(ROOT)
        self.assertFalse(oracle["positive_evidence_allowed"])
        self.assertEqual(oracle["unresolved_result_id_count"], 238)
        self.assertEqual(oracle["table10_incomplete_parent_result_id_count"], 34)
        self.assertEqual(
            set(oracle["table10_incomplete_parent_result_ids"]),
            V6.TABLE10_INCOMPLETE_PARENT_RESULT_IDS,
        )
        self.assertEqual(
            set(oracle["figure5_zero_raw_result_ids"]),
            V6.FIGURE5_ZERO_RAW_RESULT_IDS,
        )
        self.assertEqual(oracle["drift_only_raw_id"], V6.DRIFT_ONLY_RAW_ID)
        self.assertEqual(oracle["drift_only_raw_sha256"], V6.DRIFT_ONLY_RAW_SHA256)
        for result_id in oracle["unresolved_result_ids"]:
            self.assertEqual(
                V6.derive_admission_state(unresolved_record(result_id), "precopy"),
                "UNRESOLVED",
            )

    def test_table10_and_figure5_are_not_auto_promoted(self) -> None:
        for result_id in V6.TABLE10_INCOMPLETE_PARENT_RESULT_IDS:
            row = unresolved_record(result_id)
            row["required_parent_result_ids"] = ["result:base", "result:prefine"]
            row["closed_parent_result_ids"] = ["result:base"]
            self.assertEqual(V6.derive_admission_state(row, "precopy"), "UNRESOLVED")
        for number, result_id in enumerate(sorted(V6.FIGURE5_ZERO_RAW_RESULT_IDS)):
            audit_only = unresolved_record(result_id)
            self.assertEqual(
                V6.derive_admission_state(audit_only, "precopy"), "UNRESOLVED"
            )
            closed = positive_record(number, result_id=result_id)
            self.assertEqual(
                V6.validate_admission_record(closed, "precopy")["state"],
                "ADMITTED_FOR_COPY",
            )

    def test_audit_mutation_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp_root = Path(temporary)
            for relative in (
                V6.AUDIT_UNRESOLVED_RELATIVE,
                V6.AUDIT_DEFECTS_RELATIVE,
            ):
                destination = temp_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            unresolved = temp_root / V6.AUDIT_UNRESOLVED_RELATIVE
            unresolved.write_bytes(unresolved.read_bytes() + b"\n")
            with self.assertRaisesRegex(V6.V6ContractError, "hash mismatch"):
                V6.validate_immutable_audit_oracles(temp_root)


if __name__ == "__main__":
    unittest.main()
