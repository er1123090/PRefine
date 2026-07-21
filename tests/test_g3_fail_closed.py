from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/provenance/build_g3_admission_graph.py"
SPEC = importlib.util.spec_from_file_location("build_g3_admission_graph", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def admission(status: str, raw_ids: tuple[str, ...] = ()) -> dict[str, object]:
    return {
        "status": status,
        "admission_scope": "raw_file" if raw_ids else "none",
        "reason_codes": [],
        "aggregate_bindings": [],
        "raw_bindings": [{"raw_node_id": raw_id} for raw_id in raw_ids],
        "supporting_evidence": [],
    }


class G3FailClosedTests(unittest.TestCase):
    def test_precopy_gate_blocks_any_unresolved_result(self) -> None:
        state_counts = {
            "ADMITTED_FOR_COPY": 1907,
            "DERIVED_FROM_ADMITTED_RAW": 0,
            "NON_INFERENCE": 0,
            "NOT_APPLICABLE": 0,
            "UNRESOLVED": 1,
        }
        self.assertEqual(MODULE.precopy_gate(state_counts, 1908), ("BLOCKED", False))

    def test_precopy_gate_requires_exact_inventory_cardinality(self) -> None:
        state_counts = {
            "ADMITTED_FOR_COPY": 1907,
            "DERIVED_FROM_ADMITTED_RAW": 0,
            "NON_INFERENCE": 0,
            "NOT_APPLICABLE": 0,
            "UNRESOLVED": 0,
        }
        self.assertEqual(MODULE.precopy_gate(state_counts, 1907), ("BLOCKED", False))

    def test_precopy_gate_passes_only_exact_three_state_v6_admission(self) -> None:
        self.assertEqual(
            MODULE.precopy_gate(
                {"ADMITTED_FOR_COPY": 1908, "UNRESOLVED": 0},
                1908,
            ),
            ("PASS", True),
        )
        self.assertEqual(
            MODULE.precopy_gate(
                {
                    "ADMITTED_FOR_COPY": 1907,
                    "DERIVED_FROM_ADMITTED_RAW": 1,
                    "UNRESOLVED": 0,
                },
                1908,
            ),
            ("BLOCKED", False),
        )

    def test_precopy_gate_rejects_invalid_or_negative_counts(self) -> None:
        self.assertEqual(
            MODULE.precopy_gate(
                {"ADMITTED_FOR_COPY": 1909, "UNRESOLVED": -1},
                1908,
            ),
            ("BLOCKED", False),
        )
        self.assertEqual(
            MODULE.precopy_gate(
                {"ADMITTED_FOR_COPY": True, "UNRESOLVED": 1907},
                1908,
            ),
            ("BLOCKED", False),
        )

    def test_table10_requires_both_parents_admitted_with_raw(self) -> None:
        base_id = "base"
        pref_id = "pref"
        delta_id = "delta"
        results = [
            {"result_id": base_id, "evidence_id": "table:3", "method": "Base Prompting", "model": "M", "setting": "context-free", "preference_type": "Recall", "metric": "F1", "numeric_value": "10.00"},
            {"result_id": pref_id, "evidence_id": "table:3", "method": "PREFINE", "model": "M", "setting": "context-free", "preference_type": "Recall", "metric": "F1", "numeric_value": "20.00"},
            {"result_id": delta_id, "evidence_id": "table:10", "method": "PREFINE minus Base Prompting", "model": "M", "setting": "context-free", "preference_type": "Recall", "metric": "F1", "numeric_value": "10.00"},
        ]
        admissions = {
            base_id: admission("unresolved"),
            pref_id: admission("admitted_exact_raw", ("raw:pref",)),
            delta_id: admission("unresolved"),
        }
        MODULE.bind_table10(results, admissions)
        self.assertEqual(admissions[delta_id]["status"], "unresolved_derived")
        self.assertEqual(
            admissions[delta_id]["reason_codes"],
            ["table3_delta_parents_not_both_admitted_with_raw"],
        )
        self.assertEqual(admissions[delta_id]["raw_bindings"], [])

    def test_figure4_derived_gate_allows_ground_truth_but_unions_inference_raw_only(self) -> None:
        parent_results = [
            {"evidence_id": "figure:4", "method": "Ground Truth"},
            {"evidence_id": "figure:4", "method": "Base Prompting"},
            {"evidence_id": "figure:4", "method": "PREFINE"},
        ]
        parent_admissions = [
            admission("aggregate_only_dataset_reference", ("raw:ground-truth-candidate",)),
            admission("admitted_derived_exact_raw_set", ("raw:base",)),
            admission("admitted_semantic_raw_set_unlabeled_graphical_mark", ("raw:pref",)),
        ]
        valid, raw_bindings = MODULE.figure4_derived_parent_raw_union(
            parent_results, parent_admissions
        )
        self.assertTrue(valid)
        self.assertEqual(
            [binding["raw_node_id"] for binding in raw_bindings],
            ["raw:base", "raw:pref"],
        )

        parent_admissions[1] = admission("admitted_derived_exact_raw_set")
        valid, raw_bindings = MODULE.figure4_derived_parent_raw_union(
            parent_results, parent_admissions
        )
        self.assertFalse(valid)
        self.assertEqual(raw_bindings, [])

        parent_admissions[0] = admission("non_inference_dataset_statistic")
        parent_admissions[1] = admission("admitted_derived_exact_raw_set", ("raw:base",))
        valid, raw_bindings = MODULE.figure4_derived_parent_raw_union(
            parent_results, parent_admissions
        )
        self.assertFalse(valid)
        self.assertEqual(raw_bindings, [])

    def test_unresolved_serializer_separates_candidates_and_preserves_whitelisted_diagnostics(self) -> None:
        unresolved = admission("unresolved", ("raw:candidate-b", "raw:candidate-a"))
        expected = {
            "candidate_count": 4,
            "distinct_bound_raw_count": 2,
            "unbound_candidates": [{"missing_path": "candidate.json"}],
            "bound_candidate_count": 2,
            "context_candidate_counts": {"gemma": 1, "gpt4o": 0},
            "exact_combination_count": 0,
            "tolerance_combination_count": 3,
            "fully_bound_qualifying_combination_count": 1,
            "unbound_qualifying_combination_count": 2,
            "best_max_delta_percentage_points": "0.02",
            "aggregate_value": "10.10",
            "producer_value": "10.09",
            "paper_value": "10.00",
            "delta_percentage_points": "0.09",
            "parent_count": 4,
        }
        unresolved.update(expected)
        diagnostics = MODULE.unresolved_contract_diagnostics(
            unresolved,
            [binding["raw_node_id"] for binding in unresolved["raw_bindings"]],
        )
        row = {"raw_artifact_ids": []}
        row.update(diagnostics)

        self.assertEqual(row["raw_artifact_ids"], [])
        self.assertEqual(
            row["candidate_raw_artifact_ids"],
            ["raw:candidate-a", "raw:candidate-b"],
        )
        for key, value in expected.items():
            self.assertEqual(row[key], value, key)
        self.assertNotIn("raw_bindings", diagnostics)
        self.assertNotIn("protected_path", diagnostics)


if __name__ == "__main__":
    unittest.main()
