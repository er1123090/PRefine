from __future__ import annotations

import copy
import unittest
from pathlib import Path

from ecpr.contracts import CandidatePolicy
from ecpr.integrity import PQR_ROUTING_COVERAGE, build_routing_coverage
from ecpr.io import load_json
from ecpr.prompts import build_action_prompt, candidate_memory_block
from ecpr.query_gate import GateDecision, gate_public_query, validate_ontology
from ecpr.router import route_hypotheses


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = load_json(ROOT / "configs/public_domain_ontology.json")
SINGLE_SCHEMA = load_json(ROOT / "configs/schema_single.json")
MULTI_SCHEMA = load_json(ROOT / "configs/schema_multi.json")
SCHEMA_UNION = SINGLE_SCHEMA + MULTI_SCHEMA


def function_schema(name: str, slot: str = "preference") -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "parameters": {
                "type": "object",
                "properties": {slot: {"type": "string"}},
            },
        },
    }


def task(query: str, mode: str = "singleturn") -> dict:
    return {
        "case_key": "a" * 64,
        "example_id": "opaque",
        "mode": mode,
        "query": query,
        "schema_key": "single" if mode == "singleturn" else "multi",
    }


class PublicQueryGateTests(unittest.TestCase):
    def test_selects_unique_strong_domain_and_uses_max_not_sum(self):
        selected = gate_public_query(
            "Please find a HOTEL near the station.",
            "singleturn",
            SINGLE_SCHEMA,
            ONTOLOGY,
        )
        self.assertEqual(
            (selected.status, selected.canonical_domain, selected.schema_domain),
            ("SELECT", "Hotels", "GetHotels"),
        )
        generic = gate_public_query(
            "Compare checking account and savings account options.",
            "singleturn",
            SINGLE_SCHEMA,
            ONTOLOGY,
        )
        self.assertEqual((generic.status, generic.top_score), ("SELECT", 3))

    def test_token_boundaries_prevent_substring_matches(self):
        decision = gate_public_query(
            "The bankruptcy proceeding is long.",
            "singleturn",
            SINGLE_SCHEMA,
            ONTOLOGY,
        )
        self.assertEqual(decision.status, "ABSTAIN")

    def test_multiturn_uses_only_user_payload_and_is_assistant_invariant(self):
        first = gate_public_query(
            "User: Find a hotel.\nAssistant: Flights and weather are available.",
            "multiturn",
            MULTI_SCHEMA,
            ONTOLOGY,
        )
        second = gate_public_query(
            "User: Find a hotel.\nAssistant: Music, movies, and buses.",
            "multiturn",
            MULTI_SCHEMA,
            ONTOLOGY,
        )
        self.assertEqual(first, second)
        self.assertEqual(first.schema_domain, "GetHotels")

    def test_tagged_missing_user_malformed_and_unknown_mode_abstain(self):
        for query, mode in (
            ("Assistant: Find flights.\nTool: GetFlights", "multiturn"),
            ("User:\nAssistant: Find hotels.", "multiturn"),
            ("Find a hotel.", "unknown"),
        ):
            with self.subTest(query=query, mode=mode):
                decision = gate_public_query(query, mode, MULTI_SCHEMA, ONTOLOGY)
                self.assertEqual(decision.status, "ABSTAIN")

    def test_malformed_role_prefix_and_untagged_preamble_abstain(self):
        for query in (
            "User: hello\nAssistant - Find a flight.",
            "Narrator preamble\nUser: Find a flight.",
        ):
            with self.subTest(query=query):
                decision = gate_public_query(
                    query, "multiturn", MULTI_SCHEMA, ONTOLOGY
                )
                self.assertEqual(decision.status, "ABSTAIN")

    def test_frozen_routing_coverage_recomputes_public_decisions_not_accuracy(self):
        coverage = load_json(ROOT / PQR_ROUTING_COVERAGE)
        self.assertEqual(coverage, build_routing_coverage(ROOT))
        self.assertEqual(coverage["kind"], "routing_coverage_not_accuracy")
        self.assertEqual(
            coverage["authority"],
            "post_lock_determinism_snapshot_only_cannot_change_ontology_or_thresholds",
        )
        self.assertEqual(
            coverage["counts"],
            {
                "single": {"select": 1097, "abstain": 0, "total": 1097},
                "multi": {"select": 320, "abstain": 777, "total": 1097},
                "total": {
                    "select": 1417,
                    "abstain": 777,
                    "total": 2194,
                },
            },
        )

    def test_tie_vague_and_multi_intent_abstain(self):
        for query in (
            "Find a hotel and a flight.",
            "I need a meal.",
            "Show a movie and play music.",
        ):
            with self.subTest(query=query):
                decision = gate_public_query(
                    query, "singleturn", SINGLE_SCHEMA, ONTOLOGY
                )
                self.assertEqual(decision.status, "ABSTAIN")

    def test_schema_permutation_and_single_schema_asymmetry(self):
        query = "Find a restaurant for dinner."
        forward = gate_public_query(query, "singleturn", SINGLE_SCHEMA, ONTOLOGY)
        reverse = gate_public_query(
            query, "singleturn", list(reversed(SINGLE_SCHEMA)), ONTOLOGY
        )
        self.assertEqual(forward, reverse)
        weather = gate_public_query(
            "What is the weather forecast?", "singleturn", SINGLE_SCHEMA, ONTOLOGY
        )
        self.assertEqual(weather.status, "ABSTAIN")

    def test_schema_union_rejects_unknown_ontology_mapping(self):
        mutated = copy.deepcopy(ONTOLOGY)
        mutated["domains"][-1]["schema_domain"] = "GetUnknown"
        with self.assertRaisesRegex(ValueError, "frozen schema"):
            validate_ontology(mutated, SCHEMA_UNION)


class RoutedCandidateTests(unittest.TestCase):
    def setUp(self):
        self.schema = [
            function_schema("GetHotels", "preference"),
            function_schema("GetFlights", "preference"),
        ]
        self.task = task("Find a hotel near downtown.")
        self.memory = {
            "latent_abstraction": {"implicit_pref": "TOXIC_LATENT_SENTINEL"},
            "typed_hypotheses": [
                {
                    "domain": domain,
                    "slot": "preference",
                    "value": value,
                    "support": 3,
                    "counterevidence": 0,
                    "confidence": 0.75,
                    "last_seen": 2,
                    "provenance": [],
                }
                for domain, value in (
                    ("GetHotels", "quiet"),
                    ("GetFlights", "aisle"),
                )
            ],
        }
        self.preference_slots = {
            "GetHotels": ["preference"],
            "GetFlights": ["preference"],
        }

    def test_router_keeps_only_selected_domain(self):
        routed = route_hypotheses(
            self.memory,
            self.task,
            self.schema,
            self.preference_slots,
            CandidatePolicy(),
            ONTOLOGY,
        )
        self.assertEqual(
            [(item["domain"], item["value"]) for item in routed],
            [("GetHotels", "quiet")],
        )

    def test_forged_decision_and_hidden_task_field_fail_closed(self):
        forged = gate_public_query(
            "Find a flight.", "singleturn", self.schema, ONTOLOGY
        )
        with self.assertRaisesRegex(ValueError, "decision mismatch"):
            route_hypotheses(
                self.memory,
                self.task,
                self.schema,
                self.preference_slots,
                CandidatePolicy(),
                ONTOLOGY,
                decision=forged,
            )
        injected = {**self.task, "hidden_hint": "GetFlights"}
        with self.assertRaisesRegex(ValueError, "field contract"):
            route_hypotheses(
                self.memory,
                injected,
                self.schema,
                self.preference_slots,
                CandidatePolicy(),
                ONTOLOGY,
            )

    def test_abstain_or_zero_typed_omits_toxic_latent(self):
        abstain = candidate_memory_block(
            self.memory["typed_hypotheses"], 1536
        )
        selected_empty = candidate_memory_block(
            [], 1536, selected_domain="GetHotels"
        )
        for block in (abstain, selected_empty):
            self.assertNotIn("TOXIC_LATENT_SENTINEL", block)
        self.assertIn("ABSTAIN", abstain)
        self.assertIn("No typed evidence survived", selected_empty)

    def test_prompt_states_scope_without_implying_request(self):
        routed = route_hypotheses(
            self.memory,
            self.task,
            self.schema,
            self.preference_slots,
            CandidatePolicy(),
            ONTOLOGY,
        )
        block = candidate_memory_block(
            routed,
            1536,
            selected_domain="GetHotels",
        )
        prompt = build_action_prompt(
            self.task,
            self.schema,
            block,
        )
        self.assertIn("otherwise-missing preference", prompt)
        self.assertIn("does not imply that the user requested", prompt)
        self.assertNotIn("TOXIC_LATENT_SENTINEL", prompt)


if __name__ == "__main__":
    unittest.main()
