from __future__ import annotations

import copy
import inspect
import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ecpr.action_case import build_action_case
from ecpr.contracts import CandidatePolicy
from ecpr.io import canonical_json, sha256_file, sha256_text
from ecpr.latent_firewall import (
    ACTIVE_CANDIDATE_REVISION,
    CANDIDATE_REVISION,
    VLT3_CANDIDATE_REVISION,
    VLT3_LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256,
    _schema_function,
    _unsafe_or_trait,
    _vlt3_mapped_value,
    build_query_constraint_mask,
    canonical_vlt_prompt_tuple,
    decide_latent_transfer,
    qualified_trait_diagnostics,
    validate_latent_trait_ontology_v3,
    validate_vlt_schema_contract,
)
from ecpr.memory import build_typed_hypotheses
from ecpr.parsing import normalize_value
from ecpr.prompts import build_action_prompt, candidate_memory_block


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = ROOT / "configs/latent_trait_ontology.vlt3.json"


def load(path: Path):
    return json.loads(path.read_text())


def scalar_type(value):
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if type(value) is str:
        return "string"
    raise TypeError("synthetic typed evidence must be a JSON scalar")


def hypothesis(domain, slot, raw_value, *, support=2, counterevidence=0):
    value_type = scalar_type(raw_value)
    total = support + counterevidence
    return {
        "domain": domain,
        "slot": slot,
        "value": normalize_value(raw_value),
        "value_type": value_type,
        "source_literal": {"type": value_type, "value": raw_value},
        "support": support,
        "counterevidence": counterevidence,
        "confidence": support / total if total else 0.0,
        "last_seen": 0,
        "provenance": [],
    }


class AppendOnlyVLT3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ontology = load(ONTOLOGY_PATH)
        cls.preference_slots = load(ROOT / "configs/preference_slots.json")
        cls.schemas = {
            "single": load(ROOT / "configs/schema_single.json"),
            "multi": load(ROOT / "configs/schema_multi.json"),
        }
        cls.public_ontology = load(ROOT / "configs/public_domain_ontology.json")
        cls.policy = CandidatePolicy()

    def mask(self, query, domain, schema_key="single", mode="singleturn"):
        return build_query_constraint_mask(
            current_query=query,
            query_mode=mode,
            selected_domain=domain,
            schema_key=schema_key,
            schema=self.schemas[schema_key],
            ontology=self.ontology,
        )

    def decision(
        self,
        latent_text,
        selected_domain,
        typed,
        *,
        query,
        schema_key="single",
        mode="singleturn",
        query_mask=None,
    ):
        query_mask = query_mask or self.mask(
            query, selected_domain, schema_key=schema_key, mode=mode
        )
        with patch(
            "ecpr.latent_firewall._attestation_authorizes",
            return_value=(True, "AUTHORIZED"),
        ):
            return decide_latent_transfer(
                latent_abstraction={"implicit_pref": latent_text},
                latent_attestation={},
                pqr_status="SELECT",
                selected_domain=selected_domain,
                schema_key=schema_key,
                current_query=query,
                query_mode=mode,
                schema=self.schemas[schema_key],
                preference_slots=self.preference_slots,
                typed_hypotheses=typed,
                routed_hypotheses=[],
                policy=self.policy,
                ontology=self.ontology,
                query_constraint_mask=query_mask,
            )

    def test_append_only_parent_and_frozen_v1_v2_bytes(self):
        self.assertEqual(CANDIDATE_REVISION, "vlt2")
        self.assertEqual(ACTIVE_CANDIDATE_REVISION, "vlt3")
        self.assertEqual(VLT3_CANDIDATE_REVISION, "vlt3")
        self.assertEqual(
            self.ontology["parent_ontology_sha256"],
            "be2070439b5928d6f2b1468ea0b8b63c0a42f48b578b4755a6ef949880fe005b",
        )
        self.assertEqual(
            sha256_file(ROOT / "configs/latent_trait_ontology.json"),
            "1d9ffdf29a2af37367bbd63033c927473789e6eddb71926999bb88e56bb198a1",
        )
        self.assertEqual(
            sha256_file(ROOT / "configs/latent_trait_ontology.vlt2.json"),
            "be2070439b5928d6f2b1468ea0b8b63c0a42f48b578b4755a6ef949880fe005b",
        )
        self.assertEqual(
            sha256_file(ROOT / "tests/test_vlt2_audited_transfer.py"),
            "8795dd60faf16e1af4c16fa7c366d3397b2f139d96e21855899d48067355c5c9",
        )
        self.assertEqual(
            sha256_text(canonical_json(self.ontology)),
            VLT3_LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256,
        )
        validate_latent_trait_ontology_v3(self.ontology)

    def test_table5_correspondence_and_group_usage_is_excluded(self):
        correspondence = {
            item["internal_name"]: (item["paper_name"], item["status"])
            for item in self.ontology["paper_name_correspondence"]
        }
        self.assertEqual(correspondence["BUDGET"], ("Budget", "included"))
        self.assertEqual(correspondence["TRAVEL_PARTY"], ("Travel", "included"))
        self.assertEqual(correspondence["SOLO_USAGE"], ("solo", "included"))
        self.assertEqual(correspondence["GROUP_USAGE"], ("group", "excluded"))
        active = canonical_json(
            {
                "trait_order": self.ontology["trait_order"],
                "traits": self.ontology["traits"],
                "trait_groups": self.ontology["trait_groups"],
                "target_mappings": self.ontology["target_mappings"],
                "typed_support_registry": self.ontology["typed_support_registry"],
            }
        )
        self.assertNotIn("GROUP_USAGE", active)

        mutated = copy.deepcopy(self.ontology)
        mutated["trait_order"].append("GROUP_USAGE")
        mutated_hash = sha256_text(canonical_json(mutated))
        with patch(
            "ecpr.latent_firewall.VLT3_LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256",
            mutated_hash,
        ):
            with self.assertRaisesRegex(ValueError, "correspondence-only"):
                validate_latent_trait_ontology_v3(mutated)

    def test_mapping_identity_is_trait_domain_and_order_independent(self):
        mappings = self.ontology["target_mappings"]
        identities = [(item["trait_id"], item["domain"]) for item in mappings]
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(self.ontology["mapping_identity"], ["trait_id", "domain"])
        reversed_ontology = copy.deepcopy(self.ontology)
        reversed_ontology["target_mappings"] = list(reversed(mappings))
        function = _schema_function(self.schemas["single"], "GetRestaurants")
        self.assertEqual(
            _vlt3_mapped_value(
                reversed_ontology,
                "LOW_COST",
                "GetRestaurants",
                function,
                self.preference_slots,
                "single",
            ),
            ("price_range", "cheap", "AUTHORIZED"),
        )
        self.assertEqual(
            _vlt3_mapped_value(
                reversed_ontology,
                "HIGH_COST",
                "GetRestaurants",
                function,
                self.preference_slots,
                "single",
            ),
            ("price_range", "pricey", "AUTHORIZED"),
        )

    def test_query_registry_is_directly_bound_and_target_independent(self):
        mappings = {
            (item["trait_id"], item["domain"]): item
            for item in self.ontology["target_mappings"]
        }
        for constraint in self.ontology["query_constraint_policy"][
            "mapping_constraints"
        ]:
            identity = (constraint["trait_id"], constraint["domain"])
            mapping = mappings[identity]
            self.assertEqual(
                constraint["target_mapping_ref"],
                {"trait_id": identity[0], "domain": identity[1]},
            )
            self.assertEqual(constraint["slot"], mapping["slot"])
            self.assertEqual(constraint["schema_keys"], mapping["schema_keys"])
        provenance = self.ontology["query_constraint_policy"][
            "registry_component_provenance"
        ]
        self.assertEqual(
            set(provenance),
            {
                "slot_aliases",
                "explicit_value_phrases",
                "numeric_word_tokens",
                "count_alias_distance_tokens",
            },
        )
        for component in provenance.values():
            self.assertEqual(
                component["held_out_or_current_target_query_or_test_use"],
                "forbidden",
            )
            self.assertEqual(component["freeze"], "vlt3_pre_evaluation_append_only")
            self.assertTrue(component["authority"])
            self.assertTrue(component["source"])
            self.assertTrue(component["rationale"])
        self.assertEqual(
            provenance["count_alias_distance_tokens"]["frozen_value"], 3
        )

    def test_schema_contract_and_restaurant_solo_multi_only(self):
        validate_vlt_schema_contract(
            self.ontology, self.schemas, self.preference_slots
        )
        support = [hypothesis("GetFlights", "passengers", "1")]
        single = self.decision(
            "prefers solo trips",
            "GetRestaurants",
            support,
            query="Find a restaurant in Rome.",
            schema_key="single",
        )
        self.assertEqual(single.reason_code, "SCHEMA_MISMATCH")
        multi = self.decision(
            "prefers solo trips",
            "GetRestaurants",
            support,
            query="Find a restaurant in Rome.",
            schema_key="multi",
        )
        self.assertTrue(multi.authorized)
        self.assertEqual((multi.slot, multi.value), ("number_of_seats", "1"))

    def test_conservative_solo_phrases_and_fail_closed_text(self):
        for phrase in (
            "prefers to travel alone",
            "prefers traveling alone",
            "prefers travelling alone",
            "prefers solo trips",
        ):
            self.assertEqual(
                _unsafe_or_trait(phrase, self.ontology),
                ("SOLO_USAGE", "AUTHORIZED"),
            )
        self.assertEqual(_unsafe_or_trait("alone", self.ontology)[1], "NO_TRAIT_MATCH")
        self.assertEqual(
            _unsafe_or_trait("does not prefer solo trips", self.ontology)[1],
            "NEGATION_PRESENT",
        )
        self.assertEqual(
            _unsafe_or_trait("prefers solo trips and low cost", self.ontology)[1],
            "AMBIGUOUS_TRAIT",
        )

    def test_exact_source_wire_literals_reject_type_and_spelling_aliases(self):
        values = ["1", 1, 1.0, True, "01", "one"]
        history = {
            "sessions": [
                {
                    "session_index": 0,
                    "api_calls": [
                        {"name": "GetFlights", "arguments": {"passengers": value}}
                        for value in values
                    ],
                }
            ]
        }
        permissive = CandidatePolicy(
            minimum_support=1,
            minimum_confidence=0.0,
            maximum_conflict_ratio=1.0,
        )
        hypotheses = build_typed_hypotheses(
            history, self.preference_slots, permissive
        )
        diagnostics = qualified_trait_diagnostics(
            self.ontology, hypotheses, permissive
        )
        solo = [item for item in diagnostics if item["trait_id"] == "SOLO_USAGE"]
        self.assertEqual(len(solo), 1)
        self.assertEqual(
            solo[0]["hypothesis"]["source_literal"],
            {"type": "string", "value": "1"},
        )
        self.assertEqual(
            {
                (item["value_type"], canonical_json(item["source_literal"]))
                for item in hypotheses
            },
            {
                ("string", canonical_json({"type": "string", "value": "1"})),
                ("integer", canonical_json({"type": "integer", "value": 1})),
                ("number", canonical_json({"type": "number", "value": 1.0})),
                ("boolean", canonical_json({"type": "boolean", "value": True})),
                ("string", canonical_json({"type": "string", "value": "01"})),
                ("string", canonical_json({"type": "string", "value": "one"})),
            },
        )

    def test_boolean_schema_support_uses_exact_string_source_wire(self):
        history = {
            "sessions": [
                {
                    "session_index": 0,
                    "api_calls": [
                        {"name": "GetTravel", "arguments": {"free_entry": "True"}},
                        {"name": "GetTravel", "arguments": {"free_entry": True}},
                        {"name": "GetTravel", "arguments": {"free_entry": "true"}},
                    ],
                }
            ]
        }
        permissive = CandidatePolicy(
            minimum_support=1,
            minimum_confidence=0.0,
            maximum_conflict_ratio=1.0,
        )
        diagnostics = qualified_trait_diagnostics(
            self.ontology,
            build_typed_hypotheses(history, self.preference_slots, permissive),
            permissive,
        )
        low = [item for item in diagnostics if item["trait_id"] == "LOW_COST"]
        self.assertEqual(len(low), 1)
        self.assertEqual(
            low[0]["hypothesis"]["source_literal"],
            {"type": "string", "value": "True"},
        )

    def test_one_policy_qualified_non_target_hypothesis_is_uniform(self):
        cases = (
            (
                "low cost",
                "GetRestaurants",
                "Find a restaurant in Rome.",
                hypothesis("GetFlights", "flight_class", "Economy"),
                "LOW_COST",
            ),
            (
                "high cost",
                "GetRestaurants",
                "Find a restaurant in Rome.",
                hypothesis("GetRentalCars", "car_type", "Full-size"),
                "HIGH_COST",
            ),
            (
                "prefers solo trips",
                "GetBuses",
                "Find a bus from Rome to Milan.",
                hypothesis("GetFlights", "passengers", "1"),
                "SOLO_USAGE",
            ),
        )
        for text, domain, query, support, trait in cases:
            with self.subTest(trait=trait):
                result = self.decision(text, domain, [support], query=query)
                self.assertTrue(result.authorized)
                self.assertEqual(result.trait_id, trait)
                self.assertEqual(len(result.supporting_domains), 1)

    def test_group_local_opposition_and_cost_solo_neutrality(self):
        low_support = hypothesis("GetFlights", "flight_class", "Economy")
        high_support = hypothesis("GetRentalCars", "car_type", "Full-size")
        solo_support = hypothesis("GetFlights", "passengers", "1")
        opposed = self.decision(
            "low cost",
            "GetRestaurants",
            [low_support, high_support],
            query="Find a restaurant in Rome.",
        )
        self.assertEqual(opposed.reason_code, "OPPOSING_TRAIT_SUPPORT")
        low_neutral = self.decision(
            "low cost",
            "GetRestaurants",
            [low_support, hypothesis("GetBuses", "group_size", "1")],
            query="Find a restaurant in Rome.",
        )
        self.assertTrue(low_neutral.authorized)
        solo_neutral = self.decision(
            "prefers solo trips",
            "GetBuses",
            [solo_support, low_support, high_support],
            query="Find a bus from Rome to Milan.",
        )
        self.assertTrue(solo_neutral.authorized)

    def test_target_slot_redundant_conflict_and_ambiguity_preserve_type(self):
        support = hypothesis("GetFlights", "passengers", "1")
        query = "Find a bus from Rome to Milan."
        redundant = self.decision(
            "prefers solo trips",
            "GetBuses",
            [support, hypothesis("GetBuses", "group_size", "1")],
            query=query,
        )
        self.assertEqual(redundant.reason_code, "TARGET_SLOT_REDUNDANT")
        conflict = self.decision(
            "prefers solo trips",
            "GetBuses",
            [support, hypothesis("GetBuses", "group_size", 1)],
            query=query,
        )
        self.assertEqual(conflict.reason_code, "TARGET_SLOT_CONFLICT")
        ambiguous = self.decision(
            "prefers solo trips",
            "GetBuses",
            [
                support,
                hypothesis("GetBuses", "group_size", "1"),
                hypothesis("GetBuses", "group_size", "2"),
            ],
            query=query,
        )
        self.assertEqual(ambiguous.reason_code, "TARGET_SLOT_AMBIGUOUS")

    def test_user_only_multiturn_mask_and_malformed_fail_closed(self):
        assistant_only = self.mask(
            "User: Find a flight from Rome to Milan.\nAssistant: I will reserve two passengers.",
            "GetFlights",
            mode="multiturn",
        )
        self.assertEqual(assistant_only.constrained_slots, ())
        tool_only = self.mask(
            "User: Find a flight from Rome to Milan.\nTool: passengers=2",
            "GetFlights",
            mode="multiturn",
        )
        self.assertEqual(tool_only.constrained_slots, ())
        user_explicit = self.mask(
            "User: Find a flight for two passengers.\nAssistant: Understood.",
            "GetFlights",
            mode="multiturn",
        )
        self.assertEqual(user_explicit.constrained_slots, ("passengers",))
        malformed = self.mask(
            "Assistant: Find a flight for two passengers.",
            "GetFlights",
            mode="multiturn",
        )
        self.assertEqual(malformed.status, "USER_PAYLOAD_UNAVAILABLE_FAIL_CLOSED")
        self.assertEqual(
            malformed.constrained_slots,
            ("flight_class", "passengers"),
        )
        bad_prefix = self.mask(
            "User Find a flight for two passengers.",
            "GetFlights",
            mode="multiturn",
        )
        self.assertEqual(bad_prefix.status, "USER_PAYLOAD_UNAVAILABLE_FAIL_CLOSED")

    def test_restaurant_query_mask_respects_schema_key(self):
        query = "Find a restaurant table for two people."
        single = self.mask(query, "GetRestaurants", schema_key="single")
        multi = self.mask(query, "GetRestaurants", schema_key="multi")
        self.assertNotIn("number_of_seats", single.constrained_slots)
        self.assertEqual(multi.constrained_slots, ("number_of_seats",))
        self.assertEqual(single.query_sha256, sha256_text(query))
        self.assertEqual(multi.query_sha256, sha256_text(query))

    def test_current_query_constraint_suppresses_latent_transfer(self):
        query = "Find a bus for two passengers."
        result = self.decision(
            "prefers solo trips",
            "GetBuses",
            [hypothesis("GetFlights", "passengers", "1")],
            query=query,
        )
        self.assertEqual(result.reason_code, "CURRENT_QUERY_EXPLICIT_CONSTRAINT")

    def test_forged_query_mask_query_mode_and_schema_are_rejected(self):
        query = "Find a bus from Rome to Milan."
        valid = self.mask(query, "GetBuses")
        for forged in (
            replace(valid, query_sha256="0" * 64),
            replace(valid, constrained_slots=("group_size",), status="MASKED_EXPLICIT_TARGET_SLOTS"),
            replace(valid, query_mode="multiturn"),
            replace(valid, schema_sha256="f" * 64),
        ):
            with self.subTest(forged=forged):
                with self.assertRaisesRegex(ValueError, "exact ontology-frozen query mask"):
                    self.decision(
                        "prefers solo trips",
                        "GetBuses",
                        [hypothesis("GetFlights", "passengers", "1")],
                        query=query,
                        query_mask=forged,
                    )

    def test_serializer_rebinds_schema_key_query_and_mask(self):
        query = "Find a restaurant in Rome."
        support = [hypothesis("GetFlights", "passengers", "1")]
        decision = self.decision(
            "prefers solo trips",
            "GetRestaurants",
            support,
            query=query,
            schema_key="multi",
        )
        mask = self.mask(query, "GetRestaurants", schema_key="multi")
        serialized = canonical_vlt_prompt_tuple(
            decision,
            ontology=self.ontology,
            schema=self.schemas["multi"],
            preference_slots=self.preference_slots,
            selected_domain="GetRestaurants",
            schema_key="multi",
            current_query=query,
            query_mode="singleturn",
            query_constraint_mask=mask,
        )
        self.assertEqual(json.loads(serialized)["value"], "1")

        masked_query = "Find a restaurant table for two people."
        masked = self.mask(masked_query, "GetRestaurants", schema_key="multi")
        self.assertIsNone(
            canonical_vlt_prompt_tuple(
                decision,
                ontology=self.ontology,
                schema=self.schemas["multi"],
                preference_slots=self.preference_slots,
                selected_domain="GetRestaurants",
                schema_key="multi",
                current_query=masked_query,
                query_mode="singleturn",
                query_constraint_mask=masked,
            )
        )
        single_mask = self.mask(query, "GetRestaurants", schema_key="single")
        self.assertIsNone(
            canonical_vlt_prompt_tuple(
                decision,
                ontology=self.ontology,
                schema=self.schemas["single"],
                preference_slots=self.preference_slots,
                selected_domain="GetRestaurants",
                schema_key="single",
                current_query=query,
                query_mode="singleturn",
                query_constraint_mask=single_mask,
            )
        )

    def test_action_case_filters_same_mask_from_typed_and_vlt_paths(self):
        task = {
            "case_key": "synthetic-flight-mask",
            "example_id": "synthetic-example",
            "mode": "singleturn",
            "query": "Find a flight for two passengers.",
            "schema_key": "single",
        }
        typed = hypothesis("GetFlights", "passengers", "1")
        memory = {
            "latent_abstraction": {"implicit_pref": "prefers solo trips"},
            "latent_attestation": {},
            "typed_hypotheses": [typed],
        }
        with patch(
            "ecpr.latent_firewall._attestation_authorizes",
            return_value=(True, "AUTHORIZED"),
        ):
            case = build_action_case(
                arm="candidate",
                task=task,
                history={"sessions": []},
                memory=memory,
                schema=self.schemas["single"],
                preference_slots=self.preference_slots,
                policy=self.policy,
                public_domain_ontology=self.public_ontology,
                latent_trait_ontology=self.ontology,
                latent_trait_ontology_sha256=sha256_file(ONTOLOGY_PATH),
            )
        mask = case.audit_artifact["query_constraint_mask"]
        self.assertEqual(mask["constrained_slots"], ["passengers"])
        self.assertEqual(case.audit_artifact["routed_before_constraint_count"], 1)
        self.assertEqual(case.audit_artifact["routed_constraint_filtered_count"], 1)
        self.assertEqual(case.audit_artifact["routed_count"], 0)
        self.assertEqual(
            case.vlt_decision.reason_code,
            "CURRENT_QUERY_EXPLICIT_CONSTRAINT",
        )
        self.assertEqual(mask["query_sha256"], sha256_text(task["query"]))
        self.assertEqual(mask["schema_sha256"], sha256_text(canonical_json(self.schemas["single"])))
        self.assertEqual(
            case.audit_artifact["query_constraint_mask_sha256"],
            sha256_text(canonical_json(mask)),
        )

    def test_source_literal_never_reaches_candidate_prompt_projection(self):
        routed = [hypothesis("GetFlights", "passengers", "1")]
        block = candidate_memory_block(
            routed,
            1536,
            selected_domain="GetFlights",
        )
        self.assertNotIn("source_literal", block)
        self.assertNotIn("value_type", block)
        self.assertEqual(
            tuple(inspect.signature(build_action_prompt).parameters),
            ("task", "schema", "memory_block"),
        )


if __name__ == "__main__":
    unittest.main()
