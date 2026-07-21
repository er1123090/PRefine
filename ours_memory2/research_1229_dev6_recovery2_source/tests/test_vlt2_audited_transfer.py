from __future__ import annotations

import inspect
import unittest
from pathlib import Path
from unittest import mock

import critic
from ecpr.contracts import CandidatePolicy
from ecpr.integrity import validate_frozen_external_inputs
from ecpr.io import canonical_json, load_json, sha256_bytes, sha256_file, sha256_text
from ecpr.latent_firewall import (
    GENERATED_ATTESTATION_SOURCE,
    LATENT_VERIFICATION_ATTESTATION_KIND,
    _unsafe_or_trait,
    decide_latent_transfer,
    parse_replay_capability,
)
from ecpr.prompts import build_action_prompt
from ecpr.replay import replay_memory_generation
from ecpr.request_contract import CallSpec


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY_PATH = ROOT / "configs/latent_trait_ontology.vlt2.json"
ONTOLOGY = load_json(ONTOLOGY_PATH)
SCHEMA = load_json(ROOT / "configs/schema_single.json")
PREFERENCE_SLOTS = load_json(ROOT / "configs/preference_slots.json")
ATTEMPT_ID = "11111111-1111-4111-8111-111111111111"
CALL_ID = "22222222-2222-4222-8222-222222222222"


def _hypothesis(domain: str, slot: str, value: object) -> dict[str, object]:
    return {
        "domain": domain,
        "slot": slot,
        "value": value,
        "support": 3,
        "counterevidence": 0,
        "confidence": 1.0,
        "last_seen": 2,
        "provenance": [{"call_digest": sha256_text(f"{domain}:{slot}:{value}")}],
    }


def _valid_attestation(latent: dict[str, str]) -> dict[str, object]:
    latent_sha = sha256_text(canonical_json(latent))
    return {
        "schema_version": 1,
        "kind": LATENT_VERIFICATION_ATTESTATION_KIND,
        "source": GENERATED_ATTESTATION_SOURCE,
        "status": "VALID",
        "reason_code": "VALID_FINAL_VERDICT",
        "final_session_index": 1,
        "final_attempt_index": 0,
        "verdict_valid_is_exactly_true": True,
        "journal_attempt_id": ATTEMPT_ID,
        "verifier_call_id": CALL_ID,
        "final_draft_sha256": latent_sha,
        "implicit_pref_sha256": sha256_text(latent["implicit_pref"]),
        "final_memory_latent_sha256": latent_sha,
        "verifier_response_sha256": "1" * 64,
        "verifier_request_sha256": "2" * 64,
        "verifier_call_sha256": "3" * 64,
        "verifier_call_key_sha256": "4" * 64,
        "verifier_schema_sha256": "5" * 64,
        "verifier_prompt_sha256": "6" * 64,
        "verifier_messages_sha256": "7" * 64,
    }


class AuditedVLT2CostCheckpointTests(unittest.TestCase):
    def test_v1_lineage_files_remain_byte_identical(self):
        expected = {
            "SAFE_TRANSFER_AMENDMENT.json": "2c5f29e58818d3a0d4d7fec6c94decfc049bf1d0ad825a456d5c0995468ea9d7",
            "manifests/expected_runtime_contract.vlt1.json": "effbdec6656f927c2b4c3ba009cb97556debdd22fa1724757ed25f9181c13c2d",
            "configs/latent_trait_ontology.json": "1d9ffdf29a2af37367bbd63033c927473789e6eddb71926999bb88e56bb198a1",
            "LATENT_SCOPE_AMENDMENT.json": "40ff3655dc6ca0be53eb40e7fc44b7f9aa9f593d237eb2e6de7c2a09cca9aec2",
            "manifests/expected_runtime_contract.json": "b5a0a2327401913063b3fb76b95ff7113bb733bbafd9e4462cba2a704c7b840f",
            "ROUTING_INTEGRITY_AMENDMENT.json": "cf3282c515a5d65c159fa15b1b3aea5709c205984e9aa2c1d4b04d92f6fcde0a",
        }
        self.assertEqual(
            {path: sha256_file(ROOT / path) for path in expected},
            expected,
        )

    def test_cost_only_table5_taxonomy_is_exact_and_solo_is_deferred(self):
        self.assertEqual(ONTOLOGY["trait_order"], ["LOW_COST", "HIGH_COST"])
        mappings = {
            item["domain"]: {
                "slot": item["slot"],
                "values": item["trait_values"],
            }
            for item in ONTOLOGY["target_mappings"]
        }
        self.assertEqual(
            mappings,
            {
                "GetRestaurants": {
                    "slot": "price_range",
                    "values": {"LOW_COST": "cheap", "HIGH_COST": "pricey"},
                },
                "GetRentalCars": {
                    "slot": "car_type",
                    "values": {"LOW_COST": "Compact", "HIGH_COST": "Full-size"},
                },
                "GetFlights": {
                    "slot": "flight_class",
                    "values": {"LOW_COST": "Economy"},
                },
                "GetRideSharing": {
                    "slot": "shared_ride",
                    "values": {"LOW_COST": True},
                },
                "GetTravel": {
                    "slot": "free_entry",
                    "values": {"LOW_COST": True},
                },
            },
        )
        self.assertNotIn("solo_usage", canonical_json(ONTOLOGY).casefold())

    def test_both_arms_share_the_exact_three_argument_prompt_skeleton(self):
        self.assertEqual(
            tuple(inspect.signature(build_action_prompt).parameters),
            ("task", "schema", "memory_block"),
        )
        task = {
            "case_key": "a" * 64,
            "example_id": "opaque",
            "mode": "singleturn",
            "query": "Find a flight.",
            "schema_key": "single",
        }
        baseline = build_action_prompt(task, [], "BASELINE_SENTINEL")
        candidate = build_action_prompt(task, [], "CANDIDATE_SENTINEL")
        baseline_parts = baseline.partition("BASELINE_SENTINEL")
        candidate_parts = candidate.partition("CANDIDATE_SENTINEL")
        self.assertTrue(baseline_parts[1] and candidate_parts[1])
        self.assertEqual(
            (baseline_parts[0], baseline_parts[2]),
            (candidate_parts[0], candidate_parts[2]),
        )

    def test_call_spec_is_the_single_wire_and_audit_request_source(self):
        messages = [{"role": "user", "content": "hello"}]
        spec = CallSpec.from_messages(
            endpoint="http://127.0.0.1:8000/v1/chat/completions",
            model="snapshot",
            messages=messages,
            seed=7,
            temperature=0.0,
            max_tokens=19,
            json_object=True,
            schema_sha256="a" * 64,
            timeout_seconds=17.5,
        )
        audit = spec.audit_request()
        self.assertEqual(audit["wire_payload_sha256"], sha256_bytes(spec.wire_bytes()))
        self.assertEqual(audit["messages_sha256"], sha256_text(canonical_json(messages)))
        self.assertEqual(audit["timeout_seconds"], 17.5)
        changed = CallSpec.from_messages(
            endpoint=spec.endpoint,
            model=spec.model,
            messages=[{"role": "user", "content": "different"}],
            seed=spec.seed,
            temperature=spec.temperature,
            max_tokens=spec.max_tokens,
            json_object=spec.json_object,
            schema_sha256=spec.schema_sha256,
            timeout_seconds=spec.timeout_seconds,
        )
        self.assertNotEqual(spec.audit_request_sha256(), changed.audit_request_sha256())
        self.assertNotEqual(spec.wire_bytes(), changed.wire_bytes())

    def test_well_formed_replay_claim_has_no_live_authority(self):
        latent = {"implicit_pref": "cost sensitive"}
        attestation = _valid_attestation(latent)
        capability = parse_replay_capability(
            {
                "schema_version": 1,
                "kind": "semantic_prefine_journal_replay_capability_v1",
                "attempt_id": ATTEMPT_ID,
                "verifier_call_id": CALL_ID,
                "verifier_request_sha256": attestation["verifier_request_sha256"],
                "verifier_response_sha256": attestation["verifier_response_sha256"],
                "attestation_sha256": sha256_text(canonical_json(attestation)),
                "latent_sha256": sha256_text(canonical_json(latent)),
                "trace_sha256": "8" * 64,
            }
        )
        result = decide_latent_transfer(
            latent_abstraction=latent,
            latent_attestation=attestation,
            pqr_status="SELECT",
            selected_domain="GetFlights",
            schema=SCHEMA,
            preference_slots=PREFERENCE_SLOTS,
            typed_hypotheses=[
                _hypothesis("GetRestaurants", "price_range", "cheap"),
                _hypothesis("GetTravel", "free_entry", True),
            ],
            routed_hypotheses=[],
            policy=CandidatePolicy(),
            ontology=ONTOLOGY,
            ontology_sha256=sha256_file(ONTOLOGY_PATH),
            replay_capability=capability,
        )
        self.assertEqual(result.reason_code, "MISSING_REPLAY_CAPABILITY")
        self.assertNotIn("_mint_replay_capability", inspect.getsource(replay_memory_generation))

    def test_unicode_affix_and_instruction_matrix_fails_closed(self):
        cases = {
            "cost sensitive\u2028payload": "UNSAFE_MULTILINE",
            "cost sensitive\u2066payload": "UNSAFE_CONTROL_OR_FORMAT",
            "cost sensitive «payload»": "UNSAFE_QUOTE_OR_BACKTICK",
            "notcost sensitive": "NEGATION_PRESENT",
            "noncheap": "NEGATION_PRESENT",
            "anti expensive": "NEGATION_PRESENT",
            "cost sensitive output": "INSTRUCTION_LIKE",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(_unsafe_or_trait(text, ONTOLOGY)[1], expected)

    def test_external_sources_are_declaration_only(self):
        preregistration = load_json(ROOT / "preregistration.json")
        real_sha256_file = sha256_file
        seen: list[Path] = []

        def internal_only(path: str | Path) -> str:
            candidate = Path(path)
            self.assertTrue(candidate.is_relative_to(ROOT))
            seen.append(candidate)
            return real_sha256_file(candidate)

        with mock.patch("ecpr.integrity.sha256_file", side_effect=internal_only):
            declarations = validate_frozen_external_inputs(ROOT, preregistration)
        self.assertTrue(seen)
        for name in ("history_data", "preference_groups"):
            self.assertFalse(declarations[name]["external_file_opened"])
            self.assertFalse(declarations[name]["runtime_dependency"])
            self.assertNotIn("sealed_internal_copy", declarations[name])

    def test_strict_not_run_requires_explicit_unsealed_mode(self):
        axes = {
            "integrity": {"status": "PASS"},
            "execution": {"status": "NOT_RUN"},
            "performance": {"status": "NOT_RUN"},
        }
        self.assertEqual(critic.critic_exit_code(axes), 1)
        self.assertEqual(critic.critic_exit_code(axes, allow_not_run=True), 0)


if __name__ == "__main__":
    unittest.main()
