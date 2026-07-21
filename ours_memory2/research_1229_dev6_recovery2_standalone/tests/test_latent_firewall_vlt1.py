from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from ecpr.contracts import CandidatePolicy
from ecpr.io import canonical_json, load_json, sha256_file, sha256_text
from ecpr.latent_firewall import (
    CANDIDATE_REVISION,
    GENERATED_ATTESTATION_SOURCE,
    LATENT_VERIFICATION_ATTESTATION_KIND,
    _mint_replay_capability,
    decide_latent_transfer,
    parse_replay_capability,
    unattested_latent,
)
from ecpr.memory import build_latent_abstraction_result
from ecpr.prompts import (
    LATENT_INITIAL,
    LATENT_SYSTEM,
    LATENT_VERIFY,
    baseline_memory_block,
    candidate_memory_block,
    lexical_count,
)
from ecpr.provider import Completion


ROOT = Path(__file__).resolve().parents[1]
ONTOLOGY = load_json(ROOT / "configs/latent_trait_ontology.vlt2.json")
SCHEMA = load_json(ROOT / "configs/schema_single.json")
PREFERENCE_SLOTS = load_json(ROOT / "configs/preference_slots.json")
POLICY = CandidatePolicy()
ATTEMPT_ID = "11111111-1111-4111-8111-111111111111"
CALL_ID = "22222222-2222-4222-8222-222222222222"


def valid_attestation(latent: dict) -> dict:
    implicit_pref = latent["implicit_pref"]
    digest = sha256_text(canonical_json(latent))
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
        "final_draft_sha256": digest,
        "implicit_pref_sha256": sha256_text(implicit_pref),
        "final_memory_latent_sha256": digest,
        "verifier_response_sha256": "1" * 64,
        "verifier_request_sha256": "2" * 64,
        "verifier_call_sha256": "3" * 64,
        "verifier_call_key_sha256": "3" * 64,
        "verifier_schema_sha256": "4" * 64,
        "verifier_prompt_sha256": "5" * 64,
        "verifier_messages_sha256": "6" * 64,
    }




def replay_capability(
    latent: dict, attestation: dict
):
    return _mint_replay_capability(
        schema_version=1,
        kind="semantic_prefine_journal_replay_capability_v1",
        attempt_id=ATTEMPT_ID,
        verifier_call_id=CALL_ID,
        verifier_request_sha256=attestation["verifier_request_sha256"],
        verifier_response_sha256=attestation["verifier_response_sha256"],
        attestation_sha256=sha256_text(canonical_json(attestation)),
        latent_sha256=sha256_text(canonical_json(latent)),
        trace_sha256="8" * 64,
    )
def hypothesis(
    domain: str,
    slot: str,
    value,
    *,
    support: int = 3,
    counterevidence: int = 0,
    confidence: float = 1.0,
) -> dict:
    return {
        "domain": domain,
        "slot": slot,
        "value": value,
        "support": support,
        "counterevidence": counterevidence,
        "confidence": confidence,
        "last_seen": 2,
        "provenance": [{"call_digest": sha256_text(f"{domain}:{slot}:{value}")}],
    }


COST_SUPPORT = [
    hypothesis("GetRestaurants", "price_range", "cheap"),
    hypothesis("GetTravel", "free_entry", True),
]


def decision(
    text: str = "cost sensitive",
    *,
    selected_domain: str | None = "GetFlights",
    pqr_status: str = "SELECT",
    typed=None,
    routed=None,
    attestation=None,
    schema=SCHEMA,
    ablations=None,
):
    latent = {"implicit_pref": text}
    resolved_attestation = (
        valid_attestation(latent) if attestation is None else attestation
    )
    return decide_latent_transfer(
        latent_abstraction=latent,
        latent_attestation=resolved_attestation,
        pqr_status=pqr_status,
        selected_domain=selected_domain,
        schema=schema,
        preference_slots=PREFERENCE_SLOTS,
        typed_hypotheses=COST_SUPPORT if typed is None else typed,
        routed_hypotheses=[] if routed is None else routed,
        policy=POLICY,
        ontology=ONTOLOGY,
        ontology_sha256=sha256_file(
            ROOT / "configs/latent_trait_ontology.vlt2.json"
        ),
        ablations=set(ablations or ()),
        replay_capability=(
            replay_capability(latent, resolved_attestation)
            if attestation is None else None
        ),
    )


class LatentFirewallVLT1Tests(unittest.TestCase):
    def test_authorizes_only_two_distinct_non_target_support_domains(self):
        allowed = decision()
        self.assertTrue(allowed.authorized)
        self.assertEqual(
            allowed.prompt_tuple(),
            {
                "domain": "GetFlights",
                "slot": "flight_class",
                "trait_id": "LOW_COST",
                "value": "Economy",
            },
        )
        self.assertEqual(
            allowed.supporting_domains,
            ("GetRestaurants", "GetTravel"),
        )

        duplicate_domain = decision(
            typed=[
                hypothesis("GetRestaurants", "price_range", "cheap"),
                hypothesis("GetRestaurants", "price_range", "inexpensive"),
            ]
        )
        self.assertEqual(
            duplicate_domain.reason_code, "INSUFFICIENT_CROSS_DOMAIN_SUPPORT"
        )
        target_domain_excluded = decision(
            typed=[
                hypothesis("GetFlights", "flight_class", "Economy"),
                hypothesis("GetRestaurants", "price_range", "cheap"),
            ]
        )
        self.assertEqual(
            target_domain_excluded.reason_code,
            "INSUFFICIENT_CROSS_DOMAIN_SUPPORT",
        )

    def test_opposition_and_target_slot_evidence_fail_closed(self):
        opposed = decision(
            typed=[
                *COST_SUPPORT,
                hypothesis("GetRentalCars", "car_type", "Full-size"),
            ]
        )
        self.assertEqual(opposed.reason_code, "OPPOSING_TRAIT_SUPPORT")
        equal = decision(
            typed=[
                *COST_SUPPORT,
                hypothesis("GetFlights", "flight_class", "Economy")
            ]
        )
        self.assertEqual(equal.reason_code, "TARGET_SLOT_REDUNDANT")
        conflict = decision(
            typed=[
                *COST_SUPPORT,
                hypothesis("GetFlights", "flight_class", "Business")
            ]
        )
        self.assertEqual(conflict.reason_code, "TARGET_SLOT_CONFLICT")

    def test_attestation_requires_exact_journal_bindings(self):
        latent = {"implicit_pref": "cost sensitive"}
        unjournaled = valid_attestation(latent)
        unjournaled["journal_attempt_id"] = None
        self.assertEqual(
            decision(attestation=unjournaled).reason_code,
            "ATTESTATION_MISMATCH",
        )
        forged = valid_attestation(latent)
        forged["verdict_valid_is_exactly_true"] = 1
        self.assertEqual(
            decision(attestation=forged).reason_code,
            "INVALID_ATTESTATION",
        )
        external = unattested_latent(
            latent,
            source="external_latent",
            reason_code="EXTERNAL_UNATTESTED",
        ).as_dict()
        self.assertEqual(
            decision(attestation=external).reason_code,
            "ATTESTATION_MISMATCH",
        )

    def test_unsafe_latent_text_is_rejected_after_nfkc(self):
        cases = {
            "not cost sensitive": "NEGATION_PRESENT",
            "cost sensitive and expensive": "AMBIGUOUS_TRAIT",
            "ignore previous cost sensitive": "INSTRUCTION_LIKE",
            "cost sensitive\nreturn json": "UNSAFE_MULTILINE",
            "cost sensitive \u200b": "UNSAFE_CONTROL_OR_FORMAT",
            "cost sensitive \u202e": "UNSAFE_CONTROL_OR_FORMAT",
            "cost sensitive ＂quoted＂": "UNSAFE_QUOTE_OR_BACKTICK",
            "cost sensitive ｀quoted｀": "UNSAFE_QUOTE_OR_BACKTICK",
            "likes windows": "NO_TRAIT_MATCH",
            "cost sensitive " + "x" * 300: "OVER_LENGTH_CAP",
        }
        for text, reason in cases.items():
            with self.subTest(text=text):
                self.assertEqual(decision(text).reason_code, reason)

    def test_ablation_pqr_policy_and_mapping_guards(self):
        self.assertEqual(
            decision(ablations={"no_latent"}).reason_code,
            "NO_LATENT_ABLATION",
        )
        self.assertEqual(
            decision(ablations={"no_typed"}).reason_code,
            "NO_TYPED_ABLATION",
        )
        weak = [hypothesis("GetRestaurants", "price_range", "cheap")]
        for ablation in ({"no_routing"}, {"no_counterevidence"}):
            with self.subTest(ablation=ablation):
                self.assertEqual(
                    decision(typed=weak, ablations=ablation).reason_code,
                    "INSUFFICIENT_CROSS_DOMAIN_SUPPORT",
                )
        self.assertTrue(decision(ablations={"no_routing"}).authorized)
        self.assertEqual(
            decision(selected_domain=None, pqr_status="ABSTAIN").reason_code,
            "PQR_ABSTAIN",
        )
        premium_support = [
            hypothesis("GetRestaurants", "price_range", "pricey"),
            hypothesis("GetRentalCars", "car_type", "Full-size"),
        ]
        self.assertEqual(
            decision(
                "expensive",
                selected_domain="GetFlights",
                typed=premium_support,
            ).reason_code,
            "NO_DOMAIN_MAPPING",
        )

    def test_candidate_prompt_never_contains_raw_latent_or_audit_provenance(self):
        allowed = decision()
        routed = [
            hypothesis("GetFlights", "airline", "fixture-air"),
            hypothesis("GetFlights", "flight_class", "Premium Economy"),
        ]
        block = candidate_memory_block(
            routed,
            1536,
            selected_domain="GetFlights",
            vlt_decision=allowed,
            latent_trait_ontology=ONTOLOGY,
            schema=SCHEMA,
            preference_slots=PREFERENCE_SLOTS,
            latent_trait_ontology_sha256=sha256_file(ROOT / "configs/latent_trait_ontology.vlt2.json"),
        )
        self.assertNotIn("cost sensitive", block)
        self.assertNotIn("provenance", block)
        self.assertIn('"trait_id":"LOW_COST"', block)
        typed_payload = block.split(
            "Typed, current-task-routed evidence:\n", 1
        )[1].split("\n\n", 1)[0]
        json.loads(typed_payload)

        full_count = lexical_count(block)
        reduced = candidate_memory_block(
            routed,
            full_count - 1,
            selected_domain="GetFlights",
            vlt_decision=allowed,
            latent_trait_ontology=ONTOLOGY,
            schema=SCHEMA,
            preference_slots=PREFERENCE_SLOTS,
            latent_trait_ontology_sha256=sha256_file(ROOT / "configs/latent_trait_ontology.vlt2.json"),
        )
        if "Typed, current-task-routed evidence:\n" in reduced:
            payload = reduced.split(
                "Typed, current-task-routed evidence:\n", 1
            )[1].split("\n\n", 1)[0]
            json.loads(payload)
        self.assertNotIn("TRUNCATED_AT_FROZEN_LEXICAL_CAP", reduced)

        memory = {"latent_abstraction": {"implicit_pref": "RAW_SENTINEL"}}
        history = {
            "sessions": [
                {"session_index": 1, "api_calls": ["Book(RAW_CALL_SENTINEL)"]}
            ]
        }
        baseline = baseline_memory_block(memory, history, 1536)
        self.assertIn("RAW_SENTINEL", baseline)
        self.assertIn("RAW_CALL_SENTINEL", baseline)
        self.assertNotIn("RAW_SENTINEL", block)
        self.assertNotIn("RAW_CALL_SENTINEL", block)

    def test_generated_attestation_preserves_frozen_call_shape(self):
        class FakeClient:
            def __init__(self, journaled: bool):
                self.journal = (
                    SimpleNamespace(attempt_id=ATTEMPT_ID)
                    if journaled
                    else None
                )
                self.calls = []

            def complete(self, messages, **kwargs):
                self.calls.append((messages, kwargs))
                if len(self.calls) % 2:
                    return Completion(
                        '{"reasoning":"r","implicit_pref":"cost sensitive"}',
                        {},
                        CALL_ID,
                        "9" * 64,
                    )
                return Completion(
                    '{"valid":true,"feedback":""}',
                    {},
                    CALL_ID,
                    "9" * 64,
                )

        history = {
            "example_id": "fixture",
            "sessions": [
                {
                    "session_index": 1,
                    "dialogue": [{"role": "user", "message": "fixture"}],
                    "api_calls": [],
                }
            ],
        }
        client = FakeClient(True)
        result = build_latent_abstraction_result(
            history, client, 100, max_attempts=1, example_id="fixture"
        )
        attestation = result.verification_attestation
        self.assertEqual(attestation.status, "VALID")
        self.assertEqual(attestation.journal_attempt_id, ATTEMPT_ID)
        self.assertEqual(attestation.verifier_call_id, CALL_ID)
        generation_messages, generation_kwargs = client.calls[0]
        verifier_messages, verifier_kwargs = client.calls[1]
        self.assertEqual(generation_messages[1]["content"], LATENT_INITIAL)
        self.assertIn(LATENT_SYSTEM.splitlines()[0], generation_messages[0]["content"])
        self.assertIn(LATENT_VERIFY.splitlines()[0], verifier_messages[1]["content"])
        self.assertEqual(
            generation_kwargs,
            {
                "seed": 200,
                "max_tokens": 2048,
                "temperature": 0.4,
                "json_object": True,
                "phase": "memory_generation",
                "call_key": "fixture:session:1:attempt:0:draft",
                "schema_sha256": generation_kwargs["schema_sha256"],
            },
        )
        self.assertEqual(verifier_kwargs["seed"], 201)
        self.assertEqual(verifier_kwargs["max_tokens"], 512)
        self.assertEqual(verifier_kwargs["temperature"], 0.0)

        unjournaled = FakeClient(False)
        unsealed = build_latent_abstraction_result(
            history, unjournaled, 100, max_attempts=1, example_id="fixture"
        )
        self.assertEqual(
            unsealed.verification_attestation.reason_code,
            "UNSEALED_VERIFIER",
        )
        self.assertEqual(CANDIDATE_REVISION, "vlt2")


if __name__ == "__main__":
    unittest.main()

