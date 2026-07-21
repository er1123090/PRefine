from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ecpr.contracts import CandidatePolicy
from ecpr.firewall import sanitize_example, validate_sanitized_history
from ecpr.io import load_json, write_jsonl
from ecpr.memory import build_typed_hypotheses
from ecpr.prompts import baseline_memory_block, candidate_memory_block
from ecpr.router import route_hypotheses


SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "GetHotels",
            "parameters": {"type": "object", "properties": {"seat": {"type": "string"}, "date": {"type": "string"}}},
        },
    }
]

TASK = {
    "case_key": "a" * 64,
    "example_id": "u1",
    "mode": "singleturn",
    "query": "User: find a hotel with a quiet seat",
    "schema_key": "single",
}

ONTOLOGY = load_json(Path(__file__).resolve().parents[1] / "configs/public_domain_ontology.json")


class FirewallMemoryRouterTests(unittest.TestCase):
    def history(self):
        return {
            "example_id": "u1",
            "sessions": [
                {"session_index": 1, "dialogue": [], "api_calls": ['Book(seat="quiet")']},
                {"session_index": 2, "dialogue": [], "api_calls": ['Book(seat="quiet")']},
                {"session_index": 3, "dialogue": [], "api_calls": ['Book(seat="busy")']},
                {"session_index": 4, "dialogue": [], "api_calls": ['Book(seat="quiet")']},
            ],
        }

    def test_top_level_targets_are_dropped(self):
        raw = {
            "example_id": "u1",
            "api_calls": ["secret target"],
            "api_calls_pref": [{"value_group": "secret"}],
            "sessions": [{"dialogue": [{"role": "user", "message": "hello"}], "api_call": ["Book()"]}],
        }
        sanitized = sanitize_example(raw)
        self.assertEqual(set(sanitized), {"example_id", "sessions"})
        self.assertNotIn("secret", json.dumps(sanitized))

    def test_nested_target_key_fails_closed(self):
        raw = {
            "example_id": "u1",
            "sessions": [{"dialogue": [{"role": "user", "message": "hello", "gold": "x"}], "api_call": []}],
        }
        with self.assertRaisesRegex(ValueError, "forbidden key"):
            sanitize_example(raw)

    def test_sanitized_jsonl_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.jsonl"
            write_jsonl(path, [self.history()])
            self.assertEqual(validate_sanitized_history(path), 1)

    def test_multiple_hypotheses_and_digest_only_provenance(self):
        hypotheses = build_typed_hypotheses(self.history(), {"Book": ["seat"]}, CandidatePolicy())
        self.assertEqual({item["value"] for item in hypotheses}, {"quiet", "busy"})
        quiet = next(item for item in hypotheses if item["value"] == "quiet")
        self.assertEqual((quiet["support"], quiet["counterevidence"]), (3, 1))
        self.assertEqual(quiet["last_seen"], 4)
        self.assertEqual(set(quiet["provenance"][0]), {"session_index", "call_index", "call_digest"})
        self.assertNotIn("Book(", json.dumps(hypotheses))

    def test_router_gates_schema_domain_and_preference_config(self):
        hypotheses = build_typed_hypotheses(self.history(), {"Book": ["seat"]}, CandidatePolicy())
        for hypothesis in hypotheses:
            hypothesis["domain"] = "GetHotels"
        memory = {"typed_hypotheses": hypotheses}
        routed = route_hypotheses(memory, TASK, SCHEMA, {"GetHotels": ["seat"]}, CandidatePolicy(), ONTOLOGY)
        self.assertEqual([(item["slot"], item["value"]) for item in routed], [("seat", "quiet")])

    def test_router_abstains_when_two_values_both_pass(self):
        memory = {
            "typed_hypotheses": [
                {"domain": "GetHotels", "slot": "seat", "value": "a", "support": 7, "counterevidence": 3, "confidence": 0.70, "last_seen": 4, "provenance": []},
                {"domain": "GetHotels", "slot": "seat", "value": "b", "support": 7, "counterevidence": 3, "confidence": 0.68, "last_seen": 3, "provenance": []},
            ]
        }
        routed = route_hypotheses(memory, TASK, SCHEMA, {"GetHotels": ["seat"]}, CandidatePolicy(), ONTOLOGY)
        self.assertEqual(routed, [])

    def test_candidate_prompt_memory_excludes_raw_history(self):
        history = self.history()
        memory = {"latent_abstraction": {"implicit_pref": "prefers quiet environments"}}
        routed = [{"domain": "GetHotels", "slot": "seat", "value": "quiet", "support": 3, "counterevidence": 1, "confidence": 0.75, "last_seen": 4, "provenance": []}]
        baseline = baseline_memory_block(memory, history, 1536)
        candidate = candidate_memory_block(routed, 1536, selected_domain="GetHotels")
        self.assertIn("Book(", baseline)
        self.assertNotIn("Book(", candidate)


if __name__ == "__main__":
    unittest.main()
