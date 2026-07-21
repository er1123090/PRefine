from __future__ import annotations

import inspect
import json
import unittest
from pathlib import Path

from ecpr.io import sha256_file
from ecpr.parsing import call_to_string, extract_calls
from ecpr.prompts import (
    ACTION_COMMON_SAFETY_RULES,
    MULTI_ACTION_INFERENCE_TEMPLATE,
    SINGLE_ACTION_INFERENCE_TEMPLATE,
    build_action_prompt,
)


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = ROOT.parents[1]
SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "GetFlights",
            "description": "서울 출발 항공편을 찾습니다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "flight_class": {
                        "type": "string",
                        "enum": ["Economy", "Business"],
                    },
                    "passengers": {"type": "string"},
                },
                "required": [],
            },
        },
    }
]


def task(mode: str, schema_key: str) -> dict[str, str]:
    query = (
        "User: Find a flight.\nAssistant: Which class?\nUser: Business."
        if mode == "multiturn"
        else "User: Find a flight."
    )
    return {
        "case_key": "a" * 64,
        "example_id": "opaque",
        "mode": mode,
        "query": query,
        "schema_key": schema_key,
    }


class PReFineActionPromptVLT3Tests(unittest.TestCase):
    def test_three_argument_api_and_same_mode_arm_skeleton_are_preserved(self):
        self.assertEqual(
            tuple(inspect.signature(build_action_prompt).parameters),
            ("task", "schema", "memory_block"),
        )
        for mode, schema_key in (
            ("singleturn", "single"),
            ("multiturn", "multi"),
        ):
            with self.subTest(mode=mode):
                baseline = build_action_prompt(
                    task(mode, schema_key), SCHEMA, "BASELINE_SENTINEL"
                )
                candidate = build_action_prompt(
                    task(mode, schema_key), SCHEMA, "CANDIDATE_SENTINEL"
                )
                baseline_parts = baseline.partition("BASELINE_SENTINEL")
                candidate_parts = candidate.partition("CANDIDATE_SENTINEL")
                self.assertTrue(baseline_parts[1])
                self.assertTrue(candidate_parts[1])
                self.assertEqual(
                    (baseline_parts[0], baseline_parts[2]),
                    (candidate_parts[0], candidate_parts[2]),
                )
                self.assertEqual(baseline.count("BASELINE_SENTINEL"), 1)
                self.assertEqual(candidate.count("CANDIDATE_SENTINEL"), 1)

    def test_mode_selects_detailed_source_local_template(self):
        single = build_action_prompt(task("singleturn", "single"), SCHEMA, "M")
        multi = build_action_prompt(task("multiturn", "multi"), SCHEMA, "M")
        self.assertIn("[Task Definition: SINGLE-TURN]", single)
        self.assertIn("2. Relevant Memories:", single)
        self.assertNotIn("2. Current Dialogue Context:", single)
        self.assertIn("CURRENT DIALOGUE (single current user utterance):", single)
        self.assertIn("[Task Definition: MULTI-TURN]", multi)
        self.assertIn("2. Current Dialogue Context:", multi)
        self.assertIn("3. Relevant Memories:", multi)
        self.assertIn("CURRENT DIALOGUE (multi-turn context):", multi)
        for template in (
            SINGLE_ACTION_INFERENCE_TEMPLATE,
            MULTI_ACTION_INFERENCE_TEMPLATE,
        ):
            self.assertIn("Repetitiveness", template)
            self.assertIn("Cross-domain Consistency", template)
            self.assertIn("Schema Filtering (Slot Scope Control)", template)

    def test_mode_schema_key_mismatch_fails_closed(self):
        for mode, schema_key in (
            ("singleturn", "multi"),
            ("multiturn", "single"),
        ):
            with self.subTest(mode=mode, schema_key=schema_key):
                with self.assertRaisesRegex(ValueError, "mode/schema_key mismatch"):
                    build_action_prompt(task(mode, schema_key), SCHEMA, "M")
        with self.assertRaisesRegex(ValueError, "invalid action task mode"):
            build_action_prompt(task("unknown", "single"), SCHEMA, "M")

    def test_common_safety_rules_are_identical_and_complete(self):
        prompts = (
            build_action_prompt(task("singleturn", "single"), SCHEMA, "M"),
            build_action_prompt(task("multiturn", "multi"), SCHEMA, "M"),
        )
        required = (
            "CURRENT DIALOGUE always override memory",
            "Memory may fill only otherwise-missing preference slots",
            "Never change the requested operation",
            "emit a slot outside the current schema",
            "weak, conflicting, ambiguous, irrelevant, or insufficient",
            "Never copy or replay an entire historical API call",
            "transient historical details such as dates, times, locations",
            "A stable preference value may be reused only when it is relevant",
        )
        for prompt in prompts:
            self.assertEqual(prompt.count(ACTION_COMMON_SAFETY_RULES), 1)
            for phrase in required:
                self.assertIn(phrase, prompt)

    def test_schema_is_pretty_printed_unicode_and_sorted(self):
        prompt = build_action_prompt(task("singleturn", "single"), SCHEMA, "M")
        expected = json.dumps(
            SCHEMA, ensure_ascii=False, sort_keys=True, indent=2
        )
        compact = json.dumps(SCHEMA, ensure_ascii=False, sort_keys=True)
        self.assertEqual(prompt.count(expected), 1)
        self.assertNotIn(compact, prompt)
        self.assertIn("서울 출발 항공편을 찾습니다.", prompt)
        self.assertNotIn("\\uc11c\\uc6b8", prompt)

    def test_single_function_call_output_contract_parser_roundtrip(self):
        arguments = {"flight_class": "Economy", "passengers": "1"}
        wire = call_to_string("GetFlights", arguments)
        self.assertEqual(
            wire,
            'GetFlights(flight_class="Economy", passengers="1")',
        )
        self.assertEqual(
            extract_calls(wire),
            [{"name": "GetFlights", "arguments": arguments}],
        )
        for mode, schema_key in (
            ("singleturn", "single"),
            ("multiturn", "multi"),
        ):
            prompt = build_action_prompt(task(mode, schema_key), SCHEMA, "M")
            self.assertIn('FunctionName(slot="value", ...)', prompt)
            self.assertIn("produce exactly one final Service API call", prompt)
            self.assertIn("Output no explanation, JSON wrapper, list", prompt)

    def test_vlt3_source_prompt_provenance_is_repo_relative_and_hash_bound(self):
        ontology = json.loads(
            (ROOT / "configs/latent_trait_ontology.vlt3.json").read_text()
        )
        source = ontology["provenance"]["original_source_prompt"]
        self.assertEqual(
            source["repository_relative_path"],
            "ours_memory2/ours_memory2/prompts.py",
        )
        self.assertEqual(
            source["sha256"],
            "2a7e69a3ab13b40727793de7b935b4643c14069983f5bc83900e64707d3ce411",
        )
        self.assertEqual(
            sha256_file(REPOSITORY_ROOT / source["repository_relative_path"]),
            source["sha256"],
        )


if __name__ == "__main__":
    unittest.main()
