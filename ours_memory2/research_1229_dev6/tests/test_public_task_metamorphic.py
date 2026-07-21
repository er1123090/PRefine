from __future__ import annotations

import copy
import unittest

from ecpr.contracts import CandidatePolicy, TASK_FIELDS
from ecpr.inference import _case_seed
from ecpr.prepare import build_public_tasks, build_task_gold_rows
from ecpr.prompts import build_action_prompt
from ecpr.router import route_hypotheses


SCHEMA_HASHES = {"single": "1" * 64, "multi": "2" * 64}
SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "Book",
            "parameters": {
                "type": "object",
                "properties": {
                    "seat": {"type": "string"},
                    "date": {"type": "string"},
                },
            },
        },
    }
]


def legacy_rows() -> list[dict[str, object]]:
    return [
        {
            "case_key": "legacy-a",
            "example_id": "u1",
            "mode": "singleturn",
            "target_domain": "Book",
            "query": "User: reserve a seat",
            "schema_key": "single",
            "explicit_slots": ["seat"],
        },
        {
            "case_key": "legacy-b",
            "example_id": "u1",
            "mode": "singleturn",
            "target_domain": "SecretAlternative",
            "query": "User: reserve a seat",
            "schema_key": "single",
            "explicit_slots": [],
        },
    ]


class PublicTaskMetamorphicTests(unittest.TestCase):
    def test_target_metadata_mutation_cannot_change_task_seed_prompt_or_order(self):
        original = legacy_rows()
        mutated = copy.deepcopy(original)
        mutated[0]["target_domain"] = "MutatedDomain"
        mutated[0]["explicit_slots"] = ["mutated", "label"]
        mutated[1]["target_domain"] = "AnotherDomain"
        mutated[1]["explicit_slots"] = ["answer"]
        mutated.reverse()

        first = build_public_tasks(
            original, SCHEMA_HASHES, allow_legacy_projection=True
        )
        second = build_public_tasks(
            mutated, SCHEMA_HASHES, allow_legacy_projection=True
        )
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        self.assertNotEqual(first[0]["case_key"], first[1]["case_key"])
        self.assertEqual(
            first[0]["case_key"][:8], first[1]["case_key"][:8]
        )
        self.assertTrue(all(set(task) == TASK_FIELDS for task in first))

        seeds = [_case_seed(2026071600, task["case_key"]) for task in first]
        self.assertEqual(seeds[0], seeds[1])
        prompts = [
            build_action_prompt(task, SCHEMA, "same sealed memory")
            for task in first
        ]
        self.assertEqual(prompts[0], prompts[1])
        self.assertNotIn("TARGET DOMAIN", prompts[0])

    def test_label_difficulty_domain_value_and_rules_only_change_evaluator_rows(self):
        frozen = legacy_rows()
        base_examples = [
            {
                "example_id": "u1",
                "label": "label-a",
                "difficulty": "easy",
                "domain": "Book",
                "api_calls": ['Book(seat="quiet")'],
                "api_calls_pref": [
                    {
                        "value_group": "g",
                        "evidence": [{"domain": "Book", "slot": "seat"}],
                    }
                ],
            }
        ]
        mutated_examples = copy.deepcopy(base_examples)
        mutated_examples[0]["label"] = "label-b"
        mutated_examples[0]["difficulty"] = "hard"
        mutated_examples[0]["domain"] = "Changed"
        mutated_examples[0]["api_calls"] = ['Changed(secret="answer")']
        mutated_examples[0]["api_calls_pref"][0]["evidence"] = [
            {"domain": "Changed", "slot": "secret"}
        ]

        def materialize(domain, slots):
            rendered = [
                f"{domain}("
                + ",".join(
                    f"{slot}={value!r}"
                    for slot, values in sorted(slots.items())
                    for value in values
                )
                + ")"
            ]
            return "User: reserve a seat", rendered

        base_groups = {
            "g": {
                "rules": [
                    {"domain": "Book", "slot": "seat", "value": "quiet"}
                ]
            }
        }
        mutated_groups = {
            "g": {
                "rules": [
                    {
                        "domain": "Changed",
                        "slot": "secret",
                        "value": "answer",
                    }
                ]
            }
        }
        base_tasks, base_gold, _ = build_task_gold_rows(
            frozen_task_rows=frozen,
            raw_examples=base_examples,
            preference_slots={"Book": ["seat"]},
            preference_groups=base_groups,
            materializers={"singleturn": materialize},
            schema_hashes=SCHEMA_HASHES,
            allow_legacy_projection=True,
        )
        mutated_tasks, mutated_gold, _ = build_task_gold_rows(
            frozen_task_rows=frozen,
            raw_examples=mutated_examples,
            preference_slots={"Changed": ["secret"]},
            preference_groups=mutated_groups,
            materializers={"singleturn": materialize},
            schema_hashes=SCHEMA_HASHES,
            allow_legacy_projection=True,
        )
        self.assertEqual(base_tasks, mutated_tasks)
        self.assertNotEqual(base_gold, mutated_gold)

    def test_forbidden_task_field_injection_fails_prompt_and_router(self):
        clean = build_public_tasks(
            legacy_rows(), SCHEMA_HASHES, allow_legacy_projection=True
        )[0]
        injected = dict(clean)
        injected["target_domain"] = "Book"
        injected["explicit_slots"] = ["seat"]

        with self.assertRaisesRegex(ValueError, "field contract"):
            build_action_prompt(injected, SCHEMA, "memory")
        with self.assertRaisesRegex(ValueError, "field contract"):
            route_hypotheses(
                {"typed_hypotheses": []},
                injected,
                SCHEMA,
                {"Book": ["seat"]},
                CandidatePolicy(),
            )
        with self.assertRaisesRegex(ValueError, "field contract"):
            build_public_tasks([injected], SCHEMA_HASHES)


if __name__ == "__main__":
    unittest.main()
