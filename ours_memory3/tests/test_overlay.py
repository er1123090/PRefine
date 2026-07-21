from __future__ import annotations

import unittest

from ours_memory3 import OverlayPolicy, build_overlay
from ours_memory3.prompts import build_action_prompt


def history(*calls: str) -> dict[str, object]:
    return {"sessions": [{"session_index": index, "api_calls": [call]} for index, call in enumerate(calls)]}


class OverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = "[Implicit Preferences]:\nlatent\n\n[Past API History]:\nhistory"
        self.slots = {"Restaurant": ["price", "area"], "Hotel": ["stars"]}

    def test_repeated_public_fact_is_append_only(self) -> None:
        decision = build_overlay(
            baseline_memory=self.base,
            history=history('Restaurant(price="cheap")', 'Restaurant(price="cheap")'),
            query="Find a restaurant",
            mode="singleturn",
            preference_slots=self.slots,
            schema_domain="Restaurant",
        )
        self.assertFalse(decision.baseline_unchanged)
        self.assertEqual(decision.memory[: len(self.base)], self.base)
        self.assertIn("[Evidence-Calibrated Preference Overlay]", decision.memory)
        self.assertEqual(
            [(fact.slot, fact.value, fact.support_sessions) for fact in decision.facts],
            [("price", "cheap", 2)],
        )
        self.assertNotIn("source_literal", decision.memory)
        self.assertNotIn("session_index", decision.memory)

    def test_no_eligible_fact_preserves_baseline_bytes(self) -> None:
        decision = build_overlay(
            baseline_memory=self.base,
            history=history('Restaurant(price="cheap")'),
            query="Find a restaurant",
            mode="singleturn",
            preference_slots=self.slots,
            schema_domain="Restaurant",
        )
        self.assertTrue(decision.baseline_unchanged)
        self.assertEqual(decision.memory, self.base)

    def test_conflict_or_tie_is_suppressed(self) -> None:
        decision = build_overlay(
            baseline_memory=self.base,
            history=history(
                'Restaurant(price="cheap")',
                'Restaurant(price="expensive")',
                'Restaurant(price="cheap")',
                'Restaurant(price="expensive")',
            ),
            query="Find a restaurant",
            mode="singleturn",
            preference_slots=self.slots,
            schema_domain="Restaurant",
        )
        self.assertEqual(decision.memory, self.base)
        self.assertEqual(decision.facts, ())

    def test_current_explicit_slot_masks_transfer(self) -> None:
        decision = build_overlay(
            baseline_memory=self.base,
            history=history('Restaurant(price="cheap")', 'Restaurant(price="cheap")'),
            query="Find a restaurant with a price I specify",
            mode="singleturn",
            preference_slots=self.slots,
            schema_domain="Restaurant",
        )
        self.assertEqual(decision.memory, self.base)
        self.assertIn("price", decision.explicit_slots)

    def test_ambiguous_domain_fails_closed(self) -> None:
        decision = build_overlay(
            baseline_memory=self.base,
            history=history('Restaurant(price="cheap")', 'Restaurant(price="cheap")'),
            query="Find a restaurant and hotel",
            mode="singleturn",
            preference_slots=self.slots,
        )
        self.assertEqual(decision.memory, self.base)
        self.assertIsNone(decision.selected_domain)

    def test_only_memory_boundary_changes_in_action_prompt(self) -> None:
        candidate = build_overlay(
            baseline_memory=self.base,
            history=history('Restaurant(price="cheap")', 'Restaurant(price="cheap")'),
            query="Find a restaurant",
            mode="singleturn",
            preference_slots=self.slots,
            schema_domain="Restaurant",
        ).memory
        marker = "__MEMORY__"
        expected = build_action_prompt(
            mode="singleturn",
            schema={"tool": "Restaurant"},
            memory=marker,
            query="Find a restaurant",
        )
        baseline = build_action_prompt(
            mode="singleturn",
            schema={"tool": "Restaurant"},
            memory=self.base,
            query="Find a restaurant",
        )
        paired = build_action_prompt(
            mode="singleturn",
            schema={"tool": "Restaurant"},
            memory=candidate,
            query="Find a restaurant",
        )
        self.assertEqual(baseline.replace(self.base, marker, 1), expected)
        self.assertEqual(paired.replace(candidate, marker, 1), expected)

    def test_policy_requires_independent_sessions(self) -> None:
        strict = OverlayPolicy(minimum_independent_sessions=3)
        decision = build_overlay(
            baseline_memory=self.base,
            history=history('Restaurant(price="cheap")', 'Restaurant(price="cheap")'),
            query="Find a restaurant",
            mode="singleturn",
            preference_slots=self.slots,
            schema_domain="Restaurant",
            policy=strict,
        )
        self.assertEqual(decision.memory, self.base)


if __name__ == "__main__":
    unittest.main()

