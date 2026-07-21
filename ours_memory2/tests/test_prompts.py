from __future__ import annotations

import unittest

from ours_memory2.contracts import ContextMode
from ours_memory2.prompts import (
    build_generation_messages,
    build_inference_messages,
    build_refinement_messages,
    build_verifier_messages,
)


class PromptBuilderTests(unittest.TestCase):
    def test_generation_verification_and_refinement_roles(self) -> None:
        generated = build_generation_messages(
            previous_preference={"implicit_pref": "quiet"},
            dialogue="user: hello",
            api_history=("search(a=1)",),
        )
        self.assertIn("Preference Abstraction", generated[0].content)
        self.assertIn("user: hello", generated[0].content)
        self.assertIn("search(a=1)", generated[0].content)
        self.assertIn("Infer Latent Preference", generated[1].content)

        verified = build_verifier_messages(
            candidate={"implicit_pref": "quiet"},
            dialogue="user: hello",
            api_history=("search(a=1)",),
        )
        self.assertEqual(
            verified[0].content,
            "You are a Preference Verification Module. Output JSON only.",
        )
        self.assertIn("Evidence Support", verified[1].content)
        self.assertIn('"valid": true/false', verified[1].content)

        refined = build_refinement_messages(
            previous_preference={},
            dialogue="user: hello",
            api_history=(),
            previous_draft={"implicit_pref": "draft"},
            feedback="too specific",
        )
        self.assertIn("Do NOT add new information", refined[1].content)
        self.assertIn("too specific", refined[1].content)
        self.assertIn("draft", refined[1].content)

    def test_exactly_four_inference_contexts(self) -> None:
        expected = {
            ContextMode.MEMORY_ONLY: (True, False, False),
            ContextMode.MEMORY_API: (True, True, False),
            ContextMode.MEMORY_DIAG: (True, False, True),
            ContextMode.API_ONLY: (False, True, False),
        }
        self.assertEqual(set(expected), set(ContextMode))
        for mode, (has_pref, has_api, has_dialogue) in expected.items():
            with self.subTest(mode=mode):
                content = build_inference_messages(
                    context_mode=mode,
                    current_utterance="CURRENT_UNIQUE",
                    tool_schema={"tool": "SCHEMA_UNIQUE"},
                    preference="PREFERENCE_UNIQUE",
                    api_history=("API_UNIQUE",),
                    prior_dialogue="DIALOGUE_UNIQUE",
                )[0].content
                self.assertEqual("PREFERENCE_UNIQUE" in content, has_pref)
                self.assertEqual("API_UNIQUE" in content, has_api)
                self.assertEqual("DIALOGUE_UNIQUE" in content, has_dialogue)
                self.assertEqual(content.count("CURRENT_UNIQUE"), 1)
                self.assertEqual(content.count("SCHEMA_UNIQUE"), 1)

    def test_prompt_mapping_order_is_source_order_not_alphabetic(self) -> None:
        content = build_generation_messages(
            previous_preference={"z-last": 1, "a-first": 2},
            dialogue="dialogue",
            api_history=(),
        )[0].content
        self.assertLess(content.index("z-last"), content.index("a-first"))
        inference = build_inference_messages(
            context_mode=ContextMode.MEMORY_ONLY,
            current_utterance="current",
            tool_schema={"z-tool": 1, "a-tool": 2},
            preference={"z-pref": 1, "a-pref": 2},
        )[0].content
        self.assertLess(inference.index("z-tool"), inference.index("a-tool"))
        self.assertLess(inference.index("z-pref"), inference.index("a-pref"))

    def test_inference_methodology_and_single_user_wire_are_preserved(self) -> None:
        for multi_turn, expected_step in (
            (False, "2. **Relevant Memories**"),
            (True, "2. **Current Dialogue Context**"),
        ):
            with self.subTest(multi_turn=multi_turn):
                messages = build_inference_messages(
                    context_mode=ContextMode.MEMORY_API,
                    multi_turn=multi_turn,
                    current_utterance="current",
                    tool_schema={"name": "Tool"},
                    preference={"constraint": "quiet"},
                    api_history=("Tool(value='quiet')",),
                    prior_dialogue="User: prior",
                )
                self.assertEqual([message.role for message in messages], ["user"])
                prompt = messages[0].content
                for required in (
                    "Schema Filtering (Slot Scope Control)",
                    "Consider ONLY the slots defined in the schema",
                    "reasonable support from API history or memories",
                    "Do NOT create empty slots",
                    "Do NOT hallucinate values or infer beyond the schema",
                    "Get~(slot_name=\"value\", ...)",
                    "produce ONLY the final Service API call",
                    expected_step,
                ):
                    self.assertIn(required, prompt)


if __name__ == "__main__":
    unittest.main()
