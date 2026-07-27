from __future__ import annotations

import unittest

from scripts.run_ours_memory_stratified_batch import (
    ANTHROPIC_MAX_TOKENS,
    ANTHROPIC_MODEL,
    ANTHROPIC_THINKING_BUDGET,
    OPENAI_MAX_COMPLETION_TOKENS,
    OPENAI_MODEL,
    OPENAI_REASONING_EFFORT,
    anthropic_request,
    openai_request,
    proportional_quotas,
)


class OursMemoryStratifiedBatchTest(unittest.TestCase):
    def test_published_1000_row_proportional_allocation(self) -> None:
        counts = {
            "single_easy": 727,
            "single_medium": 674,
            "single_hard": 526,
            "single_conflict-medium": 298,
            "single_conflict-hard": 162,
            "multi_easy": 648,
            "multi_medium": 674,
            "multi_hard": 526,
            "multi_conflict-medium": 298,
            "multi_conflict-hard": 162,
        }
        self.assertEqual(
            proportional_quotas(counts, 1000),
            {
                "multi_conflict-hard": 34,
                "multi_conflict-medium": 63,
                "multi_easy": 138,
                "multi_hard": 112,
                "multi_medium": 144,
                "single_conflict-hard": 35,
                "single_conflict-medium": 63,
                "single_easy": 155,
                "single_hard": 112,
                "single_medium": 144,
            },
        )

    def test_openai_request_uses_gpt5_medium_with_output_guard(self) -> None:
        request = openai_request("v8-00001", "prompt")
        self.assertEqual(request["url"], "/v1/chat/completions")
        self.assertEqual(request["body"]["model"], OPENAI_MODEL)
        self.assertEqual(
            request["body"]["reasoning_effort"], OPENAI_REASONING_EFFORT
        )
        self.assertEqual(
            request["body"]["max_completion_tokens"],
            OPENAI_MAX_COMPLETION_TOKENS,
        )

    def test_anthropic_request_uses_manual_thinking_2048(self) -> None:
        request = anthropic_request("v8-00001", "prompt")
        params = request["params"]
        self.assertEqual(params["model"], ANTHROPIC_MODEL)
        self.assertEqual(params["max_tokens"], ANTHROPIC_MAX_TOKENS)
        self.assertEqual(
            params["thinking"],
            {
                "type": "enabled",
                "budget_tokens": ANTHROPIC_THINKING_BUDGET,
                "display": "omitted",
            },
        )


if __name__ == "__main__":
    unittest.main()
