from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_vanilla_batch_sample import (
    anthropic_request,
    anthropic_result_payload,
    openai_request,
    openai_result_payload,
    sample_population,
)


class VanillaBatchSampleTest(unittest.TestCase):
    def test_sampling_is_reproducible_and_without_replacement(self) -> None:
        population = [
            {"sample_id": f"v8-{index:05d}", "population_index": index}
            for index in range(20)
        ]
        first = sample_population(population, sample_size=7, seed=42)
        second = sample_population(population, sample_size=7, seed=42)
        self.assertEqual(first, second)
        self.assertEqual(len({row["sample_id"] for row in first}), 7)
        self.assertEqual(
            [row["population_index"] for row in first],
            sorted(row["population_index"] for row in first),
        )

    def test_provider_requests_share_custom_id_and_prompt(self) -> None:
        row = {"sample_id": "v8-00001", "prompt": "prompt"}
        openai = openai_request(row, "gpt-5", "minimal")
        anthropic = anthropic_request(row, "claude-haiku-4-5", 1024)
        self.assertEqual(openai["custom_id"], anthropic["custom_id"])
        self.assertEqual(openai["body"]["reasoning_effort"], "minimal")
        self.assertNotIn("thinking", anthropic["params"])
        self.assertEqual(
            openai["body"]["messages"][0]["content"],
            anthropic["params"]["messages"][0]["content"],
        )

    def test_openai_success_normalization_preserves_reasoning_usage(self) -> None:
        row = {
            "response": {
                "status_code": 200,
                "body": {
                    "choices": [
                        {
                            "message": {
                                "content": 'GetHotels(stars="5")',
                                "reasoning_content": "",
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 12,
                        "total_tokens": 112,
                        "completion_tokens_details": {"reasoning_tokens": 2},
                        "prompt_tokens_details": {"cached_tokens": 80},
                    },
                },
            },
            "error": None,
        }
        payload = openai_result_payload(row)
        self.assertIsNone(payload["error"])
        self.assertEqual(payload["content"], 'GetHotels(stars="5")')
        self.assertEqual(payload["usage"]["reasoning_tokens"], 2)
        self.assertEqual(payload["usage"]["cached_input_tokens"], 80)

    def test_anthropic_success_normalization_joins_text_blocks(self) -> None:
        row = {
            "result": {
                "type": "succeeded",
                "message": {
                    "content": [
                        {"type": "text", "text": "GetHotels"},
                        {"type": "text", "text": '(stars="5")'},
                    ],
                    "usage": {"input_tokens": 100, "output_tokens": 10},
                },
            }
        }
        payload = anthropic_result_payload(row)
        self.assertIsNone(payload["error"])
        self.assertEqual(payload["content"], 'GetHotels\n(stars="5")')
        self.assertEqual(payload["usage"]["total_tokens"], 110)
        self.assertEqual(payload["usage"]["reasoning_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
