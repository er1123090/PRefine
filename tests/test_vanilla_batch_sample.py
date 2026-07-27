from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_vanilla_batch_sample import (
    anthropic_request,
    anthropic_result_payload,
    build_population,
    openai_request,
    openai_result_payload,
    sample_population,
)
from scripts.run_vanilla_batch_remaining import (
    select_remaining,
    validate_full_rows,
)


class VanillaBatchSampleTest(unittest.TestCase):
    def test_population_can_exclude_easy_conflict(self) -> None:
        general = {
            "example_id": "general",
            "meta": {"subset": "mix"},
        }
        conflict = {
            "example_id": "conflict",
            "meta": {"subset": "conflict_ordered"},
        }

        def prepared_items(**_: object):
            return [
                {
                    "original_ex": general,
                    "utterance": "general query",
                    "ground_truth": ["General()"],
                    "sub_idx": 0,
                },
                {
                    "original_ex": conflict,
                    "utterance": "conflict query",
                    "ground_truth": ["Conflict()"],
                    "sub_idx": 0,
                },
            ]

        with (
            patch(
                "scripts.run_vanilla_batch_sample.prepare_items",
                side_effect=prepared_items,
            ),
            patch(
                "scripts.run_vanilla_batch_sample.common.load_tools_from_file",
                return_value=[],
            ),
            patch(
                "scripts.run_vanilla_batch_sample.common.build_memory_prompt",
                return_value="prompt",
            ),
        ):
            population = build_population(
                query="hint",
                context_type="diag-apilist",
                input_path="/tmp/not-read-by-mock.json",
                exclude_easy_conflict=True,
            )

        self.assertEqual(len(population), 10)
        self.assertFalse(
            any(
                row["pref_type"] == "easy"
                and row["example_id"] == "conflict"
                for row in population
            )
        )
        self.assertEqual(
            sum(row["example_id"] == "conflict" for row in population),
            4,
        )

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

    def test_remaining_selection_is_exact_complement(self) -> None:
        population = [
            {
                "sample_id": f"v8-{index:05d}",
                "population_index": index,
            }
            for index in range(8)
        ]
        sample = [population[1], population[4], population[7]]
        remaining = select_remaining(population, sample)
        self.assertEqual(
            [row["population_index"] for row in remaining],
            [0, 2, 3, 5, 6],
        )
        validate_full_rows(sample + remaining, expected_count=8)

    def test_full_validation_rejects_duplicate_or_missing_index(self) -> None:
        rows = [
            {"sample_id": "v8-00000", "population_index": 0},
            {"sample_id": "v8-00001", "population_index": 0},
        ]
        with self.assertRaises(RuntimeError):
            validate_full_rows(rows, expected_count=2)


if __name__ == "__main__":
    unittest.main()
