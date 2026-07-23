from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from ablations.token_count.analyze import (
    calculate_cost_usd,
    covered_usage_cost,
    main as analyze_main,
    usage_coverage,
)
from src.construction_usage import (
    begin_usage_collection,
    end_usage_collection,
    record_response_usage,
    set_usage_session,
    usage_from_response,
)


def openai_response(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
    reasoning_tokens: int = 0,
) -> SimpleNamespace:
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        prompt_tokens_details=SimpleNamespace(cached_tokens=cached_tokens),
        completion_tokens_details=SimpleNamespace(
            reasoning_tokens=reasoning_tokens
        ),
    )
    return SimpleNamespace(usage=usage)


class ConstructionUsageTest(unittest.TestCase):
    def test_provider_usage_is_grouped_by_component_and_session(self) -> None:
        token = begin_usage_collection()
        try:
            set_usage_session(1)
            record_response_usage(
                openai_response(
                    prompt_tokens=100,
                    completion_tokens=40,
                    cached_tokens=10,
                    reasoning_tokens=5,
                ),
                component="generator",
                provider="openai",
                model="test-model",
            )
            set_usage_session(2)
            record_response_usage(
                openai_response(prompt_tokens=60, completion_tokens=20),
                component="verifier",
                provider="openai",
                model="test-model",
            )
        finally:
            report = end_usage_collection(token)

        self.assertEqual(report["summary"]["input_tokens"], 160)
        self.assertEqual(report["summary"]["output_tokens"], 60)
        self.assertEqual(report["summary"]["cached_input_tokens"], 10)
        self.assertEqual(report["summary"]["reasoning_tokens"], 5)
        self.assertEqual(report["summary"]["usage_available_calls"], 2)
        self.assertEqual(
            report["by_component"]["generator"]["total_tokens"], 140
        )
        self.assertEqual(
            report["by_session"]["2"]["by_component"]["verifier"]["input_tokens"],
            60,
        )

    def test_langchain_and_direct_dict_usage_are_normalized(self) -> None:
        langchain = SimpleNamespace(
            llm_output={
                "token_usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 7,
                    "total_tokens": 18,
                }
            }
        )
        self.assertEqual(usage_from_response(langchain)["total_tokens"], 18)
        self.assertEqual(
            usage_from_response(
                {
                    "input_tokens": 13,
                    "output_tokens": 2,
                    "total_tokens": 15,
                }
            )["input_tokens"],
            13,
        )
        langchain_message_usage = {
            "usage_metadata": {
                "input_tokens": 20,
                "output_tokens": 8,
                "total_tokens": 28,
                "input_token_details": {"cache_read": 6},
                "output_token_details": {"reasoning": 3},
            }
        }
        normalized = usage_from_response(langchain_message_usage)
        self.assertEqual(normalized["cached_input_tokens"], 6)
        self.assertEqual(normalized["reasoning_tokens"], 3)

    def test_missing_usage_is_explicit(self) -> None:
        usage = usage_from_response({"results": [{"memory": "quiet room"}]})
        self.assertFalse(usage["usage_available"])
        self.assertEqual(usage["total_tokens"], 0)


class CostCalculationTest(unittest.TestCase):
    def test_cached_input_uses_its_own_rate(self) -> None:
        self.assertAlmostEqual(
            calculate_cost_usd(
                input_tokens=1_000_000,
                cached_input_tokens=200_000,
                output_tokens=500_000,
                input_cost_per_million=5.0,
                cached_input_cost_per_million=1.0,
                output_cost_per_million=15.0,
            ),
            11.7,
        )

    def test_missing_provider_usage_is_not_reported_as_zero_cost(self) -> None:
        missing = {
            "call_count": 3,
            "usage_available_calls": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
        }
        self.assertEqual(usage_coverage(missing), 0)
        self.assertIsNone(
            covered_usage_cost(
                missing,
                input_rate=5.0,
                output_rate=15.0,
                cached_rate=1.0,
            )
        )

    def test_analyzer_separates_provider_cost_from_local_mem0_proxy(self) -> None:
        known_summary = {
            "input_tokens": 1_000,
            "cached_input_tokens": 100,
            "output_tokens": 200,
            "reasoning_tokens": 0,
            "total_tokens": 1_200,
            "call_count": 1,
            "usage_available_calls": 1,
            "usage_missing_calls": 0,
        }
        missing_summary = {
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "total_tokens": 0,
            "call_count": 1,
            "usage_available_calls": 0,
            "usage_missing_calls": 1,
        }
        records = [
            {
                "example_id": "ours",
                "method": "ours_memory",
                "token_counts": known_summary,
                "construction_token_usage": {
                    "summary": known_summary,
                    "by_component": {"generator": known_summary},
                    "by_session": {"1": {"summary": known_summary}},
                },
                "preference_evolution_history": [
                    {
                        "session_index": 1,
                        "stored_memory_tokens_after_session": 200,
                    }
                ],
            },
            {
                "example_id": "mem0",
                "method": "mem0",
                "local_construction_input_tokens": 500,
                "token_counts": missing_summary,
                "construction_token_usage": {
                    "summary": missing_summary,
                    "by_component": {"mem0_add": missing_summary},
                    "by_session": {"1": {"summary": missing_summary}},
                },
                "session_exports": [
                    {
                        "session_index": 1,
                        "local_construction_input_tokens": 500,
                        "stored_memory_tokens_after_session": 50,
                    }
                ],
            },
            {
                "example_id": "legacy-no-usage",
                "method": "vanilla_llm",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "usage.jsonl"
            output_path = Path(directory) / "analysis.json"
            input_path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            argv = [
                "analyze.py",
                str(input_path),
                "--skip-local",
                "--input-cost-per-million",
                "2",
                "--cached-input-cost-per-million",
                "1",
                "--output-cost-per-million",
                "10",
                "--json_output",
                str(output_path),
            ]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(
                io.StringIO()
            ):
                analyze_main()
            analysis = json.loads(output_path.read_text(encoding="utf-8"))

        ours, mem0, legacy = analysis["rows"]
        self.assertAlmostEqual(ours["construction_estimated_cost_usd"], 0.0039)
        self.assertAlmostEqual(
            ours["final_stored_memory_estimated_input_cost_usd"], 0.0004
        )
        self.assertIsNone(mem0["construction_estimated_cost_usd"])
        self.assertAlmostEqual(
            mem0["local_construction_input_estimated_cost_usd"], 0.001
        )
        self.assertEqual(
            analysis["sessions"][1]["provider_usage_coverage"], 0
        )
        self.assertIsNone(legacy["input_tokens"])
        self.assertIsNone(legacy["estimated_cost_usd"])
        self.assertIsNone(legacy["construction_estimated_cost_usd"])


if __name__ == "__main__":
    unittest.main()
