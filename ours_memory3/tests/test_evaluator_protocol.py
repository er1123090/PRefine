from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))

import run_paired


def prediction(case_key: str, mode: str, output: str, arm: str) -> dict[str, object]:
    return {
        "case_key": case_key,
        "example_id": "e-" + case_key,
        "mode": mode,
        "arm": arm,
        "llm_output": output,
        "status": "ok",
        "model": "m",
        "seed": 1,
        "temperature": 0.0,
        "max_tokens": 32,
        "calls": 1,
        "prompt_hash": "p",
        "prompt_template_hash": "t",
        "schema_hash": "s",
        "memory_hash": "h",
        "usage": {},
    }


class EvaluatorProtocolTests(unittest.TestCase):
    def test_aggregate_metric_requires_both_modes_to_improve(self) -> None:
        gold = [
            {
                "case_key": "s",
                "example_id": "e-s",
                "mode": "singleturn",
                "difficulty": "x",
                "reference_ground_truth": 'Restaurant(price="cheap")',
            },
            {
                "case_key": "m",
                "example_id": "e-m",
                "mode": "multiturn",
                "difficulty": "x",
                "reference_ground_truth": 'Hotel(stars="3")',
            },
        ]
        baseline = [
            prediction("s", "singleturn", 'Restaurant(price="high")', "baseline"),
            prediction("m", "multiturn", 'Hotel(stars="4")', "baseline"),
        ]
        candidate = [
            prediction("s", "singleturn", 'Restaurant(price="cheap")', "candidate"),
            prediction("m", "multiturn", 'Hotel(stars="3")', "candidate"),
        ]
        result = run_paired._evaluate_rows(gold, baseline, candidate, 0.5)
        self.assertTrue(result["pass"])
        self.assertEqual(result["metrics"]["singleturn"]["delta_percentage_points"], 100.0)
        self.assertEqual(result["metrics"]["multiturn"]["delta_percentage_points"], 100.0)

    def test_thought_text_is_not_treated_as_an_extra_call(self) -> None:
        self.assertEqual(
            run_paired._call_identity('<think>reasoning</think>\nRestaurant(price="cheap")'),
            run_paired._call_identity('Restaurant(price="cheap")'),
        )

    def test_vault_open_is_confined_to_post_receipt_function(self) -> None:
        source = (ROOT / "eval/run_paired.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("read_bytes()"), 1)
        self.assertIn("def _evaluate_after_receipt", source)
        self.assertLess(source.index("def _evaluate_after_receipt"), source.index("gold_bytes ="))


if __name__ == "__main__":
    unittest.main()

