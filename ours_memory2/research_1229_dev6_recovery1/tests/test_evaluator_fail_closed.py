from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ecpr.contracts import PREDICTION_FIELDS
from ecpr.evaluate import compute_paired_summary
from ecpr.io import sha256_file, write_json, write_jsonl


def prediction(case_key: str, example_id: str, arm: str, output: str):
    row = {
        "case_key": case_key,
        "example_id": example_id,
        "mode": "singleturn",
        "arm": arm,
        "llm_output": output,
        "status": "ok",
        "model_snapshot": "snap",
        "seed": 1 if case_key == "a" else 2,
        "temperature": 0.0,
        "max_tokens": 10,
        "calls": 1,
        "prompt_hash": "p",
        "schema_hash": "s",
        "memory_hash": "m",
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }
    assert set(row) == PREDICTION_FIELDS
    return row


class EvaluatorFailClosedTests(unittest.TestCase):
    def test_missing_prediction_is_empty_but_coverage_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "artifacts").mkdir()
            (root / "configs").mkdir()
            (root / "evaluator_vault").mkdir()
            (root / "reports").mkdir()
            prereg = {
                "statistics": {"bootstrap_draws": 20, "bootstrap_seed": 3, "randomization_draws": 30, "randomization_seed": 4},
                "guardrails": {
                    "minimum_each_task_delta_f1": -0.005,
                    "maximum_preference_f1_drop": 0.005,
                    "maximum_nonpreference_f1_drop": 0.005,
                    "maximum_parse_failure_rate_increase": 0.005,
                    "required_coverage": 1.0,
                    "equal_action_budget": True,
                },
                "pass": {"minimum_delta_bmf1": 0.01, "maximum_p_value_exclusive": 0.05},
            }
            prereg_path = root / "preregistration.json"
            write_json(prereg_path, prereg)
            gold = [
                {"case_key": "a", "example_id": "u1", "mode": "singleturn", "difficulty": "easy", "reference_ground_truth": ['Book(seat="quiet")']},
                {"case_key": "b", "example_id": "u2", "mode": "singleturn", "difficulty": "easy", "reference_ground_truth": ['Book(seat="quiet")']},
            ]
            tasks = [
                {"case_key": key, "example_id": ex, "mode": "singleturn", "target_domain": "Book", "query": "q", "schema_key": "single", "explicit_slots": []}
                for key, ex in (("a", "u1"), ("b", "u2"))
            ]
            gold_path = root / "evaluator_vault/gold.jsonl"
            tasks_path = root / "artifacts/tasks.jsonl"
            write_jsonl(gold_path, gold)
            write_jsonl(tasks_path, tasks)
            write_json(
                root / "evaluator_vault/sealed_manifest.json",
                {
                    "preregistration_sha256": sha256_file(prereg_path),
                    "gold_sha256": sha256_file(gold_path),
                    "task_sha256": sha256_file(tasks_path),
                },
            )
            write_json(root / "configs/preference_slots.json", {"Book": ["seat"]})
            baseline_path = root / "artifacts/baseline.jsonl"
            candidate_path = root / "artifacts/candidate.jsonl"
            write_jsonl(
                baseline_path,
                [prediction("a", "u1", "baseline", 'Book(seat="quiet")'), prediction("b", "u2", "baseline", 'Book(seat="quiet")')],
            )
            write_jsonl(candidate_path, [prediction("a", "u1", "candidate", 'Book(seat="quiet")')])
            summary = compute_paired_summary(
                gold_rows=gold,
                baseline_rows=[
                    prediction("a", "u1", "baseline", 'Book(seat="quiet")'),
                    prediction("b", "u2", "baseline", 'Book(seat="quiet")'),
                ],
                candidate_rows=[prediction("a", "u1", "candidate", 'Book(seat="quiet")')],
                preference_slots={"Book": ["seat"]},
                preregistration=prereg,
            )
            self.assertEqual(summary["coverage"]["candidate"], 0.5)
            self.assertFalse(summary["gates"]["coverage"])
            self.assertFalse(summary["equal_action_budget"])
            self.assertEqual(summary["metrics"]["candidate"]["singleturn"]["fn"], 1)
            self.assertFalse(summary["pass"])

    def test_duplicate_prediction_hard_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "dupe.jsonl"
            write_jsonl(path, [prediction("a", "u1", "baseline", ""), prediction("a", "u1", "baseline", "")])
            from ecpr.io import iter_jsonl, unique_by

            with self.assertRaisesRegex(ValueError, "duplicate"):
                unique_by(iter_jsonl(path), "case_key")


if __name__ == "__main__":
    unittest.main()
