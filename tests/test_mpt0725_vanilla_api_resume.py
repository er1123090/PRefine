from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_mpt0725_vanilla_api_resume import (
    prepare_command,
    record_is_complete,
    select_pending,
)


class Mpt0725VanillaApiResumeTest(unittest.TestCase):
    def test_each_provider_gets_its_own_exact_complement(self) -> None:
        population = [
            {"sample_id": f"v8-{index:05d}", "population_index": index}
            for index in range(5)
        ]
        openai_checkpoint = {
            "v8-00000": {
                "sample_id": "v8-00000",
                "population_index": 0,
                "status": "OK",
                "llm_output": "answer",
            },
            "v8-00001": {
                "sample_id": "v8-00001",
                "population_index": 1,
                "status": "OK",
                "llm_output": "answer",
            },
        }
        anthropic_checkpoint = {
            "v8-00000": openai_checkpoint["v8-00000"],
            "v8-00003": {
                "sample_id": "v8-00003",
                "population_index": 3,
                "status": "OK",
                "llm_output": "answer",
            },
        }

        self.assertEqual(
            [row["sample_id"] for row in select_pending(population, openai_checkpoint)],
            ["v8-00002", "v8-00003", "v8-00004"],
        )
        self.assertEqual(
            [
                row["sample_id"]
                for row in select_pending(population, anthropic_checkpoint)
            ],
            ["v8-00001", "v8-00002", "v8-00004"],
        )

    def test_failed_or_truncated_checkpoint_rows_are_retried(self) -> None:
        population = [
            {"sample_id": "v8-00000", "population_index": 0},
            {"sample_id": "v8-00001", "population_index": 1},
        ]
        checkpoint = {
            "v8-00000": {
                "sample_id": "v8-00000",
                "population_index": 0,
                "status": "ERROR",
                "llm_output": "",
            },
            "v8-00001": {
                "sample_id": "v8-00001",
                "population_index": 1,
                "status": "OK",
                "finish_reason": "length",
                "llm_output": "partial",
            },
        }
        self.assertEqual(len(select_pending(population, checkpoint)), 2)

    def test_thinking_record_must_not_end_inside_think_block(self) -> None:
        self.assertFalse(
            record_is_complete(
                {
                    "status": "OK",
                    "thinking": True,
                    "llm_output": "<think>unfinished",
                }
            )
        )
        self.assertTrue(
            record_is_complete(
                {
                    "status": "OK",
                    "thinking": False,
                    "llm_output": "answer",
                }
            )
        )

    def test_checkpoint_population_drift_is_rejected(self) -> None:
        population = [{"sample_id": "v8-00000", "population_index": 0}]
        checkpoint = {
            "v8-00000": {
                "sample_id": "v8-00000",
                "population_index": 2,
                "status": "OK",
                "llm_output": "answer",
            }
        }
        with self.assertRaises(ValueError):
            select_pending(population, checkpoint)

    def test_prepare_never_overwrites_recorded_batch_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            state = output_dir / "provider" / "shards" / "condition" / "batch_state.json"
            state.parent.mkdir(parents=True)
            state.write_text("{}\n", encoding="utf-8")
            args = SimpleNamespace(
                output_dir=str(output_dir),
                target_root=str(output_dir / "target"),
                dataset=str(output_dir / "dataset.json"),
                force=True,
            )
            with self.assertRaises(RuntimeError):
                prepare_command(args)


if __name__ == "__main__":
    unittest.main()
