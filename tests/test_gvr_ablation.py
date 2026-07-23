from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ablations.gvr.build_memory import MemoryState, PreferenceAggregator


def aggregator_for(mode: str, max_retries: int = 10) -> PreferenceAggregator:
    aggregator = PreferenceAggregator.__new__(PreferenceAggregator)
    aggregator.memory_mode = mode
    aggregator.true_blind = False
    aggregator.max_retries = max_retries
    return aggregator


class GvrAblationTest(unittest.IsolatedAsyncioTestCase):
    async def test_generate_verify_calls_each_component_once_without_refining(self) -> None:
        aggregator = aggregator_for("generate_verify")
        calls = {"generate": 0, "verify": 0}

        async def generate(**_: object) -> dict:
            calls["generate"] += 1
            return {"preference": "quiet"}

        async def verify(**_: object):
            calls["verify"] += 1
            return False, "needs work", "verifier input", {"valid": False}

        aggregator._generate_preference = generate
        aggregator._verify_preference = verify
        result = await aggregator.update_memory(MemoryState(), "dialogue", ["API()"])

        self.assertEqual(calls, {"generate": 1, "verify": 1})
        self.assertEqual(json.loads(result.implicit_pref), {"preference": "quiet"})
        self.assertEqual(
            result.evolution_log[0]["refinement_process"][0]["stage"],
            "generate_then_verify_no_refine",
        )

    async def test_verified_refine_uses_feedback_until_valid(self) -> None:
        aggregator = aggregator_for("verified_refine")
        calls = {"generate": 0, "verify": 0}
        feedback_seen = []

        async def generate(**kwargs: object) -> dict:
            calls["generate"] += 1
            feedback_seen.append(kwargs.get("feedback"))
            return {"attempt": calls["generate"]}

        async def verify(**_: object):
            calls["verify"] += 1
            valid = calls["verify"] == 2
            return valid, "" if valid else "be more general", "input", {"valid": valid}

        aggregator._generate_preference = generate
        aggregator._verify_preference = verify
        result = await aggregator.update_memory(MemoryState(), "dialogue", ["API()"])

        self.assertEqual(calls, {"generate": 2, "verify": 2})
        self.assertEqual(feedback_seen, ["", "be more general"])
        self.assertEqual(json.loads(result.implicit_pref), {"attempt": 2})


if __name__ == "__main__":
    unittest.main()
