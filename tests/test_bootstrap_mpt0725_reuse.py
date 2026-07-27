from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.bootstrap_mpt0725_reuse import reusable_example_ids


class BootstrapMpt0725ReuseTest(unittest.TestCase):
    def test_reusable_examples_require_equal_construction_inputs(self) -> None:
        base = {
            "example_id": "same",
            "sessions": [{"dialogue_id": "d1"}],
            "api_calls": ["GetWeather()"],
            "api_calls_drop": [],
            "api_calls_pref": [{"value_group": "low_cost"}],
            "meta": {"subset": "old"},
        }
        changed_meta = {
            **base,
            "meta": {"subset": "mix"},
        }
        changed_sessions = {
            **changed_meta,
            "example_id": "changed",
            "sessions": [{"dialogue_id": "d2"}],
        }
        conflict = {
            **base,
            "example_id": "conflict",
            "meta": {"subset": "conflict_ordered"},
        }

        reusable = reusable_example_ids(
            [
                base,
                {**changed_sessions, "sessions": [{"dialogue_id": "old"}]},
                {**conflict, "meta": {"subset": "old"}},
            ],
            [changed_meta, changed_sessions, conflict],
        )

        self.assertEqual(reusable, {"same"})


if __name__ == "__main__":
    unittest.main()
