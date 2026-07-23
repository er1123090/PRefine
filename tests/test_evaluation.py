from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import counts_slot_and_value_or, build_gt_allowed_map, build_pred_map


class Experiment5EvaluationTest(unittest.TestCase):
    def test_or_ground_truth_and_json_prediction(self) -> None:
        ground_truth = ['GetHotels(stars="4")', 'GetHotels(stars="5")']
        prediction = json.dumps({"GetHotels": {"stars": "5"}})
        counts = counts_slot_and_value_or(
            build_gt_allowed_map(ground_truth), build_pred_map(prediction)
        )
        self.assertEqual(counts, (1, 0, 0))

    def test_native_dict_prediction_is_not_silently_accepted(self) -> None:
        prediction = {"GetHotels": {"stars": "5"}}
        self.assertEqual(build_pred_map(prediction), {})


if __name__ == "__main__":
    unittest.main()
