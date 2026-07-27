from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    build_gt_allowed_map,
    build_pred_map,
    counts_slot_and_value_or,
    extract_calls,
    is_parsing_failed,
)
from scripts.run_vanilla_batch_sample import evaluation_report


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

    def test_markdown_bold_function_name_is_parsed(self) -> None:
        prediction = '**GetHeating**(temperature="30", mode="heat")'
        self.assertEqual(
            extract_calls(prediction),
            ['GetHeating(temperature="30", mode="heat")'],
        )

    def test_quoted_braced_function_name_is_parsed(self) -> None:
        prediction = '{"GetMusic"(artist="Jay Chou")}'
        self.assertEqual(
            extract_calls(prediction),
            ['GetMusic(artist="Jay Chou")'],
        )

    def test_single_missing_closing_parenthesis_is_repaired(self) -> None:
        prediction = '{"GetHeating"(temperature="18", mode="heat"}'
        self.assertEqual(
            extract_calls(prediction),
            ['GetHeating(temperature="18", mode="heat")'],
        )

    def test_quoted_function_and_argument_names_are_normalized(self) -> None:
        prediction = (
            '{"GetRestaurants":('
            '"price_range"="moderate", "number_of_seats"="1")}'
        )
        self.assertEqual(
            build_pred_map(prediction),
            {
                ("GetRestaurants", "price_range"): {"moderate"},
                ("GetRestaurants", "number_of_seats"): {"1"},
            },
        )

    def test_comma_wrapped_function_and_arguments_are_normalized(self) -> None:
        prediction = (
            '{"GetHeating", "temperature"="16", "eco_mode"="on"}'
        )
        self.assertEqual(
            build_pred_map(prediction),
            {
                ("GetHeating", "temperature"): {"16"},
                ("GetHeating", "eco_mode"): {"on"},
            },
        )

    def test_explicit_no_call_remains_a_parsing_failure(self) -> None:
        failed, reason = is_parsing_failed(
            {"llm_output": "No function call needed."}
        )
        self.assertTrue(failed)
        self.assertEqual(reason, "no_calls_extracted")

    def test_date_and_time_values_use_shared_normalization(self) -> None:
        ground_truth = build_gt_allowed_map(
            [
                'GetHotels(check_in_date="2019-03-05")',
                'GetEvents(start_time="18:15")',
            ]
        )
        prediction = build_pred_map(
            [
                'GetHotels(check_in_date="March 5th")',
                'GetEvents(start_time="6:15 PM")',
            ]
        )
        self.assertEqual(
            counts_slot_and_value_or(ground_truth, prediction),
            (2, 0, 0),
        )

    def test_evaluation_report_splits_conflict_within_condition(self) -> None:
        base = {
            "condition": "single_medium",
            "reference_ground_truth": ['GetHotels(stars="5")'],
            "llm_output": 'GetHotels(stars="5")',
            "status": "OK",
            "token_counts": {},
        }
        report = evaluation_report(
            [
                {
                    **base,
                    "meta": {"subset": "mix"},
                },
                {
                    **base,
                    "meta": {"subset": "conflict_ordered"},
                },
            ]
        )
        split = report["condition_conflict"]["single_medium"]
        self.assertEqual(split["non_conflict"]["n"], 1)
        self.assertEqual(split["conflict"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
