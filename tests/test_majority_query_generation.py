from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_DIR = ROOT / "src/exp4_runtime"
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from common import assign_multiturn_utterances, assign_singleturn_utterances
from methods.mem0.utils_mem0 import (
    assign_user_utterances as assign_mem0_utility_utterances,
)


class MajorityQueryGenerationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)
        self.pref_list_path = temp_path / "pref_list.json"
        self.pref_group_path = temp_path / "pref_group.json"
        self.pref_list_path.write_text("{}", encoding="utf-8")
        self.pref_group_path.write_text(
            json.dumps(
                {
                    "low_cost": {
                        "rules": [
                            {
                                "domain": "GetRestaurants",
                                "slot": "price_range",
                                "value": "cheap",
                            },
                            {
                                "domain": "GetHotels",
                                "slot": "average_star",
                                "value": 1,
                            },
                        ]
                    },
                    "high_cost": {
                        "rules": [
                            {
                                "domain": "GetRestaurants",
                                "slot": "price_range",
                                "value": "pricey",
                            },
                            {
                                "domain": "GetHotels",
                                "slot": "average_star",
                                "value": 5,
                            },
                        ]
                    },
                    "prefers_star": {
                        "rules": [
                            {
                                "domain": "GetMovies",
                                "slot": "starring",
                                "value": "Kris Wu",
                            },
                            {
                                "domain": "GetMovies",
                                "slot": "starring",
                                "value": "Jimmy Lin",
                            },
                            {
                                "domain": "GetMusic",
                                "slot": "artist",
                                "value": "Kris Wu",
                            },
                            {
                                "domain": "GetMusic",
                                "slot": "artist",
                                "value": "Jimmy Lin",
                            },
                        ]
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def budget_example(include_majority: bool = True) -> dict:
        example = {
            "api_calls_pref": [
                {
                    "value_group": "low_cost",
                    "evidence": [
                        {"domain": "GetRestaurants", "slot": "price_range"}
                    ],
                },
                {
                    "value_group": "high_cost",
                    "evidence": [
                        {"domain": "GetRestaurants", "slot": "price_range"}
                    ],
                },
            ]
        }
        if include_majority:
            example["meta"] = {"majority": "low_cost", "minority": "high_cost"}
        return example

    @staticmethod
    def celebrity_example() -> dict:
        return {
            "meta": {"majority": "Kris Wu", "minority": "Jimmy Lin"},
            "api_calls_pref": [
                {
                    "value_group": "prefers_star",
                    "evidence": [
                        {
                            "domain": "GetMovies",
                            "slot": "starring",
                            "values": [{"value": "Kris Wu"}],
                        }
                    ],
                },
                {
                    "value_group": "prefers_star",
                    "evidence": [
                        {
                            "domain": "GetMusic",
                            "slot": "artist",
                            "values": [{"value": "Jimmy Lin"}],
                        }
                    ],
                },
            ],
        }

    def test_medium_uses_only_majority_value_group(self) -> None:
        results = assign_singleturn_utterances(
            pref_list_path=str(self.pref_list_path),
            example=self.budget_example(),
            query_map={"GetRestaurants": "Find a restaurant"},
            pref_type="medium",
            pref_group_path=str(self.pref_group_path),
        )

        self.assertEqual(
            results,
            [
                (
                    "Find a restaurant",
                    ['GetRestaurants(price_range="cheap")'],
                )
            ],
        )

    def test_hard_uses_only_majority_value_group(self) -> None:
        results = assign_singleturn_utterances(
            pref_list_path=str(self.pref_list_path),
            example=self.budget_example(),
            query_map={"GetHotels": "Find a hotel"},
            pref_type="hard",
            pref_group_path=str(self.pref_group_path),
        )

        self.assertEqual(
            results,
            [("Find a hotel", ['GetHotels(average_star="1")'])],
        )

    def test_celebrity_conflict_uses_only_majority_value(self) -> None:
        medium_results = assign_singleturn_utterances(
            pref_list_path=str(self.pref_list_path),
            example=self.celebrity_example(),
            query_map={
                "GetMovies": "Find a movie",
                "GetMusic": "Find music",
            },
            pref_type="medium",
            pref_group_path=str(self.pref_group_path),
        )
        hard_results = assign_singleturn_utterances(
            pref_list_path=str(self.pref_list_path),
            example=self.celebrity_example(),
            query_map={
                "GetMovies": "Find a movie",
                "GetMusic": "Find music",
            },
            pref_type="hard",
            pref_group_path=str(self.pref_group_path),
        )

        self.assertEqual(
            medium_results,
            [("Find a movie", ['GetMovies(starring="Kris Wu")'])],
        )
        self.assertEqual(
            hard_results,
            [("Find music", ['GetMusic(artist="Kris Wu")'])],
        )

    def test_missing_majority_preserves_existing_behavior(self) -> None:
        results = assign_singleturn_utterances(
            pref_list_path=str(self.pref_list_path),
            example=self.budget_example(include_majority=False),
            query_map={"GetRestaurants": "Find a restaurant"},
            pref_type="medium",
            pref_group_path=str(self.pref_group_path),
        )

        ground_truth = {item for _, items in results for item in items}
        self.assertEqual(
            ground_truth,
            {
                'GetRestaurants(price_range="cheap")',
                'GetRestaurants(price_range="pricey")',
            },
        )

    def test_multiturn_celebrity_conflict_excludes_minority(self) -> None:
        multiturn_data = {
            "GetMovies": [
                {
                    "query": [{"role": "User", "message": "Find a movie"}],
                    "api_call": ['GetMovies(date="tomorrow")'],
                }
            ],
            "GetMusic": [
                {
                    "query": [{"role": "User", "message": "Find music"}],
                    "api_call": ['GetMusic(genre="pop")'],
                }
            ],
        }

        for pref_type in ("medium", "hard"):
            with self.subTest(pref_type=pref_type):
                results = assign_multiturn_utterances(
                    pref_list_path=str(self.pref_list_path),
                    example=self.celebrity_example(),
                    multiturn_data=multiturn_data,
                    pref_type=pref_type,
                    pref_group_path=str(self.pref_group_path),
                )
                flattened = [item for _, items in results for item in items]
                self.assertEqual(len(results), 1)
                self.assertTrue(all("Kris Wu" in item for item in flattened))
                self.assertTrue(all("Jimmy Lin" not in item for item in flattened))

    def test_mem0_utility_medium_supports_aggregated_majority_evidence(self) -> None:
        results = assign_mem0_utility_utterances(
            pref_list_path=str(self.pref_list_path),
            example=self.celebrity_example(),
            query_map={
                "GetMovies": "Find a movie",
                "GetMusic": "Find music",
            },
            pref_type="medium",
            pref_group_path=str(self.pref_group_path),
        )

        self.assertEqual(
            results,
            [("Find a movie", 'GetMovies(starring="Kris Wu")')],
        )

    def test_mem0_utility_without_majority_preserves_evidence_value(self) -> None:
        results = assign_mem0_utility_utterances(
            pref_list_path=str(self.pref_list_path),
            example={
                "api_calls_pref": [
                    {
                        "value_group": "low_cost",
                        "evidence": [
                            {
                                "domain": "GetRestaurants",
                                "slot": "price_range",
                                "value": "legacy-cheap",
                            }
                        ],
                    }
                ]
            },
            query_map={"GetRestaurants": "Find a restaurant"},
            pref_type="medium",
            pref_group_path=str(self.pref_group_path),
        )

        self.assertEqual(
            results,
            [
                (
                    "Find a restaurant",
                    'GetRestaurants(price_range="legacy-cheap")',
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
