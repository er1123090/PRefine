import json
import tempfile
import unittest
from pathlib import Path

from experiments5.src.data_utils import (
    assign_user_utterances_multiturn,
    assign_user_utterances_singleturn,
)


class DataUtilsAssignmentTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        root = Path(self.tmpdir.name)
        self.pref_list_path = root / "pref_list.json"
        self.pref_group_path = root / "pref_group.json"

        self.pref_list_path.write_text(
            json.dumps(
                {
                    "GetHotels": ["star", "parking"],
                    "GetRestaurants": ["price"],
                }
            ),
            encoding="utf-8",
        )
        self.pref_group_path.write_text(
            json.dumps(
                {
                    "budget": {
                        "rules": [
                            {"domain": "GetHotels", "slot": "star", "value": "3"},
                            {
                                "domain": "GetRestaurants",
                                "slot": "price",
                                "value": "cheap",
                            },
                        ]
                    }
                }
            ),
            encoding="utf-8",
        )

        self.example = {
            "api_calls": ['GetHotels(star="4", city="Seoul")'],
            "api_calls_pref": [
                {
                    "value_group": "budget",
                    "evidence": [
                        {
                            "domain": "GetHotels",
                            "slot": "star",
                            "value": "3",
                        }
                    ],
                }
            ],
        }
        self.query_map = {
            "GetHotels": "Find me a hotel.",
            "GetRestaurants": "Find me a restaurant.",
        }
        self.multiturn_data = {
            "GetHotels": [
                {
                    "query": [
                        {"role": "User", "message": "I need a hotel."},
                        {"role": "Assistant", "message": "Which city?"},
                    ],
                    "api_call": ['GetHotels(city="Seoul")'],
                }
            ],
            "GetRestaurants": [
                {
                    "query": [
                        {"role": "User", "message": "I need dinner."},
                    ],
                    "api_call": ['GetRestaurants(city="Seoul")'],
                }
            ],
        }

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_singleturn_easy_uses_direct_preference_slots(self):
        pairs = assign_user_utterances_singleturn(
            str(self.pref_list_path),
            self.example,
            self.query_map,
            "easy",
            str(self.pref_group_path),
        )

        self.assertEqual(pairs, [("Find me a hotel.", ['GetHotels(star="4")'])])

    def test_singleturn_medium_uses_value_group_rules_for_evidence_domain(self):
        pairs = assign_user_utterances_singleturn(
            str(self.pref_list_path),
            self.example,
            self.query_map,
            "medium",
            str(self.pref_group_path),
        )

        self.assertEqual(pairs, [("Find me a hotel.", ['GetHotels(star="3")'])])

    def test_singleturn_hard_uses_unseen_domain_from_same_group(self):
        pairs = assign_user_utterances_singleturn(
            str(self.pref_list_path),
            self.example,
            self.query_map,
            "hard",
            str(self.pref_group_path),
        )

        self.assertEqual(
            pairs,
            [("Find me a restaurant.", ['GetRestaurants(price="cheap")'])],
        )

    def test_multiturn_easy_merges_template_args_with_preference_slots(self):
        pairs = assign_user_utterances_multiturn(
            str(self.pref_list_path),
            self.example,
            self.multiturn_data,
            "easy",
            str(self.pref_group_path),
        )

        self.assertEqual(
            pairs,
            [
                (
                    "User: I need a hotel.\nAssistant: Which city?",
                    ['GetHotels(city="Seoul", star="4")'],
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
