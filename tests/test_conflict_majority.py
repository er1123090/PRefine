import json
import unittest
from pathlib import Path

from experiments5.src.conflict_majority import (
    build_conflict_majority_tasks,
    select_majority_prefs,
)
from experiments5.src.data_utils import load_multiturn_data, load_query_map


ROOT = Path(__file__).resolve().parents[1]


class ConflictMajorityTaskTest(unittest.TestCase):
    def test_classifies_majority_targets_by_seen_counts(self):
        pref_group = {
            "budget": {
                "rules": [
                    {"domain": "GetHotels", "slot": "average_star", "value": 1},
                    {"domain": "GetHotels", "slot": "average_star", "value": 2},
                    {"domain": "GetRestaurants", "slot": "price_range", "value": "cheap"},
                    {"domain": "GetFlights", "slot": "flight_class", "value": "Economy"},
                ]
            }
        }
        example = {
            "example_id": "dev_x",
            "api_calls_pref": [
                {
                    "value_group": "budget",
                    "count": 3,
                    "evidence": [
                        {
                            "domain": "GetHotels",
                            "slot": "average_star",
                            "values": [
                                {"value": "1", "meta": {"count": 2}},
                            ],
                        },
                        {
                            "domain": "GetRestaurants",
                            "slot": "price_range",
                            "values": [
                                {"value": "cheap", "meta": {"count": 1}},
                            ],
                        },
                    ],
                }
            ],
        }
        query_map = {
            "GetHotels": "Find a hotel.",
            "GetRestaurants": "Find food.",
            "GetFlights": "Find a flight.",
        }

        tasks, summary = build_conflict_majority_tasks(
            examples=[example],
            pref_group_data=pref_group,
            query_catalog=query_map,
            turn_type="singleturn",
        )

        by_target = {(task["target_domain"], task["target_slot"]): task for task in tasks}
        self.assertEqual(summary["tasks_by_difficulty"], {"easy": 1, "medium": 1, "hard": 1})
        self.assertEqual(by_target[("GetHotels", "average_star")]["difficulty"], "easy")
        self.assertEqual(
            by_target[("GetHotels", "average_star")]["reference_ground_truth"],
            ['GetHotels(average_star="1")', 'GetHotels(average_star="2")'],
        )
        self.assertEqual(by_target[("GetRestaurants", "price_range")]["difficulty"], "medium")
        self.assertEqual(by_target[("GetFlights", "flight_class")]["difficulty"], "hard")

    def test_strict_majority_skips_ties(self):
        example = {
            "example_id": "dev_tie",
            "api_calls_pref": [
                {"value_group": "low_cost", "count": 2, "evidence": []},
                {"value_group": "solo_usage", "count": 2, "evidence": []},
            ],
        }

        selected, reason, max_count = select_majority_prefs(example, include_ties=False)
        self.assertEqual(selected, [])
        self.assertEqual(reason, "tie")
        self.assertEqual(max_count, 2)

        selected, reason, max_count = select_majority_prefs(example, include_ties=True)
        self.assertEqual([item["value_group"] for item in selected], ["low_cost", "solo_usage"])
        self.assertIsNone(reason)
        self.assertEqual(max_count, 2)

    def test_real_conflict_files_have_expected_strict_counts(self):
        pref_group = json.loads((ROOT / "config" / "pref_group.json").read_text(encoding="utf-8"))
        singleturn_query = load_query_map(str(ROOT / "config" / "query_singleturn.json"))
        multiturn_query = load_multiturn_data(str(ROOT / "config" / "query_multiturn-domain.json"))

        for name in ["e_dev_conflict_ordered_ratio.json", "e_dev_conflict_random_ratio.json"]:
            examples = json.loads((ROOT / "data" / name).read_text(encoding="utf-8"))
            single_tasks, single_summary = build_conflict_majority_tasks(
                examples=examples,
                pref_group_data=pref_group,
                query_catalog=singleturn_query,
                turn_type="singleturn",
            )
            multi_tasks, multi_summary = build_conflict_majority_tasks(
                examples=examples,
                pref_group_data=pref_group,
                query_catalog=multiturn_query,
                turn_type="multiturn",
            )

            expected = {"easy": 32, "medium": 48, "hard": 67}
            self.assertEqual(single_summary["tasks_by_difficulty"], expected)
            self.assertEqual(multi_summary["tasks_by_difficulty"], expected)
            self.assertEqual(len(single_tasks), 147)
            self.assertEqual(len(multi_tasks), 147)
            self.assertEqual(single_summary["skipped_examples"], {"tie": 1})
            self.assertEqual(multi_summary["skipped_examples"], {"tie": 1})
            self.assertEqual(single_summary["tie_examples"][0]["example_id"], "dev_0028")


if __name__ == "__main__":
    unittest.main()
