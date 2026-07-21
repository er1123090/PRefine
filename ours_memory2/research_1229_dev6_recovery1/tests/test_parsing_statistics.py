from __future__ import annotations

import unittest

from ecpr.parsing import extract_calls, slot_value_map
from ecpr.statistics import paired_cluster_statistics


class ParsingStatisticsTests(unittest.TestCase):
    def test_function_and_json_tool_calls_parse(self):
        self.assertEqual(extract_calls('Book(seat="Quiet", count=2)')[0]["arguments"]["count"], 2)
        payload = {"tool_calls": [{"function": {"name": "Book", "arguments": '{"seat":"Quiet"}'}}]}
        self.assertEqual(slot_value_map(payload), {("Book", "seat"): {"quiet"}})

    def test_think_tags_are_removed(self):
        value = "<think>Book(seat='wrong')</think>Book(seat='right')"
        self.assertEqual(slot_value_map(value), {("Book", "seat"): {"right"}})

    def test_cluster_statistics_are_deterministic(self):
        clusters = {}
        for index in range(8):
            clusters[str(index)] = {
                "baseline": {"singleturn": (0, 0, 1), "multiturn": (1, 0, 0)},
                "candidate": {"singleturn": (1, 0, 0), "multiturn": (1, 0, 0)},
            }
        kwargs = dict(
            bootstrap_draws=100,
            bootstrap_seed=11,
            randomization_draws=200,
            randomization_seed=12,
        )
        first = paired_cluster_statistics(clusters, **kwargs)
        second = paired_cluster_statistics(clusters, **kwargs)
        self.assertEqual(first, second)
        self.assertGreater(first["delta_bmf1"], 0)
        self.assertGreater(first["bootstrap_ci_95"][0], 0)


if __name__ == "__main__":
    unittest.main()

