from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.exp4_runtime import tabular


ROOT = Path(__file__).resolve().parents[1]


class TabularCompatTest(unittest.TestCase):
    def test_loads_mpt_v2_array_through_exp4_fallback_pattern(self) -> None:
        path = ROOT / "data/MPT_v2_mix600.json"
        with self.assertRaises(ValueError):
            tabular.read_json(str(path), lines=True)

        frame = tabular.DataFrame(__import__("json").loads(path.read_text()))
        self.assertEqual(len(frame), 600)
        _, first = next(frame.iterrows())
        self.assertIn("example_id", first.to_dict())

    def test_loads_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            path.write_text('{"id": 1}\n{"id": 2}\n', encoding="utf-8")
            frame = tabular.read_json(str(path), lines=True)
            self.assertEqual(
                [row.to_dict()["id"] for _, row in frame.iterrows()],
                [1, 2],
            )


if __name__ == "__main__":
    unittest.main()
