from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ours_memory3.cli import build
from ours_memory3.contracts import PublicInputError


class CliTests(unittest.TestCase):
    def test_rejects_gold_key_before_overlay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "public.jsonl"
            input_path.write_text(
                json.dumps({"case_id": "x", "gold": "forbidden"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(PublicInputError):
                build(input_path, root / "out.jsonl", None)


if __name__ == "__main__":
    unittest.main()

