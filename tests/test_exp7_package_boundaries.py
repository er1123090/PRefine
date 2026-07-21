from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class CanonicalPackageBoundaryTests(unittest.TestCase):
    def test_canonical_namespaces_are_importable(self) -> None:
        modules = (
            "exp7",
            "exp7.datasets",
            "exp7.methods",
            "exp7.providers",
            "exp7.evaluation",
            "exp7.provenance",
        )

        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module.__doc__)


if __name__ == "__main__":
    unittest.main()
