from __future__ import annotations

import hashlib
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facade.metadata import construct_metadata  # noqa: E402
from facade.registry import load_bundle, load_json  # noqa: E402


class SnapshotIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(ROOT)

    def test_generated_metadata_is_byte_exact(self) -> None:
        for relative, payload in construct_metadata(ROOT).items():
            self.assertEqual((ROOT / relative).read_bytes(), payload, relative)

    def test_snapshot_plan_is_exact_and_pending(self) -> None:
        plan = load_json(ROOT / "configs/snapshot-plan.json")
        self.assertEqual(len(plan["records"]), 129)
        self.assertEqual(
            {row["lineage_id"] for row in plan["records"]},
            set(self.bundle.lineage_by_id),
        )
        self.assertTrue(all(row["final_sha256"] is None for row in plan["records"]))
        self.assertTrue(all(row["snapshot_state"] == "pending_protected_snapshot" for row in plan["records"]))

    def test_root_readme_is_immutable(self) -> None:
        digest = hashlib.sha256((ROOT / "README.md").read_bytes()).hexdigest()
        self.assertEqual(digest, "3f32154f9bf480fcc077e94d26378a6997811bed4fc955160567a615dd4b32fc")


if __name__ == "__main__":
    unittest.main()
