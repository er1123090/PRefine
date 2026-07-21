from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facade.registry import load_bundle  # noqa: E402


class RegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(ROOT)

    def test_exact_counts_and_required_variants(self) -> None:
        self.assertEqual(len(self.bundle.families), 16)
        self.assertEqual(len(self.bundle.variants), 85)
        self.assertEqual(len(self.bundle.adapters_by_variant), 85)
        required = {
            "infer4_compat", "infer5_native", "infer6_release",
            "eval4_legacy", "eval5_canonical", "eval6_release",
            "eval4_mt_parse_0103a", "eval4_mt_parse_0103b",
        }
        self.assertTrue(required <= set(self.bundle.variants))

    def test_labels_preserve_origin_stage_and_variant(self) -> None:
        for variant_id, adapter in self.bundle.adapters_by_variant.items():
            self.assertIn(f" / {adapter['stage']} / {variant_id}", adapter["label"])
            if variant_id.startswith("eval4_mt_parse_0103"):
                self.assertTrue(adapter["label"].startswith("cross-origin reference / parser /"))
            else:
                self.assertTrue(adapter["label"].startswith(adapter["origin_root"] + " /"))

    def test_every_adapter_closes_to_exact_g1_lineage(self) -> None:
        for variant_id, adapter in self.bundle.adapters_by_variant.items():
            variant = self.bundle.variants[variant_id]
            self.assertEqual(adapter["source_lineage_ids"], variant["lineage_ids"])
            self.assertEqual(adapter["destinations"], variant["destinations"])
            for binding in adapter["source_bindings"]:
                lineage = self.bundle.lineage_by_id[binding["lineage_id"]]
                self.assertEqual(binding["origin_sha256"], lineage["origin_sha256"])
                self.assertEqual(
                    binding["source_manifest_record_id"],
                    lineage["source_manifest_record_id"],
                )

    def test_unpublished_snapshots_are_explicit(self) -> None:
        states = {row["snapshot_state"] for row in self.bundle.adapters_by_variant.values()}
        self.assertEqual(states, {"unpublished", "reference_unpublished"})
        for adapter in self.bundle.adapters_by_variant.values():
            self.assertTrue(all(item["sha256"] is None for item in adapter["final_snapshot_bindings"]))


if __name__ == "__main__":
    unittest.main()
