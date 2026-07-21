from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facade.metadata import REFERENCE_ONLY, STAGES  # noqa: E402
from facade.registry import load_bundle  # noqa: E402


class ProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(ROOT)

    def test_all_stages_are_explicit(self) -> None:
        for profile_id, profile in self.bundle.profiles_by_id.items():
            selected = set(profile["stages"])
            not_applicable = {row["stage"] for row in profile["not_applicable"]}
            self.assertFalse(selected & not_applicable, profile_id)
            self.assertEqual(selected | not_applicable, set(STAGES), profile_id)

    def test_all_variants_have_profile_coverage(self) -> None:
        coverage = {
            variant_id
            for profile in self.bundle.profiles_by_id.values()
            for variant_id in profile["stages"].values()
        }
        self.assertEqual(coverage, set(self.bundle.variants))

    def test_evaluation_expansions_are_exact(self) -> None:
        for profile_id, profile in self.bundle.profiles_by_id.items():
            evaluation = profile["stages"].get("evaluation")
            if not evaluation:
                continue
            expansion = self.bundle.variants[evaluation].get("expands_to") or {}
            for stage, variant_id in expansion.items():
                self.assertEqual(profile["stages"].get(stage), variant_id, profile_id)

    def test_parser_snapshots_are_distinct_reference_only(self) -> None:
        self.assertEqual(len(REFERENCE_ONLY), 2)
        for variant_id in REFERENCE_ONLY:
            adapter = self.bundle.adapters_by_variant[variant_id]
            self.assertEqual(adapter["execution_kind"], "reference_only")
            profiles = [
                profile for profile in self.bundle.profiles_by_id.values()
                if variant_id in profile["stages"].values()
            ]
            self.assertTrue(profiles)
            self.assertTrue(all(profile["kind"] == "reference" for profile in profiles))

    def test_no_implicit_alias(self) -> None:
        self.assertFalse(self.bundle.profiles["implicit_default"])
        self.assertFalse({"default", "canonical", "latest", "release"} & set(self.bundle.profiles_by_id))


if __name__ == "__main__":
    unittest.main()
