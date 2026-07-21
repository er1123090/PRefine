from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facade.planner import inspect_variant, profile_selection  # noqa: E402
from facade.registry import FacadeError, load_bundle  # noqa: E402


class FacadeCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(ROOT)
        cls.command = [sys.executable, "-B", str(ROOT / "scripts/runtime/facade.py")]

    def run_cli(self, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [*self.command, *args], cwd=ROOT, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
        )

    def test_help_for_every_command(self) -> None:
        self.assertEqual(self.run_cli("--help").returncode, 0)
        for command in ("list", "inspect", "validate", "plan", "run", "compare-reference"):
            result = self.run_cli(command, "--help")
            self.assertEqual(result.returncode, 0, (command, result.stderr))

    def test_list_exact_counts(self) -> None:
        result = self.run_cli("list", "--json")
        self.assertEqual(result.returncode, 0, result.stdout)
        value = json.loads(result.stdout)
        self.assertEqual((value["family_count"], value["variant_count"]), (16, 85))

    def test_every_variant_inspects_and_family_fails(self) -> None:
        for variant_id in self.bundle.variants:
            self.assertEqual(inspect_variant(self.bundle, variant_id)["variant"]["variant_id"], variant_id)
        with self.assertRaises(FacadeError) as caught:
            inspect_variant(self.bundle, "family.inference")
        self.assertEqual(caught.exception.code, "FAMILY_IS_NOT_SELECTABLE")

    def test_every_profile_validates(self) -> None:
        for profile_id in self.bundle.profiles_by_id:
            selection = profile_selection(
                self.bundle, profile_id, execution="validate", run_id=None, output_root=None
            )
            repeated = profile_selection(
                self.bundle, profile_id, execution="validate", run_id=None, output_root=None
            )
            self.assertEqual(selection, repeated)
            self.assertEqual(selection["profile_id"], profile_id)
            self.assertEqual(len(selection["control_hashes"]), 6)

    def test_unknown_profile_fails_closed(self) -> None:
        result = self.run_cli("validate", "--profile-id", "missing_profile", "--json")
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stdout)["reason_codes"], ["UNKNOWN_PROFILE"])


if __name__ == "__main__":
    unittest.main()
