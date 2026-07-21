from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class PrepareCliTests(unittest.TestCase):
    def test_cli_materializes_six_verified_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "prepared"
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/prepare.py"),
                    "--output-dir",
                    str(output),
                ],
                cwd=temporary,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            validation = json.loads(
                (output / "validation_report.json").read_text(encoding="utf-8")
            )

            self.assertEqual(manifest["summary"]["total_instances"], 2638)
            self.assertEqual(len(manifest["outputs"]), 6)
            self.assertEqual(validation["schema_warnings"]["total"], 463)
            self.assertEqual(
                validation["multiturn_base_preference_conflicts"]["conflict_count"],
                0,
            )
            for output_spec in manifest["outputs"].values():
                self.assertTrue((output / output_spec["path"]).is_file())
            self.assertIn("builder", manifest)
            self.assertIn("config", manifest)
            self.assertEqual(set(manifest["inputs"]), {
                "mix600",
                "pref_group",
                "pref_list",
                "query_multiturn",
                "query_singleturn",
                "schema_multiturn",
                "schema_singleturn",
            })


if __name__ == "__main__":
    unittest.main()
