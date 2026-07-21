from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "exp7-strict-v6-20260718T120001Z-" + "2" * 32


class StrictRunV6G0BootstrapCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="exp7-g0-bootstrap-cli-", dir=ROOT
        )
        self.base = Path(self.temporary.name)
        self.strict_parent = self.base / "paper_outputs" / "strict-runs"
        self.strict_parent.mkdir(parents=True)
        self.run_root = self.strict_parent / RUN_ID
        self.external = self.base / "external"
        self.external.mkdir()
        self.sources: dict[str, Path] = {}
        for root_id in ("experiments4", "experiments5", "experiments6"):
            root = self.base / root_id
            root.mkdir()
            (root / "payload.txt").write_text(f"{root_id}\n", encoding="utf-8")
            self.sources[root_id] = root
        paper_dir = self.base / "_paper"
        paper_dir.mkdir()
        self.paper = paper_dir / "paper.pdf"
        self.paper.write_bytes(b"%PDF-disposable-g0-cli\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_cli_seals_cp0_with_disposable_inputs(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-B",
                str(ROOT / "scripts" / "strict_run" / "bootstrap_g0_cp0.py"),
                "--strict-parent", str(self.strict_parent),
                "--sealed-run-id", RUN_ID,
                "--run-root", str(self.run_root),
                "--external-dir", str(self.external),
                "--project-id", "experiments7-g0-bootstrap-cli-test",
                "--owner-seed", "g0-bootstrap-cli-owner",
                "--experiments4-root", str(self.sources["experiments4"]),
                "--experiments5-root", str(self.sources["experiments5"]),
                "--experiments6-root", str(self.sources["experiments6"]),
                "--paper-path", str(self.paper),
                "--timeout-seconds", "60",
            ],
            cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=90,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        summary = json.loads(completed.stdout)
        self.assertEqual(summary["sealed_run_id"], RUN_ID)
        self.assertEqual(summary["run_root"], str(self.run_root))
        self.assertEqual(summary["checkpoint"], 0)
        self.assertTrue((self.run_root / "checkpoints" / "cp0.json").is_file())
        self.assertTrue(
            all(Path(path).is_file() for path in summary["external_paths"].values())
        )


if __name__ == "__main__":
    unittest.main()
