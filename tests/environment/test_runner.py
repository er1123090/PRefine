from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiment_env.runner import RunError, _validate_output_paths
from experiment_env.util import redact_argv


class RunnerTests(unittest.TestCase):
    def test_output_paths_must_stay_in_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            run_dir = root / "runs" / "one"
            snapshot.mkdir()
            run_dir.mkdir(parents=True)
            recipe = {"output_flags": ["--output_path"], "required_output_flags": ["--output_path"]}
            _validate_output_paths(["tool", "--output_path", str(run_dir / "ok.json")], recipe, snapshot, run_dir)
            with self.assertRaises(RunError):
                _validate_output_paths(["tool", "--output_path", str(root / "escape.json")], recipe, snapshot, run_dir)

    def test_undeclared_output_like_flag_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            snapshot = root / "snapshot"
            run_dir = root / "runs" / "one"
            snapshot.mkdir()
            run_dir.mkdir(parents=True)
            with self.assertRaises(RunError):
                _validate_output_paths(["tool", "--save_path", str(run_dir / "x")], {"output_flags": []}, snapshot, run_dir)

    def test_secret_arguments_are_redacted(self) -> None:
        self.assertEqual(
            redact_argv(["tool", "--api-key", "secret", "--token=secret", "--model", "ok"]),
            ["tool", "--api-key", "<redacted>", "--token=<redacted>", "--model", "ok"],
        )


if __name__ == "__main__":
    unittest.main()
