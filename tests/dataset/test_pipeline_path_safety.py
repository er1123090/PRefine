from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exp7.datasets.pipeline import (
    DatasetPreparationError,
    _load_json,
    _output_path,
    _validate_new_output_target,
    _validate_replace_target,
)
from exp7.provenance import admission


class PreparedOutputPathSafetyTests(unittest.TestCase):
    def test_new_output_rejects_symlink_parent_without_resolving_it_away(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(root_tmp)
            link = root / "linked-parent"
            link.symlink_to(Path(outside_tmp), target_is_directory=True)
            output = _output_path(root, "linked-parent/prepared")
            self.assertEqual(output, link / "prepared")
            with self.assertRaisesRegex(DatasetPreparationError, "real directory|symlink"):
                _validate_new_output_target(output)

    def test_replace_rejects_symlink_and_unidentified_directory(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(root_tmp)
            outside = Path(outside_tmp)
            link = root / "prepared-link"
            link.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(DatasetPreparationError, "real directory"):
                _validate_replace_target(link, "mix600-v1")

            unrelated = root / "unrelated"
            unrelated.mkdir()
            with self.assertRaisesRegex(DatasetPreparationError, "manifest"):
                _validate_replace_target(unrelated, "mix600-v1")

    def test_json_input_rejects_symlink_before_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "input.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(DatasetPreparationError, "non-symlink"):
                _load_json(link, label="test input")

    def test_json_input_rejects_mutation_during_descriptor_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.json"
            path.write_text('{"before": true}', encoding="utf-8")
            real_read = admission.os.read
            mutated = False

            def mutate_on_descriptor_read(descriptor, size):
                nonlocal mutated
                if not mutated:
                    path.write_text('{"after": true}', encoding="utf-8")
                    mutated = True
                return real_read(descriptor, size)

            with mock.patch.object(
                admission.os, "read", side_effect=mutate_on_descriptor_read
            ):
                with self.assertRaisesRegex(
                    DatasetPreparationError, "changed while being read"
                ):
                    _load_json(path, label="test input")


if __name__ == "__main__":
    unittest.main()
