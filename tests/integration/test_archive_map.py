from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exp7.provenance import archive_map
from exp7.provenance.archive_map import ArchiveMapError


EXPECTED_SCOPE = (
    "data/sources",
    "methods",
    "experiments",
    "environments",
    "lineage",
    "manifests",
    "variants",
    "paper_outputs/final",
    "paper_outputs/raw",
    "paper_outputs/admission",
    "paper_outputs/inventory",
    "paper_outputs/provenance",
    "paper_outputs/strict-runs",
    "outputs",
    "results",
    "runs",
)

EXPECTED_INVENTORY = {
    "data/sources": (41, 79_729_488, 0),
    "methods": (307, 3_599_879, 0),
    "experiments": (183, 1_096_628, 0),
    "environments": (517, 71_220_961, 26),
    "lineage": (2, 92_219, 0),
    "manifests": (71, 34_370_859, 0),
    "variants": (5, 218_624, 0),
    "paper_outputs/final": (7, 6_879_419, 0),
    "paper_outputs/raw": (523, 7_686_997_692, 0),
    "paper_outputs/admission": (7, 2_926_322, 0),
    "paper_outputs/inventory": (9, 2_115_859, 0),
    "paper_outputs/provenance": (4, 7_333_133, 0),
    "paper_outputs/strict-runs": (47, 31_790_752, 0),
    "outputs": (3, 3_221, 0),
    "results": (2, 1_462, 0),
    "runs": (46, 15_969_600, 0),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ArchiveMapTests(unittest.TestCase):
    def test_checked_in_map_is_deterministic_current_and_complete(self) -> None:
        first = archive_map.build_archive_map(ROOT)
        second = archive_map.build_archive_map(ROOT)
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], "experiments7-archive-map/v1")
        self.assertIs(first["planned_only"], True)
        self.assertIs(first["no_mutations"], True)
        self.assertEqual(first["record_count"], len(EXPECTED_SCOPE))
        self.assertEqual(
            (ROOT / "archive/archive-map.json").read_bytes(),
            archive_map.json_bytes(first),
        )
        self.assertEqual(
            (ROOT / "archive/archive-map.md").read_bytes(),
            archive_map.render_markdown(first),
        )

    def test_scope_inventory_dispositions_and_raw_gate(self) -> None:
        records = {
            row["path"]: row
            for row in archive_map.build_archive_map(ROOT)["records"]
        }
        self.assertEqual(archive_map.target_paths(), EXPECTED_SCOPE)
        self.assertEqual(tuple(records), EXPECTED_SCOPE)
        for path, expected in EXPECTED_INVENTORY.items():
            row = records[path]
            self.assertEqual(
                (row["file_count"], row["bytes"], row["symlink_count"]),
                expected,
                path,
            )
            self.assertTrue(row["reason"])
            self.assertTrue(row["cutover_prerequisites"])
            self.assertTrue(row["recovery_pointers"])

        for path in ("data/sources", "methods", "experiments"):
            self.assertEqual(records[path]["disposition"], "archive_after_cutover")
        self.assertEqual(records["environments"]["disposition"], "keep_active_audit")
        self.assertEqual(records["outputs"]["disposition"], "generated_rebuildable")
        self.assertEqual(records["results"]["disposition"], "generated_rebuildable")
        self.assertEqual(records["runs"]["disposition"], "split_before_cleanup")

        raw = records["paper_outputs/raw"]
        self.assertIs(raw["externalization_allowed"], False)
        self.assertEqual(
            raw["payload_coverage"],
            {
                "bytes": 7_686_987_770,
                "file_count": 521,
                "metadata_bytes": 9_922,
                "metadata_file_count": 2,
            },
        )
        self.assertIn("restore transcript", raw["warning"])
        self.assertIn("bulk-delete", records["runs"]["warning"])

    def test_every_evidence_path_exists_and_hash_matches(self) -> None:
        result = json.loads(
            (ROOT / "archive/archive-map.json").read_text(encoding="utf-8")
        )
        for record in result["records"]:
            self.assertTrue(record["evidence"], record["path"])
            for evidence in record["evidence"]:
                path = ROOT / evidence["path"]
                self.assertTrue(path.is_file(), evidence["path"])
                self.assertFalse(path.is_symlink(), evidence["path"])
                self.assertEqual(_sha256(path), evidence["sha256"])
                self.assertTrue(evidence["coverage"]["type"])
                self.assertTrue(evidence["coverage"]["limits"])

    def test_check_cli_is_read_only(self) -> None:
        tracked = [
            ROOT / "archive/archive-map.json",
            ROOT / "archive/archive-map.md",
            *(ROOT / path for path in EXPECTED_SCOPE),
        ]
        before = {
            str(path): (
                path.lstat().st_mtime_ns,
                path.lstat().st_size,
                path.lstat().st_mode,
            )
            for path in tracked
        }
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/archive_map.py"), "--check"],
            cwd=ROOT,
            env={
                "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["mode"], "check")
        after = {
            str(path): (
                path.lstat().st_mtime_ns,
                path.lstat().st_size,
                path.lstat().st_mode,
            )
            for path in tracked
        }
        self.assertEqual(before, after)

    def test_check_detects_drift_without_writing(self) -> None:
        value = archive_map.build_archive_map(ROOT)
        with tempfile.TemporaryDirectory() as temporary:
            json_path = Path(temporary) / "archive-map.json"
            markdown_path = Path(temporary) / "archive-map.md"
            json_path.write_text("{}\n", encoding="utf-8")
            markdown_path.write_bytes(archive_map.render_markdown(value))

            def output(_root: Path, relative: str) -> Path:
                return json_path if relative == archive_map.DEFAULT_OUTPUT else markdown_path

            with mock.patch.object(archive_map, "_output", side_effect=output):
                with self.assertRaisesRegex(ArchiveMapError, "missing or stale"):
                    archive_map.check_archive_map(ROOT)

    def test_atomic_writer_replaces_complete_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "map.json"
            path.write_bytes(b"old")
            archive_map._write(path, b"new\n")
            self.assertEqual(path.read_bytes(), b"new\n")
            self.assertEqual([item.name for item in path.parent.iterdir()], ["map.json"])

    def test_file_symlink_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repository"
            outside = Path(temporary) / "outside.json"
            (root / "data/sources").mkdir(parents=True)
            outside.write_text("outside\n", encoding="utf-8")
            os.symlink(outside, root / "data/sources/escape.json")

            with self.assertRaisesRegex(ArchiveMapError, "symlink escapes repository"):
                archive_map.inventory_tree(root, "data/sources")

    def test_symlink_root_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repository"
            outside = Path(temporary) / "outside"
            (root / "data").mkdir(parents=True)
            outside.mkdir()
            os.symlink(outside, root / "data/sources", target_is_directory=True)
            with self.assertRaisesRegex(ArchiveMapError, "may not be a symlink"):
                archive_map.inventory_tree(root, "data/sources")


if __name__ == "__main__":
    unittest.main()
