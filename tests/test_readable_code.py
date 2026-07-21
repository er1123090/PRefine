from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments7_layout import catalog, readable_code


class ReadableCodeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.plan = readable_code.build_readable_plan(REPO_ROOT)

    def test_plan_covers_all_audited_sources_and_variants(self) -> None:
        self.assertEqual(self.plan["entry_count"], 443)
        self.assertEqual(self.plan["variant_count"], 84)
        self.assertEqual(
            self.plan["source_origin_counts"],
            {"experiments4": 83, "experiments5": 72, "experiments6": 288},
        )
        self.assertEqual(
            self.plan["category_counts"],
            {"data": 41, "experiment": 97, "method": 305},
        )
        self.assertEqual(
            self.plan["source_kind_counts"],
            {"original_source": 47, "sealed_base": 288, "sealed_overlay": 108},
        )
        self.assertEqual(
            self.plan["experiments6_inventory"],
            {"regular_file_count": 288, "symlink_records_not_materialized": 26},
        )

    def test_every_target_has_an_explicit_version_and_unique_mapping(self) -> None:
        targets: set[str] = set()
        source_keys: set[tuple[str, str]] = set()
        version_counts: Counter[str] = Counter()
        for entry in self.plan["entries"]:
            target = entry["target_path"]
            key = (entry["source_origin"], entry["source_relative_path"])
            self.assertNotIn(target, targets)
            self.assertNotIn(key, source_keys)
            targets.add(target)
            source_keys.add(key)
            version = entry["version_label"]
            version_counts[version] += 1
            self.assertIn(version, {"experiments4", "experiments5", "experiments6"})
            self.assertIn(version, Path(target).parts)
            self.assertNotIn("eval6", Path(target).parts)
            self.assertNotIn("infer6", Path(target).parts)
        self.assertEqual(
            version_counts,
            Counter({"experiments6": 288, "experiments4": 83, "experiments5": 72}),
        )

    def test_committed_copies_are_unique_regular_files_with_audited_hashes(self) -> None:
        readable_code.sync_readable_code(REPO_ROOT, check=True)
        for entry in self.plan["entries"]:
            path = REPO_ROOT / entry["target_path"]
            value = os.lstat(path)
            self.assertTrue(stat.S_ISREG(value.st_mode), path)
            self.assertEqual(value.st_nlink, 1, path)
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                entry["audited_sha256"],
                path,
            )

    def test_variant_documents_point_only_to_actual_planned_code(self) -> None:
        planned_targets = {entry["target_path"] for entry in self.plan["entries"]}
        documents = self.plan["variant_documents"]
        self.assertEqual(len(documents), 84)
        for relative_path, expected in documents.items():
            path = REPO_ROOT / relative_path
            self.assertTrue(path.is_file() and not path.is_symlink(), path)
            actual = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(actual, expected)
            self.assertTrue(actual["code_paths"])
            self.assertLessEqual(set(actual["code_paths"]), planned_targets)

    def test_catalog_variants_use_the_same_actual_code_paths(self) -> None:
        documents = catalog.build_catalogs(REPO_ROOT)
        experiment_index = documents["experiments/index.json"]
        by_label = {
            document["semantic_label"]: document
            for document in self.plan["variant_documents"].values()
        }
        self.assertEqual(experiment_index["entry_count"], 84)
        for entry in experiment_index["entries"]:
            expected = by_label[entry["semantic_label"]]
            self.assertEqual(entry["code_paths"], expected["code_paths"])
            self.assertEqual(entry["canonical_path"], expected["code_paths"][0])

    def test_atomic_write_rejects_symlinked_parent(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(root_tmp)
            outside = Path(outside_tmp)
            (root / "methods").mkdir()
            (root / "methods/rag").symlink_to(outside, target_is_directory=True)
            with self.assertRaises(readable_code.ReadableCodeError):
                readable_code._atomic_write(
                    root,
                    "methods/rag/experiments4/inference.py",
                    b"unsafe\n",
                    0o644,
                )
            self.assertFalse((outside / "experiments4/inference.py").exists())

    def test_atomic_write_rejects_hardlinked_destination(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(root_tmp)
            outside = Path(outside_tmp)
            destination = root / "methods/rag/experiments4/inference.py"
            destination.parent.mkdir(parents=True)
            sentinel = outside / "sentinel.py"
            sentinel.write_text("unchanged\n", encoding="utf-8")
            os.link(sentinel, destination)
            with self.assertRaisesRegex(
                readable_code.ReadableCodeError,
                "unique regular file",
            ):
                readable_code._atomic_write(
                    root,
                    "methods/rag/experiments4/inference.py",
                    b"changed\n",
                    0o644,
                )
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged\n")

    def test_root_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            real_root = base / "real"
            real_root.mkdir()
            root = base / "link"
            root.symlink_to(real_root, target_is_directory=True)
            with self.assertRaisesRegex(
                readable_code.ReadableCodeError,
                "root is not a real directory",
            ):
                readable_code._real_root(root)

    def test_managed_files_rejects_dangling_version_root_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            family = root / "methods/emem"
            family.mkdir(parents=True)
            (family / "experiments6").symlink_to(
                root / "missing",
                target_is_directory=True,
            )

            with self.assertRaisesRegex(
                readable_code.ReadableCodeError,
                "managed code root is unsafe",
            ):
                readable_code._managed_files(root)

    def test_new_tree_write_failure_preserves_legacy_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.py"
            source.write_text("new\n", encoding="utf-8")
            legacy = root / "methods/rag/eval4/old.py"
            legacy.parent.mkdir(parents=True)
            legacy.write_text("old\n", encoding="utf-8")
            (root / readable_code.MANIFEST_PATH).write_text(
                json.dumps(
                    {
                        "entries": [{"target_path": "methods/rag/eval4/old.py"}],
                        "variant_count": 0,
                    }
                ),
                encoding="utf-8",
            )
            payload = source.read_bytes()
            plan = {
                "entries": [
                    {
                        "audited_sha256": hashlib.sha256(payload).hexdigest(),
                        "mode": "0644",
                        "source_path": str(source),
                        "target_path": "methods/rag/experiments4/new.py",
                    }
                ],
                "variant_documents": {},
            }

            with (
                mock.patch.object(readable_code, "build_readable_plan", return_value=plan),
                mock.patch.object(
                    readable_code,
                    "_atomic_write",
                    side_effect=readable_code.ReadableCodeError("injected write failure"),
                ),
                self.assertRaisesRegex(
                    readable_code.ReadableCodeError,
                    "injected write failure",
                ),
            ):
                readable_code.sync_readable_code(root)

            self.assertEqual(legacy.read_text(encoding="utf-8"), "old\n")


if __name__ == "__main__":
    unittest.main()
