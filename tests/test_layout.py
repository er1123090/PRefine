from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from experiments7_layout import catalog, cli


class LayoutCatalogTests(unittest.TestCase):
    def _stub_documents(self) -> dict[str, dict[str, str]]:
        return {path: {"schema": path} for path in catalog.CATALOG_PATHS}

    def _make_navigation_root(self, root: Path) -> None:
        for directory in catalog.NAVIGATION_FILES:
            target = root / directory
            target.mkdir()
            for child in catalog.NAVIGATION_DIRECTORIES[directory]:
                (target / child).mkdir()

    def _write_stub_catalogs(self, root: Path) -> dict[str, dict[str, str]]:
        documents = self._stub_documents()
        self._make_navigation_root(root)
        for relative_path, document in documents.items():
            (root / relative_path).write_bytes(catalog._json_bytes(document))
        return documents

    def test_committed_layout_is_current_and_valid(self) -> None:
        catalog.write_catalogs(REPO_ROOT, check=True)
        report = catalog.validate_layout(REPO_ROOT)
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["counts"], catalog.COUNT_CONTRACT)

    def test_catalog_build_is_deterministic(self) -> None:
        first = catalog.build_catalogs(REPO_ROOT)
        second = catalog.build_catalogs(REPO_ROOT)
        self.assertEqual(first, second)

    def test_audited_labels_and_source_origins_are_preserved(self) -> None:
        index = json.loads((REPO_ROOT / "experiments/index.json").read_text(encoding="utf-8"))
        registry = json.loads((REPO_ROOT / catalog.AUDITED_REGISTRY).read_text(encoding="utf-8"))
        expected = {
            variant["variant_id"]: {
                evidence["origin_root"] for evidence in variant["origin_evidence"]
            }
            for variant in registry["variants"]
        }
        actual = {
            entry["semantic_label"]: set(entry["source_origins"])
            for entry in index["entries"]
        }
        self.assertEqual(index["entry_count"], 84)
        self.assertEqual(len(actual), 84)
        self.assertEqual(actual, expected)
        for label in actual:
            self.assertTrue(label)
            tokens = set(re.findall(r"[a-z0-9]+", label.lower()))
            self.assertTrue({"eval6", "infer6"}.isdisjoint(tokens))

    def test_public_paper_navigation_has_only_final_and_raw(self) -> None:
        index = json.loads(
            (REPO_ROOT / "outputs/paper_outputs_index.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            {entry["canonical_path"] for entry in index["entries"]},
            {"paper_outputs/final", "paper_outputs/raw"},
        )

    def test_canonical_path_rejects_traversal_absolute_and_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(root_tmp)
            (root / "safe.txt").write_text("safe\n", encoding="utf-8")
            outside = Path(outside_tmp)
            (outside / "outside.txt").write_text("outside\n", encoding="utf-8")
            (root / "escape").symlink_to(outside, target_is_directory=True)
            self.assertEqual(catalog.validate_canonical_path(root, "safe.txt"), root / "safe.txt")
            for value in ("../escape", "/absolute", "safe/../safe.txt", "./safe.txt", "escape/outside.txt"):
                with self.subTest(value=value), self.assertRaises(catalog.LayoutError):
                    catalog.validate_canonical_path(root, value)

    def test_navigation_rejects_undeclared_and_symlink_files(self) -> None:
        for hostile in ("undeclared", "symlink"):
            with self.subTest(hostile=hostile), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for directory, names in catalog.NAVIGATION_FILES.items():
                    target = root / directory
                    target.mkdir()
                    for child in catalog.NAVIGATION_DIRECTORIES[directory]:
                        (target / child).mkdir()
                    for name in names:
                        (target / name).write_text("{}\n", encoding="utf-8")
                (root / "layout_manifest.json").write_text("{}\n", encoding="utf-8")
                if hostile == "undeclared":
                    (root / "data/extra.txt").write_text("hostile\n", encoding="utf-8")
                else:
                    (root / "methods/methods_index.json").unlink()
                    (root / "methods/methods_index.json").symlink_to(root / "data/datasets_index.json")
                with self.assertRaises(catalog.LayoutError):
                    catalog._validate_navigation_files(root)

    def test_write_rejects_symlinked_category_parent_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(root_tmp)
            outside = Path(outside_tmp)
            sentinel = outside / "sentinel.txt"
            sentinel.write_text("unchanged\n", encoding="utf-8")
            for directory in catalog.NAVIGATION_FILES:
                if directory != "results":
                    (root / directory).mkdir()
            (root / "results").symlink_to(outside, target_is_directory=True)

            with mock.patch.object(catalog, "build_catalogs", return_value=self._stub_documents()):
                with self.assertRaises(catalog.LayoutError):
                    catalog.write_catalogs(root)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged\n")
            self.assertFalse((outside / "results_index.json").exists())
            self.assertFalse((root / "data/datasets_index.json").exists())

    def test_write_rejects_dangling_destination_symlink_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(root_tmp)
            outside = Path(outside_tmp)
            sentinel = outside / "sentinel.txt"
            sentinel.write_text("unchanged\n", encoding="utf-8")
            for directory in catalog.NAVIGATION_FILES:
                (root / directory).mkdir()
            outside_target = outside / "escaped-results-index.json"
            (root / "results/results_index.json").symlink_to(outside_target)

            with mock.patch.object(catalog, "build_catalogs", return_value=self._stub_documents()):
                with self.assertRaises(catalog.LayoutError):
                    catalog.write_catalogs(root)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged\n")
            self.assertFalse(outside_target.exists())
            self.assertFalse((root / "data/datasets_index.json").exists())

    def test_write_commits_exact_bytes_without_temp_residue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_navigation_root(root)
            documents = self._stub_documents()

            with (
                mock.patch.object(catalog, "build_catalogs", return_value=documents),
                mock.patch.object(catalog.os, "fsync", wraps=os.fsync) as fsync,
            ):
                catalog.write_catalogs(root)

            for relative_path, document in documents.items():
                self.assertEqual(
                    (root / relative_path).read_bytes(),
                    catalog._json_bytes(document),
                )
            self.assertFalse(
                any(".tmp-" in path.name for path in root.rglob("*"))
            )
            self.assertEqual(fsync.call_count, 2 * len(catalog.CATALOG_PATHS))

    def test_write_rejects_hardlinked_destination_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(root_tmp)
            outside = Path(outside_tmp)
            self._make_navigation_root(root)
            sentinel = outside / "sentinel.txt"
            sentinel.write_text("unchanged\n", encoding="utf-8")
            os.link(sentinel, root / "results/results_index.json")

            with mock.patch.object(catalog, "build_catalogs", return_value=self._stub_documents()):
                with self.assertRaisesRegex(catalog.LayoutError, "unique regular file"):
                    catalog.write_catalogs(root)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged\n")
            self.assertFalse((root / "data/datasets_index.json").exists())

    def test_post_preflight_parent_symlink_swap_cannot_escape(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(root_tmp)
            outside = Path(outside_tmp)
            self._make_navigation_root(root)
            sentinel = outside / "sentinel.txt"
            sentinel.write_text("unchanged\n", encoding="utf-8")
            original_write_temp = catalog._write_catalog_temp
            swapped = False

            def swap_parent(parent_fd: int, name: str, payload: bytes):
                nonlocal swapped
                temp = original_write_temp(parent_fd, name, payload)
                if not swapped:
                    (root / "results").rename(root / "results-original")
                    (root / "results").symlink_to(outside, target_is_directory=True)
                    swapped = True
                return temp

            with (
                mock.patch.object(catalog, "build_catalogs", return_value=self._stub_documents()),
                mock.patch.object(catalog, "_write_catalog_temp", side_effect=swap_parent),
                self.assertRaises(catalog.LayoutError),
            ):
                catalog.write_catalogs(root)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged\n")
            self.assertFalse((outside / "results_index.json").exists())
            self.assertFalse((root / "data/datasets_index.json").exists())

    def test_post_preflight_destination_symlink_swap_cannot_escape(self) -> None:
        with tempfile.TemporaryDirectory() as root_tmp, tempfile.TemporaryDirectory() as outside_tmp:
            root = Path(root_tmp)
            outside = Path(outside_tmp)
            self._make_navigation_root(root)
            sentinel = outside / "sentinel.txt"
            sentinel.write_text("unchanged\n", encoding="utf-8")
            outside_target = outside / "escaped-results-index.json"
            original_write_temp = catalog._write_catalog_temp
            swapped = False

            def swap_destination(parent_fd: int, name: str, payload: bytes):
                nonlocal swapped
                temp = original_write_temp(parent_fd, name, payload)
                if not swapped:
                    (root / "results/results_index.json").symlink_to(outside_target)
                    swapped = True
                return temp

            with (
                mock.patch.object(catalog, "build_catalogs", return_value=self._stub_documents()),
                mock.patch.object(
                    catalog,
                    "_write_catalog_temp",
                    side_effect=swap_destination,
                ),
                self.assertRaises(catalog.LayoutError),
            ):
                catalog.write_catalogs(root)

            self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged\n")
            self.assertFalse(outside_target.exists())
            self.assertFalse((root / "data/datasets_index.json").exists())

    def test_check_rejects_destination_symlink_and_hardlink(self) -> None:
        for hostile in ("symlink", "hardlink"):
            with (
                self.subTest(hostile=hostile),
                tempfile.TemporaryDirectory() as root_tmp,
                tempfile.TemporaryDirectory() as outside_tmp,
            ):
                root = Path(root_tmp)
                outside = Path(outside_tmp)
                documents = self._write_stub_catalogs(root)
                destination = root / "results/results_index.json"
                destination.unlink()
                sentinel = outside / "sentinel.txt"
                sentinel.write_text("unchanged\n", encoding="utf-8")
                if hostile == "symlink":
                    destination.symlink_to(sentinel)
                else:
                    os.link(sentinel, destination)

                with mock.patch.object(catalog, "build_catalogs", return_value=documents):
                    with self.assertRaisesRegex(catalog.LayoutError, "unique regular file"):
                        catalog.write_catalogs(root, check=True)

                self.assertEqual(sentinel.read_text(encoding="utf-8"), "unchanged\n")

    def test_root_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            real_root = base / "real-root"
            real_root.mkdir()
            self._make_navigation_root(real_root)
            root = base / "root-link"
            root.symlink_to(real_root, target_is_directory=True)

            with mock.patch.object(catalog, "build_catalogs", return_value=self._stub_documents()):
                with self.assertRaisesRegex(catalog.LayoutError, "root is not a real directory"):
                    catalog.write_catalogs(root)

    def test_cli_passes_unresolved_root_to_writer_and_validator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            real_root = base / "real-root"
            real_root.mkdir()
            root = base / "root-link"
            root.symlink_to(real_root, target_is_directory=True)

            with (
                mock.patch.object(cli, "write_catalogs") as writer,
                mock.patch.object(cli, "_emit"),
            ):
                self.assertEqual(
                    cli.main(["--root", str(root), "build-indexes", "--check"]),
                    0,
                )
            writer.assert_called_once_with(root, check=True)

            with (
                mock.patch.object(
                    cli,
                    "validate_layout",
                    return_value={"status": "ok"},
                ) as validator,
                mock.patch.object(cli, "_emit"),
            ):
                self.assertEqual(cli.main(["--root", str(root), "validate"]), 0)
            validator.assert_called_once_with(root)

    def test_validate_layout_rejects_symlink_root_before_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            real_root = base / "real-root"
            real_root.mkdir()
            root = base / "root-link"
            root.symlink_to(real_root, target_is_directory=True)

            with (
                mock.patch.object(catalog, "write_catalogs") as writer,
                self.assertRaisesRegex(
                    catalog.LayoutError,
                    "root is not a real directory",
                ),
            ):
                catalog.validate_layout(root)
            writer.assert_not_called()

    def test_real_catalog_root_rejects_post_lstat_root_swap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root = base / "root"
            root.mkdir()
            attacker = base / "attacker"
            attacker.mkdir()
            original = base / "original"
            real_lstat = os.lstat
            swapped = False

            def swap_after_lstat(path: os.PathLike[str] | str):
                nonlocal swapped
                result = real_lstat(path)
                if Path(path) == root and not swapped:
                    root.rename(original)
                    attacker.rename(root)
                    swapped = True
                return result

            with (
                mock.patch.object(
                    catalog.os,
                    "lstat",
                    side_effect=swap_after_lstat,
                ),
                self.assertRaisesRegex(catalog.LayoutError, "root binding changed"),
            ):
                catalog._real_catalog_root(root)

    def test_preflight_closes_root_and_parent_when_parent_fstat_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_navigation_root(root)
            root, root_stat = catalog._real_catalog_root(root)
            real_fstat = os.fstat
            fstat_calls = 0

            def fail_parent_fstat(file_fd: int):
                nonlocal fstat_calls
                fstat_calls += 1
                if fstat_calls == 2:
                    raise OSError("injected parent fstat failure")
                return real_fstat(file_fd)

            with (
                mock.patch.object(
                    catalog.os,
                    "fstat",
                    side_effect=fail_parent_fstat,
                ),
                mock.patch.object(catalog.os, "close", wraps=os.close) as close,
                self.assertRaisesRegex(catalog.LayoutError, "catalog parent is unsafe"),
            ):
                catalog._preflight_catalog_destinations(
                    root,
                    root_stat,
                    require_existing=False,
                )
            self.assertGreaterEqual(close.call_count, 2)

    def test_cleanup_failure_does_not_mask_primary_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_navigation_root(root)

            with (
                mock.patch.object(
                    catalog,
                    "build_catalogs",
                    return_value=self._stub_documents(),
                ),
                mock.patch.object(
                    catalog,
                    "_verify_temp_binding",
                    side_effect=catalog.LayoutError("primary verification failure"),
                ),
                mock.patch.object(
                    catalog.os,
                    "unlink",
                    side_effect=OSError("injected cleanup failure"),
                ),
                self.assertRaisesRegex(
                    catalog.LayoutError,
                    "primary verification failure",
                ),
            ):
                catalog.write_catalogs(root)

    def test_cli_works_from_an_external_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            environment = dict(os.environ)
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            result = subprocess.run(
                [sys.executable, "-B", str(REPO_ROOT / "scripts/layout.py"), "validate"],
                cwd=tmp,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "ok")


if __name__ == "__main__":
    unittest.main()
