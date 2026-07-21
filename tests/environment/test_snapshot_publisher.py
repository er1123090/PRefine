from __future__ import annotations

import base64
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiment_env.snapshot import PublicationError, publish_snapshot, validate_snapshot


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


class SnapshotPublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "environments").mkdir()
        (self.root / "source" / "src").mkdir(parents=True)
        (self.root / "source" / "artifacts").mkdir()
        source_file = self.root / "source" / "src" / "run.py"
        source_file.write_text("print('ok')\n", encoding="utf-8")
        excluded = self.root / "source" / "artifacts" / "output.json"
        excluded.write_text("{}\n", encoding="utf-8")
        os.symlink("src", self.root / "source" / "legacy")
        os.symlink("artifacts/output.json", self.root / "source" / "bad-link")
        records = []
        for path in (source_file, excluded):
            info = path.stat()
            relative = path.relative_to(self.root / "source").as_posix()
            data = path.read_bytes()
            records.append(
                {
                    "descriptor_identity": {
                        "file_type": "regular",
                        "mode": info.st_mode & 0o777,
                        "mtime_ns": info.st_mtime_ns,
                        "size": info.st_size,
                        "st_dev": info.st_dev,
                        "st_ino": info.st_ino,
                    },
                    "record_id": "source:" + hashlib.sha256(relative.encode()).hexdigest(),
                    "relative_path_b64": base64.b64encode(relative.encode()).decode(),
                    "root_id": "experiments6",
                    "schema": "experiments7-source-pre/v6",
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                    "type": "regular",
                }
            )
        source_pre = self.root / "source-pre.jsonl"
        source_pre.write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8")
        cp0 = self.root / "cp0.json"
        dump(cp0, {"checkpoint": 0, "semantic_bindings": {"source_pre_record_count": 2}})
        recipes = self.root / "recipes.json"
        dump(recipes, {"profile_id": "test_eval6", "recipes": {}, "schema": "experiments7-environment-recipes/v1"})
        source_pre_sha = hashlib.sha256(source_pre.read_bytes()).hexdigest()
        cp0_sha = hashlib.sha256(cp0.read_bytes()).hexdigest()
        self.spec = self.root / "spec.json"
        dump(
            self.spec,
            {
                "cp0_path": "cp0.json",
                "destination": "environments/test-v1",
                "exclude": {"components": [], "prefixes": ["artifacts"], "suffixes": []},
                "expected_cp0_sha256": cp0_sha,
                "expected_excluded_record_count": 1,
                "expected_regular_bytes": len(source_file.read_bytes()),
                "expected_regular_file_count": 1,
                "expected_root_record_count": 2,
                "expected_safe_symlink_count": 1,
                "expected_skipped_symlink_count": 1,
                "expected_source_pre_record_count": 2,
                "expected_source_pre_sha256": source_pre_sha,
                "profile_id": "test_eval6",
                "recipes_path": "recipes.json",
                "schema": "experiments7-environment-publication/v1",
                "source_pre_path": "source-pre.jsonl",
                "source_root": "source",
                "source_root_id": "experiments6",
            },
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_publication_and_validation(self) -> None:
        result = publish_snapshot(self.root, self.spec)
        snapshot = Path(result["destination"])
        validation = validate_snapshot(self.root, snapshot, expected_seal_sha256=result["seal_sha256"])
        self.assertEqual(validation["regular_file_count"], 1)
        self.assertEqual(validation["safe_symlink_count"], 1)
        self.assertTrue((snapshot / "legacy").is_symlink())
        self.assertFalse((snapshot / "bad-link").exists())
        self.assertEqual((snapshot / "src" / "run.py").stat().st_mode & 0o222, 0)

    def test_publication_never_replaces_existing_snapshot(self) -> None:
        publish_snapshot(self.root, self.spec)
        with self.assertRaises(FileExistsError):
            publish_snapshot(self.root, self.spec)

    def test_source_drift_fails_closed(self) -> None:
        (self.root / "source" / "src" / "run.py").write_text("changed\n", encoding="utf-8")
        with self.assertRaises(PublicationError):
            publish_snapshot(self.root, self.spec)


if __name__ == "__main__":
    unittest.main()
