#!/usr/bin/env python3
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent
SPEC = importlib.util.spec_from_file_location("supplement", HERE / "publish_incident_baseline_supplement.py")
mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader
SPEC.loader.exec_module(mod)


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir="/tmp")
        self.root = Path(self.tmp.name) / "root"
        self.root.mkdir()
        baseline = []
        snapshot = {}
        for i in range(37):
            path = f"baseline/{i:02d}.txt"
            data = f"row-{i}".encode()
            digest = hashlib.sha256(data).hexdigest()
            baseline.append({"mode": 420, "path": path, "sha256": digest, "size": len(data)})
            snapshot[path] = digest
        self.omitted = {}
        for path, data in (("one", b"1"), ("two", b"22"), ("nested/three", b"333")):
            target = self.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            digest = hashlib.sha256(data).hexdigest()
            self.omitted[path] = digest
            snapshot[path] = digest
        incident = {"legacy_baseline": baseline}
        terminal = {"acceptance_eligible": False, "legacy_baseline": baseline}
        self.incident_hash = self._record(mod.INCIDENT_REL, incident)
        self.terminal_hash = self._record(mod.TERMINAL_REL, terminal)
        self.snapshot = Path(self.tmp.name) / "snapshot.json"
        self.snapshot.write_text(json.dumps(snapshot), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _record(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = mod.canonical_json(value)
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    def publish(self):
        return mod.publish(self.root, self.snapshot, self.incident_hash, self.terminal_hash, self.omitted)

    def test_success(self):
        result = self.publish()
        self.assertEqual(result["complete_count"], 40)
        self.assertFalse(result["acceptance_eligible"])
        self.assertEqual(len(result["omitted_rows"]), 3)

    def test_no_replace(self):
        self.publish()
        with self.assertRaises(mod.PublicationError):
            self.publish()

    def test_symlink_rejection(self):
        (self.root / "one").unlink()
        os.symlink("two", self.root / "one")
        with self.assertRaises(mod.PublicationError):
            self.publish()

    def test_original_record_drift(self):
        with (self.root / mod.INCIDENT_REL).open("ab") as stream:
            stream.write(b" ")
        with self.assertRaisesRegex(mod.PublicationError, "record hash drift"):
            self.publish()

    def test_singleton_drift(self):
        (self.root / "two").write_bytes(b"changed")
        with self.assertRaisesRegex(mod.PublicationError, "singleton drift"):
            self.publish()

    def test_snapshot_set_mismatch(self):
        value = json.loads(self.snapshot.read_text())
        value["unexpected"] = value.pop("baseline/00.txt")
        self.snapshot.write_text(json.dumps(value))
        with self.assertRaisesRegex(mod.PublicationError, "snapshot set mismatch"):
            self.publish()


if __name__ == "__main__":
    unittest.main()
