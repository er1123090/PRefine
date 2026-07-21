#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("publish_incident.py")
SPEC = importlib.util.spec_from_file_location("publish_incident", MODULE_PATH)
assert SPEC and SPEC.loader
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


class PublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="exp7-incident-", dir="/tmp")
        self.root = Path(self.temp.name)
        for prefix in publisher.INVENTORY_ROOTS:
            (self.root / prefix).mkdir(parents=True, exist_ok=True)
        self.expected = {}
        for index, relative in enumerate(publisher.CRITICAL_HASHES):
            payload = f"fixture-{index}\n".encode()
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            self.expected[relative] = hashlib.sha256(payload).hexdigest()
        extra = self.root / "paper_outputs/inventory/structure.jsonl"
        extra.write_text("{}\n")
        os.chmod(extra, 0o640)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def load_outputs(self):
        incident_path = self.root / publisher.INCIDENT_REL
        terminal_path = self.root / publisher.TERMINAL_REL
        return json.loads(incident_path.read_bytes()), json.loads(terminal_path.read_bytes())

    def test_exact_schema_null_reasons_and_baseline(self):
        incident, terminal = publisher.publish(self.root, self.expected)
        self.assertEqual(set(incident), {
            "actor", "argv", "argv_unavailable_reason", "content_bytes_read", "decision",
            "filenames_observed", "legacy_baseline", "observed_at",
            "observed_at_unavailable_reason", "operation", "protected_root_strings",
            "protected_writes", "run_id", "schema", "violated",
        })
        self.assertIsNone(incident["argv"])
        self.assertIsNone(incident["observed_at"])
        self.assertTrue(incident["argv_unavailable_reason"])
        self.assertTrue(incident["observed_at_unavailable_reason"])
        self.assertEqual(incident["actor"], {"canonical_task": "/root/planner_g1_matrix", "role": "planner"})
        self.assertEqual(incident["violated"], ["G0-ORD-01", "I-G0-01"])
        self.assertEqual(set(terminal), {"acceptance_eligible", "incident", "legacy_baseline", "run_id", "schema", "terminal_state"})
        self.assertEqual(terminal["terminal_state"], "failed")
        self.assertFalse(terminal["acceptance_eligible"])
        self.assertEqual(terminal["legacy_baseline"], incident["legacy_baseline"])
        disk_incident, disk_terminal = self.load_outputs()
        self.assertEqual((disk_incident, disk_terminal), (incident, terminal))
        raw = (self.root / publisher.INCIDENT_REL).read_bytes()
        self.assertEqual(raw, publisher.canonical_json(incident))
        self.assertEqual(disk_terminal["incident"]["sha256"], hashlib.sha256(raw).hexdigest())
        row = next(row for row in incident["legacy_baseline"] if row["path"].endswith("structure.jsonl"))
        self.assertEqual(row["mode"], 0o640)

    def test_duplicate_is_no_replace(self):
        publisher.publish(self.root, self.expected)
        incident_path = self.root / publisher.INCIDENT_REL
        before = incident_path.read_bytes()
        with self.assertRaises(publisher.PublicationError):
            publisher.publish(self.root, self.expected)
        self.assertEqual(incident_path.read_bytes(), before)

    def test_parent_symlink_refused(self):
        sealed = self.root / "manifests/sealed-runs"
        target = self.root / "elsewhere"
        target.mkdir()
        sealed.symlink_to(target, target_is_directory=True)
        with self.assertRaises(publisher.PublicationError):
            publisher.publish(self.root, self.expected)
        self.assertEqual(list(target.iterdir()), [])

    def test_critical_drift_refused(self):
        path = self.root / next(iter(self.expected))
        path.write_bytes(b"drift\n")
        with self.assertRaisesRegex(publisher.PublicationError, "critical hash drift"):
            publisher.publish(self.root, self.expected)
        self.assertFalse((self.root / publisher.INCIDENT_REL).exists())

    def test_exclusion_boundaries(self):
        excluded = (
            "manifests/sealed-runs/old/evidence.json",
            "scripts/g0/v2/later.py",
            "scripts/g0/attempts/one.json",
            "scripts/g0/caches/value.bin",
            "scripts/g0/__pycache__/value.pyc",
        )
        for relative in excluded:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"excluded")
        included = self.root / "scripts/g0/provider-attempts.json"
        included.write_bytes(b"included")
        baseline = publisher.build_legacy_baseline(self.root)
        paths = {row["path"] for row in baseline}
        self.assertIn("scripts/g0/provider-attempts.json", paths)
        self.assertTrue(set(excluded).isdisjoint(paths))

    def test_no_protected_roots_are_accessed(self):
        accessed = []
        original = publisher.os.scandir

        def guarded(path):
            value = os.fspath(path)
            accessed.append(value)
            if any(value == root or value.startswith(root + os.sep) for root in publisher.PROTECTED_ROOTS):
                raise AssertionError("protected access")
            return original(path)

        publisher.os.scandir = guarded
        try:
            publisher.publish(self.root, self.expected)
        finally:
            publisher.os.scandir = original
        self.assertTrue(accessed)
        self.assertTrue(all(value.startswith(str(self.root) + os.sep) for value in accessed))


if __name__ == "__main__":
    unittest.main()
