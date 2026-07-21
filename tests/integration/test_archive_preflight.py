from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


from exp7.provenance.archive_preflight import (
    ArchivePreflightError,
    CANONICAL_RAW_SCHEMA,
    OBJECT_SCHEMA,
    PROOF_SCHEMA,
    REPORT_SCHEMA,
    RETENTION_SCHEMA,
    TRANSCRIPT_SCHEMA,
    validate_archive_preflight,
)


NOW = datetime(2027, 1, 1, tzinfo=timezone.utc)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class EvidenceFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.canonical_manifest = root / "canonical-raw-manifest.jsonl"
        self.backup_proof = root / "backup-proof.json"
        self.object_manifest = root / "object-manifest.jsonl"
        self.retention_evidence = root / "retention-evidence.json"
        self.restore_transcript = root / "restore-transcript.json"
        self.restore_root = root / "restored"
        self.payloads = {
            "paper_outputs/raw/verified/fixture/a.bin": b"alpha\x00payload",
            "paper_outputs/raw/verified/fixture/b.bin": b"beta\x00payload",
        }
        self.canonical_rows = []
        for index, (path, payload) in enumerate(self.payloads.items(), start=1):
            digest = _sha(payload)
            self.canonical_rows.append(
                {
                    "destination_relative_path": path,
                    "fresh_source_record_id": f"source:{digest}",
                    "paper_result_ids": [f"result:fixture:{index}"],
                    "raw_node_id": f"raw:{digest}",
                    "root_id": "fixture",
                    "schema": CANONICAL_RAW_SCHEMA,
                    "sha256": digest,
                    "size": len(payload),
                    "source_relative_path": f"fixture/{index}.bin",
                }
            )
            destination = self.restore_root.joinpath(*Path(path).parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(payload)
        self.canonical_manifest.write_bytes(_jsonl_bytes(self.canonical_rows))
        self.retention = {
            "captured_at": "2026-01-01T00:00:00Z",
            "evidence_id": "retention-evidence-0001",
            "immutability_enabled": True,
            "mode": "object_lock_compliance",
            "object_count": len(self.canonical_rows),
            "provider_evidence_id": "provider-proof-0001",
            "retention_until": "2099-01-01T00:00:00Z",
            "schema": RETENTION_SCHEMA,
            "storage_location": "s3://archive-fixture/raw",
            "versioning_enabled": True,
        }
        self.object_rows = [
            {
                "object_uri": f"s3://archive-fixture/raw/object-{index:04d}",
                "path": row["destination_relative_path"],
                "retention_evidence_id": self.retention["evidence_id"],
                "schema": OBJECT_SCHEMA,
                "sha256": row["sha256"],
                "size": row["size"],
                "version_id": f"version-{index:08d}",
            }
            for index, row in enumerate(self.canonical_rows, start=1)
        ]
        self.write_evidence()

    @property
    def total_bytes(self) -> int:
        return sum(len(payload) for payload in self.payloads.values())

    def write_evidence(self) -> None:
        retention_payload = _json_bytes(self.retention)
        object_payload = _jsonl_bytes(self.object_rows)
        self.retention_evidence.write_bytes(retention_payload)
        self.object_manifest.write_bytes(object_payload)
        transcript = {
            "canonical_manifest_sha256": _sha(self.canonical_manifest.read_bytes()),
            "files": [
                {
                    "path": row.get("path", "paper_outputs/raw/verified/fallback"),
                    "sha256": row.get("sha256", "0" * 64),
                    "size": row.get("size", 0),
                    "status": "restored_verified",
                    "version_id": row.get("version_id", "version-00000000"),
                }
                for row in self.object_rows
            ],
            "object_count": len(self.canonical_rows),
            "object_manifest_sha256": _sha(object_payload),
            "restore_id": "independent-restore-0001",
            "restored_at": "2026-01-02T00:00:00Z",
            "retention_evidence_sha256": _sha(retention_payload),
            "schema": TRANSCRIPT_SCHEMA,
            "total_bytes": self.total_bytes,
        }
        transcript_payload = _json_bytes(transcript)
        self.restore_transcript.write_bytes(transcript_payload)
        proof = {
            "backup_id": "offhost-backup-0001",
            "canonical_manifest_sha256": _sha(self.canonical_manifest.read_bytes()),
            "created_at": "2026-01-03T00:00:00Z",
            "independent_restore": True,
            "object_count": len(self.canonical_rows),
            "object_manifest_sha256": _sha(object_payload),
            "off_host": True,
            "restore_transcript_sha256": _sha(transcript_payload),
            "retention_evidence_sha256": _sha(retention_payload),
            "schema": PROOF_SCHEMA,
            "total_bytes": self.total_bytes,
        }
        self.backup_proof.write_bytes(_json_bytes(proof))

    def validate(self) -> dict:
        return validate_archive_preflight(
            canonical_manifest=self.canonical_manifest,
            backup_proof=self.backup_proof,
            object_manifest=self.object_manifest,
            retention_evidence=self.retention_evidence,
            restore_transcript=self.restore_transcript,
            restore_root=self.restore_root,
            expected_count=2,
            now=NOW,
        )


def _file_state(root: Path) -> dict[str, tuple[int, int, str]]:
    result = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        value = path.stat()
        result[path.relative_to(root).as_posix()] = (
            value.st_mtime_ns,
            value.st_size,
            _sha(path.read_bytes()),
        )
    return result


class ArchivePreflightTests(unittest.TestCase):
    def test_no_arguments_reports_blocked_nonzero_and_mutates_nothing(self) -> None:
        self.assertEqual(
            REPORT_SCHEMA,
            "experiments7-archive-preflight-report/v3",
        )
        tracked = (
            ROOT / "archive/archive-map.json",
            ROOT / "archive/archive-map.md",
            ROOT / "paper_outputs/final/raw-manifest.jsonl",
        )
        before = {
            path: (path.stat().st_mtime_ns, path.stat().st_size, _sha(path.read_bytes()))
            for path in tracked
        }
        result = subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/archive_preflight.py")],
            cwd=ROOT,
            env={
                "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["schema"], REPORT_SCHEMA)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["preflight_passed"])
        self.assertEqual(report["preflight_scope"], "structural_only")
        self.assertEqual(report["validation_domain"], "archive_evidence_structure")
        self.assertFalse(report["evidence_bundle_structurally_valid"])
        self.assertFalse(report["provider_authenticity_verified"])
        self.assertFalse(report["independent_approval_verified"])
        self.assertFalse(report["cutover_receipt_verified"])
        self.assertFalse(report["externalization_authorized"])
        self.assertTrue(report["no_mutations"])
        after = {
            path: (path.stat().st_mtime_ns, path.stat().st_size, _sha(path.read_bytes()))
            for path in tracked
        }
        self.assertEqual(before, after)

    def test_success_on_tiny_fixture_hashes_restore_and_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvidenceFixture(Path(temporary))
            before = _file_state(fixture.root)
            report = fixture.validate()
            self.assertEqual(report["status"], "evidence_bundle_structurally_valid")
            self.assertTrue(report["preflight_passed"])
            self.assertEqual(report["preflight_scope"], "structural_only")
            self.assertEqual(report["validation_domain"], "archive_evidence_structure")
            self.assertTrue(report["evidence_bundle_structurally_valid"])
            self.assertFalse(report["provider_authenticity_verified"])
            self.assertFalse(report["independent_approval_verified"])
            self.assertFalse(report["cutover_receipt_verified"])
            self.assertFalse(report["externalization_authorized"])
            self.assertFalse(report["archive_map_mutated"])
            self.assertEqual(report["canonical_manifest"]["object_count"], 2)
            self.assertEqual(report["canonical_manifest"]["payload_bytes"], fixture.total_bytes)
            self.assertEqual(before, _file_state(fixture.root))

    def test_missing_proof_and_evidence_symlinks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvidenceFixture(Path(temporary))
            fixture.backup_proof.unlink()
            with self.assertRaisesRegex(ArchivePreflightError, "backup proof"):
                fixture.validate()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvidenceFixture(Path(temporary))
            target = fixture.root / "proof-target.json"
            fixture.backup_proof.rename(target)
            fixture.backup_proof.symlink_to(target)
            with self.assertRaisesRegex(ArchivePreflightError, "non-symlink"):
                fixture.validate()

    def test_digest_bound_evidence_tamper_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvidenceFixture(Path(temporary))
            changed = deepcopy(fixture.object_rows)
            changed[0]["version_id"] = "version-tampered-0001"
            fixture.object_manifest.write_bytes(_jsonl_bytes(changed))
            with self.assertRaisesRegex(
                ArchivePreflightError, "object manifest binding"
            ):
                fixture.validate()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvidenceFixture(Path(temporary))
            transcript = json.loads(
                fixture.restore_transcript.read_text(encoding="utf-8")
            )
            transcript["restore_id"] = "independent-restore-tampered"
            fixture.restore_transcript.write_bytes(_json_bytes(transcript))
            with self.assertRaisesRegex(
                ArchivePreflightError, "restore_transcript_sha256"
            ):
                fixture.validate()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvidenceFixture(Path(temporary))
            proof = json.loads(fixture.backup_proof.read_text(encoding="utf-8"))
            proof["total_bytes"] += 1
            fixture.backup_proof.write_bytes(_json_bytes(proof))
            with self.assertRaisesRegex(ArchivePreflightError, "total_bytes"):
                fixture.validate()

    def test_duplicate_missing_extra_and_escaping_object_paths_fail_closed(self) -> None:
        cases = {}
        with tempfile.TemporaryDirectory() as temporary:
            fixture = EvidenceFixture(Path(temporary))
            cases["duplicate"] = [*fixture.object_rows, deepcopy(fixture.object_rows[0])]
            cases["missing"] = fixture.object_rows[:-1]
            extra = deepcopy(fixture.object_rows)
            extra_row = deepcopy(extra[0])
            extra_row.update(
                {
                    "object_uri": "s3://archive-fixture/raw/object-extra",
                    "path": "paper_outputs/raw/verified/fixture/extra.bin",
                    "version_id": "version-extra-0001",
                }
            )
            cases["extra"] = [*extra, extra_row]
            escaping = deepcopy(fixture.object_rows)
            escaping[0]["path"] = "../escape.bin"
            cases["escape"] = escaping

        for label, rows in cases.items():
            with self.subTest(case=label), tempfile.TemporaryDirectory() as temporary:
                fixture = EvidenceFixture(Path(temporary))
                fixture.object_rows = deepcopy(rows)
                fixture.write_evidence()
                with self.assertRaises(ArchivePreflightError):
                    fixture.validate()

    def test_fake_or_missing_version_and_immutability_fail_closed(self) -> None:
        for mutation, message in (
            ("fake_version", "version identifier"),
            ("missing_version", "keys differ"),
            ("immutability_false", "immutability_enabled"),
            ("missing_immutability", "keys differ"),
        ):
            with self.subTest(case=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = EvidenceFixture(Path(temporary))
                if mutation == "fake_version":
                    fixture.object_rows[0]["version_id"] = "fake-version"
                elif mutation == "missing_version":
                    fixture.object_rows[0].pop("version_id")
                elif mutation == "immutability_false":
                    fixture.retention["immutability_enabled"] = False
                else:
                    fixture.retention.pop("immutability_enabled")
                fixture.write_evidence()
                with self.assertRaisesRegex(ArchivePreflightError, message):
                    fixture.validate()

    def test_restore_missing_extra_symlink_and_byte_mismatch_fail_closed(self) -> None:
        for mutation, message in (
            ("missing", "restore path set differs"),
            ("extra", "restore path set differs"),
            ("symlink", "symlink or unsafe file"),
            ("bytes", "size/hash differs"),
        ):
            with self.subTest(case=mutation), tempfile.TemporaryDirectory() as temporary:
                fixture = EvidenceFixture(Path(temporary))
                first_relative = next(iter(fixture.payloads))
                first = fixture.restore_root.joinpath(*Path(first_relative).parts)
                if mutation == "missing":
                    first.unlink()
                elif mutation == "extra":
                    (fixture.restore_root / "extra.bin").write_bytes(b"extra")
                elif mutation == "symlink":
                    outside = fixture.root / "outside.bin"
                    outside.write_bytes(first.read_bytes())
                    first.unlink()
                    os.symlink(outside, first)
                else:
                    payload = first.read_bytes()
                    first.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
                with self.assertRaisesRegex(ArchivePreflightError, message):
                    fixture.validate()


if __name__ == "__main__":
    unittest.main()
