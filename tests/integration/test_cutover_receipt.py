from __future__ import annotations

import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exp7.provenance.cutover_receipt import (
    ARCHIVE_MAP_PATH,
    ARCHIVE_MAP_SCHEMA,
    CUTOVER_PATHS,
    CutoverReceiptError,
    DESTINATION_INVENTORY_PATH,
    DESTINATION_INVENTORY_SCHEMA,
    PROVIDER_PROOF_PATH,
    PROVIDER_PROOF_SCHEMA,
    POST_CUTOVER_STATE_SCHEMA,
    RECEIPT_PATH,
    RECEIPT_SCHEMA,
    RESTORE_REPORT_PATH,
    RESTORE_REPORT_SCHEMA,
    SOURCE_INVENTORY_PATH,
    SOURCE_INVENTORY_SCHEMA,
    TRUST_SCHEMA,
    VALIDATION_SCHEMA,
    cutover_receipt_report,
    inspect_post_cutover_local_state,
    validate_cutover_receipt,
)


NOW = datetime(2026, 7, 22, tzinfo=timezone.utc)
OPENSSL = shutil.which("openssl")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: list[dict]) -> bytes:
    return b"".join(_canonical(row) + b"\n" for row in rows)


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class _RSAKey:
    def __init__(self, directory: Path, name: str) -> None:
        if OPENSSL is None:
            raise unittest.SkipTest("openssl is required for synthetic RSA fixture")
        self.path = directory / f"{name}.pem"
        generated = subprocess.run(
            [
                OPENSSL,
                "genpkey",
                "-algorithm",
                "RSA",
                "-pkeyopt",
                "rsa_keygen_bits:2048",
                "-out",
                str(self.path),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if generated.returncode != 0:
            raise RuntimeError(generated.stderr)
        modulus = subprocess.run(
            [OPENSSL, "rsa", "-in", str(self.path), "-noout", "-modulus"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        self.modulus_hex = modulus.removeprefix("Modulus=").lower()

    def sign(self, payload: bytes) -> str:
        signature_path = self.path.with_suffix(".sig")
        result = subprocess.run(
            [
                OPENSSL,
                "dgst",
                "-sha256",
                "-sign",
                str(self.path),
                "-out",
                str(signature_path),
            ],
            input=payload,
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))
        return base64.b64encode(signature_path.read_bytes()).decode("ascii")


class ReceiptFixture:
    def __init__(
        self,
        root: Path,
        *,
        local_provider: bool = False,
        raw_metadata_stubs: bool = False,
    ) -> None:
        self.root = root
        (root / "archive").mkdir(parents=True)
        external = root.parent / "external-trust-material"
        external.mkdir(parents=True)
        self.trust_path = external / "cutover-trust.json"
        self.provider_key = _RSAKey(external, "provider")
        self.reviewer_key = _RSAKey(external, "reviewer")
        self.provider_identity = "aws-provider-signing-service"
        self.provider_key_id = "provider-key-20260721"
        self.reviewer_identity = "independent-reviewer-0001"
        self.reviewer_key_id = "reviewer-key-20260721"
        if raw_metadata_stubs:
            raw = root / "paper_outputs/raw"
            raw.mkdir(parents=True)
            (raw / "README.md").write_text("# Externalized raw payload\n", encoding="utf-8")
            (raw / "schema.json").write_text("{}\n", encoding="utf-8")

        source_paths = (
            "data/sources/source.json",
            "experiments/run.json",
            "methods/method.py",
            "paper_outputs/raw/payload.bin",
        )
        self.source_rows = []
        for index, path in enumerate(source_paths, start=1):
            payload = f"payload-{index}".encode("utf-8")
            self.source_rows.append(
                {
                    "path": path,
                    "schema": SOURCE_INVENTORY_SCHEMA,
                    "sha256": _sha(payload),
                    "size": len(payload),
                }
            )
        self.source_payload = _jsonl_bytes(self.source_rows)
        root.joinpath(SOURCE_INVENTORY_PATH).write_bytes(self.source_payload)

        storage = (
            "file:///tmp/cutover"
            if local_provider
            else "s3://production-archive-bucket/experiments7"
        )
        self.destination_rows = [
            {
                "object_uri": f"{storage}/object-{index:04d}",
                "schema": DESTINATION_INVENTORY_SCHEMA,
                "sha256": row["sha256"],
                "size": row["size"],
                "source_path": row["path"],
                "version_id": f"version-{index:08d}",
            }
            for index, row in enumerate(self.source_rows, start=1)
        ]
        self.destination_payload = _jsonl_bytes(self.destination_rows)
        root.joinpath(DESTINATION_INVENTORY_PATH).write_bytes(self.destination_payload)
        total_bytes = sum(row["size"] for row in self.source_rows)

        records = []
        for path in CUTOVER_PATHS:
            rows = [row for row in self.source_rows if row["path"].startswith(path + "/")]
            records.append(
                {
                    "bytes": sum(row["size"] for row in rows),
                    "file_count": len(rows),
                    "path": path,
                    "symlink_count": 0,
                }
            )
        archive_map = {
            "no_mutations": True,
            "planned_only": False,
            "records": records,
            "schema": ARCHIVE_MAP_SCHEMA,
        }
        self.archive_payload = _json_bytes(archive_map)
        root.joinpath(ARCHIVE_MAP_PATH).write_bytes(self.archive_payload)

        object_manifest_sha = _sha(self.destination_payload)
        restore = {
            "archive_map_sha256": _sha(self.archive_payload),
            "destination_inventory_sha256": _sha(self.destination_payload),
            "object_count": len(self.destination_rows),
            "object_manifest_sha256": object_manifest_sha,
            "restore_id": "provider-restore-20260720",
            "restored_at": "2026-07-20T00:00:00Z",
            "schema": RESTORE_REPORT_SCHEMA,
            "source_inventory_sha256": _sha(self.source_payload),
            "status": "restored_verified",
            "total_bytes": total_bytes,
        }
        self.restore_payload = _json_bytes(restore)
        root.joinpath(RESTORE_REPORT_PATH).write_bytes(self.restore_payload)

        provider = {
            "archive_map_sha256": _sha(self.archive_payload),
            "destination_inventory_sha256": _sha(self.destination_payload),
            "immutability_enabled": True,
            "immutability_mode": "object_lock_compliance",
            "issued_at": "2026-07-20T01:00:00Z",
            "object_count": len(self.destination_rows),
            "object_manifest_sha256": object_manifest_sha,
            "provider_account_id": "account-12345678",
            "provider_id": "aws-s3-provider",
            "restore_report_sha256": _sha(self.restore_payload),
            "schema": PROVIDER_PROOF_SCHEMA,
            "source_inventory_sha256": _sha(self.source_payload),
            "storage_location": storage,
            "total_bytes": total_bytes,
            "version_pointer": f"{storage}/versions/version-20260720",
            "versioning_enabled": True,
        }
        self.provider_payload = _json_bytes(provider)
        root.joinpath(PROVIDER_PROOF_PATH).write_bytes(self.provider_payload)

        trust = {
            "provider_signers": [
                {
                    "algorithm": "rsa-pkcs1v15-sha256",
                    "exponent": 65537,
                    "identity": self.provider_identity,
                    "key_id": self.provider_key_id,
                    "modulus_hex": self.provider_key.modulus_hex,
                }
            ],
            "reviewer_signers": [
                {
                    "algorithm": "rsa-pkcs1v15-sha256",
                    "exponent": 65537,
                    "identity": self.reviewer_identity,
                    "key_id": self.reviewer_key_id,
                    "modulus_hex": self.reviewer_key.modulus_hex,
                }
            ],
            "schema": TRUST_SCHEMA,
        }
        self.trust_payload = _json_bytes(trust)
        self.trust_path.write_bytes(self.trust_payload)
        self.trust_sha256 = _sha(self.trust_payload)

        local_state = inspect_post_cutover_local_state(root)
        receipt = {
            "approval": {
                "approved_at": "2026-07-20T02:00:00Z",
                "decision": "approved_physical_cutover",
                "independent_of": "cutover-requester-0001",
                "reviewer_id": self.reviewer_identity,
                "reviewer_name": "IndependentReviewer",
            },
            "archive_map": {
                "path": ARCHIVE_MAP_PATH,
                "schema": ARCHIVE_MAP_SCHEMA,
                "sha256": _sha(self.archive_payload),
            },
            "completed_at": "2026-07-20T03:00:00Z",
            "destination_inventory": {
                "bytes": total_bytes,
                "file_count": len(self.destination_rows),
                "path": DESTINATION_INVENTORY_PATH,
                "schema": DESTINATION_INVENTORY_SCHEMA,
                "sha256": _sha(self.destination_payload),
            },
            "post_cutover_local_state": {
                "schema": POST_CUTOVER_STATE_SCHEMA,
                "sha256": local_state["sha256"],
            },
            "provider_authentication": {
                "algorithm": "rsa-pkcs1v15-sha256",
                "key_id": self.provider_key_id,
                "path": PROVIDER_PROOF_PATH,
                "provider_id": provider["provider_id"],
                "schema": PROVIDER_PROOF_SCHEMA,
                "sha256": _sha(self.provider_payload),
                "signature": self.provider_key.sign(_canonical(provider)),
                "signer_identity": self.provider_identity,
            },
            "receipt_id": "cutover-receipt-20260720",
            "restore_evidence": {
                "path": RESTORE_REPORT_PATH,
                "schema": RESTORE_REPORT_SCHEMA,
                "sha256": _sha(self.restore_payload),
            },
            "rollback": {
                "object_manifest_sha256": object_manifest_sha,
                "restore_id": restore["restore_id"],
                "storage_location": provider["storage_location"],
                "version_pointer": provider["version_pointer"],
            },
            "schema": RECEIPT_SCHEMA,
            "source_inventory": {
                "bytes": total_bytes,
                "file_count": len(self.source_rows),
                "path": SOURCE_INVENTORY_PATH,
                "schema": SOURCE_INVENTORY_SCHEMA,
                "sha256": _sha(self.source_payload),
            },
        }
        reviewer_signature = self.reviewer_key.sign(_canonical(receipt))
        receipt["reviewer_signature"] = {
            "algorithm": "rsa-pkcs1v15-sha256",
            "key_id": self.reviewer_key_id,
            "signature": reviewer_signature,
            "signer_identity": self.reviewer_identity,
        }
        root.joinpath(RECEIPT_PATH).write_bytes(_json_bytes(receipt))

    def validate(self) -> dict:
        return validate_cutover_receipt(
            self.root,
            trust_bundle=self.trust_path,
            expected_trust_sha256=self.trust_sha256,
            now=NOW,
        )


@unittest.skipIf(OPENSSL is None, "openssl is required for synthetic RSA fixture")
class CutoverReceiptTests(unittest.TestCase):
    def test_current_repository_without_receipt_is_blocked(self) -> None:
        report = cutover_receipt_report(ROOT)
        self.assertEqual(report["schema"], VALIDATION_SCHEMA)
        self.assertEqual(report["status"], "blocked")
        self.assertFalse(report["verified"])
        self.assertFalse(report["provider_authenticity_verified"])
        self.assertFalse(report["independent_approval_verified"])
        self.assertFalse(report["externalization_authorized"])
        self.assertFalse(report["physical_cutover_complete"])
        self.assertTrue(report["no_mutations"])

    def test_signed_complete_receipt_is_verified(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "repo")
            report = fixture.validate()
        self.assertEqual(report["schema"], VALIDATION_SCHEMA)
        self.assertEqual(report["status"], "verified")
        self.assertTrue(report["verified"])
        self.assertTrue(report["provider_authenticity_verified"])
        self.assertTrue(report["independent_approval_verified"])
        self.assertTrue(report["physical_cutover_complete"])

    def test_exact_raw_metadata_stubs_are_bound_and_permitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(
                Path(temporary) / "repo",
                raw_metadata_stubs=True,
            )
            report = fixture.validate()
        local_state = report["post_cutover_local_state"]
        self.assertTrue(local_state["verified"])
        self.assertEqual(local_state["snapshot"]["raw"]["status"], "metadata_stubs_only")
        self.assertEqual(len(local_state["snapshot"]["raw"]["metadata_stubs"]), 2)

    def test_local_only_provider_evidence_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "repo", local_provider=True)
            with self.assertRaisesRegex(CutoverReceiptError, "off-host"):
                fixture.validate()

    def test_signed_receipt_rejects_still_present_local_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "repo")
            payload = fixture.root / "data/sources/still-active.json"
            payload.parent.mkdir(parents=True)
            payload.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(CutoverReceiptError, "data/sources"):
                fixture.validate()

    def test_raw_metadata_stubs_do_not_allow_raw_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "repo")
            raw = fixture.root / "paper_outputs/raw"
            raw.mkdir(parents=True)
            (raw / "payload.bin").write_bytes(b"still-local")
            with self.assertRaisesRegex(CutoverReceiptError, "paper_outputs/raw"):
                fixture.validate()

    def test_archive_map_edit_invalidates_signed_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "repo")
            archive_path = fixture.root.joinpath(ARCHIVE_MAP_PATH)
            archive_map = json.loads(archive_path.read_text(encoding="utf-8"))
            archive_map["planned_only"] = True
            archive_path.write_bytes(_json_bytes(archive_map))
            with self.assertRaisesRegex(CutoverReceiptError, "archive map digest"):
                fixture.validate()

    def test_reviewer_signature_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "repo")
            receipt_path = fixture.root.joinpath(RECEIPT_PATH)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            receipt["approval"]["reviewer_name"] = "DifferentReviewer"
            receipt_path.write_bytes(_json_bytes(receipt))
            with self.assertRaisesRegex(CutoverReceiptError, "reviewer.*signature"):
                fixture.validate()

    def test_repo_contained_trust_bundle_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "repo")
            local_trust = fixture.root / "editable-trust.json"
            local_trust.write_bytes(fixture.trust_payload)
            with self.assertRaisesRegex(CutoverReceiptError, "outside the repository"):
                validate_cutover_receipt(
                    fixture.root,
                    trust_bundle=local_trust,
                    expected_trust_sha256=fixture.trust_sha256,
                    now=NOW,
                )

    def test_receipt_parent_symlink_is_rejected_by_shared_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "repo")
            archive = fixture.root / "archive"
            real_archive = fixture.root / "archive-real"
            archive.rename(real_archive)
            archive.symlink_to(real_archive, target_is_directory=True)
            with self.assertRaisesRegex(CutoverReceiptError, "non-symlink"):
                fixture.validate()

    def test_receipt_strict_json_rejects_duplicate_and_nonfinite_numbers(self) -> None:
        hostile_payloads = (
            (
                b'{"schema":"experiments7-cutover-receipt/v1","schema":"duplicate"}\n',
                "duplicate object key",
            ),
            (b'{"value":NaN}\n', "non-finite number"),
            (b'{"value":Infinity}\n', "non-finite number"),
            (b'{"value":1e999}\n', "non-finite number"),
        )
        for payload, error in hostile_payloads:
            with self.subTest(payload=payload):
                with tempfile.TemporaryDirectory() as temporary:
                    fixture = ReceiptFixture(Path(temporary) / "repo")
                    fixture.root.joinpath(RECEIPT_PATH).write_bytes(payload)
                    with self.assertRaisesRegex(CutoverReceiptError, error):
                        fixture.validate()

    def test_wrong_external_trust_digest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ReceiptFixture(Path(temporary) / "repo")
            with self.assertRaisesRegex(CutoverReceiptError, "digest differs"):
                validate_cutover_receipt(
                    fixture.root,
                    trust_bundle=fixture.trust_path,
                    expected_trust_sha256="0" * 64,
                    now=NOW,
                )


if __name__ == "__main__":
    unittest.main()
