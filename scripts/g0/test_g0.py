#!/usr/bin/env python3
"""Standard-library hostile tests for the G0-owned tooling."""
from __future__ import annotations

import base64
import builtins
import contextlib
import io
import json
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import compare_manifests
import manifest_tool
import preflight_target
import run_protected


class G0Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = Path(tempfile.mkdtemp(prefix="experiments7-g0-test-", dir="/tmp"))
        self.previous_envelope_id = os.environ.pop("EXPERIMENTS7_ENVELOPE_ID", None)

    def tearDown(self) -> None:
        if self.previous_envelope_id is not None:
            os.environ["EXPERIMENTS7_ENVELOPE_ID"] = self.previous_envelope_id
        else:
            os.environ.pop("EXPERIMENTS7_ENVELOPE_ID", None)
        shutil.rmtree(self.base)

    def root_config(self, root: Path) -> list[dict[str, object]]:
        st = os.lstat(root)
        return [{"root_id": "fixture", "path": str(root), "binding": {"dev": st.st_dev, "inode": st.st_ino}}]

    def canonical_records(self, root: Path) -> bytes:
        return b"".join(
            manifest_tool.canonical_json(item)
            for item in manifest_tool._build_source_records(
                self.root_config(root)
            )
        )

    def producer_binding(self) -> dict[str, object]:
        context = {"namespace": "synthetic"}
        acceptance = {
            "schema": "experiments7-publication-transcript/v6",
            "sealed_run_id": "synthetic-run",
            "run_root": str(self.base / "strict-run"),
            "relative_path": "reservation-acceptance.json",
            "context": context,
            "emitted_monotonic_ns": 10,
        }
        acceptance_sha256 = manifest_tool.sha256_bytes(
            manifest_tool.canonical_json(acceptance)
        )
        envelope_payload = {
            "schema": "experiments7-cp0-envelope-evidence/v6",
            "sealed_run_id": "synthetic-run",
            "run_root": str(self.base / "strict-run"),
            "acceptance_transcript_sha256": acceptance_sha256,
        }
        envelope_transcript = {
            "schema": "experiments7-publication-transcript/v6",
            "sealed_run_id": "synthetic-run",
            "run_root": str(self.base / "strict-run"),
            "relative_path": "frozen/envelope_evidence/artifact.json",
            "context": context,
            "emitted_monotonic_ns": 20,
        }
        envelope_evidence = {
            "schema": "experiments7-stage-artifact/v6",
            "relative_path": "frozen/envelope_evidence/artifact.json",
            "artifact_type": "regular",
            "sha256": manifest_tool.sha256_bytes(
                manifest_tool.canonical_json(envelope_payload)
            ),
            "bytes": len(manifest_tool.canonical_json(envelope_payload)),
            "publication_transcript_sha256": manifest_tool.sha256_bytes(
                manifest_tool.canonical_json(envelope_transcript)
            ),
        }
        return {
            "schema": "experiments7-cp0-producer-binding/v6",
            "acceptance_transcript": acceptance,
            "envelope_artifact_evidence": envelope_evidence,
            "envelope_transcript": envelope_transcript,
            "envelope_payload": envelope_payload,
        }

    def manifest_config(self, name: str) -> dict[str, object]:
        root = self.base / f"{name}-root"
        root.mkdir()
        (root / "value.txt").write_text("value", encoding="utf-8")
        paper_dir = self.base / f"{name}-paper"
        paper_dir.mkdir()
        paper = paper_dir / "paper.pdf"
        paper.write_bytes(b"synthetic-pdf")
        paper_parent = os.lstat(paper_dir)
        paper_file = os.lstat(paper)
        return {
            "sealed_run_id": "synthetic-run",
            "run_root": str(self.base / "strict-run"),
            "protected_sources": self.root_config(root),
            "paper": {
                "path": str(paper),
                "parent_binding": {
                    "dev": paper_parent.st_dev,
                    "inode": paper_parent.st_ino,
                },
                "binding": {
                    "dev": paper_file.st_dev,
                    "inode": paper_file.st_ino,
                },
            },
            "metadata_scope": ["type", "size", "dev", "inode"],
            "root_readme": {"path": str(paper)},
            "owner_record": {"path": str(paper)},
            "outputs": {
                "build-pre": {
                    "source": str(self.base / f"{name}-source.jsonl"),
                    "paper": str(self.base / f"{name}-paper.json"),
                }
            },
            "writable_during_g0": [str(self.base)],
        }

    def write_canonical(self, name: str, value: object) -> Path:
        path = self.base / name
        path.write_bytes(manifest_tool.canonical_json(value))
        return path

    def test_canonical_determinism_and_no_symlink_traversal(self) -> None:
        root = self.base / "root"
        root.mkdir()
        (root / "a.txt").write_bytes(b"alpha\n")
        (root / "nested").mkdir()
        (root / "nested" / "b.bin").write_bytes(b"\x00\xff")
        os.link(root / "a.txt", root / "hardlink.txt")
        os.symlink("/etc/passwd", root / "external-link")
        first = self.canonical_records(root)
        second = self.canonical_records(root)
        self.assertEqual(first, second)
        records = [json.loads(line) for line in first.splitlines()]
        link = next(item for item in records if base64.b64decode(item["relative_path_b64"]) == b"external-link")
        self.assertEqual(link["type"], "symlink")
        self.assertEqual(base64.b64decode(link["link_target_b64"]), b"/etc/passwd")
        self.assertNotIn("sha256", link)

    def test_content_and_metadata_drift_are_detected(self) -> None:
        root = self.base / "root"
        root.mkdir()
        path = root / "value.txt"
        path.write_text("one", encoding="utf-8")
        first = self.canonical_records(root)
        path.write_text("two", encoding="utf-8")
        second = self.canonical_records(root)
        self.assertNotEqual(first, second)
        before_mtime = path.stat().st_mtime_ns
        os.utime(path, ns=(path.stat().st_atime_ns, before_mtime + 1_000_000))
        third = self.canonical_records(root)
        self.assertNotEqual(second, third)

    def test_attested_source_builder_starts_after_validated_gate(self) -> None:
        root = self.base / "attested-root"
        root.mkdir()
        (root / "value.txt").write_text("value", encoding="utf-8")
        with patch.object(manifest_tool.time, "monotonic_ns", return_value=30):
            records, attestation = manifest_tool.build_attested_source_records(
                {
                    "sealed_run_id": "synthetic-run",
                    "run_root": str(self.base / "strict-run"),
                    "protected_sources": self.root_config(root),
                },
                self.producer_binding(),
            )
        self.assertGreaterEqual(len(records), 2)
        self.assertEqual(
            attestation["schema"],
            "experiments7-cp0-protected-read-attestation/v6",
        )
        self.assertEqual(attestation["operation"], "source-pre")
        self.assertEqual(attestation["started_monotonic_ns"], 30)

    def test_attested_builder_rejects_early_or_substituted_gate_before_open(self) -> None:
        root = self.base / "guarded-root"
        root.mkdir()
        cases: list[tuple[str, dict[str, object], int]] = []

        early = self.producer_binding()
        cases.append(("early", early, 20))

        wrong_hash = self.producer_binding()
        wrong_hash["envelope_artifact_evidence"] = dict(
            wrong_hash["envelope_artifact_evidence"]
        )
        wrong_hash["envelope_artifact_evidence"][
            "publication_transcript_sha256"
        ] = "0" * 64
        cases.append(("hash", wrong_hash, 30))

        wrong_context = self.producer_binding()
        wrong_context["envelope_transcript"] = dict(
            wrong_context["envelope_transcript"]
        )
        wrong_context["envelope_transcript"]["context"] = {
            "namespace": "other"
        }
        evidence = dict(wrong_context["envelope_artifact_evidence"])
        evidence["publication_transcript_sha256"] = manifest_tool.sha256_bytes(
            manifest_tool.canonical_json(wrong_context["envelope_transcript"])
        )
        wrong_context["envelope_artifact_evidence"] = evidence
        cases.append(("context", wrong_context, 30))

        for label, binding, started in cases:
            with self.subTest(label=label), patch.object(
                manifest_tool.time, "monotonic_ns", return_value=started
            ), patch.object(
                manifest_tool.os,
                "open",
                side_effect=AssertionError("protected open occurred before gate"),
            ):
                with self.assertRaises(RuntimeError):
                    manifest_tool.build_attested_source_records(
                        {
                            "sealed_run_id": "synthetic-run",
                            "run_root": str(self.base / "strict-run"),
                            "protected_sources": self.root_config(root),
                        },
                        binding,
                    )

    def test_manifest_cli_rejects_all_legacy_reads_before_access(self) -> None:
        variants = {
            "build-pre": ["manifest_tool.py", "build-pre"],
            "build-paper": ["manifest_tool.py", "build-paper"],
            "build-post": ["manifest_tool.py", "build-post"],
            "path-binding": [
                "manifest_tool.py",
                "build-pre",
                "--producer-binding",
                "/must/not/open/producer-binding.json",
            ],
        }
        for label, argv in variants.items():
            stderr = io.StringIO()
            with self.subTest(label=label), patch.object(
                sys, "argv", argv
            ), patch.object(
                manifest_tool.os, "open"
            ) as protected_open, patch.object(
                manifest_tool, "load_json"
            ) as load_config, patch.object(
                manifest_tool, "load_canonical_json"
            ) as load_binding, patch.object(
                manifest_tool, "_build_source_records"
            ) as source_read, patch.object(
                manifest_tool, "_build_paper_record"
            ) as paper_read, contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as blocked:
                    manifest_tool.main()
            self.assertEqual(blocked.exception.code, 2)
            self.assertIn("legacy manifest CLI is disabled", stderr.getvalue())
            protected_open.assert_not_called()
            load_config.assert_not_called()
            load_binding.assert_not_called()
            source_read.assert_not_called()
            paper_read.assert_not_called()

    def test_runner_rejects_all_legacy_reads_before_any_file_or_provider_access(
        self,
    ) -> None:
        self.assertNotIn("provider", run_protected.__dict__)
        self.assertNotIn("manifest_tool", run_protected.__dict__)
        variants = {
            "build-pre": ["run_protected.py", "build-pre"],
            "build-paper": ["run_protected.py", "build-paper"],
            "build-post": ["run_protected.py", "build-post"],
            "path-binding": [
                "run_protected.py",
                "build-pre",
                "--producer-binding",
                "/must/not/open/producer-binding.json",
            ],
        }
        for label, argv in variants.items():
            stderr = io.StringIO()
            with self.subTest(label=label), patch.object(
                sys, "argv", argv
            ), patch.object(
                builtins, "open"
            ) as any_file_open, patch.object(
                manifest_tool, "publish_source_manifest"
            ) as source_publish, patch.object(
                manifest_tool, "publish_paper_manifest"
            ) as paper_publish, contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as blocked:
                    run_protected.main()
            self.assertEqual(blocked.exception.code, 2)
            self.assertIn(
                "legacy protected-read wrapper is disabled", stderr.getvalue()
            )
            any_file_open.assert_not_called()
            source_publish.assert_not_called()
            paper_publish.assert_not_called()

    def test_atomic_publish_is_no_replace(self) -> None:
        output = self.base / "evidence.json"
        result = manifest_tool.atomic_publish(str(output), [b"{}\n"])
        self.assertEqual(result["sha256"], "ca3d163bab055381827226140568f3bef7eaac187cebd76878e0b63e9e442356")
        with self.assertRaises(FileExistsError):
            manifest_tool.atomic_publish(str(output), [b"changed\n"])
        self.assertEqual(output.read_bytes(), b"{}\n")

    def test_comparator_rejects_source_and_paper_drift(self) -> None:
        source_pre = self.base / "source-pre.jsonl"
        source_post = self.base / "source-post.jsonl"
        record = {"record_id": "r1", "root_id": "fixture", "relative_path_b64": "", "sha256": "a"}
        raw = manifest_tool.canonical_json(record)
        source_pre.write_bytes(raw)
        source_post.write_bytes(raw)
        paper_pre = self.base / "paper-pre.json"
        paper_post = self.base / "paper-post.json"
        paper = {"bundle_lock_sha256": "lock", "sha256": "paper"}
        paper_pre.write_bytes(manifest_tool.canonical_json(paper))
        paper_post.write_bytes(manifest_tool.canonical_json(paper))
        passed = compare_manifests.compare(str(source_pre), str(source_post), str(paper_pre), str(paper_post), "lock")
        self.assertEqual(passed["status"], "PASS")
        source_post.write_bytes(manifest_tool.canonical_json({**record, "sha256": "b"}))
        failed = compare_manifests.compare(str(source_pre), str(source_post), str(paper_pre), str(paper_post), "lock")
        self.assertEqual(failed["status"], "FAIL")
        self.assertIn("source_records_differ", failed["failures"])

    def test_synthetic_paper_tamper_changes_record(self) -> None:
        paper_dir = self.base / "_paper"
        paper_dir.mkdir()
        paper = paper_dir / "paper.pdf"
        paper.write_bytes(b"fake-pdf-v1")
        parent_st = os.lstat(paper_dir)
        paper_st = os.lstat(paper)
        config = {
            "paper": {
                "path": str(paper),
                "parent_binding": {"dev": parent_st.st_dev, "inode": parent_st.st_ino},
                "binding": {"dev": paper_st.st_dev, "inode": paper_st.st_ino},
            },
            "metadata_scope": ["type", "mode", "uid", "gid", "nlink", "size", "mtime_ns", "ctime_ns", "dev", "inode"],
        }
        first = manifest_tool._build_paper_record(config, "lock")
        paper.write_bytes(b"fake-pdf-v2")
        current = os.lstat(paper)
        os.utime(paper, ns=(current.st_atime_ns, int(first["metadata"]["mtime_ns"]) + 1_000_000))
        changed_st = os.lstat(paper)
        config["paper"]["binding"] = {"dev": changed_st.st_dev, "inode": changed_st.st_ino}
        second = manifest_tool._build_paper_record(config, "lock")
        self.assertNotEqual(first["sha256"], second["sha256"])
        self.assertNotEqual(first["metadata"], second["metadata"])

    def test_bootstrap_constants_and_symlink_boundary(self) -> None:
        self.assertEqual(preflight_target.README_BYTES, b"# experiments7\n\nSee [docs/README.md](docs/README.md).\n")
        real = self.base / "real"
        real.mkdir()
        link = self.base / "link"
        link.symlink_to(real, target_is_directory=True)
        with self.assertRaises(RuntimeError):
            preflight_target.check_real_dir(str(link))


if __name__ == "__main__":
    unittest.main(verbosity=2)
