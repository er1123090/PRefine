from __future__ import annotations

import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strict_run import (  # noqa: E402
    ExternalOutputReservation,
    ExternalTranscriptRecord,
    ExternalTranscriptRegistry,
    StrictRunError,
    canonical_json,
    load_pre_reservation_document,
    sha256_bytes,
)
from strict_run.filesystem import mount_identity, open_absolute_directory  # noqa: E402


RUN_ID = "exp7-strict-v6-20260717T120000Z-" + "e" * 32


class StrictRunV6ExternalRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="exp7-external-v6-")
        self.base = Path(self.temporary.name)
        self.strict_parent = self.base / "strict-runs"
        self.strict_parent.mkdir()
        self.run_root = self.strict_parent / RUN_ID
        self.run_root.mkdir()
        frozen = self.run_root / "frozen"
        frozen.mkdir()
        self.write_json(frozen / "inside.json", {"inside": True})

        self.external = self.base / "external"
        self.external.mkdir()
        self.document_path = self.external / "bundle.json"
        self.first = {"id": "a", "nested": {"value": 1}}
        self.second = {"id": "b", "nested": {"value": 2}}
        self.write_json(
            self.document_path,
            {"metadata": "synthetic", "transcripts": [self.first, self.second]},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def write_json(path: Path, value: object) -> None:
        path.write_bytes(canonical_json(value))
        path.chmod(0o444)

    def records(self) -> list[ExternalTranscriptRecord]:
        return [
            ExternalTranscriptRecord(
                str(self.document_path), ("transcripts", 0)
            ),
            ExternalTranscriptRecord(
                str(self.document_path), "/transcripts/1"
            ),
        ]

    def registry(
        self, records: list[ExternalTranscriptRecord] | None = None
    ) -> ExternalTranscriptRegistry:
        return ExternalTranscriptRegistry.load(
            self.records() if records is None else records,
            strict_parent=str(self.strict_parent),
            sealed_run_id=RUN_ID,
            run_root=str(self.run_root),
        )

    def test_unique_document_load_linear_pointer_index_and_fresh_decode(self) -> None:
        import strict_run.external as external_module

        with (
            mock.patch.object(
                external_module,
                "_load_external_document",
                wraps=external_module._load_external_document,
            ) as load_document,
            mock.patch.object(
                external_module,
                "_snapshot_strict_tree",
                wraps=external_module._snapshot_strict_tree,
            ) as snapshot_tree,
            mock.patch.object(
                external_module,
                "_resolve_pointer",
                wraps=external_module._resolve_pointer,
            ) as resolve_pointer,
            mock.patch.object(
                external_module.os,
                "read",
                wraps=os.read,
            ) as read_document,
        ):
            with self.registry() as registry:
                self.assertEqual(load_document.call_count, 1)
                self.assertEqual(snapshot_tree.call_count, 1)
                self.assertEqual(resolve_pointer.call_count, 2)
                self.assertEqual(read_document.call_count, 2)
                self.assertEqual(len(registry), 2)
                first_digest = sha256_bytes(canonical_json(self.first))
                second_digest = sha256_bytes(canonical_json(self.second))
                self.assertEqual(registry.digests, (first_digest, second_digest))
                selected = registry.get_exact(first_digest)
                selected["nested"]["value"] = 999
                self.assertEqual(registry.get_exact(first_digest), self.first)
                document_value = registry.documents[str(self.document_path)].value
                document_value["transcripts"][0]["id"] = "caller-mutated"
                self.assertEqual(
                    registry.documents[str(self.document_path)].value["transcripts"][0],
                    self.first,
                )
                self.assertEqual(resolve_pointer.call_count, 2)
                self.assertEqual(read_document.call_count, 2)
                with self.assertRaises(TypeError):
                    registry.canonical_bytes[first_digest] = b"forged"

    def test_lookup_rehashes_the_exact_cached_key(self) -> None:
        with self.registry() as registry:
            digest = sha256_bytes(canonical_json(self.first))
            with mock.patch(
                "strict_run.external.sha256_bytes", return_value="0" * 64
            ):
                with self.assertRaises(StrictRunError) as caught:
                    registry.get_exact(digest)
            self.assertEqual(caught.exception.code, "EXTERNAL_CACHE_CORRUPT")

    def test_pre_reservation_loader_never_opens_a_run_handle(self) -> None:
        planned_parent = self.base / "planned-strict-runs"
        planned_parent.mkdir()
        planned_root = planned_parent / RUN_ID
        with mock.patch(
            "strict_run.external.open_run_handle",
            side_effect=AssertionError("pre-reservation loader reopened the run"),
        ):
            document = load_pre_reservation_document(
                str(self.document_path),
                strict_parent=str(planned_parent),
                sealed_run_id=RUN_ID,
                run_root=str(planned_root),
            )
        try:
            self.assertEqual(document.value["transcripts"][0], self.first)
        finally:
            document.close()
        forbidden = planned_root / "not-yet-created.json"
        with self.assertRaises(StrictRunError) as caught:
            load_pre_reservation_document(
                str(forbidden),
                strict_parent=str(planned_parent),
                sealed_run_id=RUN_ID,
                run_root=str(planned_root),
            )
        self.assertEqual(caught.exception.code, "TRANSCRIPT_IN_RUN_TREE")
        self.assertFalse(planned_root.exists())

    def test_symlink_file_and_symlink_ancestor_are_rejected(self) -> None:
        file_alias = self.external / "file-alias.json"
        file_alias.symlink_to(self.document_path)
        ancestor_alias = self.base / "external-alias"
        ancestor_alias.symlink_to(self.external, target_is_directory=True)
        cases = (
            (
                ExternalTranscriptRecord(str(file_alias), ("transcripts", 0)),
                "UNSAFE_EXTERNAL_DOCUMENT",
            ),
            (
                ExternalTranscriptRecord(
                    str(ancestor_alias / "bundle.json"), ("transcripts", 0)
                ),
                "UNSAFE_EXTERNAL_PATH",
            ),
        )
        for record, code in cases:
            with self.subTest(path=record.path), self.assertRaises(StrictRunError) as caught:
                self.registry([record])
            self.assertEqual(caught.exception.code, code)

    def test_hardlink_inside_run_and_run_ancestor_alias_are_rejected(self) -> None:
        hardlink_source = self.external / "hardlink-source.json"
        hardlink_alias = self.external / "hardlink-alias.json"
        self.write_json(hardlink_source, {"transcript": self.first})
        os.link(hardlink_source, hardlink_alias)

        inside = self.run_root / "frozen" / "inside.json"
        tree_hardlink_alias = self.external / "tree-hardlink-alias.json"
        os.link(inside, tree_hardlink_alias)
        run_alias = self.base / "run-alias"
        run_alias.symlink_to(self.run_root, target_is_directory=True)
        cases = (
            (
                ExternalTranscriptRecord(str(hardlink_alias), ("transcript",)),
                "EXTERNAL_HARDLINK",
            ),
            (
                ExternalTranscriptRecord(str(tree_hardlink_alias), ()),
                "EXTERNAL_HARDLINK",
            ),
            (
                ExternalTranscriptRecord(str(inside), ()),
                "TRANSCRIPT_IN_RUN_TREE",
            ),
            (
                ExternalTranscriptRecord(
                    str(run_alias / "frozen" / "inside.json"), ()
                ),
                "UNSAFE_EXTERNAL_PATH",
            ),
        )
        for record, code in cases:
            with self.subTest(path=record.path), self.assertRaises(StrictRunError) as caught:
                self.registry([record])
            self.assertEqual(caught.exception.code, code)

    def test_external_mutation_invalidates_lookup_without_reread(self) -> None:
        registry = self.registry()
        digest = sha256_bytes(canonical_json(self.first))
        self.document_path.chmod(0o644)
        self.write_json(
            self.document_path,
            {"metadata": "synthetic", "transcripts": [self.second, self.first]},
        )
        try:
            with self.assertRaises(StrictRunError) as caught:
                registry.get_exact(digest)
            self.assertEqual(caught.exception.code, "EXTERNAL_DOCUMENT_MUTATED")
        finally:
            registry.close()

    def test_lookup_revalidates_every_held_document(self) -> None:
        other_path = self.external / "other-bundle.json"
        other_transcript = {"id": "other", "nested": {"value": 3}}
        self.write_json(other_path, {"selected": other_transcript})
        registry = self.registry(
            [
                ExternalTranscriptRecord(
                    str(self.document_path), ("transcripts", 0)
                ),
                ExternalTranscriptRecord(str(other_path), ("selected",)),
            ]
        )
        other_path.chmod(0o644)
        self.write_json(other_path, {"selected": {"id": "drifted"}})
        try:
            with self.assertRaises(StrictRunError) as caught:
                registry.get_exact(sha256_bytes(canonical_json(self.first)))
            self.assertEqual(caught.exception.code, "EXTERNAL_DOCUMENT_MUTATED")
        finally:
            registry.close()

    def test_parent_identity_mutation_and_document_replacement_fail(self) -> None:
        registry = self.registry()
        digest = sha256_bytes(canonical_json(self.first))
        self.write_json(self.external / "late-parent-entry.json", {"late": True})
        try:
            with self.assertRaises(StrictRunError) as parent_mutation:
                registry.get_exact(digest)
            self.assertEqual(
                parent_mutation.exception.code, "EXTERNAL_DOCUMENT_MUTATED"
            )
        finally:
            registry.close()

        registry = self.registry()
        original = self.document_path.read_bytes()
        self.document_path.unlink()
        self.document_path.write_bytes(original)
        self.document_path.chmod(0o444)
        try:
            with self.assertRaises(StrictRunError) as replacement:
                registry.get_exact(digest)
            self.assertEqual(
                replacement.exception.code, "EXTERNAL_DOCUMENT_MUTATED"
            )
        finally:
            registry.close()

    def test_strict_tree_mutation_after_construction_invalidates_lookup(self) -> None:
        registry = self.registry()
        self.write_json(
            self.run_root / "frozen" / "late-tree-entry.json", {"late": True}
        )
        try:
            with self.assertRaises(StrictRunError) as caught:
                registry.get_exact(sha256_bytes(canonical_json(self.first)))
            self.assertEqual(caught.exception.code, "STRICT_TREE_MUTATED")
        finally:
            registry.close()

    def test_duplicate_key_and_noncanonical_json_are_rejected(self) -> None:
        duplicate = self.external / "duplicate-key.json"
        duplicate.write_bytes(b'{"transcript":{"id":"x"},"transcript":{"id":"y"}}\n')
        duplicate.chmod(0o444)
        noncanonical = self.external / "noncanonical.json"
        noncanonical.write_bytes(b'{"transcript": {"id":"x"}}\n')
        noncanonical.chmod(0o444)
        cases = (
            (
                ExternalTranscriptRecord(str(duplicate), ("transcript",)),
                "DUPLICATE_JSON_KEY",
            ),
            (
                ExternalTranscriptRecord(str(noncanonical), ("transcript",)),
                "NONCANONICAL_JSON",
            ),
        )
        for record, code in cases:
            with self.subTest(path=record.path), self.assertRaises(StrictRunError) as caught:
                self.registry([record])
            self.assertEqual(caught.exception.code, code)

    def test_boolean_and_invalid_document_limits_are_rejected(self) -> None:
        for invalid in (True, False, 0, -1, (1 << 30) + 1, "1024", 1.5):
            with self.subTest(limit=invalid), self.assertRaises(StrictRunError) as caught:
                ExternalTranscriptRegistry.load(
                    self.records(),
                    strict_parent=str(self.strict_parent),
                    sealed_run_id=RUN_ID,
                    run_root=str(self.run_root),
                    max_document_bytes=invalid,
                )
            self.assertEqual(caught.exception.code, "EXTERNAL_LIMIT_INVALID")

    def test_duplicate_document_digest_pointer_and_transcript_digest_fail(self) -> None:
        duplicate_document = self.external / "duplicate-bundle.json"
        duplicate_document.write_bytes(self.document_path.read_bytes())
        duplicate_document.chmod(0o444)
        with self.assertRaises(StrictRunError) as raw_digest:
            self.registry(
                [
                    ExternalTranscriptRecord(
                        str(self.document_path), ("transcripts", 0)
                    ),
                    ExternalTranscriptRecord(
                        str(duplicate_document), ("transcripts", 1)
                    ),
                ]
            )
        self.assertEqual(
            raw_digest.exception.code, "DUPLICATE_EXTERNAL_DOCUMENT_DIGEST"
        )

        duplicate_record = ExternalTranscriptRecord(
            str(self.document_path), ("transcripts", 0)
        )
        with self.assertRaises(StrictRunError) as pointer:
            self.registry([duplicate_record, duplicate_record])
        self.assertEqual(pointer.exception.code, "DUPLICATE_TRANSCRIPT_POINTER")

        distinct_document = self.external / "distinct-bundle.json"
        self.write_json(
            distinct_document,
            {"metadata": "different-document", "selected": self.first},
        )
        with self.assertRaises(StrictRunError) as transcript_digest:
            self.registry(
                [
                    duplicate_record,
                    ExternalTranscriptRecord(str(distinct_document), ("selected",)),
                ]
            )
        self.assertEqual(
            transcript_digest.exception.code, "DUPLICATE_TRANSCRIPT_DIGEST"
        )

    def test_boolean_pointer_index_is_not_an_integer(self) -> None:
        with self.assertRaises(StrictRunError) as caught:
            ExternalTranscriptRecord(
                str(self.document_path), ("transcripts", True)
            )
        self.assertEqual(caught.exception.code, "JSON_POINTER_INVALID")

    def test_published_identity_replacement_fails_before_first_payload_read(
        self,
    ) -> None:
        import strict_run.external as external_module

        pinned_path = self.external / "pinned-transcript.json"
        payload = {"selected": self.first}
        with ExternalOutputReservation.create_for_existing_run(
            str(pinned_path),
            strict_parent=str(self.strict_parent),
            sealed_run_id=RUN_ID,
            run_root=str(self.run_root),
        ) as output:
            evidence = output.publish_json(payload)
        parent_fd = open_absolute_directory(str(self.external))
        try:
            expected_mount_id = int(mount_identity(parent_fd)["mount_id"])
        finally:
            os.close(parent_fd)
        record = ExternalTranscriptRecord(
            evidence.path,
            ("selected",),
            expected_sha256=evidence.sha256,
            expected_size=evidence.size,
            expected_identity=evidence.identity,
            expected_parent_identity=evidence.parent_identity,
            expected_mount_id=expected_mount_id,
        )

        pinned_path.unlink()
        self.write_json(pinned_path, {"selected": self.second})
        with (
            mock.patch.object(external_module.os, "read", wraps=os.read) as payload_read,
            self.assertRaises(StrictRunError) as caught,
        ):
            self.registry([record])
        self.assertEqual(
            caught.exception.code, "EXTERNAL_DOCUMENT_IDENTITY_MISMATCH"
        )
        self.assertEqual(payload_read.call_count, 0)


class StrictRunV6ExternalOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="exp7-output-v6-")
        self.base = Path(self.temporary.name)
        self.strict_parent = self.base / "strict-runs"
        self.strict_parent.mkdir()
        self.run_root = self.strict_parent / RUN_ID
        self.external = self.base / "external"
        self.external.mkdir()
        self.output = self.external / "reservation-transcript.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def preflight(self, path: Path | None = None) -> ExternalOutputReservation:
        return ExternalOutputReservation.create(
            str(self.output if path is None else path),
            strict_parent=str(self.strict_parent),
            sealed_run_id=RUN_ID,
            run_root=str(self.run_root),
        )

    def preflight_existing(
        self, path: Path | None = None
    ) -> ExternalOutputReservation:
        return ExternalOutputReservation.create_for_existing_run(
            str(self.output if path is None else path),
            strict_parent=str(self.strict_parent),
            sealed_run_id=RUN_ID,
            run_root=str(self.run_root),
        )

    def test_held_preflight_publishes_no_replace_with_full_readback(self) -> None:
        payload = {"schema": "synthetic-reservation-transcript", "value": 1}
        with self.preflight() as output:
            self.assertFalse(self.run_root.exists())
            self.assertFalse(self.output.exists())
            self.run_root.mkdir()
            evidence = output.publish_json(payload)
            self.assertEqual(self.output.read_bytes(), canonical_json(payload))
            self.assertEqual(evidence.sha256, sha256_bytes(canonical_json(payload)))
            self.assertEqual(evidence.size, len(canonical_json(payload)))
            self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o444)
            with self.assertRaises(StrictRunError) as second_publish:
                output.publish_json(payload)
            self.assertEqual(
                second_publish.exception.code,
                "EXTERNAL_OUTPUT_ALREADY_PUBLISHED",
            )

    def test_preflight_failures_create_neither_run_nor_output(self) -> None:
        existing = self.external / "existing.json"
        existing.write_bytes(b"sentinel")
        alias = self.base / "external-alias"
        alias.symlink_to(self.external, target_is_directory=True)
        cases = (
            (self.run_root / "inside.json", "TRANSCRIPT_IN_RUN_TREE"),
            (existing, "EXTERNAL_OUTPUT_EXISTS"),
            (alias / "through-alias.json", "UNSAFE_EXTERNAL_PATH"),
        )
        for path, code in cases:
            with self.subTest(path=path), self.assertRaises(StrictRunError) as caught:
                self.preflight(path)
            self.assertEqual(caught.exception.code, code)
            self.assertFalse(self.run_root.exists())
            self.assertFalse(self.output.exists())
        self.assertEqual(existing.read_bytes(), b"sentinel")

    def test_expected_mount_mismatch_blocks_external_preflight(self) -> None:
        import strict_run.external as external_module

        with (
            mock.patch.object(
                external_module,
                "mount_identity",
                return_value={"mount_id": 41},
            ),
            self.assertRaisesRegex(Exception, "mount differs from preflight"),
        ):
            ExternalOutputReservation.create(
                str(self.output),
                strict_parent=str(self.strict_parent),
                sealed_run_id=RUN_ID,
                run_root=str(self.run_root),
                expected_mount_id=42,
            )
        self.assertFalse(self.run_root.exists())
        self.assertFalse(self.output.exists())

    def test_parent_swap_after_reservation_is_rejected_before_output(self) -> None:
        reservation = self.preflight()
        moved = self.base / "external-before-swap"
        self.external.rename(moved)
        self.external.mkdir()
        self.run_root.mkdir()
        try:
            with self.assertRaises(StrictRunError) as caught:
                reservation.publish_json({"value": 1})
            self.assertEqual(caught.exception.code, "NAMESPACE_SUBSTITUTION")
            self.assertFalse(self.output.exists())
            self.assertFalse((moved / self.output.name).exists())
        finally:
            reservation.close()

    def test_output_exists_race_is_no_replace(self) -> None:
        reservation = self.preflight()
        self.run_root.mkdir()
        self.output.write_bytes(b"racing-writer")
        try:
            with self.assertRaises(StrictRunError) as caught:
                reservation.publish_json({"value": 1})
            self.assertEqual(caught.exception.code, "EXTERNAL_OUTPUT_EXISTS")
            self.assertEqual(self.output.read_bytes(), b"racing-writer")
        finally:
            reservation.close()

    def test_metadata_and_hardlink_race_before_readback_fails_closed(self) -> None:
        reservation = self.preflight()
        self.run_root.mkdir()
        alias = self.external / "reservation-transcript-alias.json"
        real_fsync = os.fsync
        mutated = False

        def mutate_after_parent_fsync(fd: int) -> None:
            nonlocal mutated
            real_fsync(fd)
            if fd == reservation._parent_fd and not mutated:
                self.output.chmod(0o666)
                os.link(self.output, alias)
                mutated = True

        try:
            with mock.patch(
                "strict_run.external.os.fsync",
                side_effect=mutate_after_parent_fsync,
            ), self.assertRaises(StrictRunError) as caught:
                reservation.publish_json({"value": 1})
            self.assertTrue(mutated)
            self.assertEqual(caught.exception.code, "EXTERNAL_OUTPUT_SUBSTITUTED")
            self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o666)
            self.assertEqual(self.output.stat().st_nlink, 2)
        finally:
            reservation.close()

    def test_new_run_tree_alias_is_rejected_before_output(self) -> None:
        reservation = self.preflight()
        self.run_root.symlink_to(self.external, target_is_directory=True)
        try:
            with self.assertRaises(StrictRunError) as caught:
                reservation.publish_json({"value": 1})
            self.assertEqual(caught.exception.code, "UNSAFE_RUN_ROOT")
            self.assertFalse(self.output.exists())
        finally:
            reservation.close()

    def test_existing_run_allows_expected_mutation_and_sibling_outputs(self) -> None:
        self.run_root.mkdir()
        second_path = self.external / "stage-transcripts.json"
        with (
            self.preflight_existing() as first,
            self.preflight_existing(second_path) as second,
        ):
            (self.run_root / "reservation-acceptance.json").write_bytes(b"accepted")
            first.publish_json({"kind": "acceptance"})
            second.publish_json({"kind": "stage"})
        self.assertEqual(
            self.output.read_bytes(), canonical_json({"kind": "acceptance"})
        )
        self.assertEqual(
            second_path.read_bytes(), canonical_json({"kind": "stage"})
        )
        self.assertEqual(stat.S_IMODE(self.output.stat().st_mode), 0o444)
        self.assertEqual(stat.S_IMODE(second_path.stat().st_mode), 0o444)

    def test_existing_run_preflight_rejects_output_path_aliases(self) -> None:
        self.run_root.mkdir()
        alias = self.base / "external-existing-alias"
        alias.symlink_to(self.external, target_is_directory=True)
        cases = (
            (self.run_root / "inside.json", "TRANSCRIPT_IN_RUN_TREE"),
            (alias / "through-alias.json", "UNSAFE_EXTERNAL_PATH"),
        )
        for path, code in cases:
            with self.subTest(path=path), self.assertRaises(StrictRunError) as caught:
                self.preflight_existing(path)
            self.assertEqual(caught.exception.code, code)
        self.assertFalse(self.output.exists())

    def test_existing_run_root_replacement_fails_before_output(self) -> None:
        self.run_root.mkdir()
        reservation = self.preflight_existing()
        moved = self.strict_parent / f"{RUN_ID}-moved"
        self.run_root.rename(moved)
        self.run_root.mkdir()
        try:
            with self.assertRaises(StrictRunError) as caught:
                reservation.publish_json({"value": 1})
            self.assertEqual(caught.exception.code, "NAMESPACE_SUBSTITUTION")
            self.assertFalse(self.output.exists())
        finally:
            reservation.close()

    def test_existing_run_rejects_alias_added_before_publication(self) -> None:
        self.run_root.mkdir()
        reservation = self.preflight_existing()
        (self.run_root / "external-alias").symlink_to(
            self.external, target_is_directory=True
        )
        try:
            with self.assertRaises(StrictRunError) as caught:
                reservation.publish_json({"value": 1})
            self.assertEqual(caught.exception.code, "UNSAFE_STRICT_TREE")
            self.assertFalse(self.output.exists())
        finally:
            reservation.close()


if __name__ == "__main__":
    unittest.main()
