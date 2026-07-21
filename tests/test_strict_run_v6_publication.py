from __future__ import annotations

import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strict_run import (  # noqa: E402
    ArtifactEvidence,
    StagePublisher,
    StrictRunError,
    WriterIdentity,
    default_writer_policy,
)
from strict_run.publication import validate_publication_transcript  # noqa: E402
from strict_run.filesystem import RunDescriptorBinding, open_run_handle  # noqa: E402


RUN_ID = "exp7-strict-v6-20260717T120000Z-" + "f" * 32


class StrictRunV6PublicationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="exp7-publication-v6-")
        self.base = Path(self.temporary.name)
        self.parent = self.base / "strict-runs"
        self.parent.mkdir()
        self.run_root = self.parent / RUN_ID
        self.run_root.mkdir()
        self.writer = WriterIdentity("g0", "g0:publication-test", "g0-writer")
        self.publisher = StagePublisher(
            str(self.parent),
            RUN_ID,
            str(self.run_root),
            self.writer,
            default_writer_policy(),
        )
        self.payload = b"x"
        self.publication = self.publisher.publish_bytes(
            "frozen/payload.bin", self.payload
        )[-1]
        self.root_fd = os.open(
            self.run_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )

    def tearDown(self) -> None:
        os.close(self.root_fd)
        self.temporary.cleanup()

    def validate(self, transcript: object) -> dict[str, object]:
        return validate_publication_transcript(
            self.root_fd,
            RUN_ID,
            str(self.run_root),
            transcript,
            expected_relative_path="frozen/payload.bin",
            expected_writer=self.writer,
        )

    def test_valid_and_empty_publications_round_trip(self) -> None:
        self.validate(self.publication.transcript)
        empty = self.publisher.publish_bytes("frozen/empty.bin", b"")[-1]
        validated = validate_publication_transcript(
            self.root_fd,
            RUN_ID,
            str(self.run_root),
            empty.transcript,
            expected_relative_path="frozen/empty.bin",
            expected_writer=self.writer,
        )
        self.assertEqual(validated["bytes"], 0)
        self.assertEqual(
            [event["syscall"] for event in validated["events"]],
            ["openat_exclusive", "fsync", "fsync", "openat_readback", "read"],
        )

    def test_bound_publisher_rejects_descendant_mount_before_payload_open(
        self,
    ) -> None:
        import strict_run.publication as publication_module

        with open_run_handle(
            str(self.parent), RUN_ID, str(self.run_root)
        ) as handle:
            binding = RunDescriptorBinding.capture(
                handle.parent_fd, handle.root_fd
            )
        publisher = StagePublisher(
            str(self.parent),
            RUN_ID,
            str(self.run_root),
            self.writer,
            default_writer_policy(),
            run_binding=binding,
        )
        real_mount_identity = publication_module.mount_identity

        def substituted_mount(fd: int) -> dict[str, object]:
            evidence = real_mount_identity(fd)
            if os.readlink(f"/proc/self/fd/{fd}") == str(
                self.run_root / "frozen"
            ):
                evidence = copy.deepcopy(evidence)
                evidence["mount_id"] = int(evidence["mount_id"]) + 1
            return evidence

        with (
            mock.patch.object(
                publication_module,
                "mount_identity",
                side_effect=substituted_mount,
            ),
            self.assertRaises(StrictRunError) as caught,
        ):
            publisher.publish_bytes("frozen/mount-crossing.bin", b"forbidden")
        self.assertEqual(caught.exception.code, "MOUNT_SUBSTITUTION")
        self.assertFalse((self.run_root / "frozen" / "mount-crossing.bin").exists())

    def test_artifact_evidence_rejects_boolean_identity_numbers(self) -> None:
        forged = copy.deepcopy(self.publication.evidence.to_dict())
        forged["identity"]["dev"] = True
        with self.assertRaises(StrictRunError) as caught:
            ArtifactEvidence.from_dict(forged)
        self.assertEqual(caught.exception.code, "TRANSCRIPT_INVALID")

    def test_transcript_rejects_boolean_numeric_aliases(self) -> None:
        mutations = []

        top_bytes = copy.deepcopy(self.publication.transcript)
        top_bytes["bytes"] = True
        mutations.append(top_bytes)

        event_publisher = copy.deepcopy(self.publication.transcript)
        event_publisher["events"][0]["effective_publisher"]["effective_uid"] = False
        mutations.append(event_publisher)

        write_requested = copy.deepcopy(self.publication.transcript)
        write_event = next(
            event for event in write_requested["events"] if event["syscall"] == "write"
        )
        write_event["arguments"]["requested_bytes"] = True
        mutations.append(write_requested)

        descriptor_number = copy.deepcopy(self.publication.transcript)
        descriptor_number["events"][0]["descriptor"]["status_flags"] = True
        mutations.append(descriptor_number)

        for forged in mutations:
            with self.subTest(forged=forged), self.assertRaises(StrictRunError):
                self.validate(forged)

    def test_transcript_replays_recorded_chunk_digests(self) -> None:
        forged = copy.deepcopy(self.publication.transcript)
        read_event = next(
            event
            for event in forged["events"]
            if event["syscall"] == "read" and "chunk_sha256" in event["result"]
        )
        read_event["result"]["chunk_sha256"] = "0" * 64
        with self.assertRaises(StrictRunError) as caught:
            self.validate(forged)
        self.assertEqual(caught.exception.code, "TRANSCRIPT_STALE")

    def test_multi_chunk_transcript_replays_aggregate_digest(self) -> None:
        payload = b"m" * (1024 * 1024 + 17)
        publication = self.publisher.publish_bytes(
            "frozen/multi-chunk.bin", payload
        )[-1]
        validated = validate_publication_transcript(
            self.root_fd,
            RUN_ID,
            str(self.run_root),
            publication.transcript,
            expected_relative_path="frozen/multi-chunk.bin",
            expected_writer=self.writer,
        )
        read_events = [
            event for event in validated["events"] if event["syscall"] == "read"
        ]
        self.assertGreaterEqual(len(read_events), 2)
        self.assertEqual(validated["bytes"], len(payload))

    def test_reserved_path_guard_precedes_every_publication_api_mutation(self) -> None:
        attempts = (
            lambda: self.publisher.ensure_directory("checkpoints"),
            lambda: self.publisher.publish_bytes("terminal/blocked.json", b"{}"),
        )
        for attempt in attempts:
            with self.subTest(attempt=attempt), self.assertRaises(StrictRunError) as caught:
                attempt()
            self.assertEqual(caught.exception.code, "RESERVED_CONTROLLER_PATH")
            self.assertFalse((self.run_root / "checkpoints").exists())
            self.assertFalse((self.run_root / "terminal").exists())

        parent_fd = os.open(
            self.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            with self.assertRaises(StrictRunError) as caught:
                self.publisher.publish_bytes_on_reserved_descriptors(
                    parent_fd,
                    self.root_fd,
                    "checkpoints/cp0.json",
                    b"{}",
                )
        finally:
            os.close(parent_fd)
        self.assertEqual(caught.exception.code, "RESERVED_CONTROLLER_PATH")
        self.assertFalse((self.run_root / "checkpoints").exists())

    def test_full_replay_detects_same_inode_same_size_rewrite(self) -> None:
        artifact = self.run_root / "frozen" / "payload.bin"
        real_read = os.read
        calls = 0

        def mutate_after_segment_replay(fd: int, size: int) -> bytes:
            nonlocal calls
            block = real_read(fd, size)
            calls += 1
            if calls == 4:
                artifact.chmod(0o644)
                with artifact.open("r+b") as stream:
                    stream.write(b"y")
                    stream.flush()
                    os.fsync(stream.fileno())
                artifact.chmod(0o444)
            return block

        with mock.patch(
            "strict_run.publication.os.read", side_effect=mutate_after_segment_replay
        ):
            with self.assertRaises(StrictRunError) as caught:
                self.validate(self.publication.transcript)
        self.assertGreaterEqual(calls, 5)
        self.assertEqual(caught.exception.code, "TRANSCRIPT_STALE")


if __name__ == "__main__":
    unittest.main()
