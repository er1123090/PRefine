from __future__ import annotations

import copy
import os
from pathlib import Path
import stat
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strict_run import (  # noqa: E402
    ExternalTranscriptRecord,
    ExternalTranscriptRegistry,
    ReservationInputs,
    StrictRunError,
    WriterIdentity,
    accept_reservation,
    canonical_json,
    default_writer_policy,
    publish_owner_binding,
    reserve_strict_run,
    sha256_bytes,
    validate_sealed_run_id,
)
from strict_run.reservation import (  # noqa: E402
    validate_acceptance_payload,
    validate_reservation_payload,
)
import strict_run.reservation as reservation_module  # noqa: E402


RUN_ID = "exp7-strict-v6-20260717T120000Z-" + "a" * 32


def writer(role: str, task_prefix: str, writer_id: str) -> WriterIdentity:
    return WriterIdentity(role, f"{task_prefix}synthetic", writer_id)


def inputs(reservation_writer: WriterIdentity) -> ReservationInputs:
    digest = sha256_bytes(b"synthetic-v6")
    return ReservationInputs(
        project_id="experiments7-synthetic",
        owner_seed="owner-seed-synthetic",
        writer=reservation_writer,
        tool_sha256=digest,
        configuration_sha256=digest,
        runtime_sha256=digest,
        environment_path_policy_sha256=digest,
        canonical_argv=("python", "-B", "synthetic-reserve"),
    )


class StrictRunV6ReservationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="exp7-strict-v6-")
        self.base = Path(self.temporary.name)
        self.parent = self.base / "paper_outputs" / "strict-runs"
        self.parent.mkdir(parents=True)
        self.run_root = self.parent / RUN_ID
        self.reservation_writer = writer(
            "run_reservation", "reservation:", "run_reservation-writer"
        )
        self.acceptance_writer = writer(
            "reservation_acceptance",
            "acceptance:",
            "reservation_acceptance-writer",
        )
        self.g0_writer = writer("g0", "g0:", "g0-writer")
        self.policy = default_writer_policy()
        self.external_counter = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def reserve(self, **kwargs: object):
        return reserve_strict_run(
            str(self.parent),
            RUN_ID,
            str(self.run_root),
            inputs(self.reservation_writer),
            policy=self.policy,
            **kwargs,
        )

    def accept(
        self,
        reservation_transcript: object,
        acceptance_writer: WriterIdentity | None = None,
    ):
        self.external_counter += 1
        path = self.base / f"reservation-transcript-{self.external_counter}.json"
        payload = canonical_json(reservation_transcript)
        path.write_bytes(payload)
        with ExternalTranscriptRegistry(
            [ExternalTranscriptRecord(str(path))],
            strict_parent=str(self.parent),
            sealed_run_id=RUN_ID,
            run_root=str(self.run_root),
        ) as registry:
            return accept_reservation(
                str(self.parent),
                RUN_ID,
                str(self.run_root),
                registry,
                self.acceptance_writer
                if acceptance_writer is None
                else acceptance_writer,
                reservation_transcript_sha256=sha256_bytes(payload),
                policy=self.policy,
            )

    def test_canonical_id_accepts_only_exact_byte_spelling(self) -> None:
        self.assertEqual(validate_sealed_run_id(RUN_ID), RUN_ID)
        malformed = (
            "",
            ".",
            "..",
            f"/{RUN_ID}",
            f"{RUN_ID}/child",
            RUN_ID.upper(),
            RUN_ID + "-suffix",
            "exp7-strict-v06-20260230T120000Z-" + "a" * 32,
            "exp7-strict-v6-20260717T120000Z-" + "A" * 32,
            RUN_ID + "\x00",
        )
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(StrictRunError):
                validate_sealed_run_id(value)
        with self.assertRaises(StrictRunError) as caught:
            reserve_strict_run(
                str(self.parent),
                RUN_ID,
                str(self.run_root) + "/",
                inputs(self.reservation_writer),
                policy=self.policy,
            )
        self.assertEqual(caught.exception.code, "NONCANONICAL_PATH")
        self.assertFalse(self.run_root.exists())

    def test_reservation_is_prepublication_only_and_no_replace(self) -> None:
        result = self.reserve()
        reservation_path = self.run_root / "reservation.json"
        before = reservation_path.read_bytes()
        self.assertEqual(stat.S_IMODE(reservation_path.stat().st_mode), 0o444)
        self.assertNotIn("success", result.payload)
        self.assertNotIn("created_at", result.payload)
        self.assertNotIn("publication_transcript", result.payload)
        self.assertNotIn("cp0", str(result.payload).lower())
        events = result.transcript["events"]
        syscalls = [event["syscall"] for event in events]
        first_fsync = syscalls.index("fsync")
        self.assertEqual(syscalls[0], "openat_exclusive")
        self.assertTrue(syscalls[1:first_fsync])
        self.assertEqual(set(syscalls[1:first_fsync]), {"write"})
        self.assertEqual(
            syscalls[first_fsync:first_fsync + 3],
            ["fsync", "fsync", "openat_readback"],
        )
        self.assertTrue(syscalls[first_fsync + 3:])
        self.assertEqual(set(syscalls[first_fsync + 3:]), {"read"})
        self.assertIs(events[-1]["result"]["eof"], True)
        self.assertFalse(
            any(path.name.endswith("transcript.json") for path in self.run_root.iterdir())
        )
        with self.assertRaises(StrictRunError) as caught:
            self.reserve()
        self.assertEqual(caught.exception.code, "RUN_ROOT_EXISTS")
        self.assertEqual(reservation_path.read_bytes(), before)

    def test_reservation_publication_does_not_reopen_run_by_path(self) -> None:
        with mock.patch(
            "strict_run.publication.open_run_handle",
            side_effect=AssertionError("reservation must keep genesis descriptors open"),
        ):
            result = self.reserve()
        self.assertEqual(result.evidence.relative_path, "reservation.json")
        self.assertTrue((self.run_root / "reservation.json").is_file())

    def test_distinct_acceptance_and_owner_binding_are_causal(self) -> None:
        reservation = self.reserve()
        with self.assertRaises(StrictRunError) as caught:
            accept_reservation(
                str(self.parent),
                RUN_ID,
                str(self.run_root),
                reservation.transcript,  # type: ignore[arg-type]
                self.acceptance_writer,
                reservation_transcript_sha256=sha256_bytes(
                    canonical_json(reservation.transcript)
                ),
                policy=self.policy,
            )
        self.assertEqual(caught.exception.code, "EXTERNAL_REGISTRY_INVALID")
        self.assertFalse((self.run_root / "reservation-acceptance.json").exists())
        acceptance = self.accept(reservation.transcript)
        self.assertEqual(
            acceptance.payload["reservation_sha256"], reservation.evidence.sha256
        )
        self.assertNotIn("publication_success", acceptance.payload)
        acceptance_self_attesting = copy.deepcopy(acceptance.payload)
        acceptance_self_attesting["own_directory_sync"] = True
        with self.assertRaises(StrictRunError) as caught:
            validate_acceptance_payload(
                acceptance_self_attesting, RUN_ID, str(self.run_root)
            )
        self.assertEqual(caught.exception.code, "SCHEMA_INVALID")
        owner = publish_owner_binding(
            str(self.parent),
            RUN_ID,
            str(self.run_root),
            "owner-seed-synthetic",
            "experiments7-synthetic",
            self.g0_writer,
            policy=self.policy,
        )
        self.assertEqual(owner.evidence.relative_path, "owner-binding.json")
        before = (self.run_root / "reservation-acceptance.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.accept(reservation.transcript)
        self.assertEqual((self.run_root / "reservation-acceptance.json").read_bytes(), before)

    def test_same_writer_mismatch_early_and_self_sync_are_rejected(self) -> None:
        reservation = self.reserve()
        same_identity = writer(
            "reservation_acceptance", "acceptance:", self.reservation_writer.writer_id
        )
        with self.assertRaises(StrictRunError) as caught:
            self.accept(reservation.transcript, same_identity)
        self.assertEqual(caught.exception.code, "ACCEPTANCE_WRITER_NOT_DISTINCT")

        mismatch = copy.deepcopy(reservation.transcript)
        mismatch["identity"]["inode"] += 1
        with self.assertRaises(StrictRunError):
            self.accept(mismatch)
        early = copy.deepcopy(reservation.transcript)
        early["emitted_monotonic_ns"] = early["events"][-1]["completed_monotonic_ns"]
        with self.assertRaises(StrictRunError) as caught:
            self.accept(early)
        self.assertEqual(caught.exception.code, "TRANSCRIPT_EARLY")

        self_attesting = copy.deepcopy(reservation.payload)
        self_attesting["publication_success"] = True
        with self.assertRaises(StrictRunError) as caught:
            validate_reservation_payload(
                self_attesting, RUN_ID, str(self.parent), str(self.run_root)
            )
        self.assertEqual(caught.exception.code, "SCHEMA_INVALID")

        nested_self_attesting = copy.deepcopy(reservation.payload)
        nested_self_attesting["parent"]["descriptor"]["publication_success"] = True
        with self.assertRaises(StrictRunError) as caught:
            validate_reservation_payload(
                nested_self_attesting, RUN_ID, str(self.parent), str(self.run_root)
            )
        self.assertEqual(caught.exception.code, "SCHEMA_INVALID")

        wrong_flags = copy.deepcopy(reservation.transcript)
        wrong_flags["events"][0]["arguments"]["flags"] ^= os.O_EXCL
        with self.assertRaises(StrictRunError) as caught:
            self.accept(wrong_flags)
        self.assertEqual(caught.exception.code, "TRANSCRIPT_INVALID")

    def test_acceptance_never_reopens_a_replaced_run_root(self) -> None:
        reservation = self.reserve()
        original = self.base / "original-run-root"
        real_validate = validate_acceptance_payload

        def swap_after_validation(
            value: object,
            sealed_run_id: str,
            run_root: str,
        ) -> dict[str, object]:
            result = real_validate(value, sealed_run_id, run_root)
            self.run_root.rename(original)
            self.run_root.mkdir()
            return result

        with mock.patch(
            "strict_run.reservation.validate_acceptance_payload",
            side_effect=swap_after_validation,
        ), self.assertRaises(StrictRunError):
            self.accept(reservation.transcript)
        self.assertTrue((original / "reservation.json").is_file())
        self.assertFalse((original / "reservation-acceptance.json").exists())
        self.assertFalse((self.run_root / "reservation-acceptance.json").exists())

    def test_existing_symlink_noncanonical_and_ancestor_swap_fail_closed(self) -> None:
        other = self.base / "other"
        other.mkdir()
        symlink_parent = self.base / "strict-link"
        symlink_parent.symlink_to(self.parent, target_is_directory=True)
        with self.assertRaises(StrictRunError):
            reserve_strict_run(
                str(symlink_parent), RUN_ID, str(symlink_parent / RUN_ID),
                inputs(self.reservation_writer), policy=self.policy,
            )

        moved_parent = self.base / "strict-runs-moved"

        def swap() -> None:
            self.parent.rename(moved_parent)
            self.parent.symlink_to(other, target_is_directory=True)

        with self.assertRaises(StrictRunError) as caught:
            self.reserve(race_hook=swap)
        self.assertEqual(caught.exception.code, "NAMESPACE_SUBSTITUTION")
        self.assertEqual(list(other.iterdir()), [])
        self.assertFalse((moved_parent / RUN_ID).exists())

    def test_expected_parent_mount_mismatch_blocks_before_mkdir(self) -> None:
        with (
            mock.patch.object(
                reservation_module,
                "mount_identity",
                return_value={"mount_id": 41},
            ),
            mock.patch.object(reservation_module.os, "mkdir") as mkdir,
            self.assertRaisesRegex(Exception, "mount differs from preflight"),
        ):
            self.reserve(expected_parent_mount_id=42)
        mkdir.assert_not_called()
        self.assertFalse(self.run_root.exists())

    def test_expected_child_mount_mismatch_blocks_before_publication(self) -> None:
        mount_rows = iter(
            (
                {"mount_id": 42},
                {"mount_id": 42},
                {"mount_id": 43},
            )
        )
        with (
            mock.patch.object(
                reservation_module,
                "mount_identity",
                side_effect=lambda _fd: next(mount_rows),
            ),
            mock.patch.object(
                reservation_module.StagePublisher,
                "publish_json_on_reserved_descriptors",
            ) as publisher,
            self.assertRaisesRegex(
                Exception, "child descriptor mount differs before publication"
            ),
        ):
            self.reserve(expected_parent_mount_id=42)
        publisher.assert_not_called()
        self.assertTrue(self.run_root.is_dir())
        self.assertEqual(list(self.run_root.iterdir()), [])

    def test_racing_reservations_have_exactly_one_winner(self) -> None:
        def attempt() -> str:
            try:
                self.reserve()
            except (StrictRunError, FileExistsError):
                return "blocked"
            return "created"

        with ThreadPoolExecutor(max_workers=6) as pool:
            outcomes = list(pool.map(lambda _: attempt(), range(6)))
        self.assertEqual(outcomes.count("created"), 1)
        self.assertEqual(outcomes.count("blocked"), 5)

    def test_crash_boundaries_never_become_accepted_or_reusable(self) -> None:
        points = (
            "child_created", "exclusive_open", "write_complete", "file_sync",
            "directory_sync",
        )
        for index, point in enumerate(points):
            run_id = f"exp7-strict-v6-20260717T12000{index}Z-" + f"{index + 1:032x}"
            run_root = self.parent / run_id
            with self.subTest(point=point):
                with self.assertRaises(StrictRunError) as caught:
                    reserve_strict_run(
                        str(self.parent), run_id, str(run_root),
                        inputs(self.reservation_writer), policy=self.policy, fail_after=point,
                    )
                self.assertEqual(caught.exception.code, "INJECTED_CRASH")
                self.assertTrue(run_root.is_dir())
                transcript_path = self.base / f"invalid-transcript-{index}.json"
                transcript_payload = canonical_json({})
                transcript_path.write_bytes(transcript_payload)
                with ExternalTranscriptRegistry(
                    [ExternalTranscriptRecord(str(transcript_path))],
                    strict_parent=str(self.parent),
                    sealed_run_id=run_id,
                    run_root=str(run_root),
                ) as registry, self.assertRaises(Exception):
                    accept_reservation(
                        str(self.parent),
                        run_id,
                        str(run_root),
                        registry,
                        self.acceptance_writer,
                        reservation_transcript_sha256=sha256_bytes(
                            transcript_payload
                        ),
                        policy=self.policy,
                    )
                with self.assertRaises(StrictRunError) as reused:
                    reserve_strict_run(
                        str(self.parent), run_id, str(run_root),
                        inputs(self.reservation_writer), policy=self.policy,
                    )
                self.assertEqual(reused.exception.code, "RUN_ROOT_EXISTS")


if __name__ == "__main__":
    unittest.main()
