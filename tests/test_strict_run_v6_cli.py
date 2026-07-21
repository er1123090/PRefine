from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import stat
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from strict_run import WriterIdentity, canonical_json, sha256_bytes  # noqa: E402


RUN_ID = "exp7-strict-v6-20260717T140000Z-" + "c" * 32


class StrictRunV6CliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="exp7-strict-cli-v6-")
        self.base = Path(self.temporary.name)
        self.parent = self.base / "paper_outputs" / "strict-runs"
        self.parent.mkdir(parents=True)
        self.run_root = self.parent / RUN_ID
        self.external = self.base / "external"
        self.external.mkdir()
        digest = sha256_bytes(b"cli-synthetic")
        self.inputs = self.external / "reservation-inputs.json"
        self.inputs.write_bytes(
            canonical_json(
                {
                    "project_id": "experiments7-cli-synthetic",
                    "owner_seed": "cli-owner-seed",
                    "writer": WriterIdentity(
                        "run_reservation",
                        "reservation:cli",
                        "run_reservation-writer",
                    ).to_dict(),
                    "tool_sha256": digest,
                    "configuration_sha256": digest,
                    "runtime_sha256": digest,
                    "environment_path_policy_sha256": digest,
                    "canonical_argv": ["python", "-B", "reserve-cli"],
                }
            )
        )
        self.acceptance_writer = self.external / "acceptance-writer.json"
        self.acceptance_writer.write_bytes(
            canonical_json(
                WriterIdentity(
                    "reservation_acceptance",
                    "acceptance:cli",
                    "reservation_acceptance-writer",
                ).to_dict()
            )
        )
        self.environment = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.base),
            "TMPDIR": str(self.base),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def run_script(self, script: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            self.script_command(script, *arguments),
            cwd=ROOT,
            env=self.environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )

    @staticmethod
    def script_command(script: str, *arguments: str) -> list[str]:
        return [
            sys.executable,
            "-B",
            str(ROOT / "scripts" / "strict_run" / script),
            *arguments,
        ]

    def test_explicit_reserve_then_distinct_acceptance_cli(self) -> None:
        reservation_transcript = self.external / "reservation-transcript.json"
        reserve = self.run_script(
            "reserve.py",
            "--strict-parent", str(self.parent),
            "--sealed-run-id", RUN_ID,
            "--run-root", str(self.run_root),
            "--inputs-json", str(self.inputs),
            "--external-transcript", str(reservation_transcript),
        )
        self.assertEqual(reserve.returncode, 0, reserve.stderr)
        self.assertEqual(json.loads(reserve.stdout)["sealed_run_id"], RUN_ID)
        self.assertTrue(reservation_transcript.is_file())
        self.assertFalse((self.run_root / "reservation-transcript.json").exists())

        acceptance_transcript = self.external / "acceptance-transcript.json"
        accept = self.run_script(
            "accept_reservation.py",
            "--strict-parent", str(self.parent),
            "--sealed-run-id", RUN_ID,
            "--run-root", str(self.run_root),
            "--reservation-transcript", str(reservation_transcript),
            "--writer-json", str(self.acceptance_writer),
            "--external-transcript", str(acceptance_transcript),
        )
        self.assertEqual(accept.returncode, 0, accept.stderr)
        self.assertTrue((self.run_root / "reservation-acceptance.json").is_file())
        self.assertTrue(acceptance_transcript.is_file())

    def test_acceptance_output_preflight_failure_precedes_mutation(self) -> None:
        reservation_transcript = self.external / "reservation-before-failure.json"
        reserve = self.run_script(
            "reserve.py",
            "--strict-parent", str(self.parent),
            "--sealed-run-id", RUN_ID,
            "--run-root", str(self.run_root),
            "--inputs-json", str(self.inputs),
            "--external-transcript", str(reservation_transcript),
        )
        self.assertEqual(reserve.returncode, 0, reserve.stderr)
        blocked_output = self.external / "acceptance-output-exists.json"
        blocked_output.write_bytes(b"sentinel")
        accept = self.run_script(
            "accept_reservation.py",
            "--strict-parent", str(self.parent),
            "--sealed-run-id", RUN_ID,
            "--run-root", str(self.run_root),
            "--reservation-transcript", str(reservation_transcript),
            "--writer-json", str(self.acceptance_writer),
            "--external-transcript", str(blocked_output),
        )
        self.assertNotEqual(accept.returncode, 0)
        self.assertIn("EXTERNAL_OUTPUT_EXISTS", accept.stderr)
        self.assertFalse((self.run_root / "reservation-acceptance.json").exists())
        self.assertEqual(blocked_output.read_bytes(), b"sentinel")

    def test_checkpoint_output_preflight_failure_precedes_inputs_and_mutation(self) -> None:
        self.run_root.mkdir()
        blocked_output = self.external / "checkpoint-output-exists.json"
        blocked_output.write_bytes(b"sentinel")
        stage_output = self.external / "checkpoint-stage-transcripts.json"
        result = self.run_script(
            "publish_checkpoint.py",
            "--strict-parent", str(self.parent),
            "--sealed-run-id", RUN_ID,
            "--run-root", str(self.run_root),
            "--checkpoint", "0",
            "--stage-bundle", str(self.external / "missing-stage-bundle.json"),
            "--controller-json", str(self.external / "missing-controller.json"),
            "--external-transcript", str(blocked_output),
            "--external-stage-transcripts", str(stage_output),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EXTERNAL_OUTPUT_EXISTS", result.stderr)
        self.assertFalse((self.run_root / "checkpoints").exists())
        self.assertFalse(stage_output.exists())
        self.assertEqual(blocked_output.read_bytes(), b"sentinel")

    def test_checkpoint_duplicate_outputs_fail_before_mutation(self) -> None:
        self.run_root.mkdir()
        output = self.external / "same-checkpoint-output.json"
        result = self.run_script(
            "publish_checkpoint.py",
            "--strict-parent", str(self.parent),
            "--sealed-run-id", RUN_ID,
            "--run-root", str(self.run_root),
            "--checkpoint", "0",
            "--stage-bundle", str(self.external / "missing-stage-bundle.json"),
            "--controller-json", str(self.external / "missing-controller.json"),
            "--external-transcript", str(output),
            "--external-stage-transcripts", str(output),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EXTERNAL_OUTPUT_PATH_COLLISION", result.stderr)
        self.assertFalse((self.run_root / "checkpoints").exists())
        self.assertFalse(output.exists())

    def test_cp3_cli_requires_complete_prior_transcript_bundle_before_mutation(self) -> None:
        self.run_root.mkdir()
        transcript_output = self.external / "cp3-checkpoint-transcript.json"
        stage_output = self.external / "cp3-stage-transcripts.json"
        result = self.run_script(
            "publish_checkpoint.py",
            "--strict-parent", str(self.parent),
            "--sealed-run-id", RUN_ID,
            "--run-root", str(self.run_root),
            "--checkpoint", "3",
            "--stage-bundle", str(self.external / "missing-stage-bundle.json"),
            "--controller-json", str(self.external / "missing-controller.json"),
            "--external-transcript", str(transcript_output),
            "--external-stage-transcripts", str(stage_output),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PRIOR_TRANSCRIPT_BUNDLE_REQUIRED", result.stderr)
        self.assertFalse((self.run_root / "checkpoints").exists())
        self.assertFalse(transcript_output.exists())
        self.assertFalse(stage_output.exists())

    def test_prior_transcript_bundle_and_count_are_atomic_cli_inputs(self) -> None:
        self.run_root.mkdir()
        prior = self.external / "prior-transcripts.json"
        prior.write_bytes(canonical_json({"transcripts": []}))
        transcript_output = self.external / "paired-checkpoint-transcript.json"
        stage_output = self.external / "paired-stage-transcripts.json"
        result = self.run_script(
            "publish_checkpoint.py",
            "--strict-parent", str(self.parent),
            "--sealed-run-id", RUN_ID,
            "--run-root", str(self.run_root),
            "--checkpoint", "0",
            "--stage-bundle", str(self.external / "missing-stage-bundle.json"),
            "--controller-json", str(self.external / "missing-controller.json"),
            "--prior-transcript-bundle", str(prior),
            "--external-transcript", str(transcript_output),
            "--external-stage-transcripts", str(stage_output),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("PRIOR_TRANSCRIPT_BUNDLE_INVALID", result.stderr)
        self.assertFalse((self.run_root / "checkpoints").exists())
        self.assertFalse(transcript_output.exists())
        self.assertFalse(stage_output.exists())

    def test_checkpoint_cli_publishes_two_preflighted_sibling_outputs(self) -> None:
        from tests.test_strict_run_v6_checkpoints import (
            RUN_ID as CHECKPOINT_RUN_ID,
            SyntheticStrictRun,
        )

        fixture = SyntheticStrictRun(self.base / "checkpoint-cli-fixture")
        stage = fixture.bootstrap_pre_cp0()
        stage_bundle = self.external / "checkpoint-stage-bundle.json"
        stage_bundle.write_bytes(
            canonical_json(
                {
                    "publications": [
                        {
                            "evidence": publication.evidence.to_dict(),
                            "transcript": publication.transcript,
                        }
                        for publication in stage
                    ]
                }
            )
        )
        controller = self.external / "checkpoint-controller.json"
        controller.write_bytes(canonical_json(fixture.writers["controller"].to_dict()))
        transcript_output = self.external / "checkpoint-transcript.json"
        stage_output = self.external / "checkpoint-stage-transcripts.json"
        child_evidence_arguments = [
            argument
            for record in fixture.child_records
            for argument in ("--external-evidence-document", record.path)
        ]
        result = self.run_script(
            "publish_checkpoint.py",
            "--strict-parent", str(fixture.parent),
            "--sealed-run-id", CHECKPOINT_RUN_ID,
            "--run-root", str(fixture.run_root),
            "--checkpoint", "0",
            "--stage-bundle", str(stage_bundle),
            "--controller-json", str(controller),
            *child_evidence_arguments,
            "--external-transcript", str(transcript_output),
            "--external-stage-transcripts", str(stage_output),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((fixture.run_root / "checkpoints" / "cp0.json").is_file())
        self.assertTrue(transcript_output.is_file())
        self.assertTrue(stage_output.is_file())
        self.assertEqual(stat.S_IMODE(transcript_output.stat().st_mode), 0o444)
        self.assertEqual(stat.S_IMODE(stage_output.stat().st_mode), 0o444)

    def test_external_transcript_path_inside_run_is_rejected_without_artifact(self) -> None:
        second_id = "exp7-strict-v6-20260717T140001Z-" + "d" * 32
        second_root = self.parent / second_id
        forbidden_output = second_root / "transcript.json"
        result = self.run_script(
            "reserve.py",
            "--strict-parent", str(self.parent),
            "--sealed-run-id", second_id,
            "--run-root", str(second_root),
            "--inputs-json", str(self.inputs),
            "--external-transcript", str(forbidden_output),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TRANSCRIPT_IN_RUN_TREE", result.stderr)
        self.assertFalse(second_root.exists())
        self.assertFalse(forbidden_output.exists())

    def test_reserve_preflight_rejects_symlink_input_and_output_parent(self) -> None:
        input_alias = self.external / "reservation-inputs-alias.json"
        input_alias.symlink_to(self.inputs)
        input_id = "exp7-strict-v6-20260717T140002Z-" + "e" * 32
        input_root = self.parent / input_id
        input_output = self.external / "input-alias-transcript.json"
        input_result = self.run_script(
            "reserve.py",
            "--strict-parent", str(self.parent),
            "--sealed-run-id", input_id,
            "--run-root", str(input_root),
            "--inputs-json", str(input_alias),
            "--external-transcript", str(input_output),
        )
        self.assertNotEqual(input_result.returncode, 0)
        self.assertIn("UNSAFE_EXTERNAL_DOCUMENT", input_result.stderr)
        self.assertFalse(input_root.exists())
        self.assertFalse(input_output.exists())

        output_alias = self.base / "external-output-alias"
        output_alias.symlink_to(self.external, target_is_directory=True)
        output_id = "exp7-strict-v6-20260717T140003Z-" + "f" * 32
        output_root = self.parent / output_id
        aliased_output = output_alias / "aliased-transcript.json"
        output_result = self.run_script(
            "reserve.py",
            "--strict-parent", str(self.parent),
            "--sealed-run-id", output_id,
            "--run-root", str(output_root),
            "--inputs-json", str(self.inputs),
            "--external-transcript", str(aliased_output),
        )
        self.assertNotEqual(output_result.returncode, 0)
        self.assertIn("UNSAFE_EXTERNAL_PATH", output_result.stderr)
        self.assertFalse(output_root.exists())
        self.assertFalse(aliased_output.exists())

    def test_reserve_preflight_rejects_existing_output_without_run(self) -> None:
        output = self.external / "already-exists.json"
        output.write_bytes(b"existing-output")
        run_id = "exp7-strict-v6-20260717T140004Z-" + "a" * 32
        run_root = self.parent / run_id
        result = self.run_script(
            "reserve.py",
            "--strict-parent", str(self.parent),
            "--sealed-run-id", run_id,
            "--run-root", str(run_root),
            "--inputs-json", str(self.inputs),
            "--external-transcript", str(output),
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("EXTERNAL_OUTPUT_EXISTS", result.stderr)
        self.assertFalse(run_root.exists())
        self.assertEqual(output.read_bytes(), b"existing-output")

    def test_reserve_rejects_output_parent_swap_after_run_creation(self) -> None:
        run_id = "exp7-strict-v6-20260717T140005Z-" + "b" * 32
        run_root = self.parent / run_id
        slow_inputs = self.base / "slow-reservation-inputs.json"
        row = json.loads(self.inputs.read_bytes())
        row["canonical_argv"] = ["python", "x" * (4 * 1024 * 1024)]
        slow_inputs.write_bytes(canonical_json(row))
        output_parent = self.base / "swap-output"
        output_parent.mkdir()
        output = output_parent / "reservation-transcript.json"
        moved_parent = self.base / "swap-output-before-rename"
        process = subprocess.Popen(
            self.script_command(
                "reserve.py",
                "--strict-parent", str(self.parent),
                "--sealed-run-id", run_id,
                "--run-root", str(run_root),
                "--inputs-json", str(slow_inputs),
                "--external-transcript", str(output),
            ),
            cwd=ROOT,
            env=self.environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.monotonic() + 30
        while not run_root.exists() and process.poll() is None:
            if time.monotonic() >= deadline:
                process.kill()
                self.fail("reserve CLI did not create the reserved root in time")
            time.sleep(0.0005)
        self.assertTrue(run_root.exists())
        output_parent.rename(moved_parent)
        output_parent.mkdir()
        stdout, stderr = process.communicate(timeout=30)
        self.assertNotEqual(process.returncode, 0, stdout)
        self.assertIn("NAMESPACE_SUBSTITUTION", stderr)
        self.assertTrue((run_root / "reservation.json").is_file())
        self.assertFalse(output.exists())
        self.assertFalse((moved_parent / output.name).exists())


if __name__ == "__main__":
    unittest.main()
