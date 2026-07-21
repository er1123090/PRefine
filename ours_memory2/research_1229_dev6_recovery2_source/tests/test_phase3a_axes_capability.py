from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import critic
from ecpr import cli
from ecpr.evaluate import evaluate_paired
from ecpr.integrity import IMPLEMENTATION_MANIFEST
from ecpr.ledger import enforce_unsealed_output_arguments, final_paths


class CriticAxisTests(unittest.TestCase):
    def test_not_run_never_invokes_gold_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch.object(critic, "static_audit", return_value={"fixture": True}):
                axes = critic.audit_axes(
                    root,
                    gold_loader=lambda _path: (_ for _ in ()).throw(
                        AssertionError("gold loader must not run")
                    ),
                )
        self.assertEqual(axes["integrity"]["status"], "PASS")
        self.assertEqual(axes["execution"]["status"], "NOT_RUN")
        self.assertEqual(axes["performance"]["status"], "NOT_RUN")
        self.assertEqual(critic.critic_exit_code(axes), 1)
        self.assertEqual(critic.critic_exit_code(axes, allow_not_run=True), 0)

    def test_honest_performance_failure_preserves_integrity(self):
        axes = critic.classify_completed_summary(
            static={"fixture": True},
            strict={"case_count": 2},
            reported_pass=False,
            recomputed_gates={"minimum_delta_bmf1": False, "coverage": True},
        )
        self.assertEqual(axes["integrity"]["status"], "PASS")
        self.assertEqual(axes["execution"]["status"], "COMPLETE")
        self.assertEqual(axes["performance"]["status"], "FAIL")
        self.assertEqual(critic.critic_exit_code(axes), 1)

    def test_reported_pass_mismatch_is_integrity_failure(self):
        axes = critic.classify_completed_summary(
            static={},
            strict={},
            reported_pass=True,
            recomputed_gates={"minimum_delta_bmf1": False},
        )
        self.assertEqual(axes["integrity"]["status"], "FAIL")
        self.assertEqual(axes["execution"]["status"], "COMPLETE")
        self.assertEqual(critic.critic_exit_code(axes), 1)


class CapabilityBoundaryTests(unittest.TestCase):
    def test_cli_evaluate_requires_attempt_id(self):
        with self.assertRaises(SystemExit):
            cli._parser().parse_args(
                ["evaluate", "--baseline", "b.jsonl", "--candidate", "c.jsonl"]
            )
        with self.assertRaises(TypeError):
            evaluate_paired(
                root=Path("/fixture"),
                baseline_path=Path("b"),
                candidate_path=Path("c"),
                output=Path("o"),
                preregistration=Path("p"),
            )

    def test_unsealed_writes_reject_every_final_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, path in final_paths(root).items():
                with self.subTest(name=name):
                    with self.assertRaisesRegex(ValueError, "official final output"):
                        enforce_unsealed_output_arguments(root, path)

    def test_prepare_rejects_after_implementation_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / IMPLEMENTATION_MANIFEST
            path.parent.mkdir(parents=True)
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "prepare is forbidden"):
                cli.main(["prepare", "--root", str(root)])

    def test_preference_validation_precedes_any_gold_iteration(self):
        attempt = {"attempt_id": "fixture"}
        implementation = {"preference_slots_sha256": "0" * 64}
        with mock.patch("ecpr.evaluate.validate_final_attempt_v2", return_value=(attempt, implementation, "seal")), mock.patch(
            "ecpr.evaluate.enforce_final_evaluation_arguments"
        ), mock.patch("ecpr.evaluate.validate_run_dag", return_value={"equal_action_budget": True}), mock.patch(
            "ecpr.evaluate.validate_preference_slots", side_effect=ValueError("bad slots")
        ), mock.patch("ecpr.evaluate.iter_jsonl") as rows:
            with self.assertRaisesRegex(ValueError, "bad slots"):
                evaluate_paired(
                    root=Path("/fixture"),
                    baseline_path=Path("b"),
                    candidate_path=Path("c"),
                    output=Path("o"),
                    preregistration=Path("p"),
                    final_attempt_id="fixture",
                    runner_nonce=b"r" * 32,
                    gold_open_fd=99,
                )
        rows.assert_not_called()


if __name__ == "__main__":
    unittest.main()
