from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("exp7_cutover_validate", ROOT / "scripts/validate.py")
assert SPEC is not None and SPEC.loader is not None
validate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(validate)


class CutoverValidatorTests(unittest.TestCase):
    def run_validator(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts/validate.py"), *arguments],
            cwd=ROOT,
            env={
                "PATH": str(Path(sys.executable).parent) + ":/usr/bin:/bin",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
            check=False,
            capture_output=True,
            text=True,
        )

    def assert_completion_semantics(
        self,
        report: dict,
        *,
        mode: str,
        structural_readiness: str,
        project_cutover_complete: bool,
    ) -> None:
        expected_domain = (
            "project_structure"
            if mode == validate.PHASE_MODE
            else "project_physical_cutover"
        )
        self.assertEqual(report["completion_domain"], expected_domain)
        self.assertEqual(report["structural_readiness"], structural_readiness)
        self.assertEqual(report["experiment_completion"], "not_evaluated")
        self.assertFalse(report["actual_experiment_completion_claimed"])
        self.assertIs(
            report["project_cutover_complete"],
            project_cutover_complete,
        )
        self.assertIs(report["project_cutover_complete"], report["final_complete"])

    def test_current_repository_is_phase_ready_but_not_final_complete(self) -> None:
        self.assertEqual(
            validate.REPORT_SCHEMA,
            "experiments7-cutover-validation/v5",
        )
        phase = self.run_validator()
        self.assertEqual(phase.returncode, 0, phase.stdout + phase.stderr)
        phase_report = json.loads(phase.stdout)
        blocker_ids = [
            gap["id"] for gap in phase_report["completion_blockers"]
        ]
        self.assertEqual(phase_report["schema"], validate.REPORT_SCHEMA)
        self.assertEqual(phase_report["mode"], validate.PHASE_MODE)
        self.assertEqual(phase_report["status"], "phase_ready")
        self.assertTrue(phase_report["phase_ready"])
        self.assertFalse(phase_report["final_complete"])
        self.assertEqual(phase_report["failures"], [])
        self.assert_completion_semantics(
            phase_report,
            mode=validate.PHASE_MODE,
            structural_readiness="ready",
            project_cutover_complete=False,
        )
        self.assertEqual(
            blocker_ids,
            [
                "historical_paper_cleanup_planned_only",
                "cutover_receipt_missing_or_invalid",
            ],
        )
        self.assertEqual(phase_report["known_gaps"], phase_report["completion_blockers"])
        self.assertNotIn("only_vanillallm", blocker_ids)
        self.assertNotIn("evaluation_metrics_pending", blocker_ids)
        self.assertNotIn("canonical_experiment_config_missing", blocker_ids)
        self.assertNotIn("docs_navigation_gap", blocker_ids)

        final = self.run_validator("--mode", validate.FINAL_MODE)
        self.assertEqual(final.returncode, 2, final.stdout + final.stderr)
        final_report = json.loads(final.stdout)
        self.assertEqual(final_report["mode"], validate.FINAL_MODE)
        self.assertEqual(final_report["status"], "final_blocked")
        self.assertTrue(final_report["phase_ready"])
        self.assertFalse(final_report["final_complete"])
        self.assert_completion_semantics(
            final_report,
            mode=validate.FINAL_MODE,
            structural_readiness="ready",
            project_cutover_complete=False,
        )
        self.assertEqual(final_report["completion_blockers"], phase_report["completion_blockers"])

    def test_mode_status_and_exit_contract_fail_closed(self) -> None:
        self.assertEqual(
            validate.REPORT_SCHEMA,
            "experiments7-cutover-validation/v5",
        )
        passing_checks = (("synthetic", lambda root: {"status": "pass"}),)
        blocker = [{"id": "pending", "detail": "still pending"}]
        with mock.patch.object(validate, "completion_blockers", return_value=blocker):
            phase = validate.run_checks(ROOT, passing_checks)
            final = validate.run_checks(
                ROOT,
                passing_checks,
                mode=validate.FINAL_MODE,
            )
        self.assertEqual(phase["status"], "phase_ready")
        self.assertEqual(validate.report_exit_code(phase), 0)
        self.assert_completion_semantics(
            phase,
            mode=validate.PHASE_MODE,
            structural_readiness="ready",
            project_cutover_complete=False,
        )
        self.assertEqual(final["status"], "final_blocked")
        self.assertNotEqual(validate.report_exit_code(final), 0)
        self.assert_completion_semantics(
            final,
            mode=validate.FINAL_MODE,
            structural_readiness="ready",
            project_cutover_complete=False,
        )

        with mock.patch.object(validate, "completion_blockers", return_value=[]):
            clean_phase = validate.run_checks(ROOT, passing_checks)
            still_blocked = validate.run_checks(
                ROOT,
                passing_checks,
                mode=validate.FINAL_MODE,
            )
        self.assertEqual(clean_phase["status"], "phase_ready")
        self.assertFalse(clean_phase["final_complete"])
        self.assertEqual(still_blocked["status"], "final_blocked")
        self.assertFalse(still_blocked["final_complete"])
        self.assertNotEqual(validate.report_exit_code(still_blocked), 0)

        verified_receipt = {
            "externalization_authorized": True,
            "physical_cutover_complete": True,
            "schema": "experiments7-cutover-receipt-validation/v1",
            "status": "verified",
            "verified": True,
        }
        verified_local_state = {
            "schema": "experiments7-post-cutover-local-state-validation/v1",
            "status": "verified",
            "verified": True,
        }
        with (
            mock.patch.object(validate, "completion_blockers", return_value=[]),
            mock.patch.object(
                validate,
                "cutover_receipt_report",
                return_value=verified_receipt,
            ),
            mock.patch.object(
                validate,
                "post_cutover_local_state_report",
                return_value=verified_local_state,
            ),
        ):
            complete = validate.run_checks(
                ROOT,
                passing_checks,
                mode=validate.FINAL_MODE,
            )
        self.assertEqual(complete["status"], "final_complete")
        self.assertTrue(complete["final_complete"])
        self.assertEqual(validate.report_exit_code(complete), 0)
        self.assert_completion_semantics(
            complete,
            mode=validate.FINAL_MODE,
            structural_readiness="ready",
            project_cutover_complete=True,
        )

        failing_checks = (("synthetic", lambda root: 1 / 0),)
        failed = validate.run_checks(ROOT, failing_checks, mode=validate.FINAL_MODE)
        self.assertEqual(failed["status"], "validation_failed")
        self.assertFalse(failed["phase_ready"])
        self.assertNotEqual(validate.report_exit_code(failed), 0)
        self.assert_completion_semantics(
            failed,
            mode=validate.FINAL_MODE,
            structural_readiness="failed",
            project_cutover_complete=False,
        )
        with self.assertRaisesRegex(validate.CutoverValidationError, "unsupported"):
            validate.run_checks(ROOT, passing_checks, mode="ambiguous")

    def test_startup_failure_never_claims_experiment_or_cutover_completion(self) -> None:
        missing_root = ROOT / ".missing-validator-root"
        result = self.run_validator(
            "--root",
            str(missing_root),
            "--mode",
            validate.FINAL_MODE,
        )
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "validation_failed")
        self.assert_completion_semantics(
            report,
            mode=validate.FINAL_MODE,
            structural_readiness="failed",
            project_cutover_complete=False,
        )

    def test_completion_blockers_are_derived_from_current_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "archive").mkdir()
            (root / "archive/archive-map.json").write_text(
                json.dumps({"planned_only": False, "records": []}),
                encoding="utf-8",
            )
            (root / "docs").mkdir()
            (root / "README.md").write_text(
                "[Quickstart](docs/quickstart.md)\n",
                encoding="utf-8",
            )
            (root / "docs/README.md").write_text("# Docs\n", encoding="utf-8")
            (root / "docs/quickstart.md").write_text(
                "# Quickstart\n",
                encoding="utf-8",
            )
            self.assertEqual(
                [gap["id"] for gap in validate.completion_blockers(root)],
                ["cutover_receipt_missing_or_invalid"],
            )

            (root / "README.md").write_text("# Project\n", encoding="utf-8")
            self.assertEqual(
                [gap["id"] for gap in validate.completion_blockers(root)],
                [
                    "cutover_receipt_missing_or_invalid",
                    "docs_navigation_gap",
                ],
            )
            (root / "archive/archive-map.json").write_text(
                json.dumps({"planned_only": True, "records": []}),
                encoding="utf-8",
            )
            self.assertEqual(
                [gap["id"] for gap in validate.completion_blockers(root)],
                [
                    "historical_paper_cleanup_planned_only",
                    "cutover_receipt_missing_or_invalid",
                    "docs_navigation_gap",
                ],
            )

    def test_map_edits_and_source_deletions_cannot_bypass_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "archive").mkdir()
            (root / "docs").mkdir()
            (root / "archive/archive-map.json").write_text(
                json.dumps(
                    {
                        "schema": "experiments7-archive-map/v1",
                        "planned_only": False,
                        "records": [],
                    }
                ),
                encoding="utf-8",
            )
            (root / "README.md").write_text(
                "[Quickstart](docs/quickstart.md)\n",
                encoding="utf-8",
            )
            (root / "docs/README.md").write_text("# Docs\n", encoding="utf-8")
            (root / "docs/quickstart.md").write_text(
                "# Quickstart\n",
                encoding="utf-8",
            )

            blockers = validate.completion_blockers(root)
            self.assertEqual(
                [gap["id"] for gap in blockers],
                ["cutover_receipt_missing_or_invalid"],
            )

    def test_verified_receipt_cannot_override_present_cutover_payload(self) -> None:
        verified_receipt = {
            "externalization_authorized": True,
            "physical_cutover_complete": True,
            "schema": "experiments7-cutover-receipt-validation/v1",
            "status": "verified",
            "verified": True,
        }
        passing_checks = (("synthetic", lambda root: {"status": "pass"}),)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "archive").mkdir()
            (root / "archive/archive-map.json").write_text(
                json.dumps(
                    {
                        "planned_only": False,
                        "records": [
                            {
                                "externalization_allowed": True,
                                "path": "paper_outputs/raw",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            payload = root / "data/sources/still-active.json"
            payload.parent.mkdir(parents=True)
            payload.write_text("{}\n", encoding="utf-8")
            with (
                mock.patch.object(validate, "completion_blockers", return_value=[]),
                mock.patch.object(
                    validate,
                    "cutover_receipt_report",
                    return_value=verified_receipt,
                ),
            ):
                report = validate.run_checks(
                    root,
                    passing_checks,
                    mode=validate.FINAL_MODE,
                )
        self.assertEqual(report["status"], "final_blocked")
        self.assertFalse(report["final_complete"])
        self.assertFalse(report["post_cutover_local_state"]["verified"])
        self.assertIn("data/sources", report["post_cutover_local_state"]["error"])

    def test_completion_blockers_include_present_payload_with_verified_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive"
            archive.mkdir()
            (archive / "archive-map.json").write_text(
                json.dumps(
                    {
                        "planned_only": False,
                        "records": [
                            {
                                "externalization_allowed": True,
                                "path": "paper_outputs/raw",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            raw = root / "paper_outputs/raw"
            raw.mkdir(parents=True)
            (raw / "still-local.bin").write_bytes(b"payload")

            blockers = validate.completion_blockers(
                root,
                receipt_report={"verified": True},
            )

        self.assertIn(
            "historical_paper_cleanup_planned_only",
            {blocker["id"] for blocker in blockers},
        )

    def test_api_free_target_includes_registry_evaluator_and_config_entrypoint(self) -> None:
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="",
            stderr="Ran 114 tests in 1.0s\n\nOK\n",
        )
        with mock.patch.object(
            validate.subprocess,
            "run",
            return_value=completed,
        ) as invoked:
            report = validate.check_api_free_targeted_tests(ROOT)
        command = invoked.call_args.args[0]
        self.assertEqual(report["tests"], 114)
        expected_modules = {
            "tests.dataset.test_pipeline_path_safety",
            "tests.integration.test_run_suite_artifacts",
            "tests.integration.test_experiment_config_entrypoint",
            "tests.evaluation.test_manifest_evaluator",
            "tests.methods.test_adapter_contract",
            "tests.methods.test_method_registry",
            "tests.methods.test_vanilla_llm_runner",
            "tests.methods.test_rag_adapter",
            "tests.methods.test_mem0_adapter",
            "tests.methods.test_langmem_adapter",
            "tests.methods.test_preference_memory_adapter",
        }
        self.assertTrue(expected_modules.issubset(command))

    def test_forbidden_historical_import_fails_boundary_checker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            hostile = Path(temporary) / "hostile.py"
            hostile.write_text("from experiments4.vanillaLLM import run\n", encoding="utf-8")
            with self.assertRaisesRegex(validate.CutoverValidationError, "historical import"):
                validate.check_python_boundaries(ROOT, [hostile])

    def test_provenance_literal_exceptions_are_exact_and_do_not_exempt_imports(self) -> None:
        allowlisted = (
            ROOT / "src/exp7/provenance/archive_map.py",
            ROOT / "src/exp7/provenance/cutover_receipt.py",
        )
        for path in allowlisted:
            with self.subTest(path=path):
                report = validate.check_python_boundaries(ROOT, [path])
                self.assertTrue(report["historical_literal_exceptions"])

        with tempfile.TemporaryDirectory() as temporary:
            lookalike = Path(temporary) / "archive_map.py"
            lookalike.write_text('PATH = "paper_outputs/raw"\n', encoding="utf-8")
            with self.assertRaisesRegex(validate.CutoverValidationError, "historical path literal"):
                validate.check_python_boundaries(ROOT, [lookalike])

        for path in allowlisted:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path, violation="unlisted literal"), mock.patch.object(
                Path,
                "read_text",
                autospec=True,
                side_effect=lambda candidate, **kwargs: (
                    source + '\nEXTRA = "archive/not-allowlisted.json"\n'
                    if candidate == path
                    else source
                ),
            ):
                with self.assertRaisesRegex(validate.CutoverValidationError, "historical path literal"):
                    validate.check_python_boundaries(ROOT, [path])

            with self.subTest(path=path, violation="import"), mock.patch.object(
                Path,
                "read_text",
                autospec=True,
                side_effect=lambda candidate, **kwargs: (
                    "from experiments4 import legacy\n" if candidate == path else source
                ),
            ):
                with self.assertRaisesRegex(validate.CutoverValidationError, "historical import"):
                    validate.check_python_boundaries(ROOT, [path])

    def test_wrong_warning_value_produces_failure_and_nonzero_exit(self) -> None:
        good = {
            "schema_warnings": {
                "total": 463,
                "scenarios": dict(validate.SCHEMA_WARNING_COUNTS),
                "missing_slots": copy.deepcopy(validate.SCHEMA_WARNING_SLOTS),
            },
            "preference_groups": {
                "source_examples": {"unsupported_only": 477},
                "ignored_groups": dict(validate.IGNORED_GROUPS),
            },
            "multiturn_base_preference_conflicts": {
                "template_count": 25,
                "conflict_count": 0,
            },
        }
        bad = copy.deepcopy(good)
        bad["schema_warnings"]["total"] = 462
        with self.assertRaisesRegex(validate.CutoverValidationError, "must be 463"):
            validate.validate_warning_contract(bad)

        failed_report = {
            "checks": {},
            "completion_blockers": [],
            "failures": [{"check": "wrong_warning", "error": "must be 463"}],
            "final_complete": False,
            "known_gaps": [],
            "mode": validate.PHASE_MODE,
            "phase_ready": False,
            "schema": validate.REPORT_SCHEMA,
            "status": "validation_failed",
        }
        with mock.patch.object(validate, "run_checks", return_value=failed_report):
            self.assertNotEqual(validate.main([]), 0)


if __name__ == "__main__":
    unittest.main()
