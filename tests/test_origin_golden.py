from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil
import stat
import sys
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facade.planner import profile_selection  # noqa: E402
from facade.registry import FacadeError, load_bundle  # noqa: E402
from facade.runner import compare_reference, run_external, run_fixture  # noqa: E402


REQUIRED = {
    "infer4_compat", "infer5_native", "infer6_release",
    "eval4_legacy", "eval5_canonical", "eval6_release",
    "eval4_mt_parse_0103a", "eval4_mt_parse_0103b",
}


def _remove_test_run(target: Path) -> None:
    """Remove a test-owned run after restoring owner permissions."""
    if not target.exists():
        return
    for directory, directories, files in os.walk(target, topdown=False):
        root = Path(directory)
        for name in files:
            path = root / name
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | stat.S_IWUSR)
        for name in directories:
            path = root / name
            if not path.is_symlink():
                path.chmod(path.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
    target.chmod(target.stat().st_mode | stat.S_IWUSR | stat.S_IXUSR)
    shutil.rmtree(target)


class OriginGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bundle = load_bundle(ROOT)

    def profile_for(self, variant_id: str) -> str:
        return sorted(
            profile_id
            for profile_id, profile in self.bundle.profiles_by_id.items()
            if variant_id in profile["stages"].values()
        )[0]

    def test_required_stable_variant_fixture_routes(self) -> None:
        for variant_id in sorted(REQUIRED - {"eval4_mt_parse_0103a", "eval4_mt_parse_0103b"}):
            profile_id = self.profile_for(variant_id)
            run_id = f"test-fixture-{variant_id}-{uuid.uuid4().hex}"
            target = ROOT / "runs" / run_id
            try:
                result = run_fixture(self.bundle, profile_id, f"runs/{run_id}", "routing_basic")
                self.assertEqual(result["state"], "PASS")
                self.assertIn(variant_id, result["selection"]["selected_stages"].values())
                self.assertFalse(result["fixture_output"]["semantic_execution"])
            finally:
                _remove_test_run(target)

    def test_representative_origin_profiles_keep_exact_variants(self) -> None:
        cases = {
            "exp4_api_eval4": {"inference": "infer4_api"},
            "exp4_vllm_eval4": {"inference": "infer4_vllm"},
            "exp5_v1_api_eval5": {"inference": "infer5_v1_api"},
            "exp5_v1_vllm_eval5": {"inference": "infer5_v1_vllm"},
            "exp5_v2_api_eval5": {"inference": "infer5_v2_api"},
            "exp5_v2_vllm_eval5": {"inference": "infer5_v2_vllm"},
            "exp5_v3_api_eval5": {"inference": "infer5_v3_api"},
            "exp5_v3_vllm_eval5": {"inference": "infer5_v3_vllm"},
            "exp6_api_eval6": {"inference": "infer6_api"},
            "exp6_vllm_eval6": {"inference": "infer6_vllm"},
            "emem4_eval4": {"inference": "emem4_infer"},
            "emem5_eval5": {"inference": "emem5_infer"},
            "session4_report_eval4": {"reporting": "report4_session_memory"},
            "session6_report_eval6": {"reporting": "report6_session_memory"},
        }
        for profile_id, expected in cases.items():
            selection = profile_selection(
                self.bundle, profile_id, execution="validate", run_id=None, output_root=None
            )
            for stage, variant_id in expected.items():
                self.assertEqual(selection["selected_stages"][stage], variant_id)

    def test_reference_profiles_never_execute_parser(self) -> None:
        cases = (
            ("parser_reference_0103a_eval4", "parser_reference_0103a", "eval4_mt_parse_0103a"),
            ("parser_reference_0103b_eval4", "parser_reference_0103b", "eval4_mt_parse_0103b"),
        )
        for profile_id, fixture_id, variant_id in cases:
            run_id = f"test-reference-{uuid.uuid4().hex}"
            target = ROOT / "runs" / run_id
            try:
                result = compare_reference(self.bundle, profile_id, f"runs/{run_id}", fixture_id)
                self.assertEqual(result["comparison"]["reference_variant_id"], variant_id)
                self.assertFalse(result["comparison"]["live_parser_executed"])
                self.assertFalse(result["selection"]["external_runnable"])
            finally:
                _remove_test_run(target)

            external_run_id = f"test-reference-external-{uuid.uuid4().hex}"
            external_target = ROOT / "runs" / external_run_id
            try:
                with self.assertRaises(FacadeError) as caught:
                    run_external(
                        self.bundle, profile_id, f"runs/{external_run_id}",
                        allow_external=True,
                    )
                self.assertEqual(caught.exception.code, "REFERENCE_ONLY_EXECUTION_FORBIDDEN")
                self.assertEqual(
                    caught.exception.detail["reference_variant_ids"], [variant_id]
                )
                self.assertTrue((external_target / "selection.json").is_file())
                self.assertTrue((external_target / "blocked.json").is_file())
            finally:
                _remove_test_run(external_target)

    def test_external_refusal_occurs_after_selection(self) -> None:
        run_id = f"test-external-{uuid.uuid4().hex}"
        target = ROOT / "runs" / run_id
        try:
            with self.assertRaises(FacadeError) as caught:
                run_external(
                    self.bundle, "exp4_compat_eval4", f"runs/{run_id}", allow_external=True
                )
            self.assertEqual(caught.exception.code, "EXTERNAL_PREREQUISITES_UNAVAILABLE")
            self.assertTrue((target / "selection.json").is_file())
            self.assertTrue((target / "blocked.json").is_file())
        finally:
            _remove_test_run(target)

    def test_cleanup_removes_read_only_run_tree(self) -> None:
        target = ROOT / "runs" / f"test-cleanup-{uuid.uuid4().hex}"
        child = target / "sealed"
        child.mkdir(parents=True)
        artifact = child / "artifact.json"
        artifact.write_text("{}\n", encoding="utf-8")
        artifact.chmod(0o444)
        child.chmod(0o555)
        target.chmod(0o555)
        try:
            _remove_test_run(target)
            self.assertFalse(target.exists())
        finally:
            _remove_test_run(target)

    def test_plan_is_deterministic_except_run_identity(self) -> None:
        first = profile_selection(
            self.bundle, "exp5_v3_api_eval5", execution="plan",
            run_id="one", output_root=ROOT / "runs/one",
        )
        second = profile_selection(
            self.bundle, "exp5_v3_api_eval5", execution="plan",
            run_id="two", output_root=ROOT / "runs/two",
        )
        for value in (first, second):
            value.pop("selection_sha256")
            value["run_id"] = None
            value["output_root"] = None
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
