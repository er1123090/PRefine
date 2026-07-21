from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from experiment_env.audited import (
    EXPECTED_REQUIRED_LABELS,
    FORBIDDEN_LABEL_RE,
    audited_doctor,
    audited_variants,
    build_audited_plan,
    selected_audit_files,
)


class AuditedEnvironmentTests(unittest.TestCase):
    def test_audit_selection_and_registry_policy(self) -> None:
        audit_path = next((ROOT / "configs" / "environment").glob("audit-overlay-782df2d*.json"))
        audit = json.loads(audit_path.read_bytes())
        selected = selected_audit_files(audit)
        self.assertEqual(len(selected), 108)
        self.assertEqual(sum(item["root_id"] == "experiments4" for item in selected), 62)
        self.assertEqual(sum(item["root_id"] == "experiments5" for item in selected), 46)
        variants = audited_variants(ROOT)
        labels = {item["variant_id"] for item in variants}
        self.assertEqual(len(variants), 84)
        self.assertTrue(EXPECTED_REQUIRED_LABELS <= labels)
        self.assertFalse(any(FORBIDDEN_LABEL_RE.search(label) for label in labels))
        self.assertNotIn("preprocess5_v1", labels)

    def test_doctor_validates_base_and_overlay(self) -> None:
        result = audited_doctor(ROOT)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["overlay"]["manifest_record_count"], 108)
        self.assertEqual(result["overlay"]["variant_count"], 84)
        self.assertEqual(result["base_semantics"], "experiments4_release_alias")

    def test_dry_run_resolves_sealed_entrypoint_without_execution(self) -> None:
        plan = build_audited_plan(ROOT, "eval4_single_legacy", 0, "TEST_DRY_RUN")
        self.assertFalse(plan["execution_supported"])
        self.assertEqual(plan["variant_id"], "eval4_single_legacy")
        self.assertIn("exp45-audited-overlay", " ".join(plan["argv"]))
        self.assertFalse((ROOT / "runs" / "TEST_DRY_RUN").exists())


if __name__ == "__main__":
    unittest.main()
