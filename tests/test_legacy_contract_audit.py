from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/provenance/audit_legacy_contract.py"
SPEC = importlib.util.spec_from_file_location("audit_legacy_contract", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LegacyContractAuditTests(unittest.TestCase):
    def test_current_legacy_contract_defects_are_reproduced(self) -> None:
        defects, summary = MODULE.build_audit(ROOT)
        self.assertEqual(summary["legacy_admission_results"], 1908)
        self.assertEqual(summary["legacy_unresolved_results"], 238)
        self.assertEqual(summary["legacy_copy_records"], 521)
        self.assertEqual(summary["incomplete_table10_admissions"], 34)
        self.assertEqual(summary["drift_only_copied_raw_nodes"], 1)
        self.assertFalse(summary["sealed_run_identity_match"])
        self.assertFalse(summary["strict_reuse_allowed"])
        self.assertEqual(summary["terminal_state"], "BLOCKED")
        self.assertEqual(len(defects), 37)


if __name__ == "__main__":
    unittest.main()
