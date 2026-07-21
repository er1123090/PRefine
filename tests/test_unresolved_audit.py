from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/provenance/audit_unresolved.py"
SPEC = importlib.util.spec_from_file_location("audit_unresolved", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class UnresolvedAuditTests(unittest.TestCase):
    def test_current_sealed_admission_is_exhaustively_classified(self) -> None:
        records, summary = MODULE.build_audit(ROOT)
        self.assertEqual(summary["total_paper_results"], 1908)
        self.assertEqual(summary["unresolved_results"], 238)
        self.assertEqual(len(records), 238)
        self.assertEqual(summary["direct_unresolved_results"], 128)
        self.assertEqual(summary["derived_unresolved_results"], 110)
        self.assertEqual(summary["resolvable_from_experiments7_only"], 3)
        self.assertEqual(summary["terminal_state"], "BLOCKED")
        self.assertEqual(summary["protected_source_reads_performed"], 0)
        self.assertFalse(summary["public_dependency_graph_complete"])

    def test_reason_mapping_is_fail_closed(self) -> None:
        self.assertEqual(
            MODULE.evidence_requirement("unknown_reason"),
            "manual_evidence_review_required",
        )


if __name__ == "__main__":
    unittest.main()
