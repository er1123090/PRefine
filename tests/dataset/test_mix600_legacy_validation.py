from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
import unittest

from tests.dataset import test_mix600_legacy_contract as contract_module


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests/fixtures/dataset/mix600_legacy_validation.json"


def _schema_slots(path: Path) -> dict[str, set[str]]:
    schema = json.loads(path.read_text(encoding="utf-8"))
    return {
        tool["function"]["name"]: set(
            tool["function"]["parameters"].get("properties", {})
        )
        for tool in schema
    }


def _call_domain_and_slots(api_call: str) -> tuple[str, set[str]]:
    domain = api_call.split("(", 1)[0].strip()
    return domain, set(re.findall(r"(\w+)=", api_call))


class Mix600LegacyValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        contract_module.Mix600LegacyContractTests.setUpClass()
        cls.baseline = contract_module.Mix600LegacyContractTests

    def test_medium_hard_unsupported_only_source_count(self) -> None:
        supported_groups = set(
            self.baseline.contract["supported_medium_hard_groups"]
        )
        actual = Counter()
        for row in self.baseline.rows:
            groups = {
                preference.get("value_group")
                for preference in row.get("api_calls_pref", [])
            }
            key = "supported" if groups & supported_groups else "unsupported_only"
            actual[key] += 1

        self.assertEqual(
            dict(actual),
            self.expected["medium_hard_source_examples"],
        )

    def test_multiturn_template_base_slots_never_overlap_preference_slots(self) -> None:
        templates = json.loads(
            self.baseline.paths["query_multiturn"].read_text(encoding="utf-8")
        )
        overlap_count = 0
        for template in templates:
            preference_slots = {
                target["slot"]
                for target in template.get("target", [])
                if target.get("slot")
            }
            base_slots: set[str] = set()
            for api_call in template.get("api_call", []):
                _, slots = _call_domain_and_slots(api_call)
                base_slots.update(slots)
            if base_slots & preference_slots:
                overlap_count += 1

        expected = self.expected["multiturn_templates"]
        self.assertEqual(len(templates), expected["count"])
        self.assertEqual(
            overlap_count,
            expected["base_preference_slot_overlap_count"],
        )

    def test_schema_missing_slot_instance_counts(self) -> None:
        expected = self.expected["schema_missing_slot_instances"]
        actual_counts: dict[str, int] = {}
        missing_by_turn = {"singleturn": set(), "multiturn": set()}

        for scenario, data in self.baseline.generated.items():
            turn = scenario.split(".", 1)[0]
            schema_name = (
                "schema_singleturn" if turn == "singleturn" else "schema_multiturn"
            )
            allowed = _schema_slots(self.baseline.paths[schema_name])
            missing_instances = 0
            for record in data["records"]:
                record_missing: set[str] = set()
                for api_call in record["ground_truth"]:
                    domain, slots = _call_domain_and_slots(api_call)
                    record_missing.update(slots - allowed.get(domain, set()))
                if record_missing:
                    missing_instances += 1
                    missing_by_turn[turn].update(record_missing)
            actual_counts[scenario] = missing_instances

        self.assertEqual(actual_counts, expected["scenarios"])
        self.assertEqual(sum(actual_counts.values()), expected["total"])
        self.assertEqual(
            {turn: sorted(slots) for turn, slots in missing_by_turn.items()},
            expected["missing_slots"],
        )


if __name__ == "__main__":
    unittest.main()
