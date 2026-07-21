import unittest
from pathlib import Path

import recovery2_critic
from recovery2_common import (
    GOLD_RELATIVE_PATH,
    RECOVERY_OUTPUTS,
    build_source_inventory,
    stage_record,
    validate_stage_record,
)


class Recovery2ProtocolTests(unittest.TestCase):
    def test_stage_receipts_form_a_hash_bound_chain(self) -> None:
        amendment = "a" * 64
        attempt = stage_record(
            stage="attempt",
            sequence=0,
            payload={"gold_content_opens_so_far": 0},
            amendment_sha256=amendment,
            previous=None,
        )
        pre_gold = stage_record(
            stage="pre_gold",
            sequence=1,
            payload={"gold_content_opens_so_far": 0},
            amendment_sha256=amendment,
            previous=attempt,
        )
        validate_stage_record(
            attempt,
            stage="attempt",
            sequence=0,
            amendment_sha256=amendment,
            previous=None,
        )
        validate_stage_record(
            pre_gold,
            stage="pre_gold",
            sequence=1,
            amendment_sha256=amendment,
            previous=attempt,
        )

    def test_target_blind_inventory_never_hashes_gold_bytes(self) -> None:
        inventory = build_source_inventory()
        gold_rows = [row for row in inventory["rows"] if row["path"] == GOLD_RELATIVE_PATH]
        self.assertIs(inventory["gold_content_opened"], False)
        self.assertEqual(len(gold_rows), 1)
        self.assertEqual(gold_rows[0]["kind"], "sealed_target_declared_only_not_opened")
        self.assertNotIn("sha256", gold_rows[0])
        self.assertEqual(len(gold_rows[0]["declared_sha256"]), 64)

    def test_critic_source_cannot_reopen_gold(self) -> None:
        source = Path(recovery2_critic.__file__).read_text(encoding="utf-8")
        self.assertNotIn("open_regular_bytes_once", source)
        self.assertNotIn("gold.jsonl", source)

    def test_recovery_output_paths_are_unique_and_outside_source_run(self) -> None:
        resolved = [path.resolve() for path in RECOVERY_OUTPUTS.values()]
        self.assertEqual(len(resolved), len(set(resolved)))
        self.assertTrue(all(path.is_relative_to(RECOVERY_OUTPUTS["attempt"].parents[1]) for path in resolved))


if __name__ == "__main__":
    unittest.main()
