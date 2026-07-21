from __future__ import annotations

import ast
from collections import Counter
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import unittest


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "tests/fixtures/dataset/mix600_legacy_contract.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_legacy_functions(path: Path, names: set[str]) -> dict[str, Any]:
    """Execute only selected legacy function definitions, avoiding API SDK imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    found = {node.name for node in functions}
    if found != names:
        raise AssertionError(f"legacy function mismatch in {path}: {sorted(names - found)}")

    namespace: dict[str, Any] = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "Tuple": Tuple,
        "itertools": itertools,
        "json": json,
        "os": os,
        "re": re,
    }
    module = ast.Module(body=functions, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _lines_sha256(values: list[str]) -> str:
    payload = "\n".join(values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Mix600LegacyContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        cls.legacy_root = (ROOT / cls.contract["legacy_root"]).resolve()

        sources = cls.contract["sources"]
        cls.paths = {
            name: cls.legacy_root / details["path"]
            for name, details in sources.items()
        }
        cls.rows = json.loads(cls.paths["mix600"].read_text(encoding="utf-8"))

        cls.single = _load_legacy_functions(
            cls.paths["builder_singleturn"],
            {"assign_user_utterances", "generate_func_strings"},
        )
        cls.multi = _load_legacy_functions(
            cls.paths["builder_multiturn"],
            {
                "assign_user_utterances",
                "format_multiturn_dialogue",
                "generate_single_api_string",
                "get_multiturn_target_slots",
                "load_multiturn_data",
                "merge_and_generate_api_strings",
                "parse_api_call_to_dict",
                "select_multiturn_template",
            },
        )
        cls.query_data = {
            "singleturn": json.loads(cls.paths["query_singleturn"].read_text(encoding="utf-8")),
            "multiturn": cls.multi["load_multiturn_data"](str(cls.paths["query_multiturn"])),
        }
        cls.builders = {
            "singleturn": cls.single["assign_user_utterances"],
            "multiturn": cls.multi["assign_user_utterances"],
        }
        cls.generated = cls._generate_all_scenarios()

    @classmethod
    def _generate_all_scenarios(cls) -> dict[str, dict[str, Any]]:
        generated: dict[str, dict[str, Any]] = {}
        for turn in ("singleturn", "multiturn"):
            builder = cls.builders[turn]
            query_data = cls.query_data[turn]
            for difficulty in ("easy", "medium", "hard"):
                records: list[dict[str, Any]] = []
                instance_ids: list[str] = []
                source_examples = 0
                for row in cls.rows:
                    pairs = builder(
                        str(cls.paths["pref_list"]),
                        row,
                        query_data,
                        difficulty,
                        str(cls.paths["pref_group"]),
                    )
                    if pairs:
                        source_examples += 1
                    for sub_idx, (query, ground_truth) in enumerate(pairs):
                        instance_ids.append(f"{row['example_id']}_{sub_idx}")
                        records.append(
                            {
                                "example_id": row["example_id"],
                                "query": query,
                                "ground_truth": sorted(ground_truth),
                            }
                        )
                generated[f"{turn}.{difficulty}"] = {
                    "records": records,
                    "instance_ids": instance_ids,
                    "source_examples": source_examples,
                }
        return generated

    def test_frozen_source_hashes_and_sizes(self) -> None:
        for name, expected in self.contract["sources"].items():
            with self.subTest(source=name):
                path = self.paths[name]
                self.assertTrue(path.is_file(), path)
                self.assertEqual(path.stat().st_size, expected["bytes"])
                self.assertEqual(_sha256(path), expected["sha256"])

    def test_input_preference_group_coverage_is_frozen(self) -> None:
        actual = Counter(
            preference.get("value_group")
            for row in self.rows
            for preference in row.get("api_calls_pref", [])
        )
        self.assertEqual(len(self.rows), self.contract["source_example_count"])
        self.assertEqual(dict(sorted(actual.items())), self.contract["input_preference_group_counts"])

        pref_group = json.loads(self.paths["pref_group"].read_text(encoding="utf-8"))
        self.assertEqual(sorted(pref_group), self.contract["supported_medium_hard_groups"])

    def test_all_scenario_counts_ids_and_semantic_multisets(self) -> None:
        total = 0
        for scenario, expected in self.contract["scenarios"].items():
            with self.subTest(scenario=scenario):
                actual = self.generated[scenario]
                records = actual["records"]
                instance_ids = actual["instance_ids"]
                total += len(records)

                self.assertEqual(len(records), expected["count"])
                self.assertEqual(actual["source_examples"], expected["source_examples"])
                self.assertEqual(len(instance_ids), len(set(instance_ids)))
                self.assertEqual(
                    _lines_sha256(sorted(instance_ids)),
                    expected["instance_ids_sha256"],
                )
                self.assertEqual(
                    _lines_sha256(sorted(_canonical_json(record) for record in records)),
                    expected["semantic_multiset_sha256"],
                )

        self.assertEqual(total, self.contract["expected_total_instances"])

    def test_representative_query_and_gt_records_remain_present(self) -> None:
        for scenario, expected in self.contract["representative_records"].items():
            with self.subTest(scenario=scenario):
                canonical_records = {
                    _canonical_json(record) for record in self.generated[scenario]["records"]
                }
                self.assertIn(_canonical_json(expected), canonical_records)


if __name__ == "__main__":
    unittest.main()
