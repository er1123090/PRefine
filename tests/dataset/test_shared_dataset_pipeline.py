from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exp7.datasets.build_instances import (
    build_all_scenarios,
    canonical_json,
    semantic_multiset_sha256,
)
from exp7.datasets.validation import build_validation_report


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _inputs():
    dataset_config = _load(ROOT / "configs/datasets/mix600-v1.json")
    query_config = _load(ROOT / dataset_config["queries_config"])
    preference_config = _load(ROOT / dataset_config["preferences_config"])
    schema_config = _load(ROOT / dataset_config["schemas_config"])

    def configured(config, name):
        return _load(ROOT / config["inputs"][name]["path"])

    rows = _load((ROOT / dataset_config["source"]["default_path"]).resolve())
    query_singleturn = configured(query_config, "singleturn")
    query_multiturn = configured(query_config, "multiturn")
    pref_list = configured(preference_config, "pref_list")
    pref_groups = configured(preference_config, "pref_group")
    scenarios = build_all_scenarios(
        rows,
        dataset_id=dataset_config["dataset_id"],
        turns=dataset_config["turns"],
        difficulties=dataset_config["difficulties"],
        query_singleturn=query_singleturn,
        query_multiturn_raw=query_multiturn,
        pref_list=pref_list,
        pref_groups=pref_groups,
    )
    schemas = {
        "singleturn": configured(schema_config, "singleturn"),
        "multiturn": configured(schema_config, "multiturn"),
    }
    return dataset_config, rows, query_multiturn, pref_groups, schemas, scenarios


def determinism_signature() -> str:
    _, _, _, _, _, scenarios = _inputs()
    values = []
    for scenario, records in scenarios.items():
        values.append(
            canonical_json(
                {
                    "ids": [record["instance_id"] for record in records],
                    "legacy_ids": [record["legacy_example_id_sub"] for record in records],
                    "scenario": scenario,
                    "semantic": semantic_multiset_sha256(records),
                }
            )
        )
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


class SharedDatasetPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        (
            cls.config,
            cls.rows,
            cls.query_multiturn,
            cls.pref_groups,
            cls.schemas,
            cls.scenarios,
        ) = _inputs()

    def test_counts_and_semantic_multisets_match_g001(self) -> None:
        expected = self.config["expected"]["scenarios"]
        self.assertEqual(set(self.scenarios), set(expected))
        for scenario, records in self.scenarios.items():
            with self.subTest(scenario=scenario):
                wanted = expected[scenario]
                self.assertEqual(len(records), wanted["count"])
                self.assertEqual(
                    len({record["source_example_id"] for record in records}),
                    wanted["source_examples"],
                )
                self.assertEqual(
                    semantic_multiset_sha256(records),
                    wanted["semantic_multiset_sha256"],
                )

    def test_canonical_ids_are_stable_unique_and_condition_scoped(self) -> None:
        all_ids = []
        for scenario, records in self.scenarios.items():
            for record in records:
                self.assertIn(f":{scenario.replace('.', ':')}:", record["instance_id"])
                self.assertRegex(record["legacy_example_id_sub"], r"^.+_\d+$")
                self.assertEqual(len(record["semantic_sha256"]), 64)
                all_ids.append(record["instance_id"])
        self.assertEqual(len(all_ids), 2638)
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_validation_report_preserves_known_warnings(self) -> None:
        report = build_validation_report(
            rows=self.rows,
            scenarios=self.scenarios,
            pref_groups=self.pref_groups,
            query_multiturn=self.query_multiturn,
            schemas=self.schemas,
            expected=self.config["expected"],
        )
        self.assertEqual(report["contract_parity"]["status"], "ok")
        self.assertEqual(report["schema_warnings"]["total"], 463)
        self.assertEqual(
            report["multiturn_base_preference_conflicts"]["conflict_count"], 0
        )
        self.assertEqual(
            report["preference_groups"]["ignored_groups"],
            {"eco": 130, "prefers_star": 468},
        )

    def test_active_input_maps_use_only_canonical_small_configs(self) -> None:
        for config_path in (
            ROOT / self.config["queries_config"],
            ROOT / self.config["preferences_config"],
            ROOT / self.config["schemas_config"],
        ):
            config = _load(config_path)
            for spec in config["inputs"].values():
                self.assertTrue(spec["path"].startswith("configs/"), spec["path"])
                path = ROOT / spec["path"]
                self.assertEqual(path.stat().st_size, spec["expected_bytes"])
                self.assertEqual(
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                    spec["expected_sha256"],
                )

    def test_output_is_independent_of_python_hash_seed(self) -> None:
        command = [
            sys.executable,
            "-c",
            (
                "from tests.dataset.test_shared_dataset_pipeline "
                "import determinism_signature; print(determinism_signature())"
            ),
        ]
        signatures = []
        for seed in ("1", "2"):
            environment = dict(os.environ)
            environment["PYTHONHASHSEED"] = seed
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            signatures.append(result.stdout.strip())
        self.assertEqual(signatures[0], signatures[1])


if __name__ == "__main__":
    unittest.main()
