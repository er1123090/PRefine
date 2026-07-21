from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

from ours_memory2.contracts import InputContractError, ProviderResponse
from tests.support import ScriptedProvider
from ours_memory2.step2_common import load_manifest_v1
from ours_memory2.step2_multi import build_multi_cases, infer_multi


FIXTURES = Path(__file__).parent / "fixtures" / "step2"
GROUPED = FIXTURES / "manifest_grouped.json"
LIST = FIXTURES / "manifest_list.json"
TWO_EXAMPLE_MANIFEST = FIXTURES / "manifest_two_examples.json"


class MultiNormalizationTests(unittest.TestCase):
    def test_grouped_and_list_raw_resources_normalize_equivalently(self) -> None:
        grouped = load_manifest_v1(GROUPED, multi_shape="grouped")
        listed = load_manifest_v1(LIST, multi_shape="list")
        self.assertEqual(grouped.multiturn_templates, listed.multiturn_templates)
        self.assertEqual(
            build_multi_cases(grouped, difficulty="all"),
            build_multi_cases(listed, difficulty="all"),
        )
        with self.assertRaises(InputContractError) as caught:
            load_manifest_v1(GROUPED, multi_shape="list")
        self.assertEqual(caught.exception.path, "resources.multiturn_templates")

    def test_source_faithful_selection_merge_and_all_order(self) -> None:
        resources = load_manifest_v1(GROUPED)
        cases = build_multi_cases(resources, difficulty="all")
        self.assertEqual([case.difficulty.value for case in cases], ["easy", "medium", "hard"])
        self.assertEqual([case.template_id for case in cases], ["r-pref", "r-pref", "h-pref"])
        self.assertEqual(
            cases[1].ground_truth,
            (
                'Restaurant(city="Seoul", seating="cozy")',
                'Restaurant(city="Seoul", seating="quiet")',
            ),
        )
        self.assertEqual(
            cases[2].ground_truth,
            ('Hotel(city="Seoul", room="quiet")', 'Hotel(city="Seoul", room="suite")'),
        )
        self.assertEqual(cases[0].utterance, "User: RESTAURANT-PREF-FIRST\nAssistant: RESTAURANT-PREF-SECOND")

    def test_two_examples_reset_source_ordinal_and_preserve_all_order(self) -> None:
        resources = load_manifest_v1(TWO_EXAMPLE_MANIFEST)
        cases = build_multi_cases(resources, difficulty="all")
        self.assertEqual(
            [case.difficulty.value for case in cases],
            ["easy", "easy", "medium", "medium", "hard", "hard"],
        )
        self.assertEqual(
            [case.example_id for case in cases], ["7", "8", "7", "8", "7", "8"]
        )
        self.assertEqual(
            [case.case_id for case in cases],
            [
                "7:multi:easy:0",
                "8:multi:easy:0",
                "7:multi:medium:0",
                "8:multi:medium:0",
                "7:multi:hard:0",
                "8:multi:hard:0",
            ],
        )

    def test_all_four_contexts_and_grouped_list_requests_match(self) -> None:
        for context in ("memory_only", "memory_api", "memory_diag", "api_only"):
            with self.subTest(context=context):
                providers = [ScriptedProvider(lambda _: ProviderResponse(text='{"action":"ok"}')) for _ in range(2)]
                grouped = infer_multi(str(GROUPED), input_shape="auto", difficulty="all", context=context, model="m", provider=providers[0])
                listed = infer_multi(str(LIST), input_shape="auto", difficulty="all", context=context, model="m", provider=providers[1])
                self.assertEqual(grouped, listed)
                self.assertEqual(providers[0].requests, providers[1].requests)
                self.assertEqual([row["context"] for row in grouped.results], [context] * 3)
                self.assertEqual([row["mode"] for row in grouped.results], ["multi"] * 3)
        provider = ScriptedProvider([])
        with self.assertRaises(InputContractError) as caught:
            infer_multi(str(GROUPED), input_shape="auto", difficulty="all", context="api-only", model="m", provider=provider)
        self.assertEqual(caught.exception.path, "context")
        self.assertEqual(provider.requests, [])

    def test_result_and_diagnostic_rows_are_deterministic(self) -> None:
        factory = lambda: ScriptedProvider(
            lambda request: ProviderResponse(text=json.dumps({"prompt": request.prompt[-16:]}), raw_response={"secret": "hidden"})
        )
        first_provider, second_provider = factory(), factory()
        first = infer_multi(str(GROUPED), input_shape="grouped", difficulty="all", context="memory_diag", model="m", provider=first_provider)
        second = infer_multi(str(GROUPED), input_shape="grouped", difficulty="all", context="memory_diag", model="m", provider=second_provider)
        self.assertEqual(first, second)
        self.assertEqual(len(first.results), 3)
        self.assertEqual([row["case_id"] for row in first.results], [row["case_id"] for row in first.diagnostics])
        self.assertNotIn("hidden", json.dumps(first.diagnostics))
        for request, diagnostic in zip(first_provider.requests, first.diagnostics):
            provenance = diagnostic["provider"]["request"]
            self.assertEqual(provenance["model"], request.model)
            self.assertEqual(provenance["purpose"], "infer")
            self.assertEqual(
                [message["role"] for message in provenance["messages"]],
                ["user"],
            )
            self.assertEqual(
                provenance["temperature"], {"present": False, "value": None}
            )
            self.assertFalse(provenance["json_intent"])

    def test_hard_rules_skip_null_values_and_all_null_is_pre_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "step2"
            shutil.copytree(FIXTURES, root)
            group_path = root / "resources" / "preference_groups.json"
            data = json.loads(group_path.read_text(encoding="utf-8"))
            data["comfort"]["rules"] = [
                rule
                for rule in data["comfort"]["rules"]
                if rule["domain"] != "Hotel"
            ] + [
                {"domain": "Hotel", "slot": "room", "value": None},
                {"domain": "Hotel", "slot": "room", "value": "suite"},
                {"domain": "Hotel", "slot": "room", "value": None},
                {"domain": "Hotel", "slot": "room", "value": "quiet"},
            ]
            group_path.write_text(json.dumps(data), encoding="utf-8")
            resources = load_manifest_v1(root / "manifest_grouped.json")
            cases = build_multi_cases(resources, difficulty="hard")
            self.assertEqual(
                cases[0].ground_truth,
                (
                    'Hotel(city="Seoul", room="suite")',
                    'Hotel(city="Seoul", room="quiet")',
                ),
            )

            for rule in data["comfort"]["rules"]:
                if rule["domain"] == "Hotel":
                    rule["value"] = None
            group_path.write_text(json.dumps(data), encoding="utf-8")
            provider = ScriptedProvider([])
            with self.assertRaises(InputContractError) as caught:
                infer_multi(
                    str(root / "manifest_grouped.json"),
                    input_shape="grouped",
                    difficulty="hard",
                    context="memory_only",
                    model="m",
                    provider=provider,
                )
            self.assertEqual(caught.exception.path, "cases")
            self.assertEqual(provider.requests, [])


if __name__ == "__main__":
    unittest.main()
