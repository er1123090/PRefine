from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from typing import Callable
import unittest
from unittest import mock

from ours_memory2 import cli
from ours_memory2.contracts import ContextMode, InputContractError, JoinError, ProviderResponse
from tests.support import ScriptedProvider
from ours_memory2.step2_common import load_manifest_v1, make_provider_request
from ours_memory2.step2_single import build_single_cases, infer_single


FIXTURES = Path(__file__).parent / "fixtures" / "step2"
MANIFEST = FIXTURES / "manifest_grouped.json"
TWO_EXAMPLE_MANIFEST = FIXTURES / "manifest_two_examples.json"


class _FixtureCopy:
    def __init__(self) -> None:
        self.temp = tempfile.TemporaryDirectory()

    def __enter__(self) -> Path:
        root = Path(self.temp.name) / "step2"
        shutil.copytree(FIXTURES, root)
        return root

    def __exit__(self, *args: object) -> None:
        self.temp.cleanup()


def _copy_fixture() -> _FixtureCopy:
    return _FixtureCopy()


class SingleDerivationTests(unittest.TestCase):
    def test_manifest_relative_loading_strict_join_and_all_order(self) -> None:
        resources = load_manifest_v1(MANIFEST)
        self.assertEqual([item["example_id"] for item in resources.examples], ["7"])
        cases = build_single_cases(resources, difficulty="all")
        self.assertEqual([case.difficulty.value for case in cases], ["easy", "medium", "hard"])
        self.assertEqual([case.domain for case in cases], ["Restaurant", "Restaurant", "Hotel"])
        self.assertEqual(cases[0].ground_truth, ('Restaurant(seating="quiet")',))
        self.assertEqual(
            cases[1].ground_truth,
            ('Restaurant(seating="cozy")', 'Restaurant(seating="quiet")'),
        )
        self.assertEqual(
            cases[2].ground_truth,
            ('Hotel(room="quiet")', 'Hotel(room="suite")'),
        )
        self.assertEqual(cases[1].set_derived_slots, ("seating",))
        self.assertEqual(
            [case.case_id for case in cases],
            ["7:single:easy:0", "7:single:medium:0", "7:single:hard:0"],
        )
        self.assertEqual(cases, build_single_cases(resources, difficulty="all"))

    def test_two_examples_reset_source_ordinal_and_preserve_all_order(self) -> None:
        resources = load_manifest_v1(TWO_EXAMPLE_MANIFEST)
        cases = build_single_cases(resources, difficulty="all")
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
                "7:single:easy:0",
                "8:single:easy:0",
                "7:single:medium:0",
                "8:single:medium:0",
                "7:single:hard:0",
                "8:single:hard:0",
            ],
        )

    def test_exactly_four_context_prompts_have_only_selected_components(self) -> None:
        resources = load_manifest_v1(MANIFEST)
        case = build_single_cases(resources, difficulty="easy")[0]
        expected = {
            "memory_only": (True, False, False),
            "memory_api": (True, True, False),
            "memory_diag": (True, False, True),
            "api_only": (False, True, False),
        }
        self.assertEqual(set(expected), {mode.value for mode in ContextMode})
        for context, (preference, api, dialogue) in expected.items():
            with self.subTest(context=context):
                prompt = make_provider_request(
                    resources=resources, case=case, context=context, model="model"
                ).prompt
                self.assertEqual(prompt.count("pref-unique"), int(preference))
                self.assertEqual(prompt.count("API-HISTORY-UNIQUE-1"), int(api))
                self.assertEqual(prompt.count("Find a quiet restaurant"), int(dialogue))
                self.assertEqual(prompt.count("SINGLE-RESTAURANT-UTTERANCE"), 1)
                self.assertEqual(prompt.count("SCHEMA-UNIQUE-RESTAURANT"), 1)
        for value in ("with-memory", "without-memory", "api-only", "unknown"):
            with self.subTest(value=value), self.assertRaises(InputContractError) as caught:
                make_provider_request(resources=resources, case=case, context=value, model="model")
            self.assertEqual(caught.exception.path, "context")

    def test_inference_rows_are_correlated_stable_and_redacted(self) -> None:
        provider = ScriptedProvider(
            lambda request: ProviderResponse(
                text=json.dumps({"action": "ok"}),
                request_id="request-1",
                usage={
                    "prompt_tokens": 2,
                    "access_token": "access-must-hide",
                    "monkey": "usage-monkey-kept",
                },
                raw_response={
                    "authorization": "auth-must-hide",
                    "nested": {
                        "refreshToken": "refresh-must-hide",
                        "client_secret": "client-must-hide",
                        "password": "password-must-hide",
                        "credential_id": "credential-must-hide",
                        "apikey": "apikey-must-hide",
                        "aws_access_key_id": "aws-must-hide",
                        "monkey": "raw-monkey-kept",
                    },
                },
            )
        )
        run = infer_single(
            str(MANIFEST), difficulty="all", context="memory_api", model="model", provider=provider
        )
        self.assertEqual(len(provider.requests), 3)
        self.assertEqual(len(run.results), 3)
        self.assertEqual(len(run.diagnostics), 3)
        self.assertEqual([row["case_id"] for row in run.results], [row["case_id"] for row in run.diagnostics])
        self.assertEqual([row["mode"] for row in run.results], ["single"] * 3)
        self.assertEqual(
            [row["memory_id"] for row in run.results],
            [row["memory_id"] for row in run.diagnostics],
        )
        diagnostic_text = json.dumps(run.diagnostics)
        self.assertNotIn("must-hide", diagnostic_text)
        self.assertIn("[REDACTED]", diagnostic_text)
        self.assertIn("usage-monkey-kept", diagnostic_text)
        self.assertIn("raw-monkey-kept", diagnostic_text)
        self.assertIn('"prompt_tokens": 2', diagnostic_text)
        for request, diagnostic in zip(provider.requests, run.diagnostics):
            provenance = diagnostic["provider"]["request"]
            self.assertEqual(provenance["model"], request.model)
            self.assertEqual(provenance["purpose"], "infer")
            self.assertEqual(
                [message["role"] for message in provenance["messages"]],
                ["user"],
            )
            self.assertEqual(provenance["prompt"], request.prompt)
            self.assertEqual(
                provenance["temperature"], {"present": False, "value": None}
            )
            self.assertFalse(provenance["json_intent"])

    def test_request_provenance_redacts_nested_compound_keys_but_not_monkey(self) -> None:
        with _copy_fixture() as root:
            memory_path = root / "resources" / "memories.jsonl"
            memory = json.loads(memory_path.read_text(encoding="utf-8"))
            memory["final_implicit_preference"] = {
                "nested": {
                    "aws_access_key_id": "REQUEST-AWS-SECRET",
                    "monkey": "REQUEST-MONKEY-KEPT",
                }
            }
            memory_path.write_text(json.dumps(memory) + "\n", encoding="utf-8")
            provider = ScriptedProvider(
                [ProviderResponse(text='{"action":"ok"}')]
            )
            run = infer_single(
                str(root / "manifest_grouped.json"),
                difficulty="easy",
                context="memory_only",
                model="model",
                provider=provider,
            )

        rendered = json.dumps(run.diagnostics, ensure_ascii=False)
        self.assertNotIn("REQUEST-AWS-SECRET", rendered)
        self.assertIn("REQUEST-MONKEY-KEPT", rendered)
        self.assertIn("[REDACTED]", rendered)

    def test_hard_rules_skip_null_values_and_preserve_nonnull_order(self) -> None:
        with _copy_fixture() as root:
            group_path = root / "resources" / "preference_groups.json"
            data = json.loads(group_path.read_text(encoding="utf-8"))
            rules = [
                rule
                for rule in data["comfort"]["rules"]
                if rule["domain"] != "Hotel"
            ]
            rules.extend(
                [
                    {"domain": "Hotel", "slot": "room", "value": None},
                    {"domain": "Hotel", "slot": "room", "value": "suite"},
                    {"domain": "Hotel", "slot": "room", "value": None},
                    {"domain": "Hotel", "slot": "room", "value": "quiet"},
                ]
            )
            data["comfort"]["rules"] = rules
            group_path.write_text(json.dumps(data), encoding="utf-8")
            resources = load_manifest_v1(root / "manifest_grouped.json")
            cases = build_single_cases(resources, difficulty="hard")
            self.assertEqual(
                cases[0].ground_truth,
                ('Hotel(room="suite")', 'Hotel(room="quiet")'),
            )

            for rule in data["comfort"]["rules"]:
                if rule["domain"] == "Hotel":
                    rule["value"] = None
            group_path.write_text(json.dumps(data), encoding="utf-8")
            provider = ScriptedProvider([])
            with self.assertRaises(InputContractError) as caught:
                infer_single(
                    str(root / "manifest_grouped.json"),
                    difficulty="hard",
                    context="memory_only",
                    model="m",
                    provider=provider,
                )
            self.assertEqual(caught.exception.path, "cases")
            self.assertEqual(provider.requests, [])


class ManifestFailureTests(unittest.TestCase):
    def test_malformed_later_raw_api_call_has_exact_indexed_path_pre_provider(self) -> None:
        malformed = (
            ("Hotel", "must be a typed API call"),
            ("Hotel(room=unquoted)", "contains malformed arguments"),
            (None, "must be a string"),
        )
        for command in ("single", "multi"):
            for bad_call, message in malformed:
                with self.subTest(command=command, bad_call=bad_call), _copy_fixture() as root:
                    manifest_path = root / "manifest_two_examples.json"
                    examples_path = root / "resources" / "examples_two.json"
                    examples = json.loads(examples_path.read_text(encoding="utf-8"))
                    examples[1]["api_calls"].append(bad_call)
                    examples_path.write_text(json.dumps(examples), encoding="utf-8")
                    output = root / f"output-{command}"
                    output.mkdir()
                    sentinel = output / "sentinel.txt"
                    sentinel.write_bytes(b"unchanged")
                    before = {
                        path.name: (path.stat(), path.read_bytes())
                        for path in output.iterdir()
                    }
                    args = [
                        "step2",
                        command,
                        "--manifest",
                        str(manifest_path),
                        "--difficulty",
                        "all",
                        "--context",
                        "memory_only",
                        "--output-root",
                        str(output),
                        "--endpoint",
                        "http://127.0.0.1:1/x",
                        "--model",
                        "m",
                    ]
                    if command == "multi":
                        args.extend(("--input-shape", "auto"))
                    stderr = io.StringIO()
                    with mock.patch(
                        "ours_memory2.cli.OpenAICompatibleProvider",
                        side_effect=AssertionError(
                            "provider constructed before raw API validation"
                        ),
                    ) as provider_type, contextlib.redirect_stderr(stderr):
                        code = cli.main(tuple(args))
                    self.assertEqual(code, 2)
                    self.assertEqual(provider_type.call_count, 0)
                    self.assertIn(
                        "resources.examples[1].api_calls[1]", stderr.getvalue()
                    )
                    self.assertIn(message, stderr.getvalue())
                    self.assertEqual(
                        {
                            path.name: (path.stat(), path.read_bytes())
                            for path in output.iterdir()
                        },
                        before,
                    )
    def test_normalized_mapping_collisions_are_exact_and_pre_side_effect(self) -> None:
        originals = {
            "resources/single_queries.json": (
                "resources.single_query_map",
                "Restaurant",
                "SINGLE",
            ),
            "resources/preference_slots.json": (
                "resources.preference_slots",
                "Restaurant",
                ["seating"],
            ),
            "resources/preference_groups.json": (
                "resources.preference_groups",
                "comfort",
                {
                    "rules": [
                        {
                            "domain": "Restaurant",
                            "slot": "seating",
                            "value": "quiet",
                        }
                    ]
                },
            ),
            "resources/multiturn_grouped.json": (
                "resources.multiturn_templates",
                "Restaurant",
                json.loads(
                    (FIXTURES / "resources" / "multiturn_grouped.json").read_text(
                        encoding="utf-8"
                    )
                )["Restaurant"],
            ),
        }
        for relative, (logical_base, normalized_key, value) in originals.items():
            for raw_keys in (
                (normalized_key, f" {normalized_key} "),
                (f" {normalized_key} ", normalized_key),
            ):
                with self.subTest(relative=relative, raw_keys=raw_keys), _copy_fixture() as root:
                    path = root / relative
                    path.write_text(
                        json.dumps(
                            {raw_keys[0]: value, raw_keys[1]: value},
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                    output = root / "must-not-exist"
                    stderr = io.StringIO()
                    with mock.patch(
                        "ours_memory2.cli.OpenAICompatibleProvider",
                        side_effect=AssertionError(
                            "provider constructed before collision rejection"
                        ),
                    ), contextlib.redirect_stderr(stderr):
                        code = cli.main(
                            (
                                "step2",
                                "single",
                                "--manifest",
                                str(root / "manifest_grouped.json"),
                                "--difficulty",
                                "all",
                                "--context",
                                "memory_only",
                                "--output-root",
                                str(output),
                                "--endpoint",
                                "http://127.0.0.1:1/x",
                                "--model",
                                "m",
                            )
                        )
                    expected_path = (
                        f"{logical_base}[{json.dumps(raw_keys[1], ensure_ascii=False)}]"
                    )
                    self.assertEqual(code, 2)
                    self.assertIn(expected_path, stderr.getvalue())
                    self.assertFalse(output.exists())

    def test_noncolliding_mapping_insertion_order_is_preserved(self) -> None:
        resources = load_manifest_v1(MANIFEST)
        self.assertEqual(
            list(resources.single_query_map), ["Restaurant", "Hotel"]
        )
        self.assertEqual(
            list(resources.preference_slots), ["Restaurant", "Hotel"]
        )
        self.assertEqual(list(resources.preference_groups), ["comfort"])
        self.assertEqual(
            list(resources.multiturn_templates), ["Restaurant", "Hotel"]
        )

    def test_template_targets_are_strict_and_city_api_argument_is_known(self) -> None:
        resources = load_manifest_v1(MANIFEST)
        self.assertEqual(resources.multiturn_templates["Restaurant"][0].target_slots, ("city",))
        mutations = (
            (lambda target: target.__setitem__(0, "bad"), "resources.multiturn_templates.Restaurant[0].target[0]"),
            (lambda target: target[0].pop("domain"), "resources.multiturn_templates.Restaurant[0].target[0].domain"),
            (lambda target: target[0].__setitem__("domain", "Hotel"), "resources.multiturn_templates.Restaurant[0].target[0].domain"),
            (lambda target: target[0].__setitem__("domain", "Unknown"), "resources.multiturn_templates.Restaurant[0].target[0].domain"),
            (lambda target: target[0].pop("slot"), "resources.multiturn_templates.Restaurant[0].target[0].slot"),
            (lambda target: target[0].__setitem__("slot", "fabricated"), "resources.multiturn_templates.Restaurant[0].target[0].slot"),
        )
        for mutate, expected_path in mutations:
            with self.subTest(expected_path=expected_path), _copy_fixture() as root:
                path = root / "resources" / "multiturn_grouped.json"
                data = json.loads(path.read_text())
                mutate(data["Restaurant"][0]["target"])
                path.write_text(json.dumps(data), encoding="utf-8")
                with self.assertRaises((InputContractError, JoinError)) as caught:
                    load_manifest_v1(root / "manifest_grouped.json")
                self.assertEqual(caught.exception.path, expected_path)

        with _copy_fixture() as root:
            path = root / "resources" / "multiturn_list.json"
            data = json.loads(path.read_text())
            data[0]["target"][0].pop("slot")
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(InputContractError) as caught:
                load_manifest_v1(root / "manifest_list.json", multi_shape="list")
            self.assertEqual(
                caught.exception.path,
                "resources.multiturn_templates[0].target[0].slot",
            )

    def test_missing_duplicate_orphan_and_missing_join_fail_before_provider(self) -> None:
        mutations = {
            "missing_resource": ("manifest_grouped.json", lambda data: data.__setitem__("tool_schema", "resources/missing.json"), InputContractError, "resources.tool_schema"),
            "collision": ("resources/examples.json", lambda data: data.append(dict(data[0], example_id="7")), JoinError, "resources.examples[1].example_id"),
            "missing_memory": ("resources/memories.jsonl", lambda data: data[0].__setitem__("example_id", "8"), JoinError, "joins.examples[7].memory"),
            "duplicate_memory": ("resources/memories.jsonl", lambda data: data.append(dict(data[0], example_id=7)), JoinError, "resources.memories.line[2].example_id"),
            "orphan": ("resources/memories.jsonl", lambda data: data.append(dict(data[0], example_id="8")), JoinError, "joins.memories[8].example"),
            "group": ("resources/examples.json", lambda data: data[0]["api_calls_pref"][0].__setitem__("value_group", "missing"), JoinError, "joins.preference_groups.missing"),
            "slot": ("resources/examples.json", lambda data: data[0]["api_calls_pref"][0]["evidence"][0].__setitem__("slot", "missing"), JoinError, "joins.preference_slots.Restaurant.missing"),
            "template": ("resources/multiturn_grouped.json", lambda data: data.pop("Hotel"), JoinError, "joins.domains.Hotel"),
            "shape": ("resources/examples.json", lambda data: data[0].__setitem__("sessions", [1]), InputContractError, "resources.examples[0].sessions[0]"),
        }
        for name, (relative, mutate, error_type, expected_path) in mutations.items():
            with self.subTest(name=name), _copy_fixture() as root:
                path = root / relative
                if path.suffix == ".jsonl":
                    data = [json.loads(line) for line in path.read_text().splitlines() if line]
                    _apply_mutation(mutate, data)
                    path.write_text("".join(json.dumps(row) + "\n" for row in data), encoding="utf-8")
                else:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    _apply_mutation(mutate, data)
                    path.write_text(json.dumps(data), encoding="utf-8")
                provider = ScriptedProvider([])
                with self.assertRaises(error_type) as caught:
                    infer_single(str(root / "manifest_grouped.json"), difficulty="all", context="memory_only", model="m", provider=provider)
                self.assertEqual(caught.exception.path, expected_path)
                self.assertEqual(provider.requests, [])

    def test_zero_derived_cases_is_typed_and_pre_provider(self) -> None:
        with _copy_fixture() as root:
            path = root / "resources" / "examples.json"
            data = json.loads(path.read_text())
            data[0]["api_calls"] = []
            data[0]["api_calls_pref"] = []
            path.write_text(json.dumps(data), encoding="utf-8")
            provider = ScriptedProvider([])
            with self.assertRaises(InputContractError) as caught:
                infer_single(str(root / "manifest_grouped.json"), difficulty="all", context="memory_only", model="m", provider=provider)
            self.assertEqual(caught.exception.path, "cases")
            self.assertEqual(provider.requests, [])

def _apply_mutation(mutation: Callable[[object], object], value: object) -> None:
    mutation(value)


if __name__ == "__main__":
    unittest.main()
