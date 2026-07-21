from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
PROBES = Path(__file__).parent / "probes"
REFERENCE_FILES = (
    "Preference_Memory_step1_LATENTPREF.py",
    "prompt_update3.py",
    "prompt_inference.py",
    "Preference_Memory_step2_ACTION_singleturn_api.py",
    "Preference_Memory_step2_ACTION_multiturn_api.py",
    "Preference_Memory_step2_ACTION_singleturn_VLLM.py",
    "Preference_Memory_step2_ACTION_multiturn_VLLM.py",
)
APPROVED_DEPARTURES = {
    "provider_errors_are_explicit": "ProviderError",
    "source_context_alias": "rejected",
    "strict_line_local_utf8_jsonl": "one-object-per-nonblank-line",
    "strict_verifier_shape": "rejected",
    "memory_diag_dialogue_injection": "approved-target-normalization",
    "target_step2_additions": [
        "record_type",
        "case_id",
        "memory_id",
        "request_provenance",
        "response_diagnostics",
    ],
}
TIMEOUT_SECONDS = 8


def differential_request(family: str) -> dict:
    sessions = [
        {
            "dialogue": [{"role": "user", "message": "DIALOGUE-UNIQUE first"}],
            "api_call": ["Restaurant(city='Seoul', seating='quiet')"],
        },
        {
            "dialogue": [{"role": "assistant", "message": "DIALOGUE-UNIQUE second"}],
            "api_call": ["Hotel(city='Seoul', room='quiet')"],
        },
    ]
    memory = {
        "example_id": "7",
        "final_implicit_preference": {"constraint": "PREFERENCE-UNIQUE"},
        "final_accumulated_api_calls": ["API-HISTORY-UNIQUE first", "API-HISTORY-UNIQUE second"],
        "final_accumulated_dialogue": "DIALOGUE-UNIQUE first\nDIALOGUE-UNIQUE second",
        "total_sessions_processed": 2,
        "preference_evolution_history": [{"session_index": 1}, {"session_index": 2}],
    }
    example = {
        "example_id": "7",
        "sessions": sessions,
        "api_calls": ["Restaurant(city='Seoul', seating='quiet')"],
        "api_calls_pref": [
            {
                "value_group": "comfort",
                "evidence": [{"domain": "Restaurant", "slot": "seating"}],
            }
        ],
    }
    second_example = {
        "example_id": "8",
        "sessions": [
            {
                "dialogue": [
                    {"role": "user", "message": "DIALOGUE-SECOND hotel"}
                ],
                "api_call": ["Hotel(city='Busan', room='suite')"],
            }
        ],
        "api_calls": ["Hotel(city='Busan', room='suite')"],
        "api_calls_pref": [
            {
                "value_group": "comfort",
                "evidence": [{"domain": "Hotel", "slot": "room"}],
            }
        ],
    }
    second_memory = {
        "example_id": "8",
        "final_implicit_preference": {"constraint": "PREFERENCE-SECOND"},
        "final_accumulated_api_calls": ["API-HISTORY-SECOND"],
        "final_accumulated_dialogue": "DIALOGUE-SECOND hotel",
        "total_sessions_processed": 1,
        "preference_evolution_history": [{"session_index": 1}],
    }
    request = {
        "family": family,
        "parse_inputs": [
            '{"implicit_pref":{"constraint":"one"}}',
            '```json\n{"implicit_pref":{"constraint":"two",},}\n```',
            "not-json",
        ],
        "sessions": sessions,
        "two_session_scripts": [
            {"draft": {"constraint": "session-one"}, "valid": True, "feedback": "ok-one"},
            {"draft": {"constraint": "session-two"}, "valid": True, "feedback": "ok-two"},
        ],
        "ten_slot_scripts": [
            {"draft": {"constraint": f"candidate-{index}"}, "valid": False, "feedback": f"feedback-{index}"}
            for index in range(1, 11)
        ],
        "step1_retry_cases": {
            "malformed_first_generation": [
                "not-json",
                {"constraint": "candidate-after-malformed"},
                {"valid": True, "feedback": "supported"},
            ],
            "false_with_empty_feedback": [
                {"constraint": "candidate-empty-feedback-1"},
                {"valid": False, "feedback": ""},
                {"constraint": "candidate-empty-feedback-2"},
                {"valid": True, "feedback": "supported"},
            ],
        },
        "example": example,
        "examples": [example, second_example],
        "memory": memory,
        "memories": [memory, second_memory],
        "single_query_map": {
            "Restaurant": "CURRENT-UNIQUE restaurant",
            "Hotel": "CURRENT-UNIQUE hotel",
        },
        "multiturn_data": {
            "Restaurant": [
                {"query_id": "r-base", "query": [{"role": "User", "message": "CURRENT-UNIQUE base"}], "api_call": ["Restaurant(city='Seoul')"], "target": [{"domain": "Restaurant", "slot": "city"}]},
                {"query_id": "r-pref", "query": [{"role": "User", "message": "CURRENT-UNIQUE restaurant"}], "api_call": ["Restaurant(city='Seoul')"], "target": [{"domain": "Restaurant", "slot": "seating"}]},
            ],
            "Hotel": [
                {"query_id": "h-pref", "query": [{"role": "User", "message": "CURRENT-UNIQUE hotel"}], "api_call": ["Hotel(city='Seoul')"], "target": [{"domain": "Hotel", "slot": "room"}]}
            ],
        },
        "preference_slots": {"Restaurant": ["seating"], "Hotel": ["room"]},
        "preference_groups": {
            "comfort": {"rules": [
                {"domain": "Restaurant", "slot": "seating", "value": "quiet"},
                {"domain": "Restaurant", "slot": "seating", "value": "cozy"},
                {"domain": "Hotel", "slot": "room", "value": "quiet"},
                {"domain": "Hotel", "slot": "room", "value": "suite"},
            ]}
        },
        "hard_null_rules": {
            "mixed": [
                {"domain": "Restaurant", "slot": "seating", "value": "quiet"},
                {"domain": "Hotel", "slot": "room", "value": None},
                {"domain": "Hotel", "slot": "room", "value": "suite"},
                {"domain": "Hotel", "slot": "room", "value": None},
                {"domain": "Hotel", "slot": "room", "value": "quiet"},
            ],
            "all_null": [
                {"domain": "Restaurant", "slot": "seating", "value": "quiet"},
                {"domain": "Hotel", "slot": "room", "value": None},
                {"domain": "Hotel", "slot": "room", "value": None},
            ],
        },
        "tool_schema": [{"name": "Restaurant", "description": "SCHEMA-UNIQUE"}],
    }
    return request


class ReferenceDifferentialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.required = os.environ.get("REQUIRE_REFERENCE_DIFFERENTIAL") == "1"
        value = os.environ.get("OURS_MEMORY_REFERENCE_ROOT")
        if not value:
            if cls.required:
                raise AssertionError("OURS_MEMORY_REFERENCE_ROOT is required by the differential gate")
            raise unittest.SkipTest("reference differential unavailable: OURS_MEMORY_REFERENCE_ROOT is not set")
        cls.reference_root = Path(value).expanduser().resolve(strict=False)
        missing = [name for name in REFERENCE_FILES if not (cls.reference_root / name).is_file()]
        if not cls.reference_root.is_dir() or missing:
            message = f"reference differential unavailable: missing {missing or [str(cls.reference_root)]}"
            if cls.required:
                raise AssertionError(message)
            raise unittest.SkipTest(message)

    def test_step1_parsing_carried_state_and_ten_slot_projection_match(self) -> None:
        self._assert_family("step1")

    def test_step1_malformed_and_empty_feedback_retry_roles_match_source(self) -> None:
        result = self._run_pair(differential_request("step1"))
        malformed = result["retry_cases"]["malformed_first_generation"]
        self.assertEqual(
            [item["purpose"] for item in malformed["provider_requests"]],
            ["generate", "generate", "verify"],
        )
        self.assertEqual(malformed["generation_purposes"], ["generate"])
        self.assertEqual(malformed["steps"], [2])
        empty = result["retry_cases"]["false_with_empty_feedback"]
        self.assertEqual(
            [item["purpose"] for item in empty["provider_requests"]],
            ["generate", "verify", "generate", "verify"],
        )
        self.assertEqual(empty["generation_purposes"], ["generate", "generate"])
        self.assertEqual(empty["lineage"], [
            {"parent": None, "feedback": None},
            {"parent": None, "feedback": None},
        ])

    def test_single_raw_derivation_contexts_and_projection_match(self) -> None:
        result = self._run_pair(differential_request("single"))
        self._assert_every_case_has_all_contexts(result)
        self.assertEqual(
            result["hard_null"],
            {
                "all_null": [],
                "mixed": [
                    {
                        "example_id": "7",
                        "domain": "Hotel",
                        "utterance": "CURRENT-UNIQUE hotel",
                        "ground_truth": [
                            'Hotel(room="suite")',
                            'Hotel(room="quiet")',
                        ],
                        "template_id": None,
                    }
                ],
            },
        )

    def test_multi_raw_derivation_contexts_and_projection_match(self) -> None:
        request = differential_request("multi")
        grouped_result = self._run_pair(request)
        self._assert_every_case_has_all_contexts(grouped_result)
        request["multiturn_data"] = [
            item for values in request["multiturn_data"].values() for item in values
        ]
        listed_result = self._run_pair(request)
        self.assertEqual(grouped_result, listed_result, "grouped/list-equivalent source order changed")

    def test_lossless_schema_rejects_prompt_lineage_and_list_corruption(self) -> None:
        step1 = self._run_probe(
            "new_probe.py", differential_request("step1"), reference=False
        )
        single = self._run_probe(
            "new_probe.py", differential_request("single"), reference=False
        )
        mutations = []

        verifier_prompt = copy.deepcopy(step1)
        next(
            item
            for item in verifier_prompt["two_session"]["provider_requests"]
            if item["purpose"] == "verify"
        )["prompt"] += " CORRUPT"
        mutations.append((step1, verifier_prompt, "verifier prompt corruption"))

        verifier_role = copy.deepcopy(step1)
        next(
            item
            for item in verifier_role["two_session"]["provider_requests"]
            if item["purpose"] == "verify"
        )["messages"][0]["role"] = "user"
        mutations.append((step1, verifier_role, "verifier role corruption"))

        lineage = copy.deepcopy(step1)
        lineage["ten_slot"]["streams"]["drafts"][1].pop("refinement_parent")
        mutations.append((step1, lineage, "refinement lineage loss"))

        duplicate = copy.deepcopy(single)
        duplicate["request_matrix"][0]["wire"]["messages"].append(
            copy.deepcopy(duplicate["request_matrix"][0]["wire"]["messages"][0])
        )
        mutations.append((single, duplicate, "prompt duplication"))

        reordered_prompt = copy.deepcopy(single)
        reordered_prompt["request_matrix"].reverse()
        mutations.append((single, reordered_prompt, "prompt reordering"))

        reordered_list = copy.deepcopy(single)
        reordered_list["carried_by_example"][0]["api_history"].reverse()
        mutations.append((single, reordered_list, "list-derived reordering"))

        for expected, corrupted, label in mutations:
            with self.subTest(label=label), self.assertRaises(AssertionError):
                self._assert_lossless_equal(expected, corrupted, label)

    def test_runtime_mutation_of_later_request_role_and_lineage_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            package_root = self._copy_package(Path(temp))
            prompt_path = package_root / "ours_memory2" / "prompts.py"
            self._replace_once(
                prompt_path,
                (
                    '    return (ChatMessage("system", system), ChatMessage("user", user))\n'
                    "\n\n"
                    "def build_verifier_messages"
                ),
                (
                    '    return (ChatMessage("assistant", system), ChatMessage("user", user))\n'
                    "\n\n"
                    "def build_verifier_messages"
                ),
            )
            step1_path = package_root / "ours_memory2" / "step1.py"
            self._replace_once(
                step1_path,
                "            refinement_parent=parent,\n",
                "            refinement_parent=None,\n",
            )
            self._assert_mutated_pair_fails(
                differential_request("step1"), package_root, "request role/lineage"
            )

    def test_runtime_mutation_removing_step2_memory_identity_fails(self) -> None:
        for family in ("single", "multi"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temp:
                package_root = self._copy_package(Path(temp))
                common_path = package_root / "ours_memory2" / "step2_common.py"
                text = common_path.read_text(encoding="utf-8")
                marker = '        "memory_id": case.example_id,\n'
                self.assertEqual(text.count(marker), 2)
                common_path.write_text(text.replace(marker, ""), encoding="utf-8")
                self._assert_mutated_pair_fails(
                    differential_request(family), package_root, "memory identity"
                )

    def test_runtime_mutation_reversing_each_step2_example_traversal_fails(self) -> None:
        for family, module_name in (
            ("single", "step2_single.py"),
            ("multi", "step2_multi.py"),
        ):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temp:
                package_root = self._copy_package(Path(temp))
                module_path = package_root / "ours_memory2" / module_name
                self._replace_once(
                    module_path,
                    "        for example in resources.examples:\n",
                    "        for example in reversed(resources.examples):\n",
                )
                self._assert_mutated_pair_fails(
                    differential_request(family), package_root, "example traversal"
                )

    def test_runtime_mutation_joining_every_case_to_first_memory_fails(self) -> None:
        for family in ("single", "multi"):
            with self.subTest(family=family), tempfile.TemporaryDirectory() as temp:
                package_root = self._copy_package(Path(temp))
                common_path = package_root / "ours_memory2" / "step2_common.py"
                self._replace_once(
                    common_path,
                    "    memory = resources.memories[case.example_id]\n",
                    "    memory = next(iter(resources.memories.values()))\n",
                )
                self._assert_mutated_pair_fails(
                    differential_request(family), package_root, "first-memory join"
                )

    def test_source_prompt_rule_and_target_inference_wire_mutations_fail(self) -> None:
        for family in ("single", "multi"):
            with self.subTest(family=family, mutation="source-prompt"), tempfile.TemporaryDirectory() as temp:
                reference_root = self._copy_reference_files(Path(temp))
                prompt_path = reference_root / "prompt_inference.py"
                text = prompt_path.read_text(encoding="utf-8")
                marker = "Do NOT hallucinate values or infer beyond the schema."
                self.assertGreaterEqual(text.count(marker), 2)
                prompt_path.write_text(
                    text.replace(marker, "IGNORE SOURCE EVIDENCE RULE"),
                    encoding="utf-8",
                )
                reference = self._run_probe(
                    "reference_probe.py",
                    differential_request(family),
                    reference=True,
                    reference_root=reference_root,
                )
                target = self._run_probe(
                    "new_probe.py", differential_request(family), reference=False
                )
                with self.assertRaises(AssertionError):
                    self._assert_step2_pair(reference, target)

            with self.subTest(family=family, mutation="wire"), tempfile.TemporaryDirectory() as temp:
                package_root = self._copy_package(Path(temp))
                common_path = package_root / "ours_memory2" / "step2_common.py"
                self._replace_once(
                    common_path,
                    "        temperature=None,\n        json_intent=False,\n",
                    "        temperature=0.0,\n        json_intent=True,\n",
                )
                self._assert_mutated_pair_fails(
                    differential_request(family), package_root, "inference wire"
                )

    def test_reference_projection_uses_observed_legacy_fields_only(self) -> None:
        reference = self._run_probe(
            "reference_probe.py", differential_request("single"), reference=True
        )
        self.assertNotIn("target_projection", reference)
        self.assertTrue(
            all("target_case_id" not in item for item in reference["derivation"])
        )
        self.assertTrue(
            all("target_request" not in item for item in reference["request_matrix"])
        )
        self.assertEqual(
            set(reference["projection"]["legacy_logs"][0]),
            {
                "example_id",
                "example_id_sub",
                "model_name",
                "context_type",
                "pref_type",
                "injected_utterance",
                "reference_ground_truth",
                "model_input",
                "model_output",
                "reasoning_content",
                "token_counts",
            },
        )
        target = self._run_probe(
            "new_probe.py", differential_request("single"), reference=False
        )

        invented = copy.deepcopy(reference)
        invented["target_projection"] = copy.deepcopy(target["target_projection"])
        with self.assertRaises(AssertionError):
            self._assert_step2_pair(invented, target)

        invented = copy.deepcopy(reference)
        invented["request_matrix"][0]["target_request"] = copy.deepcopy(
            target["request_matrix"][0]["target_request"]
        )
        with self.assertRaises(AssertionError):
            self._assert_step2_pair(invented, target)

        invented = copy.deepcopy(reference)
        invented["projection"]["legacy_logs"][0]["memory_id"] = "fabricated"
        with self.assertRaises(AssertionError):
            self._assert_step2_pair(invented, target)

    def _assert_family(self, family: str) -> None:
        self._run_pair(differential_request(family))

    def _run_pair(self, request: dict) -> dict:
        reference = self._run_probe("reference_probe.py", request, reference=True)
        new = self._run_probe("new_probe.py", request, reference=False)
        if request["family"] == "step1":
            self._assert_lossless_equal(
                reference,
                new,
                (
                    f"differential mismatch for {request['family']}\n"
                    f"reference={json.dumps(reference, ensure_ascii=False, sort_keys=True)}\n"
                    f"new={json.dumps(new, ensure_ascii=False, sort_keys=True)}"
                ),
            )
        else:
            self._assert_step2_pair(reference, new)
        self.assertEqual(reference["departures"], APPROVED_DEPARTURES)
        return new

    def _assert_step2_pair(self, reference: dict, target: dict) -> None:
        self.assertEqual(
            set(reference),
            {
                "schema_version",
                "family",
                "derivation",
                "request_matrix",
                "projection",
                "hard_null",
                "identity",
                "carried_by_example",
                "departures",
            },
        )
        self.assertEqual(set(target), set(reference) | {"target_projection"})
        for key in (
            "schema_version",
            "family",
            "identity",
            "carried_by_example",
            "hard_null",
            "projection",
            "departures",
        ):
            self.assertEqual(reference[key], target[key], key)

        self.assertEqual(len(reference["derivation"]), len(target["derivation"]))
        target_case_ids = {}
        for source_case, target_case in zip(
            reference["derivation"], target["derivation"]
        ):
            target_case = dict(target_case)
            case_id = target_case.pop("target_case_id")
            self.assertEqual(source_case, target_case)
            expected = (
                f"{source_case['example_id']}:{target['family']}:"
                f"{source_case['difficulty']}:{source_case['source_ordinal']}"
            )
            self.assertEqual(case_id, expected)
            target_case_ids[self._case_key(source_case)] = case_id

        self.assertEqual(
            len(reference["request_matrix"]), len(target["request_matrix"])
        )
        for source_request, target_request in zip(
            reference["request_matrix"], target["request_matrix"]
        ):
            self.assertEqual(
                set(source_request),
                {
                    "example_id",
                    "difficulty",
                    "source_ordinal",
                    "context",
                    "components",
                    "wire",
                },
            )
            self.assertEqual(
                set(target_request), set(source_request) | {"target_request"}
            )
            target_request = copy.deepcopy(target_request)
            typed = target_request.pop("target_request")
            for key in (
                "example_id",
                "difficulty",
                "source_ordinal",
                "context",
                "components",
            ):
                self.assertEqual(source_request[key], target_request[key], key)
            source_wire = source_request["wire"]
            target_wire = target_request["wire"]
            self.assertEqual(
                set(source_wire),
                {"model", "messages", "prompt", "temperature", "json_intent"},
            )
            self.assertEqual(set(target_wire), set(source_wire))
            self.assertEqual(source_wire["model"], target_wire["model"])
            self.assertEqual(source_wire["temperature"], target_wire["temperature"])
            self.assertEqual(source_wire["json_intent"], target_wire["json_intent"])
            self.assertEqual(
                [message["role"] for message in source_wire["messages"]], ["user"]
            )
            self.assertEqual(
                [message["role"] for message in target_wire["messages"]], ["user"]
            )
            self.assertEqual(source_wire["messages"][0]["content"], source_wire["prompt"])
            self.assertEqual(target_wire["messages"][0]["content"], target_wire["prompt"])
            if source_request["context"] == "memory_diag":
                expected_prompt = self._inject_dialogue(
                    source_wire["prompt"], source_request["components"]["dialogue"]
                )
                self.assertEqual(target_wire["prompt"], expected_prompt)
            else:
                self.assertEqual(source_wire["prompt"], target_wire["prompt"])

            case_key = self._case_key(source_request)
            self.assertEqual(typed["case_id"], target_case_ids[case_key])
            self.assertEqual(typed["purpose"], "infer")
            for key in ("model", "messages", "prompt", "temperature", "json_intent"):
                self.assertEqual(typed[key], target_wire[key], key)
            self._assert_target_components_in_prompt(target_request)

        self._assert_target_projection(
            target["target_projection"],
            target["derivation"],
            target["request_matrix"],
        )

    def _assert_target_projection(
        self, projection: dict, derivation: list[dict], matrix: list[dict]
    ) -> None:
        results = projection["results"]
        diagnostics = projection["diagnostics"]
        self.assertEqual(len(results), len(derivation))
        self.assertEqual(len(diagnostics), len(derivation))
        memory_api_requests = {
            item["target_request"]["case_id"]: item["target_request"]
            for item in matrix
            if item["context"] == "memory_api"
        }
        expected_result_fields = {
            "record_type", "case_id", "example_id", "mode", "difficulty",
            "context", "memory_id", "domain", "template_id", "utterance",
            "reference_ground_truth", "inferred_action",
        }
        expected_diagnostic_fields = {
            "record_type", "case_id", "example_id", "mode", "difficulty",
            "context", "memory_id", "template_id", "set_derived_slots", "provider",
        }
        for result, diagnostic, case in zip(results, diagnostics, derivation):
            self.assertEqual(set(result), expected_result_fields)
            self.assertEqual(set(diagnostic), expected_diagnostic_fields)
            case_id = case["target_case_id"]
            self.assertEqual(result["case_id"], case_id)
            self.assertEqual(diagnostic["case_id"], case_id)
            self.assertEqual(result["example_id"], case["example_id"])
            self.assertEqual(diagnostic["example_id"], case["example_id"])
            self.assertEqual(result["memory_id"], case["example_id"])
            self.assertEqual(diagnostic["memory_id"], case["example_id"])
            provider = diagnostic["provider"]
            self.assertEqual(
                set(provider), {"request", "request_id", "usage", "raw_response"}
            )
            expected_request = dict(memory_api_requests[case_id])
            expected_request.pop("case_id")
            self.assertEqual(provider["request"], expected_request)
            self.assertEqual(provider["request_id"], f"request-{case_id}")
            self.assertEqual(provider["usage"], {})
            self.assertEqual(provider["raw_response"], {"echo": case_id})

    def _assert_target_components_in_prompt(self, request: dict) -> None:
        prompt = request["wire"]["prompt"]
        components = request["components"]
        self.assertIn(components["tool_schema"], prompt)
        self.assertEqual(prompt.count(components["current"]), 1)
        for name in ("preference", "api_history", "dialogue"):
            content = components[name]
            if content is not None:
                self.assertEqual(prompt.count(content), 1, (name, request))

    @staticmethod
    def _inject_dialogue(prompt: str, dialogue: str) -> str:
        marker = "\nCurrent User Utterance:\n"
        return prompt.replace(
            marker,
            f"\nCurrent Dialogue Context:\n{dialogue}\n" + marker,
            1,
        )

    @staticmethod
    def _case_key(item: dict) -> tuple[str, str, int]:
        return item["example_id"], item["difficulty"], item["source_ordinal"]

    def _assert_every_case_has_all_contexts(self, result: dict) -> None:
        expected = {
            self._case_key(case): {"memory_only", "memory_api", "memory_diag", "api_only"}
            for case in result["derivation"]
        }
        observed = {key: set() for key in expected}
        for item in result["request_matrix"]:
            observed[self._case_key(item)].add(item["context"])
        self.assertEqual(observed, expected)

    def _assert_lossless_equal(self, expected: dict, actual: dict, message: str) -> None:
        self.assertEqual(expected, actual, msg=message)

    def _run_probe(
        self,
        filename: str,
        request: dict,
        *,
        reference: bool,
        package_root: Path | None = None,
        reference_root: Path | None = None,
    ) -> dict:
        env = self._scrubbed_environment()
        if reference:
            selected_reference = reference_root or self.reference_root
            env["OURS_MEMORY_REFERENCE_ROOT"] = str(selected_reference)
            env["PYTHONPATH"] = str(selected_reference)
        else:
            env.pop("OURS_MEMORY_REFERENCE_ROOT", None)
            env["PYTHONPATH"] = str(package_root or ROOT)
        command = [sys.executable, "-S", str(PROBES / filename)]
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(request, ensure_ascii=False),
                text=True,
                cwd="/tmp",
                env=env,
                capture_output=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self.fail(f"{filename} timed out after {TIMEOUT_SECONDS}s: {exc.stderr or ''}")
        if completed.returncode != 0:
            self.fail(f"{filename} exited {completed.returncode}: {completed.stderr}")
        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError:
            self.fail(f"{filename} emitted non-canonical JSON: {completed.stdout!r}; stderr={completed.stderr!r}")

    def _assert_mutated_pair_fails(
        self, request: dict, package_root: Path, label: str
    ) -> None:
        reference = self._run_probe(
            "reference_probe.py", request, reference=True
        )
        mutated = self._run_probe(
            "new_probe.py",
            request,
            reference=False,
            package_root=package_root,
        )
        with self.assertRaises(AssertionError):
            if request["family"] == "step1":
                self._assert_lossless_equal(reference, mutated, label)
            else:
                self._assert_step2_pair(reference, mutated)

    @staticmethod
    def _copy_package(temp_root: Path) -> Path:
        package_root = temp_root / "package"
        shutil.copytree(
            ROOT,
            package_root,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build", "dist"),
        )
        return package_root

    def _copy_reference_files(self, temp_root: Path) -> Path:
        reference_root = temp_root / "reference"
        reference_root.mkdir()
        for name in REFERENCE_FILES:
            shutil.copy2(self.reference_root / name, reference_root / name)
        return reference_root

    def _replace_once(self, path: Path, old: str, new: str) -> None:
        text = path.read_text(encoding="utf-8")
        self.assertEqual(text.count(old), 1, (path, old))
        path.write_text(text.replace(old, new, 1), encoding="utf-8")

    @staticmethod
    def _scrubbed_environment() -> dict[str, str]:
        env = dict(os.environ)
        forbidden_fragments = (
            "API_KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "OPENAI",
            "VLLM", "CUDA", "GPU", "PYTHONHOME", "PYTHONPATH",
        )
        for key in list(env):
            if any(fragment in key.upper() for fragment in forbidden_fragments):
                env.pop(key, None)
        env.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
        return env


if __name__ == "__main__":
    unittest.main()
