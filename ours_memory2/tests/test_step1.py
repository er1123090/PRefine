from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ours_memory2.contracts import ProviderError, ProviderPurpose, ProviderResponse, Step1OutputNames
from ours_memory2.jsonl import read_examples_jsonl
from tests.support import ScriptedProvider
from ours_memory2.step1 import (
    MAX_GENERATION_SLOTS,
    VERIFIER_PARSE_FEEDBACK,
    build_and_write_examples,
    build_example_memory,
)


FIXTURES = Path(__file__).parent / "fixtures"


def response(value: object) -> ProviderResponse:
    if isinstance(value, str):
        text = value
    else:
        text = json.dumps(value, ensure_ascii=False)
    return ProviderResponse(text=text)


def one_session_example():
    return read_examples_jsonl(FIXTURES / "valid_two_examples.jsonl")[0]


class Step1MethodTests(unittest.TestCase):
    def test_immediate_valid(self) -> None:
        provider = ScriptedProvider(
            [
                response({"reasoning": "evidence", "implicit_pref": "quiet"}),
                response({"valid": True, "feedback": "supported"}),
            ]
        )
        run = build_example_memory(one_session_example(), provider)

        self.assertEqual(
            [request.purpose for request in provider.requests],
            [ProviderPurpose.GENERATE, ProviderPurpose.VERIFY],
        )
        evolution = run.final_row["preference_evolution_history"][0]
        self.assertEqual(evolution["termination_reason"], "valid")
        self.assertEqual(evolution["generation_slots_used"], 1)
        self.assertEqual(evolution["verified_candidates"], 1)
        self.assertEqual(evolution["final_verifier_feedback"], "supported")
        self.assertEqual(run.final_row["final_implicit_preference"]["implicit_pref"], "quiet")
        self.assertEqual(len(run.draft_rows), 1)
        self.assertEqual(len(run.verifier_rows), 1)

    def test_refine_then_valid_preserves_lineage_and_request_order(self) -> None:
        provider = ScriptedProvider(
            [
                response({"implicit_pref": "draft-1"}),
                response({"valid": False, "feedback": "too specific"}),
                response({"implicit_pref": "draft-2"}),
                response({"valid": True, "feedback": "now supported"}),
            ]
        )
        run = build_example_memory(one_session_example(), provider)

        self.assertEqual(
            [request.purpose for request in provider.requests],
            [
                ProviderPurpose.GENERATE,
                ProviderPurpose.VERIFY,
                ProviderPurpose.REFINE,
                ProviderPurpose.VERIFY,
            ],
        )
        second = run.draft_rows[1]
        self.assertEqual(second["step"], 2)
        self.assertEqual(second["generation_purpose"], "refine")
        self.assertEqual(second["refinement_parent"]["implicit_pref"], "draft-1")
        self.assertEqual(second["refinement_feedback"], "too specific")
        self.assertIn("draft-1", provider.requests[2].messages[1].content)
        self.assertIn("too specific", provider.requests[2].messages[1].content)
        self.assertEqual(run.final_row["final_implicit_preference"]["implicit_pref"], "draft-2")

    def test_malformed_first_slot_retries_with_initial_generation_role(self) -> None:
        provider = ScriptedProvider(
            [
                response("not-json"),
                response({"implicit_pref": "candidate-2"}),
                response({"valid": True, "feedback": "supported"}),
            ]
        )
        run = build_example_memory(one_session_example(), provider)

        self.assertEqual(
            [request.purpose for request in provider.requests],
            [
                ProviderPurpose.GENERATE,
                ProviderPurpose.GENERATE,
                ProviderPurpose.VERIFY,
            ],
        )
        for request in provider.requests[:2]:
            self.assertEqual([message.role for message in request.messages], ["system", "user"])
            self.assertIn("Infer Latent Preference", request.prompt)
            self.assertNotIn("Refine Preference", request.prompt)
        self.assertEqual(len(run.draft_rows), 1)
        self.assertEqual(len(run.verifier_rows), 1)
        self.assertEqual(run.draft_rows[0]["step"], 2)
        self.assertEqual(run.draft_rows[0]["generation_purpose"], "generate")
        self.assertIsNone(run.draft_rows[0]["refinement_parent"])
        self.assertIsNone(run.draft_rows[0]["refinement_feedback"])
        evolution = run.final_row["preference_evolution_history"][0]
        self.assertEqual(evolution["generation_slots_used"], 2)
        self.assertEqual(evolution["verified_candidates"], 1)
        self.assertEqual(evolution["termination_reason"], "valid")
        self.assertEqual(
            run.final_row["final_implicit_preference"],
            {"implicit_pref": "candidate-2"},
        )

    def test_false_verification_with_empty_feedback_retries_initial_generation(self) -> None:
        provider = ScriptedProvider(
            [
                response({"implicit_pref": "candidate-1"}),
                response({"valid": False, "feedback": ""}),
                response({"implicit_pref": "candidate-2"}),
                response({"valid": True, "feedback": "supported"}),
            ]
        )
        run = build_example_memory(one_session_example(), provider)

        self.assertEqual(
            [request.purpose for request in provider.requests],
            [
                ProviderPurpose.GENERATE,
                ProviderPurpose.VERIFY,
                ProviderPurpose.GENERATE,
                ProviderPurpose.VERIFY,
            ],
        )
        retry = provider.requests[2]
        self.assertIn("Infer Latent Preference", retry.prompt)
        self.assertNotIn("Refine Preference", retry.prompt)
        self.assertEqual(
            [row["generation_purpose"] for row in run.draft_rows],
            ["generate", "generate"],
        )
        self.assertIsNone(run.draft_rows[1]["refinement_parent"])
        self.assertIsNone(run.draft_rows[1]["refinement_feedback"])
        evolution = run.final_row["preference_evolution_history"][0]
        self.assertEqual(evolution["generation_slots_used"], 2)
        self.assertEqual(evolution["verified_candidates"], 2)
        self.assertEqual(evolution["final_verifier_feedback"], "supported")
        self.assertEqual(
            run.final_row["final_implicit_preference"],
            {"implicit_pref": "candidate-2"},
        )

    def test_never_valid_uses_exact_ten_candidates_and_nine_refinements(self) -> None:
        scripted = []
        for step in range(1, MAX_GENERATION_SLOTS + 1):
            scripted.extend(
                [
                    response({"implicit_pref": f"candidate-{step}"}),
                    response({"valid": False, "feedback": f"feedback-{step}"}),
                ]
            )
        provider = ScriptedProvider(scripted)
        run = build_example_memory(one_session_example(), provider)

        purposes = [request.purpose for request in provider.requests]
        self.assertEqual(len(purposes), 20)
        self.assertEqual(purposes.count(ProviderPurpose.GENERATE), 1)
        self.assertEqual(purposes.count(ProviderPurpose.REFINE), 9)
        self.assertEqual(purposes.count(ProviderPurpose.VERIFY), 10)
        self.assertEqual([row["step"] for row in run.draft_rows], list(range(1, 11)))
        self.assertEqual([row["step"] for row in run.verifier_rows], list(range(1, 11)))
        evolution = run.final_row["preference_evolution_history"][0]
        self.assertEqual(evolution["termination_reason"], "max_attempts_invalid")
        self.assertEqual(evolution["final_verifier_feedback"], "feedback-10")
        self.assertEqual(evolution["verified_candidates"], 10)
        self.assertEqual(run.final_row["final_implicit_preference"]["implicit_pref"], "candidate-10")

    def test_two_sessions_carry_state_order_unicode_metadata_and_lossless_streams(self) -> None:
        example = read_examples_jsonl(FIXTURES / "step1_two_session.jsonl")[0]
        provider = ScriptedProvider(
            [
                response({"implicit_pref": "조용한 장소"}),
                response({"valid": True, "feedback": "근거 있음"}),
                response({"implicit_pref": "조용함을 지속 선호"}),
                response({"valid": True, "feedback": "두 세션에서 지지됨"}),
            ]
        )
        run = build_example_memory(example, provider)

        session_two_request = provider.requests[2]
        session_two_message = session_two_request.messages[0].content
        self.assertIn("조용한 장소", session_two_message)
        self.assertIn("서울에서는 조용한 식당을 선호해요.", session_two_message)
        self.assertIn("다음 주에도 같은 취향으로 예약해 줘.", session_two_message)
        self.assertIn("restaurant.search", session_two_message)
        self.assertIn("restaurant.reserve", session_two_message)
        self.assertEqual(
            session_two_request.prompt, session_two_request.messages[1].content
        )

        final = run.final_row
        self.assertEqual(final["example_id"], "unicode-user")
        self.assertEqual(final["total_sessions_processed"], 2)
        self.assertEqual(final["metadata"], {"cohort": "테스트", "source_order": 3})
        self.assertEqual(
            final["final_accumulated_api_calls"],
            [
                "[Session 1] restaurant.search(city='서울',noise='quiet')",
                "[Session 2] restaurant.reserve(day='next-week')",
            ],
        )
        self.assertEqual(
            [(row["session_index"], row["step"]) for row in run.draft_rows],
            [(1, 1), (2, 1)],
        )
        self._assert_lossless(run)

    def test_malformed_verifier_counts_as_false_and_preserves_draft_input(self) -> None:
        malformed_values = (
            "not-json",
            {"feedback": "missing valid"},
            {"valid": "yes", "feedback": "wrong type"},
        )
        for malformed in malformed_values:
            with self.subTest(malformed=malformed):
                provider = ScriptedProvider(
                    [
                        response({"implicit_pref": "draft-1"}),
                        response(malformed),
                        response({"implicit_pref": "draft-2"}),
                        response({"valid": True, "feedback": "valid"}),
                    ]
                )
                run = build_example_memory(one_session_example(), provider)
                first = run.verifier_rows[0]
                self.assertFalse(first["is_valid"])
                self.assertEqual(first["verifier_feedback"], VERIFIER_PARSE_FEEDBACK)
                self.assertEqual(first["verifier_output"], {})
                self.assertIn("draft-1", first["verifier_input"])
                self.assertEqual(
                    run.final_row["preference_evolution_history"][0]["diagnostics"][0]["kind"],
                    "verifier_parse_error",
                )

    def test_malformed_generation_consumes_slots_and_tenth_retains_last_candidate(self) -> None:
        scripted = [response({"implicit_pref": "candidate-1"}), response({"valid": False, "feedback": "f1"})]
        for _ in range(2, 11):
            scripted.append(response("not-json"))
        provider = ScriptedProvider(scripted)
        run = build_example_memory(one_session_example(), provider)

        self.assertEqual(len(provider.requests), 11)
        self.assertEqual(provider.requests[0].purpose, ProviderPurpose.GENERATE)
        self.assertEqual(provider.requests[1].purpose, ProviderPurpose.VERIFY)
        self.assertTrue(
            all(
                request.purpose is ProviderPurpose.REFINE
                for request in provider.requests[2:]
            )
        )
        for request in provider.requests[2:]:
            self.assertIn("Refine Preference", request.prompt)
            self.assertIn("candidate-1", request.prompt)
            self.assertIn("f1", request.prompt)
        self.assertEqual(len(run.draft_rows), 1)
        evolution = run.final_row["preference_evolution_history"][0]
        self.assertEqual(evolution["generation_slots_used"], 10)
        self.assertEqual(evolution["termination_reason"], "max_attempts_generation_error")
        self.assertEqual(len(evolution["diagnostics"]), 9)
        self.assertEqual(run.final_row["final_implicit_preference"]["implicit_pref"], "candidate-1")

    def test_provider_failure_writes_none_of_current_example(self) -> None:
        class FailingProvider:
            model = "fixture-model"

            def __init__(self) -> None:
                self.requests = []

            def complete(self, request):
                self.requests.append(request)
                raise ProviderError("transport failed")

        provider = FailingProvider()
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "outputs"
            with self.assertRaises(ProviderError):
                build_and_write_examples((one_session_example(),), provider, output)
            self.assertFalse(output.exists())
            self.assertEqual(len(provider.requests), 1)

    def test_provider_failure_preserves_prior_example_without_partial_current_rows(self) -> None:
        examples = read_examples_jsonl(FIXTURES / "valid_two_examples.jsonl")
        calls = 0

        def scripted(request):
            nonlocal calls
            calls += 1
            if calls == 1:
                return response({"implicit_pref": "first-complete"})
            if calls == 2:
                return response({"valid": True, "feedback": "ok"})
            raise ProviderError("second example transport failed")

        provider = ScriptedProvider(scripted)
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "outputs"
            with self.assertRaises(ProviderError):
                build_and_write_examples(examples, provider, output)
            names = Step1OutputNames()
            final_rows = self._read_rows(output / names.final)
            draft_rows = self._read_rows(output / names.drafts)
            verifier_rows = self._read_rows(output / names.verifiers)

        self.assertEqual([row["example_id"] for row in final_rows], ["7"])
        self.assertEqual([row["example_id"] for row in draft_rows], ["7"])
        self.assertEqual([row["example_id"] for row in verifier_rows], ["7"])
        self.assertEqual(calls, 3)

    def test_jsonl_streams_round_trip_with_frozen_field_contract(self) -> None:
        example = read_examples_jsonl(FIXTURES / "step1_two_session.jsonl")[0]
        provider = ScriptedProvider(
            [
                response({"implicit_pref": "초기"}),
                response({"valid": True, "feedback": "ok-1"}),
                response({"implicit_pref": "최종"}),
                response({"valid": True, "feedback": "ok-2"}),
            ]
        )
        contract = json.loads(
            (FIXTURES / "step1_stream_contract.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "outputs"
            runs = build_and_write_examples((example,), provider, output)
            names = Step1OutputNames()
            final_rows = self._read_rows(output / names.final)
            draft_rows = self._read_rows(output / names.drafts)
            verifier_rows = self._read_rows(output / names.verifiers)

        self.assertEqual(final_rows, list(runs[0].final_rows))
        self.assertEqual(draft_rows, list(runs[0].draft_rows))
        self.assertEqual(verifier_rows, list(runs[0].verifier_rows))
        self.assertEqual(set(final_rows[0]), set(contract["final_required"]))
        self.assertEqual(set(draft_rows[0]), set(contract["draft_required"]))
        self.assertEqual(set(verifier_rows[0]), set(contract["verifier_required"]))
        self.assertEqual(
            contract["source_to_target"]["MemoryState.evolution_log"],
            "final.preference_evolution_history",
        )
        self.assertEqual(
            contract["source_to_target"]["attempt.generation_lineage"],
            "draft.generation_purpose+refinement_parent+refinement_feedback",
        )
        self._assert_lossless(runs[0])

    def _assert_lossless(self, run) -> None:
        reconstructed = []
        verifier_by_key = {
            (row["example_id"], row["session_index"], row["step"]): row
            for row in run.verifier_rows
        }
        for draft in run.draft_rows:
            key = (draft["example_id"], draft["session_index"], draft["step"])
            verifier = verifier_by_key[key]
            reconstructed.append(
                {
                    "step": draft["step"],
                    "draft_preference": draft["draft_preference"],
                    "is_valid": verifier["is_valid"],
                    "verifier_feedback": verifier["verifier_feedback"],
                    "verifier_input": verifier["verifier_input"],
                    "verifier_output": verifier["verifier_output"],
                    "generation_purpose": draft["generation_purpose"],
                    "refinement_parent": draft["refinement_parent"],
                    "refinement_feedback": draft["refinement_feedback"],
                }
            )
        expected = [
            attempt
            for session in run.final_row["preference_evolution_history"]
            for attempt in session["refinement_process"]
        ]
        self.assertEqual(reconstructed, expected)

    @staticmethod
    def _read_rows(path: Path) -> list[dict]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
