"""Execute only the standalone package and emit evidence for differential review."""

from __future__ import annotations

from dataclasses import replace
import json
import os
import socket
import subprocess
import sys
from typing import Any
import urllib.request


CONTEXTS = ("memory_only", "memory_api", "memory_diag", "api_only")


def _blocked(*_args: object, **_kwargs: object) -> None:
    raise RuntimeError("probe isolation blocked an external side effect")


def install_tripwires() -> None:
    socket.socket.connect = _blocked  # type: ignore[assignment]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    urllib.request.urlopen = _blocked  # type: ignore[assignment]
    subprocess.Popen = _blocked  # type: ignore[assignment]
    subprocess.run = _blocked  # type: ignore[assignment]
    subprocess.call = _blocked  # type: ignore[assignment]


def response(value: object):
    from ours_memory2.contracts import ProviderResponse

    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return ProviderResponse(text=text)


def _session_inputs(request: dict[str, Any]) -> list[Any]:
    from ours_memory2.contracts import DialogueTurn, SessionInput

    return [
        SessionInput(
            tuple(
                DialogueTurn(turn["role"], turn["message"])
                for turn in item["dialogue"]
            ),
            tuple(item["api_call"]),
        )
        for item in request["sessions"]
    ]


def step1_case(request: dict[str, Any]) -> dict[str, Any]:
    from tests.support import ScriptedProvider
    from ours_memory2.step1 import (
        MemoryState,
        format_dialogue,
        parse_generated_preference,
        update_session,
    )

    sessions = _session_inputs(request)
    parsing = []
    for value in request["parse_inputs"]:
        try:
            parsing.append(parse_generated_preference(value))
        except ValueError:
            parsing.append({})

    scripted = []
    for item in request["two_session_scripts"]:
        scripted.extend(
            (
                response(item["draft"]),
                response({"valid": item["valid"], "feedback": item["feedback"]}),
            )
        )
    provider = ScriptedProvider(scripted, model="probe")
    state = MemoryState()
    states = []
    for session in sessions:
        state = update_session(state, session, provider)
        states.append(project_state(state))

    cap_script = []
    for item in request["ten_slot_scripts"]:
        cap_script.extend(
            (
                response(item["draft"]),
                response({"valid": item["valid"], "feedback": item["feedback"]}),
            )
        )
    cap_provider = ScriptedProvider(cap_script, model="probe")
    cap_state = update_session(MemoryState(), sessions[0], cap_provider)
    attempts = cap_state.evolution_history[0]["refinement_process"]

    retry_cases = {
        name: run_retry_case(values, sessions[0], str(request["example"]["example_id"]))
        for name, values in request["step1_retry_cases"].items()
    }
    return {
        "schema_version": 3,
        "family": "step1",
        "identity": {"example_id": str(request["example"]["example_id"])},
        "parsing": parsing,
        "formatted_dialogue": format_dialogue(sessions[0]),
        "two_session": {
            "states": states,
            "second_generate_input": {
                "previous_preference": states[0]["preference"],
                "dialogue": states[0]["dialogue"]
                + "\n=== Session 2 ===\n"
                + format_dialogue(sessions[1]),
                "api_history": states[0]["api_history"]
                + [f"[Session 2] {item}" for item in sessions[1].api_call],
            },
            "streams": canonical_step1_streams(
                str(request["example"]["example_id"]), state
            ),
            "provider_requests": project_provider_requests(provider.requests),
        },
        "ten_slot": {
            "generation_slots": len(
                [
                    item
                    for item in cap_provider.requests
                    if item.purpose.value in {"generate", "refine"}
                ]
            ),
            "verifications": len(
                [item for item in cap_provider.requests if item.purpose.value == "verify"]
            ),
            "refinements": len(
                [item for item in cap_provider.requests if item.purpose.value == "refine"]
            ),
            "steps": [item["step"] for item in attempts],
            "drafts": [item["draft_preference"] for item in attempts],
            "validity": [item["is_valid"] for item in attempts],
            "feedback": [item["verifier_feedback"] for item in attempts],
            "final_preference": cap_state.implicit_preference,
            "termination": cap_state.evolution_history[0]["termination_reason"],
            "streams": canonical_step1_streams(
                str(request["example"]["example_id"]), cap_state
            ),
            "provider_requests": project_provider_requests(cap_provider.requests),
        },
        "retry_cases": retry_cases,
        "departures": departure_evidence(),
    }


def run_retry_case(values: list[object], session: Any, example_id: str) -> dict[str, Any]:
    from tests.support import ScriptedProvider
    from ours_memory2.step1 import MemoryState, update_session

    provider = ScriptedProvider([response(value) for value in values], model="probe")
    state = update_session(MemoryState(), session, provider)
    evolution = state.evolution_history[0]
    attempts = evolution["refinement_process"]
    return {
        "provider_requests": project_provider_requests(provider.requests),
        "generation_slots": len(
            [
                request
                for request in provider.requests
                if request.purpose.value in {"generate", "refine"}
            ]
        ),
        "verifications": len(
            [request for request in provider.requests if request.purpose.value == "verify"]
        ),
        "steps": [attempt["step"] for attempt in attempts],
        "generation_purposes": [
            attempt["generation_purpose"] for attempt in attempts
        ],
        "lineage": [
            {
                "parent": attempt["refinement_parent"],
                "feedback": attempt["refinement_feedback"],
            }
            for attempt in attempts
        ],
        "slots_used": evolution["generation_slots_used"],
        "verified_candidates": evolution["verified_candidates"],
        "final_feedback": evolution["final_verifier_feedback"],
        "termination": evolution["termination_reason"],
        "final_preference": state.implicit_preference,
        "streams": canonical_step1_streams(example_id, state),
    }


def project_state(state: Any) -> dict[str, Any]:
    return {
        "preference": state.implicit_preference,
        "dialogue": state.accumulated_dialogue,
        "api_history": list(state.accumulated_api_calls),
        "session_count": state.session_count,
        "evolution": canonical_evolution(state.evolution_history),
    }


def canonical_evolution(evolution: Any) -> list[dict[str, Any]]:
    copied = json.loads(json.dumps(evolution, ensure_ascii=False))
    for session in copied:
        session.pop("diagnostics", None)
    return copied


def canonical_step1_streams(example_id: str, state: Any) -> dict[str, Any]:
    from ours_memory2.contracts import ExampleInput
    from ours_memory2.step1 import _project_example

    run = _project_example(ExampleInput(example_id, ()), state)
    final = json.loads(json.dumps(run.final_rows, ensure_ascii=False))
    for session in final[0]["preference_evolution_history"]:
        session.pop("diagnostics", None)
    return {
        "final": final,
        "drafts": list(run.draft_rows),
        "verifiers": list(run.verifier_rows),
    }


def project_provider_requests(requests: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "purpose": item.purpose.value,
            "model": item.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in item.messages
            ],
            "prompt": item.prompt,
            "temperature": item.temperature,
            "json_intent": item.json_intent,
        }
        for item in requests
    ]


def departure_evidence() -> dict[str, Any]:
    from ours_memory2.contracts import InputContractError
    from ours_memory2.step1 import parse_verifier_decision
    from ours_memory2.step2_common import normalize_context

    try:
        parse_verifier_decision('{"valid":"yes"}')
        verifier = "accepted"
    except ValueError:
        verifier = "rejected"
    try:
        normalize_context("api-only")
        alias = "accepted"
    except InputContractError:
        alias = "rejected"
    return {
        "provider_errors_are_explicit": "ProviderError",
        "source_context_alias": alias,
        "strict_line_local_utf8_jsonl": "one-object-per-nonblank-line",
        "strict_verifier_shape": verifier,
        "memory_diag_dialogue_injection": "approved-target-normalization",
        "target_step2_additions": [
            "record_type",
            "case_id",
            "memory_id",
            "request_provenance",
            "response_diagnostics",
        ],
    }


def make_resources(request: dict[str, Any]):
    from ours_memory2.step2_common import ManifestResources, normalize_multiturn_templates

    groups = {
        name: tuple(item["rules"])
        for name, item in request["preference_groups"].items()
    }
    return ManifestResources(
        examples=tuple(request["examples"]),
        memories={str(item["example_id"]): item for item in request["memories"]},
        single_query_map=request["single_query_map"],
        multiturn_templates=normalize_multiturn_templates(
            request["multiturn_data"], shape="auto"
        ),
        preference_slots={
            key: tuple(value) for key, value in request["preference_slots"].items()
        },
        preference_groups=groups,
        tool_schema=request["tool_schema"],
    )


def step2_case(request: dict[str, Any], family: str) -> dict[str, Any]:
    from ours_memory2.contracts import ProviderResponse
    from ours_memory2.step2_common import (
        make_provider_request,
        normalize_context,
        project_rows,
    )
    from ours_memory2.step2_multi import build_multi_cases
    from ours_memory2.step2_single import build_single_cases

    resources = make_resources(request)
    builder = build_single_cases if family == "single" else build_multi_cases
    cases = builder(resources, difficulty="all")
    derivation = [target_derivation_record(case) for case in cases]
    request_matrix = []
    for case in cases:
        for context in CONTEXTS:
            provider_request = make_provider_request(
                resources=resources,
                case=case,
                context=context,
                model="probe",
            )
            request_matrix.append(
                target_request_record(resources, case, context, provider_request)
            )

    shared_logs = []
    shared_results = []
    target_results = []
    target_diagnostics = []
    projection_context = normalize_context("memory_api")
    for case in cases:
        provider_request = make_provider_request(
            resources=resources,
            case=case,
            context=projection_context,
            model="probe",
        )
        output = json.dumps({"case": case.case_id}, ensure_ascii=False)
        provider_response = ProviderResponse(
            text=output,
            request_id=f"request-{case.case_id}",
            usage={},
            raw_response={"echo": case.case_id},
        )
        result, diagnostic = project_rows(
            case=case,
            mode=family,
            context=projection_context,
            request=provider_request,
            response=provider_response,
        )
        target_results.append(result)
        target_diagnostics.append(diagnostic)
        ordinal = int(case.case_id.rsplit(":", 1)[1])
        shared_logs.append(
            {
                "example_id": case.example_id,
                "example_id_sub": f"{case.example_id}_{ordinal}",
                "model_name": provider_request.model,
                "context_type": "memory_api",
                "pref_type": "implicit",
                "injected_utterance": case.utterance,
                "reference_ground_truth": list(case.ground_truth),
                "model_input": provider_request.prompt,
                "model_output": output,
                "reasoning_content": "",
                "token_counts": {},
            }
        )
        shared_results.append(
            {
                "example_id_sub": f"{case.example_id}_{ordinal}",
                "test_utterance": case.utterance,
                "reference_ground_truth": list(case.ground_truth),
                "llm_output": output,
                "reasoning_content": "",
                "token_counts": {},
                "reasoning_token_count": 0,
            }
        )
    return {
        "schema_version": 3,
        "family": family,
        "derivation": derivation,
        "request_matrix": request_matrix,
        "projection": {
            "legacy_logs": shared_logs,
            "legacy_results": shared_results,
        },
        "target_projection": {
            "results": target_results,
            "diagnostics": target_diagnostics,
        },
        "hard_null": hard_null_evidence(request, family, resources),
        "identity": {
            "example_ids": [str(item["example_id"]) for item in request["examples"]],
            "memory_resource_ids": [
                str(item["example_id"]) for item in request["memories"]
            ],
        },
        "carried_by_example": carried_records(request),
        "departures": departure_evidence(),
    }


def target_derivation_record(case: Any) -> dict[str, Any]:
    return {
        "example_id": case.example_id,
        "difficulty": case.difficulty.value,
        "source_ordinal": int(case.case_id.rsplit(":", 1)[1]),
        "domain": case.domain,
        "utterance": case.utterance,
        "ground_truth": list(case.ground_truth),
        "template_id": case.template_id,
        "set_derived_slots": list(case.set_derived_slots),
        "target_case_id": case.case_id,
    }


def target_request_record(
    resources: Any, case: Any, context: str, provider_request: Any
) -> dict[str, Any]:
    return {
        "example_id": case.example_id,
        "difficulty": case.difficulty.value,
        "source_ordinal": int(case.case_id.rsplit(":", 1)[1]),
        "context": context,
        "components": context_components(resources, case, context),
        "wire": {
            "model": provider_request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in provider_request.messages
            ],
            "prompt": provider_request.prompt,
            "temperature": {
                "present": provider_request.temperature is not None,
                "value": provider_request.temperature,
            },
            "json_intent": provider_request.json_intent,
        },
        "target_request": {
            "case_id": case.case_id,
            "purpose": provider_request.purpose.value,
            "model": provider_request.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in provider_request.messages
            ],
            "prompt": provider_request.prompt,
            "temperature": {
                "present": provider_request.temperature is not None,
                "value": provider_request.temperature,
            },
            "json_intent": provider_request.json_intent,
        },
    }


def context_components(resources: Any, case: Any, context: str) -> dict[str, Any]:
    example = next(
        item for item in resources.examples if item["example_id"] == case.example_id
    )
    memory = resources.memories[case.example_id]
    preference = json.dumps(
        memory.get("final_implicit_preference"), ensure_ascii=False, indent=2
    )
    api_history = "\n".join(memory["final_accumulated_api_calls"]) or "None"
    sessions = []
    for index, session in enumerate(example.get("sessions", []), start=1):
        lines = [f"[Session {index}]"]
        for turn in session.get("dialogue", []):
            role = str(turn.get("role", "")).capitalize()
            message = turn.get("message", "")
            if role and message:
                lines.append(f"{role}: {message}")
        sessions.append("\n".join(lines))
    dialogue = "\n\n".join(sessions) or "None"
    return {
        "tool_schema": json.dumps(
            resources.tool_schema, ensure_ascii=False, indent=2
        ),
        "preference": (
            preference if context in {"memory_only", "memory_api", "memory_diag"} else None
        ),
        "api_history": api_history if context in {"memory_api", "api_only"} else None,
        "dialogue": dialogue if context == "memory_diag" else None,
        "current": case.utterance.strip(),
    }


def hard_null_evidence(
    request: dict[str, Any], family: str, resources: Any
) -> dict[str, Any]:
    from ours_memory2.contracts import InputContractError
    from ours_memory2.step2_multi import build_multi_cases
    from ours_memory2.step2_single import build_single_cases

    builder = build_single_cases if family == "single" else build_multi_cases
    evidence = {}
    for name, rules in request["hard_null_rules"].items():
        current = replace(
            resources,
            examples=(request["example"],),
            preference_groups={"comfort": tuple(rules)},
        )
        try:
            cases = builder(current, difficulty="hard")
        except InputContractError as exc:
            if exc.path != "cases":
                raise
            cases = ()
        evidence[name] = [
            {
                "example_id": case.example_id,
                "domain": case.domain,
                "utterance": case.utterance,
                "ground_truth": list(case.ground_truth),
                "template_id": case.template_id,
            }
            for case in cases
        ]
    return evidence


def carried_records(request: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "example_id": str(memory["example_id"]),
            "dialogue": memory["final_accumulated_dialogue"],
            "api_history": memory["final_accumulated_api_calls"],
            "preference": memory["final_implicit_preference"],
        }
        for memory in request["memories"]
    ]


def main() -> int:
    if os.environ.get("OURS_MEMORY_REFERENCE_ROOT"):
        raise RuntimeError("new probe received the forbidden reference root")
    install_tripwires()
    import ours_memory2

    forbidden = {"openai", "torch", "vllm", "google", "tqdm"}.intersection(
        sys.modules
    )
    if forbidden:
        raise RuntimeError(
            f"new package imported forbidden dependencies: {sorted(forbidden)}"
        )
    request = json.loads(sys.stdin.read())
    output = (
        step1_case(request)
        if request["family"] == "step1"
        else step2_case(request, request["family"])
    )
    sys.stdout.write(
        json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
