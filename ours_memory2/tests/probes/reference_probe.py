"""Execute cited legacy boundaries and retain only directly observed evidence."""

from __future__ import annotations

import ast
import asyncio
import copy
from dataclasses import dataclass, field
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple
import urllib.request


CONTEXTS = ("memory_only", "memory_api", "memory_diag", "api_only")
FILES = {
    "step1": "Preference_Memory_step1_LATENTPREF.py",
    "single": "Preference_Memory_step2_ACTION_singleturn_api.py",
    "multi": "Preference_Memory_step2_ACTION_multiturn_api.py",
    "single_api_only": "Preference_Memory_step2_ACTION_singleturn_VLLM.py",
    "multi_api_only": "Preference_Memory_step2_ACTION_multiturn_VLLM.py",
    "prompts": "prompt_inference.py",
}
AsyncOpenAI = Any
tqdm = SimpleNamespace(tqdm=Any)


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


def load_nodes(
    path: Path, names: set[str], *, include_memory_state: bool = False
) -> dict[str, Any]:
    """AST-load only named functions and the small Step 1 state dataclass."""

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    selected: list[ast.stmt] = []
    for node in tree.body:
        if include_memory_state and isinstance(node, ast.ClassDef) and node.name == "MemoryState":
            selected.append(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.ClassDef):
            selected.extend(
                item
                for item in node.body
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                and item.name in names
            )
    found = {getattr(node, "name", "") for node in selected}
    missing = names - found
    if missing:
        raise RuntimeError(f"missing cited helper(s) in {path.name}: {sorted(missing)}")
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = globals().copy()
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def load_string_constant(path: Path, name: str) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, str):
                return value
    raise RuntimeError(f"missing string constant {name} in {path.name}")


def _wire_text(value: object) -> str:
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def step1_case(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    source = root / FILES["step1"]
    ns = load_nodes(
        source,
        {
            "_generate_preference",
            "_parse_json",
            "_verify_preference",
            "format_dialogue",
            "update_memory",
        },
        include_memory_state=True,
    )
    prompt_source = root / "prompt_update3.py"
    prompt_constants = {
        name: load_string_constant(prompt_source, name)
        for name in (
            "LATENT_PREF_SYSTEM_PROMPT",
            "LATENT_PREF_INITIAL_PROMPT",
            "LATENT_PREF_REFINEMENT_PROMPT",
            "LATENT_PREF_VERIFIER_PROMPT",
        )
    }
    for function_name in ("_generate_preference", "_verify_preference"):
        ns[function_name].__globals__.update(prompt_constants)
    source_generate = ns["_generate_preference"]

    class Oracle:
        max_retries = 10
        _parse_json = ns["_parse_json"]
        update_memory = ns["update_memory"]
        _verify_preference = ns["_verify_preference"]
        model = "probe"

        def __init__(self, scripts: list[object]) -> None:
            self.scripts = iter(scripts)
            self.calls: list[dict[str, Any]] = []
            self.generation_inputs: list[dict[str, Any]] = []

        async def _generate_preference(self, **kwargs: Any) -> dict[str, Any]:
            result = await source_generate(self, **kwargs)
            self.generation_inputs.append({**kwargs, "result": result})
            return result

        async def _call_llm(
            self, system_prompt: str, user_prompt: str, temperature: float = 0.0
        ) -> str:
            if system_prompt == "You are a Preference Verification Module. Output JSON only.":
                purpose = "verify"
            elif user_prompt == prompt_constants["LATENT_PREF_INITIAL_PROMPT"]:
                purpose = "generate"
            elif "Refine Preference Based on Evidence Gaps" in user_prompt:
                purpose = "refine"
            else:
                raise RuntimeError("unrecognized actual Step 1 prompt role")
            self.calls.append(
                {
                    "purpose": purpose,
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "prompt": user_prompt,
                    "temperature": temperature,
                    "json_intent": True,
                }
            )
            return _wire_text(next(self.scripts))

    format_dialogue = ns["format_dialogue"]
    MemoryState = ns["MemoryState"]
    parse_owner = Oracle([])
    parsing = [parse_owner._parse_json(value) for value in request["parse_inputs"]]

    carried = Oracle(iter_standard_step1_values(request["two_session_scripts"]))
    state = MemoryState()
    session_snapshots = []
    for session in request["sessions"]:
        state = asyncio.run(
            carried.update_memory(
                state, format_dialogue(session["dialogue"]), session["api_call"]
            )
        )
        session_snapshots.append(project_source_state(state))

    capped = Oracle(iter_standard_step1_values(request["ten_slot_scripts"]))
    cap_state = asyncio.run(
        capped.update_memory(
            MemoryState(),
            format_dialogue(request["sessions"][0]["dialogue"]),
            request["sessions"][0]["api_call"],
        )
    )
    cap_projected = project_source_state(cap_state)
    retry_cases = {}
    for name, values in request["step1_retry_cases"].items():
        oracle = Oracle(values)
        retry_state = asyncio.run(
            oracle.update_memory(
                MemoryState(),
                format_dialogue(request["sessions"][0]["dialogue"]),
                request["sessions"][0]["api_call"],
            )
        )
        retry_cases[name] = project_source_retry(
            retry_state, oracle, str(request["example"]["example_id"])
        )

    return {
        "schema_version": 3,
        "family": "step1",
        "identity": {"example_id": str(request["example"]["example_id"])},
        "parsing": parsing,
        "formatted_dialogue": format_dialogue(request["sessions"][0]["dialogue"]),
        "two_session": {
            "states": session_snapshots,
            "second_generate_input": canonical_generation_input(
                next(
                    call
                    for call in carried.generation_inputs
                    if "=== Session 2 ===" in call["full_dialogue"]
                )
            ),
            "streams": project_streams(
                str(request["example"]["example_id"]), session_snapshots[-1]
            ),
            "provider_requests": carried.calls,
        },
        "ten_slot": {
            **canonical_cap(cap_projected, capped.calls),
            "streams": project_streams(
                str(request["example"]["example_id"]), cap_projected
            ),
            "provider_requests": capped.calls,
        },
        "retry_cases": retry_cases,
        "departures": departure_evidence(),
    }


def iter_standard_step1_values(items: list[dict[str, Any]]) -> list[object]:
    expanded: list[object] = []
    for item in items:
        expanded.extend(
            (
                item["draft"],
                {"valid": item["valid"], "feedback": item["feedback"]},
            )
        )
    return expanded


def project_source_state(state: Any) -> dict[str, Any]:
    preference = normalize_source_preference(state.implicit_pref)
    evolution = []
    for session in state.evolution_log:
        attempts = []
        for attempt_index, attempt in enumerate(session["refinement_process"]):
            normalized = dict(attempt)
            normalized["generation_purpose"] = (
                "generate" if attempt_index == 0 else "refine"
            )
            normalized["refinement_parent"] = (
                None if attempt_index == 0 else attempts[-1]["draft_preference"]
            )
            normalized["refinement_feedback"] = (
                None if attempt_index == 0 else attempts[-1]["verifier_feedback"]
            )
            attempts.append(normalized)
        evolution.append(
            canonical_source_evolution_entry(
                session["session_index"], attempts, session["final_preference_at_session"]
            )
        )
    return {
        "preference": preference,
        "dialogue": state.accumulated_dialogue,
        "api_history": list(state.accumulated_api_calls),
        "session_count": state.session_count,
        "evolution": evolution,
    }


def project_source_retry(state: Any, oracle: Any, example_id: str) -> dict[str, Any]:
    session = state.evolution_log[0]
    generation_calls = [
        call for call in oracle.calls if call["purpose"] in {"generate", "refine"}
    ]
    attempts = []
    for attempt in session["refinement_process"]:
        slot_index = attempt["step"] - 1
        call = generation_calls[slot_index]
        generation_input = oracle.generation_inputs[slot_index]
        normalized = dict(attempt)
        normalized["generation_purpose"] = call["purpose"]
        normalized["refinement_parent"] = (
            normalize_source_preference(generation_input["previous_draft"])
            if call["purpose"] == "refine"
            else None
        )
        normalized["refinement_feedback"] = (
            generation_input["feedback"] if call["purpose"] == "refine" else None
        )
        attempts.append(normalized)
    entry = canonical_source_evolution_entry(
        session["session_index"], attempts, session["final_preference_at_session"]
    )
    entry["generation_slots_used"] = len(generation_calls)
    projected = {
        "preference": normalize_source_preference(state.implicit_pref),
        "dialogue": state.accumulated_dialogue,
        "api_history": list(state.accumulated_api_calls),
        "session_count": state.session_count,
        "evolution": [entry],
    }
    return {
        "provider_requests": oracle.calls,
        "generation_slots": len(generation_calls),
        "verifications": len(
            [call for call in oracle.calls if call["purpose"] == "verify"]
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
        "slots_used": entry["generation_slots_used"],
        "verified_candidates": entry["verified_candidates"],
        "final_feedback": entry["final_verifier_feedback"],
        "termination": entry["termination_reason"],
        "final_preference": projected["preference"],
        "streams": project_streams(example_id, projected),
    }


def canonical_source_evolution_entry(
    session_index: int, attempts: list[dict[str, Any]], final_preference: object
) -> dict[str, Any]:
    valid = bool(attempts and attempts[-1]["is_valid"])
    return {
        "session_index": session_index,
        "refinement_process": attempts,
        "final_preference_at_session": final_preference,
        "termination_reason": "valid" if valid else "max_attempts_invalid",
        "generation_slots_used": attempts[-1]["step"] if attempts else 10,
        "verified_candidates": len(attempts),
        "final_verifier_feedback": (
            attempts[-1]["verifier_feedback"] if attempts else ""
        ),
    }


def normalize_source_preference(value: object) -> object:
    if isinstance(value, str) and value not in ("{}", "None"):
        return json.loads(value)
    return {} if value in ("{}", "None") else value


def project_streams(example_id: str, state: dict[str, Any]) -> dict[str, Any]:
    final = {
        "example_id": example_id,
        "final_implicit_preference": state["preference"],
        "final_accumulated_api_calls": state["api_history"],
        "final_accumulated_dialogue": state["dialogue"],
        "total_sessions_processed": state["session_count"],
        "preference_evolution_history": state["evolution"],
        "metadata": {},
    }
    drafts = []
    verifiers = []
    for session in state["evolution"]:
        for attempt in session["refinement_process"]:
            key = {
                "example_id": example_id,
                "session_index": session["session_index"],
                "step": attempt["step"],
            }
            drafts.append(
                {
                    **key,
                    "draft_preference": attempt["draft_preference"],
                    "generation_purpose": attempt["generation_purpose"],
                    "refinement_parent": attempt["refinement_parent"],
                    "refinement_feedback": attempt["refinement_feedback"],
                }
            )
            verifiers.append(
                {
                    **key,
                    "is_valid": attempt["is_valid"],
                    "verifier_feedback": attempt["verifier_feedback"],
                    "verifier_input": attempt["verifier_input"],
                    "verifier_output": attempt["verifier_output"],
                }
            )
    return {"final": [final], "drafts": drafts, "verifiers": verifiers}


def canonical_generation_input(call: dict[str, Any]) -> dict[str, Any]:
    return {
        "previous_preference": normalize_source_preference(call["prev_implicit"]),
        "dialogue": call["full_dialogue"],
        "api_history": call["full_api_calls"].splitlines(),
    }


def canonical_cap(state: dict[str, Any], calls: list[dict[str, Any]]) -> dict[str, Any]:
    attempts = state["evolution"][0]["refinement_process"]
    return {
        "generation_slots": len(
            [call for call in calls if call["purpose"] in {"generate", "refine"}]
        ),
        "verifications": len([call for call in calls if call["purpose"] == "verify"]),
        "refinements": len([call for call in calls if call["purpose"] == "refine"]),
        "steps": [item["step"] for item in attempts],
        "drafts": [item["draft_preference"] for item in attempts],
        "validity": [item["is_valid"] for item in attempts],
        "feedback": [item["verifier_feedback"] for item in attempts],
        "final_preference": state["preference"],
        "termination": "max_attempts_invalid",
    }


def departure_evidence() -> dict[str, Any]:
    return {
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


def step2_case(root: Path, request: dict[str, Any], family: str) -> dict[str, Any]:
    names = {
        "parse_deepseek_reasoning",
        "assign_user_utterances",
        "build_memory_input_prompt",
        "format_memory_content",
        "format_api_calls",
        "call_llm_api_async",
        "process_single_item",
    }
    if family == "single":
        names.add("generate_func_strings")
    else:
        names.update(
            {
                "load_multiturn_data",
                "parse_api_call_to_dict",
                "generate_single_api_string",
                "merge_and_generate_api_strings",
                "format_multiturn_dialogue",
                "get_multiturn_target_slots",
                "select_multiturn_template",
            }
        )
    source = root / FILES[family]
    ns = load_nodes(source, names)
    api_only_ns = load_nodes(
        root / FILES[f"{family}_api_only"],
        {"build_memory_input_prompt", "format_memory_content", "format_api_calls"},
    )
    prompt_name = (
        "IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE"
        if family == "single"
        else "IMPLICIT_ZS_PROMPT_MEMORY_TEMPLATE_MULTITURN"
    )
    prompt_template = load_string_constant(root / FILES["prompts"], prompt_name)

    with tempfile.TemporaryDirectory(prefix="reference-probe-") as temp:
        temp_root = Path(temp)
        slots = temp_root / "slots.json"
        groups = temp_root / "groups.json"
        slots.write_text(json.dumps(request["preference_slots"]), encoding="utf-8")
        groups.write_text(json.dumps(request["preference_groups"]), encoding="utf-8")
        if family == "single":
            query_source = request["single_query_map"]
        else:
            multiturn_file = temp_root / "multiturn.json"
            multiturn_file.write_text(
                json.dumps(request["multiturn_data"]), encoding="utf-8"
            )
            query_source = ns["load_multiturn_data"](str(multiturn_file))
        derivation = source_derivation(
            ns, slots, groups, query_source, request, family
        )
        request_matrix = source_request_matrix(
            ns,
            api_only_ns,
            prompt_template,
            derivation,
            request,
            family,
        )
        projection = source_projection(
            ns,
            prompt_template,
            derivation,
            request,
            temp_root,
        )
        hard_null = source_hard_null(
            ns, slots, groups, query_source, request, family
        )
    return {
        "schema_version": 3,
        "family": family,
        "derivation": derivation,
        "request_matrix": request_matrix,
        "projection": projection,
        "hard_null": hard_null,
        "identity": {
            "example_ids": [str(item["example_id"]) for item in request["examples"]],
            "memory_resource_ids": [
                str(item["example_id"]) for item in request["memories"]
            ],
        },
        "carried_by_example": carried_records(request),
        "departures": departure_evidence(),
    }


def source_derivation(
    ns: dict[str, Any],
    slots: Path,
    groups: Path,
    query_source: object,
    request: dict[str, Any],
    family: str,
) -> list[dict[str, Any]]:
    derivation = []
    for difficulty in ("easy", "medium", "hard"):
        for example in request["examples"]:
            rows = ns["assign_user_utterances"](
                str(slots), example, query_source, difficulty, str(groups)
            )
            derivation.extend(
                canonical_reference_rows(example, family, difficulty, rows, request)
            )
    return derivation


def canonical_reference_rows(
    example: dict[str, Any],
    family: str,
    difficulty: str,
    rows: list[Any],
    request: dict[str, Any],
) -> list[dict[str, Any]]:
    result = []
    raw_templates = request["multiturn_data"]
    if isinstance(raw_templates, list):
        templates: dict[str, list[dict[str, Any]]] = {}
        for item in raw_templates:
            templates.setdefault(item["target"][0]["domain"], []).append(item)
    else:
        templates = raw_templates
    if difficulty == "hard":
        rows = sorted(rows, key=lambda row: row[1][0].split("(", 1)[0])
    for index, (utterance, ground_truth) in enumerate(rows):
        domain = ground_truth[0].split("(", 1)[0]
        template_id = None
        if family == "multi":
            matching = [
                item
                for item in templates[domain]
                if "\n".join(
                    f"{turn.get('role', 'User')}: {turn.get('message', '')}"
                    for turn in item["query"]
                )
                == utterance
            ]
            template_id = matching[0]["query_id"] if matching else None
        result.append(
            {
                "example_id": str(example["example_id"]),
                "difficulty": difficulty,
                "source_ordinal": index,
                "domain": domain,
                "utterance": utterance,
                "ground_truth": (
                    sorted(ground_truth) if difficulty == "medium" else ground_truth
                ),
                "template_id": template_id,
                "set_derived_slots": (
                    sorted(
                        {
                            evidence["slot"]
                            for pref in example["api_calls_pref"]
                            for evidence in pref["evidence"]
                            if evidence["domain"] == domain
                        }
                    )
                    if difficulty == "medium"
                    else []
                ),
            }
        )
    return result


def source_request_matrix(
    ns: dict[str, Any],
    api_only_ns: dict[str, Any],
    prompt_template: str,
    derivation: list[dict[str, Any]],
    request: dict[str, Any],
    family: str,
) -> list[dict[str, Any]]:
    records = []
    for case in derivation:
        example = example_for(request, case["example_id"])
        memory = memory_for(request, case["example_id"])
        for context in CONTEXTS:
            builder = (
                api_only_ns["build_memory_input_prompt"]
                if context == "api_only"
                else ns["build_memory_input_prompt"]
            )
            source_context = "api-only" if context == "api_only" else context
            observation = builder(
                example,
                memory,
                case["utterance"],
                "SCHEMA={preference_schema}\n\nMEM={retrieved_memories}\n\nDIALOGUE={dialogue_history}\n\nCURRENT={user_utterance}",
                source_context,
                request["tool_schema"],
            )
            prompt = builder(
                example,
                memory,
                case["utterance"],
                prompt_template,
                source_context,
                request["tool_schema"],
            )
            wire = capture_source_wire(ns["call_llm_api_async"], prompt, "ok")
            records.append(
                {
                    "example_id": case["example_id"],
                    "difficulty": case["difficulty"],
                    "source_ordinal": case["source_ordinal"],
                    "context": context,
                    "components": parse_observation_prompt(observation),
                    "wire": wire,
                }
            )
    return records


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.requests: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.requests.append(copy.deepcopy(kwargs))
        message = SimpleNamespace(content=self.content, reasoning_content="")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=None
        )


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def capture_source_wire(function: Any, prompt: str, content: str) -> dict[str, Any]:
    client = _FakeClient(content)
    result = asyncio.run(function(prompt, "probe", client, None, None))
    if result.get("error"):
        raise RuntimeError(f"legacy provider probe failed: {result['error']}")
    if len(client.completions.requests) != 1:
        raise RuntimeError("legacy provider probe did not make exactly one call")
    body = client.completions.requests[0]
    return {
        "model": body.get("model"),
        "messages": body.get("messages"),
        "prompt": prompt,
        "temperature": {
            "present": "temperature" in body,
            "value": body.get("temperature"),
        },
        "json_intent": "response_format" in body,
    }


def parse_observation_prompt(prompt: str) -> dict[str, Any]:
    schema, remainder = prompt.split("\n\nMEM=", 1)
    memory, remainder = remainder.split("\n\nDIALOGUE=", 1)
    dialogue, current = remainder.split("\n\nCURRENT=", 1)
    preference = None
    api_history = None
    if memory.startswith("[Implicit Preferences]:\n"):
        preference = memory.removeprefix("[Implicit Preferences]:\n")
        if "\n\n[Past API History]:\n" in preference:
            preference, api_history = preference.split(
                "\n\n[Past API History]:\n", 1
            )
    elif memory.startswith("[Past API History]:\n"):
        api_history = memory.removeprefix("[Past API History]:\n")
    return {
        "tool_schema": schema.removeprefix("SCHEMA="),
        "preference": preference,
        "api_history": api_history,
        "dialogue": None if dialogue == "None" else dialogue,
        "current": current,
    }


class _Progress:
    def update(self, _count: int) -> None:
        return None


def source_projection(
    ns: dict[str, Any],
    prompt_template: str,
    derivation: list[dict[str, Any]],
    request: dict[str, Any],
    temp_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    logs = []
    results = []
    for index, case in enumerate(derivation):
        example = example_for(request, case["example_id"])
        memory = memory_for(request, case["example_id"])
        output = json.dumps(
            {
                "case": f"{case['example_id']}:{request['family']}:{case['difficulty']}:{case['source_ordinal']}"
            },
            ensure_ascii=False,
        )
        client = _FakeClient(output)
        log_path = temp_root / f"projection-{index}.jsonl"

        async def run() -> Any:
            return await ns["process_single_item"](
                original_ex=example,
                user_memory=memory,
                utterance=case["utterance"],
                ground_truth=case["ground_truth"],
                sub_idx=case["source_ordinal"],
                model_name="probe",
                prompt_template=prompt_template,
                context_type="memory_api",
                openai_client=client,
                tools_schema=request["tool_schema"],
                log_path=str(log_path),
                semaphore=asyncio.Semaphore(1),
                file_lock=asyncio.Lock(),
                pbar=_Progress(),
                reasoning_effort=None,
            )

        result = asyncio.run(run())
        log = json.loads(log_path.read_text(encoding="utf-8"))
        log.pop("timestamp", None)
        logs.append(log)
        results.append(
            {
                key: result[key]
                for key in (
                    "example_id_sub",
                    "test_utterance",
                    "reference_ground_truth",
                    "llm_output",
                    "reasoning_content",
                    "token_counts",
                    "reasoning_token_count",
                )
            }
        )
    return {"legacy_logs": logs, "legacy_results": results}


def source_hard_null(
    ns: dict[str, Any],
    slots: Path,
    groups: Path,
    query_source: object,
    request: dict[str, Any],
    family: str,
) -> dict[str, Any]:
    evidence = {}
    for name, rules in request["hard_null_rules"].items():
        groups.write_text(
            json.dumps({"comfort": {"rules": rules}}), encoding="utf-8"
        )
        rows = ns["assign_user_utterances"](
            str(slots),
            request["example"],
            query_source,
            "hard",
            str(groups),
        )
        evidence[name] = [
            {
                "example_id": str(request["example"]["example_id"]),
                "domain": ground_truth[0].split("(", 1)[0],
                "utterance": utterance,
                "ground_truth": ground_truth,
                "template_id": template_id_for(request, family, utterance, ground_truth),
            }
            for utterance, ground_truth in sorted(
                rows, key=lambda row: row[1][0].split("(", 1)[0]
            )
        ]
    return evidence


def template_id_for(
    request: dict[str, Any], family: str, utterance: str, ground_truth: list[str]
) -> str | None:
    if family == "single":
        return None
    raw = request["multiturn_data"]
    if isinstance(raw, list):
        candidates = raw
    else:
        domain = ground_truth[0].split("(", 1)[0]
        candidates = raw[domain]
    for item in candidates:
        rendered = "\n".join(
            f"{turn.get('role', 'User')}: {turn.get('message', '')}"
            for turn in item["query"]
        )
        if rendered == utterance:
            return item["query_id"]
    return None


def example_for(request: dict[str, Any], example_id: str) -> dict[str, Any]:
    return next(
        item for item in request["examples"] if str(item["example_id"]) == example_id
    )


def memory_for(request: dict[str, Any], example_id: str) -> dict[str, Any]:
    return next(
        item for item in request["memories"] if str(item["example_id"]) == example_id
    )


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
    root_value = os.environ.get("OURS_MEMORY_REFERENCE_ROOT")
    if not root_value:
        raise RuntimeError("OURS_MEMORY_REFERENCE_ROOT is required")
    root = Path(root_value).resolve(strict=True)
    request = json.loads(sys.stdin.read())
    install_tripwires()
    family = request["family"]
    output = (
        step1_case(root, request)
        if family == "step1"
        else step2_case(root, request, family)
    )
    sys.stdout.write(
        json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
