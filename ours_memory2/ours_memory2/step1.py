"""Ordered-session preference-memory construction and lossless projections."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    ExampleInput,
    InputContractError,
    ProviderPurpose,
    ProviderRequest,
    SessionInput,
    Step1OutputNames,
)
from .jsonl import append_jsonl_streams, validate_output_root
from .prompts import (
    build_generation_messages,
    build_refinement_messages,
    build_verifier_messages,
)
from .providers import Provider


MAX_GENERATION_SLOTS = 10
GENERATION_PARSE_FEEDBACK = "Failed to parse generation output"
VERIFIER_PARSE_FEEDBACK = "Failed to parse verifier output"


class Step1ParseError(ValueError):
    """A provider response cannot satisfy a Step 1 JSON contract."""

    def __init__(self, message: str, *, response_kind: str) -> None:
        self.response_kind = response_kind
        super().__init__(message)


@dataclass(frozen=True)
class AttemptRecord:
    step: int
    draft_preference: Mapping[str, Any]
    is_valid: bool
    verifier_feedback: str
    verifier_input: str
    verifier_output: Mapping[str, Any]
    generation_purpose: str
    refinement_parent: Mapping[str, Any] | None = None
    refinement_feedback: str | None = None

    def as_evolution_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "draft_preference": _copy_json(self.draft_preference),
            "is_valid": self.is_valid,
            "verifier_feedback": self.verifier_feedback,
            "verifier_input": self.verifier_input,
            "verifier_output": _copy_json(self.verifier_output),
            "generation_purpose": self.generation_purpose,
            "refinement_parent": _copy_json(self.refinement_parent),
            "refinement_feedback": self.refinement_feedback,
        }


@dataclass(frozen=True)
class MemoryState:
    implicit_preference: Mapping[str, Any] = field(default_factory=dict)
    accumulated_dialogue: str = ""
    accumulated_api_calls: tuple[str, ...] = ()
    session_count: int = 0
    evolution_history: tuple[Mapping[str, Any], ...] = ()


@dataclass(frozen=True)
class ExampleMemoryRun:
    example_id: str
    state: MemoryState
    final_rows: tuple[Mapping[str, Any], ...]
    draft_rows: tuple[Mapping[str, Any], ...]
    verifier_rows: tuple[Mapping[str, Any], ...]

    @property
    def final_row(self) -> Mapping[str, Any]:
        return self.final_rows[0]


def update_session(
    state: MemoryState, session: SessionInput, provider: Provider
) -> MemoryState:
    """Process one ordered session with exactly ten available generation slots."""

    session_index = state.session_count + 1
    session_dialogue = format_dialogue(session)
    full_dialogue = (
        state.accumulated_dialogue
        + f"\n=== Session {session_index} ===\n"
        + session_dialogue
    )
    session_api_calls = tuple(
        f"[Session {session_index}] {api_call}" for api_call in session.api_call
    )
    full_api_calls = state.accumulated_api_calls + session_api_calls

    incoming_preference = _copy_json(state.implicit_preference)
    candidate = _copy_json(state.implicit_preference)
    has_prior_parsed_draft = False
    feedback = ""
    attempts: list[AttemptRecord] = []
    diagnostics: list[dict[str, Any]] = []
    termination_reason = "max_attempts_generation_error"
    final_feedback = ""

    for step in range(1, MAX_GENERATION_SLOTS + 1):
        use_refinement = has_prior_parsed_draft and bool(feedback)
        purpose = (
            ProviderPurpose.REFINE
            if use_refinement
            else ProviderPurpose.GENERATE
        )
        parent = _copy_json(candidate) if use_refinement else None
        refinement_feedback = feedback if use_refinement else None
        if purpose is ProviderPurpose.GENERATE:
            messages = build_generation_messages(
                previous_preference=incoming_preference,
                dialogue=full_dialogue,
                api_history=full_api_calls,
            )
        else:
            messages = build_refinement_messages(
                previous_preference=incoming_preference,
                dialogue=full_dialogue,
                api_history=full_api_calls,
                previous_draft=parent,
                feedback=feedback,
            )
        generation_request = _provider_request(
            provider=provider,
            purpose=purpose,
            messages=messages,
            temperature=0.4,
        )
        generation_response = provider.complete(generation_request)
        try:
            draft = parse_generated_preference(generation_response.text)
        except Step1ParseError as exc:
            diagnostics.append(
                {
                    "session_index": session_index,
                    "step": step,
                    "kind": "generation_parse_error",
                    "message": str(exc),
                    "feedback": GENERATION_PARSE_FEEDBACK,
                }
            )
            if step == MAX_GENERATION_SLOTS:
                termination_reason = "max_attempts_generation_error"
            continue

        candidate = draft
        has_prior_parsed_draft = True
        verifier_messages = build_verifier_messages(
            candidate=candidate,
            dialogue=full_dialogue,
            api_history=full_api_calls,
        )
        verifier_request = _provider_request(
            provider=provider,
            purpose=ProviderPurpose.VERIFY,
            messages=verifier_messages,
            temperature=0.0,
        )
        verifier_response = provider.complete(verifier_request)
        verifier_input = verifier_messages[-1].content
        try:
            is_valid, new_feedback, verifier_output = parse_verifier_decision(
                verifier_response.text
            )
        except Step1ParseError as exc:
            is_valid = False
            new_feedback = VERIFIER_PARSE_FEEDBACK
            verifier_output = {}
            diagnostics.append(
                {
                    "session_index": session_index,
                    "step": step,
                    "kind": "verifier_parse_error",
                    "message": str(exc),
                }
            )

        attempt = AttemptRecord(
            step=step,
            draft_preference=_copy_json(candidate),
            is_valid=is_valid,
            verifier_feedback=new_feedback,
            verifier_input=verifier_input,
            verifier_output=_copy_json(verifier_output),
            generation_purpose=purpose.value,
            refinement_parent=parent,
            refinement_feedback=refinement_feedback,
        )
        attempts.append(attempt)
        feedback = new_feedback
        final_feedback = new_feedback
        if is_valid:
            termination_reason = "valid"
            break
        termination_reason = "max_attempts_invalid"

    evolution_entry = {
        "session_index": session_index,
        "refinement_process": [attempt.as_evolution_dict() for attempt in attempts],
        "final_preference_at_session": _copy_json(candidate),
        "termination_reason": termination_reason,
        "generation_slots_used": MAX_GENERATION_SLOTS
        if termination_reason.startswith("max_attempts")
        else attempts[-1].step,
        "verified_candidates": len(attempts),
        "final_verifier_feedback": final_feedback,
        "diagnostics": diagnostics,
    }
    return MemoryState(
        implicit_preference=_copy_json(candidate),
        accumulated_dialogue=full_dialogue,
        accumulated_api_calls=full_api_calls,
        session_count=session_index,
        evolution_history=state.evolution_history + (evolution_entry,),
    )


def build_example_memory(
    example: ExampleInput, provider: Provider
) -> ExampleMemoryRun:
    """Build a complete example in memory before exposing any output stream."""

    state = MemoryState()
    for session in example.sessions:
        state = update_session(state, session, provider)
    run = _project_example(example, state)
    _validate_run(run)
    return run


def build_and_write_examples(
    examples: Iterable[ExampleInput], provider: Provider, output_root: str | Path
) -> tuple[ExampleMemoryRun, ...]:
    """Build then append each complete example; provider failure writes no current row."""

    root = validate_output_root(output_root)
    runs: list[ExampleMemoryRun] = []
    for example in examples:
        run = build_example_memory(example, provider)
        write_example_run(run, root)
        runs.append(run)
    return tuple(runs)


def write_example_run(run: ExampleMemoryRun, output_root: str | Path) -> None:
    """Append one already-complete, correlated run to the three JSONL streams."""

    _validate_run(run)
    root = validate_output_root(output_root)
    names = Step1OutputNames()
    append_jsonl_streams(
        root,
        {
            names.final: run.final_rows,
            names.drafts: run.draft_rows,
            names.verifiers: run.verifier_rows,
        },
    )


def format_dialogue(session: SessionInput) -> str:
    """Format turns in their supplied order."""

    return "\n".join(f"{turn.role}: {turn.message}" for turn in session.dialogue)


def parse_generated_preference(text: str) -> dict[str, Any]:
    value = _parse_json_object(text, response_kind="generation")
    if not value:
        raise Step1ParseError(
            "generation response must be a nonempty JSON object",
            response_kind="generation",
        )
    return value


def parse_verifier_decision(
    text: str,
) -> tuple[bool, str, dict[str, Any]]:
    value = _parse_json_object(text, response_kind="verifier")
    if type(value.get("valid")) is not bool:
        raise Step1ParseError(
            "verifier response requires a boolean valid field",
            response_kind="verifier",
        )
    feedback = value.get("feedback", "")
    if not isinstance(feedback, str):
        raise Step1ParseError(
            "verifier feedback must be a string",
            response_kind="verifier",
        )
    return value["valid"], feedback, value


def _project_example(example: ExampleInput, state: MemoryState) -> ExampleMemoryRun:
    evolution = [_copy_json(item) for item in state.evolution_history]
    final_row = {
        "example_id": example.example_id,
        "final_implicit_preference": _copy_json(state.implicit_preference),
        "final_accumulated_api_calls": list(state.accumulated_api_calls),
        "final_accumulated_dialogue": state.accumulated_dialogue,
        "total_sessions_processed": state.session_count,
        "preference_evolution_history": evolution,
        "metadata": _copy_json(dict(example.metadata)),
    }
    drafts: list[dict[str, Any]] = []
    verifiers: list[dict[str, Any]] = []
    for session_entry in evolution:
        session_index = session_entry["session_index"]
        for attempt in session_entry["refinement_process"]:
            key = {
                "example_id": example.example_id,
                "session_index": session_index,
                "step": attempt["step"],
            }
            drafts.append(
                {
                    **key,
                    "draft_preference": _copy_json(attempt["draft_preference"]),
                    "generation_purpose": attempt["generation_purpose"],
                    "refinement_parent": _copy_json(attempt["refinement_parent"]),
                    "refinement_feedback": attempt["refinement_feedback"],
                }
            )
            verifiers.append(
                {
                    **key,
                    "is_valid": attempt["is_valid"],
                    "verifier_feedback": attempt["verifier_feedback"],
                    "verifier_input": attempt["verifier_input"],
                    "verifier_output": _copy_json(attempt["verifier_output"]),
                }
            )
    return ExampleMemoryRun(
        example_id=example.example_id,
        state=state,
        final_rows=(final_row,),
        draft_rows=tuple(drafts),
        verifier_rows=tuple(verifiers),
    )


def _validate_run(run: ExampleMemoryRun) -> None:
    if len(run.final_rows) != 1:
        raise InputContractError("must contain exactly one row", path="step1.final_rows")
    draft_keys = [_row_key(row) for row in run.draft_rows]
    verifier_keys = [_row_key(row) for row in run.verifier_rows]
    if draft_keys != verifier_keys:
        raise InputContractError(
            "draft and verifier streams are not correlated", path="step1.streams"
        )
    evolution_count = sum(
        len(item["refinement_process"])
        for item in run.final_row["preference_evolution_history"]
    )
    if evolution_count != len(run.draft_rows):
        raise InputContractError(
            "evolution history is not lossless", path="step1.streams"
        )


def _provider_request(
    *,
    provider: Provider,
    purpose: ProviderPurpose,
    messages: Sequence[Any],
    temperature: float,
) -> ProviderRequest:
    model = getattr(provider, "model", None)
    if not isinstance(model, str) or not model.strip():
        raise InputContractError("must be nonempty", path="provider.model")
    return ProviderRequest(
        purpose=purpose,
        model=model,
        messages=tuple(messages),
        prompt=messages[-1].content,
        temperature=temperature,
        json_intent=True,
    )


def _parse_json_object(text: str, *, response_kind: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise Step1ParseError(
            f"{response_kind} response is empty", response_kind=response_kind
        )
    candidates = [text.strip()]
    without_fence = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    if without_fence != candidates[0]:
        candidates.append(without_fence)
    start = without_fence.find("{")
    end = without_fence.rfind("}")
    if start >= 0 and end > start:
        candidates.append(without_fence[start : end + 1])
    for candidate in candidates:
        for normalized in (candidate, re.sub(r",\s*([}\]])", r"\1", candidate)):
            try:
                value = json.loads(normalized)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise Step1ParseError(
        f"{response_kind} response is not a JSON object",
        response_kind=response_kind,
    )


def _copy_json(value: Any) -> Any:
    if value is None:
        return None
    return json.loads(json.dumps(value, ensure_ascii=False))


def _row_key(row: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return row["example_id"], row["session_index"], row["step"]
