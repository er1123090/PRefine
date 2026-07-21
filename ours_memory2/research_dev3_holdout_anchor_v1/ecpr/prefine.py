"""Deterministic PReFine state machine shared by live execution and replay."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from .io import canonical_json, sha256_text
from .latent_firewall import (
    GENERATED_ATTESTATION_SOURCE,
    LATENT_VERIFICATION_ATTESTATION_KIND,
    LatentVerificationAttestation,
)
from .prompts import LATENT_INITIAL, LATENT_REFINE, LATENT_SYSTEM, LATENT_VERIFY


LATENT_GENERATION_SCHEMA_SHA256 = sha256_text(
    canonical_json({"kind": "prefine_latent_generation", "schema_version": 1})
)
LATENT_VERIFICATION_SCHEMA_SHA256 = sha256_text(
    canonical_json({"kind": "prefine_latent_verification", "schema_version": 1})
)


def parse_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start : end + 1])
                return value if isinstance(value, dict) else {}
            except json.JSONDecodeError:
                return {}
        return {}


def history_strings(history: dict[str, Any], through: int) -> tuple[str, str]:
    dialogue_parts: list[str] = []
    call_parts: list[str] = []
    for session in history.get("sessions", [])[:through]:
        index = session["session_index"]
        turns = "\n".join(
            f"{turn['role']}: {turn['message']}"
            for turn in session.get("dialogue", [])
        )
        dialogue_parts.append(f"[Session {index}]\n{turns}")
        call_parts.extend(
            f"[Session {index}] {call}"
            for call in session.get("api_calls", [])
        )
    return (
        "\n\n".join(dialogue_parts),
        "\n".join(call_parts) or "No API calls recorded.",
    )


@dataclass(frozen=True)
class LatentBuildResult:
    latent_abstraction: dict[str, Any]
    verification_attestation: LatentVerificationAttestation
    semantic_trace_sha256: str


@dataclass(frozen=True)
class PrefineCall:
    """One deterministic method-level request shared by live and replay."""

    messages: tuple[tuple[str, str], ...]
    seed: int
    max_tokens: int
    temperature: float
    json_object: bool
    phase: str
    call_key: str
    schema_sha256: str
    session_index: int
    attempt_index: int
    role: str

    @classmethod
    def from_messages(
        cls,
        messages: list[dict[str, str]],
        **values: Any,
    ) -> "PrefineCall":
        normalized = tuple(
            (str(message["role"]), str(message["content"]))
            for message in messages
        )
        return cls(messages=normalized, **values)

    def message_list(self) -> list[dict[str, str]]:
        return [
            {"role": role, "content": content}
            for role, content in self.messages
        ]

    def provider_kwargs(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "json_object": self.json_object,
            "phase": self.phase,
            "call_key": self.call_key,
            "schema_sha256": self.schema_sha256,
        }

    def call_sha256(self) -> str:
        return sha256_text(
            canonical_json(
                {
                    "phase": self.phase,
                    "call_key": self.call_key,
                    "messages": self.message_list(),
                    "seed": self.seed,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "json_object": self.json_object,
                    "schema_sha256": self.schema_sha256,
                }
            )
        )


class CompletionLike(Protocol):
    content: str
    usage: dict[str, int]
    call_id: str | None
    request_sha256: str | None


PrefineCallback = Callable[[PrefineCall], CompletionLike]


def run_prefine_state_machine(
    history: dict[str, Any],
    seed: int,
    callback: PrefineCallback,
    max_attempts: int = 10,
    *,
    example_id: str = "anonymous",
    journal_attempt_id: str | None = None,
) -> LatentBuildResult:
    """Execute the sole live/replay PReFine transition graph."""
    if type(seed) is not int or type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("PReFine seed/attempt budget is invalid")
    previous: dict[str, Any] = {}
    final_session_index: int | None = None
    final_attempt_index: int | None = None
    final_draft: dict[str, Any] | None = None
    final_verdict: dict[str, Any] | None = None
    final_verifier_response: str | None = None
    final_verifier_call: PrefineCall | None = None
    final_verifier_call_id: str | None = None
    final_verifier_request_sha256: str | None = None
    semantic_trace: list[dict[str, Any]] = []
    for through in range(1, len(history.get("sessions", [])) + 1):
        final_session_index = through
        dialogue, calls = history_strings(history, through)
        feedback = ""
        draft: dict[str, Any] = previous
        for attempt in range(max_attempts):
            task = LATENT_INITIAL if attempt == 0 else LATENT_REFINE.format(
                draft=canonical_json(draft), feedback=feedback
            )
            generation_messages = [
                {
                    "role": "system",
                    "content": LATENT_SYSTEM.format(
                        previous=canonical_json(previous) if previous else "None",
                        dialogue=dialogue,
                        api_calls=calls,
                    ),
                },
                {"role": "user", "content": task},
            ]
            generation_call = PrefineCall.from_messages(
                generation_messages,
                seed=seed + through * 100 + attempt * 2,
                max_tokens=2048,
                temperature=0.4,
                json_object=True,
                phase="memory_generation",
                call_key=f"{example_id}:session:{through}:attempt:{attempt}:draft",
                schema_sha256=LATENT_GENERATION_SCHEMA_SHA256,
                session_index=through,
                attempt_index=attempt,
                role="draft",
            )
            generated = callback(generation_call)
            draft = parse_json_object(generated.content)
            semantic_trace.append(
                {
                    "call_sha256": generation_call.call_sha256(),
                    "call_id": generated.call_id,
                    "request_sha256": generated.request_sha256,
                    "response_sha256": sha256_text(generated.content),
                    "parsed_sha256": sha256_text(canonical_json(draft)),
                }
            )
            verifier_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Preference Verification Module. Output JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": LATENT_VERIFY.format(
                        dialogue=dialogue,
                        api_calls=calls,
                        candidate=canonical_json(draft),
                    ),
                },
            ]
            verifier_call = PrefineCall.from_messages(
                verifier_messages,
                seed=seed + through * 100 + attempt * 2 + 1,
                max_tokens=512,
                temperature=0.0,
                json_object=True,
                phase="memory_generation",
                call_key=f"{example_id}:session:{through}:attempt:{attempt}:verify",
                schema_sha256=LATENT_VERIFICATION_SCHEMA_SHA256,
                session_index=through,
                attempt_index=attempt,
                role="verify",
            )
            verified = callback(verifier_call)
            verdict = parse_json_object(verified.content)
            semantic_trace.append(
                {
                    "call_sha256": verifier_call.call_sha256(),
                    "call_id": verified.call_id,
                    "request_sha256": verified.request_sha256,
                    "response_sha256": sha256_text(verified.content),
                    "parsed_sha256": sha256_text(canonical_json(verdict)),
                }
            )
            final_attempt_index = attempt
            final_draft = draft
            final_verdict = verdict
            final_verifier_response = verified.content
            final_verifier_call = verifier_call
            final_verifier_call_id = verified.call_id
            final_verifier_request_sha256 = verified.request_sha256
            if verdict.get("valid") is True:
                break
            feedback = str(verdict.get("feedback", "verification failed"))
        previous = draft or {"implicit_pref": "insufficient evidence"}

    latent = previous or {"implicit_pref": "insufficient evidence"}
    implicit_pref = (
        final_draft.get("implicit_pref")
        if isinstance(final_draft, dict)
        and isinstance(final_draft.get("implicit_pref"), str)
        else None
    )
    exact_valid = bool(
        isinstance(final_verdict, dict)
        and final_verdict.get("valid") is True
    )
    draft_sha256 = (
        sha256_text(canonical_json(final_draft))
        if final_draft is not None
        else None
    )
    memory_sha256 = sha256_text(canonical_json(latent))
    if final_session_index is None:
        reason_code = "NO_FINAL_SESSION"
    elif final_attempt_index is None:
        reason_code = "RETRY_EXHAUSTED"
    elif exact_valid and (
        not isinstance(journal_attempt_id, str)
        or not isinstance(final_verifier_call_id, str)
        or not isinstance(final_verifier_request_sha256, str)
    ):
        reason_code = "UNSEALED_VERIFIER"
    elif exact_valid and implicit_pref is None:
        reason_code = "VERIFIED_DRAFT_MALFORMED"
    elif exact_valid and draft_sha256 != memory_sha256:
        reason_code = "VERIFIED_DRAFT_NOT_FINAL_MEMORY"
    elif exact_valid:
        reason_code = "VALID_FINAL_VERDICT"
    elif (
        not isinstance(final_verdict, dict)
        or type(final_verdict.get("valid")) is not bool
    ):
        reason_code = "MALFORMED_FINAL_VERDICT"
    else:
        reason_code = "RETRY_EXHAUSTED"
    status = "VALID" if reason_code == "VALID_FINAL_VERDICT" else "INVALID"
    verifier_messages = (
        final_verifier_call.message_list()
        if final_verifier_call is not None
        else None
    )
    attestation = LatentVerificationAttestation(
        schema_version=1,
        kind=LATENT_VERIFICATION_ATTESTATION_KIND,
        source=GENERATED_ATTESTATION_SOURCE,
        status=status,
        reason_code=reason_code,
        final_session_index=final_session_index,
        final_attempt_index=final_attempt_index,
        verdict_valid_is_exactly_true=exact_valid,
        journal_attempt_id=journal_attempt_id,
        verifier_call_id=final_verifier_call_id,
        final_draft_sha256=draft_sha256,
        implicit_pref_sha256=(
            sha256_text(implicit_pref) if implicit_pref is not None else None
        ),
        final_memory_latent_sha256=memory_sha256,
        verifier_response_sha256=(
            sha256_text(final_verifier_response)
            if final_verifier_response is not None
            else None
        ),
        verifier_request_sha256=final_verifier_request_sha256,
        verifier_call_sha256=(
            final_verifier_call.call_sha256()
            if final_verifier_call is not None
            else None
        ),
        verifier_call_key_sha256=(
            sha256_text(final_verifier_call.call_key)
            if final_verifier_call is not None
            else None
        ),
        verifier_schema_sha256=(
            final_verifier_call.schema_sha256
            if final_verifier_call is not None
            else None
        ),
        verifier_prompt_sha256=(
            sha256_text(verifier_messages[1]["content"])
            if verifier_messages is not None
            else None
        ),
        verifier_messages_sha256=(
            sha256_text(canonical_json(verifier_messages))
            if verifier_messages is not None
            else None
        ),
    )
    return LatentBuildResult(
        latent,
        attestation,
        sha256_text(canonical_json(semantic_trace)),
    )

