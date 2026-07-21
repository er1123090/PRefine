"""PREFINE latent memory plus independently retained typed evidence hypotheses."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .contracts import CandidatePolicy
from .io import canonical_json, iter_jsonl, sha256_text, write_jsonl, write_jsonl_once
from .latent_firewall import (
    ACTIVE_CANDIDATE_REVISION,
    LatentVerificationAttestation,
    unattested_latent,
)
from .parsing import extract_calls, normalize_value
from .prefine import LatentBuildResult, run_prefine_state_machine
from .provider import LoopbackChatClient, ProviderError


def _canonical_value_type(value: Any) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "boolean"
    if type(value) is int:
        return "integer"
    if type(value) is float:
        return "number"
    if type(value) is str:
        return "string"
    if type(value) is list:
        return "array"
    if type(value) is dict:
        return "object"
    return "other"


def _source_literal(value: Any) -> dict[str, Any] | None:
    value_type = _canonical_value_type(value)
    if value_type not in {"null", "boolean", "integer", "number", "string"}:
        return None
    return {"type": value_type, "value": value}


def build_typed_hypotheses(
    history: dict[str, Any], preference_slots: dict[str, list[str]], policy: CandidatePolicy
) -> list[dict[str, Any]]:
    observations: dict[
        tuple[str, str], list[tuple[str, str, str, int, int, str]]
    ] = defaultdict(list)
    allowed = {str(domain): {str(slot) for slot in slots} for domain, slots in preference_slots.items()}
    for session in history.get("sessions", []):
        session_index = int(session["session_index"])
        for call_index, raw_call in enumerate(session.get("api_calls", [])):
            for call in extract_calls(raw_call):
                domain = str(call["name"])
                digest = sha256_text(canonical_json(call))
                for slot, raw_value in call["arguments"].items():
                    slot = str(slot)
                    value = normalize_value(raw_value)
                    value_type = _canonical_value_type(raw_value)
                    source_literal = _source_literal(raw_value)
                    if value and source_literal is not None and slot in allowed.get(domain, set()):
                        observations[(domain, slot)].append(
                            (value, value_type, canonical_json(source_literal), session_index, call_index, digest)
                        )

    hypotheses: list[dict[str, Any]] = []
    for (domain, slot), entries in sorted(observations.items()):
        total = len(entries)
        grouped: dict[tuple[str, str, str], list[tuple[str, str, str, int, int, str]]] = defaultdict(list)
        for entry in entries:
            grouped[(entry[0], entry[1], entry[2])].append(entry)
        for (value, value_type, source_literal), supporters in grouped.items():
            last_seen = max(entry[3] for entry in supporters)
            provenance = [
                {"session_index": entry[3], "call_index": entry[4], "call_digest": entry[5]}
                for entry in supporters
            ]
            support = len(supporters)
            hypotheses.append(
                {
                    "domain": domain,
                    "slot": slot,
                    "value": value,
                    "value_type": value_type,
                    "support": support,
                    "source_literal": json.loads(source_literal),
                    "counterevidence": total - support,
                    "confidence": round(support / total, 12),
                    "last_seen": last_seen,
                    "provenance": provenance,
                }
            )
    hypotheses.sort(
        key=lambda item: (
            -item["confidence"],
            -item["support"],
            -item["last_seen"],
            item["domain"],
            item["slot"],
            item["value"],
            item["value_type"],
            canonical_json(item["source_literal"]),
        )
    )
    return hypotheses[: policy.maximum_stored_hypotheses]


def build_latent_abstraction_result(
    history: dict[str, Any],
    client: LoopbackChatClient,
    seed: int,
    max_attempts: int = 10,
    *,
    example_id: str = "anonymous",
) -> LatentBuildResult:
    journal_attempt_id = getattr(
        getattr(client, "journal", None), "attempt_id", None
    )
    return run_prefine_state_machine(
        history,
        seed,
        lambda call: client.complete(
            call.message_list(), **call.provider_kwargs()
        ),
        max_attempts,
        example_id=example_id,
        journal_attempt_id=journal_attempt_id,
    )


def build_latent_abstraction(
    history: dict[str, Any],
    client: LoopbackChatClient,
    seed: int,
    max_attempts: int = 10,
    *,
    example_id: str = "anonymous",
) -> dict[str, Any]:
    """Backward-compatible latent-only result surface."""
    return build_latent_abstraction_result(
        history,
        client,
        seed,
        max_attempts,
        example_id=example_id,
    ).latent_abstraction


def build_memory_records(
    history_path: str | Path,
    preference_slots: dict[str, list[str]],
    policy: CandidatePolicy,
    *,
    client: LoopbackChatClient | None = None,
    seed: int = 2026071600,
    latent_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    latent_by_id: dict[str, Any] = {}
    if latent_path:
        for row in iter_jsonl(latent_path):
            example_id = str(row.get("example_id", ""))
            if example_id in latent_by_id:
                raise ValueError(f"duplicate latent example_id: {example_id}")
            latent_by_id[example_id] = row.get("latent_abstraction", row.get("final_implicit_preference"))

    records: list[dict[str, Any]] = []
    for history in iter_jsonl(history_path):
        example_id = str(history["example_id"])
        latent = latent_by_id.get(example_id)
        attestation: LatentVerificationAttestation
        if latent is None:
            if client is None:
                latent = {"implicit_pref": "insufficient evidence"}
                attestation = unattested_latent(
                    latent,
                    source="clientless_fallback",
                    reason_code="CLIENTLESS_FALLBACK",
                )
            else:
                try:
                    result = build_latent_abstraction_result(
                        history,
                        client,
                        seed + int(sha256_text(example_id)[:8], 16),
                        example_id=example_id,
                    )
                    latent = result.latent_abstraction
                    attestation = result.verification_attestation
                except ProviderError as exc:
                    raise RuntimeError(
                        "latent build failed for opaque example "
                        + sha256_text(example_id)[:12]
                    ) from exc
        else:
            attestation = unattested_latent(
                latent,
                source="external_latent",
                reason_code="EXTERNAL_UNATTESTED",
            )
        records.append(
            {
                "example_id": example_id,
                "latent_abstraction": latent,
                "latent_attestation": attestation.as_dict(),
                "typed_hypotheses": build_typed_hypotheses(
                    history, preference_slots, policy
                ),
                "method": "ecpr_v1",
                "candidate_revision": ACTIVE_CANDIDATE_REVISION,
            }
        )
    return records


def write_memory_records(
    output: str | Path,
    records: list[dict[str, Any]],
    *,
    commit_once: bool = False,
) -> None:
    writer = write_jsonl_once if commit_once else write_jsonl
    writer(output, records)
