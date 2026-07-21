"""Hash-only semantic replay for sealed PReFine provider journals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .io import canonical_json, sha256_text
from .latent_firewall import (
    JournalReplayCapability,
    parse_replay_capability,
    validate_latent_attestation,
)
from .prefine import PrefineCall, run_prefine_state_machine
from .request_contract import CallSpec


@dataclass(frozen=True)
class _ReplayCompletion:
    content: str
    usage: dict[str, int]
    call_id: str
    request_sha256: str


@dataclass
class _ReplayCursor:
    records: list[dict[str, Any]]
    endpoint: str
    model: str
    timeout_seconds: float
    index: int = 0

    def callback(self, call: PrefineCall) -> _ReplayCompletion:
        if self.index + 2 > len(self.records):
            raise ValueError("PReFine replay journal ended before the state machine")
        intent = self.records[self.index]
        terminal = self.records[self.index + 1]
        self.index += 2
        if (
            intent.get("kind") != "provider_call_intent"
            or terminal.get("kind") != "provider_call_terminal"
        ):
            raise ValueError("PReFine replay call order is not intent/terminal")
        if (
            intent.get("phase") != call.phase
            or terminal.get("phase") != call.phase
            or intent.get("call_key_sha256") != sha256_text(call.call_key)
        ):
            raise ValueError("PReFine replay phase or call-key mismatch")
        if (
            terminal.get("call_id") != intent.get("call_id")
            or terminal.get("intent_record_sha256") != intent.get("record_sha256")
        ):
            raise ValueError("PReFine replay terminal does not bind its intent")
        expected = CallSpec.from_messages(
            endpoint=self.endpoint,
            model=self.model,
            messages=call.message_list(),
            seed=call.seed,
            temperature=call.temperature,
            max_tokens=call.max_tokens,
            json_object=call.json_object,
            schema_sha256=call.schema_sha256,
            timeout_seconds=self.timeout_seconds,
        )
        expected_request = expected.audit_request()
        if intent.get("request") != expected_request:
            raise ValueError("PReFine replay request differs from canonical CallSpec")
        if terminal.get("status") != "ok":
            raise ValueError("PReFine replay requires a successful terminal")
        response_text = terminal.get("response_text")
        if (
            not isinstance(response_text, str)
            or terminal.get("response_sha256") != sha256_text(response_text)
        ):
            raise ValueError("PReFine replay terminal content mismatch")
        return _ReplayCompletion(
            content=response_text,
            usage=dict(terminal.get("usage", {})),
            call_id=str(intent["call_id"]),
            request_sha256=expected.audit_request_sha256(),
        )


def replay_memory_generation(
    *,
    histories: Mapping[str, dict[str, Any]],
    memory_rows: Mapping[str, dict[str, Any]],
    latent_rows: Mapping[str, dict[str, Any]],
    records: list[dict[str, Any]],
    attempt_id: str,
    endpoint: str,
    model: str,
    timeout_seconds: float,
    base_seed: int,
    max_attempts: int = 10,
) -> tuple[dict[str, Any], dict[str, JournalReplayCapability]]:
    """Replay every history row and issue capabilities only for exact valid rows."""
    if set(histories) != set(memory_rows) or set(histories) != set(latent_rows):
        raise ValueError("PReFine replay row universes differ")
    cursor = _ReplayCursor(
        records=records,
        endpoint=endpoint,
        model=model,
        timeout_seconds=timeout_seconds,
    )
    capabilities: dict[str, JournalReplayCapability] = {}
    row_artifacts: list[dict[str, Any]] = []
    for example_id, history in histories.items():
        memory = memory_rows[example_id]
        latent_projection = latent_rows[example_id]
        seed = base_seed + int(sha256_text(example_id)[:8], 16)
        start = cursor.index
        result = run_prefine_state_machine(
            history,
            seed,
            cursor.callback,
            max_attempts,
            example_id=example_id,
            journal_attempt_id=attempt_id,
        )
        end = cursor.index
        if result.latent_abstraction != memory.get("latent_abstraction"):
            raise ValueError("PReFine replay final latent differs from ECPR memory")
        if result.verification_attestation.as_dict() != memory.get(
            "latent_attestation"
        ):
            raise ValueError("PReFine replay attestation differs from ECPR memory")
        expected_projection = {
            "example_id": example_id,
            "latent_abstraction": result.latent_abstraction,
            "method": "prefine_v1",
        }
        if latent_projection != expected_projection:
            raise ValueError("PReFine replay differs from PREFINE projection")
        attestation = validate_latent_attestation(
            result.latent_abstraction,
            result.verification_attestation.as_dict(),
        )
        record_trace = [
            {
                "sequence": record["sequence"],
                "record_sha256": record["record_sha256"],
                "call_id": record["call_id"],
                "kind": record["kind"],
            }
            for record in records[start:end]
        ]
        trace_sha256 = sha256_text(
            canonical_json(
                {
                    "semantic_trace_sha256": result.semantic_trace_sha256,
                    "record_trace": record_trace,
                }
            )
        )
        capability: JournalReplayCapability | None = None
        if attestation.status == "VALID":
            required = (
                attestation.verifier_call_id,
                attestation.verifier_request_sha256,
                attestation.verifier_response_sha256,
            )
            if not all(isinstance(value, str) for value in required):
                raise ValueError("valid replay attestation lacks verifier binding")
            capability = parse_replay_capability(
                {
                    "schema_version": 1,
                    "kind": "semantic_prefine_journal_replay_capability_v1",
                    "attempt_id": attempt_id,
                    "verifier_call_id": str(attestation.verifier_call_id),
                    "verifier_request_sha256": str(
                        attestation.verifier_request_sha256
                    ),
                    "verifier_response_sha256": str(
                        attestation.verifier_response_sha256
                    ),
                    "attestation_sha256": sha256_text(
                        canonical_json(attestation.as_dict())
                    ),
                    "latent_sha256": sha256_text(
                        canonical_json(result.latent_abstraction)
                    ),
                    "trace_sha256": trace_sha256,
                }
            )
            capabilities[example_id] = capability
        row_artifacts.append(
            {
                "example_id_sha256": sha256_text(example_id),
                "memory_row_sha256": sha256_text(canonical_json(memory)),
                "prefine_row_sha256": sha256_text(
                    canonical_json(latent_projection)
                ),
                "semantic_trace_sha256": result.semantic_trace_sha256,
                "journal_trace_sha256": trace_sha256,
                "capability": (
                    capability.artifact() if capability is not None else None
                ),
            }
        )
    if cursor.index != len(records):
        raise ValueError("PReFine replay left unconsumed journal records")
    artifact = {
        "schema_version": 1,
        "kind": "semantic_prefine_journal_replay_artifact_v1",
        "attempt_id": attempt_id,
        "row_count": len(row_artifacts),
        "record_count": len(records),
        "rows": row_artifacts,
        "rows_sha256": sha256_text(canonical_json(row_artifacts)),
        "capability_count": len(capabilities),
        "capabilities_sha256": sha256_text(
            canonical_json(
                [
                    {
                        "example_id_sha256": sha256_text(example_id),
                        "capability": capabilities[example_id].artifact(),
                    }
                    for example_id in sorted(capabilities)
                ]
            )
        ),
    }
    return artifact, capabilities

