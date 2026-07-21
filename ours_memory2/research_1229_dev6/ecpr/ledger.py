"""Attempt-bound provider journal and forward provenance manifests."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping

from .action_case import build_action_case
from .contracts import CandidatePolicy, PREDICTION_FIELDS
from .integrity import ATTEMPT_LOCK, commit_json_once, validate_final_attempt
from .io import canonical_json, iter_jsonl, load_json, sha256_file, sha256_text, unique_by
from .latent_firewall import (
    JournalReplayCapability,
    VLT2_CANDIDATE_REVISION,
    VLT3_CANDIDATE_REVISION,
    _mint_replay_capability,
    validate_latent_attestation,
    validate_latent_trait_ontology_v2,
    validate_latent_trait_ontology_v3,
    validate_vlt_schema_contract,
)
from .query_gate import validate_ontology
from .replay import replay_memory_generation
from .request_contract import CallSpec


ZERO_SHA256 = "0" * 64
PROVIDER_JOURNAL = Path("artifacts/provider_calls.final.jsonl")
PREFINE_MEMORY = Path("artifacts/memory.prefine.jsonl")
PREFINE_MEMORY_MANIFEST = Path("artifacts/memory.prefine.jsonl.manifest.json")
ECPR_MEMORY = Path("artifacts/memory.ecpr.jsonl")
ECPR_MEMORY_MANIFEST = Path("artifacts/memory.ecpr.jsonl.manifest.json")
BASELINE_PREDICTIONS = Path("artifacts/predictions.baseline.jsonl")
BASELINE_PREDICTIONS_MANIFEST = Path("artifacts/predictions.baseline.jsonl.manifest.json")
CANDIDATE_PREDICTIONS = Path("artifacts/predictions.candidate.jsonl")
CANDIDATE_PREDICTIONS_MANIFEST = Path("artifacts/predictions.candidate.jsonl.manifest.json")
SUMMARY = Path("reports/summary.json")
JOURNAL_PHASES = frozenset(
    {"memory_generation", "action_baseline", "action_candidate"}
)

INTENT_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "attempt_id",
        "sequence",
        "previous_record_sha256",
        "record_sha256",
        "call_id",
        "phase",
        "call_key_sha256",
        "recorded_unix_ns",
        "request",
    }
)
TERMINAL_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "attempt_id",
        "sequence",
        "previous_record_sha256",
        "record_sha256",
        "call_id",
        "phase",
        "intent_record_sha256",
        "recorded_unix_ns",
        "status",
        "usage",
        "response_text",
        "response_sha256",
        "error_text",
        "error_sha256",
    }
)
REQUEST_FIELDS = frozenset(
    {
        "endpoint",
        "model",
        "seed",
        "temperature",
        "max_tokens",
        "n",
        "stream",
        "json_object",
        "schema_sha256",
        "messages_sha256",
        "timeout_seconds",
        "wire_payload_sha256",
    }
)
USAGE_FIELDS = frozenset(
    {"prompt_tokens", "completion_tokens", "total_tokens"}
)


def _digest_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _require_sha256(value: Any, label: str, *, allow_zero: bool = False) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        or (not allow_zero and value == ZERO_SHA256)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _canonical_uuid(value: Any, label: str) -> str:
    try:
        canonical = str(uuid.UUID(str(value)))
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if canonical != value:
        raise ValueError(f"{label} is not canonical")
    return canonical


def journal_record_sha256(record: dict[str, Any]) -> str:
    payload = dict(record)
    payload.pop("record_sha256", None)
    return _digest_json(payload)


def _validate_request(request: Any) -> None:
    if not isinstance(request, dict) or set(request) != REQUEST_FIELDS:
        raise ValueError("provider request field contract violation")
    if not isinstance(request["endpoint"], str) or not request["endpoint"]:
        raise ValueError("provider request endpoint is missing")
    if not isinstance(request["model"], str) or not request["model"]:
        raise ValueError("provider request model is missing")
    if type(request["seed"]) is not int:
        raise ValueError("provider request seed must be an integer")
    if isinstance(request["temperature"], bool) or not isinstance(
        request["temperature"], (int, float)
    ):
        raise ValueError("provider request temperature must be numeric")
    if not isinstance(request["max_tokens"], int) or request["max_tokens"] <= 0:
        raise ValueError("provider request max_tokens must be positive")
    if request["n"] != 1 or request["stream"] is not False:
        raise ValueError("provider request must be a single nonstreaming call")
    if not isinstance(request["json_object"], bool):
        raise ValueError("provider request JSON mode must be boolean")
    _require_sha256(request["schema_sha256"], "provider schema digest")
    _require_sha256(request["messages_sha256"], "provider messages digest")
    _require_sha256(
        request["wire_payload_sha256"], "provider wire-payload digest"
    )
    if (
        isinstance(request["timeout_seconds"], bool)
        or not isinstance(request["timeout_seconds"], (int, float))
        or request["timeout_seconds"] <= 0
    ):
        raise ValueError("provider request timeout must be positive")


def _validate_usage(usage: Any) -> None:
    if not isinstance(usage, dict) or set(usage) != USAGE_FIELDS:
        raise ValueError("provider terminal usage field contract violation")
    if any(not isinstance(value, int) or value < 0 for value in usage.values()):
        raise ValueError("provider terminal usage must be nonnegative integers")


def validate_journal_records(
    records: list[dict[str, Any]],
    attempt_id: str,
    *,
    allow_incomplete: bool = False,
) -> list[dict[str, Any]]:
    """Validate sequence, hash chain, and exactly one terminal per intent."""
    _canonical_uuid(attempt_id, "attempt ID")
    previous = ZERO_SHA256
    intents: dict[str, dict[str, Any]] = {}
    terminals: dict[str, dict[str, Any]] = {}
    for expected_sequence, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError("journal record must be an object")
        kind = record.get("kind")
        expected_fields = (
            INTENT_FIELDS if kind == "provider_call_intent" else TERMINAL_FIELDS
            if kind == "provider_call_terminal"
            else frozenset()
        )
        if not expected_fields or set(record) != expected_fields:
            raise ValueError("provider journal record field contract violation")
        if record.get("schema_version") != 1:
            raise ValueError("provider journal schema mismatch")
        if record.get("attempt_id") != attempt_id:
            raise ValueError("provider journal attempt binding mismatch")
        if record.get("sequence") != expected_sequence:
            raise ValueError("provider journal sequence is not monotonic")
        if record.get("previous_record_sha256") != previous:
            raise ValueError("provider journal previous-hash chain mismatch")
        if record.get("record_sha256") != journal_record_sha256(record):
            raise ValueError("provider journal record digest mismatch")
        _canonical_uuid(record.get("call_id"), "provider call ID")
        if record.get("phase") not in JOURNAL_PHASES:
            raise ValueError("provider journal phase mismatch")
        if not isinstance(record.get("recorded_unix_ns"), int):
            raise ValueError("provider journal timestamp mismatch")
        call_id = str(record["call_id"])
        if kind == "provider_call_intent":
            if call_id in intents:
                raise ValueError("duplicate provider intent")
            _require_sha256(record.get("call_key_sha256"), "provider call-key digest")
            _validate_request(record.get("request"))
            intents[call_id] = record
        else:
            if call_id not in intents:
                raise ValueError("orphan provider terminal")
            if call_id in terminals:
                raise ValueError("duplicate provider terminal")
            intent = intents[call_id]
            if record.get("phase") != intent.get("phase"):
                raise ValueError("provider terminal phase mismatch")
            if record.get("intent_record_sha256") != intent.get("record_sha256"):
                raise ValueError("provider terminal intent binding mismatch")
            _validate_usage(record.get("usage"))
            if record.get("status") == "ok":
                response_text = record.get("response_text")
                if not isinstance(response_text, str):
                    raise ValueError("successful provider terminal lacks response text")
                if record.get("response_sha256") != sha256_text(response_text):
                    raise ValueError("provider response text/digest mismatch")
                if (
                    record.get("error_text") is not None
                    or record.get("error_sha256") is not None
                ):
                    raise ValueError("successful provider terminal has an error")
            elif record.get("status") == "error":
                error_text = record.get("error_text")
                if not isinstance(error_text, str):
                    raise ValueError("failed provider terminal lacks error text")
                if record.get("error_sha256") != sha256_text(error_text):
                    raise ValueError("provider error text/digest mismatch")
                if (
                    record.get("response_text") is not None
                    or record.get("response_sha256") is not None
                ):
                    raise ValueError("failed provider terminal has a response")
            else:
                raise ValueError("provider terminal status mismatch")
            terminals[call_id] = record
        previous = str(record["record_sha256"])
    missing = set(intents) - set(terminals)
    if missing and not allow_incomplete:
        raise ValueError("provider intent is missing its terminal")
    if len(missing) > 1:
        raise ValueError("provider journal has multiple incomplete calls")
    return records


def _read_locked(handle: Any) -> list[dict[str, Any]]:
    handle.seek(0)
    payload = handle.read()
    if payload and not payload.endswith("\n"):
        raise ValueError("provider journal has a torn final line")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"provider journal contains a blank line at {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("provider journal row must be an object")
        records.append(value)
    return records


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class ProviderJournal:
    """Single-writer append API with an OS lock and durable fsync per record."""

    def __init__(
        self,
        path: str | Path,
        attempt_id: str,
        *,
        clock: Callable[[], int] = time.time_ns,
        call_id_factory: Callable[[], Any] = uuid.uuid4,
    ):
        self.path = Path(path)
        self.attempt_id = _canonical_uuid(attempt_id, "attempt ID")
        self.clock = clock
        self.call_id_factory = call_id_factory
        self.require_preinitialized = self.path.name == PROVIDER_JOURNAL.name

    def initialize_once(self) -> None:
        """Create the official empty journal exclusively before any provider call."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(
            self.path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("provider journal must be a regular file")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(self.path.parent)

    def _append(
        self,
        payload: dict[str, Any],
        *,
        pending_call_id: str | None,
    ) -> dict[str, Any]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = (
            os.O_RDWR
            | os.O_APPEND
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        if not self.require_preinitialized:
            flags |= os.O_CREAT
        descriptor = os.open(
            self.path,
            flags,
            0o600,
        )
        with os.fdopen(descriptor, "r+", encoding="utf-8") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("provider journal must be a regular file")
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            records = _read_locked(handle)
            validate_journal_records(records, self.attempt_id, allow_incomplete=True)
            intent_ids = {
                str(record["call_id"])
                for record in records
                if record["kind"] == "provider_call_intent"
            }
            terminal_ids = {
                str(record["call_id"])
                for record in records
                if record["kind"] == "provider_call_terminal"
            }
            pending = intent_ids - terminal_ids
            expected_pending = set() if pending_call_id is None else {pending_call_id}
            if pending != expected_pending:
                raise ValueError("provider journal pending-call state mismatch")
            record = {
                **payload,
                "schema_version": 1,
                "attempt_id": self.attempt_id,
                "sequence": len(records) + 1,
                "previous_record_sha256": (
                    str(records[-1]["record_sha256"]) if records else ZERO_SHA256
                ),
            }
            record["record_sha256"] = journal_record_sha256(record)
            validate_journal_records(
                [*records, record],
                self.attempt_id,
                allow_incomplete=record["kind"] == "provider_call_intent",
            )
            handle.seek(0, os.SEEK_END)
            handle.write(canonical_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(self.path.parent)
        return record

    def begin_call(
        self,
        *,
        phase: str,
        call_key: str,
        call_spec: CallSpec,
    ) -> dict[str, Any]:
        if not isinstance(call_spec, CallSpec):
            raise TypeError("provider journal requires a canonical CallSpec")
        call_spec.validate()
        call_id = str(uuid.UUID(str(self.call_id_factory())))
        return self._append(
            {
                "kind": "provider_call_intent",
                "call_id": call_id,
                "phase": phase,
                "call_key_sha256": sha256_text(str(call_key)),
                "recorded_unix_ns": int(self.clock()),
                "request": call_spec.audit_request(),
            },
            pending_call_id=None,
        )

    def finish_call(
        self,
        intent: dict[str, Any],
        *,
        status: str,
        usage: dict[str, int],
        response: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        if status not in {"ok", "error"}:
            raise ValueError("invalid provider terminal status")
        if status == "ok":
            if not isinstance(response, str) or error is not None:
                raise ValueError("successful provider terminal requires exact response text")
            response_text, error_text = response, None
        else:
            if not isinstance(error, str) or response is not None:
                raise ValueError("failed provider terminal requires exact error text")
            response_text, error_text = None, error
        call_id = str(intent.get("call_id", ""))
        return self._append(
            {
                "kind": "provider_call_terminal",
                "call_id": call_id,
                "phase": intent.get("phase"),
                "intent_record_sha256": intent.get("record_sha256"),
                "recorded_unix_ns": int(self.clock()),
                "status": status,
                "usage": {
                    key: int(usage.get(key, 0) or 0)
                    for key in sorted(USAGE_FIELDS)
                },
                "response_text": response_text,
                "response_sha256": (
                    sha256_text(response_text)
                    if response_text is not None
                    else None
                ),
                "error_text": error_text,
                "error_sha256": (
                    sha256_text(error_text) if error_text is not None else None
                ),
            },
            pending_call_id=call_id,
        )

    def validate(self) -> list[dict[str, Any]]:
        try:
            descriptor = os.open(
                self.path,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError as exc:
            raise ValueError("provider journal is missing or unsafe") from exc
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise ValueError("provider journal must be a regular file")
            records = _read_locked(handle)
        return validate_journal_records(records, self.attempt_id)

    def checkpoint(self) -> dict[str, Any]:
        try:
            records = self.validate()
        except ValueError:
            if self.require_preinitialized:
                raise
            records = []
        return {
            "sequence": len(records),
            "record_sha256": str(records[-1]["record_sha256"]) if records else ZERO_SHA256,
        }

    def _describe_slice(
        self,
        records: list[dict[str, Any]],
        first_sequence: int,
        last_sequence: int,
        phase: str,
    ) -> dict[str, Any]:
        if first_sequence < 1 or last_sequence < first_sequence or last_sequence > len(records):
            raise ValueError("invalid provider journal slice bounds")
        subset = records[first_sequence - 1 : last_sequence]
        if any(record["phase"] != phase for record in subset):
            raise ValueError("provider journal slice crosses phase boundaries")
        intents = [record for record in subset if record["kind"] == "provider_call_intent"]
        terminals = [record for record in subset if record["kind"] == "provider_call_terminal"]
        if not intents or len(intents) != len(terminals):
            raise ValueError("provider journal slice is not terminal-complete")
        return {
            "schema_version": 1,
            "journal_path": str(PROVIDER_JOURNAL),
            "attempt_id": self.attempt_id,
            "phase": phase,
            "first_sequence": first_sequence,
            "last_sequence": last_sequence,
            "previous_record_sha256": str(subset[0]["previous_record_sha256"]),
            "terminal_record_sha256": str(subset[-1]["record_sha256"]),
            "record_count": len(subset),
            "call_count": len(intents),
            "call_ids_sha256": _digest_json(sorted(record["call_id"] for record in intents)),
            "records_sha256": _digest_json(subset),
        }

    def slice_from(self, checkpoint: dict[str, Any], phase: str) -> dict[str, Any]:
        records = self.validate()
        prior_sequence = int(checkpoint.get("sequence", -1))
        prior_digest = checkpoint.get("record_sha256")
        if prior_sequence < 0 or prior_sequence >= len(records):
            raise ValueError("provider journal checkpoint did not advance")
        expected_prior = ZERO_SHA256 if prior_sequence == 0 else records[prior_sequence - 1]["record_sha256"]
        if prior_digest != expected_prior:
            raise ValueError("provider journal checkpoint digest mismatch")
        return self._describe_slice(records, prior_sequence + 1, len(records), phase)

    def validate_slice(self, descriptor: dict[str, Any]) -> list[dict[str, Any]]:
        records = self.validate()
        if descriptor.get("attempt_id") != self.attempt_id:
            raise ValueError("provider journal slice attempt mismatch")
        expected = self._describe_slice(
            records,
            int(descriptor.get("first_sequence", -1)),
            int(descriptor.get("last_sequence", -1)),
            str(descriptor.get("phase", "")),
        )
        if descriptor != expected:
            raise ValueError("provider journal slice manifest mismatch")
        return records[expected["first_sequence"] - 1 : expected["last_sequence"]]


def action_seed(base_seed: int, case_key: str) -> int:
    return (base_seed ^ int(case_key[:8], 16)) % 2_147_483_647


def _final_context(root: Path, attempt_id: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    attempt, implementation, implementation_sha256 = validate_final_attempt(root)
    if attempt.get("attempt_id") != attempt_id:
        raise ValueError("CLI attempt ID does not match the sealed final attempt")
    return attempt, implementation, implementation_sha256


def final_paths(root: Path) -> dict[str, Path]:
    return {
        "journal": root / PROVIDER_JOURNAL,
        "baseline_latent": root / PREFINE_MEMORY,
        "baseline_latent_manifest": root / PREFINE_MEMORY_MANIFEST,
        "memory": root / ECPR_MEMORY,
        "memory_manifest": root / ECPR_MEMORY_MANIFEST,
        "baseline": root / BASELINE_PREDICTIONS,
        "baseline_manifest": root / BASELINE_PREDICTIONS_MANIFEST,
        "candidate": root / CANDIDATE_PREDICTIONS,
        "candidate_manifest": root / CANDIDATE_PREDICTIONS_MANIFEST,
        "summary": root / SUMMARY,
    }


def enforce_unsealed_output_arguments(root: Path, *outputs: str | Path) -> None:
    """Keep scratch calls from creating any confirmatory output or manifest."""
    protected = {path.resolve() for path in final_paths(root.resolve()).values()}
    for output in outputs:
        resolved = Path(output).resolve()
        derived_manifest = Path(str(output) + ".manifest.json").resolve()
        if resolved in protected or derived_manifest in protected:
            raise ValueError("unsealed operation cannot write an official final output")


def _same_path(left: str | Path, right: str | Path) -> bool:
    return Path(left).resolve() == Path(right).resolve()


def enforce_final_build_memory_arguments(
    root: Path,
    contract: dict[str, Any],
    *,
    preregistration: Path,
    history: Path,
    preference_slots: Path,
    output: Path,
    base_url: str | None,
    latent_path: Path | None,
) -> None:
    paths = final_paths(root)
    if latent_path is not None:
        raise ValueError("--latent-path is forbidden in the confirmatory final run")
    if base_url != contract["pipeline"]["base_url"]:
        raise ValueError("confirmatory memory generation requires the frozen loopback endpoint")
    expected = {
        preregistration: root / "preregistration.json",
        history: root / "artifacts/history.sanitized.jsonl",
        preference_slots: root / "configs/preference_slots.json",
        output: paths["memory"],
    }
    if any(not _same_path(actual, wanted) for actual, wanted in expected.items()):
        raise ValueError("confirmatory memory generation received an alternate path")


def enforce_final_inference_arguments(
    root: Path,
    contract: dict[str, Any],
    *,
    arm: str,
    preregistration: Path,
    output: Path,
    base_url: str,
    ablations: set[str],
) -> None:
    paths = final_paths(root)
    if ablations:
        raise ValueError("ablations are forbidden in the confirmatory final run")
    if base_url != contract["pipeline"]["base_url"]:
        raise ValueError("confirmatory inference requires the frozen loopback endpoint")
    if not _same_path(preregistration, root / "preregistration.json"):
        raise ValueError("confirmatory inference received an alternate preregistration")
    if not _same_path(output, paths[arm]):
        raise ValueError("confirmatory inference received an alternate output path")


def enforce_final_evaluation_arguments(
    root: Path,
    *,
    preregistration: Path,
    baseline: Path,
    candidate: Path,
    output: Path,
) -> None:
    paths = final_paths(root)
    expected = {
        preregistration: root / "preregistration.json",
        baseline: paths["baseline"],
        candidate: paths["candidate"],
        output: paths["summary"],
    }
    if any(not _same_path(actual, wanted) for actual, wanted in expected.items()):
        raise ValueError("confirmatory evaluation received an alternate path")


def assert_final_artifacts_absent(root: Path) -> None:
    existing = [
        str(path.relative_to(root))
        for path in final_paths(root).values()
        if path.exists() or path.is_symlink()
    ]
    if existing:
        raise FileExistsError(f"confirmatory output already exists: {sorted(existing)}")


def _identity(root: Path, path: Path, key: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    rows = unique_by(iter_jsonl(path), key)
    return (
        {
            "path": str(path.relative_to(root)),
            "sha256": sha256_file(path),
            "row_count": len(rows),
            f"{key}_universe_sha256": _digest_json(sorted(rows)),
        },
        rows,
    )


def _attempt_binding(root: Path, attempt_id: str) -> dict[str, Any]:
    attempt, implementation, implementation_sha256 = _final_context(root, attempt_id)
    return {
        "attempt_id": attempt_id,
        "attempt_record_sha256": sha256_file(root / ATTEMPT_LOCK),
        "implementation_manifest_sha256": implementation_sha256,
        "model_contract_sha256": implementation["model_contract_sha256"],
        "runtime_contract_sha256": implementation["runtime_contract_sha256"],
        "pipeline_contract_sha256": implementation["pipeline_contract_sha256"],
        "action_budget_sha256": implementation["action_budget_sha256"],
        "candidate_revision": implementation["candidate_revision"],
        "historical_expected_runtime_contract_sha256": implementation[
            "historical_expected_runtime_contract_sha256"
        ],
        "safe_transfer_amendment_sha256": implementation[
            "safe_transfer_amendment_sha256"
        ],
        "v1_expected_runtime_contract_sha256": implementation[
            "v1_expected_runtime_contract_sha256"
        ],
        "vlt_audit_amendment_sha256": implementation[
            "vlt_audit_amendment_sha256"
        ],
        "latent_trait_ontology_sha256": implementation[
            "latent_trait_ontology_sha256"
        ],
        "safe_transfer_contract_sha256": implementation[
            "safe_transfer_contract_sha256"
        ],
        "sealed_evaluator_manifest_sha256": attempt["binding"]["sealed_evaluator_manifest_sha256"],
    }


def _memory_manifest_values(
    root: Path,
    attempt_id: str,
    generation_slice: dict[str, Any],
    latent_manifest_sha256: str | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, JournalReplayCapability]]:
    binding = _attempt_binding(root, attempt_id)
    _attempt, implementation, _implementation_sha256 = _final_context(
        root, attempt_id
    )
    sealed_path = root / "evaluator_vault/sealed_manifest.json"
    sealed = load_json(sealed_path)
    history_identity, history_rows = _identity(
        root, root / "artifacts/history.sanitized.jsonl", "example_id"
    )
    latent_identity, latent_rows = _identity(root, root / PREFINE_MEMORY, "example_id")
    memory_identity, memory_rows = _identity(root, root / ECPR_MEMORY, "example_id")
    if set(history_rows) != set(latent_rows) or set(history_rows) != set(memory_rows):
        raise ValueError("memory artifacts do not cover the exact sanitized-history universe")
    for example_id, memory in memory_rows.items():
        if set(memory) != {
            "example_id",
            "latent_abstraction",
            "latent_attestation",
            "typed_hypotheses",
            "method",
            "candidate_revision",
        }:
            raise ValueError("ECPR memory row field contract violation")
        if memory["method"] != "ecpr_v1":
            raise ValueError("ECPR memory method mismatch")
        if memory["candidate_revision"] != implementation["candidate_revision"]:
            raise ValueError("ECPR memory candidate revision mismatch")
        validate_latent_attestation(
            memory["latent_abstraction"],
            memory["latent_attestation"],
        )
        latent = latent_rows[example_id]
        if set(latent) != {"example_id", "latent_abstraction", "method"}:
            raise ValueError("PREFINE projection row field contract violation")
        if latent != {
            "example_id": example_id,
            "latent_abstraction": memory["latent_abstraction"],
            "method": "prefine_v1",
        }:
            raise ValueError("PREFINE projection is not an exact ECPR latent projection")
    contract = implementation["expected_contract"]
    generation_records = ProviderJournal(
        root / PROVIDER_JOURNAL, attempt_id
    ).validate_slice(generation_slice)
    semantic_replay, _capabilities = replay_memory_generation(
        histories=history_rows,
        memory_rows=memory_rows,
        latent_rows=latent_rows,
        records=generation_records,
        attempt_id=attempt_id,
        endpoint=contract["pipeline"]["base_url"].rstrip("/")
        + "/v1/chat/completions",
        model=contract["model"]["id"],
        timeout_seconds=float(contract["provider_timeout_seconds"]),
        base_seed=int(contract["action_budget"]["seed"]),
    )
    history = {
        **history_identity,
        "declared_history_sha256": sealed["history_sha256"],
        "sealed_manifest_sha256": sha256_file(sealed_path),
    }
    if history["sha256"] != history["declared_history_sha256"]:
        raise ValueError("sanitized history no longer matches the sealed evaluator manifest")
    common = {
        "schema_version": 3,
        **binding,
        "sanitized_history": history,
        "generation_call_ledger": generation_slice,
        "semantic_replay": semantic_replay,
    }
    latent_manifest = {
        **common,
        "kind": "sealed_prefine_latent_projection_manifest",
        "baseline_latent": latent_identity,
    }
    memory_manifest = {
        **common,
        "kind": "sealed_ecpr_memory_manifest",
        "pipeline_predecessor_manifest_sha256": latent_manifest_sha256,
        "baseline_latent_manifest_sha256": latent_manifest_sha256,
        "memory": memory_identity,
    }
    return latent_manifest, memory_manifest, _capabilities


def publish_memory_manifests(
    root: Path,
    attempt_id: str,
    generation_slice: dict[str, Any],
) -> dict[str, Any]:
    journal = ProviderJournal(root / PROVIDER_JOURNAL, attempt_id)
    journal.validate_slice(generation_slice)
    if generation_slice.get("phase") != "memory_generation":
        raise ValueError("memory manifest received a non-memory journal slice")
    latent_manifest, _, _ = _memory_manifest_values(root, attempt_id, generation_slice, None)
    commit_json_once(root / PREFINE_MEMORY_MANIFEST, latent_manifest)
    latent_sha256 = sha256_file(root / PREFINE_MEMORY_MANIFEST)
    _, memory_manifest, _ = _memory_manifest_values(
        root, attempt_id, generation_slice, latent_sha256
    )
    commit_json_once(root / ECPR_MEMORY_MANIFEST, memory_manifest)
    return memory_manifest


def validate_memory_manifests(root: Path, attempt_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    latent_path = root / PREFINE_MEMORY_MANIFEST
    memory_path = root / ECPR_MEMORY_MANIFEST
    if any(path.is_symlink() or not path.is_file() for path in (latent_path, memory_path)):
        raise ValueError("sealed memory manifest is missing")
    latent = load_json(latent_path)
    memory = load_json(memory_path)
    descriptor = memory.get("generation_call_ledger")
    if not isinstance(descriptor, dict) or latent.get("generation_call_ledger") != descriptor:
        raise ValueError("memory manifests disagree on their generation ledger")
    if descriptor.get("phase") != "memory_generation":
        raise ValueError("sealed memory manifest has a non-memory ledger slice")
    journal = ProviderJournal(root / PROVIDER_JOURNAL, attempt_id)
    journal.validate_slice(descriptor)
    expected_latent, expected_memory, _ = _memory_manifest_values(
        root, attempt_id, descriptor, sha256_file(latent_path)
    )
    if latent != expected_latent or memory != expected_memory:
        raise ValueError("sealed memory manifest content/hash DAG mismatch")
    return latent, memory


def validate_memory_replay_capabilities(
    root: Path,
    attempt_id: str,
) -> dict[str, JournalReplayCapability]:
    """Re-run exact replay and return only freshly identity-sealed capabilities."""
    latent, memory = validate_memory_manifests(root, attempt_id)
    descriptor = memory["generation_call_ledger"]
    expected_latent, expected_memory, capabilities = _memory_manifest_values(
        root,
        attempt_id,
        descriptor,
        sha256_file(root / PREFINE_MEMORY_MANIFEST),
    )
    if latent != expected_latent or memory != expected_memory:
        raise ValueError("memory replay capability manifest mismatch")
    return {
        example_id: _mint_replay_capability(**claim.artifact())
        for example_id, claim in capabilities.items()
    }



def validate_arm_memory_binding(
    arm: str,
    *,
    baseline_latent_manifest_sha256: str | None,
    memory_manifest_sha256: str | None,
) -> None:
    if arm == "baseline":
        _require_sha256(baseline_latent_manifest_sha256, "baseline latent manifest digest")
        if memory_manifest_sha256 is not None:
            raise ValueError("baseline cannot consume the sealed ECPR memory manifest")
    elif arm == "candidate":
        if baseline_latent_manifest_sha256 is not None:
            raise ValueError("candidate must consume the sealed ECPR memory manifest directly")
        _require_sha256(memory_manifest_sha256, "candidate memory manifest digest")
    else:
        raise ValueError("unknown prediction arm")


def _prediction_manifest_value(
    root: Path,
    attempt_id: str,
    arm: str,
    output: Path,
    action_slice: dict[str, Any],
) -> dict[str, Any]:
    binding = _attempt_binding(root, attempt_id)
    paths = final_paths(root)
    if not _same_path(output, paths[arm]):
        raise ValueError("prediction manifest received an alternate output path")
    tasks_identity, tasks = _identity(
        root, root / "artifacts/tasks.jsonl", "case_key"
    )
    predictions_identity, predictions = _identity(root, output, "case_key")
    if set(tasks) != set(predictions):
        raise ValueError("predictions do not cover the exact public task universe")

    journal = ProviderJournal(root / PROVIDER_JOURNAL, attempt_id)
    slice_records = journal.validate_slice(action_slice)
    expected_phase = f"action_{arm}"
    if (
        action_slice.get("phase") != expected_phase
        or action_slice.get("call_count") != len(tasks)
        or action_slice.get("record_count") != 2 * len(tasks)
    ):
        raise ValueError("action ledger slice phase or call count mismatch")
    intent_records = [
        record
        for record in slice_records
        if record["kind"] == "provider_call_intent"
    ]
    terminal_records = [
        record
        for record in slice_records
        if record["kind"] == "provider_call_terminal"
    ]
    expected_call_keys = [sha256_text(case_key) for case_key in tasks]
    if (
        len(intent_records) != len(tasks)
        or len(terminal_records) != len(tasks)
        or [record["call_key_sha256"] for record in intent_records]
        != expected_call_keys
    ):
        raise ValueError(
            "action ledger call order differs from the public task order"
        )
    terminals = {record["call_id"]: record for record in terminal_records}
    if len(terminals) != len(tasks):
        raise ValueError("action ledger contains duplicate terminal call IDs")

    _, implementation, _ = _final_context(root, attempt_id)
    contract = implementation["expected_contract"]
    budget = contract["action_budget"]
    model = contract["model"]
    timeout_seconds = float(contract["provider_timeout_seconds"])
    endpoint = (
        contract["pipeline"]["base_url"].rstrip("/")
        + "/v1/chat/completions"
    )
    policy = CandidatePolicy(**contract["candidate_policy"])
    preference_slots = load_json(root / "configs/preference_slots.json")
    schemas = {
        "single": load_json(root / "configs/schema_single.json"),
        "multi": load_json(root / "configs/schema_multi.json"),
    }
    histories = unique_by(
        iter_jsonl(root / "artifacts/history.sanitized.jsonl"),
        "example_id",
    )
    memory_path = root / (PREFINE_MEMORY if arm == "baseline" else ECPR_MEMORY)
    memory_rows = unique_by(iter_jsonl(memory_path), "example_id")

    public_ontology = None
    latent_ontology = None
    latent_ontology_sha256 = None
    replay_capabilities: dict[str, JournalReplayCapability] = {}
    if arm == "candidate":
        public_path = root / str(contract["routing_gate"]["ontology_path"])
        public_ontology = load_json(public_path)
        validate_ontology(
            public_ontology,
            schemas["single"] + schemas["multi"],
        )
        latent_path_value = contract.get("latent_trait_ontology_path")
        if not isinstance(latent_path_value, str) or not latent_path_value:
            raise ValueError(
                "runtime contract lacks an explicit VLT ontology path"
            )
        latent_path = root / latent_path_value
        latent_ontology = load_json(latent_path)
        candidate_revision = contract.get("candidate_revision")
        if candidate_revision == VLT2_CANDIDATE_REVISION:
            validate_latent_trait_ontology_v2(latent_ontology)
        elif candidate_revision == VLT3_CANDIDATE_REVISION:
            validate_latent_trait_ontology_v3(latent_ontology)
        else:
            raise ValueError("unsupported runtime VLT candidate revision")
        validate_vlt_schema_contract(
            latent_ontology,
            schemas,
            preference_slots,
        )
        latent_ontology_sha256 = sha256_file(latent_path)
        if latent_ontology_sha256 != binding["latent_trait_ontology_sha256"]:
            raise ValueError(
                "runtime VLT ontology differs from the attempt binding"
            )
        replay_capabilities = validate_memory_replay_capabilities(
            root,
            attempt_id,
        )

    total_usage = {key: 0 for key in sorted(USAGE_FIELDS)}
    action_decisions: list[dict[str, Any]] = []
    for index, (case_key, task) in enumerate(tasks.items()):
        row = predictions[case_key]
        if set(row) != PREDICTION_FIELDS:
            raise ValueError("prediction row field contract violation")
        example_id = str(task["example_id"])
        if example_id not in histories or example_id not in memory_rows:
            raise ValueError("prediction task lacks sealed history or memory")
        if (
            row.get("case_key") != case_key
            or row.get("example_id") != example_id
            or row.get("mode") != task.get("mode")
            or row.get("arm") != arm
        ):
            raise ValueError("prediction row public identity mismatch")

        schema = schemas[task["schema_key"]]
        action_case = build_action_case(
            arm=arm,
            task=task,
            history=histories[example_id],
            memory=memory_rows[example_id],
            schema=schema,
            preference_slots=preference_slots,
            policy=policy,
            public_domain_ontology=public_ontology,
            latent_trait_ontology=latent_ontology,
            latent_trait_ontology_sha256=latent_ontology_sha256,
            replay_capability=replay_capabilities.get(example_id),
            ablations=set(),
        )
        action_decisions.append(action_case.audit_artifact)
        call_spec = CallSpec.from_messages(
            endpoint=endpoint,
            model=model["id"],
            messages=[{"role": "user", "content": action_case.prompt}],
            seed=action_seed(int(budget["seed"]), case_key),
            temperature=float(budget["temperature"]),
            max_tokens=int(budget["max_tokens"]),
            json_object=False,
            schema_sha256=_digest_json(schema),
            timeout_seconds=timeout_seconds,
        )
        intent = intent_records[index]
        if intent["request"] != call_spec.audit_request():
            raise ValueError(
                "actual action request differs from canonical prompt CallSpec"
            )
        terminal = terminals.get(intent["call_id"])
        if terminal is None:
            raise ValueError("action intent has no exact terminal")
        expected_status = "ok" if terminal["status"] == "ok" else "error"
        expected_row_values = {
            "status": expected_status,
            "model_snapshot": model["snapshot"],
            "seed": call_spec.seed,
            "temperature": call_spec.temperature,
            "max_tokens": call_spec.max_tokens,
            "calls": 1,
            "prompt_hash": sha256_text(action_case.prompt),
            "schema_hash": call_spec.schema_sha256,
            "memory_hash": sha256_text(action_case.memory_block),
            "usage": terminal["usage"],
        }
        if any(row.get(key) != value for key, value in expected_row_values.items()):
            raise ValueError(
                "prediction row runtime metadata differs from canonical replay"
            )
        if terminal["status"] == "ok":
            response_text = terminal["response_text"]
            if (
                response_text != row.get("llm_output")
                or terminal["response_sha256"] != sha256_text(response_text)
            ):
                raise ValueError(
                    "prediction output differs from exact provider response"
                )
        elif row.get("llm_output") != "":
            raise ValueError(
                "failed provider call must have an empty prediction output"
            )
        for key in total_usage:
            total_usage[key] += terminal["usage"][key]

    latent_sha256 = (
        sha256_file(root / PREFINE_MEMORY_MANIFEST)
        if arm == "baseline"
        else None
    )
    memory_sha256 = (
        sha256_file(root / ECPR_MEMORY_MANIFEST)
        if arm == "candidate"
        else None
    )
    validate_arm_memory_binding(
        arm,
        baseline_latent_manifest_sha256=latent_sha256,
        memory_manifest_sha256=memory_sha256,
    )
    predecessor = (
        sha256_file(root / ECPR_MEMORY_MANIFEST)
        if arm == "baseline"
        else sha256_file(root / BASELINE_PREDICTIONS_MANIFEST)
    )
    return {
        "schema_version": 3,
        "kind": "sealed_prediction_manifest",
        "arm": arm,
        **binding,
        "pipeline_predecessor_manifest_sha256": predecessor,
        "tasks": tasks_identity,
        "baseline_latent_manifest_sha256": latent_sha256,
        "memory_manifest_sha256": memory_sha256,
        "predictions": predictions_identity,
        "action_call_ledger": action_slice,
        "action_case_decisions": action_decisions,
        "action_case_decisions_sha256": _digest_json(action_decisions),
        "authoritative_action_contract": {
            "model": model,
            "action_budget": budget,
            "provider_timeout_seconds": timeout_seconds,
            "call_count": len(tasks),
            "usage": total_usage,
            "row_self_reported_runtime_metadata": "verified_against_canonical_replay",
        },
        "ablation_flags": [],
    }

def publish_prediction_manifest(
    root: Path,
    attempt_id: str,
    arm: str,
    output: Path,
    action_slice: dict[str, Any],
) -> dict[str, Any]:
    validate_memory_manifests(root, attempt_id)
    if arm == "candidate":
        validate_prediction_manifest(root, attempt_id, "baseline", final_paths(root)["baseline"])
    manifest = _prediction_manifest_value(root, attempt_id, arm, output, action_slice)
    commit_json_once(final_paths(root)[f"{arm}_manifest"], manifest)
    return manifest


def validate_prediction_manifest(
    root: Path,
    attempt_id: str,
    arm: str,
    output: Path,
) -> dict[str, Any]:
    manifest_path = final_paths(root)[f"{arm}_manifest"]
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"sealed {arm} prediction manifest is missing")
    manifest = load_json(manifest_path)
    action_slice = manifest.get("action_call_ledger")
    if not isinstance(action_slice, dict):
        raise ValueError("prediction manifest action ledger is missing")
    expected = _prediction_manifest_value(root, attempt_id, arm, output, action_slice)
    if manifest != expected:
        raise ValueError(f"sealed {arm} prediction manifest content/hash DAG mismatch")
    return manifest


def validate_run_dag(
    root: Path,
    attempt_id: str,
    baseline_path: Path,
    candidate_path: Path,
) -> dict[str, Any]:
    _, memory = validate_memory_manifests(root, attempt_id)
    baseline = validate_prediction_manifest(root, attempt_id, "baseline", baseline_path)
    candidate = validate_prediction_manifest(root, attempt_id, "candidate", candidate_path)
    slices = [
        memory["generation_call_ledger"],
        baseline["action_call_ledger"],
        candidate["action_call_ledger"],
    ]
    if slices[0]["first_sequence"] != 1:
        raise ValueError("provider journal has calls before sealed memory generation")
    for left, right in zip(slices, slices[1:]):
        if right["first_sequence"] != left["last_sequence"] + 1:
            raise ValueError("provider journal phase slices are not adjacent")
        if right["previous_record_sha256"] != left["terminal_record_sha256"]:
            raise ValueError("provider journal phase slices break the forward hash chain")
    records = ProviderJournal(root / PROVIDER_JOURNAL, attempt_id).validate()
    if slices[-1]["last_sequence"] != len(records):
        raise ValueError("provider journal contains calls outside the final DAG")
    left_contract = baseline["authoritative_action_contract"]
    right_contract = candidate["authoritative_action_contract"]
    equal = (
        left_contract["model"] == right_contract["model"]
        and left_contract["action_budget"] == right_contract["action_budget"]
        and left_contract["provider_timeout_seconds"]
        == right_contract["provider_timeout_seconds"]
        and left_contract["call_count"] == right_contract["call_count"]
    )
    if not equal:
        raise ValueError("baseline and candidate authoritative action contracts differ")
    task_count = int(baseline["tasks"]["row_count"])
    memory_call_count = int(slices[0]["call_count"])
    baseline_call_count = int(slices[1]["call_count"])
    candidate_call_count = int(slices[2]["call_count"])
    if baseline_call_count != task_count or candidate_call_count != task_count:
        raise ValueError("official action-arm call counts differ from task count")
    total_call_count = memory_call_count + baseline_call_count + candidate_call_count
    expected_total_call_count = memory_call_count + 2 * task_count
    if total_call_count != expected_total_call_count:
        raise ValueError("provider call cardinality differs from memory plus both task arms")
    if len(records) != 2 * total_call_count:
        raise ValueError("provider journal record count differs from exact call DAG")
    authoritative_contract = {
        "model": left_contract["model"],
        "action_budget": left_contract["action_budget"],
        "provider_timeout_seconds": left_contract["provider_timeout_seconds"],
        "call_count_per_arm": task_count,
    }
    return {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "provider_journal_sha256": sha256_file(root / PROVIDER_JOURNAL),
        "memory_manifest_sha256": sha256_file(root / ECPR_MEMORY_MANIFEST),
        "baseline_prediction_manifest_sha256": sha256_file(root / BASELINE_PREDICTIONS_MANIFEST),
        "candidate_prediction_manifest_sha256": sha256_file(root / CANDIDATE_PREDICTIONS_MANIFEST),
        "equal_action_budget": equal,
        "task_count": task_count,
        "arms": ["baseline", "candidate"],
        "memory_call_count": memory_call_count,
        "baseline_action_call_count": baseline_call_count,
        "candidate_action_call_count": candidate_call_count,
        "total_call_count": total_call_count,
        "expected_total_call_count": expected_total_call_count,
        "journal_record_count": len(records),
        "expected_journal_record_count": 2 * total_call_count,
        "final_journal_sequence": int(records[-1]["sequence"]),
        "final_journal_record_sha256": str(records[-1]["record_sha256"]),
        "phases": ["memory_generation", "action_baseline", "action_candidate"],
        "no_extra_provider_calls_or_phases": True,
        "authoritative_action_contract": authoritative_contract,
        "authoritative_action_contract_sha256": _digest_json(
            authoritative_contract
        ),
    }
