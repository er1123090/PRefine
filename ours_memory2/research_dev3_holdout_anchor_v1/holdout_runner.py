"""Gold-free PReFine memory construction and paired action inference.

This runner can read only sanitized history, public tasks, frozen schemas, and
the locked preregistration.  It has no import or path to evaluator gold.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ecpr.anchor import build_anchored_action_case
from ecpr.contracts import CandidatePolicy, PREDICTION_FIELDS, TASK_FIELDS
from ecpr.firewall import validate_sanitized_history
from ecpr.io import (
    canonical_json,
    iter_jsonl,
    load_json,
    sha256_file,
    sha256_text,
    unique_by,
    write_json_once,
    write_jsonl_once,
)
from ecpr.memory import build_latent_abstraction_result, build_typed_hypotheses
from ecpr.prompts import baseline_memory_block, build_action_prompt


ROOT = Path(__file__).resolve().parent
MEMORY_PATH = Path("artifacts/memory.prefine_anchor.jsonl")
PREDICTION_PATHS = {
    "baseline": Path("artifacts/predictions.baseline.jsonl"),
    "candidate": Path("artifacts/predictions.candidate.jsonl"),
}


class ProviderError(RuntimeError):
    """A loopback provider failed without serializing a response body."""


@dataclass(frozen=True)
class Completion:
    content: str
    usage: dict[str, int]
    call_id: str | None
    request_sha256: str


@dataclass
class DigestJournal:
    """Non-content provider audit trail for pre-gold integrity checking."""

    attempt_id: str
    records: list[dict[str, Any]] = field(default_factory=list)

    def record(
        self,
        *,
        phase: str | None,
        call_key: str | None,
        request_sha256: str,
        content: str | None,
        usage: dict[str, int],
        status: str,
    ) -> str:
        call_id = f"{self.attempt_id}:{len(self.records):08d}"
        self.records.append(
            {
                "schema_version": 1,
                "kind": "provider_call_digest_v1",
                "call_id": call_id,
                "phase": str(phase or "unspecified"),
                "call_key_sha256": sha256_text(str(call_key or "")),
                "request_sha256": request_sha256,
                "response_sha256": sha256_text(content) if content is not None else None,
                "usage": dict(usage),
                "status": status,
            }
        )
        return call_id


class LoopbackChatClient:
    """A small OpenAI-compatible client that accepts only localhost endpoints."""

    def __init__(self, base_url: str, model: str, *, journal: DigestJournal) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("only an explicit loopback model endpoint is permitted")
        self.endpoint = base_url.rstrip("/") + "/v1/chat/completions"
        self.model = model
        self.journal = journal

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        seed: int,
        max_tokens: int,
        temperature: float,
        json_object: bool = False,
        phase: str | None = None,
        call_key: str | None = None,
        schema_sha256: str | None = None,
    ) -> Completion:
        if not schema_sha256:
            raise ValueError("model call lacks schema digest")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "seed": int(seed),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "n": 1,
            "stream": False,
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}
        request_bytes = canonical_json(payload).encode("utf-8")
        request_sha256 = sha256_text(request_bytes.decode("utf-8"))
        zero_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        request = urllib.request.Request(
            self.endpoint,
            data=request_bytes,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                body = json.loads(response.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            if not isinstance(content, str) or not content:
                raise ValueError("empty model completion")
            raw_usage = body.get("usage", {})
            usage = {
                key: int(raw_usage.get(key, 0) or 0)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
        except (
            urllib.error.URLError,
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            IndexError,
            TypeError,
            ValueError,
        ) as exc:
            self.journal.record(
                phase=phase,
                call_key=call_key,
                request_sha256=request_sha256,
                content=None,
                usage=zero_usage,
                status="error",
            )
            raise ProviderError(type(exc).__name__) from None
        call_id = self.journal.record(
            phase=phase,
            call_key=call_key,
            request_sha256=request_sha256,
            content=content,
            usage=usage,
            status="ok",
        )
        return Completion(
            content=content,
            usage=usage,
            call_id=call_id,
            request_sha256=request_sha256,
        )


def _assert_absent(path: Path) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite sealed output: {path}")


def _load_preregistration(root: Path) -> dict[str, Any]:
    prereg = load_json(root / "preregistration.json")
    if prereg.get("status") != "locked_before_target_metrics":
        raise ValueError("runner requires a locked preregistration")
    return prereg


def _policy(preregistration: dict[str, Any]) -> CandidatePolicy:
    policy = CandidatePolicy(**preregistration["candidate_parameters"])
    if not policy.recency_tiebreak_only or not policy.raw_api_history_in_candidate_prompt:
        raise ValueError("anchored candidate policy must preserve source API history")
    return policy


def _action_seed(base_seed: int, case_key: str) -> int:
    return int(base_seed) + int(sha256_text(case_key)[:8], 16)


def _write_journal(root: Path, stage: str, journal: DigestJournal) -> str:
    path = root / "artifacts" / f"provider_calls.{stage}.jsonl"
    _assert_absent(path)
    write_jsonl_once(path, journal.records)
    return sha256_file(path)


def build_memory(root: Path, base_url: str) -> dict[str, Any]:
    root = root.resolve()
    output = root / MEMORY_PATH
    _assert_absent(output)
    prereg = _load_preregistration(root)
    policy = _policy(prereg)
    history_path = root / "artifacts/history.sanitized.jsonl"
    history_count = validate_sanitized_history(history_path)
    slots = load_json(root / "configs/preference_slots.json")
    if not isinstance(slots, dict):
        raise ValueError("preference slot config must be an object")
    journal = DigestJournal("memory")
    client = LoopbackChatClient(base_url, str(prereg["model"]["id"]), journal=journal)
    max_attempts = int(prereg["prefine"]["max_attempts"])
    seed = int(prereg["inference_budget"]["seed"])
    records: list[dict[str, Any]] = []
    for history in iter_jsonl(history_path):
        example_id = str(history["example_id"])
        result = build_latent_abstraction_result(
            history,
            client,
            seed + int(sha256_text(example_id)[:8], 16),
            max_attempts=max_attempts,
            example_id=example_id,
        )
        records.append(
            {
                "example_id": example_id,
                "latent_abstraction": result.latent_abstraction,
                "latent_attestation": result.verification_attestation.as_dict(),
                "semantic_trace_sha256": result.semantic_trace_sha256,
                "typed_hypotheses": build_typed_hypotheses(history, slots, policy),
                "method": "prefine_anchored_typed_overlay_v1",
            }
        )
    if len(records) != history_count:
        raise AssertionError("memory build coverage mismatch")
    write_jsonl_once(output, records)
    journal_sha256 = _write_journal(root, "memory", journal)
    manifest = {
        "schema_version": 1,
        "record_count": len(records),
        "history_sha256": sha256_file(history_path),
        "memory_sha256": sha256_file(output),
        "provider_journal_sha256": journal_sha256,
        "prefine_max_attempts": max_attempts,
    }
    write_json_once(root / "artifacts/memory.manifest.json", manifest)
    return manifest


def infer_arm(root: Path, base_url: str, arm: str) -> dict[str, Any]:
    if arm not in PREDICTION_PATHS:
        raise ValueError("arm must be baseline or candidate")
    root = root.resolve()
    output = root / PREDICTION_PATHS[arm]
    _assert_absent(output)
    prereg = _load_preregistration(root)
    policy = _policy(prereg)
    budget = prereg["inference_budget"]
    if int(budget["calls_per_case"]) != 1 or int(budget["retries"]) != 0:
        raise ValueError("runner requires one action call and zero retries")
    tasks = unique_by(iter_jsonl(root / "artifacts/tasks.jsonl"), "case_key")
    histories = unique_by(iter_jsonl(root / "artifacts/history.sanitized.jsonl"), "example_id")
    memories = unique_by(iter_jsonl(root / MEMORY_PATH), "example_id")
    slots = load_json(root / "configs/preference_slots.json")
    ontology = load_json(root / "configs/public_domain_ontology.json")
    constraint_ontology = load_json(root / "configs/latent_trait_ontology.vlt3.json")
    schemas = {
        "single": load_json(root / "configs/schema_single.json"),
        "multi": load_json(root / "configs/schema_multi.json"),
    }
    if not isinstance(slots, dict) or not isinstance(ontology, dict) or not isinstance(constraint_ontology, dict):
        raise ValueError("frozen routing inputs have invalid shapes")
    journal = DigestJournal(f"action_{arm}")
    client = LoopbackChatClient(base_url, str(prereg["model"]["id"]), journal=journal)
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    for case_key, task in tasks.items():
        if set(task) != TASK_FIELDS:
            raise ValueError("public task field contract violation")
        example_id = str(task["example_id"])
        if example_id not in histories or example_id not in memories:
            raise ValueError("public task lacks a matching sanitized memory")
        schema = schemas.get(str(task["schema_key"]))
        if schema is None:
            raise ValueError("public task has an unknown schema key")
        if arm == "baseline":
            block = baseline_memory_block(memories[example_id], histories[example_id], policy.memory_lexical_token_cap)
            prompt = build_action_prompt(task, schema, block)
            audit = {
                "schema_version": 1,
                "kind": "source_prefine_baseline_action_case_v1",
                "case_key_sha256": sha256_text(case_key),
                "memory_block_sha256": sha256_text(block),
                "prompt_sha256": sha256_text(prompt),
            }
        else:
            anchored = build_anchored_action_case(
                task=task,
                history=histories[example_id],
                memory=memories[example_id],
                schema=schema,
                preference_slots=slots,
                policy=policy,
                public_domain_ontology=ontology,
                constraint_ontology=constraint_ontology,
            )
            block, prompt, audit = (
                anchored.memory_block,
                anchored.prompt,
                anchored.audit_artifact,
            )
        status = "ok"
        content = ""
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        try:
            completion = client.complete(
                [{"role": "user", "content": prompt}],
                seed=_action_seed(int(budget["seed"]), case_key),
                max_tokens=int(budget["max_tokens"]),
                temperature=float(budget["temperature"]),
                phase=f"action_{arm}",
                call_key=case_key,
                schema_sha256=sha256_text(canonical_json(schema)),
            )
            content, usage = completion.content, completion.usage
        except ProviderError:
            status = "error"
        row = {
            "case_key": case_key,
            "example_id": example_id,
            "mode": task["mode"],
            "arm": arm,
            "llm_output": content,
            "status": status,
            "model_snapshot": prereg["model"]["snapshot"],
            "seed": _action_seed(int(budget["seed"]), case_key),
            "temperature": float(budget["temperature"]),
            "max_tokens": int(budget["max_tokens"]),
            "calls": 1,
            "prompt_hash": sha256_text(prompt),
            "schema_hash": sha256_text(canonical_json(schema)),
            "memory_hash": sha256_text(block),
            "usage": usage,
        }
        if set(row) != PREDICTION_FIELDS:
            raise AssertionError("prediction field contract drift")
        rows.append(row)
        audits.append(audit)
    if len(rows) != len(tasks):
        raise AssertionError("action inference coverage mismatch")
    write_jsonl_once(output, rows)
    journal_sha256 = _write_journal(root, f"action_{arm}", journal)
    audit_path = root / "artifacts" / f"action_audit.{arm}.json"
    write_json_once(
        audit_path,
        {
            "schema_version": 1,
            "arm": arm,
            "case_count": len(audits),
            "action_cases_sha256": sha256_text(canonical_json(audits)),
            "provider_journal_sha256": journal_sha256,
        },
    )
    return {
        "arm": arm,
        "case_count": len(rows),
        "predictions_sha256": sha256_file(output),
        "provider_journal_sha256": journal_sha256,
        "action_audit_sha256": sha256_file(audit_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="gold-free fresh-holdout runner")
    parser.add_argument("--root", type=Path, default=ROOT)
    sub = parser.add_subparsers(dest="command", required=True)
    memory = sub.add_parser("build-memory")
    memory.add_argument("--base-url", required=True)
    infer = sub.add_parser("infer")
    infer.add_argument("--base-url", required=True)
    infer.add_argument("--arm", choices=("baseline", "candidate"), required=True)
    args = parser.parse_args(argv)
    result = (
        build_memory(args.root, args.base_url)
        if args.command == "build-memory"
        else infer_arm(args.root, args.base_url, args.arm)
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
