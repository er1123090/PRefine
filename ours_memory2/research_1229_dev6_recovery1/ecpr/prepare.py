"""Evaluator-only sealed task builder. It never emits target values to stdout."""

from __future__ import annotations

import itertools
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .contracts import TASK_FIELDS
from .firewall import sanitize_dataset
from .io import (
    canonical_json,
    iter_jsonl,
    load_json,
    sha256_file,
    sha256_text,
    verify_sha256,
    write_json,
    write_jsonl,
)
from .parsing import call_to_string, extract_calls


LEGACY_TASK_FIELDS = frozenset(TASK_FIELDS | {"target_domain", "explicit_slots"})
MODE_SCHEMA_KEYS = {"singleturn": "single", "multiturn": "multi"}


def canonical_query(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("public task query must be a string")
    normalized = value.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n")).strip()
    if not normalized:
        raise ValueError("public task query must be nonempty")
    return normalized


def _public_identity(
    row: dict[str, Any], schema_hashes: dict[str, str]
) -> dict[str, str]:
    example_id = str(row.get("example_id", "")).strip()
    mode = str(row.get("mode", "")).strip()
    schema_key = str(row.get("schema_key", "")).strip()
    if not example_id:
        raise ValueError("public task lacks example_id")
    if mode not in MODE_SCHEMA_KEYS:
        raise ValueError(f"invalid public task mode: {mode}")
    if schema_key != MODE_SCHEMA_KEYS[mode]:
        raise ValueError("public task mode/schema_key mismatch")
    schema_hash = str(schema_hashes.get(schema_key, "")).strip()
    if not schema_hash:
        raise ValueError(f"missing frozen schema hash for {schema_key}")
    return {
        "example_id": example_id,
        "mode": mode,
        "query": canonical_query(row.get("query", "")),
        "schema_sha256": schema_hash,
    }


def public_task_fingerprint(
    row: dict[str, Any], schema_hashes: dict[str, str]
) -> str:
    return sha256_text(canonical_json(_public_identity(row, schema_hashes)))


def _public_case_key(public_fingerprint: str, duplicate_ordinal: int) -> str:
    ordinal_digest = sha256_text(
        canonical_json(
            {
                "duplicate_ordinal": duplicate_ordinal,
                "public_fingerprint": public_fingerprint,
            }
        )
    )
    # Inference mixes the first eight hexadecimal characters into its seed.
    # Exact public duplicates therefore receive distinct opaque keys while
    # sharing the same public-fingerprint seed.
    return public_fingerprint[:8] + ordinal_digest[8:]


def build_public_tasks(
    frozen_rows: list[dict[str, Any]],
    schema_hashes: dict[str, str],
    *,
    allow_legacy_projection: bool = False,
) -> list[dict[str, Any]]:
    """Project a frozen public-task multiset without consulting evaluator targets."""
    multiplicities: dict[str, int] = defaultdict(int)
    identities: dict[str, dict[str, str]] = {}
    fingerprints: dict[str, str] = {}
    seen_fingerprints: dict[str, str] = {}

    for row in frozen_rows:
        fields = frozenset(row)
        if fields != TASK_FIELDS:
            if not allow_legacy_projection or fields != LEGACY_TASK_FIELDS:
                raise ValueError(
                    f"frozen public task field contract violation: {sorted(fields)}"
                )
        identity = _public_identity(row, schema_hashes)
        serialized = canonical_json(identity)
        fingerprint = sha256_text(serialized)
        prior = seen_fingerprints.setdefault(fingerprint, serialized)
        if prior != serialized:
            raise ValueError("public task fingerprint collision")
        multiplicities[serialized] += 1
        identities[serialized] = identity
        fingerprints[serialized] = fingerprint

    tasks: list[dict[str, Any]] = []
    for serialized in sorted(identities, key=lambda value: fingerprints[value]):
        identity = identities[serialized]
        fingerprint = fingerprints[serialized]
        for ordinal in range(multiplicities[serialized]):
            tasks.append(
                {
                    "case_key": _public_case_key(fingerprint, ordinal),
                    "example_id": identity["example_id"],
                    "mode": identity["mode"],
                    "query": identity["query"],
                    "schema_key": MODE_SCHEMA_KEYS[identity["mode"]],
                }
            )

    keys = [task["case_key"] for task in tasks]
    if len(keys) != len(set(keys)):
        raise ValueError("public task case-key collision")
    return tasks


def _single_queries(raw: Any) -> dict[str, str]:
    if isinstance(raw, dict):
        return {str(key): str(value) for key, value in raw.items() if isinstance(value, str)}
    result: dict[str, str] = {}
    if not isinstance(raw, list):
        return result
    for item in raw:
        if not isinstance(item, dict):
            continue
        domain = ""
        for target in item.get("target", []):
            if isinstance(target, dict) and target.get("domain"):
                domain = str(target["domain"])
                break
        if not domain:
            calls = extract_calls(item.get("api_call", []))
            domain = str(calls[0]["name"]) if calls else ""
        utterance = ""
        for turn in item.get("query", []):
            if not isinstance(turn, dict):
                continue
            if str(turn.get("role", "")).casefold() == "user":
                message = turn.get("message", turn.get("content"))
                if isinstance(message, str) and message.strip():
                    utterance = message.strip()
                    break
        if domain and utterance:
            result[domain] = utterance
    return result


def _multi_queries(raw: Any) -> dict[str, list[dict[str, Any]]]:
    if isinstance(raw, dict):
        return {
            str(key): value
            for key, value in raw.items()
            if isinstance(value, list) and all(isinstance(item, dict) for item in value)
        }
    result: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(raw, list):
        return result
    for item in raw:
        if not isinstance(item, dict):
            continue
        domain = ""
        for target in item.get("target", []):
            if isinstance(target, dict) and target.get("domain"):
                domain = str(target["domain"])
                break
        if not domain:
            calls = extract_calls(item.get("api_call", []))
            domain = str(calls[0]["name"]) if calls else ""
        if domain:
            result.setdefault(domain, []).append(item)
    return result


def _dialogue(template: dict[str, Any]) -> str:
    lines: list[str] = []
    for turn in template.get("query", []):
        if isinstance(turn, dict):
            role = str(turn.get("role", "User"))
            message = turn.get("message", turn.get("content", ""))
            if isinstance(message, str):
                lines.append(f"{role}: {message}")
    return "\n".join(lines)


def _product_calls(domain: str, slot_values: dict[str, list[Any]], base: dict[str, Any] | None = None) -> list[str]:
    base = dict(base or {})
    slots = sorted(slot_values)
    if not slots:
        return [call_to_string(domain, base)]
    result: list[str] = []
    for values in itertools.product(*(slot_values[slot] for slot in slots)):
        arguments = dict(base)
        arguments.update(zip(slots, values))
        result.append(call_to_string(domain, arguments))
    return result


def _single_materializer(query_map: dict[str, str]) -> Callable[[str, dict[str, list[Any]]], tuple[str, list[str]] | None]:
    def materialize(domain: str, slots: dict[str, list[Any]]) -> tuple[str, list[str]] | None:
        query = query_map.get(domain)
        return (query, _product_calls(domain, slots)) if query else None

    return materialize


def _multi_materializer(templates: dict[str, list[dict[str, Any]]]) -> Callable[[str, dict[str, list[Any]]], tuple[str, list[str]] | None]:
    def materialize(domain: str, slots: dict[str, list[Any]]) -> tuple[str, list[str]] | None:
        candidates = templates.get(domain, [])
        if not candidates:
            return None
        template = candidates[0]
        calls = extract_calls(template.get("api_call", []))
        base = dict(calls[0]["arguments"]) if calls else {}
        return _dialogue(template), _product_calls(domain, slots, base)

    return materialize


def _pairs(
    example: dict[str, Any],
    difficulty: str,
    preference_slots: dict[str, list[str]],
    preference_groups: dict[str, Any],
    materialize: Callable[[str, dict[str, list[Any]]], tuple[str, list[str]] | None],
) -> list[tuple[str, str, list[str]]]:
    result: list[tuple[str, str, list[str]]] = []
    if difficulty == "easy":
        for call in extract_calls(example.get("api_calls", [])):
            domain = str(call["name"])
            allowed = {str(slot) for slot in preference_slots.get(domain, [])}
            selected = {str(slot): [value] for slot, value in call["arguments"].items() if str(slot) in allowed}
            if selected:
                rendered = materialize(domain, selected)
                if rendered:
                    result.append((domain, rendered[0], rendered[1]))
        return result

    for preference in example.get("api_calls_pref", []):
        if not isinstance(preference, dict):
            continue
        group = preference_groups.get(preference.get("value_group"), {})
        rules = group.get("rules", []) if isinstance(group, dict) else []
        evidence = [item for item in preference.get("evidence", []) if isinstance(item, dict)]
        if difficulty == "medium":
            by_domain: dict[str, dict[str, list[Any]]] = {}
            for item in evidence:
                domain, slot = item.get("domain"), item.get("slot")
                if not domain or not slot:
                    continue
                values = sorted(
                    {str(rule.get("value")) for rule in rules if rule.get("domain") == domain and rule.get("slot") == slot and rule.get("value") is not None}
                )
                if values:
                    by_domain.setdefault(str(domain), {})[str(slot)] = values
            for domain in sorted(by_domain):
                rendered = materialize(domain, by_domain[domain])
                if rendered:
                    result.append((domain, rendered[0], rendered[1]))
        elif difficulty == "hard":
            used_domains = {str(item["domain"]) for item in evidence if item.get("domain")}
            candidate_domains = sorted(
                {str(rule["domain"]) for rule in rules if rule.get("domain") and str(rule["domain"]) not in used_domains}
            )
            for domain in candidate_domains:
                slots: dict[str, list[Any]] = {}
                for rule in rules:
                    if str(rule.get("domain", "")) != domain or not rule.get("slot") or rule.get("value") is None:
                        continue
                    slot = str(rule["slot"])
                    value = str(rule["value"])
                    if value not in slots.setdefault(slot, []):
                        slots[slot].append(value)
                for slot in slots:
                    slots[slot].sort()
                if slots:
                    rendered = materialize(domain, slots)
                    if rendered:
                        result.append((domain, rendered[0], rendered[1]))
    return result


def _target_variants(
    raw_examples: list[Any],
    preference_slots: dict[str, list[str]],
    preference_groups: dict[str, Any],
    materializers: dict[
        str,
        Callable[[str, dict[str, list[Any]]], tuple[str, list[str]] | None],
    ],
    schema_hashes: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """Build evaluator-only alternatives keyed by a public fingerprint."""
    variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for mode, materialize in materializers.items():
        for difficulty in ("easy", "medium", "hard"):
            for example in raw_examples:
                if not isinstance(example, dict) or not example.get("example_id"):
                    continue
                example_id = str(example["example_id"])
                for _domain, query, ground_truth in _pairs(
                    example,
                    difficulty,
                    preference_slots,
                    preference_groups,
                    materialize,
                ):
                    public_row = {
                        "example_id": example_id,
                        "mode": mode,
                        "query": query,
                        "schema_key": MODE_SCHEMA_KEYS[mode],
                    }
                    fingerprint = public_task_fingerprint(
                        public_row, schema_hashes
                    )
                    variants[fingerprint].append(
                        {
                            "difficulty": difficulty,
                            "reference_ground_truth": list(ground_truth),
                        }
                    )
    return variants


def _gold_rows_for_public_tasks(
    tasks: list[dict[str, Any]],
    variants: dict[str, list[dict[str, Any]]],
    schema_hashes: dict[str, str],
) -> tuple[list[dict[str, Any]], int]:
    """Assign evaluator alternatives only inside the vault-facing row set."""
    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        task_groups[public_task_fingerprint(task, schema_hashes)].append(task)

    assignments: dict[str, list[dict[str, Any]]] = {
        task["case_key"]: [] for task in tasks
    }
    unassigned = 0
    for fingerprint, target_rows in variants.items():
        public_tasks = task_groups.get(fingerprint, [])
        if not public_tasks:
            unassigned += len(target_rows)
            continue
        ordered_targets = sorted(target_rows, key=canonical_json)
        for index, target_row in enumerate(ordered_targets):
            task = public_tasks[index % len(public_tasks)]
            assignments[task["case_key"]].append(target_row)

    gold: list[dict[str, Any]] = []
    for task in tasks:
        assigned = assignments[task["case_key"]]
        references = sorted(
            {
                str(reference)
                for target_row in assigned
                for reference in target_row["reference_ground_truth"]
            }
        )
        difficulties = sorted(
            {str(target_row["difficulty"]) for target_row in assigned}
        )
        gold.append(
            {
                "case_key": task["case_key"],
                "example_id": task["example_id"],
                "mode": task["mode"],
                "difficulty": "|".join(difficulties)
                if difficulties
                else "evaluator_only_unmatched",
                "reference_ground_truth": references,
            }
        )
    return gold, unassigned


def build_task_gold_rows(
    *,
    frozen_task_rows: list[dict[str, Any]],
    raw_examples: list[Any],
    preference_slots: dict[str, list[str]],
    preference_groups: dict[str, Any],
    materializers: dict[
        str,
        Callable[[str, dict[str, list[Any]]], tuple[str, list[str]] | None],
    ],
    schema_hashes: dict[str, str],
    allow_legacy_projection: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    tasks = build_public_tasks(
        frozen_task_rows,
        schema_hashes,
        allow_legacy_projection=allow_legacy_projection,
    )
    variants = _target_variants(
        raw_examples,
        preference_slots,
        preference_groups,
        materializers,
        schema_hashes,
    )
    gold, unassigned = _gold_rows_for_public_tasks(
        tasks, variants, schema_hashes
    )
    return tasks, gold, unassigned


def prepare_bundle(root: str | Path, preregistration: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    prereg = load_json(preregistration)
    if prereg.get("status") != "locked_before_target_metrics":
        raise ValueError("preregistration is not locked")
    inputs = prereg["inputs"]
    for spec in inputs.values():
        verify_sha256(spec["path"], spec["sha256"])

    artifacts = root / "artifacts"
    configs = root / "configs"
    vault = root / "evaluator_vault"
    manifests = root / "manifests"
    for directory in (artifacts, configs, vault, manifests):
        directory.mkdir(parents=True, exist_ok=True)

    frozen_task_path = artifacts / "tasks.jsonl"
    if not frozen_task_path.is_file():
        raise ValueError(
            "frozen public-task multiset is required; target-derived task generation is forbidden"
        )
    frozen_task_rows = list(iter_jsonl(frozen_task_path))
    legacy_projection = any(
        frozenset(row) == LEGACY_TASK_FIELDS for row in frozen_task_rows
    )
    if legacy_projection:
        prior_seal_path = vault / "sealed_manifest.json"
        if not prior_seal_path.is_file():
            raise ValueError("legacy task projection requires its prior sealed manifest")
        prior_seal = load_json(prior_seal_path)
        if prior_seal.get("task_sha256") != sha256_file(frozen_task_path):
            raise ValueError("legacy frozen task multiset hash mismatch")

    local_names = {
        "single_query": "query_singleturn.json",
        "single_schema": "schema_single.json",
        "multi_query": "query_multiturn-domain.json",
        "multi_schema": "schema_multi.json",
        "preference_slots": "preference_slots.json",
        "preference_groups": "preference_groups.json",
    }
    for key, name in local_names.items():
        shutil.copyfile(inputs[key]["path"], configs / name)

    history_path = artifacts / "history.sanitized.jsonl"
    history_count = sanitize_dataset(
        inputs["history_data"]["path"], inputs["history_data"]["sha256"], history_path
    )

    raw_examples = load_json(inputs["history_data"]["path"])
    if isinstance(raw_examples, dict):
        raw_examples = raw_examples.get("dataset")
    if not isinstance(raw_examples, list):
        raise ValueError("dataset must be a list")
    preference_slots = load_json(configs / "preference_slots.json")
    preference_groups = load_json(configs / "preference_groups.json")
    single_query_map = _single_queries(load_json(configs / "query_singleturn.json"))
    multi_templates = _multi_queries(load_json(configs / "query_multiturn-domain.json"))
    modes = {
        "singleturn": _single_materializer(single_query_map),
        "multiturn": _multi_materializer(multi_templates),
    }
    schema_hashes = {
        "single": sha256_file(configs / "schema_single.json"),
        "multi": sha256_file(configs / "schema_multi.json"),
    }
    tasks, gold, unassigned_target_variants = build_task_gold_rows(
        frozen_task_rows=frozen_task_rows,
        raw_examples=raw_examples,
        preference_slots=preference_slots,
        preference_groups=preference_groups,
        materializers=modes,
        schema_hashes=schema_hashes,
        allow_legacy_projection=legacy_projection,
    )

    keys = [row["case_key"] for row in tasks]
    if len(keys) != len(set(keys)) or {row["case_key"] for row in gold} != set(keys):
        raise ValueError("task/gold key universe is not one-to-one")
    write_jsonl(artifacts / "tasks.jsonl", tasks)
    write_jsonl(vault / "gold.jsonl", gold)

    for mode, schema_name in (("singleturn", "schema_single.json"), ("multiturn", "schema_multi.json")):
        resources = {
            "examples": "artifacts/history.sanitized.jsonl",
            "memories": "artifacts/memory.ecpr.jsonl",
            "single_query_map": "configs/query_singleturn.json",
            "multiturn_templates": "configs/query_multiturn-domain.json",
            "preference_slots": "configs/preference_slots.json",
            "preference_groups": "configs/preference_groups.json",
            "tool_schema": f"configs/{schema_name}",
        }
        write_json(manifests / f"{mode}.bundle.json", {"schema_version": 1, "mode": mode, "resources": resources})

    sealed = {
        "schema_version": 1,
        "preregistration_sha256": sha256_file(preregistration),
        "task_sha256": sha256_file(artifacts / "tasks.jsonl"),
        "gold_sha256": sha256_file(vault / "gold.jsonl"),
        "history_sha256": sha256_file(history_path),
        "history_count": history_count,
        "case_count": len(tasks),
        "unassigned_target_variant_count": unassigned_target_variants,
    }
    write_json(vault / "sealed_manifest.json", sealed)
    return {"history_count": history_count, "case_count": len(tasks), "sealed_manifest": str(vault / "sealed_manifest.json")}
