"""Strict manifest loading and deterministic shared Step 2 helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    ContextMode,
    Difficulty,
    InputContractError,
    JoinError,
    ProviderPurpose,
    ProviderRequest,
    ProviderResponse,
    normalize_scalar_id,
)
from .prompts import build_inference_messages
from .providers import Provider, redact_diagnostic


_RESOURCE_NAMES = (
    "examples",
    "memories",
    "single_query_map",
    "multiturn_templates",
    "preference_slots",
    "preference_groups",
    "tool_schema",
)
_CALL_PATTERN = re.compile(
    r"\s*([A-Za-z_]\w*)\s*=\s*([\"'])([^\"']+)\2\s*(?:,|$)"
)
_CALL_ENVELOPE = re.compile(r"\s*([A-Za-z_]\w*)\s*\((.*)\)\s*", re.DOTALL)


@dataclass(frozen=True)
class MultiTemplate:
    template_id: str
    domain: str
    query: tuple[Mapping[str, Any], ...]
    api_calls: tuple[str, ...]
    target_slots: tuple[str, ...]
    api_argument_slots: tuple[str, ...]
    target_slot_paths: tuple[str, ...] = field(compare=False)
    domain_path: str = field(compare=False)


@dataclass(frozen=True)
class ManifestResources:
    examples: tuple[Mapping[str, Any], ...]
    memories: Mapping[str, Mapping[str, Any]]
    single_query_map: Mapping[str, str]
    multiturn_templates: Mapping[str, tuple[MultiTemplate, ...]]
    preference_slots: Mapping[str, tuple[str, ...]]
    preference_groups: Mapping[str, tuple[Mapping[str, Any], ...]]
    tool_schema: object


@dataclass(frozen=True)
class Step2Case:
    case_id: str
    example_id: str
    difficulty: Difficulty
    domain: str
    utterance: str
    ground_truth: tuple[str, ...]
    template_id: str | None = None
    set_derived_slots: tuple[str, ...] = ()


@dataclass(frozen=True)
class Step2Run:
    results: tuple[Mapping[str, Any], ...]
    diagnostics: tuple[Mapping[str, Any], ...]


def load_manifest_v1(
    manifest_path: str | Path, *, multi_shape: str = "auto"
) -> ManifestResources:
    """Load all manifest-v1 resources relative to the manifest directory."""

    manifest_file = Path(manifest_path)
    manifest = _read_json(manifest_file, "manifest")
    if not isinstance(manifest, dict):
        raise InputContractError("must be an object", path="manifest")
    if type(manifest.get("version")) is not int or manifest["version"] != 1:
        raise InputContractError("must equal 1", path="manifest.version")
    base = manifest_file.expanduser().resolve(strict=False).parent
    raw: dict[str, object] = {}
    for name in _RESOURCE_NAMES:
        value = manifest.get(name)
        if not isinstance(value, str) or not value.strip():
            raise InputContractError("must be a nonempty relative path", path=f"manifest.{name}")
        resource_path = Path(value)
        if resource_path.is_absolute():
            raise InputContractError("must be relative", path=f"manifest.{name}")
        resolved = (base / resource_path).resolve(strict=False)
        try:
            resolved.relative_to(base)
        except ValueError:
            raise InputContractError("must remain beneath the manifest directory", path=f"manifest.{name}") from None
        raw[name] = _read_jsonl(resolved, f"resources.{name}") if name == "memories" else _read_json(resolved, f"resources.{name}")

    examples = _parse_examples(raw["examples"])
    memories = _parse_memories(raw["memories"])
    single_queries = _parse_query_map(raw["single_query_map"])
    slots = _parse_slots(raw["preference_slots"])
    groups = _parse_groups(raw["preference_groups"])
    templates = normalize_multiturn_templates(raw["multiturn_templates"], shape=multi_shape)
    tool_schema = raw["tool_schema"]
    if not isinstance(tool_schema, (dict, list)) or not tool_schema:
        raise JoinError("must be a nonempty object or list", path="resources.tool_schema")
    _validate_joins(examples, memories, single_queries, templates, slots, groups)
    return ManifestResources(examples, memories, single_queries, templates, slots, groups, tool_schema)


def normalize_context(value: str | ContextMode) -> ContextMode:
    if isinstance(value, ContextMode):
        return value
    try:
        return ContextMode(value)
    except (ValueError, TypeError):
        raise InputContractError(
            "must be one of memory_only, memory_api, memory_diag, api_only", path="context"
        ) from None


def normalize_difficulties(value: str | Difficulty) -> tuple[Difficulty, ...]:
    if value == "all":
        return (Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD)
    if isinstance(value, Difficulty):
        return (value,)
    try:
        return (Difficulty(value),)
    except (ValueError, TypeError):
        raise InputContractError("must be one of easy, medium, hard, all", path="difficulty") from None


def normalize_multiturn_templates(
    value: object, *, shape: str = "auto"
) -> Mapping[str, tuple[MultiTemplate, ...]]:
    """Normalize historical grouped and list resources without reordering them."""

    if shape not in {"auto", "grouped", "list"}:
        raise InputContractError("must be one of auto, grouped, list", path="input_shape")
    actual = "grouped" if isinstance(value, dict) else "list" if isinstance(value, list) else None
    if actual is None or (shape != "auto" and shape != actual):
        raise InputContractError(f"must have {shape if shape != 'auto' else 'grouped or list'} shape", path="resources.multiturn_templates")
    grouped: dict[str, list[MultiTemplate]] = {}
    seen_ids: set[str] = set()
    if actual == "grouped":
        normalized_groups: list[tuple[str, object]] = []
        seen_domains: set[str] = set()
        for raw_domain, raw_items in value.items():  # type: ignore[union-attr]
            domain = _nonempty_string(
                raw_domain, "resources.multiturn_templates.domain"
            )
            if domain in seen_domains:
                raise JoinError(
                    "normalizes to an earlier mapping key",
                    path=_mapping_key_path(
                        "resources.multiturn_templates", raw_domain
                    ),
                )
            seen_domains.add(domain)
            normalized_groups.append((domain, raw_items))
        grouped_items: Iterable[tuple[object | None, int, object]] = (
            (domain, index, item)
            for domain, raw_items in normalized_groups
            for index, item in _group_items(domain, raw_items)
        )
    else:
        grouped_items = (
            (None, index, item) for index, item in enumerate(value)  # type: ignore[arg-type]
        )
    for raw_domain, index, item in grouped_items:
        if raw_domain is None:
            path = f"resources.multiturn_templates[{index}]"
            grouped_domain = None
        else:
            grouped_domain = _nonempty_string(
                raw_domain, "resources.multiturn_templates.domain"
            )
            path = f"resources.multiturn_templates.{grouped_domain}[{index}]"
        if not isinstance(item, dict):
            raise InputContractError("must be an object", path=path)
        target = item.get("target", [])
        if not isinstance(target, list):
            raise InputContractError("must be a list", path=f"{path}.target")
        target_domains: list[str] = []
        target_slots: list[str] = []
        target_slot_paths: list[str] = []
        for target_index, entry in enumerate(target):
            target_path = f"{path}.target[{target_index}]"
            if not isinstance(entry, dict):
                raise InputContractError("must be an object", path=target_path)
            target_domain = _nonempty_string(
                entry.get("domain"), f"{target_path}.domain"
            )
            target_slot = _nonempty_string(
                entry.get("slot"), f"{target_path}.slot"
            )
            target_domains.append(target_domain)
            target_slots.append(target_slot)
            target_slot_paths.append(f"{target_path}.slot")
        calls = item.get("api_call")
        if (
            not isinstance(calls, list)
            or not calls
            or not all(isinstance(call, str) for call in calls)
        ):
            raise InputContractError(
                "must be a nonempty string list", path=f"{path}.api_call"
            )
        parsed_calls = tuple(
            parse_api_call(call, path=f"{path}.api_call[{call_index}]")
            for call_index, call in enumerate(calls)
        )
        call_domain = parsed_calls[0][0]
        for call_index, (current_domain, _arguments) in enumerate(parsed_calls):
            if current_domain != call_domain:
                raise JoinError(
                    "API call domain disagrees with the first call",
                    path=f"{path}.api_call[{call_index}]",
                )
        domain = grouped_domain or (target_domains[0] if target_domains else call_domain)
        domain_path = (
            f"{path}.target[0].domain" if target_domains else f"{path}.api_call[0]"
        )
        if grouped_domain is not None and domain != grouped_domain:
            raise JoinError(
                "domain disagrees with grouped key", path=domain_path
            )
        for target_index, target_domain in enumerate(target_domains):
            if target_domain != domain:
                raise JoinError(
                    "target domain disagrees with template domain",
                    path=f"{path}.target[{target_index}].domain",
                )
        if call_domain != domain:
            raise JoinError(
                "API call domain disagrees with template domain",
                path=(
                    f"{path}.target[0].domain"
                    if target_domains
                    else f"{path}.api_call[0]"
                ),
            )
        domain_index = len(grouped.get(domain, ()))
        template_id = normalize_scalar_id(
            item.get("query_id", f"{domain}:{domain_index}"),
            path=f"{path}.query_id",
        )
        if template_id in seen_ids:
            raise JoinError("duplicates an earlier normalized identifier", path=f"{path}.query_id")
        seen_ids.add(template_id)
        query = item.get("query")
        if not isinstance(query, list) or not query:
            raise InputContractError("must be a nonempty list", path=f"{path}.query")
        turns: list[Mapping[str, Any]] = []
        for turn_index, turn in enumerate(query):
            if not isinstance(turn, dict) or not isinstance(turn.get("message"), str):
                raise InputContractError("must contain a string message", path=f"{path}.query[{turn_index}]")
            turns.append(dict(turn))
        argument_slots: list[str] = []
        for _current_domain, arguments in parsed_calls:
            for argument in arguments:
                if argument not in argument_slots:
                    argument_slots.append(argument)
        grouped.setdefault(domain, []).append(
            MultiTemplate(
                template_id,
                domain,
                tuple(turns),
                tuple(calls),
                tuple(target_slots),
                tuple(argument_slots),
                tuple(target_slot_paths),
                domain_path,
            )
        )
    if not grouped:
        raise InputContractError("must contain at least one template", path="resources.multiturn_templates")
    return {domain: tuple(items) for domain, items in grouped.items()}


def make_provider_request(
    *, resources: ManifestResources, case: Step2Case, context: str | ContextMode, model: str
) -> ProviderRequest:
    mode = normalize_context(context)
    example = next(item for item in resources.examples if item["example_id"] == case.example_id)
    memory = resources.memories[case.example_id]
    messages = build_inference_messages(
        context_mode=mode,
        multi_turn=case.template_id is not None,
        current_utterance=case.utterance,
        tool_schema=resources.tool_schema,
        preference=memory.get("final_implicit_preference"),
        api_history=tuple(memory["final_accumulated_api_calls"]),
        prior_dialogue=_format_prior_dialogue(example),
    )
    return ProviderRequest(
        purpose=ProviderPurpose.INFER,
        model=model,
        messages=messages,
        prompt=messages[-1].content,
        temperature=None,
        json_intent=False,
    )


def project_rows(
    *,
    case: Step2Case,
    mode: str,
    context: ContextMode,
    request: ProviderRequest,
    response: ProviderResponse,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    result = {
        "record_type": "step2_result",
        "case_id": case.case_id,
        "example_id": case.example_id,
        "mode": mode,
        "difficulty": case.difficulty.value,
        "context": context.value,
        "memory_id": case.example_id,
        "domain": case.domain,
        "template_id": case.template_id,
        "utterance": case.utterance,
        "reference_ground_truth": list(case.ground_truth),
        "inferred_action": response.text,
    }
    diagnostic = {
        "record_type": "step2_diagnostic",
        "case_id": case.case_id,
        "example_id": case.example_id,
        "mode": mode,
        "difficulty": case.difficulty.value,
        "context": context.value,
        "memory_id": case.example_id,
        "template_id": case.template_id,
        "set_derived_slots": list(case.set_derived_slots),
        "provider": redact_diagnostic(
            {
                "request": {
                    "model": request.model,
                    "purpose": request.purpose.value,
                    "messages": [
                        {"role": message.role, "content": message.content}
                        for message in request.messages
                    ],
                    "prompt": request.prompt,
                    "temperature": {
                        "present": request.temperature is not None,
                        "value": request.temperature,
                    },
                    "json_intent": request.json_intent,
                },
                "request_id": response.request_id,
                "usage": response.usage,
                "raw_response": response.raw_response,
            }
        ),
    }
    return result, diagnostic


def _execute_step2(
    *,
    resources: ManifestResources,
    cases: tuple[Step2Case, ...],
    context: ContextMode,
    model: str,
    provider: Provider,
    mode: str,
) -> Step2Run:
    results, diagnostics = [], []
    for case in cases:
        request = make_provider_request(
            resources=resources, case=case, context=context, model=model
        )
        response = provider.complete(request)
        result, diagnostic = project_rows(
            case=case,
            mode=mode,
            context=context,
            request=request,
            response=response,
        )
        results.append(result)
        diagnostics.append(diagnostic)
    return Step2Run(tuple(results), tuple(diagnostics))


def parse_api_call(
    value: str, *, path: str = "api_call"
) -> tuple[str, dict[str, str]]:
    if not isinstance(value, str):
        raise InputContractError("must be a string", path=path)
    match = _CALL_ENVELOPE.fullmatch(value)
    if match is None:
        raise InputContractError("must be a typed API call", path=path)
    domain, content = match.groups()
    arguments: dict[str, str] = {}
    position = 0
    while position < len(content):
        argument = _CALL_PATTERN.match(content, position)
        if argument is None:
            raise InputContractError("contains malformed arguments", path=path)
        key, _quote, item = argument.groups()
        if key in arguments:
            raise InputContractError("contains duplicate argument keys", path=path)
        arguments[key] = item
        position = argument.end()
    return domain, arguments


def generate_calls(
    domain: str, base: Mapping[str, str], preferences: Mapping[str, Sequence[str]]
) -> tuple[str, ...]:
    keys = sorted(preferences, key=str)
    combinations = itertools.product(*(preferences[key] for key in keys)) if keys else [()]
    results: list[str] = []
    for combination in combinations:
        arguments = dict(base)
        arguments.update(zip(keys, combination))
        args = ", ".join(f'{key}="{arguments[key]}"' for key in sorted(arguments, key=str))
        results.append(f"{domain}({args})")
    return tuple(results)


def preference_maps(
    resources: ManifestResources, example: Mapping[str, Any], difficulty: Difficulty
) -> tuple[tuple[str, Mapping[str, tuple[str, ...]], tuple[str, ...]], ...]:
    """Return source-ordered maps; callers concatenate easy, medium, then hard.

    Only source-set-derived medium values/slots and hard domains are sorted.
    API calls, preference records, evidence, and group-rule values retain input order.
    """

    rows: list[tuple[str, Mapping[str, tuple[str, ...]], tuple[str, ...]]] = []
    if difficulty is Difficulty.EASY:
        for call in example["api_calls"]:
            domain, arguments = parse_api_call(call)
            selected = {
                slot: (value,)
                for slot, value in arguments.items()
                if slot in resources.preference_slots[domain]
            }
            if selected:
                rows.append((domain, selected, ()))
        return tuple(rows)
    for pref in example["api_calls_pref"]:
        rules = resources.preference_groups[pref["value_group"]]
        if difficulty is Difficulty.MEDIUM:
            by_domain: dict[str, dict[str, set[str]]] = {}
            for evidence in pref["evidence"]:
                domain, slot = evidence["domain"], evidence["slot"]
                values = {
                    _source_string(rule.get("value"))
                    for rule in rules
                    if rule["domain"] == domain and rule["slot"] == slot
                }
                if values:
                    by_domain.setdefault(domain, {}).setdefault(slot, set()).update(values)
            for domain, slot_map in by_domain.items():
                serialized = {
                    slot: tuple(sorted(values, key=str)) for slot, values in slot_map.items()
                }
                rows.append((domain, serialized, tuple(sorted(slot_map, key=str))))
        else:
            used = {entry["domain"] for entry in pref["evidence"]}
            candidate_domains = sorted(
                {rule["domain"] for rule in rules if rule["domain"] not in used}, key=str
            )
            for domain in candidate_domains:
                slot_map: dict[str, list[str]] = {}
                for rule in rules:
                    if rule["domain"] == domain:
                        if rule.get("value") is None:
                            continue
                        value = _source_string(rule.get("value"))
                        if value not in slot_map.setdefault(rule["slot"], []):
                            slot_map[rule["slot"]].append(value)
                if slot_map:
                    rows.append((domain, {key: tuple(values) for key, values in slot_map.items()}, ()))
    return tuple(rows)


def _parse_examples(value: object) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise InputContractError("must be a nonempty list", path="resources.examples")
    seen: set[str] = set()
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(value):
        path = f"resources.examples[{index}]"
        if not isinstance(item, dict):
            raise InputContractError("must be an object", path=path)
        identifier = normalize_scalar_id(item.get("example_id"), path=f"{path}.example_id")
        if identifier in seen:
            raise JoinError("duplicates an earlier normalized identifier", path=f"{path}.example_id")
        seen.add(identifier)
        api_calls = item.get("api_calls", [])
        prefs = item.get("api_calls_pref", [])
        sessions = item.get("sessions", [])
        if not isinstance(api_calls, list):
            raise InputContractError("must be a list", path=f"{path}.api_calls")
        for call_index, call in enumerate(api_calls):
            parse_api_call(call, path=f"{path}.api_calls[{call_index}]")
        if not isinstance(prefs, list) or not all(isinstance(pref, dict) for pref in prefs):
            raise InputContractError("must be an object list", path=f"{path}.api_calls_pref")
        if not isinstance(sessions, list):
            raise InputContractError("must be a list", path=f"{path}.sessions")
        normalized_sessions = []
        for session_index, session in enumerate(sessions):
            session_path = f"{path}.sessions[{session_index}]"
            if not isinstance(session, dict):
                raise InputContractError("must be an object", path=session_path)
            dialogue = session.get("dialogue")
            if not isinstance(dialogue, list):
                raise InputContractError("must be a list", path=f"{session_path}.dialogue")
            normalized_turns = []
            for turn_index, turn in enumerate(dialogue):
                turn_path = f"{session_path}.dialogue[{turn_index}]"
                if not isinstance(turn, dict):
                    raise InputContractError("must be an object", path=turn_path)
                role = _nonempty_string(turn.get("role"), f"{turn_path}.role")
                message = _nonempty_string(turn.get("message"), f"{turn_path}.message")
                normalized_turns.append({"role": role, "message": message})
            normalized_session = dict(session)
            normalized_session["dialogue"] = normalized_turns
            normalized_sessions.append(normalized_session)
        normalized_prefs = []
        for pref_index, pref in enumerate(prefs):
            pref_path = f"{path}.api_calls_pref[{pref_index}]"
            group = _nonempty_string(pref.get("value_group"), f"{pref_path}.value_group")
            evidence = pref.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                raise InputContractError("must be a nonempty list", path=f"{pref_path}.evidence")
            normalized_evidence = []
            for evidence_index, entry in enumerate(evidence):
                evidence_path = f"{pref_path}.evidence[{evidence_index}]"
                if not isinstance(entry, dict):
                    raise InputContractError("must be an object", path=evidence_path)
                normalized_evidence.append({
                    "domain": _nonempty_string(entry.get("domain"), f"{evidence_path}.domain"),
                    "slot": _nonempty_string(entry.get("slot"), f"{evidence_path}.slot"),
                })
            normalized_prefs.append({"value_group": group, "evidence": normalized_evidence})
        normalized = dict(item)
        normalized.update(
            example_id=identifier,
            api_calls=list(api_calls),
            api_calls_pref=normalized_prefs,
            sessions=normalized_sessions,
        )
        result.append(normalized)
    return tuple(result)


def _parse_memories(value: object) -> Mapping[str, Mapping[str, Any]]:
    if not isinstance(value, list) or not value:
        raise InputContractError("must contain at least one row", path="resources.memories")
    result: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(value):
        path = f"resources.memories.line[{index + 1}]"
        if not isinstance(item, dict):
            raise InputContractError("must be an object", path=path)
        identifier = normalize_scalar_id(item.get("example_id"), path=f"{path}.example_id")
        if identifier in result:
            raise JoinError("duplicates an earlier normalized identifier", path=f"{path}.example_id")
        if "final_implicit_preference" not in item:
            raise JoinError("is required", path=f"{path}.final_implicit_preference")
        history = item.get("final_accumulated_api_calls")
        if not isinstance(history, list) or not all(isinstance(call, str) for call in history):
            raise JoinError("must be a string list", path=f"{path}.final_accumulated_api_calls")
        result[identifier] = dict(item, example_id=identifier)
    return result


def _parse_query_map(value: object) -> Mapping[str, str]:
    if not isinstance(value, dict) or not value:
        raise InputContractError("must be a nonempty object", path="resources.single_query_map")
    result: dict[str, str] = {}
    for domain, utterance in value.items():
        key = _nonempty_string(domain, "resources.single_query_map.domain")
        if key in result:
            raise JoinError(
                "normalizes to an earlier mapping key",
                path=_mapping_key_path("resources.single_query_map", domain),
            )
        result[key] = _nonempty_string(utterance, f"resources.single_query_map.{key}")
    return result


def _parse_slots(value: object) -> Mapping[str, tuple[str, ...]]:
    if not isinstance(value, dict) or not value:
        raise InputContractError("must be a nonempty object", path="resources.preference_slots")
    result = {}
    for domain, slots in value.items():
        key = _nonempty_string(domain, "resources.preference_slots.domain")
        if key in result:
            raise JoinError(
                "normalizes to an earlier mapping key",
                path=_mapping_key_path("resources.preference_slots", domain),
            )
        if not isinstance(slots, list) or not slots:
            raise InputContractError("must be a nonempty list", path=f"resources.preference_slots.{key}")
        result[key] = tuple(_nonempty_string(slot, f"resources.preference_slots.{key}") for slot in slots)
    return result


def _parse_groups(value: object) -> Mapping[str, tuple[Mapping[str, Any], ...]]:
    if not isinstance(value, dict) or not value:
        raise InputContractError("must be a nonempty object", path="resources.preference_groups")
    result = {}
    for name, group in value.items():
        key = _nonempty_string(name, "resources.preference_groups.name")
        if key in result:
            raise JoinError(
                "normalizes to an earlier mapping key",
                path=_mapping_key_path("resources.preference_groups", name),
            )
        if not isinstance(group, dict) or not isinstance(group.get("rules"), list) or not group["rules"]:
            raise InputContractError("must contain nonempty rules", path=f"resources.preference_groups.{key}.rules")
        rules = []
        for index, rule in enumerate(group["rules"]):
            path = f"resources.preference_groups.{key}.rules[{index}]"
            if not isinstance(rule, dict) or "value" not in rule:
                raise InputContractError("must contain domain, slot, and value", path=path)
            rules.append({
                "domain": _nonempty_string(rule.get("domain"), f"{path}.domain"),
                "slot": _nonempty_string(rule.get("slot"), f"{path}.slot"),
                "value": rule["value"],
            })
        result[key] = tuple(rules)
    return result


def _validate_joins(examples, memories, queries, templates, slots, groups) -> None:
    example_ids = {item["example_id"] for item in examples}
    memory_ids = set(memories)
    missing = example_ids - memory_ids
    if missing:
        identifier = next(item["example_id"] for item in examples if item["example_id"] in missing)
        raise JoinError("has no matching memory", path=f"joins.examples[{identifier}].memory")
    orphan = memory_ids - example_ids
    if orphan:
        identifier = next(item for item in memories if item in orphan)
        raise JoinError("has no matching example", path=f"joins.memories[{identifier}].example")
    for domain in templates:
        for template in templates[domain]:
            _require_domain(
                domain, queries, templates, slots, path=template.domain_path
            )
            known_slots = set(slots[domain]).union(template.api_argument_slots)
            for slot, slot_path in zip(
                template.target_slots, template.target_slot_paths
            ):
                if slot not in known_slots:
                    raise JoinError("referenced slot is missing", path=slot_path)
    for example in examples:
        for call in example["api_calls"]:
            domain, _ = parse_api_call(call)
            _require_domain(domain, queries, templates, slots, path=f"joins.domains.{domain}")
        for pref in example["api_calls_pref"]:
            group = pref["value_group"]
            if group not in groups:
                raise JoinError("referenced group is missing", path=f"joins.preference_groups.{group}")
            for evidence in pref["evidence"]:
                domain, slot = evidence["domain"], evidence["slot"]
                _require_domain(domain, queries, templates, slots, path=f"joins.domains.{domain}")
                if slot not in slots[domain]:
                    raise JoinError("referenced slot is missing", path=f"joins.preference_slots.{domain}.{slot}")
            for rule in groups[group]:
                domain, slot = rule["domain"], rule["slot"]
                _require_domain(domain, queries, templates, slots, path=f"joins.domains.{domain}")
                if slot not in slots[domain]:
                    raise JoinError("referenced slot is missing", path=f"joins.preference_slots.{domain}.{slot}")


def _require_domain(domain, queries, templates, slots, *, path: str) -> None:
    if domain not in queries:
        raise JoinError("single query is missing", path=path)
    if domain not in templates:
        raise JoinError("multi-turn template is missing", path=path)
    if domain not in slots:
        raise JoinError("preference slot schema is missing", path=path)


def _group_items(raw_domain: object, raw_items: object) -> Iterable[tuple[int, object]]:
    domain = _nonempty_string(raw_domain, "resources.multiturn_templates.domain")
    if not isinstance(raw_items, list):
        raise InputContractError(
            "must be a list", path=f"resources.multiturn_templates.{domain}"
        )
    return enumerate(raw_items)


def _read_json(path: Path, logical_path: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputContractError("cannot be read as UTF-8 JSON", path=logical_path) from exc


def _read_jsonl(path: Path, logical_path: str) -> object:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise InputContractError("cannot be read as UTF-8 JSONL", path=logical_path) from exc
    rows = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise InputContractError("is not valid JSON", path=f"{logical_path}.line[{index}]") from exc
    return rows


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputContractError("must be a nonempty string", path=path)
    return value.strip()


def _mapping_key_path(base: str, raw_key: object) -> str:
    return f"{base}[{json.dumps(str(raw_key), ensure_ascii=False)}]"


def _source_string(value: object) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _format_prior_dialogue(example: Mapping[str, Any]) -> str:
    sessions = []
    for index, session in enumerate(example.get("sessions", []), start=1):
        lines = [f"[Session {index}]"]
        for turn in session.get("dialogue", []):
            role = str(turn.get("role", "")).capitalize()
            message = turn.get("message", "")
            if role and message:
                lines.append(f"{role}: {message}")
        sessions.append("\n".join(lines))
    return "\n\n".join(sessions)
