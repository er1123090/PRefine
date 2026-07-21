from __future__ import annotations

import argparse
import copy
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from core import (
    ContractError,
    DIFFICULTIES,
    MODEL_ID,
    MODEL_REVISION,
    MODES,
    build_request,
    calls_from_slot_values,
    canonical_json,
    content_id,
    current_dirty_snapshot,
    flatten_multiturn,
    load_json,
    normalize_arguments,
    normalize_scalar,
    parse_call_string,
    render_prompt,
    schema_map,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)


EXPECTED = {"easy": 554, "medium": 293, "hard": 472}


def _load_prompt(path: Path) -> str:
    namespace: dict[str, Any] = {}
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
    prompt = namespace.get("IMPLICIT_ZS_PROMPT_TEMPLATE")
    if not isinstance(prompt, str):
        raise ContractError(f"prompt template missing in {path}")
    return prompt


def _group_multi_templates(raw: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for template in raw:
        query_id = template.get("query_id")
        if not isinstance(query_id, str) or not query_id or query_id in seen_ids:
            raise ContractError(f"invalid or duplicate multi query_id: {query_id!r}")
        seen_ids.add(query_id)
        domains = {target.get("domain") for target in template.get("target") or [] if target.get("domain")}
        if len(domains) != 1:
            raise ContractError(f"multi template {query_id} must have one target domain")
        domain = next(iter(domains))
        result[domain].append(dict(template))
    for domain in result:
        result[domain].sort(key=lambda item: item["query_id"])
    return dict(result)


def _derive_schema_overlay(
    tools: Sequence[Mapping[str, Any]],
    raw_templates: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Add only explicit multi-turn base arguments to a run-local tool schema.

    The checked-in schema is preference-pruned, while the multi-turn templates
    contain transient request slots and values. The original schema remains an
    immutable hashed input; this deterministic overlay is recorded and applied
    to both modes so paired tool declarations stay identical.
    """
    derived = copy.deepcopy(list(tools))
    functions = schema_map(derived)
    operations: list[dict[str, Any]] = []
    for template in sorted(raw_templates, key=lambda item: str(item.get("query_id", ""))):
        query_id = str(template.get("query_id", ""))
        for raw_call in template.get("api_call") or []:
            domain, arguments = parse_call_string(raw_call)
            if domain not in functions:
                raise ContractError(f"template {query_id} references unknown function {domain}")
            parameters = functions[domain].setdefault("parameters", {"type": "object", "properties": {}})
            properties = parameters.setdefault("properties", {})
            for slot, raw_value in sorted(arguments.items()):
                if slot not in properties:
                    if isinstance(raw_value, bool):
                        inferred_type = "boolean"
                    elif isinstance(raw_value, int):
                        inferred_type = "integer"
                    elif isinstance(raw_value, float):
                        inferred_type = "number"
                    else:
                        inferred_type = "string"
                    properties[slot] = {"type": inferred_type}
                    operations.append({
                        "action": "add_explicit_property",
                        "domain": domain,
                        "query_id": query_id,
                        "slot": slot,
                        "schema": {"type": inferred_type},
                    })
                    continue
                spec = properties[slot]
                normalized = normalize_scalar(raw_value, {key: value for key, value in spec.items() if key != "enum"})
                if "enum" in spec:
                    allowed = [normalize_scalar(value, {"type": spec.get("type", "string")}) for value in spec["enum"]]
                    if canonical_json(normalized) not in {canonical_json(value) for value in allowed}:
                        spec["enum"] = sorted([*allowed, normalized], key=canonical_json)
                        operations.append({
                            "action": "extend_explicit_enum",
                            "domain": domain,
                            "query_id": query_id,
                            "slot": slot,
                            "value": normalized,
                        })
    operations.sort(key=canonical_json)
    return derived, operations


def _template_slots(template: Mapping[str, Any]) -> set[str]:
    return {str(target["slot"]) for target in template.get("target") or [] if target.get("slot")}


def _select_template(templates: Sequence[Mapping[str, Any]], preference_slots: set[str]) -> dict[str, Any]:
    scored = [(len(preference_slots & _template_slots(item)), str(item["query_id"]), dict(item)) for item in templates]
    positive = [item for item in scored if item[0] > 0]
    if not positive:
        raise ContractError(f"no positive-overlap multi template for slots {sorted(preference_slots)}")
    best_score = max(item[0] for item in positive)
    return min((item for item in positive if item[0] == best_score), key=lambda item: item[1])[2]


def _target_payload(
    example: Mapping[str, Any],
    source_index: int,
    difficulty: str,
    domain: str,
    slot_values: Mapping[str, Sequence[Any]],
    schemas: Mapping[str, Mapping[str, Any]],
    preference_group: str | None,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    gt = calls_from_slot_values(domain, slot_values, schemas)
    identity = {
        "source_example_id": example["example_id"],
        "difficulty": difficulty,
        "target_domain": domain,
        "reference_ground_truth_preference": gt,
        "preference_group": preference_group,
        "provenance": provenance,
    }
    return {
        "target_id": content_id("target", identity),
        "pair_id": content_id("pair", identity),
        "source_example_id": example["example_id"],
        "source_index": source_index,
        "difficulty": difficulty,
        "target_domain": domain,
        "preference_group": preference_group,
        "preference_slot_values": {
            call["name"]: call["arguments"] for call in gt[:1]
        },
        "reference_ground_truth_preference": gt,
        "provenance": provenance,
    }


def derive_targets(
    dataset: Sequence[Mapping[str, Any]],
    single_queries: Mapping[str, str],
    multi_templates: Mapping[str, Sequence[Mapping[str, Any]]],
    pref_list: Mapping[str, Sequence[str]],
    pref_groups: Mapping[str, Mapping[str, Any]],
    schemas: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    common_domains = set(single_queries) & set(multi_templates) & set(schemas)
    targets: list[dict[str, Any]] = []
    rejections: list[dict[str, Any]] = []
    for source_index, example in enumerate(dataset):
        example_id = example.get("example_id")
        if not isinstance(example_id, str) or not example_id:
            raise ContractError(f"source index {source_index} has no example_id")
        for difficulty in DIFFICULTIES:
            before = len(targets)
            if difficulty == "easy":
                for raw_call in example.get("api_calls") or []:
                    try:
                        domain, raw_arguments = parse_call_string(raw_call)
                        if domain not in common_domains or domain not in pref_list:
                            continue
                        preference_raw = {
                            slot: raw_arguments[slot]
                            for slot in sorted(pref_list[domain])
                            if slot in raw_arguments
                        }
                        if not preference_raw:
                            continue
                        preference_arguments = normalize_arguments(domain, preference_raw, schemas)
                        if not preference_arguments:
                            continue
                        targets.append(
                            _target_payload(
                                example,
                                source_index,
                                difficulty,
                                domain,
                                {slot: [value] for slot, value in preference_arguments.items()},
                                schemas,
                                None,
                                {"observed_api_call": raw_call},
                            )
                        )
                    except ContractError as exc:
                        rejections.append({
                            "source_example_id": example_id,
                            "difficulty": difficulty,
                            "reason": "invalid_observed_api_call",
                            "detail": str(exc),
                            "raw": raw_call,
                        })
            else:
                preferences = sorted(example.get("api_calls_pref") or [], key=canonical_json)
                for preference in preferences:
                    group_name = preference.get("value_group")
                    if group_name not in pref_groups:
                        continue
                    rules = pref_groups[group_name].get("rules") or []
                    if difficulty == "medium":
                        domain_slots: dict[str, set[str]] = defaultdict(set)
                        for evidence in preference.get("evidence") or []:
                            domain = evidence.get("domain")
                            slot = evidence.get("slot")
                            if domain in common_domains and slot:
                                domain_slots[str(domain)].add(str(slot))
                        for domain in sorted(domain_slots):
                            slot_values: dict[str, list[Any]] = {}
                            for slot in sorted(domain_slots[domain]):
                                values = [
                                    rule.get("value")
                                    for rule in rules
                                    if rule.get("domain") == domain and rule.get("slot") == slot
                                ]
                                if values:
                                    slot_values[slot] = values
                            if slot_values:
                                targets.append(
                                    _target_payload(
                                        example,
                                        source_index,
                                        difficulty,
                                        domain,
                                        slot_values,
                                        schemas,
                                        str(group_name),
                                        {"preference": preference},
                                    )
                                )
                    else:
                        used_domains = {
                            evidence.get("domain") for evidence in preference.get("evidence") or [] if evidence.get("domain")
                        }
                        candidate_domains = sorted(
                            {
                                str(rule["domain"])
                                for rule in rules
                                if rule.get("domain") in common_domains and rule.get("domain") not in used_domains
                            }
                        )
                        for domain in candidate_domains:
                            slot_values: dict[str, list[Any]] = defaultdict(list)
                            for rule in sorted(rules, key=canonical_json):
                                if rule.get("domain") == domain and rule.get("slot") and rule.get("value") is not None:
                                    slot_values[str(rule["slot"])].append(rule["value"])
                            if slot_values:
                                targets.append(
                                    _target_payload(
                                        example,
                                        source_index,
                                        difficulty,
                                        domain,
                                        dict(slot_values),
                                        schemas,
                                        str(group_name),
                                        {"preference": preference, "used_domains": sorted(used_domains)},
                                    )
                                )
            if len(targets) == before:
                rejections.append({
                    "source_example_id": example_id,
                    "difficulty": difficulty,
                    "reason": "no_applicable_target",
                })
    targets.sort(key=lambda item: item["target_id"])
    ids = [item["target_id"] for item in targets]
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise ContractError(f"duplicate content-derived target IDs: {duplicates[:5]}")
    return targets, sorted(rejections, key=canonical_json)


def render_cases(
    targets: Sequence[Mapping[str, Any]],
    dataset_by_id: Mapping[str, Mapping[str, Any]],
    single_queries: Mapping[str, str],
    multi_templates: Mapping[str, Sequence[Mapping[str, Any]]],
    tools: Sequence[Mapping[str, Any]],
    schemas: Mapping[str, Mapping[str, Any]],
    prompt_template: str,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for target in targets:
        example = dataset_by_id[target["source_example_id"]]
        domain = target["target_domain"]
        preference_gt = target["reference_ground_truth_preference"]
        preference_slots = set().union(*(call["arguments"].keys() for call in preference_gt))
        for mode in MODES:
            explicit_base: dict[str, Any] = {}
            if mode == "single":
                query_id = f"single:{domain}"
                user_utterance = single_queries[domain]
                full_gt = preference_gt
            else:
                template = _select_template(multi_templates[domain], preference_slots)
                query_id = str(template["query_id"])
                user_utterance = flatten_multiturn(template.get("query") or [])
                api_calls = template.get("api_call") or []
                if len(api_calls) != 1:
                    raise ContractError(f"template {query_id} must have exactly one base API call")
                base_domain, base_raw = parse_call_string(api_calls[0])
                if base_domain != domain:
                    raise ContractError(f"template {query_id} base domain mismatch")
                explicit_base = normalize_arguments(domain, base_raw, schemas)
                slot_values: dict[str, list[Any]] = defaultdict(list)
                for call in preference_gt:
                    for slot, value in call["arguments"].items():
                        slot_values[slot].append(value)
                full_gt = calls_from_slot_values(domain, dict(slot_values), schemas, explicit_base)
            prompt = render_prompt(example, user_utterance, prompt_template, tools)
            identity = {
                "pair_id": target["pair_id"],
                "mode": mode,
                "query_id": query_id,
                "user_utterance": user_utterance,
            }
            cases.append({
                "case_id": content_id("case", identity),
                "pair_id": target["pair_id"],
                "target_id": target["target_id"],
                "source_example_id": target["source_example_id"],
                "source_index": target["source_index"],
                "difficulty": target["difficulty"],
                "mode": mode,
                "target_domain": domain,
                "preference_group": target["preference_group"],
                "query_id": query_id,
                "user_utterance": user_utterance,
                "explicit_base_arguments": explicit_base,
                "reference_ground_truth_preference": preference_gt,
                "reference_ground_truth_full": full_gt,
                "model_input": prompt,
                "request": build_request(prompt, tools),
            })
    cases.sort(key=lambda item: item["case_id"])
    ids = [item["case_id"] for item in cases]
    if len(ids) != len(set(ids)):
        raise ContractError("duplicate case IDs")
    return cases


def validate_matrix(targets: Sequence[Mapping[str, Any]], cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    target_counts = Counter(target["difficulty"] for target in targets)
    case_counts = Counter((case["mode"], case["difficulty"]) for case in cases)
    if dict(target_counts) != EXPECTED:
        raise ContractError(f"unexpected target matrix: {dict(target_counts)} != {EXPECTED}")
    for mode in MODES:
        for difficulty, expected in EXPECTED.items():
            if case_counts[(mode, difficulty)] != expected:
                raise ContractError(f"unexpected {mode}/{difficulty} count: {case_counts[(mode, difficulty)]}")
    by_pair: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for case in cases:
        by_pair[case["pair_id"]].append(case)
    if len(by_pair) != len(targets):
        raise ContractError("pair count differs from target count")
    for pair_id, pair in by_pair.items():
        if {case["mode"] for case in pair} != set(MODES) or len(pair) != 2:
            raise ContractError(f"malformed pair {pair_id}")
        left, right = pair
        for key in ("target_domain", "reference_ground_truth_preference"):
            if canonical_json(left[key]) != canonical_json(right[key]):
                raise ContractError(f"paired {key} mismatch for {pair_id}")
    source_counts = {
        difficulty: len({target["source_example_id"] for target in targets if target["difficulty"] == difficulty})
        for difficulty in DIFFICULTIES
    }
    maxima = {
        difficulty: max(
            Counter(target["source_example_id"] for target in targets if target["difficulty"] == difficulty).values()
        )
        for difficulty in DIFFICULTIES
    }
    if source_counts != {"easy": 335, "medium": 123, "hard": 123}:
        raise ContractError(f"unexpected eligible source counts: {source_counts}")
    if maxima != {"easy": 5, "medium": 5, "hard": 7}:
        raise ContractError(f"unexpected per-source maxima: {maxima}")
    return {
        "targets": len(targets),
        "cases": len(cases),
        "target_counts": dict(sorted(target_counts.items())),
        "case_counts": {f"{mode}:{difficulty}": case_counts[(mode, difficulty)] for mode in MODES for difficulty in DIFFICULTIES},
        "eligible_sources": source_counts,
        "max_per_source": maxima,
        "paired_identity": "pass",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default="/data/minseo/experiments4")
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    run_root = Path(args.run_root).resolve()
    if run_root.exists():
        raise ContractError(f"run root already exists: {run_root}")
    run_root.mkdir(parents=True)

    paths = {
        "dataset": repo / "data/mix600.json",
        "single_queries": repo / "query_singleturn.json",
        "multi_queries": repo / "query_multiturn.json",
        "pref_list": repo / "pref_list.json",
        "pref_groups": repo / "pref_group.json",
        "tools": repo / "schema_all.json",
        "prompt": repo / "vanillaLLM/prompt.py",
        "reference_single": repo / "vanillaLLM/vanillaLLM_inference-vllm-single.py",
        "reference_multi": repo / "vanillaLLM/vanillaLLM_inference-vllm-multi.py",
    }
    code_paths = sorted((repo / "vanillaLLM/gemma4_mix600").glob("*.py"))
    input_hashes = {name: sha256_file(path) for name, path in paths.items()}
    code_hashes = {str(path.relative_to(repo)): sha256_file(path) for path in code_paths}
    write_json_atomic(run_root / "dirty_before.json", current_dirty_snapshot(repo))
    snapshot_root = run_root / "snapshots"
    for name, path in paths.items():
        destination = snapshot_root / name / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    for path in code_paths:
        destination = snapshot_root / "code" / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)

    dataset = load_json(paths["dataset"])
    single_queries = load_json(paths["single_queries"])
    raw_multi_templates = load_json(paths["multi_queries"])
    multi_templates = _group_multi_templates(raw_multi_templates)
    pref_list = load_json(paths["pref_list"])
    pref_groups = load_json(paths["pref_groups"])
    source_tools = load_json(paths["tools"])
    tools, schema_overlay = _derive_schema_overlay(source_tools, raw_multi_templates)
    schemas = schema_map(tools)
    prompt_template = _load_prompt(paths["prompt"])
    targets, rejections = derive_targets(dataset, single_queries, multi_templates, pref_list, pref_groups, schemas)
    dataset_by_id = {item["example_id"]: item for item in dataset}
    if len(dataset_by_id) != len(dataset):
        raise ContractError("duplicate source example_id")
    cases = render_cases(targets, dataset_by_id, single_queries, multi_templates, tools, schemas, prompt_template)
    validation = validate_matrix(targets, cases)

    prepared = run_root / "prepared"
    write_json_atomic(prepared / "tools_schema_derived.json", tools)
    write_json_atomic(prepared / "schema_overlay.json", schema_overlay)
    write_jsonl_atomic(prepared / "targets.jsonl", targets)
    write_jsonl_atomic(prepared / "cases.jsonl", cases)
    write_jsonl_atomic(prepared / "rejections.jsonl", rejections)
    write_json_atomic(prepared / "validation.json", validation)
    artifact_hashes = {
        name: sha256_file(prepared / name)
        for name in (
            "tools_schema_derived.json",
            "schema_overlay.json",
            "targets.jsonl",
            "cases.jsonl",
            "rejections.jsonl",
            "validation.json",
        )
    }
    manifest = {
        "experiment": "gemma4_mix600_vanilla_llm",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "construction": "mode-neutral-targets-then-paired-renderings",
        "schema_policy": "hashed source schema plus deterministic explicit multi-base overlay, shared by both modes",
        "input_paths": {name: str(path) for name, path in paths.items()},
        "input_sha256": input_hashes,
        "code_sha256": code_hashes,
        "artifact_sha256": artifact_hashes,
        "counts": validation,
        "request_contract": {
            "messages": "one user message",
            "context": "diag-apilist",
            "prompt": "imp-zs",
            "temperature": 0.0,
            "tool_choice": "auto",
            "max_tokens": 256,
            "thinking": False,
        },
    }
    write_json_atomic(run_root / "run_manifest.json", manifest)
    print(json.dumps({"run_root": str(run_root), "validation": validation}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
