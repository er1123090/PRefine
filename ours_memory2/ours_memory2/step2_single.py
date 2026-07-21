"""Deterministic single-turn Step 2 action inference."""

from __future__ import annotations

from .contracts import Difficulty, InputContractError
from .providers import Provider
from .step2_common import (
    _execute_step2,
    ManifestResources,
    Step2Case,
    Step2Run,
    generate_calls,
    load_manifest_v1,
    normalize_context,
    normalize_difficulties,
    preference_maps,
)


def build_single_cases(
    resources: ManifestResources, *, difficulty: str | Difficulty
) -> tuple[Step2Case, ...]:
    cases: list[Step2Case] = []
    for selected in normalize_difficulties(difficulty):
        for example in resources.examples:
            sub_index = 0
            for domain, preferences, set_slots in preference_maps(resources, example, selected):
                cases.append(
                    Step2Case(
                        case_id=f"{example['example_id']}:single:{selected.value}:{sub_index}",
                        example_id=example["example_id"],
                        difficulty=selected,
                        domain=domain,
                        utterance=resources.single_query_map[domain],
                        ground_truth=generate_calls(domain, {}, preferences),
                        set_derived_slots=set_slots,
                    )
                )
                sub_index += 1
    if not cases:
        raise InputContractError("derived no cases", path="cases")
    return tuple(cases)


def infer_single(
    manifest_path: str,
    *,
    difficulty: str | Difficulty,
    context: str,
    model: str,
    provider: Provider,
) -> Step2Run:
    mode = normalize_context(context)
    resources = load_manifest_v1(manifest_path)
    cases = build_single_cases(resources, difficulty=difficulty)
    return _execute_step2(
        resources=resources,
        cases=cases,
        context=mode,
        model=model,
        provider=provider,
        mode="single",
    )
