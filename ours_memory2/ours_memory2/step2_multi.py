"""Deterministic multi-turn Step 2 action inference."""

from __future__ import annotations

from .contracts import Difficulty, InputContractError
from .providers import Provider
from .step2_common import (
    _execute_step2,
    ManifestResources,
    MultiTemplate,
    Step2Case,
    Step2Run,
    generate_calls,
    load_manifest_v1,
    normalize_context,
    normalize_difficulties,
    parse_api_call,
    preference_maps,
)


def select_template(
    templates: tuple[MultiTemplate, ...], preferences: dict[str, tuple[str, ...]]
) -> MultiTemplate:
    target_slots = set(preferences)
    best = templates[0]
    best_score = 0
    for template in templates:
        score = len(target_slots.intersection(template.target_slots))
        if score > best_score:
            best, best_score = template, score
    return best


def build_multi_cases(
    resources: ManifestResources, *, difficulty: str | Difficulty
) -> tuple[Step2Case, ...]:
    cases: list[Step2Case] = []
    for selected in normalize_difficulties(difficulty):
        for example in resources.examples:
            sub_index = 0
            for domain, preferences, set_slots in preference_maps(resources, example, selected):
                template = select_template(resources.multiturn_templates[domain], dict(preferences))
                _, base_args = parse_api_call(template.api_calls[0])
                utterance = "\n".join(
                    f"{turn.get('role', 'User')}: {turn.get('message', '')}" for turn in template.query
                )
                cases.append(
                    Step2Case(
                        case_id=f"{example['example_id']}:multi:{selected.value}:{sub_index}",
                        example_id=example["example_id"],
                        difficulty=selected,
                        domain=domain,
                        utterance=utterance,
                        ground_truth=generate_calls(domain, base_args, preferences),
                        template_id=template.template_id,
                        set_derived_slots=set_slots,
                    )
                )
                sub_index += 1
    if not cases:
        raise InputContractError("derived no cases", path="cases")
    return tuple(cases)


def infer_multi(
    manifest_path: str,
    *,
    input_shape: str,
    difficulty: str | Difficulty,
    context: str,
    model: str,
    provider: Provider,
) -> Step2Run:
    mode = normalize_context(context)
    resources = load_manifest_v1(manifest_path, multi_shape=input_shape)
    cases = build_multi_cases(resources, difficulty=difficulty)
    return _execute_step2(
        resources=resources,
        cases=cases,
        context=mode,
        model=model,
        provider=provider,
        mode="multi",
    )
