#!/usr/bin/env python3
"""Aggregate VLT3 structural evidence without queries, gold, or model outputs."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ecpr.contracts import CandidatePolicy
from ecpr.firewall import validate_sanitized_history
from ecpr.io import iter_jsonl, load_json, sha256_file, write_json
from ecpr.latent_firewall import (
    _policy_qualified,
    qualified_trait_diagnostics,
    validate_latent_trait_ontology_v3,
)
from ecpr.memory import build_typed_hypotheses


ROOT = Path(__file__).resolve().parent
HISTORY = ROOT / "artifacts/history.sanitized.jsonl"
PREFERENCE_SLOTS = ROOT / "configs/preference_slots.json"
ONTOLOGY = ROOT / "configs/latent_trait_ontology.vlt3.json"
PREREGISTRATION = ROOT / "preregistration.json"
DEFAULT_OUTPUT = ROOT / "reports/TARGET_FREE_VLT3_DIAGNOSTICS.json"


def _histogram(values: list[int]) -> dict[str, int]:
    return {
        str(key): count
        for key, count in sorted(Counter(values).items())
    }


def _policy(preregistration: dict[str, Any]) -> CandidatePolicy:
    raw = preregistration.get("candidate_parameters")
    if not isinstance(raw, dict) or set(raw) != set(asdict(CandidatePolicy())):
        raise ValueError("preregistered CandidatePolicy field contract mismatch")
    return CandidatePolicy(**raw)


def build_report() -> dict[str, Any]:
    """Run the production typed-evidence matcher and emit aggregate-only facts."""
    history_count = validate_sanitized_history(HISTORY)
    preference_slots = load_json(PREFERENCE_SLOTS)
    ontology = validate_latent_trait_ontology_v3(load_json(ONTOLOGY))
    preregistration = load_json(PREREGISTRATION)
    policy = _policy(preregistration)

    trait_ids = tuple(str(value) for value in ontology["trait_order"])
    trait_opposes = {
        str(item["trait_id"]): {str(value) for value in item["opposes"]}
        for item in ontology["traits"]
    }
    if set(trait_ids) != set(trait_opposes):
        raise ValueError("VLT3 trait opposition contract mismatch")

    examples_with_trait = Counter()
    qualified_hypotheses = Counter()
    domain_histograms: dict[str, list[int]] = defaultdict(list)
    hypothesis_histograms: dict[str, list[int]] = defaultdict(list)
    qualified_domain_counts: dict[str, Counter[str]] = defaultdict(Counter)

    mapping_stats = {
        (str(mapping["trait_id"]), str(mapping["domain"]), str(mapping["slot"])): {
            "examples_with_non_target_support": 0,
            "examples_with_opposing_trait_veto": 0,
            "examples_with_policy_qualified_target_slot": 0,
            "examples_meeting_typed_only_preconditions": 0,
        }
        for mapping in ontology["target_mappings"]
    }

    observed = 0
    for history in iter_jsonl(HISTORY):
        observed += 1
        hypotheses = build_typed_hypotheses(history, preference_slots, policy)
        diagnostics = qualified_trait_diagnostics(ontology, hypotheses, policy)
        by_trait: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in diagnostics:
            trait_id = str(item["trait_id"])
            by_trait[trait_id].append(item["hypothesis"])
            qualified_hypotheses[trait_id] += 1
            qualified_domain_counts[trait_id][str(item["hypothesis"]["domain"])] += 1

        for trait_id in trait_ids:
            evidence = by_trait.get(trait_id, [])
            domains = {str(item["domain"]) for item in evidence}
            if evidence:
                examples_with_trait[trait_id] += 1
            domain_histograms[trait_id].append(len(domains))
            hypothesis_histograms[trait_id].append(len(evidence))

        for mapping in ontology["target_mappings"]:
            trait_id = str(mapping["trait_id"])
            domain = str(mapping["domain"])
            slot = str(mapping["slot"])
            stats = mapping_stats[(trait_id, domain, slot)]
            non_target_support = any(
                str(item["domain"]) != domain for item in by_trait.get(trait_id, [])
            )
            opposing_veto = any(
                by_trait.get(opposed) for opposed in trait_opposes[trait_id]
            )
            target_slot_present = any(
                isinstance(item, dict)
                and str(item.get("domain", "")) == domain
                and str(item.get("slot", "")) == slot
                and _policy_qualified(item, policy)
                for item in hypotheses
            )
            if non_target_support:
                stats["examples_with_non_target_support"] += 1
            if opposing_veto:
                stats["examples_with_opposing_trait_veto"] += 1
            if target_slot_present:
                stats["examples_with_policy_qualified_target_slot"] += 1
            if non_target_support and not opposing_veto and not target_slot_present:
                stats["examples_meeting_typed_only_preconditions"] += 1

    if observed != history_count:
        raise ValueError("sanitized-history count changed during diagnostics")

    trait_report = {}
    for trait_id in trait_ids:
        trait_report[trait_id] = {
            "examples_with_at_least_one_policy_qualified_hypothesis": examples_with_trait[trait_id],
            "policy_qualified_hypothesis_count": qualified_hypotheses[trait_id],
            "distinct_support_domains_per_example_histogram": _histogram(
                domain_histograms[trait_id]
            ),
            "qualified_hypotheses_per_example_histogram": _histogram(
                hypothesis_histograms[trait_id]
            ),
            "aggregate_qualified_hypotheses_by_source_domain": {
                domain: count
                for domain, count in sorted(qualified_domain_counts[trait_id].items())
            },
        }

    mapping_report = []
    for (trait_id, domain, slot), stats in sorted(mapping_stats.items()):
        mapping_report.append(
            {
                "trait_id": trait_id,
                "target_domain": domain,
                "target_slot": slot,
                **stats,
            }
        )

    return {
        "schema_version": 1,
        "kind": "target_free_vlt3_structural_diagnostics",
        "status": "aggregate_only_no_query_gold_latent_model_output_or_metric",
        "candidate_id": "ecpr_v1",
        "candidate_revision": "vlt3",
        "inputs": {
            "sanitized_history": {
                "path": "artifacts/history.sanitized.jsonl",
                "sha256": sha256_file(HISTORY),
                "records": history_count,
            },
            "preference_slots": {
                "path": "configs/preference_slots.json",
                "sha256": sha256_file(PREFERENCE_SLOTS),
            },
            "latent_trait_ontology": {
                "path": "configs/latent_trait_ontology.vlt3.json",
                "sha256": sha256_file(ONTOLOGY),
            },
            "preregistration": {
                "path": "preregistration.json",
                "sha256": sha256_file(PREREGISTRATION),
            },
        },
        "production_code": {
            "typed_hypothesis_builder": "ecpr.memory.build_typed_hypotheses",
            "trait_matcher": "ecpr.latent_firewall.qualified_trait_diagnostics",
            "policy_predicate": "ecpr.latent_firewall._policy_qualified",
        },
        "candidate_policy": asdict(policy),
        "trait_aggregates": trait_report,
        "target_mapping_typed_only_preconditions": mapping_report,
        "interpretation_limit": (
            "These are history-only typed-evidence preconditions. They do not inspect or "
            "predict final PQR selection, query explicit-constraint masks, verified latent "
            "trait authorization, action outputs, gold, or task metrics."
        ),
        "forbidden_sources_not_opened": [
            "artifacts/tasks.jsonl",
            "configs/query_singleturn.json",
            "configs/query_multiturn-domain.json",
            "evaluator_vault/gold.jsonl",
            "current_or_legacy_latent_outputs",
            "current_or_legacy_predictions",
            "evaluator_outputs_or_metrics",
            "preference_groups",
        ],
    }


def main() -> int:
    write_json(DEFAULT_OUTPUT, build_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
