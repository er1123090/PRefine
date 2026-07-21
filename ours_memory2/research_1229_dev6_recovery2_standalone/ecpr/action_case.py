"""Sole action-case builder shared by live inference and provenance replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contracts import CandidatePolicy
from .io import canonical_json, sha256_text
from .latent_firewall import (
    JournalReplayCapability,
    LatentFirewallDecision,
    QueryConstraintMask,
    build_query_constraint_mask,
    decide_latent_transfer,
)
from .prompts import (
    baseline_memory_block,
    build_action_prompt,
    candidate_memory_block,
)
from .query_gate import GateDecision, gate_public_query
from .router import route_hypotheses


@dataclass(frozen=True)
class ActionCase:
    memory_block: str
    prompt: str
    routing_decision: GateDecision | None
    vlt_decision: LatentFirewallDecision | None
    audit_artifact: dict[str, Any]


def build_action_case(
    *,
    arm: str,
    task: dict[str, Any],
    history: dict[str, Any],
    memory: dict[str, Any],
    schema: Any,
    preference_slots: Mapping[str, Sequence[str]],
    policy: CandidatePolicy,
    public_domain_ontology: dict[str, Any] | None = None,
    latent_trait_ontology: dict[str, Any] | None = None,
    latent_trait_ontology_sha256: str | None = None,
    replay_capability: JournalReplayCapability | None = None,
    ablations: set[str] | None = None,
) -> ActionCase:
    """Build one prompt and its raw-free deterministic audit decision."""
    query_constraint_mask: QueryConstraintMask | None = None
    routed_before_constraint_count = 0
    if arm == "baseline":
        block = baseline_memory_block(
            memory,
            history,
            policy.memory_lexical_token_cap,
        )
        prompt = build_action_prompt(task, schema, block)
        routing_decision = None
        vlt_decision = None
        routed: list[dict[str, Any]] = []
    elif arm == "candidate":
        if (
            public_domain_ontology is None
            or latent_trait_ontology is None
            or latent_trait_ontology_sha256 is None
        ):
            raise ValueError("candidate action build lacks frozen ontology inputs")
        normalized_slots = {
            str(domain): [str(slot) for slot in slots]
            for domain, slots in preference_slots.items()
        }
        routing_decision = gate_public_query(
            str(task["query"]),
            str(task["mode"]),
            schema,
            public_domain_ontology,
        )
        query_constraint_mask = build_query_constraint_mask(
            current_query=str(task["query"]),
            query_mode=str(task["mode"]),
            selected_domain=routing_decision.schema_domain,
            schema_key=str(task["schema_key"]),
            schema=schema,
            ontology=latent_trait_ontology,
        )
        routed_unmasked = route_hypotheses(
            memory,
            task,
            schema,
            normalized_slots,
            policy,
            public_domain_ontology,
            ablations=ablations,
            decision=routing_decision,
        )
        routed_before_constraint_count = len(routed_unmasked)
        masked_slots = set(query_constraint_mask.constrained_slots)
        routed = [
            item
            for item in routed_unmasked
            if not (
                str(item.get("domain", "")) == routing_decision.schema_domain
                and str(item.get("slot", "")) in masked_slots
            )
        ]
        vlt_decision = decide_latent_transfer(
            latent_abstraction=memory.get("latent_abstraction"),
            latent_attestation=memory.get("latent_attestation"),
            pqr_status=routing_decision.status,
            selected_domain=routing_decision.schema_domain,
            schema_key=str(task["schema_key"]),
            current_query=str(task["query"]),
            query_mode=str(task["mode"]),
            schema=schema,
            preference_slots=normalized_slots,
            typed_hypotheses=memory.get("typed_hypotheses", []),
            routed_hypotheses=routed,
            policy=policy,
            ontology=latent_trait_ontology,
            ontology_sha256=latent_trait_ontology_sha256,
            ablations=set(ablations or ()),
            replay_capability=replay_capability,
            query_constraint_mask=query_constraint_mask,
        )
        block = candidate_memory_block(
            routed,
            policy.memory_lexical_token_cap,
            selected_domain=routing_decision.schema_domain,
            vlt_decision=vlt_decision,
            latent_trait_ontology=latent_trait_ontology,
            schema=schema,
            preference_slots=normalized_slots,
            latent_trait_ontology_sha256=latent_trait_ontology_sha256,
            schema_key=str(task["schema_key"]),
            current_query=str(task["query"]),
            query_mode=str(task["mode"]),
            query_constraint_mask=query_constraint_mask,
        )
        # The preregistered action-prompt skeleton is shared byte-for-byte by
        # both arms. Routing authority is serialized only inside the memory
        # block; it must never add candidate-only prompt instructions.
        prompt = build_action_prompt(task, schema, block)
    else:
        raise ValueError("action case arm must be baseline or candidate")

    routing_artifact = (
        {
            "status": routing_decision.status,
            "canonical_domain": routing_decision.canonical_domain,
            "schema_domain": routing_decision.schema_domain,
            "top_score": routing_decision.top_score,
            "second_score": routing_decision.second_score,
            "scores_sha256": sha256_text(
                canonical_json(routing_decision.scores)
            ),
        }
        if routing_decision is not None
        else None
    )
    query_constraint_artifact = (
        query_constraint_mask.artifact()
        if query_constraint_mask is not None
        else None
    )
    artifact = {
        "schema_version": 1,
        "kind": "action_case_build_audit_v1",
        "arm": arm,
        "case_key_sha256": sha256_text(str(task["case_key"])),
        "example_id_sha256": sha256_text(str(task["example_id"])),
        "schema_sha256": sha256_text(canonical_json(schema)),
        "schema_key": str(task["schema_key"]),
        "query_mode": str(task["mode"]),
        "query_sha256": sha256_text(str(task["query"])),
        "query_constraint_mask": query_constraint_artifact,
        "query_constraint_mask_sha256": (
            sha256_text(canonical_json(query_constraint_artifact))
            if query_constraint_artifact is not None
            else None
        ),
        "routing": routing_artifact,
        "routed_before_constraint_count": routed_before_constraint_count,
        "routed_constraint_filtered_count": routed_before_constraint_count - len(routed),
        "routed_count": len(routed),
        "routed_sha256": sha256_text(canonical_json(routed)),
        "vlt": vlt_decision.artifact() if vlt_decision is not None else None,
        "replay_capability_sha256": (
            sha256_text(canonical_json(replay_capability.artifact()))
            if replay_capability is not None
            else None
        ),
        "memory_block_sha256": sha256_text(block),
        "prompt_sha256": sha256_text(prompt),
    }
    artifact["decision_sha256"] = sha256_text(canonical_json(artifact))
    return ActionCase(
        memory_block=block,
        prompt=prompt,
        routing_decision=routing_decision,
        vlt_decision=vlt_decision,
        audit_artifact=artifact,
    )

