"""Pure, deterministic firewall for verified latent preference transfer.

The module never calls a provider and never serializes free-form latent text.  It
reduces an exactly-attested final PREFINE draft to one frozen trait, requires
independent cross-domain typed corroboration, and emits at most one schema-valid
closed tuple.
"""

from __future__ import annotations

import re
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .contracts import CandidatePolicy
from .io import canonical_json, sha256_text
from .parsing import normalize_value
from .query_gate import extract_user_payload


VLT1_CANDIDATE_REVISION = "vlt1"
VLT1_METHOD = "deterministic_verified_latent_transfer_v1"
VLT1_LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256 = (
    "d1462847c0094eb080b4f5a4be5685e3cba344b2cc34b85c8283386ec7268070"
)
LATENT_VERIFICATION_ATTESTATION_KIND = (
    "prefine_final_session_verification_attestation_v1"
)
GENERATED_ATTESTATION_SOURCE = "generated_prefine_current_run"

VALID_ATTESTATION_REASON = "VALID_FINAL_VERDICT"
ATTESTATION_REASON_CODES = frozenset(
    {
        VALID_ATTESTATION_REASON,
        "CLIENTLESS_FALLBACK",
        "EXTERNAL_UNATTESTED",
        "MALFORMED_FINAL_VERDICT",
        "NO_FINAL_SESSION",
        "RETRY_EXHAUSTED",
        "UNSEALED_VERIFIER",
        "VERIFIED_DRAFT_MALFORMED",
        "VERIFIED_DRAFT_NOT_FINAL_MEMORY",
    }
)
VLT_REASON_CODES = frozenset(
    {
        "AUTHORIZED",
        "AMBIGUOUS_TRAIT",
        "ATTESTATION_MISMATCH",
        "CURRENT_QUERY_EXPLICIT_CONSTRAINT",
        "INSTRUCTION_LIKE",
        "INSUFFICIENT_CROSS_DOMAIN_SUPPORT",
        "INSUFFICIENT_TYPED_SUPPORT",
        "INVALID_ATTESTATION",
        "INVALID_LATENT_SHAPE",
        "NEGATION_PRESENT",
        "NO_DOMAIN_MAPPING",
        "NO_LATENT_ABLATION",
        "NO_TRAIT_MATCH",
        "MISSING_REPLAY_CAPABILITY",
        "NO_TYPED_ABLATION",
        "OPPOSING_TRAIT_SUPPORT",
        "OVER_LENGTH_CAP",
        "PQR_ABSTAIN",
        "SCHEMA_MISMATCH",
        "TARGET_SLOT_AMBIGUOUS",
        "TARGET_SLOT_CONFLICT",
        "TARGET_SLOT_REDUNDANT",
        "UNSAFE_CONTROL_OR_FORMAT",
        "UNSAFE_MULTILINE",
        "UNSAFE_QUOTE_OR_BACKTICK",
    }
)

VLT1_CONTRACT = {
    "method": VLT1_METHOD,
    "candidate_revision": VLT1_CANDIDATE_REVISION,
    "trait_count": 2,
    "normalization": "NFKC_then_casefold_then_unicode_token_boundary",
    "unsafe_input_policy": (
        "reject_control_bidi_zero_width_quotes_backticks_multiline_overcap_"
        "instruction_like_any_negation_zero_trait_or_multiple_traits"
    ),
    "attestation_policy": (
        "final_generated_session_verifier_valid_is_exactly_true_with_full_"
        "draft_implicit_pref_request_response_and_journal_attempt_call_bindings"
    ),
    "corroboration_policy": (
        "at_least_two_policy_qualified_supporting_hypotheses_from_distinct_"
        "non_target_domains_and_no_policy_qualified_opposing_trait_in_any_domain"
    ),
    "target_slot_policy": (
        "omit_if_any_policy_routed_selected_target_same_slot_evidence;different_"
        "value_is_conflict_and_equal_value_is_redundant"
    ),
    "output_policy": "one_schema_valid_closed_json_tuple_or_NONE_never_raw_latent",
    "ablation_policy": (
        "no_latent_or_no_typed_forces_NONE;no_counterevidence_or_no_routing_"
        "never_relaxes_vlt_qualification"
    ),
    "baseline_policy": "source_faithful_final_draft_projection_unchanged",
}

VLT1_SUPERSESSION_CONTRACT = {
    "supersedes": "only_zero_surviving_typed_policy_from_typed_only_scope_v1",
    "restored_authority": (
        "optional_closed_tuple_when_zero_target_routed_typed_evidence_survives_"
        "only_after_exact_attestation_and_two_distinct_non_target_domain_supports"
    ),
    "preserves": [
        "PQR_ABSTAIN_serializes_no_memory",
        "raw_free_form_latent_is_never_serialized",
        "target_routed_typed_evidence_is_selected_domain_only",
        "baseline_projection_and_prompt_are_unchanged",
    ],
}

VLT1_CANDIDATE_MEMORY_SCOPE_CONTRACT = {
    "method": "candidate_typed_routed_plus_verified_closed_tuple_vlt1",
    "parent_method": "candidate_typed_routed_evidence_only_v1",
    "candidate_revision": VLT1_CANDIDATE_REVISION,
    "selection_authority": "public_query_relevance_gate_v1",
    "selected_domain_scope": "selected_schema_domain_only",
    "serialized_evidence": (
        "selected_target_domain_routed_typed_hypotheses_plus_optional_"
        "authorized_closed_vlt_tuple"
    ),
    "untyped_latent_serialization": "forbidden_in_all_candidate_prompts",
    "abstain_policy": "no_memory_evidence",
    "zero_surviving_target_typed_policy": (
        "closed_vlt_tuple_only_when_cross_domain_corroborated_and_fully_authorized"
    ),
    "cross_domain_evidence_serialization": "forbidden_audit_digests_only",
    "typed_provenance_serialization": "audit_only_never_candidate_prompt",
    "same_target_slot_policy": "omit_vlt_even_if_equal;different_is_conflict",
    "selection_implication": "never_implies_requested_domain_or_operation",
    "baseline_memory_scope": (
        "unchanged_source_faithful_prefine_final_draft_without_candidate_attestation_filter"
    ),
    "no_latent_ablation_compatibility": "forces_no_vlt_tuple",
    "no_typed_ablation_compatibility": "forces_no_vlt_tuple_without_hidden_typed_use",
    "qualification_ablation_policy": (
        "no_routing_and_no_counterevidence_never_relax_vlt_policy_thresholds"
    ),
}

CANDIDATE_REVISION = "vlt2"
VLT_METHOD = "audited_verified_latent_transfer_v2"
LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256 = (
    "7237fdcfae80be7018e784d0dffa101e981fa9e3c2ae966fb1850b4a22acc8a1"
)

VLT_CONTRACT = {
    "method": VLT_METHOD,
    "candidate_revision": CANDIDATE_REVISION,
    "trait_count": 2,
    "trait_taxonomy": "paper_table_5_LOW_COST_and_HIGH_COST",
    "normalization": "NFKC_then_casefold_then_unicode_token_boundary",
    "unsafe_input_policy": (
        "reject_frozen_control_format_bidi_line_separator_paragraph_separator_"
        "quote_like_instruction_phrase_negation_token_affix_marker_multiline_"
        "overcap_zero_trait_or_multiple_traits"
    ),
    "attestation_policy": (
        "exact_final_verifier_attestation_plus_semantically_replayed_journal_"
        "capability_binding_attempt_call_request_messages_schema_call_key_response"
    ),
    "corroboration_policy": (
        "at_least_two_policy_qualified_supporting_hypotheses_from_distinct_"
        "non_target_domains_and_no_policy_qualified_opposing_trait_in_any_domain"
    ),
    "target_slot_policy": (
        "inspect_all_policy_qualified_raw_target_same_slot_hypotheses;multiple_"
        "values_are_ambiguous;one_equal_is_redundant;one_different_is_conflict"
    ),
    "output_policy": (
        "one_schema_revalidated_closed_json_tuple_or_NONE_never_raw_latent"
    ),
    "ablation_policy": (
        "no_latent_or_no_typed_forces_NONE;target_slot_and_policy_checks_are_"
        "ablation_independent"
    ),
    "routing_latent_policy": (
        "ABSTAIN_always_omits;SELECT_may_use_authorized_closed_tuple_with_zero_"
        "target_routed_typed_evidence"
    ),
    "baseline_policy": "source_faithful_final_draft_projection_unchanged",
}

VLT_SUPERSESSION_CONTRACT = {
    "parents": [
        "verified_latent_transfer_safe_restoration_amendment_vlt1",
        "expected_runtime_contract_vlt1",
        "latent_trait_ontology_vlt1",
    ],
    "corrections": [
        "replace_service_level_traits_with_paper_table_5_cost_taxonomy",
        "target_same_slot_guard_uses_all_policy_qualified_raw_typed_hypotheses",
        "require_semantic_journal_replay_capability",
        "serializer_independently_revalidates_closed_tuple",
        "resolve_SELECT_zero_target_typed_routing_policy",
    ],
    "preserves": [
        "PQR_ABSTAIN_serializes_no_memory",
        "raw_free_form_latent_is_never_serialized",
        "target_routed_typed_evidence_is_selected_domain_only",
        "baseline_projection_and_prompt_are_unchanged",
    ],
}

VLT_CANDIDATE_MEMORY_SCOPE_CONTRACT = {
    "method": "candidate_typed_routed_plus_audited_closed_tuple_vlt2",
    "parent_method": "candidate_typed_routed_plus_verified_closed_tuple_vlt1",
    "candidate_revision": CANDIDATE_REVISION,
    "selection_authority": "public_query_relevance_gate_v1",
    "selected_domain_scope": "selected_schema_domain_only",
    "serialized_evidence": (
        "selected_target_domain_routed_typed_hypotheses_plus_optional_"
        "schema_revalidated_replay_authorized_closed_vlt_tuple"
    ),
    "untyped_latent_serialization": "forbidden_in_all_candidate_prompts",
    "abstain_policy": "always_no_memory_evidence",
    "zero_surviving_target_typed_policy": (
        "SELECT_may_serialize_closed_tuple_only_when_all_vlt2_checks_authorize"
    ),
    "cross_domain_evidence_serialization": "forbidden_audit_digests_only",
    "typed_provenance_serialization": "audit_only_never_candidate_prompt",
    "same_target_slot_policy": (
        "all_policy_qualified_raw_values;multiple_ambiguous;equal_redundant;different_conflict"
    ),
    "tuple_serializer_policy": (
        "independent_vlt2_ontology_selected_schema_preference_slot_revalidation"
    ),
    "selection_implication": "never_implies_requested_domain_or_operation",
    "baseline_memory_scope": (
        "unchanged_source_faithful_prefine_final_draft_without_candidate_attestation_filter"
    ),
    "no_latent_ablation_compatibility": "forces_no_vlt_tuple",
    "no_typed_ablation_compatibility": "forces_no_vlt_tuple_without_hidden_typed_use",
    "qualification_ablation_policy": (
        "all_vlt_policy_and_target_slot_checks_are_ablation_independent"
    ),
}

VLT2_CANDIDATE_REVISION = CANDIDATE_REVISION
VLT2_METHOD = VLT_METHOD
VLT2_LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256 = LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256
VLT2_CONTRACT = VLT_CONTRACT
VLT2_SUPERSESSION_CONTRACT = VLT_SUPERSESSION_CONTRACT
VLT2_CANDIDATE_MEMORY_SCOPE_CONTRACT = VLT_CANDIDATE_MEMORY_SCOPE_CONTRACT
VLT3_CANDIDATE_REVISION = "vlt3"
VLT3_METHOD = "append_only_table5_verified_latent_transfer_v3"
VLT3_LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256 = "45cdbc6c715bc341cdc8df0d4fd6c096f168eefea9bc4b8ae876afea04375210"
VLT3_CONTRACT = {
    "method": VLT3_METHOD,
    "candidate_revision": VLT3_CANDIDATE_REVISION,
    "trait_count": 3,
    "trait_taxonomy": "paper_table_5_LOW_COST_HIGH_COST_and_SOLO_USAGE",
    "mapping_identity": ["trait_id", "domain"],
    "opposition_policy": "explicit_same_group_only_LOW_COST_opposes_HIGH_COST_and_cost_is_neutral_to_SOLO_USAGE",
    "corroboration_policy": "at_least_one_policy_qualified_non_target_hypothesis_uniformly_for_every_trait",
    "target_slot_policy": "all_policy_qualified_raw_values_preserve_type;multiple_ambiguous;one_exact_equal_redundant;one_different_conflict",
    "current_query_policy": "ontology_frozen_mapping_specific_public_phrases_enums_slot_aliases_and_conservative_count_detection",
    "current_query_limit": "deterministic_mask_covers_only_frozen_Table5_mapped_slots;other_natural_language_or_non_Table5_slots_remain_under_prompt_precedence",
    "typed_support_wire_policy": "exact_JSON_scalar_source_literal_type_and_value_for_support;target_mapping_keeps_exact_public_schema_output_type",
    "output_policy": "one_schema_key_revalidated_closed_json_tuple_or_NONE_never_raw_latent",
    "group_usage_policy": "never_create_transfer_or_global_veto",
    "baseline_policy": "source_local_mode_specific_prefine_action_prompt_shared_byte_identically_across_arms",
    "action_prompt_policy": "detailed_single_multi_templates_selected_by_exact_mode_schema_key_pair",
}
VLT3_SUPERSESSION_CONTRACT = {
    "parent": "audited_verified_latent_transfer_v2",
    "parent_ontology_sha256": "be2070439b5928d6f2b1468ea0b8b63c0a42f48b578b4755a6ef949880fe005b",
    "append_only_changes": [
        "add_paper_table_5_SOLO_USAGE",
        "key_mappings_by_trait_id_and_domain",
        "make_opposition_explicit_and_group_local",
        "accept_exact_raw_string_one_for_party_size_support",
        "require_one_policy_qualified_non_target_hypothesis_for_every_trait",
        "suppress_transfer_when_current_query_explicitly_constrains_target_slot",
    ],
    "preserves": [
        "frozen_v1_and_v2_ontology_and_contract_validation",
        "PQR_ABSTAIN_serializes_no_memory",
        "raw_free_form_latent_is_never_serialized",
        "same_mode_baseline_and_candidate_prompt_skeleton_are_byte_identical",
    ],
}
VLT3_CANDIDATE_MEMORY_SCOPE_CONTRACT = {
    **VLT2_CANDIDATE_MEMORY_SCOPE_CONTRACT,
    "method": "candidate_typed_routed_plus_append_only_table5_closed_tuple_vlt3",
    "parent_method": VLT2_CANDIDATE_MEMORY_SCOPE_CONTRACT["method"],
    "candidate_revision": VLT3_CANDIDATE_REVISION,
    "same_target_slot_policy": VLT3_CONTRACT["target_slot_policy"],
    "tuple_serializer_policy": "independent_vlt3_ontology_schema_key_query_scope_and_preference_slot_revalidation",
}
ACTIVE_CANDIDATE_REVISION = VLT3_CANDIDATE_REVISION
ACTIVE_VLT_METHOD = VLT3_METHOD
ACTIVE_LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256 = (
    VLT3_LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256
)
ACTIVE_VLT_CONTRACT = VLT3_CONTRACT
ACTIVE_VLT_SUPERSESSION_CONTRACT = VLT3_SUPERSESSION_CONTRACT
ACTIVE_VLT_CANDIDATE_MEMORY_SCOPE_CONTRACT = VLT3_CANDIDATE_MEMORY_SCOPE_CONTRACT
_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
_HEX = frozenset("0123456789abcdef")
_ATTESTATION_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "source",
        "status",
        "reason_code",
        "final_session_index",
        "final_attempt_index",
        "verdict_valid_is_exactly_true",
        "journal_attempt_id",
        "verifier_call_id",
        "final_draft_sha256",
        "implicit_pref_sha256",
        "final_memory_latent_sha256",
        "verifier_response_sha256",
        "verifier_request_sha256",
        "verifier_call_sha256",
        "verifier_call_key_sha256",
        "verifier_schema_sha256",
        "verifier_prompt_sha256",
        "verifier_messages_sha256",
    }
)


def _digest_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _HEX for character in value)
    )


def _implicit_pref(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    implicit_pref = value.get("implicit_pref")
    return implicit_pref if isinstance(implicit_pref, str) else None


@dataclass(frozen=True)
class LatentVerificationAttestation:
    schema_version: int
    kind: str
    source: str
    status: str
    reason_code: str
    final_session_index: int | None
    final_attempt_index: int | None
    verdict_valid_is_exactly_true: bool
    journal_attempt_id: str | None
    verifier_call_id: str | None
    final_draft_sha256: str | None
    implicit_pref_sha256: str | None
    final_memory_latent_sha256: str
    verifier_response_sha256: str | None
    verifier_request_sha256: str | None
    verifier_call_sha256: str | None
    verifier_call_key_sha256: str | None
    verifier_schema_sha256: str | None
    verifier_prompt_sha256: str | None
    verifier_messages_sha256: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            field: getattr(self, field)
            for field in sorted(_ATTESTATION_FIELDS)
        }


def unattested_latent(
    latent_abstraction: Any,
    *,
    source: str,
    reason_code: str,
) -> LatentVerificationAttestation:
    if reason_code not in {"CLIENTLESS_FALLBACK", "EXTERNAL_UNATTESTED"}:
        raise ValueError("invalid unattested latent reason")
    implicit_pref = _implicit_pref(latent_abstraction)
    return LatentVerificationAttestation(
        schema_version=1,
        kind=LATENT_VERIFICATION_ATTESTATION_KIND,
        source=source,
        status="UNATTESTED",
        reason_code=reason_code,
        final_session_index=None,
        final_attempt_index=None,
        verdict_valid_is_exactly_true=False,
        journal_attempt_id=None,
        verifier_call_id=None,
        final_draft_sha256=None,
        implicit_pref_sha256=(
            sha256_text(implicit_pref) if implicit_pref is not None else None
        ),
        final_memory_latent_sha256=_digest_json(latent_abstraction),
        verifier_response_sha256=None,
        verifier_request_sha256=None,
        verifier_call_sha256=None,
        verifier_call_key_sha256=None,
        verifier_schema_sha256=None,
        verifier_prompt_sha256=None,
        verifier_messages_sha256=None,
    )


def parse_latent_attestation(value: Any) -> LatentVerificationAttestation:
    if not isinstance(value, dict) or set(value) != _ATTESTATION_FIELDS:
        raise ValueError("latent verification attestation field contract violation")
    if value.get("schema_version") != 1:
        raise ValueError("latent verification attestation schema mismatch")
    if value.get("kind") != LATENT_VERIFICATION_ATTESTATION_KIND:
        raise ValueError("latent verification attestation kind mismatch")
    if value.get("reason_code") not in ATTESTATION_REASON_CODES:
        raise ValueError("latent verification attestation reason mismatch")
    if value.get("status") not in {"VALID", "INVALID", "UNATTESTED"}:
        raise ValueError("latent verification attestation status mismatch")
    if type(value.get("verdict_valid_is_exactly_true")) is not bool:
        raise ValueError("latent verification attestation verdict type mismatch")
    if not _is_sha256(value.get("final_memory_latent_sha256")):
        raise ValueError("latent verification memory digest mismatch")
    optional_digests = {
        "final_draft_sha256",
        "implicit_pref_sha256",
        "verifier_response_sha256",
        "verifier_request_sha256",
        "verifier_call_sha256",
        "verifier_call_key_sha256",
        "verifier_schema_sha256",
        "verifier_prompt_sha256",
        "verifier_messages_sha256",
    }
    for field in optional_digests:
        if value.get(field) is not None and not _is_sha256(value.get(field)):
            raise ValueError(f"latent verification {field} mismatch")
    for field in ("final_session_index", "final_attempt_index"):
        if value.get(field) is not None and (
            type(value.get(field)) is not int or int(value[field]) < 0
        ):
            raise ValueError(f"latent verification {field} mismatch")
    for field in ("journal_attempt_id", "verifier_call_id"):
        identifier = value.get(field)
        if identifier is not None:
            try:
                canonical = str(uuid.UUID(str(identifier)))
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(f"latent verification {field} mismatch") from exc
            if canonical != identifier:
                raise ValueError(f"latent verification {field} mismatch")
    status = value["status"]
    reason_code = value["reason_code"]
    if (status == "VALID") != (reason_code == VALID_ATTESTATION_REASON):
        raise ValueError("latent verification status/reason mismatch")
    if status == "UNATTESTED" and reason_code not in {
        "CLIENTLESS_FALLBACK",
        "EXTERNAL_UNATTESTED",
    }:
        raise ValueError("latent verification unattested reason mismatch")
    if status == "INVALID" and reason_code in {
        VALID_ATTESTATION_REASON,
        "CLIENTLESS_FALLBACK",
        "EXTERNAL_UNATTESTED",
    }:
        raise ValueError("latent verification invalid reason mismatch")
    return LatentVerificationAttestation(**value)


def validate_latent_attestation(
    latent_abstraction: Any,
    value: Any,
) -> LatentVerificationAttestation:
    attestation = parse_latent_attestation(value)
    if attestation.final_memory_latent_sha256 != _digest_json(latent_abstraction):
        raise ValueError("latent attestation does not bind final memory latent")
    implicit_pref = _implicit_pref(latent_abstraction)
    if implicit_pref is not None and attestation.implicit_pref_sha256 is not None:
        if attestation.implicit_pref_sha256 != sha256_text(implicit_pref):
            raise ValueError("latent attestation implicit preference digest mismatch")
    return attestation


_REPLAY_AUTHORITY = object()


@dataclass(frozen=True)
class JournalReplayCapability:
    schema_version: int
    kind: str
    attempt_id: str
    verifier_call_id: str
    verifier_request_sha256: str
    verifier_response_sha256: str
    attestation_sha256: str
    latent_sha256: str
    trace_sha256: str
    _authority: object | None = None

    def artifact(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "attempt_id": self.attempt_id,
            "verifier_call_id": self.verifier_call_id,
            "verifier_request_sha256": self.verifier_request_sha256,
            "verifier_response_sha256": self.verifier_response_sha256,
            "attestation_sha256": self.attestation_sha256,
            "latent_sha256": self.latent_sha256,
            "trace_sha256": self.trace_sha256,
        }


def parse_replay_capability(value: Any) -> JournalReplayCapability:
    fields = {
        "schema_version",
        "kind",
        "attempt_id",
        "verifier_call_id",
        "verifier_request_sha256",
        "verifier_response_sha256",
        "attestation_sha256",
        "latent_sha256",
        "trace_sha256",
    }
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("journal replay capability field contract violation")
    if value.get("schema_version") != 1:
        raise ValueError("journal replay capability schema mismatch")
    if value.get("kind") != "semantic_prefine_journal_replay_capability_v1":
        raise ValueError("journal replay capability kind mismatch")
    for field in ("attempt_id", "verifier_call_id"):
        try:
            canonical = str(uuid.UUID(str(value.get(field))))
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("journal replay capability UUID mismatch") from exc
        if canonical != value.get(field):
            raise ValueError("journal replay capability UUID mismatch")
    for field in (
        "verifier_request_sha256",
        "verifier_response_sha256",
        "attestation_sha256",
        "latent_sha256",
        "trace_sha256",
    ):
        if not _is_sha256(value.get(field)):
            raise ValueError(f"journal replay capability {field} mismatch")
    return JournalReplayCapability(**value)


def _mint_replay_capability(**value: Any) -> JournalReplayCapability:
    """Internal mint used only after exact semantic replay succeeds."""
    capability = parse_replay_capability(value)
    object.__setattr__(capability, "_authority", _REPLAY_AUTHORITY)
    return capability


def _has_replay_authority(value: Any) -> bool:
    return (
        type(value) is JournalReplayCapability
        and value._authority is _REPLAY_AUTHORITY
    )



def _attestation_authorizes(
    latent_abstraction: Any,
    value: Any,
    replay_capability: JournalReplayCapability | None,
) -> tuple[bool, str]:
    try:
        attestation = validate_latent_attestation(latent_abstraction, value)
    except (TypeError, ValueError):
        return False, "INVALID_ATTESTATION"
    required_digests = (
        attestation.final_draft_sha256,
        attestation.implicit_pref_sha256,
        attestation.verifier_response_sha256,
        attestation.verifier_request_sha256,
        attestation.verifier_call_sha256,
        attestation.verifier_call_key_sha256,
        attestation.verifier_schema_sha256,
        attestation.verifier_prompt_sha256,
        attestation.verifier_messages_sha256,
    )
    if (
        attestation.source != GENERATED_ATTESTATION_SOURCE
        or attestation.status != "VALID"
        or attestation.reason_code != VALID_ATTESTATION_REASON
        or attestation.verdict_valid_is_exactly_true is not True
        or attestation.final_session_index is None
        or attestation.final_session_index < 1
        or attestation.final_attempt_index is None
        or attestation.journal_attempt_id is None
        or attestation.verifier_call_id is None
        or not all(_is_sha256(digest) for digest in required_digests)
        or attestation.final_draft_sha256
        != attestation.final_memory_latent_sha256
    ):
        return False, "ATTESTATION_MISMATCH"
    if not _has_replay_authority(replay_capability):
        return False, "MISSING_REPLAY_CAPABILITY"
    try:
        parsed_capability = parse_replay_capability(
            replay_capability.artifact()
        )
    except (AttributeError, TypeError, ValueError):
        return False, "MISSING_REPLAY_CAPABILITY"
    if parsed_capability.artifact() != replay_capability.artifact():
        return False, "MISSING_REPLAY_CAPABILITY"
    if (
        replay_capability.attempt_id != attestation.journal_attempt_id
        or replay_capability.verifier_call_id != attestation.verifier_call_id
        or replay_capability.verifier_request_sha256
        != attestation.verifier_request_sha256
        or replay_capability.verifier_response_sha256
        != attestation.verifier_response_sha256
        or replay_capability.attestation_sha256
        != _digest_json(attestation.as_dict())
        or replay_capability.latent_sha256 != _digest_json(latent_abstraction)
    ):
        return False, "ATTESTATION_MISMATCH"
    return True, "AUTHORIZED"


@dataclass(frozen=True)
class LatentFirewallDecision:
    status: str
    reason_code: str
    trait_id: str | None
    domain: str | None
    vlt_method: str
    slot: str | None
    value: str | bool | None
    source_latent_sha256: str
    ontology_sha256: str
    schema_scope_sha256: str
    selected_domain: str | None
    supporting_domains: tuple[str, ...]
    supporting_hypothesis_digests: tuple[str, ...]
    supporting_evidence_sha256: str
    opposing_hypothesis_digests: tuple[str, ...]

    @property
    def authorized(self) -> bool:
        return self.status == "ALLOW"

    def artifact(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "method": self.vlt_method,
            "status": self.status,
            "reason_code": self.reason_code,
            "trait_id": self.trait_id,
            "domain": self.domain,
            "slot": self.slot,
            "value": self.value,
            "source_latent_sha256": self.source_latent_sha256,
            "ontology_sha256": self.ontology_sha256,
            "schema_scope_sha256": self.schema_scope_sha256,
            "selected_domain": self.selected_domain,
            "supporting_domains": list(self.supporting_domains),
            "supporting_hypothesis_digests": list(
                self.supporting_hypothesis_digests
            ),
            "supporting_evidence_sha256": self.supporting_evidence_sha256,
            "opposing_hypothesis_digests": list(
                self.opposing_hypothesis_digests
            ),
        }

    def prompt_tuple(self) -> dict[str, Any] | None:
        if not self.authorized:
            return None
        return {
            "domain": self.domain,
            "slot": self.slot,
            "trait_id": self.trait_id,
            "value": self.value,
        }


def validate_latent_trait_ontology_v1(ontology: Any) -> dict[str, Any]:
    if not isinstance(ontology, dict):
        raise ValueError("latent trait ontology must be an object")
    if _digest_json(ontology) != VLT1_LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256:
        raise ValueError("latent trait ontology differs from the frozen v1 contract")
    if ontology.get("candidate_id") != "ecpr_v1":
        raise ValueError("latent trait ontology candidate mismatch")
    if ontology.get("candidate_revision") != VLT1_CANDIDATE_REVISION:
        raise ValueError("latent trait ontology v1 revision mismatch")
    return ontology


def validate_latent_trait_ontology_v2(ontology: Any) -> dict[str, Any]:
    if not isinstance(ontology, dict):
        raise ValueError("latent trait ontology must be an object")
    if _digest_json(ontology) != VLT2_LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256:
        raise ValueError("latent trait ontology differs from the frozen v2 contract")
    if ontology.get("candidate_id") != "ecpr_v1":
        raise ValueError("latent trait ontology candidate mismatch")
    if ontology.get("candidate_revision") != VLT2_CANDIDATE_REVISION:
        raise ValueError("latent trait ontology v2 revision mismatch")
    return ontology


def validate_latent_trait_ontology_v3(ontology: Any) -> dict[str, Any]:
    if not isinstance(ontology, dict):
        raise ValueError("latent trait ontology must be an object")
    if _digest_json(ontology) != VLT3_LATENT_TRAIT_ONTOLOGY_CANONICAL_SHA256:
        raise ValueError("latent trait ontology differs from the frozen v3 contract")
    if ontology.get("candidate_id") != "ecpr_v1":
        raise ValueError("latent trait ontology candidate mismatch")
    if ontology.get("candidate_revision") != VLT3_CANDIDATE_REVISION:
        raise ValueError("latent trait ontology v3 revision mismatch")
    if ontology.get("parent_ontology_sha256") != "be2070439b5928d6f2b1468ea0b8b63c0a42f48b578b4755a6ef949880fe005b":
        raise ValueError("latent trait ontology v3 parent mismatch")
    if ontology.get("mapping_identity") != ["trait_id", "domain"]:
        raise ValueError("latent trait ontology v3 mapping identity mismatch")
    if ontology.get("schema_keys") != ["single", "multi"]:
        raise ValueError("latent trait ontology v3 schema-key contract mismatch")
    source_prompt = ontology.get("provenance", {}).get(
        "original_source_prompt"
    )
    if (
        not isinstance(source_prompt, dict)
        or source_prompt.get("repository_relative_path")
        != "ours_memory2/ours_memory2/prompts.py"
        or source_prompt.get("sha256")
        != "2a7e69a3ab13b40727793de7b935b4643c14069983f5bc83900e64707d3ce411"
        or source_prompt.get("runtime_dependency") is not False
    ):
        raise ValueError("latent trait ontology v3 source prompt provenance mismatch")
    expected_correspondence = [
        {
            "internal_name": "BUDGET",
            "paper_name": "Budget",
            "kind": "trait_group",
            "status": "included",
        },
        {
            "internal_name": "TRAVEL_PARTY",
            "paper_name": "Travel",
            "kind": "trait_group",
            "status": "included",
        },
        {
            "internal_name": "SOLO_USAGE",
            "paper_name": "solo",
            "kind": "trait",
            "status": "included",
        },
        {
            "internal_name": "GROUP_USAGE",
            "paper_name": "group",
            "kind": "trait",
            "status": "excluded",
        },
    ]
    if ontology.get("paper_name_correspondence") != expected_correspondence:
        raise ValueError("latent trait ontology paper-name correspondence mismatch")
    active_trait_ids = {str(item.get("trait_id")) for item in ontology["traits"]}
    active_group_members = {
        str(member)
        for group in ontology["trait_groups"].values()
        for member in group.get("members", [])
    }
    if (
        "GROUP_USAGE" in ontology["trait_order"]
        or "GROUP_USAGE" in active_trait_ids
        or "GROUP_USAGE" in active_group_members
        or "GROUP_USAGE" in ontology["typed_support_registry"]
        or any(
            mapping.get("trait_id") == "GROUP_USAGE"
            for mapping in ontology["target_mappings"]
        )
    ):
        raise ValueError("GROUP_USAGE must remain correspondence-only and excluded")
    for entries in ontology["typed_support_registry"].values():
        for entry in entries:
            literal = entry.get("exact_literal")
            if (
                entry.get("match_policy") != "exact_literal"
                or not isinstance(literal, dict)
                or set(literal) != {"type", "value"}
                or literal.get("type") != "string"
                or type(literal.get("value")) is not str
            ):
                raise ValueError("VLT3 typed support must use exact string source literals")
    query_policy = ontology["query_constraint_policy"]
    components = query_policy.get("registry_component_provenance")
    expected_components = {
        "slot_aliases",
        "explicit_value_phrases",
        "numeric_word_tokens",
        "count_alias_distance_tokens",
    }
    if (
        query_policy.get("count_alias_distance_tokens") != 3
        or not isinstance(components, dict)
        or set(components) != expected_components
        or components["count_alias_distance_tokens"].get("frozen_value") != 3
        or any(
            component.get("freeze") != "vlt3_pre_evaluation_append_only"
            or component.get("held_out_or_current_target_query_or_test_use")
            != "forbidden"
            for component in components.values()
        )
    ):
        raise ValueError("VLT3 query registry provenance contract mismatch")
    return ontology


def validate_latent_trait_ontology(ontology: Any) -> dict[str, Any]:
    return validate_latent_trait_ontology_v3(ontology)


def _validate_supported_latent_trait_ontology(ontology: Any) -> dict[str, Any]:
    if isinstance(ontology, dict) and ontology.get("candidate_revision") == "vlt2":
        return validate_latent_trait_ontology_v2(ontology)
    return validate_latent_trait_ontology_v3(ontology)


def allowed_domains(ontology: Any) -> tuple[str, ...]:
    validated = _validate_supported_latent_trait_ontology(ontology)
    return tuple(dict.fromkeys(mapping["domain"] for mapping in validated["target_mappings"]))


def _tokens(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(_TOKEN_RE.findall(normalized))


def _contains_phrase(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    width = len(phrase)
    return bool(width) and any(
        tokens[index : index + width] == phrase
        for index in range(len(tokens) - width + 1)
    )


@dataclass(frozen=True)
class QueryConstraintMask:
    status: str
    selected_domain: str | None
    schema_key: str
    query_mode: str
    constrained_slots: tuple[str, ...]
    query_sha256: str
    user_payload_sha256: str | None
    schema_sha256: str
    ontology_sha256: str

    def artifact(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": "ontology_frozen_query_constraint_mask_v1",
            "status": self.status,
            "selected_domain": self.selected_domain,
            "schema_key": self.schema_key,
            "query_mode": self.query_mode,
            "constrained_slots": list(self.constrained_slots),
            "query_sha256": self.query_sha256,
            "user_payload_sha256": self.user_payload_sha256,
            "schema_sha256": self.schema_sha256,
            "ontology_sha256": self.ontology_sha256,
        }


def _count_near_alias(
    tokens: tuple[str, ...],
    constraint: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> bool:
    if constraint.get("detect_count_near_slot_alias") is not True:
        return False
    numeric_words = {str(item) for item in policy["numeric_word_tokens"]}
    numeric_indices = {
        index
        for index, token in enumerate(tokens)
        if token in numeric_words or any(character.isdigit() for character in token)
    }
    if not numeric_indices:
        return False
    limit = int(policy["count_alias_distance_tokens"])
    for alias in constraint["slot_aliases"]:
        phrase = _tokens(str(alias))
        width = len(phrase)
        if not width:
            continue
        for start in range(len(tokens) - width + 1):
            if tokens[start : start + width] != phrase:
                continue
            end = start + width
            for index in numeric_indices:
                distance = start - index - 1 if index < start else index - end if index >= end else 0
                if distance <= limit:
                    return True
    return False


def build_query_constraint_mask(*, current_query: str, query_mode: str, selected_domain: str | None, schema_key: str, schema: Any, ontology: Any) -> QueryConstraintMask:
    validated = validate_latent_trait_ontology_v3(ontology)
    if not isinstance(current_query, str):
        raise ValueError("current query must be a string")
    if query_mode not in {"singleturn", "multiturn"}:
        raise ValueError("current query mode is invalid")
    if schema_key not in validated["schema_keys"]:
        raise ValueError("current query schema key is invalid")
    user_payload = extract_user_payload(current_query, query_mode)
    query_tokens = _tokens(user_payload) if user_payload is not None else ()
    policy = validated["query_constraint_policy"]
    constrained: set[str] = set()
    if selected_domain is not None:
        for constraint in policy["mapping_constraints"]:
            if constraint["domain"] != selected_domain:
                continue
            mappings = [
                mapping
                for mapping in validated["target_mappings"]
                if mapping["trait_id"] == constraint["trait_id"] and mapping["domain"] == constraint["domain"]
            ]
            if len(mappings) != 1:
                raise ValueError("query constraint lacks one exact trait-domain mapping")
            mapping = mappings[0]
            expected_ref = {"trait_id": mapping["trait_id"], "domain": mapping["domain"]}
            if (
                constraint.get("target_mapping_ref") != expected_ref
                or constraint.get("slot") != mapping["slot"]
                or constraint.get("schema_keys") != mapping["schema_keys"]
            ):
                raise ValueError("query constraint target mapping binding mismatch")
            if schema_key not in constraint["schema_keys"]:
                continue
            if user_payload is None:
                constrained.add(str(mapping["slot"]))
                continue
            explicit_value = any(
                _contains_phrase(query_tokens, _tokens(str(phrase)))
                for phrase in constraint["explicit_value_phrases"]
            )
            if explicit_value or _count_near_alias(query_tokens, constraint, policy):
                constrained.add(str(mapping["slot"]))
    if user_payload is None:
        status = "USER_PAYLOAD_UNAVAILABLE_FAIL_CLOSED"
    elif constrained:
        status = "MASKED_EXPLICIT_TARGET_SLOTS"
    else:
        status = "NO_CONFIDENT_EXPLICIT_CONSTRAINT"
    return QueryConstraintMask(
        status=status,
        selected_domain=selected_domain,
        schema_key=schema_key,
        query_mode=query_mode,
        constrained_slots=tuple(sorted(constrained)),
        query_sha256=sha256_text(current_query),
        user_payload_sha256=sha256_text(user_payload) if user_payload is not None else None,
        schema_sha256=_digest_json(schema),
        ontology_sha256=_digest_json(validated),
    )


def _unsafe_or_trait(text: str, ontology: dict[str, Any]) -> tuple[str | None, str]:
    if "\n" in text or "\r" in text:
        return None, "UNSAFE_MULTILINE"
    normalized = unicodedata.normalize("NFKC", text)
    if len(normalized) > int(ontology["maximum_normalized_codepoints"]):
        return None, "OVER_LENGTH_CAP"
    if any(
        unicodedata.category(character) in {"Zl", "Zp"}
        for character in normalized
    ):
        return None, "UNSAFE_MULTILINE"
    if any(
        unicodedata.category(character) in {"Cc", "Cf"}
        or unicodedata.bidirectional(character)
        in {"LRE", "RLE", "LRO", "RLO", "PDF", "LRI", "RLI", "FSI", "PDI"}
        for character in normalized
    ):
        return None, "UNSAFE_CONTROL_OR_FORMAT"
    quote_like = set(ontology["quote_like_characters"])
    if any(
        character in quote_like
        or unicodedata.category(character) in {"Pi", "Pf"}
        for character in normalized
    ):
        return None, "UNSAFE_QUOTE_OR_BACKTICK"
    tokens = _tokens(normalized)
    negation_tokens = set(ontology["negation_tokens"])
    if any(token in negation_tokens for token in tokens):
        return None, "NEGATION_PRESENT"
    negatable_heads = set(ontology.get("negatable_heads", ("cost", "price", "budget", "cheap", "inexpensive", "expensive", "affordable")))
    if any(
        token.startswith(prefix)
        and token[len(prefix) :] in negatable_heads
        for token in tokens
        for prefix in ontology["affix_negation_prefixes"]
    ):
        return None, "NEGATION_PRESENT"
    if any(
        _contains_phrase(tokens, _tokens(phrase))
        for phrase in ontology["instruction_phrases"]
    ):
        return None, "INSTRUCTION_LIKE"
    matched = {
        trait["trait_id"]
        for trait in ontology["traits"]
        if any(
            _contains_phrase(tokens, _tokens(phrase))
            for phrase in trait["strong_phrases"]
        )
    }
    if not matched:
        return None, "NO_TRAIT_MATCH"
    if len(matched) != 1:
        return None, "AMBIGUOUS_TRAIT"
    return next(iter(matched)), "AUTHORIZED"


def _schema_function(schema: Any, domain: str) -> dict[str, Any] | None:
    tools = schema.get("tools", []) if isinstance(schema, dict) else schema
    if not isinstance(tools, list):
        return None
    matches: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if function.get("name") == domain:
            matches.append(function)
    return matches[0] if len(matches) == 1 else None


def _schema_scope(
    schema: Any,
    selected_domain: str | None,
    preference_slots: Mapping[str, Sequence[str]],
) -> tuple[dict[str, Any] | None, str]:
    function = (
        _schema_function(schema, selected_domain)
        if isinstance(selected_domain, str)
        else None
    )
    payload = {
        "selected_domain": selected_domain,
        "function": function,
        "preference_slots": sorted(
            str(slot) for slot in preference_slots.get(selected_domain or "", ())
        ),
    }
    return function, _digest_json(payload)


def _vlt3_mapped_value(
    ontology: dict[str, Any],
    trait_id: str,
    selected_domain: str,
    function: dict[str, Any] | None,
    preference_slots: Mapping[str, Sequence[str]],
    schema_key: str | None,
) -> tuple[str | None, str | bool | None, str]:
    matches = [
        item
        for item in ontology["target_mappings"]
        if item["trait_id"] == trait_id and item["domain"] == selected_domain
    ]
    if not matches:
        return None, None, "NO_DOMAIN_MAPPING"
    if len(matches) != 1:
        return None, None, "SCHEMA_MISMATCH"
    mapping = matches[0]
    slot = str(mapping["slot"])
    if schema_key not in mapping["schema_keys"]:
        return slot, None, "SCHEMA_MISMATCH"
    if slot not in {str(value) for value in preference_slots.get(selected_domain, ())} or function is None:
        return slot, None, "SCHEMA_MISMATCH"
    parameters = function.get("parameters", {})
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    specification = properties.get(slot) if isinstance(properties, dict) else None
    if not isinstance(specification, dict) or specification.get("type") != mapping["schema_type"]:
        return slot, None, "SCHEMA_MISMATCH"
    literal = mapping.get("exact_literal")
    if not isinstance(literal, dict) or set(literal) != {"type", "value"}:
        return slot, None, "SCHEMA_MISMATCH"
    value = literal["value"]
    literal_type = literal["type"]
    if literal_type != mapping["schema_type"]:
        return slot, None, "SCHEMA_MISMATCH"
    if literal_type == "string" and type(value) is not str:
        return slot, None, "SCHEMA_MISMATCH"
    if literal_type == "boolean" and type(value) is not bool:
        return slot, None, "SCHEMA_MISMATCH"
    if literal_type not in {"string", "boolean"}:
        return slot, None, "SCHEMA_MISMATCH"
    available = specification.get("enum")
    if available is not None and (not isinstance(available, list) or value not in available):
        return slot, None, "SCHEMA_MISMATCH"
    return slot, value, "AUTHORIZED"


def _mapped_value(
    ontology: dict[str, Any],
    trait_id: str,
    selected_domain: str,
    function: dict[str, Any] | None,
    preference_slots: Mapping[str, Sequence[str]],
    schema_key: str | None = None,
) -> tuple[str | None, str | bool | None, str]:
    if ontology.get("candidate_revision") == VLT3_CANDIDATE_REVISION:
        return _vlt3_mapped_value(
            ontology,
            trait_id,
            selected_domain,
            function,
            preference_slots,
            schema_key,
        )
    mapping = next(
        (
            item
            for item in ontology["target_mappings"]
            if item["domain"] == selected_domain
        ),
        None,
    )
    if mapping is None:
        return None, None, "NO_DOMAIN_MAPPING"
    slot = str(mapping["slot"])
    if slot not in {str(value) for value in preference_slots.get(selected_domain, ())}:
        return slot, None, "SCHEMA_MISMATCH"
    if function is None:
        return slot, None, "SCHEMA_MISMATCH"
    parameters = function.get("parameters", {})
    properties = parameters.get("properties", {}) if isinstance(parameters, dict) else {}
    specification = properties.get(slot) if isinstance(properties, dict) else None
    if not isinstance(specification, dict) or specification.get("type") != mapping["schema_type"]:
        return slot, None, "SCHEMA_MISMATCH"
    if "trait_values" in mapping:
        if trait_id not in mapping["trait_values"]:
            return slot, None, "NO_DOMAIN_MAPPING"
        value = mapping["trait_values"][trait_id]
        if mapping["schema_type"] == "boolean":
            return (
                (slot, value, "AUTHORIZED")
                if type(value) is bool
                else (slot, None, "SCHEMA_MISMATCH")
            )
        available = specification.get("enum")
        if (
            not isinstance(value, str)
            or not isinstance(available, list)
            or value not in available
        ):
            return slot, None, "SCHEMA_MISMATCH"
        return slot, value, "AUTHORIZED"
    direction = mapping.get("trait_directions", {}).get(trait_id)
    if direction is None:
        return slot, None, "NO_DOMAIN_MAPPING"
    available = specification.get("enum")
    order = mapping.get("ordered_values")
    if (
        not isinstance(available, list)
        or not available
        or any(not isinstance(value, str) for value in available)
        or len(set(available)) != len(available)
        or any(value not in order for value in available)
    ):
        return slot, None, "SCHEMA_MISMATCH"
    ordered_available = [value for value in order if value in available]
    if not ordered_available:
        return slot, None, "SCHEMA_MISMATCH"
    if direction == "lowest_available":
        return slot, ordered_available[0], "AUTHORIZED"
    if direction == "highest_available":
        return slot, ordered_available[-1], "AUTHORIZED"
    return slot, None, "SCHEMA_MISMATCH"


def validate_vlt_schema_contract(
    ontology: Any,
    schemas: Mapping[str, Any],
    preference_slots: Mapping[str, Sequence[str]],
) -> None:
    validated = _validate_supported_latent_trait_ontology(ontology)
    is_vlt3 = validated.get("candidate_revision") == VLT3_CANDIDATE_REVISION
    if is_vlt3 and set(schemas) != set(validated["schema_keys"]):
        raise ValueError("VLT3 schemas must use the exact frozen keys")
    mapping_ids = [(item.get("trait_id"), item.get("domain")) for item in validated["target_mappings"]]
    if is_vlt3 and len(mapping_ids) != len(set(mapping_ids)):
        raise ValueError("VLT3 trait-domain mapping identity is not unique")
    for schema_name, schema in schemas.items():
        if not isinstance(schema_name, str):
            raise ValueError("VLT schema key must be a string")
        for mapping in validated["target_mappings"]:
            if is_vlt3 and schema_name not in mapping["schema_keys"]:
                continue
            domain = mapping["domain"]
            function = _schema_function(schema, domain)
            if function is None:
                raise ValueError(f"VLT schema lacks mapped domain: {schema_name}/{domain}")
            traits = (
                [mapping["trait_id"]]
                if is_vlt3
                else (
                    mapping.get("trait_directions") or mapping.get("trait_values") or {}
                )
            )
            for trait_id in traits:
                _slot, _value, reason = _mapped_value(
                    validated,
                    trait_id,
                    domain,
                    function,
                    preference_slots,
                    schema_name,
                )
                if reason != "AUTHORIZED":
                    raise ValueError(
                        f"VLT mapping is not schema valid: {schema_name}/{domain}/{trait_id}"
                    )


def _policy_qualified(hypothesis: Any, policy: CandidatePolicy) -> bool:
    if not isinstance(hypothesis, dict):
        return False
    support = hypothesis.get("support")
    counter = hypothesis.get("counterevidence")
    confidence = hypothesis.get("confidence")
    if (
        type(support) is not int
        or type(counter) is not int
        or support < 0
        or counter < 0
        or isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
    ):
        return False
    total = support + counter
    conflict_ratio = counter / total if total else 1.0
    return (
        support >= policy.minimum_support
        and float(confidence) >= policy.minimum_confidence
        and conflict_ratio <= policy.maximum_conflict_ratio
    )


def _validated_source_literal(hypothesis: Mapping[str, Any]) -> dict[str, Any] | None:
    literal = hypothesis.get("source_literal")
    if not isinstance(literal, dict) or set(literal) != {"type", "value"}:
        return None
    value_type = literal.get("type")
    raw_value = literal.get("value")
    type_matches = {
        "null": raw_value is None,
        "boolean": type(raw_value) is bool,
        "integer": type(raw_value) is int,
        "number": type(raw_value) is float,
        "string": type(raw_value) is str,
    }
    if type_matches.get(value_type) is not True:
        return None
    if hypothesis.get("value_type") != value_type:
        return None
    normalized = hypothesis.get("value")
    if not isinstance(normalized, str) or normalized != normalize_value(raw_value):
        return None
    return {"type": value_type, "value": raw_value}


def _support_trait(
    ontology: dict[str, Any], hypothesis: dict[str, Any]
) -> str | None:
    is_vlt3 = ontology.get("candidate_revision") == VLT3_CANDIDATE_REVISION
    source_literal = _validated_source_literal(hypothesis) if is_vlt3 else None
    if is_vlt3 and source_literal is None:
        return None
    domain = str(hypothesis.get("domain", ""))
    slot = str(hypothesis.get("slot", ""))
    raw_value = hypothesis.get("value")
    value = normalize_value(raw_value)
    matches: list[str] = []
    for trait_id, entries in ontology["typed_support_registry"].items():
        for entry in entries:
            if entry["domain"] != domain or entry["slot"] != slot:
                continue
            if entry.get("match_policy") == "exact_literal":
                literal = entry.get("exact_literal", {})
                if is_vlt3:
                    matched = source_literal == literal
                else:
                    matched = (
                        hypothesis.get("value_type") == literal.get("type")
                        and raw_value == literal.get("value")
                    )
            else:
                matched = value in {normalize_value(item) for item in entry["values"]}
            if matched:
                matches.append(trait_id)
    return matches[0] if len(matches) == 1 else None


def _hypothesis_value_identity(
    hypothesis: Mapping[str, Any], *, require_declared_type: bool
) -> tuple[str, str]:
    value_type = hypothesis.get("value_type")
    if not isinstance(value_type, str):
        if require_declared_type:
            value_type = "undeclared"
        else:
            raw = hypothesis.get("value")
            if type(raw) is bool:
                value_type = "boolean"
            elif type(raw) is int:
                value_type = "integer"
            elif type(raw) is float:
                value_type = "number"
            elif type(raw) is str:
                value_type = "string"
            else:
                value_type = "other"
    return value_type, normalize_value(hypothesis.get("value"))


def _hypothesis_summary(
    hypothesis: dict[str, Any], *, include_source_literal: bool = False
) -> dict[str, Any]:
    summary = {
        "domain": str(hypothesis.get("domain", "")),
        "slot": str(hypothesis.get("slot", "")),
        "value": normalize_value(hypothesis.get("value")),
        "value_type": hypothesis.get("value_type"),
        "support": hypothesis.get("support"),
        "counterevidence": hypothesis.get("counterevidence"),
        "confidence": hypothesis.get("confidence"),
        "last_seen": hypothesis.get("last_seen"),
        "provenance_digests": sorted(
            str(item.get("call_digest", ""))
            for item in hypothesis.get("provenance", [])
            if isinstance(item, dict)
        ),
    }
    if include_source_literal:
        summary["source_literal"] = _validated_source_literal(hypothesis)
    return summary


def _qualified_evidence(
    ontology: dict[str, Any],
    typed_hypotheses: Sequence[Any],
    policy: CandidatePolicy,
) -> list[tuple[str, dict[str, Any], str]]:
    result: list[tuple[str, dict[str, Any], str]] = []
    for hypothesis in typed_hypotheses:
        if not _policy_qualified(hypothesis, policy):
            continue
        assert isinstance(hypothesis, dict)
        trait_id = _support_trait(ontology, hypothesis)
        if trait_id is None:
            continue
        summary = _hypothesis_summary(
            hypothesis,
            include_source_literal=ontology.get("candidate_revision") == VLT3_CANDIDATE_REVISION,
        )
        result.append((trait_id, summary, _digest_json(summary)))
    return result


def qualified_trait_diagnostics(
    ontology: Any,
    typed_hypotheses: Sequence[Any],
    policy: CandidatePolicy,
) -> list[dict[str, Any]]:
    """Expose production matcher results without a second diagnostic ontology."""
    validated = _validate_supported_latent_trait_ontology(ontology)
    return [
        {
            "trait_id": trait_id,
            "hypothesis": summary,
            "hypothesis_sha256": digest,
        }
        for trait_id, summary, digest in _qualified_evidence(
            validated, typed_hypotheses, policy
        )
    ]

def _none_decision(
    reason_code: str,
    *,
    source_latent_sha256: str,
    ontology_sha256: str,
    schema_scope_sha256: str,
    selected_domain: str | None,
    vlt_method: str,
    trait_id: str | None = None,
    domain: str | None = None,
    slot: str | None = None,
    value: str | bool | None = None,
    supporting: Sequence[tuple[str, dict[str, Any], str]] = (),
    opposing: Sequence[tuple[str, dict[str, Any], str]] = (),
) -> LatentFirewallDecision:
    if reason_code not in VLT_REASON_CODES or reason_code == "AUTHORIZED":
        raise ValueError("invalid VLT rejection reason")
    selected_by_domain: dict[str, tuple[dict[str, Any], str]] = {}
    for _trait, summary, digest in sorted(
        supporting,
        key=lambda item: (
            item[1]["domain"],
            -float(item[1].get("confidence") or 0.0),
            -int(item[1].get("support") or 0),
            item[2],
        ),
    ):
        selected_by_domain.setdefault(str(summary["domain"]), (summary, digest))
    summaries = [selected_by_domain[key][0] for key in sorted(selected_by_domain)]
    return LatentFirewallDecision(
        status="NONE",
        reason_code=reason_code,
        trait_id=trait_id,
        vlt_method=vlt_method,
        domain=domain,
        slot=slot,
        value=value,
        source_latent_sha256=source_latent_sha256,
        ontology_sha256=ontology_sha256,
        schema_scope_sha256=schema_scope_sha256,
        selected_domain=selected_domain,
        supporting_domains=tuple(sorted(selected_by_domain)),
        supporting_hypothesis_digests=tuple(
            selected_by_domain[key][1] for key in sorted(selected_by_domain)
        ),
        supporting_evidence_sha256=_digest_json(summaries),
        opposing_hypothesis_digests=tuple(sorted(item[2] for item in opposing)),
    )


def decide_latent_transfer(
    *,
    latent_abstraction: Any,
    latent_attestation: Any,
    pqr_status: str,
    selected_domain: str | None,
    schema_key: str | None = None,
    current_query: str | None = None,
    query_mode: str | None = None,
    schema: Any,
    preference_slots: Mapping[str, Sequence[str]],
    typed_hypotheses: Sequence[Any],
    routed_hypotheses: Sequence[Mapping[str, Any]],
    policy: CandidatePolicy,
    ontology: Any,
    ontology_sha256: str | None = None,
    ablations: set[str] | None = None,
    replay_capability: JournalReplayCapability | None = None,
    query_constraint_mask: QueryConstraintMask | None = None,
) -> LatentFirewallDecision:
    """Return one closed transfer tuple or a finite fail-closed NONE decision."""
    validated = _validate_supported_latent_trait_ontology(ontology)
    semantic_ontology_sha256 = _digest_json(validated)
    is_vlt3 = validated.get("candidate_revision") == VLT3_CANDIDATE_REVISION
    if is_vlt3 and schema_key not in validated["schema_keys"]:
        raise ValueError("VLT3 decision requires a frozen schema key")
    expected_mask: QueryConstraintMask | None = None
    if is_vlt3:
        if not isinstance(current_query, str) or query_mode not in {"singleturn", "multiturn"}:
            raise ValueError("VLT3 decision requires the exact task query and mode")
        assert isinstance(schema_key, str)
        expected_mask = build_query_constraint_mask(
            current_query=current_query,
            query_mode=query_mode,
            selected_domain=selected_domain,
            schema_key=schema_key,
            schema=schema,
            ontology=validated,
        )
        if (
            type(query_constraint_mask) is not QueryConstraintMask
            or query_constraint_mask.artifact() != expected_mask.artifact()
        ):
            raise ValueError("VLT3 decision requires the exact ontology-frozen query mask")
    if ontology_sha256 is not None and not _is_sha256(ontology_sha256):
        raise ValueError("VLT ontology file digest must be SHA-256")
    bound_ontology_sha256 = ontology_sha256 or semantic_ontology_sha256
    source_latent_sha256 = _digest_json(latent_abstraction)
    function, schema_scope_sha256 = _schema_scope(
        schema, selected_domain, preference_slots
    )
    common = {
        "source_latent_sha256": source_latent_sha256,
        "ontology_sha256": bound_ontology_sha256,
        "vlt_method": VLT3_METHOD if is_vlt3 else VLT2_METHOD,
        "schema_scope_sha256": schema_scope_sha256,
        "selected_domain": selected_domain,
    }
    active_ablations = set(ablations or ())
    if "no_latent" in active_ablations:
        return _none_decision("NO_LATENT_ABLATION", **common)
    if "no_typed" in active_ablations:
        return _none_decision("NO_TYPED_ABLATION", **common)
    if pqr_status != "SELECT" or not isinstance(selected_domain, str):
        return _none_decision("PQR_ABSTAIN", **common)
    authorized, reason = _attestation_authorizes(
        latent_abstraction, latent_attestation, replay_capability
    )
    if not authorized:
        return _none_decision(reason, **common)
    implicit_pref = _implicit_pref(latent_abstraction)
    if implicit_pref is None:
        return _none_decision("INVALID_LATENT_SHAPE", **common)
    trait_id, reason = _unsafe_or_trait(implicit_pref, validated)
    if trait_id is None:
        return _none_decision(reason, **common)
    slot, value, reason = _mapped_value(
        validated,
        trait_id,
        selected_domain,
        function,
        preference_slots,
        schema_key,
    )
    if reason != "AUTHORIZED":
        return _none_decision(
            reason,
            trait_id=trait_id,
            domain=selected_domain,
            slot=slot,
            value=value,
            **common,
        )

    if (
        is_vlt3
        and isinstance(query_constraint_mask, QueryConstraintMask)
        and slot in query_constraint_mask.constrained_slots
    ):
        return _none_decision(
            "CURRENT_QUERY_EXPLICIT_CONSTRAINT",
            trait_id=trait_id,
            domain=selected_domain,
            slot=slot,
            value=value,
            **common,
        )

    qualified = _qualified_evidence(validated, typed_hypotheses, policy)
    supporting = [
        item
        for item in qualified
        if item[0] == trait_id and item[1]["domain"] != selected_domain
    ]
    trait_contracts = [
        item
        for item in validated["traits"]
        if item["trait_id"] == trait_id
    ]
    if is_vlt3 and len(trait_contracts) != 1:
        raise ValueError("VLT3 trait opposition contract is ambiguous")
    opposed_trait_ids = set(trait_contracts[0]["opposes"]) if is_vlt3 else {item[0] for item in qualified if item[0] != trait_id}
    opposing = [item for item in qualified if item[0] in opposed_trait_ids]
    if opposing:
        return _none_decision(
            "OPPOSING_TRAIT_SUPPORT",
            trait_id=trait_id,
            domain=selected_domain,
            slot=slot,
            value=value,
            supporting=supporting,
            opposing=opposing,
            **common,
        )
    support_domains = {str(item[1]["domain"]) for item in supporting}
    insufficient_support = (
        not support_domains if is_vlt3 else len(support_domains) < 2
    )
    if insufficient_support:
        return _none_decision(
            "INSUFFICIENT_TYPED_SUPPORT"
            if is_vlt3
            else "INSUFFICIENT_CROSS_DOMAIN_SUPPORT",
            trait_id=trait_id,
            domain=selected_domain,
            slot=slot,
            value=value,
            supporting=supporting,
            **common,
        )
    same_slot = [
        item
        for item in typed_hypotheses
        if isinstance(item, dict)
        and _policy_qualified(item, policy)
        and str(item.get("domain", "")) == selected_domain
        and str(item.get("slot", "")) == slot
    ]
    if is_vlt3:
        same_slot_values = {
            _hypothesis_value_identity(item, require_declared_type=True)
            for item in same_slot
        }
        mapped = ("boolean" if type(value) is bool else "string", normalize_value(value))
    else:
        same_slot_values = {normalize_value(item.get("value")) for item in same_slot}
        mapped = normalize_value(value)
    if same_slot_values:
        if len(same_slot_values) > 1:
            reason = "TARGET_SLOT_AMBIGUOUS"
        elif next(iter(same_slot_values)) == mapped:
            reason = "TARGET_SLOT_REDUNDANT"
        else:
            reason = "TARGET_SLOT_CONFLICT"
        return _none_decision(
            reason,
            trait_id=trait_id,
            domain=selected_domain,
            slot=slot,
            value=value,
            supporting=supporting,
            **common,
        )

    selected_by_domain: dict[str, tuple[dict[str, Any], str]] = {}
    for _trait, summary, digest in sorted(
        supporting,
        key=lambda item: (
            item[1]["domain"],
            -float(item[1].get("confidence") or 0.0),
            -int(item[1].get("support") or 0),
            item[2],
        ),
    ):
        selected_by_domain.setdefault(str(summary["domain"]), (summary, digest))
    summaries = [selected_by_domain[key][0] for key in sorted(selected_by_domain)]
    return LatentFirewallDecision(
        status="ALLOW",
        reason_code="AUTHORIZED",
        trait_id=trait_id,
        domain=selected_domain,
        slot=slot,
        vlt_method=VLT3_METHOD if is_vlt3 else VLT2_METHOD,
        value=value,
        source_latent_sha256=source_latent_sha256,
        ontology_sha256=bound_ontology_sha256,
        schema_scope_sha256=schema_scope_sha256,
        selected_domain=selected_domain,
        supporting_domains=tuple(sorted(selected_by_domain)),
        supporting_hypothesis_digests=tuple(
            selected_by_domain[key][1] for key in sorted(selected_by_domain)
        ),
        supporting_evidence_sha256=_digest_json(summaries),
        opposing_hypothesis_digests=(),
    )


def canonical_vlt_prompt_tuple(
    decision: LatentFirewallDecision,
    *,
    ontology: Any = None,
    schema: Any = None,
    preference_slots: Mapping[str, Sequence[str]] | None = None,
    selected_domain: str | None = None,
    schema_key: str | None = None,
    current_query: str | None = None,
    query_mode: str | None = None,
    query_constraint_mask: QueryConstraintMask | None = None,
    ontology_sha256: str | None = None,
) -> str | None:
    """Fail-closed serializer with an independent ontology/schema recheck."""
    if not isinstance(decision, LatentFirewallDecision) or not decision.authorized:
        return None
    if (
        ontology is None
        or schema is None
        or preference_slots is None
        or not isinstance(selected_domain, str)
    ):
        return None
    try:
        validated = _validate_supported_latent_trait_ontology(ontology)
        is_vlt3 = validated.get("candidate_revision") == VLT3_CANDIDATE_REVISION
        expected_mask: QueryConstraintMask | None = None
        if is_vlt3:
            if (
                schema_key not in validated["schema_keys"]
                or not isinstance(current_query, str)
                or query_mode not in {"singleturn", "multiturn"}
            ):
                raise ValueError("missing VLT3 serializer query binding")
            expected_mask = build_query_constraint_mask(
                current_query=current_query,
                query_mode=query_mode,
                selected_domain=selected_domain,
                schema_key=schema_key,
                schema=schema,
                ontology=validated,
            )
            if (
                type(query_constraint_mask) is not QueryConstraintMask
                or query_constraint_mask.artifact() != expected_mask.artifact()
            ):
                raise ValueError("VLT3 serializer query mask mismatch")

        function, schema_scope_sha256 = _schema_scope(
            schema, selected_domain, preference_slots
        )
        slot, mapped, reason = _mapped_value(
            validated,
            str(decision.trait_id),
            selected_domain,
            function,
            preference_slots,
            schema_key,
        )
    except (KeyError, TypeError, ValueError):
        return None
    expected_ontology_sha256 = ontology_sha256 or _digest_json(validated)
    if (
        (
            is_vlt3
            and expected_mask is not None
            and slot in expected_mask.constrained_slots
        )
        or reason != "AUTHORIZED"
        or decision.reason_code != "AUTHORIZED"
        or decision.domain != selected_domain
        or decision.selected_domain != selected_domain
        or decision.slot != slot
        or type(decision.value) is not type(mapped)
        or decision.value != mapped
        or decision.schema_scope_sha256 != schema_scope_sha256
        or decision.ontology_sha256 != expected_ontology_sha256
    ):
        return None
    value = decision.prompt_tuple()
    return canonical_json(value) if value is not None else None
