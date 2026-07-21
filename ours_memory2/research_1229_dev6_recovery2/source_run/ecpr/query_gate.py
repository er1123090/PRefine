"""Deterministic, target-free public query relevance gate."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any

from .parsing import schema_domain_slots


CANONICAL_DOMAINS = (
    "Banks",
    "Buses",
    "Events",
    "Flights",
    "Homes",
    "Hotels",
    "Media",
    "Movies",
    "Music",
    "RentalCars",
    "Restaurants",
    "RideSharing",
    "Travel",
    "Weather",
)
MINIMUM_TOP_SCORE = 3
MINIMUM_MARGIN = 2
STRONG_DOMAIN_SCORE = 3
ROUTING_GATE_CONTRACT = {
    "method": "public_query_relevance_gate_v1",
    "ontology_path": "configs/public_domain_ontology.json",
    "normalization": "NFKC_then_casefold_then_whitespace_collapse",
    "multiturn_extraction": "recognized_role_tags_present=>nonempty_User_payloads_only_else_whole_query;malformed_or_no_User=>ABSTAIN",
    "matching": "contiguous_unicode_token_boundary_phrase_match",
    "domain_score": "maximum_matching_alias_weight_never_sum",
    "alias_weight_classes": {
        "canonical_or_domain_specific": 4,
        "generic": 3,
        "weak": 1,
    },
    "minimum_top_score": MINIMUM_TOP_SCORE,
    "minimum_margin": MINIMUM_MARGIN,
    "tie_policy": "ABSTAIN",
    "multi_intent_policy": "ABSTAIN_if_two_or_more_domains_score_at_least_3",
    "selection_policy": "SELECT_only_unique_top_at_least_3_with_margin_at_least_2",
    "candidate_scope": "selected_schema_domain_only;selection_never_implies_requested_operation",
    "latent_policy": "omit_on_ABSTAIN_or_zero_surviving_typed_evidence",
}
CANDIDATE_MEMORY_SCOPE_CONTRACT = {
    "method": "candidate_typed_routed_evidence_only_v1",
    "selection_authority": "public_query_relevance_gate_v1",
    "selected_domain_scope": "selected_schema_domain_only",
    "serialized_evidence": "typed_hypotheses_only",
    "untyped_latent_serialization": "forbidden_in_all_candidate_prompts",
    "abstain_policy": "no_memory_evidence",
    "zero_surviving_typed_policy": "no_memory_evidence",
    "selection_implication": "never_implies_requested_domain_or_operation",
    "baseline_memory_scope": "unchanged_prefine_latent_abstraction",
    "no_latent_ablation_compatibility": "accepted_scratch_flag_no_runtime_effect",
}

_ROLE_RE = re.compile(
    r"^[ \t]*(User|Assistant|Tool|System)[ \t]*:[ \t]*(.*)$",
    re.IGNORECASE,
)
_ROLE_PREFIX_RE = re.compile(
    r"^[ \t]*(User|Assistant|Tool|System)\b",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)


@dataclass(frozen=True)
class GateDecision:
    status: str
    canonical_domain: str | None
    schema_domain: str | None
    top_score: int
    second_score: int
    scores: tuple[tuple[str, int], ...]

    @property
    def selected(self) -> bool:
        return self.status == "SELECT"


def normalize_text(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("query and aliases must be strings")
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(normalize_text(value)))


def _contains_phrase(tokens: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    width = len(phrase)
    return bool(width) and any(
        tokens[index : index + width] == phrase
        for index in range(len(tokens) - width + 1)
    )


def extract_user_payload(query: str, mode: str) -> str | None:
    """Return public user text; ``None`` is a fail-closed extraction result."""
    normalized_query = unicodedata.normalize("NFKC", query)
    if mode not in {"singleturn", "multiturn"}:
        return None
    if mode != "multiturn":
        return normalized_query
    lines = normalized_query.splitlines()
    matches = [_ROLE_RE.match(line) for line in lines]
    if any(
        _ROLE_PREFIX_RE.match(line) and match is None
        for line, match in zip(lines, matches)
    ):
        return None
    tagged = any(matches)
    if not tagged:
        return normalized_query
    first_tag = next(index for index, match in enumerate(matches) if match)
    if any(line.strip() for line in lines[:first_tag]):
        return None

    payloads: list[str] = []
    current_role: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        if current_role == "user":
            payload = "\n".join(current_lines).strip()
            if payload:
                payloads.append(payload)

    for line, match in zip(lines, matches):
        if match:
            flush()
            current_role = match.group(1).casefold()
            current_lines = [match.group(2)]
        elif current_role is not None:
            current_lines.append(line)
    flush()
    return "\n".join(payloads) if payloads else None


def validate_ontology(
    ontology: Any,
    schema: Any,
    *,
    allow_schema_subset: bool = False,
) -> tuple[dict[str, Any], ...]:
    if not isinstance(ontology, dict) or set(ontology) != {
        "schema_version",
        "kind",
        "weight_classes",
        "domains",
    }:
        raise ValueError("public domain ontology field contract violation")
    if ontology.get("schema_version") != 1:
        raise ValueError("unsupported public domain ontology schema version")
    if ontology.get("kind") != "public_domain_ontology":
        raise ValueError("public domain ontology kind mismatch")
    if ontology.get("weight_classes") != ROUTING_GATE_CONTRACT[
        "alias_weight_classes"
    ]:
        raise ValueError("public domain ontology weight classes mismatch")
    domains = ontology.get("domains")
    if not isinstance(domains, list) or len(domains) != len(CANONICAL_DOMAINS):
        raise ValueError("public domain ontology canonical domain count mismatch")

    schema_domains = set(schema_domain_slots(schema))
    canonical_seen: set[str] = set()
    schema_seen: set[str] = set()
    alias_seen: dict[tuple[str, ...], str] = {}
    validated: list[dict[str, Any]] = []
    for expected_canonical, domain in zip(CANONICAL_DOMAINS, domains):
        if not isinstance(domain, dict) or set(domain) != {
            "canonical",
            "schema_domain",
            "aliases",
        }:
            raise ValueError("public domain ontology domain shape violation")
        canonical = domain.get("canonical")
        schema_domain = domain.get("schema_domain")
        aliases = domain.get("aliases")
        if canonical != expected_canonical:
            raise ValueError("public domain ontology canonical ordering mismatch")
        if canonical in canonical_seen or not isinstance(schema_domain, str):
            raise ValueError("public domain ontology domain collision")
        if schema_domain in schema_seen:
            raise ValueError("public domain ontology schema-domain collision")
        if not allow_schema_subset and schema_domain not in schema_domains:
            raise ValueError(
                f"ontology domain is not contained in frozen schema: {schema_domain}"
            )
        if not isinstance(aliases, list) or not aliases:
            raise ValueError("public domain ontology aliases must be nonempty")

        validated_aliases: list[tuple[tuple[str, ...], int]] = []
        weights: set[int] = set()
        for alias in aliases:
            if not isinstance(alias, dict) or set(alias) != {"text", "weight"}:
                raise ValueError("public domain ontology alias shape violation")
            phrase = _tokens(alias.get("text"))
            weight = alias.get("weight")
            if not phrase or type(weight) is not int or weight not in {1, 3, 4}:
                raise ValueError("public domain ontology alias value violation")
            prior = alias_seen.setdefault(phrase, canonical)
            if prior != canonical or phrase in {
                item[0] for item in validated_aliases
            }:
                raise ValueError("public domain ontology normalized alias collision")
            validated_aliases.append((phrase, weight))
            weights.add(weight)
        if 4 not in weights:
            raise ValueError("each public domain requires a canonical/domain-specific alias")
        canonical_seen.add(canonical)
        schema_seen.add(schema_domain)
        validated.append(
            {
                "canonical": canonical,
                "schema_domain": schema_domain,
                "aliases": tuple(validated_aliases),
            }
        )
    if not schema_domains.issubset(schema_seen):
        missing = sorted(schema_domains - schema_seen)
        raise ValueError(f"frozen schema domain has no ontology mapping: {missing}")
    return tuple(validated)


def gate_public_query(
    query: str,
    mode: str,
    schema: Any,
    ontology: Any,
) -> GateDecision:
    domains = validate_ontology(ontology, schema, allow_schema_subset=True)
    active_domains = set(schema_domain_slots(schema))
    payload = extract_user_payload(query, mode)
    query_tokens = _tokens(payload) if payload is not None else ()
    scored: list[tuple[str, str, int]] = []
    for domain in domains:
        if domain["schema_domain"] not in active_domains:
            continue
        score = max(
            (
                weight
                for phrase, weight in domain["aliases"]
                if _contains_phrase(query_tokens, phrase)
            ),
            default=0,
        )
        scored.append((domain["canonical"], domain["schema_domain"], score))

    ranked = sorted(scored, key=lambda item: (-item[2], item[0]))
    if not ranked:
        return GateDecision("ABSTAIN", None, None, 0, 0, ())
    top_score = ranked[0][2]
    second_score = ranked[1][2] if len(ranked) > 1 else 0
    top_count = sum(score == top_score for _, _, score in ranked)
    strong_count = sum(score >= STRONG_DOMAIN_SCORE for _, _, score in ranked)
    scores = tuple((canonical, score) for canonical, _, score in scored)
    if (
        top_score >= MINIMUM_TOP_SCORE
        and top_count == 1
        and top_score - second_score >= MINIMUM_MARGIN
        and strong_count == 1
    ):
        canonical, schema_domain, _score = ranked[0]
        return GateDecision(
            "SELECT", canonical, schema_domain, top_score, second_score, scores
        )
    return GateDecision("ABSTAIN", None, None, top_score, second_score, scores)
