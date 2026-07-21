"""Frozen contracts shared by preparation, inference, evaluation, and audit."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "answer",
        "difficulty",
        "gold",
        "ground_truth",
        "label",
        "reference",
        "reference_ground_truth",
        "target",
        "target_domain",
        "explicit_slots",
        "api_calls_pref",
        "pref_group",
        "value_group",
    }
)

PREDICTION_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "case_key",
        "example_id",
        "mode",
        "arm",
        "llm_output",
        "status",
        "model_snapshot",
        "seed",
        "temperature",
        "max_tokens",
        "calls",
        "prompt_hash",
        "schema_hash",
        "memory_hash",
        "usage",
    }
)

TASK_FIELDS: Final[frozenset[str]] = frozenset(
    {"case_key", "example_id", "mode", "query", "schema_key"}
)

GOLD_FIELDS: Final[frozenset[str]] = frozenset(
    {"case_key", "example_id", "mode", "difficulty", "reference_ground_truth"}
)


@dataclass(frozen=True)
class CandidatePolicy:
    minimum_support: int = 2
    minimum_confidence: float = 0.67
    maximum_conflict_ratio: float = 0.34
    maximum_stored_hypotheses: int = 24
    maximum_routed_hypotheses: int = 6
    overlay_lexical_token_cap: int = 384
    memory_lexical_token_cap: int = 1536
    recency_tiebreak_only: bool = True
    raw_api_history_in_candidate_prompt: bool = False


@dataclass(frozen=True)
class InferenceBudget:
    temperature: float = 0.0
    seed: int = 2026071600
    max_tokens: int = 1024
    calls_per_case: int = 1
    retries: int = 0


ALLOWED_ABLATIONS: Final[frozenset[str]] = frozenset(
    {"no_latent", "no_typed", "no_counterevidence", "no_routing"}
)
