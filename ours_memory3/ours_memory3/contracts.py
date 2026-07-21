"""Pure public-data contracts for the ours_memory3 overlay."""

from __future__ import annotations

from dataclasses import dataclass


class PublicInputError(ValueError):
    """Raised when an input is malformed or attempts to carry private labels."""


@dataclass(frozen=True)
class OverlayPolicy:
    """Conservative thresholds fixed before a paired final evaluation."""

    minimum_independent_sessions: int = 2
    minimum_consensus: float = 0.75
    maximum_competing_sessions: int = 1
    minimum_support_margin: int = 2
    maximum_facts: int = 3
    maximum_value_characters: int = 80
    maximum_overlay_characters: int = 720

    def __post_init__(self) -> None:
        if self.minimum_independent_sessions < 2:
            raise PublicInputError("minimum_independent_sessions must be at least 2")
        if not 0.5 < self.minimum_consensus <= 1.0:
            raise PublicInputError("minimum_consensus must be in (0.5, 1]")
        if self.maximum_competing_sessions < 0:
            raise PublicInputError("maximum_competing_sessions must be nonnegative")
        if self.minimum_support_margin < 1:
            raise PublicInputError("minimum_support_margin must be positive")
        if self.maximum_facts < 1:
            raise PublicInputError("maximum_facts must be positive")
        if self.maximum_value_characters < 1 or self.maximum_overlay_characters < 80:
            raise PublicInputError("overlay character caps are invalid")


@dataclass(frozen=True)
class EvidenceFact:
    """A public, typed fact safe to serialize in an overlay."""

    domain: str
    slot: str
    value: str
    support_sessions: int
    competing_sessions: int
    consensus: float
    last_seen_session: int

    def as_public_dict(self) -> dict[str, object]:
        return {
            "domain": self.domain,
            "slot": self.slot,
            "value": self.value,
            "support_sessions": self.support_sessions,
            "competing_sessions": self.competing_sessions,
            "consensus": self.consensus,
        }


@dataclass(frozen=True)
class OverlayDecision:
    """A deterministic decision; no target answer or prediction is represented."""

    memory: str
    selected_domain: str | None
    facts: tuple[EvidenceFact, ...]
    explicit_slots: tuple[str, ...]
    reason: str

    @property
    def baseline_unchanged(self) -> bool:
        return not self.facts
