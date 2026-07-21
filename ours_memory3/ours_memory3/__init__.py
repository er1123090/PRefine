"""Standalone query-conditioned evidence-calibrated preference overlay."""

from .contracts import EvidenceFact, OverlayDecision, OverlayPolicy, PublicInputError
from .overlay import build_overlay

__all__ = [
    "EvidenceFact",
    "OverlayDecision",
    "OverlayPolicy",
    "PublicInputError",
    "build_overlay",
]

__version__ = "0.1.0"
