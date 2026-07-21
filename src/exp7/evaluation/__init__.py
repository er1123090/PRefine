"""Manifest-driven prediction evaluation against canonical prepared truth."""

from .core import (
    EVALUATOR_VERSION,
    SUPPORTED_VARIANTS,
    EvaluationError,
    EvaluationSummary,
    evaluate_run,
)

__all__ = [
    "EVALUATOR_VERSION",
    "SUPPORTED_VARIANTS",
    "EvaluationError",
    "EvaluationSummary",
    "evaluate_run",
]
