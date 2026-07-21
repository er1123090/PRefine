"""VanillaLLM over canonical prepared Experiment 7 records.

Real inference is intentionally limited to OpenAI-compatible chat-completions
endpoints. Tests and alternate runtimes can inject a sync or async callable.
"""

from .provider import OpenAICompatibleProvider
from .runner import (
    InferenceRequest,
    InferenceResult,
    ManifestVerificationError,
    RunSummary,
    run_prepared_file,
    run_records,
    verify_prepared_input,
)

__all__ = [
    "InferenceRequest",
    "InferenceResult",
    "ManifestVerificationError",
    "OpenAICompatibleProvider",
    "RunSummary",
    "run_prepared_file",
    "run_records",
    "verify_prepared_input",
]
