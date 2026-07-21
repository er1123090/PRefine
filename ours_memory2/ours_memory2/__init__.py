"""Standalone two-stage preference-memory construction and inference method."""

from .contracts import (
    ChatMessage,
    ContextMode,
    Difficulty,
    ExampleInput,
    InputContractError,
    JoinError,
    ProviderError,
    ProviderPurpose,
    ProviderRequest,
    ProviderResponse,
    SessionInput,
    normalize_scalar_id,
)

__all__ = [
    "ChatMessage",
    "ContextMode",
    "Difficulty",
    "ExampleInput",
    "InputContractError",
    "JoinError",
    "ProviderError",
    "ProviderPurpose",
    "ProviderRequest",
    "ProviderResponse",
    "SessionInput",
    "normalize_scalar_id",
]

__version__ = "0.1.0"
