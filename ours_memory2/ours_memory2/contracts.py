"""Typed, side-effect-free shared contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from typing import Any, Mapping


class OursMemory2Error(Exception):
    """Base class for stable package errors."""


class InputContractError(OursMemory2Error):
    """An external input failed validation at a stable logical path."""

    def __init__(self, message: str, *, path: str) -> None:
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


class JoinError(InputContractError):
    """Two otherwise valid resources cannot be joined unambiguously."""


class ProviderError(OursMemory2Error):
    """A provider failed without exposing credentials or response secrets."""


class Difficulty(str, Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ContextMode(str, Enum):
    MEMORY_ONLY = "memory_only"
    MEMORY_API = "memory_api"
    MEMORY_DIAG = "memory_diag"
    API_ONLY = "api_only"


class ProviderPurpose(str, Enum):
    GENERATE = "generate"
    REFINE = "refine"
    VERIFY = "verify"
    INFER = "infer"


class OutputFilename(str, Enum):
    """The complete set of direct-child output artifacts."""

    MEMORIES = "memories.jsonl"
    DRAFTS = "drafts.jsonl"
    VERIFIERS = "verifiers.jsonl"
    RESULTS = "results.jsonl"
    DIAGNOSTICS = "diagnostics.jsonl"

    def __str__(self) -> str:
        return self.value


def normalize_scalar_id(value: object, *, path: str) -> str:
    """Normalize a nonempty JSON scalar identifier to a deterministic string."""

    if value is None or isinstance(value, (dict, list)):
        raise InputContractError("must be a nonempty scalar identifier", path=path)
    if isinstance(value, str):
        normalized = value.strip()
    elif isinstance(value, bool):
        normalized = "true" if value else "false"
    elif isinstance(value, int):
        normalized = str(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise InputContractError("must be a finite scalar identifier", path=path)
        normalized = str(int(value)) if value.is_integer() else format(value, ".15g")
    else:
        raise InputContractError("must be a JSON scalar identifier", path=path)
    if not normalized:
        raise InputContractError("must be a nonempty scalar identifier", path=path)
    return normalized


@dataclass(frozen=True)
class DialogueTurn:
    role: str
    message: str


@dataclass(frozen=True)
class SessionInput:
    dialogue: tuple[DialogueTurn, ...]
    api_call: tuple[str, ...]


@dataclass(frozen=True)
class ExampleInput:
    example_id: str
    sessions: tuple[SessionInput, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise InputContractError("must be nonempty", path="provider.messages.role")
        if not self.content:
            raise InputContractError("must be nonempty", path="provider.messages.content")


@dataclass(frozen=True)
class ProviderRequest:
    """One typed provider call.

    ``temperature=None`` means the wire must omit the setting; ``json_intent``
    controls whether the adapter requests a JSON response format. This lets
    Step 1 retain its explicit temperatures/JSON intent while Step 2 mirrors
    the source inference wire, which supplies neither option.
    """

    purpose: ProviderPurpose
    model: str
    messages: tuple[ChatMessage, ...]
    prompt: str | None = None
    temperature: float | None = None
    json_intent: bool = False

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise InputContractError("must be nonempty", path="provider.model")
        if not self.messages:
            raise InputContractError("must be nonempty", path="provider.messages")
        if self.prompt is not None and not self.prompt:
            raise InputContractError("must be nonempty when supplied", path="provider.prompt")
        if self.temperature is not None and (
            not math.isfinite(self.temperature) or self.temperature < 0
        ):
            raise InputContractError(
                "must be a finite nonnegative number", path="provider.temperature"
            )


@dataclass(frozen=True)
class ProviderResponse:
    text: str
    request_id: str | None = None
    usage: Mapping[str, Any] | None = None
    raw_response: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.text:
            raise ProviderError("provider response content is empty")
        try:
            json.dumps(self.usage, ensure_ascii=False)
            json.dumps(self.raw_response, ensure_ascii=False)
        except (TypeError, ValueError):
            raise ProviderError("provider response diagnostics must be JSON-safe") from None


@dataclass(frozen=True)
class Step1OutputNames:
    final: OutputFilename = OutputFilename.MEMORIES
    drafts: OutputFilename = OutputFilename.DRAFTS
    verifiers: OutputFilename = OutputFilename.VERIFIERS


@dataclass(frozen=True)
class Step2OutputNames:
    results: OutputFilename = OutputFilename.RESULTS
    diagnostics: OutputFilename = OutputFilename.DIAGNOSTICS
