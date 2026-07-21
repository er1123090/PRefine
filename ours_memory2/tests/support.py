"""Typed deterministic provider support for tests and probes only."""

from __future__ import annotations

from collections import deque
from typing import Callable, Iterable

from ours_memory2.contracts import ProviderError, ProviderRequest, ProviderResponse


class ScriptedProvider:
    """Finite deterministic provider implementing the runtime protocol."""

    def __init__(
        self,
        responses: Iterable[ProviderResponse]
        | Callable[[ProviderRequest], ProviderResponse],
        *,
        model: str = "fixture-model",
    ) -> None:
        if not model.strip():
            raise ValueError("model must be nonempty")
        self.model = model
        self.requests: list[ProviderRequest] = []
        self._callable = responses if callable(responses) else None
        self._responses = deque() if self._callable else deque(responses)

    def complete(self, provider_request: ProviderRequest) -> ProviderResponse:
        self.requests.append(provider_request)
        if self._callable is not None:
            response = self._callable(provider_request)
        else:
            if not self._responses:
                raise ProviderError("scripted provider has no response remaining")
            response = self._responses.popleft()
        if not isinstance(response, ProviderResponse):
            raise ProviderError("scripted provider returned an invalid response contract")
        return response
