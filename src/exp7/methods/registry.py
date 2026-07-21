"""Fail-closed registry for implemented canonical method adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

from .contracts import (
    CanonicalPreparedRecord,
    MethodAdapter,
    MethodContractError,
    MethodRunContext,
    MethodStateHandle,
    canonical_prediction,
    validate_method_id,
    validate_prepared_records,
)
from .langmem import LangMemAdapter
from .mem0 import Mem0Adapter
from .preference_memory import PreferenceMemoryAdapter
from .rag import RAGAdapter
from .vanilla_llm.runner import run_records


class MethodRegistryError(LookupError):
    """Raised for unknown, duplicate, or malformed method registrations."""


MethodAdapterFactory = Callable[..., MethodAdapter[Any]]


@dataclass(frozen=True)
class _FactoryRegistration:
    factory: MethodAdapterFactory


class MethodRegistry:
    def __init__(
        self,
        adapters: Iterable[MethodAdapter[Any]] = (),
    ) -> None:
        self._adapters: dict[str, MethodAdapter[Any]] = {}
        self._factories: dict[str, _FactoryRegistration] = {}
        for adapter in adapters:
            self.register(adapter)

    @staticmethod
    def _validate_adapter(
        adapter: MethodAdapter[Any],
        *,
        expected_method_id: str | None = None,
    ) -> str:
        try:
            method_id = validate_method_id(adapter.method_id)
        except (AttributeError, MethodContractError) as exc:
            raise MethodRegistryError(f"invalid method adapter: {exc}") from exc
        if not callable(getattr(adapter, "build", None)) or not callable(
            getattr(adapter, "infer", None)
        ):
            raise MethodRegistryError(
                f"method adapter {method_id!r} must implement build() and infer()"
            )
        if expected_method_id is not None and method_id != expected_method_id:
            raise MethodRegistryError(
                "method factory identity mismatch: "
                f"registered {expected_method_id!r}, constructed {method_id!r}"
            )
        return method_id

    def _reject_duplicate(self, method_id: str) -> None:
        if method_id in self._adapters or method_id in self._factories:
            raise MethodRegistryError(f"duplicate method adapter: {method_id}")

    def register(self, adapter: MethodAdapter[Any]) -> None:
        method_id = self._validate_adapter(adapter)
        self._reject_duplicate(method_id)
        self._adapters[method_id] = adapter

    def register_factory(
        self,
        method_id: str,
        factory: MethodAdapterFactory,
    ) -> None:
        try:
            normalized = validate_method_id(method_id)
        except MethodContractError as exc:
            raise MethodRegistryError(str(exc)) from exc
        if not callable(factory):
            raise MethodRegistryError(
                f"method factory for {normalized!r} must be callable"
            )
        self._reject_duplicate(normalized)
        self._factories[normalized] = _FactoryRegistration(factory=factory)

    def create(
        self,
        method_id: str,
        **factory_options: Any,
    ) -> MethodAdapter[Any]:
        try:
            normalized = validate_method_id(method_id)
        except MethodContractError as exc:
            raise MethodRegistryError(str(exc)) from exc
        adapter = self._adapters.get(normalized)
        if adapter is not None:
            if factory_options:
                raise MethodRegistryError(
                    f"registered adapter {normalized!r} does not accept factory options"
                )
            return adapter
        registration = self._factories.get(normalized)
        if registration is None:
            raise MethodRegistryError(
                f"unknown method adapter: {normalized}"
            )
        try:
            created = registration.factory(**factory_options)
        except Exception as exc:
            raise MethodRegistryError(
                f"cannot construct method adapter {normalized!r}: {exc}"
            ) from exc
        self._validate_adapter(created, expected_method_id=normalized)
        return created

    def get(self, method_id: str) -> MethodAdapter[Any]:
        """Construct an adapter with its default options.

        Methods with required configuration, currently Mem0's namespace, fail
        closed here and must be requested through :meth:`create`.
        """

        return self.create(method_id)

    @property
    def method_ids(self) -> tuple[str, ...]:
        return tuple(sorted((*self._adapters, *self._factories)))


class VanillaLLMAdapter:
    """Honest stateless adapter over the implemented VanillaLLM runner."""

    method_id = "vanilla_llm"

    def build(
        self,
        records: Sequence[CanonicalPreparedRecord],
        context: MethodRunContext,
        backend: Any,
    ) -> None:
        del context, backend
        validate_prepared_records(records)
        return None

    async def infer(
        self,
        record: CanonicalPreparedRecord,
        context: MethodRunContext,
        state: MethodStateHandle | None,
        backend: Any,
    ) -> dict[str, Any]:
        if state is not None:
            raise MethodContractError(
                "vanilla_llm does not accept method state"
            )
        if not context.model_name:
            raise MethodContractError(
                "vanilla_llm requires a non-empty model_name"
            )
        outputs = await run_records(
            [record.as_mapping()],
            inference=backend,
            model_name=context.model_name,
            tools_schema=context.tools_schema,
            reasoning_effort=context.reasoning_effort,
        )
        return canonical_prediction(
            record,
            method_id=self.method_id,
            prediction=outputs[0],
        )


def builtin_method_registry() -> MethodRegistry:
    """Return fresh lazy registrations for every canonical built-in method."""

    registry = MethodRegistry()
    for adapter_type in (
        VanillaLLMAdapter,
        RAGAdapter,
        Mem0Adapter,
        LangMemAdapter,
        PreferenceMemoryAdapter,
    ):
        registry.register_factory(adapter_type.method_id, adapter_type)
    return registry
