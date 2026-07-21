"""Experiment methods that consume canonical prepared instances."""

from .contracts import (
    CanonicalPreparedRecord,
    MethodAdapter,
    MethodContractError,
    MethodRunContext,
    MethodStateHandle,
    await_if_needed,
    bind_method_state,
    canonical_prediction,
    method_state_path,
    validate_method_id,
    validate_prepared_record,
    validate_prepared_records,
)
from .langmem import LangMemAdapter
from .mem0 import Mem0Adapter
from .preference_memory import PreferenceMemoryAdapter
from .rag import RAGAdapter
from .registry import (
    MethodAdapterFactory,
    MethodRegistry,
    MethodRegistryError,
    VanillaLLMAdapter,
    builtin_method_registry,
)

__all__ = [
    "CanonicalPreparedRecord",
    "MethodAdapter",
    "MethodAdapterFactory",
    "MethodContractError",
    "MethodRegistry",
    "MethodRegistryError",
    "MethodRunContext",
    "MethodStateHandle",
    "LangMemAdapter",
    "Mem0Adapter",
    "PreferenceMemoryAdapter",
    "RAGAdapter",
    "VanillaLLMAdapter",
    "await_if_needed",
    "bind_method_state",
    "builtin_method_registry",
    "canonical_prediction",
    "method_state_path",
    "validate_method_id",
    "validate_prepared_record",
    "validate_prepared_records",
]
