"""Runtime-only compatibility hooks for the isolated vLLM environment."""

from __future__ import annotations

import os
from pathlib import Path
import sysconfig
from typing import Any

_original_get_paths = sysconfig.get_paths


def _get_paths(*args: Any, **kwargs: Any) -> dict[str, str]:
    paths = _original_get_paths(*args, **kwargs)
    include = paths.get("include")
    if include and (Path(include) / "Python.h").is_file():
        return paths

    override = os.environ.get("PREFINE_PYTHON_INCLUDE")
    if override and (Path(override) / "Python.h").is_file():
        paths = dict(paths)
        paths["include"] = override
        paths["platinclude"] = override
    return paths


sysconfig.get_paths = _get_paths


def _register_gemma4_unified_alias() -> None:
    """Route the new Gemma4 Unified checkpoint name to vLLM native LM."""
    try:
        from vllm.model_executor.models.registry import ModelRegistry
    except Exception:
        return

    ModelRegistry.register_model(
        "Gemma4UnifiedForConditionalGeneration",
        "gemma4_text_only:Gemma4UnifiedTextOnlyForCausalLM",
    )


_register_gemma4_unified_alias()
