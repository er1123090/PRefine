"""Provider endpoint resolution shared by experiment8 CLIs."""

from __future__ import annotations

import os
from typing import Optional, Tuple
from urllib.parse import urlparse


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_DEFAULT_EMBEDDING_MODEL = "openai/text-embedding-3-small"
_OPENAI_EMBEDDING_MODELS = {
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
}


def is_openrouter_endpoint(base_url: Optional[str]) -> bool:
    if not base_url:
        return False
    hostname = (urlparse(base_url).hostname or "").lower()
    return hostname == "openrouter.ai" or hostname.endswith(".openrouter.ai")


def resolve_openai_compatible_endpoint(
    provider: str,
    base_url: Optional[str],
    api_key: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
    use_openrouter = provider == "openrouter" or is_openrouter_endpoint(base_url)
    if not use_openrouter:
        return base_url, api_key

    resolved_api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is required for OpenRouter. "
            "Set the environment variable or pass --api_key."
        )
    return base_url or OPENROUTER_BASE_URL, resolved_api_key


def resolve_embedding_endpoint(
    provider: str,
    embedding_model: str,
    base_url: Optional[str],
    api_key: Optional[str],
) -> Tuple[str, Optional[str], Optional[str]]:
    resolved_base_url, resolved_api_key = resolve_openai_compatible_endpoint(
        provider=provider,
        base_url=base_url,
        api_key=api_key,
    )
    resolved_model = embedding_model
    if (
        provider == "openrouter" or is_openrouter_endpoint(resolved_base_url)
    ) and embedding_model in _OPENAI_EMBEDDING_MODELS:
        resolved_model = f"openai/{embedding_model}"
    return resolved_model, resolved_base_url, resolved_api_key


def provider_label(provider: str, base_url: Optional[str]) -> str:
    if provider == "openrouter" or is_openrouter_endpoint(base_url):
        return "openrouter"
    return "openai"
