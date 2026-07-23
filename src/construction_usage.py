"""Task-local token accounting for concurrent memory construction."""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Dict, Iterable, List, Optional


TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)

_CALLS: ContextVar[Optional[List[Dict[str, Any]]]] = ContextVar(
    "construction_usage_calls", default=None
)
_SESSION_INDEX: ContextVar[Optional[int]] = ContextVar(
    "construction_usage_session_index", default=None
)


def begin_usage_collection():
    """Start an isolated collector for the current asyncio task."""
    return _CALLS.set([])


def set_usage_session(session_index: int) -> None:
    _SESSION_INDEX.set(session_index)


def _numeric_attr(value: Any, names: Iterable[str]) -> Optional[int]:
    for name in names:
        if isinstance(value, dict):
            candidate = value.get(name)
        else:
            candidate = getattr(value, name, None)
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            return int(candidate)
    return None


def _nested_attr(value: Any, names: Iterable[str]) -> Any:
    for name in names:
        if isinstance(value, dict):
            candidate = value.get(name)
        else:
            candidate = getattr(value, name, None)
        if candidate is not None:
            return candidate
    return None


def usage_from_response(response: Any) -> Dict[str, Any]:
    """Normalize OpenAI/vLLM and Gemini usage objects."""
    usage = _nested_attr(response, ("usage", "usage_metadata"))
    if usage is None and isinstance(response, dict):
        direct_usage_keys = {
            "prompt_tokens",
            "input_tokens",
            "prompt_token_count",
            "completion_tokens",
            "output_tokens",
            "candidates_token_count",
            "total_tokens",
            "total_token_count",
        }
        if direct_usage_keys.intersection(response):
            usage = response
    if usage is None:
        llm_output = _nested_attr(response, ("llm_output",))
        if isinstance(llm_output, dict):
            usage = llm_output.get("token_usage") or llm_output.get("usage")
    if usage is None:
        generations = _nested_attr(response, ("generations",))
        try:
            message = generations[0][0].message
        except (AttributeError, IndexError, TypeError):
            message = None
        if message is not None:
            usage = _nested_attr(message, ("usage_metadata",))
            if usage is None:
                response_metadata = _nested_attr(message, ("response_metadata",))
                if isinstance(response_metadata, dict):
                    usage = response_metadata.get("token_usage") or response_metadata.get(
                        "usage"
                    )
    if usage is None:
        return {
            "usage_available": False,
            **{field: 0 for field in TOKEN_FIELDS},
        }

    input_tokens = _numeric_attr(
        usage, ("prompt_tokens", "input_tokens", "prompt_token_count")
    )
    output_tokens = _numeric_attr(
        usage, ("completion_tokens", "output_tokens", "candidates_token_count")
    )
    total_tokens = _numeric_attr(usage, ("total_tokens", "total_token_count"))

    input_details = _nested_attr(
        usage,
        (
            "prompt_tokens_details",
            "input_tokens_details",
            "input_token_details",
        ),
    )
    output_details = _nested_attr(
        usage,
        (
            "completion_tokens_details",
            "output_tokens_details",
            "output_token_details",
        ),
    )
    cached_input_tokens = _numeric_attr(
        input_details,
        ("cached_tokens", "cached_input_tokens", "cache_read"),
    )
    if cached_input_tokens is None:
        cached_input_tokens = _numeric_attr(
            usage, ("cached_content_token_count", "cached_input_tokens")
        )
    reasoning_tokens = _numeric_attr(
        output_details,
        ("reasoning_tokens", "reasoning_token_count", "reasoning"),
    )
    if reasoning_tokens is None:
        reasoning_tokens = _numeric_attr(
            usage, ("thoughts_token_count", "reasoning_tokens")
        )

    input_tokens = input_tokens or 0
    output_tokens = output_tokens or 0
    cached_input_tokens = cached_input_tokens or 0
    reasoning_tokens = reasoning_tokens or 0
    if total_tokens is None:
        total_tokens = input_tokens + output_tokens

    return {
        "usage_available": True,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
    }


def record_response_usage(
    response: Any,
    *,
    component: str,
    provider: str,
    model: str,
) -> Dict[str, Any]:
    record = {
        "component": component,
        "session_index": _SESSION_INDEX.get(),
        "provider": provider,
        "model": model,
        **usage_from_response(response),
    }
    calls = _CALLS.get()
    if calls is not None:
        calls.append(record)
    return record


def _summarize_calls(calls: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(calls)
    result: Dict[str, Any] = {
        field: sum(int(row.get(field) or 0) for row in rows)
        for field in TOKEN_FIELDS
    }
    result["call_count"] = len(rows)
    result["usage_available_calls"] = sum(
        int(bool(row.get("usage_available"))) for row in rows
    )
    result["usage_missing_calls"] = (
        result["call_count"] - result["usage_available_calls"]
    )
    return result


def _build_usage_report(calls: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(calls)
    components = sorted(
        {str(call.get("component")) for call in rows if call.get("component")}
    )
    sessions = sorted(
        {
            int(call["session_index"])
            for call in rows
            if isinstance(call.get("session_index"), int)
        }
    )
    return {
        "summary": _summarize_calls(rows),
        "by_component": {
            component: _summarize_calls(
                call for call in rows if call.get("component") == component
            )
            for component in components
        },
        "by_session": {
            str(session): {
                "summary": _summarize_calls(
                    call for call in rows if call.get("session_index") == session
                ),
                "by_component": {
                    component: _summarize_calls(
                        call
                        for call in rows
                        if call.get("session_index") == session
                        and call.get("component") == component
                    )
                    for component in components
                    if any(
                        call.get("session_index") == session
                        and call.get("component") == component
                        for call in rows
                    )
                },
            }
            for session in sessions
        },
        "calls": rows,
    }


def current_usage_report() -> Dict[str, Any]:
    return _build_usage_report(_CALLS.get() or [])


def end_usage_collection(token) -> Dict[str, Any]:
    report = current_usage_report()
    _CALLS.reset(token)
    return report
