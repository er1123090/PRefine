"""Lazy OpenAI-compatible provider for the prepared-record runner."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Mapping, Sequence

from .runner import InferenceRequest, InferenceResult


PROMPT_TEMPLATE = """You are a Personalized Preference Extraction Specialist.
Infer the user's preferences from the dialogue history and current utterance.
Use only slots from the target service schema and return exactly one service API call.

Target service schema:
{preference_schema}

Dialogue history:
{dialogue_history}

Current user utterance:
{user_utterance}

Output format: Domain(slot_name=\"value\", ...)
"""


def _dialogue_history(source_example: Mapping[str, Any]) -> str:
    lines: list[str] = []
    sessions = source_example.get("sessions", [])
    if not isinstance(sessions, list):
        return ""
    for session in sessions:
        if not isinstance(session, Mapping):
            continue
        turns = session.get("dialogue", [])
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if isinstance(turn, Mapping):
                lines.append(
                    f"{turn.get('role', 'User')}: "
                    f"{turn.get('message', turn.get('content', ''))}"
                )
    return "\n".join(lines)


def build_prompt(
    record: Mapping[str, Any], tools_schema: Sequence[Mapping[str, Any]]
) -> str:
    source = record.get("source_example")
    return PROMPT_TEMPLATE.format(
        preference_schema=json.dumps(tools_schema, ensure_ascii=False, separators=(",", ":")),
        dialogue_history=_dialogue_history(source if isinstance(source, Mapping) else {}),
        user_utterance=record.get("query", ""),
    )


def _clean_content(content: str) -> str:
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    match = re.search(r"([A-Za-z0-9_]+)\((.*?)\)", content, flags=re.DOTALL)
    return match.group(0).strip() if match else content


class OpenAICompatibleProvider:
    """Call an OpenAI-compatible chat-completions endpoint on first use."""

    def __init__(
        self,
        *,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float | None = None,
    ) -> None:
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "OpenAI-compatible inference requires the optional 'openai' package"
            ) from exc
        api_key = os.environ.get(self.api_key_env)
        if not api_key and not self.base_url:
            raise RuntimeError(f"missing API key environment variable: {self.api_key_env}")
        kwargs: dict[str, Any] = {"api_key": api_key or "EMPTY"}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        self._client = AsyncOpenAI(**kwargs)
        return self._client

    async def __call__(
        self, record: Mapping[str, Any], request: InferenceRequest
    ) -> InferenceResult:
        kwargs: dict[str, Any] = {
            "model": request.model_name,
            "messages": [
                {"role": "user", "content": build_prompt(record, request.tools_schema)}
            ],
        }
        if request.tools_schema:
            kwargs.update({"tools": list(request.tools_schema), "tool_choice": "auto"})
        if request.reasoning_effort:
            kwargs["reasoning_effort"] = request.reasoning_effort
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens

        response = await self._get_client().chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = message.content or ""
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            call = tool_calls[0]
            try:
                arguments = json.loads(call.function.arguments)
                rendered = ", ".join(
                    f'{key}="{arguments[key]}"' for key in sorted(arguments)
                )
                content = f"{call.function.name}({rendered})"
            except (TypeError, json.JSONDecodeError):
                content = f"ERROR_JSON_PARSE: {call.function.arguments}"
        else:
            content = _clean_content(content)

        reasoning = getattr(message, "reasoning_content", "") or ""
        usage = getattr(response, "usage", None)
        token_counts: dict[str, int] = {}
        if usage is not None:
            for output_key, source_key in (
                ("total_tokens", "total_tokens"),
                ("input_tokens", "prompt_tokens"),
                ("output_tokens", "completion_tokens"),
            ):
                value = getattr(usage, source_key, None)
                if isinstance(value, int):
                    token_counts[output_key] = value
            details = getattr(usage, "completion_tokens_details", None)
            reasoning_tokens = getattr(details, "reasoning_tokens", None)
            if isinstance(reasoning_tokens, int):
                token_counts["reasoning_tokens"] = reasoning_tokens
        return InferenceResult(
            content=content,
            reasoning_content=str(reasoning),
            token_counts=token_counts,
            response=content,
        )
