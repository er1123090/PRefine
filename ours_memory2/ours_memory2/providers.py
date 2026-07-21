"""Synchronous provider protocol and standard-library adapters."""

from __future__ import annotations

from http.client import InvalidURL
import ipaddress
import json
import math
import os
import re
import socket
from typing import Iterable, Protocol, runtime_checkable
from urllib import error, parse, request

from .contracts import ProviderError, ProviderRequest, ProviderResponse


@runtime_checkable
class Provider(Protocol):
    model: str

    def complete(self, provider_request: ProviderRequest) -> ProviderResponse:
        """Complete one request synchronously."""


class _NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


class OpenAICompatibleProvider:
    """Lazy loopback-only chat-completions adapter using ``urllib.request``."""

    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        key_env: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        if not math.isfinite(timeout) or timeout <= 0:
            raise ProviderError("provider timeout must be finite and positive")
        if not isinstance(model, str) or not model.strip():
            raise ProviderError("provider model must be nonempty")
        self.endpoint = _validate_endpoint(endpoint)
        self.model = model.strip()
        self.key_env = key_env
        self.timeout = float(timeout)

    def complete(self, provider_request: ProviderRequest) -> ProviderResponse:
        secret = os.environ.get(self.key_env) if self.key_env else None
        body = {
            "model": provider_request.model or self.model,
            "messages": [
                {"role": item.role, "content": item.content}
                for item in provider_request.messages
            ],
        }
        if provider_request.temperature is not None:
            body["temperature"] = provider_request.temperature
        if provider_request.json_intent:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if secret:
            headers["Authorization"] = f"Bearer {secret}"
        try:
            http_request = request.Request(
                self.endpoint,
                data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            opener = request.build_opener(request.ProxyHandler({}), _NoRedirect())
            with opener.open(http_request, timeout=self.timeout) as response:
                raw_bytes = response.read()
                response_headers = response.headers
        except error.HTTPError as exc:
            raise ProviderError(f"provider HTTP error {exc.code}") from None
        except (error.URLError, InvalidURL, TimeoutError, socket.timeout, OSError, ValueError):
            raise ProviderError("provider transport or timeout error") from None
        try:
            envelope = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ProviderError("provider returned a malformed JSON envelope") from None
        if not isinstance(envelope, dict):
            raise ProviderError("provider returned an invalid response envelope")
        try:
            content = envelope["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ProviderError("provider response content is missing") from None
        if not isinstance(content, str) or not content:
            raise ProviderError("provider response content is missing")
        request_id = envelope.get("id") or response_headers.get("x-request-id")
        safe_request_id = redact_diagnostic(request_id, secrets=(secret,))
        safe_usage = redact_diagnostic(envelope.get("usage"), secrets=(secret,))
        return ProviderResponse(
            text=content,
            request_id=str(safe_request_id) if safe_request_id is not None else None,
            usage=safe_usage if isinstance(safe_usage, dict) else None,
            raw_response=redact_diagnostic(envelope, secrets=(secret,)),
        )


def _validate_endpoint(endpoint: object) -> str:
    try:
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in endpoint):
            raise ValueError
        parsed = parse.urlsplit(endpoint)
        if (
            parsed.scheme != "http"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or not parsed.path
            or parsed.fragment
        ):
            raise ValueError
        _ = parsed.port
        address = ipaddress.ip_address(parsed.hostname)
        if not address.is_loopback:
            raise ValueError
    except (ValueError, TypeError):
        raise ProviderError(
            "provider endpoint must be a valid explicit loopback HTTP address"
        ) from None
    return endpoint


def redact_diagnostic(
    value: object, *, secrets: Iterable[str | None] = ()
) -> object:
    """Return a JSON-safe recursive copy with credential material removed.

    Key matching is token-aware: common compound credential names are
    redacted, while unrelated words such as ``monkey`` are preserved.  Exact
    configured secret values are also replaced wherever they occur in string
    keys or values.
    """

    safe_value = _json_safe(value)
    secret_values = tuple(secret for secret in secrets if secret)
    return _redact_json_value(safe_value, secrets=secret_values)


def _json_safe(value: object) -> object:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False))
    except (TypeError, ValueError):
        return None


def _redact_json_value(value: object, *, secrets: tuple[str, ...]) -> object:
    if isinstance(value, dict):
        redacted: dict[str, object] = {}
        for key, item in value.items():
            safe_key = _replace_secrets(str(key), secrets)
            redacted[safe_key] = (
                "[REDACTED]"
                if _is_sensitive_key(str(key))
                else _redact_json_value(item, secrets=secrets)
            )
        return redacted
    if isinstance(value, list):
        return [_redact_json_value(item, secrets=secrets) for item in value]
    if isinstance(value, str):
        return _redact_embedded_credentials(_replace_secrets(value, secrets))
    return value


def _replace_secrets(value: str, secrets: tuple[str, ...]) -> str:
    result = value
    for secret in secrets:
        result = result.replace(secret, "[REDACTED]")
    return result


def _is_sensitive_key(value: str) -> bool:
    expanded = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    tokens = tuple(re.findall(r"[a-z0-9]+", expanded.lower()))
    if not tokens:
        return False
    collapsed = "".join(tokens)
    if collapsed in {
        "apikey",
        "authorization",
        "credential",
        "credentials",
        "password",
        "passwd",
        "passphrase",
        "secret",
        "token",
        "key",
    }:
        return True
    if any(
        token
        in {
            "authorization",
            "cookie",
            "credential",
            "credentials",
            "header",
            "headers",
            "password",
            "passwd",
            "passphrase",
            "secret",
            "setcookie",
            "token",
        }
        for token in tokens
    ):
        return True
    return "key" in tokens and any(
        token in {"access", "api", "auth", "client", "private", "signing"}
        for token in tokens
    )


_QUOTED_ASSIGNMENT = re.compile(
    r'(?P<prefix>["\'](?P<key>[^"\'\n]+)["\']\s*:\s*)'
    r'(?P<value>["\'][^"\'\n]*["\']|[^\s,}\]\n]+)'
)
_PLAIN_ASSIGNMENT = re.compile(
    r"(?P<prefix>\b(?P<key>[A-Za-z0-9_.-]+)\s*(?:=|:)\s*)"
    r"(?P<value>Bearer\s+[^\s,;]+|[^\s,;]+)",
    re.IGNORECASE,
)


def _redact_embedded_credentials(value: str) -> str:
    """Redact credential assignments embedded in serialized prompt text."""

    def replace(match: re.Match[str]) -> str:
        if not _is_sensitive_key(match.group("key")):
            return match.group(0)
        return match.group("prefix") + '"[REDACTED]"'

    return _PLAIN_ASSIGNMENT.sub(replace, _QUOTED_ASSIGNMENT.sub(replace, value))
