from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading
import time
import unittest
from unittest import mock

from ours_memory2.contracts import (
    ChatMessage,
    ProviderError,
    ProviderPurpose,
    ProviderRequest,
    ProviderResponse,
)
from ours_memory2.providers import OpenAICompatibleProvider
from ours_memory2.prompts import build_generation_messages, build_verifier_messages
from ours_memory2.step1 import _provider_request
from ours_memory2.step2_common import load_manifest_v1, make_provider_request
from ours_memory2.step2_single import build_single_cases
from tests.support import ScriptedProvider


class ScriptedProviderTests(unittest.TestCase):
    def test_finite_deterministic_script_and_recording(self) -> None:
        expected = ProviderResponse(text="one")
        provider = ScriptedProvider([expected])
        request = _request()
        self.assertEqual(provider.complete(request), expected)
        self.assertEqual(provider.requests, [request])
        with self.assertRaises(ProviderError):
            provider.complete(request)

    def test_response_diagnostics_and_timeout_are_strict(self) -> None:
        with self.assertRaises(ProviderError):
            ProviderResponse(text="ok", raw_response={"bad": object()})
        for timeout in (0, -1, float("inf"), float("nan")):
            with self.subTest(timeout=timeout), self.assertRaises(ProviderError):
                OpenAICompatibleProvider(
                    endpoint="http://127.0.0.1/x", model="m", timeout=timeout
                )


class HttpProviderTests(unittest.TestCase):
    def test_success_captures_safe_diagnostics_and_redacts(self) -> None:
        secret = "TOP-SECRET-VALUE"
        keyed_secrets = {
            "access_token": "ACCESS-VALUE",
            "refreshToken": "REFRESH-VALUE",
            "client-secret": "CLIENT-SECRET-VALUE",
            "PASSWORD": "PASSWORD-VALUE",
            "awsCredentials": "CREDENTIAL-VALUE",
            "authorization": "AUTHORIZATION-VALUE",
            "api_key": "API-KEY-VALUE",
            "apikey": "APIKEY-VALUE",
            "aws_access_key_id": "AWS-ACCESS-KEY-VALUE",
        }
        envelope = {
            "id": f"request-{secret}",
            "choices": [{"message": {"content": '{"ok":true}'}}],
            "usage": {
                "prompt_tokens": 2,
                "note": secret,
                "monkey": "banana",
                "nested": keyed_secrets,
            },
            "api_key": secret,
            "nested": {
                "value": f"prefix-{secret}",
                "monkey": "kept-monkey-value",
                **keyed_secrets,
            },
        }
        with self._server(status=200, payload=envelope) as endpoint, mock.patch.dict(
            os.environ, {"TEST_PROVIDER_KEY": secret}, clear=False
        ):
            provider = OpenAICompatibleProvider(
                endpoint=endpoint, model="fixture-model", key_env="TEST_PROVIDER_KEY"
            )
            response = provider.complete(_request())
        self.assertEqual(response.text, '{"ok":true}')
        self.assertEqual(response.request_id, "request-[REDACTED]")
        self.assertEqual(response.usage["prompt_tokens"], 2)
        self.assertEqual(response.usage["note"], "[REDACTED]")
        self.assertEqual(response.usage["monkey"], "banana")
        rendered = json.dumps(
            {"request_id": response.request_id, "usage": response.usage, "raw": response.raw_response}
        )
        self.assertNotIn(secret, rendered)
        for keyed_secret in keyed_secrets.values():
            self.assertNotIn(keyed_secret, rendered)
        self.assertIn("[REDACTED]", rendered)
        self.assertIn("kept-monkey-value", rendered)

    def test_actual_step1_and_inference_http_bodies_preserve_distinct_wire_contracts(self) -> None:
        manifest = Path(__file__).parent / "fixtures" / "step2" / "manifest_grouped.json"
        resources = load_manifest_v1(manifest)
        case = build_single_cases(resources, difficulty="easy")[0]
        inference = make_provider_request(
            resources=resources,
            case=case,
            context="memory_api",
            model="fixture-model",
        )
        owner = SimpleNamespace(model="fixture-model")
        generation = _provider_request(
            provider=owner,
            purpose=ProviderPurpose.GENERATE,
            messages=build_generation_messages(
                previous_preference={},
                dialogue="User: hello",
                api_history=(),
            ),
            temperature=0.4,
        )
        verification = _provider_request(
            provider=owner,
            purpose=ProviderPurpose.VERIFY,
            messages=build_verifier_messages(
                candidate={"implicit_pref": "quiet"},
                dialogue="User: hello",
                api_history=(),
            ),
            temperature=0.0,
        )
        server = self._server(
            status=200,
            payload={"choices": [{"message": {"content": "ok"}}]},
        )
        with server as endpoint:
            provider = OpenAICompatibleProvider(
                endpoint=endpoint, model="fixture-model"
            )
            for provider_request in (inference, generation, verification):
                provider.complete(provider_request)

        infer_body, generate_body, verify_body = server.bodies
        self.assertEqual(set(infer_body), {"model", "messages"})
        self.assertEqual(
            [message["role"] for message in infer_body["messages"]], ["user"]
        )
        self.assertIn(
            "Schema Filtering (Slot Scope Control)",
            infer_body["messages"][0]["content"],
        )
        for body, temperature in ((generate_body, 0.4), (verify_body, 0.0)):
            self.assertEqual(
                [message["role"] for message in body["messages"]],
                ["system", "user"],
            )
            self.assertEqual(body["temperature"], temperature)
            self.assertEqual(body["response_format"], {"type": "json_object"})

    def test_http_redirect_malformed_missing_and_timeout_map_to_provider_error(self) -> None:
        cases = (
            (302, b"", {"Location": "http://127.0.0.1:1/elsewhere"}, 30.0),
            (200, b"not-json", {}, 30.0),
            (200, b'{"choices":[]}', {}, 30.0),
            (200, b'{"choices":[{"message":{"content":"ok"}}]}', {}, 0.02),
        )
        for index, (status, payload, headers, timeout) in enumerate(cases):
            with self.subTest(index=index), self._server(
                status=status,
                payload=payload,
                headers=headers,
                delay=0.1 if timeout < 1 else 0,
            ) as endpoint:
                provider = OpenAICompatibleProvider(
                    endpoint=endpoint, model="fixture-model", timeout=timeout
                )
                with self.assertRaises(ProviderError):
                    provider.complete(_request())

    def test_non_loopback_and_dns_are_rejected_before_connection(self) -> None:
        endpoints = (
            "http://example.com/v1/chat/completions",
            "http://192.0.2.1/x",
            "http://127.0.0.1:not-a-port/x",
            "http://127.0.0.1:65536/x",
            "http://127.0.0.1:80/x\r\nInjected: yes",
            "http://user@127.0.0.1/x",
        )
        for endpoint in endpoints:
            with self.subTest(endpoint=endpoint), mock.patch(
                "socket.create_connection", side_effect=AssertionError("must not connect")
            ):
                with self.assertRaises(ProviderError):
                    OpenAICompatibleProvider(endpoint=endpoint, model="m")

    def test_model_is_required_by_runtime_provider(self) -> None:
        for model in ("", "   "):
            with self.subTest(model=model), self.assertRaises(ProviderError):
                OpenAICompatibleProvider(
                    endpoint="http://127.0.0.1:1/x", model=model
                )

    class _ServerContext:
        def __init__(self, *, status: int, payload: object, headers: dict[str, str], delay: float) -> None:
            if isinstance(payload, bytes):
                self.payload = payload
            else:
                self.payload = json.dumps(payload).encode("utf-8")
            self.status = status
            self.headers = headers
            self.delay = delay
            self.server: ThreadingHTTPServer | None = None
            self.thread: threading.Thread | None = None
            self.bodies: list[dict[str, object]] = []

        def __enter__(self) -> str:
            context = self

            class Handler(BaseHTTPRequestHandler):
                def do_POST(self) -> None:
                    length = int(self.headers.get("Content-Length", "0"))
                    context.bodies.append(
                        json.loads(self.rfile.read(length).decode("utf-8"))
                    )
                    if context.delay:
                        time.sleep(context.delay)
                    self.send_response(context.status)
                    for key, value in context.headers.items():
                        self.send_header(key, value)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    try:
                        self.wfile.write(context.payload)
                    except (BrokenPipeError, ConnectionResetError):
                        pass

                def log_message(self, format: str, *args: object) -> None:
                    pass

            self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            self.server.daemon_threads = True
            self.thread = threading.Thread(target=self.server.serve_forever)
            self.thread.start()
            host, port = self.server.server_address
            return f"http://{host}:{port}/v1/chat/completions"

        def __exit__(self, *args: object) -> None:
            assert self.server is not None and self.thread is not None
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(2)
            if self.thread.is_alive():
                raise AssertionError("loopback server thread did not stop")

    def _server(
        self,
        *,
        status: int,
        payload: object,
        headers: dict[str, str] | None = None,
        delay: float = 0,
    ) -> _ServerContext:
        return self._ServerContext(
            status=status, payload=payload, headers=headers or {}, delay=delay
        )


def _request() -> ProviderRequest:
    return ProviderRequest(
        purpose=ProviderPurpose.INFER,
        model="fixture-model",
        messages=(ChatMessage(role="user", content="hello"),),
    )


if __name__ == "__main__":
    unittest.main()
