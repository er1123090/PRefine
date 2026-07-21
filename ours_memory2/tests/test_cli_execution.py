from __future__ import annotations

import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest import mock

from ours_memory2 import cli
from ours_memory2.contracts import ProviderResponse
from ours_memory2.jsonl import read_examples_jsonl
from ours_memory2.step1 import build_and_write_examples
from ours_memory2.step2_multi import infer_multi
from ours_memory2.step2_single import infer_single
from tests.support import ScriptedProvider


FIXTURES = Path(__file__).parent / "fixtures"
STEP2_MANIFEST = FIXTURES / "step2" / "manifest_grouped.json"


def _success(content: str, request_id: str) -> tuple[int, dict[str, object]]:
    return (
        200,
        {
            "id": request_id,
            "choices": [{"message": {"content": content}}],
            "usage": {"total_tokens": 1},
        },
    )


class _LoopbackServer:
    def __init__(self, responses: list[tuple[int, dict[str, object]]]) -> None:
        self.responses = responses
        self.requests: list[dict[str, object]] = []
        self.server: ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def __enter__(self) -> str:
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8"))
                owner.requests.append({"path": self.path, "body": body})
                index = len(owner.requests) - 1
                status, envelope = owner.responses[index]
                payload = json.dumps(envelope).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

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


class CliExecutionTests(unittest.TestCase):
    def test_actual_step1_output_drives_single_and_multi_all_four_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_path = root / "input.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "example_id": 7,
                        "sessions": [
                            {
                                "dialogue": [
                                    {
                                        "role": "user",
                                        "message": "Find a quiet restaurant",
                                    }
                                ],
                                "api_call": [
                                    "Restaurant(city='Seoul', seating='quiet')"
                                ],
                            },
                            {
                                "dialogue": [
                                    {
                                        "role": "user",
                                        "message": "A hotel should feel similar",
                                    }
                                ],
                                "api_call": ["Hotel(city='Seoul', room='quiet')"],
                            },
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            step1_output = root / "step1-output"
            step1_provider = ScriptedProvider(
                [
                    ProviderResponse(text='{"implicit_pref":"quiet"}'),
                    ProviderResponse(text='{"valid":true,"feedback":"ok"}'),
                    ProviderResponse(text='{"implicit_pref":"quiet and calm"}'),
                    ProviderResponse(text='{"valid":true,"feedback":"ok"}'),
                ]
            )
            build_and_write_examples(
                read_examples_jsonl(input_path), step1_provider, step1_output
            )

            fixture_copy = root / "step2"
            shutil.copytree(FIXTURES / "step2", fixture_copy)
            shutil.copyfile(
                step1_output / "memories.jsonl",
                fixture_copy / "resources" / "memories.jsonl",
            )
            manifest = fixture_copy / "manifest_grouped.json"
            for context in (
                "memory_only",
                "memory_api",
                "memory_diag",
                "api_only",
            ):
                with self.subTest(context=context):
                    single_provider = ScriptedProvider(
                        lambda _request: ProviderResponse(text='{"action":"single"}')
                    )
                    multi_provider = ScriptedProvider(
                        lambda _request: ProviderResponse(text='{"action":"multi"}')
                    )
                    single = infer_single(
                        str(manifest),
                        difficulty="all",
                        context=context,
                        model="fixture-model",
                        provider=single_provider,
                    )
                    multi = infer_multi(
                        str(manifest),
                        input_shape="grouped",
                        difficulty="all",
                        context=context,
                        model="fixture-model",
                        provider=multi_provider,
                    )
                    self.assertEqual(len(single.results), 3)
                    self.assertEqual(len(multi.results), 3)
                    for request in (*single_provider.requests, *multi_provider.requests):
                        prompt = request.prompt
                        self.assertEqual(
                            "quiet and calm" in prompt,
                            context != "api_only",
                        )
                        self.assertEqual(
                            "[Session 1] Restaurant" in prompt,
                            context in {"memory_api", "api_only"},
                        )
                        self.assertEqual(
                            "Find a quiet restaurant" in prompt,
                            context == "memory_diag",
                        )

    def test_all_three_commands_execute_from_foreign_cwd_with_ordered_outputs(self) -> None:
        responses = [
            _success('{"implicit_pref":"first"}', "step1-g1"),
            _success('{"valid":true,"feedback":"ok-1"}', "step1-v1"),
            _success('{"implicit_pref":"second-a"}', "step1-g2"),
            _success('{"valid":true,"feedback":"ok-2"}', "step1-v2"),
            _success('{"implicit_pref":"second-b"}', "step1-g3"),
            _success('{"valid":true,"feedback":"ok-3"}', "step1-v3"),
        ]
        responses.extend(_success(f'{{"action":"single-{index}"}}', f"single-{index}") for index in range(3))
        responses.extend(_success(f'{{"action":"multi-{index}"}}', f"multi-{index}") for index in range(3))

        server = _LoopbackServer(responses)
        with tempfile.TemporaryDirectory() as temp, server as endpoint:
            root = Path(temp)
            step1_output = root / "step1"
            single_output = root / "single"
            multi_output = root / "multi"
            original_cwd = Path.cwd()
            try:
                os.chdir("/tmp")
                self.assertEqual(
                    cli.main(self._step1_args(endpoint, step1_output)),
                    0,
                )
                self.assertEqual(
                    cli.main(self._step2_args("single", endpoint, single_output)),
                    0,
                )
                self.assertEqual(
                    cli.main(self._step2_args("multi", endpoint, multi_output)),
                    0,
                )
                self.assertEqual(Path.cwd(), Path("/tmp"))
            finally:
                os.chdir(original_cwd)

            finals = self._rows(step1_output / "memories.jsonl")
            drafts = self._rows(step1_output / "drafts.jsonl")
            verifiers = self._rows(step1_output / "verifiers.jsonl")
            self.assertEqual([row["example_id"] for row in finals], ["7", "8"])
            self.assertEqual(
                [(row["example_id"], row["session_index"], row["step"]) for row in drafts],
                [("7", 1, 1), ("8", 1, 1), ("8", 2, 1)],
            )
            self.assertEqual(
                [(row["example_id"], row["session_index"], row["step"]) for row in verifiers],
                [("7", 1, 1), ("8", 1, 1), ("8", 2, 1)],
            )
            for output, mode in ((single_output, "single"), (multi_output, "multi")):
                results = self._rows(output / "results.jsonl")
                diagnostics = self._rows(output / "diagnostics.jsonl")
                self.assertEqual(len(results), 3)
                self.assertEqual(len(diagnostics), 3)
                self.assertEqual(
                    [row["difficulty"] for row in results],
                    ["easy", "medium", "hard"],
                )
                self.assertEqual([row["mode"] for row in results], [mode] * 3)
                self.assertEqual(
                    [row["case_id"] for row in results],
                    [row["case_id"] for row in diagnostics],
                )
                self.assertTrue(
                    all(
                        row["provider"]["request"]["purpose"] == "infer"
                        for row in diagnostics
                    )
                )

        self.assertEqual(len(server.requests), 12)
        self.assertEqual({row["path"] for row in server.requests}, {"/v1/chat/completions"})
        for row in server.requests[:6]:
            body = row["body"]
            self.assertIn("temperature", body)
            self.assertEqual(body["response_format"], {"type": "json_object"})
            self.assertEqual(
                [message["role"] for message in body["messages"]],
                ["system", "user"],
            )
        for row in server.requests[6:]:
            body = row["body"]
            self.assertNotIn("temperature", body)
            self.assertNotIn("response_format", body)
            self.assertEqual(
                [message["role"] for message in body["messages"]], ["user"]
            )

    def test_all_null_hard_rules_exit_two_before_provider_and_output_for_both_families(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            fixture = root / "step2"
            shutil.copytree(FIXTURES / "step2", fixture)
            group_path = fixture / "resources" / "preference_groups.json"
            groups = json.loads(group_path.read_text(encoding="utf-8"))
            for rule in groups["comfort"]["rules"]:
                if rule["domain"] == "Hotel":
                    rule["value"] = None
            group_path.write_text(json.dumps(groups), encoding="utf-8")
            manifest = fixture / "manifest_grouped.json"
            with mock.patch(
                "ours_memory2.cli.OpenAICompatibleProvider",
                side_effect=AssertionError(
                    "provider constructed for all-null hard derivation"
                ),
            ) as provider_type:
                for command in ("single", "multi"):
                    output = root / f"output-{command}"
                    args = list(
                        self._step2_args(
                            command,
                            "http://127.0.0.1:1/x",
                            output,
                            manifest,
                        )
                    )
                    args[args.index("--difficulty") + 1] = "hard"
                    with self.subTest(command=command), contextlib.redirect_stderr(
                        io.StringIO()
                    ):
                        self.assertEqual(cli.main(tuple(args)), 2)
                        self.assertFalse(output.exists())
            self.assertEqual(provider_type.call_count, 0)

    def test_invalid_inputs_are_pre_provider_and_create_no_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            invalid_step1 = root / "invalid.jsonl"
            invalid_step1.write_text("[]\n", encoding="utf-8")
            invalid_manifest = root / "manifest.json"
            invalid_manifest.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "examples": "missing.json",
                        "memories": "missing.jsonl",
                        "single_query_map": "missing.json",
                        "multiturn_templates": "missing.json",
                        "preference_slots": "missing.json",
                        "preference_groups": "missing.json",
                        "tool_schema": "missing.json",
                    }
                ),
                encoding="utf-8",
            )
            cases = (
                self._step1_args("http://127.0.0.1:1/x", root / "step1", invalid_step1),
                self._step2_args("single", "http://127.0.0.1:1/x", root / "single", invalid_manifest),
                self._step2_args("multi", "http://127.0.0.1:1/x", root / "multi", invalid_manifest),
                (*self._step1_args("http://127.0.0.1:1/x", root / "timeout")[:-1], "nan"),
                tuple(
                    "unknown" if value == "memory_api" else value
                    for value in self._step2_args(
                        "single", "http://127.0.0.1:1/x", root / "context"
                    )
                ),
            )
            with mock.patch(
                "ours_memory2.cli.OpenAICompatibleProvider",
                side_effect=AssertionError("provider constructed before validation"),
            ):
                for args in cases:
                    output = Path(args[args.index("--output-root") + 1])
                    with self.subTest(command=args[:2]), contextlib.redirect_stderr(io.StringIO()):
                        self.assertEqual(cli.main(args), 2)
                        self.assertFalse(output.exists())

    def test_complete_step1_jsonl_negative_matrix_is_pre_provider(self) -> None:
        valid = {
            "example_id": "x",
            "sessions": [
                {
                    "dialogue": [{"role": "user", "message": "message"}],
                    "api_call": [],
                }
            ],
        }

        def encoded(value: object) -> bytes:
            return (json.dumps(value) + "\n").encode()

        cases = {
            "collision": (
                encoded(dict(valid, example_id=7))
                + encoded(dict(valid, example_id="7")),
                "input.line[2].example_id",
            ),
            "duplicate": (encoded(valid) + encoded(valid), "input.line[2].example_id"),
            "missing_id": (encoded({"sessions": valid["sessions"]}), "input.line[1].example_id"),
            "empty_id": (encoded(dict(valid, example_id=" ")), "input.line[1].example_id"),
            "missing_sessions": (encoded({"example_id": "x"}), "input.line[1].sessions"),
            "sessions_type": (encoded(dict(valid, sessions={})), "input.line[1].sessions"),
            "empty_sessions": (encoded(dict(valid, sessions=[])), "input.line[1].sessions"),
            "session_object": (encoded(dict(valid, sessions=[1])), "input.line[1].sessions[0]"),
            "missing_dialogue": (encoded(dict(valid, sessions=[{"api_call": []}])), "input.line[1].sessions[0].dialogue"),
            "dialogue_type": (encoded(dict(valid, sessions=[{"dialogue": {}, "api_call": []}])), "input.line[1].sessions[0].dialogue"),
            "empty_dialogue": (encoded(dict(valid, sessions=[{"dialogue": [], "api_call": []}])), "input.line[1].sessions[0].dialogue"),
            "turn_object": (encoded(dict(valid, sessions=[{"dialogue": [1], "api_call": []}])), "input.line[1].sessions[0].dialogue[0]"),
            "role": (encoded(dict(valid, sessions=[{"dialogue": [{"role": "", "message": "m"}], "api_call": []}])), "input.line[1].sessions[0].dialogue[0].role"),
            "message": (encoded(dict(valid, sessions=[{"dialogue": [{"role": "u", "message": 1}], "api_call": []}])), "input.line[1].sessions[0].dialogue[0].message"),
            "missing_api": (encoded(dict(valid, sessions=[{"dialogue": valid["sessions"][0]["dialogue"]}])), "input.line[1].sessions[0].api_call"),
            "api_type": (encoded(dict(valid, sessions=[{"dialogue": valid["sessions"][0]["dialogue"], "api_call": {}}])), "input.line[1].sessions[0].api_call"),
            "api_item": (encoded(dict(valid, sessions=[{"dialogue": valid["sessions"][0]["dialogue"], "api_call": [1]}])), "input.line[1].sessions[0].api_call[0]"),
            "array": (b"[]\n", "input.line[1]"),
            "wrapper": (b'{"examples":[]}\n', "input.line[1].example_id"),
            "pretty": (b'{\n"example_id":"x",\n"sessions":[]\n}\n', "input.line[1]"),
            "utf8": (b"\xff\n", "input.line[1]"),
            "json": (b"{bad}\n", "input.line[1]"),
        }
        with tempfile.TemporaryDirectory() as temp, mock.patch(
            "ours_memory2.cli.OpenAICompatibleProvider",
            side_effect=AssertionError("provider constructed before validation"),
        ):
            root = Path(temp)
            for index, (name, (payload, expected_path)) in enumerate(cases.items()):
                input_path = root / f"{index}.jsonl"
                input_path.write_bytes(payload)
                output = root / f"output-{index}"
                stderr = io.StringIO()
                with self.subTest(name=name), contextlib.redirect_stderr(stderr):
                    self.assertEqual(
                        cli.main(
                            self._step1_args(
                                "http://127.0.0.1:1/x", output, input_path
                            )
                        ),
                        2,
                    )
                    self.assertIn(expected_path, stderr.getvalue())
                    self.assertFalse(output.exists())

    def test_provider_failures_write_no_current_run_outputs(self) -> None:
        failures = [
            (500, {"error": "do not expose"}),
            _success('{"action":"first-case"}', "first-case"),
            (500, {"error": "do not expose"}),
        ]
        with tempfile.TemporaryDirectory() as temp, _LoopbackServer(failures) as endpoint:
            root = Path(temp)
            step1_output = root / "step1"
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(cli.main(self._step1_args(endpoint, step1_output)), 1)
            self.assertFalse(step1_output.exists())

            step2_output = root / "single"
            step2_output.mkdir()
            existing = {
                "results.jsonl": b'{"existing":"result"}\n',
                "diagnostics.jsonl": b'{"existing":"diagnostic"}\n',
            }
            for name, payload in existing.items():
                (step2_output / name).write_bytes(payload)
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(
                    cli.main(self._step2_args("single", endpoint, step2_output)),
                    1,
                )
            self.assertEqual(
                {name: (step2_output / name).read_bytes() for name in existing},
                existing,
            )

    def test_malformed_loopback_endpoint_is_exit_one_and_writes_nothing(self) -> None:
        malformed = (
            "http://127.0.0.1:not-numeric/x",
            "http://127.0.0.1:65536/x",
            "http://127.0.0.1:80/x\r\nHeader: value",
        )
        with tempfile.TemporaryDirectory() as temp:
            for index, endpoint in enumerate(malformed):
                output = Path(temp) / str(index)
                stderr = io.StringIO()
                with self.subTest(endpoint=endpoint), contextlib.redirect_stderr(stderr):
                    self.assertEqual(cli.main(self._step1_args(endpoint, output)), 1)
                    self.assertFalse(output.exists())
                    self.assertNotIn(endpoint, stderr.getvalue())

    @staticmethod
    def _step1_args(
        endpoint: str, output: Path, input_path: Path = FIXTURES / "valid_two_examples.jsonl"
    ) -> tuple[str, ...]:
        return (
            "step1", "build", "--input", str(input_path), "--output-root", str(output),
            "--endpoint", endpoint, "--model", "fixture-model", "--timeout", "2",
        )

    @staticmethod
    def _step2_args(
        command: str,
        endpoint: str,
        output: Path,
        manifest: Path = STEP2_MANIFEST,
    ) -> tuple[str, ...]:
        args = [
            "step2", command, "--manifest", str(manifest), "--difficulty", "all",
            "--context", "memory_api", "--output-root", str(output),
            "--endpoint", endpoint, "--model", "fixture-model", "--timeout", "2",
        ]
        if command == "multi":
            args.extend(("--input-shape", "auto"))
        return tuple(args)

    @staticmethod
    def _rows(path: Path) -> list[dict[str, object]]:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


if __name__ == "__main__":
    unittest.main()
