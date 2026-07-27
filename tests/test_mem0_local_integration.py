from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest

from methods.mem0_local.build_memory import run_ingestion
from methods.mem0_local.runtime import DEFAULT_UPSTREAM_PATH


class FakeVllmHandler(BaseHTTPRequestHandler):
    requests: list[dict] = []

    def log_message(self, _format, *_args):
        return

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": [{"id": "gpt-oss-20b", "object": "model"}],
                }
            )
            return
        self.send_error(404)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        type(self).requests.append(request)
        self._send_json(
            {
                "id": "chatcmpl-mem0-local-test",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-oss-20b",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": json.dumps(
                                {
                                    "memory": [
                                        {
                                            "text": (
                                                "The user prefers quiet hotel "
                                                "rooms."
                                            ),
                                            "attributed_to": "user",
                                        }
                                    ]
                                }
                            ),
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 37,
                    "completion_tokens": 11,
                    "total_tokens": 48,
                },
            }
        )


@unittest.skipUnless(
    os.environ.get("RUN_MEM0_LOCAL_INTEGRATION") == "1",
    "set RUN_MEM0_LOCAL_INTEGRATION=1 to run local OSS integration",
)
class Mem0LocalIntegrationTest(unittest.TestCase):
    def test_official_mem0_vllm_fastembed_qdrant_pipeline(self) -> None:
        FakeVllmHandler.requests = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), FakeVllmHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                input_path = root / "dataset.json"
                metrics_output = root / "construction.jsonl"
                input_path.write_text(
                    json.dumps(
                        [
                            {
                                "example_id": "local-user",
                                "sessions": [
                                    {
                                        "dialogue_id": "d1",
                                        "dialogue": [
                                            {
                                                "role": "user",
                                                "message": (
                                                    "I prefer quiet hotel rooms."
                                                ),
                                            }
                                        ],
                                        "api_call": [],
                                    }
                                ],
                            }
                        ]
                    ),
                    encoding="utf-8",
                )
                manifest = run_ingestion(
                    input_path=input_path,
                    metrics_output=metrics_output,
                    vector_store_path=root / "qdrant",
                    history_db_path=root / "history.sqlite",
                    upstream_repo_path=DEFAULT_UPSTREAM_PATH,
                    base_url=(
                        f"http://127.0.0.1:{server.server_port}/v1"
                    ),
                    model="gpt-oss-20b",
                    reasoning_effort="low",
                    disable_response_format=False,
                    max_tokens=512,
                    embedding_provider="fastembed",
                    embedding_model="BAAI/bge-small-en-v1.5",
                    embedding_dims=384,
                    embedding_api_key=None,
                    embedding_base_url=None,
                    collection_name="mem0_local_integration",
                    token_encoding="cl100k_base",
                    start_example=0,
                    end_example=None,
                    max_examples=None,
                    max_sessions=None,
                    concurrency=1,
                    resume=False,
                    retry_count=0,
                    retry_base_sleep=0,
                    snapshot_mode="all",
                    skip_preflight=False,
                    dry_run=False,
                )
                result = json.loads(
                    metrics_output.read_text(encoding="utf-8").strip()
                )

            self.assertEqual(manifest["status"], "COMPLETE")
            self.assertEqual(result["memory_count_final"], 1, result)
            self.assertEqual(result["token_counts"]["input_tokens"], 37)
            self.assertEqual(result["token_counts"]["output_tokens"], 11)
            self.assertEqual(
                result["session_exports"][0]["memory_snapshot_after_session"][
                    0
                ]["memory"],
                "The user prefers quiet hotel rooms.",
            )
            self.assertEqual(
                FakeVllmHandler.requests[0]["reasoning_effort"],
                "low",
            )
            self.assertEqual(
                FakeVllmHandler.requests[0]["max_completion_tokens"],
                512,
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
