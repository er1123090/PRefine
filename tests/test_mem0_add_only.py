from __future__ import annotations

import json
from pathlib import Path
import tempfile
import threading
import unittest

from methods.mem0.build_memory_add_only import (
    key_fingerprint,
    load_checkpoint,
    process_example,
)


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeHttpClient:
    def get(self, path):
        event_id = path.rstrip("/").split("/")[-1]
        return _FakeResponse({"event_id": event_id, "status": "SUCCEEDED"})

    def close(self):
        return None


class _FakeMemoryClient:
    calls = []

    def __init__(self, api_key):
        self.api_key = api_key
        self.client = _FakeHttpClient()

    def add(self, messages, user_id):
        self.calls.append((messages, user_id))
        return {"event_id": f"event-{len(self.calls)}"}


class AddOnlyTest(unittest.TestCase):
    def test_checkpoint_tracks_sessions_and_examples(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "checkpoint.jsonl"
            path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "record_type": "session",
                                "status": "OK",
                                "example_id": "u1",
                                "session_index": 1,
                            }
                        ),
                        json.dumps(
                            {
                                "record_type": "example",
                                "status": "OK",
                                "example_id": "u2",
                            }
                        ),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            sessions, examples = load_checkpoint(path)
        self.assertEqual(sessions, {("u1", 1)})
        self.assertEqual(examples, {"u2"})

    def test_add_only_never_reads_memories_or_counts_tokens(self):
        import methods.mem0.build_memory_add_only as module

        original = module.MemoryClient
        module.MemoryClient = _FakeMemoryClient
        _FakeMemoryClient.calls = []
        try:
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "checkpoint.jsonl"
                result = process_example(
                    {
                        "example_id": "u1",
                        "sessions": [
                            {
                                "dialogue_id": "d1",
                                "dialogue": [
                                    {"role": "user", "message": "hello"}
                                ],
                                "api_call": [],
                            },
                            {
                                "dialogue_id": "d2",
                                "dialogue": [
                                    {"role": "user", "message": "again"}
                                ],
                                "api_call": [],
                            },
                        ],
                    },
                    dataset_index=0,
                    api_key="test",
                    completed_sessions={("u1", 1)},
                    output=output,
                    output_lock=threading.Lock(),
                    retry_count=0,
                    retry_base_sleep=0,
                    event_timeout_seconds=1,
                    poll_interval_seconds=0,
                )
                rows = [
                    json.loads(line)
                    for line in output.read_text(encoding="utf-8").splitlines()
                ]
        finally:
            module.MemoryClient = original

        self.assertEqual(len(_FakeMemoryClient.calls), 1)
        self.assertEqual(result["added_sessions_this_run"], 1)
        self.assertEqual(result["skipped_sessions_from_checkpoint"], 1)
        self.assertEqual([row["record_type"] for row in rows], ["session", "example"])
        self.assertNotIn("token_counts", rows[0])
        self.assertNotIn("memory_snapshot", rows[0])

    def test_key_fingerprint_does_not_expose_key(self):
        secret = "m0-example-secret"
        fingerprint = key_fingerprint(secret)
        self.assertNotIn(secret, fingerprint)
        self.assertEqual(len(fingerprint), 16)


if __name__ == "__main__":
    unittest.main()
