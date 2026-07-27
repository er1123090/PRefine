from __future__ import annotations

import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from methods.mem0_local.build_memory import load_dataset_records, process_example
from methods.mem0_local.runtime import (
    LockedVectorStore,
    TrackedOpenAIClient,
    build_mem0_config,
    lock_local_vector_stores,
    normalize_json_chat_response,
    upstream_metadata,
)
from src.construction_usage import begin_usage_collection, end_usage_collection


class FakeMemory:
    def __init__(self) -> None:
        self.memories: dict[str, list[dict]] = {}
        self.get_all_calls = 0
        self.db = SimpleNamespace(
            connection=FakeConnection(),
            _lock=FakeLock(),
        )

    def get_all(self, *, filters, top_k):
        self.get_all_calls += 1
        return {"results": list(self.memories.get(filters["user_id"], []))[:top_k]}

    def delete_all(self, *, user_id):
        self.memories[user_id] = []
        return {"message": "ok"}

    def add(self, messages, *, user_id, metadata):
        memory = {
            "id": f"{user_id}-{len(self.memories.get(user_id, []))}",
            "memory": messages[0]["content"],
            "metadata": metadata,
        }
        self.memories.setdefault(user_id, []).append(memory)
        return {"results": [{**memory, "event": "ADD"}]}


class FakeConnection:
    def execute(self, _query, _parameters):
        return self

    def commit(self):
        return None


class FakeLock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


class FakeCompletions:
    def __init__(self) -> None:
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"memory": []}'),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=7,
                total_tokens=18,
            )
        )


class Mem0LocalConfigTest(unittest.TestCase):
    def test_compact_json_array_is_loaded_as_records(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "dataset.json"
            path.write_text(
                '[{"example_id":"u1","sessions":[]}]',
                encoding="utf-8",
            )
            records = load_dataset_records(path)
        self.assertEqual(records[0]["example_id"], "u1")

    def test_config_uses_official_vllm_and_local_qdrant(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = build_mem0_config(
                model="gpt-oss-20b",
                base_url="http://127.0.0.1:8000/v1/",
                vector_store_path=root / "qdrant",
                history_db_path=root / "history.sqlite",
                collection_name="test_mem0_local",
                max_tokens=2048,
                embedding_provider="fastembed",
                embedding_model="BAAI/bge-small-en-v1.5",
                embedding_dims=384,
            )

        self.assertEqual(config["llm"]["provider"], "vllm")
        self.assertEqual(config["llm"]["config"]["model"], "gpt-oss-20b")
        self.assertEqual(
            config["llm"]["config"]["vllm_base_url"],
            "http://127.0.0.1:8000/v1",
        )
        self.assertEqual(config["embedder"]["provider"], "fastembed")
        self.assertEqual(config["vector_store"]["provider"], "qdrant")
        self.assertEqual(
            config["vector_store"]["config"]["embedding_model_dims"],
            384,
        )

    def test_checkout_matches_pinned_official_commit(self) -> None:
        metadata = upstream_metadata()
        self.assertTrue(metadata["checkout_present"])
        self.assertTrue(metadata["matches_lock"])
        self.assertEqual(
            metadata["repository"],
            "https://github.com/mem0ai/mem0.git",
        )


class UsageTrackingClientTest(unittest.TestCase):
    def test_reasoning_effort_and_raw_usage_are_recorded(self) -> None:
        completions = FakeCompletions()
        delegate = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        client = TrackedOpenAIClient(
            delegate,
            model="gpt-oss-20b",
            reasoning_effort="low",
            max_completion_tokens=2048,
            disable_response_format=True,
        )
        token = begin_usage_collection()
        try:
            client.chat.completions.create(
                model="gpt-oss-20b",
                messages=[],
                response_format={"type": "json_object"},
            )
        finally:
            report = end_usage_collection(token)

        self.assertEqual(completions.kwargs["reasoning_effort"], "low")
        self.assertEqual(completions.kwargs["max_completion_tokens"], 2048)
        self.assertNotIn("response_format", completions.kwargs)
        self.assertEqual(report["summary"]["input_tokens"], 11)
        self.assertEqual(report["summary"]["output_tokens"], 7)
        self.assertEqual(
            report["by_component"]["mem0_local_llm"]["total_tokens"],
            18,
        )

    def test_json_response_is_canonicalized_from_harmony_wrapper(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='<|channel|>final<|message|>{"memory": []}'
                    ),
                    finish_reason="stop",
                )
            ]
        )

        normalize_json_chat_response(response)

        self.assertEqual(response.choices[0].message.content, '{"memory": []}')

    def test_invalid_json_response_raises_for_retry(self) -> None:
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content='{"memory": ['),
                    finish_reason="length",
                )
            ]
        )

        with self.assertRaisesRegex(ValueError, "finish_reason='length'"):
            normalize_json_chat_response(response)

    def test_local_vector_store_calls_are_serialized(self) -> None:
        class FakeStore:
            is_local = True

            def __init__(self) -> None:
                self.active = 0
                self.max_active = 0
                self.state_lock = threading.Lock()

            def insert(self, value):
                with self.state_lock:
                    self.active += 1
                    self.max_active = max(self.max_active, self.active)
                time.sleep(0.01)
                with self.state_lock:
                    self.active -= 1
                return value

        delegate = FakeStore()
        store = LockedVectorStore(delegate, threading.RLock())
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(store.insert, range(16)))

        self.assertEqual(results, list(range(16)))
        self.assertEqual(delegate.max_active, 1)
        self.assertTrue(store.is_local)

    def test_entity_store_shares_local_store_lock(self) -> None:
        primary = SimpleNamespace(is_local=True)
        entity = SimpleNamespace(is_local=False)
        memory = SimpleNamespace(
            vector_store=primary,
            entity_store=entity,
            _entity_store=entity,
        )

        enabled = lock_local_vector_stores(memory)

        self.assertTrue(enabled)
        self.assertIsInstance(memory.vector_store, LockedVectorStore)
        self.assertIsInstance(memory._entity_store, LockedVectorStore)
        self.assertIs(memory.vector_store._lock, memory._entity_store._lock)


class ConstructionTest(unittest.TestCase):
    def test_snapshots_and_accumulation_are_exported(self) -> None:
        example = {
            "example_id": "user-1",
            "sessions": [
                {
                    "dialogue_id": "d1",
                    "dialogue": [{"role": "user", "message": "quiet room"}],
                    "api_call": [],
                },
                {
                    "dialogue_id": "d2",
                    "dialogue": [{"role": "user", "message": "window seat"}],
                    "api_call": [],
                },
            ],
        }
        result = process_example(
            example,
            dataset_index=3,
            memory=FakeMemory(),
            token_encoding="cl100k_base",
            retry_count=0,
            retry_base_sleep=0,
            snapshot_mode="all",
            configuration_fingerprint="fingerprint",
            upstream={"commit": "test"},
        )

        self.assertEqual(result["method"], "mem0_local")
        self.assertEqual(result["memory_count_final"], 2)
        self.assertEqual(len(result["session_exports"]), 2)
        self.assertEqual(
            result["session_exports"][1]["memory_count_before_session"],
            1,
        )
        self.assertEqual(
            len(
                result["session_exports"][1][
                    "memory_snapshot_after_session"
                ]
            ),
            2,
        )
        self.assertGreater(
            result["construction_total_tokens_lower_bound"],
            0,
        )

    def test_final_snapshot_fetches_state_only_once_after_sessions(self) -> None:
        example = {
            "example_id": "user-1",
            "sessions": [
                {
                    "dialogue_id": "d1",
                    "dialogue": [{"role": "user", "message": "quiet room"}],
                    "api_call": [],
                },
                {
                    "dialogue_id": "d2",
                    "dialogue": [{"role": "user", "message": "window seat"}],
                    "api_call": [],
                },
            ],
        }
        memory = FakeMemory()

        result = process_example(
            example,
            dataset_index=3,
            memory=memory,
            token_encoding="cl100k_base",
            retry_count=0,
            retry_base_sleep=0,
            snapshot_mode="final",
            configuration_fingerprint="fingerprint",
            upstream={"commit": "test"},
        )

        self.assertEqual(memory.get_all_calls, 2)
        self.assertEqual(result["memory_count_final"], 2)
        self.assertGreater(result["stored_memory_tokens_final"], 0)
        self.assertGreater(result["construction_total_tokens_lower_bound"], 0)
        self.assertEqual(
            result["construction_accounting"]["session_state_accounting"],
            "deferred_to_final_snapshot",
        )
        self.assertTrue(
            all(
                export["memory_count_after_session"] is None
                and export["stored_memory_tokens_after_session"] is None
                for export in result["session_exports"]
            )
        )


if __name__ == "__main__":
    unittest.main()
