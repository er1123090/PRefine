from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


from exp7.methods.contracts import (
    MethodContractError,
    MethodRunContext,
    validate_prepared_records,
)
from exp7.methods.mem0 import Mem0Adapter, Mem0Runtime


def source_example(*, second_message: str = "I still prefer four stars."):
    return {
        "example_id": "source-7",
        "sessions": [
            {
                "dialogue_id": "dialogue-1",
                "dialogue": [
                    {"role": "USER", "message": "I prefer four-star hotels."},
                    {"role": "ASSISTANT", "message": "Understood."},
                ],
                "api_call": ['GetHotels(star="4")'],
            },
            {
                "dialogue_id": "dialogue-2",
                "dialogue": [
                    {"role": "user", "message": second_message},
                ],
                "api_call": [],
            },
        ],
    }


def prepared(
    *,
    instance_id: str,
    turn: str,
    query: str,
    source=None,
):
    return {
        "dataset_id": "mix600-v1",
        "difficulty": "easy" if turn == "singleturn" else "hard",
        "ground_truth": ['GetHotels(star="4")'],
        "instance_id": instance_id,
        "query": query,
        "source_example": source if source is not None else source_example(),
        "source_example_id": "source-7",
        "turn": turn,
    }


class FakeMemoryBackend:
    def __init__(self) -> None:
        self.adds = []
        self.searches = []
        self.destructive_calls = []

    async def add(self, messages, *, user_id, metadata):
        self.adds.append(
            {
                "messages": [dict(message) for message in messages],
                "metadata": dict(metadata),
                "user_id": user_id,
            }
        )
        return {"status": "accepted"}

    def search(self, *, query, user_id, filters):
        self.searches.append(
            {"filters": dict(filters), "query": query, "user_id": user_id}
        )
        namespace, source_id = user_id.split("::", 1)
        return {
            "results": [
                {
                    "id": f"memory-{len(self.searches)}",
                    "memory": "The user prefers four-star hotels.",
                    "metadata": {
                        "namespace": namespace,
                        "source_example_id": source_id,
                    },
                    "user_id": user_id,
                }
            ]
        }

    def delete(self, *args, **kwargs):
        self.destructive_calls.append(("delete", args, kwargs))
        raise AssertionError("adapter must not delete remote memory")

    def delete_all(self, *args, **kwargs):
        self.destructive_calls.append(("delete_all", args, kwargs))
        raise AssertionError("adapter must not clear remote memory")


class FakeGenerator:
    def __init__(self) -> None:
        self.calls = []

    async def __call__(self, record, request):
        self.calls.append((dict(record), request))
        return {
            "content": 'GetHotels(star="4")',
            "reasoning_content": "used the scoped retrieved preference",
        }


class Mem0AdapterTests(unittest.TestCase):
    def context(self, root: Path, digest: str = "a" * 64):
        return MethodRunContext(
            run_dir=root,
            dataset_manifest_sha256=digest,
            model_name="fake-model",
            tools_schema=({"type": "function"},),
            reasoning_effort="low",
        )

    def runtime(self):
        memory = FakeMemoryBackend()
        generator = FakeGenerator()
        return Mem0Runtime(memory, generator), memory, generator

    def records(self):
        return validate_prepared_records(
            [
                prepared(
                    instance_id="single-0",
                    turn="singleturn",
                    query="Find a suitable hotel.",
                ),
                prepared(
                    instance_id="multi-0",
                    turn="multiturn",
                    query="USER: What kind did I prefer?\nASSISTANT: Let me check.",
                ),
            ]
        )

    def test_build_deduplicates_source_and_reconstructs_session_batches(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime, memory, generator = self.runtime()
            state = asyncio.run(
                Mem0Adapter(namespace="suite-a").build(
                    self.records(), self.context(root), runtime
                )
            )

            self.assertEqual(len(memory.adds), 2)
            self.assertEqual(generator.calls, [])
            self.assertEqual(memory.destructive_calls, [])
            self.assertEqual(
                {call["user_id"] for call in memory.adds},
                {"suite-a::source-7"},
            )
            self.assertEqual(memory.adds[0]["messages"][0]["role"], "user")
            self.assertEqual(memory.adds[0]["messages"][1]["role"], "assistant")
            self.assertEqual(
                memory.adds[0]["messages"][-1]["content"],
                "[System Summary] API Calls executed in this session: "
                "['GetHotels(star=\"4\")']",
            )
            self.assertEqual(
                memory.adds[0]["metadata"]["namespace"], "suite-a"
            )
            self.assertEqual(state.metadata["source_count"], 1)
            self.assertEqual(state.metadata["session_batch_count"], 2)
            self.assertTrue(state.path.is_relative_to(root))
            manifest = json.loads(
                (state.path / "state_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["namespace"], "suite-a")
            self.assertEqual(
                manifest["dataset_manifest_sha256"], "a" * 64
            )
            self.assertEqual(manifest["records_sha256"], state.records_sha256)

    def test_single_and_multiturn_inference_use_exact_query_and_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = self.context(root)
            records = self.records()
            runtime, memory, generator = self.runtime()
            adapter = Mem0Adapter(namespace="suite-b")
            state = asyncio.run(adapter.build(records, context, runtime))

            outputs = [
                asyncio.run(adapter.infer(record, context, state, runtime))
                for record in records
            ]

            self.assertEqual(
                [call["query"] for call in memory.searches],
                [record.query for record in records],
            )
            self.assertEqual(
                {call["user_id"] for call in memory.searches},
                {"suite-b::source-7"},
            )
            self.assertTrue(
                all(
                    call["filters"] == {"user_id": "suite-b::source-7"}
                    for call in memory.searches
                )
            )
            self.assertEqual(
                [call[1].turn for call in generator.calls],
                ["singleturn", "multiturn"],
            )
            self.assertEqual(
                [call[1].retrieval_query for call in generator.calls],
                [record.query for record in records],
            )
            for record, output in zip(records, outputs):
                self.assertEqual(output["instance_id"], record.instance_id)
                self.assertEqual(output["source_example_id"], "source-7")
                self.assertEqual(output["query"], record.query)
                self.assertEqual(
                    output["ground_truth"], ['GetHotels(star="4")']
                )
                self.assertEqual(output["method_id"], "mem0")
                self.assertEqual(
                    output["retrieved_memories"][0]["memory"],
                    "The user prefers four-star hotels.",
                )
            self.assertEqual(memory.destructive_calls, [])

    def test_namespaces_isolate_remote_identity_and_local_state(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = self.context(root)
            records = self.records()
            runtime, memory, _ = self.runtime()
            first = Mem0Adapter(namespace="run-one")
            second = Mem0Adapter(namespace="run-two")
            first_state = asyncio.run(first.build(records, context, runtime))
            second_state = asyncio.run(second.build(records, context, runtime))

            self.assertNotEqual(first_state.path, second_state.path)
            self.assertTrue(first_state.path.is_relative_to(root))
            self.assertTrue(second_state.path.is_relative_to(root))
            self.assertEqual(
                {call["user_id"] for call in memory.adds},
                {"run-one::source-7", "run-two::source-7"},
            )
            self.assertEqual(memory.destructive_calls, [])

    def test_missing_or_tampered_state_fails_before_external_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = self.context(root)
            records = self.records()
            runtime, memory, generator = self.runtime()
            adapter = Mem0Adapter(namespace="tamper-test")
            state = asyncio.run(adapter.build(records, context, runtime))

            with self.assertRaisesRegex(MethodContractError, "requires built state"):
                asyncio.run(adapter.infer(records[0], context, None, runtime))
            manifest_path = state.path / "state_manifest.json"
            manifest_path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(MethodContractError, "SHA-256 mismatch"):
                asyncio.run(adapter.infer(records[0], context, state, runtime))
            self.assertEqual(memory.searches, [])
            self.assertEqual(generator.calls, [])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = self.context(root)
            records = self.records()
            runtime, memory, generator = self.runtime()
            adapter = Mem0Adapter(namespace="missing-test")
            state = asyncio.run(adapter.build(records, context, runtime))
            (state.path / "state_manifest.json").unlink()
            with self.assertRaisesRegex(MethodContractError, "missing or unsafe"):
                asyncio.run(adapter.infer(records[0], context, state, runtime))
            self.assertEqual(memory.searches, [])
            self.assertEqual(generator.calls, [])

    def test_state_manifest_rejects_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = self.context(root)
            records = self.records()
            runtime, memory, generator = self.runtime()
            adapter = Mem0Adapter(namespace="strict-json")
            state = asyncio.run(adapter.build(records, context, runtime))
            path = state.path / "state_manifest.json"
            payload = path.read_bytes().replace(
                b"{\n", b'{\n  "format_version": 1,\n', 1
            )
            path.write_bytes(payload)
            forged = replace(
                state,
                metadata={
                    **state.metadata,
                    "state_manifest_sha256": hashlib.sha256(payload).hexdigest(),
                },
            )
            with self.assertRaisesRegex(MethodContractError, "duplicate object key"):
                asyncio.run(adapter.infer(records[0], context, forged, runtime))
            self.assertEqual(memory.searches, [])
            self.assertEqual(generator.calls, [])

    def test_state_path_escape_and_symlink_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime, memory, _ = self.runtime()
            adapter = Mem0Adapter(namespace="safe", state_root="../outside")
            with self.assertRaisesRegex(MethodContractError, "unsafe method state path"):
                asyncio.run(adapter.build(self.records(), self.context(root), runtime))
            self.assertEqual(memory.adds, [])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root.parent / f"{root.name}-outside"
            outside.mkdir()
            try:
                (root / "method_state").symlink_to(outside, target_is_directory=True)
                runtime, memory, _ = self.runtime()
                with self.assertRaisesRegex(MethodContractError, "symlinks"):
                    asyncio.run(
                        Mem0Adapter(namespace="safe").build(
                            self.records(), self.context(root), runtime
                        )
                    )
                self.assertEqual(memory.adds, [])
            finally:
                (root / "method_state").unlink()
                outside.rmdir()

    def test_conflicting_source_payloads_fail_before_memory_writes(self):
        records = validate_prepared_records(
            [
                prepared(
                    instance_id="first",
                    turn="singleturn",
                    query="first query",
                ),
                prepared(
                    instance_id="second",
                    turn="multiturn",
                    query="second query",
                    source=source_example(second_message="A conflicting history."),
                ),
            ]
        )
        with tempfile.TemporaryDirectory() as temporary:
            runtime, memory, _ = self.runtime()
            with self.assertRaisesRegex(MethodContractError, "conflicting payloads"):
                asyncio.run(
                    Mem0Adapter(namespace="conflict").build(
                        records,
                        self.context(Path(temporary)),
                        runtime,
                    )
                )
            self.assertEqual(memory.adds, [])

    def test_cross_identity_memory_and_generator_identity_drift_are_rejected(self):
        class CrossIdentityMemory(FakeMemoryBackend):
            def search(self, *, query, user_id, filters):
                del query, filters
                return {
                    "results": [
                        {
                            "memory": "wrong user's preference",
                            "user_id": f"other::{user_id}",
                        }
                    ]
                }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = self.context(root)
            records = self.records()
            memory = CrossIdentityMemory()
            generator = FakeGenerator()
            runtime = Mem0Runtime(memory, generator)
            adapter = Mem0Adapter(namespace="scope-test")
            state = asyncio.run(adapter.build(records, context, runtime))
            with self.assertRaisesRegex(MethodContractError, "cross-identity"):
                asyncio.run(adapter.infer(records[0], context, state, runtime))
            self.assertEqual(generator.calls, [])

        class DriftingGenerator:
            def __call__(self, record, request):
                del record, request
                return {"prediction": "x", "instance_id": "other"}

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = self.context(root)
            records = self.records()
            memory = FakeMemoryBackend()
            runtime = Mem0Runtime(memory, DriftingGenerator())
            adapter = Mem0Adapter(namespace="drift-test")
            state = asyncio.run(adapter.build(records, context, runtime))
            with self.assertRaisesRegex(MethodContractError, "instance_id"):
                asyncio.run(adapter.infer(records[0], context, state, runtime))


if __name__ == "__main__":
    unittest.main()
