from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from methods.langmem.build_memory_batch import (
    allocate_integer,
    apply_operations_to_items,
    build_batch_request,
    initialize,
    manager_messages,
    paths_for,
    plan_memory_operations,
    replay_operations,
    request_tools,
    retrieve_memories,
    run_direct_loop,
    run_loop,
    stable_memory_id,
)


def semantic_value(content: str, value: str) -> dict:
    return {
        "kind": "SemanticMemory",
        "content": {
            "content": content,
            "category": "explicit_preference",
            "domain": "flight",
            "slot": "seat",
            "value": value,
            "context": None,
        },
    }


def memory_item(
    key: str,
    content: str,
    value: str,
    embedding: list[float] | None = None,
) -> dict:
    item = {
        "namespace": ["langmem", "u1", "semantic"],
        "key": key,
        "value": semantic_value(content, value),
        "created_at": "2026-01-01T00:00:00+00:00",
        "updated_at": "2026-01-01T00:00:00+00:00",
        "score": None,
        "construction_provenance": {
            "created_session_index": 1,
            "updated_session_index": 1,
        },
    }
    if embedding is not None:
        item["_embedding"] = np.asarray(embedding, dtype=np.float32)
    return item


class LangMemBatchPayloadTest(unittest.TestCase):
    def test_existing_memory_request_uses_required_patch_insert_delete_tools(self) -> None:
        item = memory_item("k1", "The user prefers aisle seats.", "aisle")
        item["stable_id"] = stable_memory_id("u1", "k1")
        messages = manager_messages(
            [{"role": "user", "content": "I prefer window seats now."}],
            [item],
            "session-tag",
        )
        tools = request_tools([item])
        request = build_batch_request(
            custom_id="lm-i0000-s02",
            model="gpt-5-mini",
            reasoning_effort="minimal",
            max_completion_tokens=2048,
            messages=messages,
            retrieved=[item],
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in tools],
            ["PatchDoc", "SemanticMemory", "RemoveDoc"],
        )
        self.assertEqual(request["body"]["tool_choice"], "required")
        self.assertEqual(request["body"]["reasoning_effort"], "minimal")
        self.assertNotIn("temperature", request["body"])
        self.assertIn(item["stable_id"], json.dumps(request["body"]))

    def test_first_session_keeps_tool_choice_optional(self) -> None:
        request = build_batch_request(
            custom_id="lm-i0000-s01",
            model="gpt-5-mini",
            reasoning_effort="minimal",
            max_completion_tokens=2048,
            messages=manager_messages(
                [{"role": "user", "content": "I prefer aisle seats."}],
                [],
                "session-tag",
            ),
            retrieved=[],
        )
        self.assertNotIn("tool_choice", request["body"])
        self.assertEqual(
            [tool["function"]["name"] for tool in request["body"]["tools"]],
            ["SemanticMemory"],
        )


class LangMemBatchOperationTest(unittest.TestCase):
    def test_patch_is_validated_applied_and_replayable(self) -> None:
        item = memory_item("k1", "The user prefers aisle seats.", "aisle")
        stable_id = stable_memory_id("u1", "k1")
        retrieved = {
            **item,
            "stable_id": stable_id,
            "score": 0.9,
        }
        body = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "type": "function",
                                "function": {
                                    "name": "PatchDoc",
                                    "arguments": json.dumps(
                                        {
                                            "json_doc_id": stable_id,
                                            "planned_edits": "replace seat",
                                            "patches": [
                                                {
                                                    "op": "replace",
                                                    "path": "/content",
                                                    "value": (
                                                        "The user now prefers "
                                                        "window seats."
                                                    ),
                                                },
                                                {
                                                    "op": "replace",
                                                    "path": "/value",
                                                    "value": "window",
                                                },
                                            ],
                                        }
                                    ),
                                },
                            }
                        ]
                    }
                }
            ]
        }
        manifest = {
            "example_id": "u1",
            "session_index": 2,
            "retrieved_memories": [retrieved],
        }
        operations = plan_memory_operations(
            "lm-i0000-s02", body, manifest, [item]
        )
        after = apply_operations_to_items(
            [item],
            operations,
            "2026-01-02T00:00:00+00:00",
            2,
        )
        replayed: dict[str, dict] = {"k1": item}
        replay_operations(replayed, operations)

        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0]["operation"], "update")
        self.assertEqual(after[0]["value"]["content"]["value"], "window")
        self.assertEqual(
            replayed["k1"]["value"]["content"]["value"], "window"
        )
        self.assertEqual(
            replayed["k1"]["construction_provenance"][
                "updated_session_index"
            ],
            2,
        )

    def test_retrieval_uses_cosine_similarity_and_limits_results(self) -> None:
        items = [
            memory_item("x", "x", "x", [1.0, 0.0]),
            memory_item("y", "y", "y", [0.0, 1.0]),
        ]
        retrieved = retrieve_memories(
            items, np.asarray([0.9, 0.1], dtype=np.float32), limit=1
        )
        self.assertEqual([item["key"] for item in retrieved], ["x"])
        self.assertIn("stable_id", retrieved[0])

    def test_usage_allocation_preserves_provider_total(self) -> None:
        allocations = allocate_integer(17, [1, 2, 3])
        self.assertEqual(sum(allocations), 17)
        self.assertGreaterEqual(allocations[2], allocations[1])
        self.assertGreaterEqual(allocations[1], allocations[0])

    def test_concatenated_optional_fields_are_repaired_and_audited(self) -> None:
        body = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-malformed",
                                "type": "function",
                                "function": {
                                    "name": "SemanticMemory",
                                    "arguments": json.dumps(
                                        {
                                            "content": "Initially requested San Diego.",
                                            "category": (
                                                "profile_fact','domain':'rental_car',"
                                                "'slot':'initial_pickup_city',"
                                                "'value':'San Diego',"
                                                "'context':'Initial request'"
                                            ),
                                        }
                                    ),
                                },
                            }
                        ]
                    }
                }
            ]
        }
        operations = plan_memory_operations(
            "lm-i0455-s01",
            body,
            {
                "example_id": "conflict_dev_0036_mix1_noise",
                "session_index": 1,
                "retrieved_memories": [],
            },
            [],
        )

        content = operations[0]["after"]["value"]["content"]
        self.assertEqual(content["category"], "profile_fact")
        self.assertEqual(content["domain"], "rental_car")
        self.assertEqual(content["slot"], "initial_pickup_city")
        self.assertEqual(content["value"], "San Diego")
        self.assertEqual(
            operations[0]["normalization_repairs"][0]["type"],
            "unpacked_concatenated_semantic_fields",
        )

    def test_nested_remove_doc_is_reclassified(self) -> None:
        item = memory_item("k1", "Old preference.", "old")
        stable_id = stable_memory_id("u1", "k1")
        operations = plan_memory_operations(
            "lm-i0000-s02",
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "bad-remove",
                                    "type": "function",
                                    "function": {
                                        "name": "SemanticMemory",
                                        "arguments": json.dumps(
                                            {
                                                "recipient_name": (
                                                    "functions.RemoveDoc"
                                                ),
                                                "parameters": {
                                                    "json_doc_id": stable_id
                                                },
                                            }
                                        ),
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "example_id": "u1",
                "session_index": 2,
                "retrieved_memories": [
                    {**item, "stable_id": stable_id}
                ],
            },
            [item],
        )
        self.assertEqual(operations[0]["operation"], "delete")
        self.assertEqual(
            operations[0]["normalization_repairs"][0]["type"],
            "reclassified_nested_remove_doc",
        )

    def test_invalid_aggregate_patch_becomes_audited_noop(self) -> None:
        item = memory_item("k1", "Existing preference.", "existing")
        stable_id = stable_memory_id("u1", "k1")
        operations = plan_memory_operations(
            "lm-i0000-s02",
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "bad-patch",
                                    "type": "function",
                                    "function": {
                                        "name": "PatchDoc",
                                        "arguments": json.dumps(
                                            {
                                                "json_doc_id": "memory_store",
                                                "planned_edits": "aggregate",
                                                "patches": [],
                                            }
                                        ),
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "example_id": "u1",
                "session_index": 2,
                "retrieved_memories": [
                    {**item, "stable_id": stable_id}
                ],
            },
            [item],
        )
        self.assertEqual(operations[0]["operation"], "noop")
        self.assertEqual(
            operations[0]["normalization_repairs"][0]["type"],
            "ignored_aggregate_patch_unknown_target",
        )

    def test_existing_placeholder_patch_becomes_audited_noop(self) -> None:
        item = memory_item("k1", "Existing preference.", "existing")
        stable_id = stable_memory_id("u1", "k1")
        operations = plan_memory_operations(
            "lm-i0000-s02",
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "placeholder-patch",
                                    "type": "function",
                                    "function": {
                                        "name": "PatchDoc",
                                        "arguments": json.dumps(
                                            {
                                                "json_doc_id": "existing",
                                                "planned_edits": "aggregate",
                                                "patches": [],
                                            }
                                        ),
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "example_id": "u1",
                "session_index": 2,
                "retrieved_memories": [
                    {**item, "stable_id": stable_id}
                ],
            },
            [item],
        )
        self.assertEqual(operations[0]["operation"], "noop")
        self.assertEqual(
            operations[0]["normalization_repairs"][0]["json_doc_id"],
            "existing",
        )

    def test_truncated_remove_id_is_repaired(self) -> None:
        item = memory_item("k1", "Existing preference.", "existing")
        stable_id = stable_memory_id("u1", "k1")
        operations = plan_memory_operations(
            "lm-i0000-s02",
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "truncated-remove",
                                    "type": "function",
                                    "function": {
                                        "name": "RemoveDoc",
                                        "arguments": json.dumps(
                                            {"json_doc_id": stable_id[:-1]}
                                        ),
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "example_id": "u1",
                "session_index": 2,
                "retrieved_memories": [
                    {**item, "stable_id": stable_id}
                ],
            },
            [item],
        )
        self.assertEqual(operations[0]["operation"], "delete")
        self.assertEqual(
            operations[0]["normalization_repairs"][0]["type"],
            "repaired_truncated_memory_id",
        )

    def test_malformed_semantic_reference_is_audited_noop(self) -> None:
        operations = plan_memory_operations(
            "lm-i0000-s02",
            {
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "malformed-reference",
                                    "type": "function",
                                    "function": {
                                        "name": "SemanticMemory",
                                        "arguments": json.dumps(
                                            {
                                                "json_doc_id": "unknown",
                                                "parameters": {},
                                            }
                                        ),
                                    },
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "example_id": "u1",
                "session_index": 2,
                "retrieved_memories": [],
            },
            [],
        )
        self.assertEqual(operations[0]["operation"], "noop")
        self.assertEqual(
            operations[0]["normalization_repairs"][0]["type"],
            "ignored_malformed_semantic_memory_reference",
        )


class FakeFiles:
    def __init__(self) -> None:
        self.values: dict[str, bytes] = {}
        self.counter = 0

    def create(self, file, purpose: str):
        self.counter += 1
        file_id = f"input-{self.counter}"
        self.values[file_id] = file.read()
        return SimpleNamespace(id=file_id)

    def content(self, file_id: str):
        return SimpleNamespace(content=self.values[file_id])


class FakeEmbeddings:
    def create(self, model: str, input: list[str]):
        data = [
            SimpleNamespace(index=index, embedding=[1.0, float(index) / 10])
            for index, _ in enumerate(input)
        ]
        token_count = sum(max(1, len(text.split())) for text in input)
        usage = SimpleNamespace(
            prompt_tokens=token_count,
            total_tokens=token_count,
        )
        return SimpleNamespace(data=data, usage=usage)


class FakeBatches:
    def __init__(self, files: FakeFiles) -> None:
        self.files = files
        self.values: dict[str, SimpleNamespace] = {}
        self.counter = 0

    def create(
        self,
        input_file_id: str,
        endpoint: str,
        completion_window: str,
        metadata: dict,
    ):
        self.counter += 1
        batch_id = f"batch-{self.counter}"
        request_rows = [
            json.loads(line)
            for line in self.files.values[input_file_id].decode().splitlines()
            if line.strip()
        ]
        output_rows = []
        for request in request_rows:
            body = request["body"]
            if len(body["tools"]) == 1:
                tool_call = {
                    "id": f"insert-{request['custom_id']}",
                    "type": "function",
                    "function": {
                        "name": "SemanticMemory",
                        "arguments": json.dumps(
                            {
                                "content": "The user prefers aisle seats.",
                                "category": "explicit_preference",
                                "domain": "flight",
                                "slot": "seat",
                                "value": "aisle",
                                "context": None,
                            }
                        ),
                    },
                }
            else:
                system_content = body["messages"][0]["content"]
                stable_id = system_content.split("<instance id=", 1)[1].split(
                    " ", 1
                )[0]
                tool_call = {
                    "id": f"patch-{request['custom_id']}",
                    "type": "function",
                    "function": {
                        "name": "PatchDoc",
                        "arguments": json.dumps(
                            {
                                "json_doc_id": stable_id,
                                "planned_edits": "No changes required.",
                                "patches": [],
                            }
                        ),
                    },
                }
            output_rows.append(
                {
                    "custom_id": request["custom_id"],
                    "response": {
                        "status_code": 200,
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "content": "",
                                        "tool_calls": [tool_call],
                                    }
                                }
                            ],
                            "usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 20,
                                "total_tokens": 120,
                                "completion_tokens_details": {
                                    "reasoning_tokens": 2
                                },
                            },
                        },
                    },
                    "error": None,
                }
            )
        output_file_id = f"output-{self.counter}"
        self.files.values[output_file_id] = (
            "".join(json.dumps(row) + "\n" for row in output_rows).encode()
        )
        request_counts = SimpleNamespace(
            model_dump=lambda: {
                "total": len(request_rows),
                "completed": len(request_rows),
                "failed": 0,
            }
        )
        batch = SimpleNamespace(
            id=batch_id,
            status="completed",
            output_file_id=output_file_id,
            error_file_id=None,
            request_counts=request_counts,
        )
        self.values[batch_id] = batch
        return batch

    def retrieve(self, batch_id: str):
        return self.values[batch_id]


class FakeChatResponse:
    def __init__(self, body: dict) -> None:
        self.body = body

    def model_dump(self, mode: str = "python") -> dict:
        return self.body


class FakeChatCompletions:
    def create(self, **body):
        if len(body["tools"]) == 1:
            tool_call = {
                "id": "direct-insert",
                "type": "function",
                "function": {
                    "name": "SemanticMemory",
                    "arguments": json.dumps(
                        {
                            "content": "The user prefers aisle seats.",
                            "category": "explicit_preference",
                            "domain": "flight",
                            "slot": "seat",
                            "value": "aisle",
                            "context": None,
                        }
                    ),
                },
            }
        else:
            system_content = body["messages"][0]["content"]
            stable_id = system_content.split("<instance id=", 1)[1].split(
                " ", 1
            )[0]
            tool_call = {
                "id": "direct-patch",
                "type": "function",
                "function": {
                    "name": "PatchDoc",
                    "arguments": json.dumps(
                        {
                            "json_doc_id": stable_id,
                            "planned_edits": "No changes required.",
                            "patches": [],
                        }
                    ),
                },
            }
        return FakeChatResponse(
            {
                "id": "chatcmpl-direct",
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "tool_calls": [tool_call],
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "total_tokens": 120,
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )


class FakeOpenAI:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.embeddings = FakeEmbeddings()
        self.batches = FakeBatches(self.files)
        self.chat = SimpleNamespace(completions=FakeChatCompletions())


class LangMemBatchLifecycleTest(unittest.TestCase):
    def test_two_round_lifecycle_finalizes_replayable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "dataset.jsonl"
            rows = [
                {
                    "example_id": f"user_{index}",
                    "sessions": [
                        {
                            "dialogue_id": f"d{index}-1",
                            "dialogue": [
                                {
                                    "role": "user",
                                    "message": "I prefer aisle seats.",
                                }
                            ],
                            "api_call": [],
                        },
                        {
                            "dialogue_id": f"d{index}-2",
                            "dialogue": [
                                {
                                    "role": "user",
                                    "message": "Keep the same seat preference.",
                                }
                            ],
                            "api_call": [],
                        },
                    ],
                }
                for index in range(2)
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                input_path=str(input_path),
                output_dir=str(root / "output"),
                model="gpt-5-mini",
                reasoning_effort="minimal",
                embedding_model="text-embedding-3-small",
                api_key="test",
                poll_seconds=5,
                embedding_chunk_size=128,
                max_completion_tokens=2048,
                max_examples=None,
                force_init=False,
            )
            fake_client = FakeOpenAI()
            with patch(
                "methods.langmem.build_memory_batch.client_for",
                return_value=fake_client,
            ):
                state = initialize(args)
                run_loop(args, state)

            output_paths = paths_for(args)
            final_state = json.loads(
                output_paths["state"].read_text(encoding="utf-8")
            )
            artifacts = [
                json.loads(line)
                for line in output_paths["artifact"]
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            accumulation = [
                json.loads(line)
                for line in output_paths["accumulation"]
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(final_state["phase"], "complete")
            self.assertEqual(len(final_state["batch_history"]), 2)
            self.assertEqual(len(artifacts), 2)
            self.assertEqual(len(accumulation), 4)
            self.assertEqual(len(artifacts[0]["memory_items"]), 1)
            self.assertGreater(
                artifacts[0]["token_counts"]["total_tokens"], 0
            )
            self.assertEqual(
                accumulation[0]["memory_operations"][0]["operation"],
                "insert",
            )
            self.assertEqual(
                accumulation[1]["cumulative_construction_token_usage"][
                    "total_tokens"
                ],
                artifacts[0]["token_counts"]["total_tokens"],
            )

    def test_direct_lifecycle_uses_no_batch_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_path = root / "dataset.jsonl"
            rows = [
                {
                    "example_id": "user_0",
                    "sessions": [
                        {
                            "dialogue_id": "d0-1",
                            "dialogue": [
                                {
                                    "role": "user",
                                    "message": "I prefer aisle seats.",
                                }
                            ],
                            "api_call": [],
                        },
                        {
                            "dialogue_id": "d0-2",
                            "dialogue": [
                                {
                                    "role": "user",
                                    "message": "Keep that preference.",
                                }
                            ],
                            "api_call": [],
                        },
                    ],
                }
            ]
            input_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                input_path=str(input_path),
                output_dir=str(root / "output"),
                model="gpt-5-mini",
                reasoning_effort="minimal",
                embedding_model="text-embedding-3-small",
                api_key="test",
                poll_seconds=5,
                embedding_chunk_size=128,
                max_completion_tokens=2048,
                max_examples=None,
                force_init=False,
                direct_concurrency=2,
                direct_max_attempts=2,
                direct_retry_seconds=0.01,
            )
            fake_client = FakeOpenAI()
            with patch(
                "methods.langmem.build_memory_batch.client_for",
                return_value=fake_client,
            ):
                state = initialize(args)
                run_direct_loop(args, state)

            output_paths = paths_for(args)
            final_state = json.loads(
                output_paths["state"].read_text(encoding="utf-8")
            )
            artifacts = [
                json.loads(line)
                for line in output_paths["artifact"]
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertEqual(final_state["phase"], "complete")
            self.assertEqual(final_state["batch_history"], [])
            self.assertEqual(len(final_state["direct_history"]), 2)
            self.assertEqual(
                artifacts[0]["memory_mode"], "langmem_openai_direct"
            )
            self.assertTrue(
                all(
                    session["execution_mode"] == "direct"
                    for session in artifacts[0]["session_exports"]
                )
            )


if __name__ == "__main__":
    unittest.main()
