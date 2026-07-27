from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from methods.amem import build_memory_local as module


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "window" in lowered:
                vectors.append([1.0, 0.0])
            elif "aisle" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return vectors

    async def async_encode(self, texts):
        return self.encode(texts)


class FakeChatResponse:
    def __init__(self, payload: dict, total_tokens: int) -> None:
        self._payload = payload
        self._total_tokens = total_tokens

    def model_dump(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(self._payload),
                    }
                }
            ],
            "usage": {
                "prompt_tokens": self._total_tokens - 10,
                "completion_tokens": 10,
                "total_tokens": self._total_tokens,
            },
        }


class FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        schema = kwargs["response_format"]["json_schema"]["name"]
        if schema == "amem_note_metadata":
            content = kwargs["messages"][1]["content"]
            if "window" in content.lower():
                payload = {
                    "context": "Seat preference.",
                    "keywords": ["window"],
                    "tags": ["flight"],
                }
            else:
                payload = {
                    "context": "Earlier aisle choice.",
                    "keywords": ["aisle"],
                    "tags": ["flight"],
                }
            return FakeChatResponse(payload, 50)
        if schema == "amem_note_evolution":
            candidates = json.loads(kwargs["messages"][1]["content"])[
                "candidate_notes"
            ]
            payload = {
                "should_evolve": bool(candidates),
                "actions": ["strengthen"] if candidates else [],
                "suggested_connections": (
                    [candidates[0]["note_id"]] if candidates else []
                ),
                "tags_to_update": ["flight", "seat"] if candidates else [],
                "new_context_neighborhood": [
                    item["context"] for item in candidates
                ],
                "new_tags_neighborhood": [item["tags"] for item in candidates],
            }
            return FakeChatResponse(payload, 40)
        raise AssertionError(f"unexpected schema: {schema}")


class FakeAsyncOpenAI:
    instance: "FakeAsyncOpenAI | None" = None

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.chat = SimpleNamespace(completions=FakeCompletions())
        FakeAsyncOpenAI.instance = self


def args_for(root: Path, dataset: Path) -> SimpleNamespace:
    return SimpleNamespace(
        input_path=dataset,
        output_dir=root / "out",
        base_url="http://127.0.0.1:8000/v1",
        api_key=None,
        model="gpt-oss-20b",
        reasoning_effort="low",
        embedding_model="all-MiniLM-L6-v2",
        top_k=5,
        max_completion_tokens=1536,
        max_examples=None,
        max_sessions=None,
        start_example=0,
        end_example=None,
        concurrency=2,
        embedding_concurrency=2,
        request_timeout_seconds=123.0,
        timeout_seconds=None,
        retry_count=0,
        retry_base_sleep=0.0,
        resume=False,
        skip_preflight=True,
        response_format="json_schema",
        dry_run=False,
    )


class AmemLocalTest(unittest.IsolatedAsyncioTestCase):
    def test_parse_local_message_json_tolerates_wrappers_and_controls(self) -> None:
        body = {
            "choices": [
                {
                    "message": {
                        "content": (
                            "<|channel|>final<|message|>```json\n"
                            '{"context":"quiet\x0broom","keywords":[],"tags":[]}'
                            "\n```"
                        )
                    }
                }
            ]
        }

        parsed = module.parse_local_message_json(body)

        self.assertEqual(parsed["context"], "quiet\x0broom")

    async def test_process_user_preserves_causal_order_and_schema(self) -> None:
        completions = FakeCompletions()
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=completions)
        )
        args = args_for(Path("/tmp"), Path("/tmp/dataset.json"))
        result = await module.process_user(
            client=client,
            example={
                "example_id": "u1",
                "sessions": [
                    {
                        "dialogue_id": "d1",
                        "dialogue": [
                            {"role": "User", "message": "aisle seat"},
                            {"role": "User", "message": "window seat"},
                        ],
                    }
                ],
            },
            dataset_index=0,
            args=args,
            embedder=FakeEmbedder(),
        )

        self.assertEqual(result["method"], "amem")
        self.assertEqual(result["backend"], "openai_compatible_local")
        self.assertEqual(len(result["notes"]), 2)
        self.assertEqual(result["notes"][1]["links"], [result["notes"][0]["note_id"]])
        self.assertEqual(len(completions.calls), 4)
        self.assertEqual(
            [
                call["response_format"]["json_schema"]["name"]
                for call in completions.calls
            ],
            [
                "amem_note_metadata",
                "amem_note_evolution",
                "amem_note_metadata",
                "amem_note_evolution",
            ],
        )
        self.assertTrue(
            all(call["max_completion_tokens"] == 1536 for call in completions.calls)
        )
        self.assertTrue(all(call["reasoning_effort"] == "low" for call in completions.calls))

    async def test_run_ingestion_writes_resumeable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = root / "dataset.json"
            dataset.write_text(
                json.dumps(
                    [
                        {
                            "example_id": "u1",
                            "sessions": [
                                {
                                    "dialogue_id": "d1",
                                    "dialogue": [
                                        {
                                            "role": "User",
                                            "message": "window seat",
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = args_for(root, dataset)
            with (
                patch.object(module, "AsyncOpenAI", FakeAsyncOpenAI),
                patch.object(module, "make_embedder", lambda _model: FakeEmbedder()),
            ):
                summary = await module.run_ingestion(args)

            artifact = module.read_jsonl(root / "out" / "memory.jsonl")
            self.assertEqual(summary["status"], "COMPLETE")
            self.assertEqual(summary["example_count"], 1)
            self.assertEqual(len(artifact), 1)
            self.assertEqual(artifact[0]["example_id"], "u1")
            self.assertEqual(artifact[0]["memory_model"], "gpt-oss-20b")
            self.assertEqual(
                FakeAsyncOpenAI.instance.kwargs["timeout"],
                123.0,
            )


if __name__ == "__main__":
    unittest.main()
