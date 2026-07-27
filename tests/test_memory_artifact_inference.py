from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from methods.amem.common import sha256_file
from scripts.run_memory_artifact_inference import (
    batch_request,
    load_or_create_embedding_cache,
    local_command,
    output_paths,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def local_args(prepared_dir: Path, output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        prepared_dir=str(prepared_dir),
        output_dir=str(output_dir),
        method="rag",
        expected_count=1,
        resume=False,
    )


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, *, model, input, encoding_format):
        self.calls += 1
        data = [
            SimpleNamespace(index=index, embedding=[float(index + 1), 1.0])
            for index, _ in enumerate(input)
        ]
        usage = SimpleNamespace(
            prompt_tokens=len(input),
            total_tokens=len(input),
        )
        return SimpleNamespace(data=data, usage=usage)


class FailingEmbeddings:
    def create(self, **kwargs):
        raise AssertionError("valid cache should avoid a second embedding call")


class MemoryArtifactInferenceTest(unittest.TestCase):
    def test_batch_request_uses_gpt5_mini_minimal(self) -> None:
        request = batch_request(
            {"sample_id": "rag-00000", "prompt": "same prompt"},
            model="gpt-5-mini",
            reasoning_effort="minimal",
            max_completion_tokens=2048,
        )

        self.assertEqual(request["url"], "/v1/chat/completions")
        self.assertEqual(request["body"]["model"], "gpt-5-mini")
        self.assertEqual(request["body"]["reasoning_effort"], "minimal")
        self.assertEqual(request["body"]["max_completion_tokens"], 2048)
        self.assertEqual(
            request["body"]["messages"],
            [{"role": "user", "content": "same prompt"}],
        )

    def test_embedding_cache_is_content_addressed_and_reused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = output_paths(Path(temporary))
            first_client = SimpleNamespace(embeddings=FakeEmbeddings())
            documents, queries, meta = load_or_create_embedding_cache(
                paths=paths,
                labels=["user-a\0memory-a", "user-a\0memory-b"],
                texts=["memory a", "memory b"],
                query_texts=["query"],
                model="text-embedding-3-small",
                client=first_client,
                batch_size=8,
                retry_count=0,
                source_sha256="source-hash",
            )
            cached_documents, cached_queries, cached_meta = (
                load_or_create_embedding_cache(
                    paths=paths,
                    labels=["user-a\0memory-a", "user-a\0memory-b"],
                    texts=["memory a", "memory b"],
                    query_texts=["query"],
                    model="text-embedding-3-small",
                    client=SimpleNamespace(embeddings=FailingEmbeddings()),
                    batch_size=8,
                    retry_count=0,
                    source_sha256="source-hash",
                )
            )

            np.testing.assert_array_equal(documents, cached_documents)
            np.testing.assert_array_equal(queries, cached_queries)
            self.assertEqual(meta["cache_key"], cached_meta["cache_key"])
            self.assertEqual(first_client.embeddings.calls, 2)

    def test_local_run_rejects_manifest_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            output = root / "output"
            paths = output_paths(prepared)
            write_jsonl(paths["manifest"], [{"sample_id": "rag-00000"}])
            paths["summary"].write_text(
                json.dumps(
                    {
                        "method": "rag",
                        "manifest_sha256": "not-the-current-hash",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "non-comparable"):
                asyncio.run(local_command(local_args(prepared, output)))

    def test_local_run_requires_explicit_resume_for_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared = root / "prepared"
            output = root / "output"
            prepared_paths = output_paths(prepared)
            output_run_paths = output_paths(output)
            manifest = [{"sample_id": "rag-00000"}]
            write_jsonl(prepared_paths["manifest"], manifest)
            prepared_paths["summary"].write_text(
                json.dumps(
                    {
                        "method": "rag",
                        "manifest_sha256": sha256_file(
                            prepared_paths["manifest"]
                        ),
                    }
                ),
                encoding="utf-8",
            )
            output_run_paths["directory"].mkdir(parents=True)
            output_run_paths["checkpoint"].touch()

            with self.assertRaisesRegex(FileExistsError, "--resume"):
                asyncio.run(local_command(local_args(prepared, output)))


if __name__ == "__main__":
    unittest.main()
