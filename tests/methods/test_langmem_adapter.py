from __future__ import annotations

import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exp7.methods.contracts import MethodRunContext, validate_prepared_record
from exp7.methods.langmem import (
    LangMemAdapter,
    LangMemAdapterError,
    LangMemBackend,
)


SOURCE_EXAMPLE = {
    "example_id": "source-1",
    "sessions": [
        {
            "api_calls": ['GetHotels(star="4")'],
            "dialogue": [
                {"role": "user", "content": "I usually choose four-star hotels."},
                {"role": "assistant", "content": "Understood."},
            ],
        }
    ],
}


def prepared(
    *,
    turn: str,
    instance_id: str,
    query: str,
    ground_truth: str,
):
    return {
        "dataset_id": "mix600-v1",
        "difficulty": "easy",
        "ground_truth": [ground_truth],
        "instance_id": instance_id,
        "query": query,
        "source_example": SOURCE_EXAMPLE,
        "source_example_id": "source-1",
        "turn": turn,
    }


class FakeEmbedder:
    def __init__(self) -> None:
        self.documents = []
        self.queries = []

    def embed_documents(self, texts):
        self.documents.append(tuple(texts))
        return [[float(index + 1), float(len(text))] for index, text in enumerate(texts)]

    async def embed_query(self, text):
        self.queries.append(text)
        return [float(len(text)), 1.0]


class FakeStore:
    def __init__(self) -> None:
        self.build_calls = []
        self.search_calls = []

    async def build(self, *, items, embeddings):
        self.build_calls.append((tuple(items), tuple(embeddings)))
        return {
            "memories": [
                {
                    "category": "explicit_preference",
                    "content": "The user prefers four-star hotels.",
                    "source_example_id": items[0].source_example_id,
                }
            ],
            "source_ids": [item.source_example_id for item in items],
        }

    def search(self, snapshot, *, query, query_embedding, limit):
        self.search_calls.append((query, tuple(query_embedding), limit))
        return snapshot["memories"][:limit]


class FakeModel:
    def __init__(self) -> None:
        self.requests = []

    async def generate(self, request):
        self.requests.append(request)
        return {
            "prediction": request.record["ground_truth"][0],
            "provider_turn": request.turn,
            "status": "ok",
        }


class LangMemAdapterTests(unittest.TestCase):
    def context(
        self,
        root: Path,
        *,
        manifest_sha256: str = "a" * 64,
    ) -> MethodRunContext:
        return MethodRunContext(
            run_dir=root,
            dataset_manifest_sha256=manifest_sha256,
            model_name="fake-model",
            tools_schema=({"type": "function"},),
            reasoning_effort="low",
        )

    def records(self):
        return (
            validate_prepared_record(
                prepared(
                    turn="singleturn",
                    instance_id="single",
                    query="Find my usual hotel.",
                    ground_truth='GetHotels(star="4")',
                )
            ),
            validate_prepared_record(
                prepared(
                    turn="multiturn",
                    instance_id="multi",
                    query="User: Find a hotel.\nAssistant: Which kind?\nUser: My usual.",
                    ground_truth='GetHotels(star="4")',
                )
            ),
        )

    def backend(self):
        embedder = FakeEmbedder()
        store = FakeStore()
        model = FakeModel()
        return LangMemBackend(store=store, embedder=embedder, model=model)

    def build_state(self, root: Path):
        adapter = LangMemAdapter(top_k=3)
        backend = self.backend()
        context = self.context(root)
        records = self.records()
        state = asyncio.run(adapter.build(records, context, backend))
        return adapter, backend, context, records, state

    def test_build_deduplicates_source_and_writes_bound_state_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            adapter, backend, context, records, state = self.build_state(root)

            self.assertEqual(state.method_id, "langmem")
            self.assertEqual(state.path, root / "method_state/langmem")
            self.assertEqual(state.dataset_manifest_sha256, context.dataset_manifest_sha256)
            self.assertEqual(state.metadata["instance_count"], 2)
            self.assertEqual(state.metadata["source_count"], 1)
            items, embeddings = backend.store.build_calls[0]
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0].source_example_id, "source-1")
            self.assertEqual(items[0].instance_ids, ("single", "multi"))
            self.assertEqual(len(embeddings), 1)
            self.assertEqual(len(backend.embedder.documents), 1)

            manifest = json.loads(
                (state.path / "state_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["method_id"], adapter.method_id)
            self.assertEqual(
                manifest["dataset_manifest_sha256"],
                context.dataset_manifest_sha256,
            )
            self.assertEqual(manifest["instance_ids"], ["single", "multi"])
            self.assertEqual(
                set(manifest["record_sha256_by_instance_id"]),
                {"single", "multi"},
            )
            self.assertEqual(len(manifest["prepared_records_sha256"]), 64)
            self.assertEqual(len(manifest["source_records_sha256"]), 64)
            self.assertEqual(manifest["contract_records_sha256"], state.records_sha256)
            self.assertEqual(
                state.metadata["state_manifest_sha256"],
                __import__("hashlib").sha256(
                    (state.path / "state_manifest.json").read_bytes()
                ).hexdigest(),
            )

    def test_single_and_multiturn_inference_preserve_identity_query_gt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter, backend, context, records, state = self.build_state(
                Path(temporary)
            )
            outputs = [
                asyncio.run(adapter.infer(record, context, state, backend))
                for record in records
            ]

            self.assertEqual([item["instance_id"] for item in outputs], ["single", "multi"])
            self.assertEqual([item["turn"] for item in outputs], ["singleturn", "multiturn"])
            self.assertEqual(
                [item["query"] for item in outputs],
                [record.query for record in records],
            )
            self.assertEqual(
                [item["ground_truth"] for item in outputs],
                [list(record.ground_truth) for record in records],
            )
            self.assertTrue(
                all(item["method_id"] == "langmem" for item in outputs)
            )
            self.assertTrue(
                all(
                    item["retrieved_memories"][0]["content"]
                    == "The user prefers four-star hotels."
                    for item in outputs
                )
            )
            self.assertEqual(
                [request.turn for request in backend.model.requests],
                ["singleturn", "multiturn"],
            )
            self.assertEqual(
                [call[0] for call in backend.store.search_calls],
                [record.query for record in records],
            )

    def test_state_paths_cannot_escape_or_traverse_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = self.context(root)
            records = self.records()
            backend = self.backend()
            with self.assertRaisesRegex(
                LangMemAdapterError,
                "unsafe method state path",
            ):
                asyncio.run(
                    LangMemAdapter(relative_state_path="../outside").build(
                        records,
                        context,
                        backend,
                    )
                )

        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            (root / "linked").symlink_to(Path(outside), target_is_directory=True)
            with self.assertRaisesRegex(LangMemAdapterError, "symlink"):
                asyncio.run(
                    LangMemAdapter(relative_state_path="linked/langmem").build(
                        self.records(),
                        self.context(root),
                        self.backend(),
                    )
                )

    def test_missing_and_tampered_state_fail_closed(self) -> None:
        for mutation, expected in (
            ("missing_snapshot", "snapshot"),
            ("tampered_snapshot", "snapshot SHA-256"),
            ("tampered_manifest", "manifest SHA-256"),
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                adapter, backend, context, records, state = self.build_state(
                    Path(temporary)
                )
                if mutation == "missing_snapshot":
                    (state.path / "snapshot.json").unlink()
                elif mutation == "tampered_snapshot":
                    (state.path / "snapshot.json").write_text(
                        "{}",
                        encoding="utf-8",
                    )
                else:
                    manifest_path = state.path / "state_manifest.json"
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    manifest["dataset_manifest_sha256"] = "b" * 64
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                with self.assertRaisesRegex(LangMemAdapterError, expected):
                    asyncio.run(
                        adapter.infer(records[0], context, state, backend)
                    )

    def test_state_manifest_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            adapter, backend, context, records, state = self.build_state(
                Path(temporary)
            )
            path = state.path / "state_manifest.json"
            original = path.read_bytes()
            payload = b'{"format_version":1,' + original[1:]
            path.write_bytes(payload)
            forged = replace(
                state,
                metadata={
                    **state.metadata,
                    "state_manifest_sha256": hashlib.sha256(payload).hexdigest(),
                },
            )
            with self.assertRaisesRegex(LangMemAdapterError, "duplicate object key"):
                asyncio.run(adapter.infer(records[0], context, forged, backend))

    def test_state_is_bound_to_context_record_content_and_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, tempfile.TemporaryDirectory() as outside:
            root = Path(temporary)
            adapter, backend, context, records, state = self.build_state(root)
            wrong_context = self.context(root, manifest_sha256="b" * 64)
            with self.assertRaisesRegex(
                LangMemAdapterError,
                "different dataset manifest",
            ):
                asyncio.run(
                    adapter.infer(records[0], wrong_context, state, backend)
                )

            changed_payload = records[0].as_mapping()
            changed_payload["query"] = "A different query."
            changed_record = validate_prepared_record(changed_payload)
            with self.assertRaisesRegex(
                LangMemAdapterError,
                "content differs",
            ):
                asyncio.run(
                    adapter.infer(changed_record, context, state, backend)
                )

            escaped_state = replace(state, path=Path(outside))
            with self.assertRaisesRegex(LangMemAdapterError, "escapes run_dir"):
                asyncio.run(
                    adapter.infer(records[0], context, escaped_state, backend)
                )

    def test_import_has_no_optional_dependency_or_network_requirement(self) -> None:
        script = (
            "import sys;"
            f"sys.path.insert(0, {str(ROOT / 'src')!r});"
            "import exp7.methods.langmem.adapter;"
            "forbidden={'langmem','openai','langchain','langchain_core'};"
            "loaded={name.split('.')[0] for name in sys.modules};"
            "assert not (forbidden & loaded), forbidden & loaded"
        )
        result = subprocess.run(
            [sys.executable, "-S", "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
