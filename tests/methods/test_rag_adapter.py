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
    validate_prepared_record,
    validate_prepared_records,
)
from exp7.methods.rag import RAGAdapter, RAGRuntime, RetrievedContext


def source_example():
    return {
        "example_id": "source-1",
        "sessions": [
            {
                "api_call": ['GetHotels(city="Seoul")'],
                "dialogue": [
                    {"role": "User", "message": "I usually stay in Seoul."},
                    {"role": "Assistant", "message": "I will remember that."},
                ],
            }
        ],
    }


def prepared(turn: str, suffix: str):
    return {
        "dataset_id": "mix600-v1",
        "difficulty": "easy",
        "ground_truth": [f'GetHotels(city="Seoul", mode="{suffix}")'],
        "instance_id": f"mix600-v1:{turn}:easy:source-1:{suffix}:0",
        "query": f"canonical-{turn}-query",
        "source_example": source_example(),
        "source_example_id": "source-1",
        "turn": turn,
    }


class FakeIndexBackend:
    def __init__(self):
        self.build_documents = []
        self.retrieval_requests = []
        self.raise_on_retrieve = False
        self.cross_identity = False

    def build_index(self, documents, state_dir):
        self.build_documents = [document.as_mapping() for document in documents]
        (state_dir / "index.json").write_text(
            json.dumps(self.build_documents, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return {"backend": "deterministic-fake", "embedding": "fake-v1"}

    def retrieve(self, state_dir, request):
        self.retrieval_requests.append(request)
        if self.raise_on_retrieve:
            raise RuntimeError("retrieval failed")
        documents = json.loads((state_dir / "index.json").read_text(encoding="utf-8"))
        selected = next(
            document
            for document in documents
            if document["source_example_id"] == request.source_example_id
            and document["user_id"] == request.user_id
        )
        source_id = "other-source" if self.cross_identity else request.source_example_id
        return [
            RetrievedContext(
                document_id=selected["document_id"],
                source_example_id=source_id,
                user_id=request.user_id,
                text=f"retrieved::{request.query}::{selected['text']}",
                score=1.0,
            )
        ]


class ReplacingIndexBackend(FakeIndexBackend):
    def __init__(self):
        super().__init__()
        self.attack_path = None

    def retrieve(self, state_dir, request):
        original = self.attack_path.with_name(self.attack_path.name + "-original")
        self.attack_path.rename(original)
        self.attack_path.mkdir()
        hostile = [
            {
                "document_id": "hostile:document",
                "kind": "dialogue",
                "metadata": {},
                "source_example_id": request.source_example_id,
                "text": "attacker-controlled state",
                "user_id": request.user_id,
            }
        ]
        (self.attack_path / "index.json").write_text(
            json.dumps(hostile, sort_keys=True),
            encoding="utf-8",
        )
        return super().retrieve(state_dir, request)


class RecordingGenerator:
    def __init__(self):
        self.calls = []

    async def __call__(self, record, request):
        self.calls.append((record, request))
        if f"retrieved::{record['query']}" not in request.retrieved_context:
            raise AssertionError("retrieved context was not used")
        return {
            "prediction": f"GeneratedFrom({request.retrieved[0].document_id})",
            "status": "ok",
        }


class RAGAdapterTests(unittest.TestCase):
    def context(self, root: Path) -> MethodRunContext:
        return MethodRunContext(
            run_dir=root,
            dataset_manifest_sha256="a" * 64,
            model_name="fake-model",
            tools_schema=({"type": "function"},),
        )

    def runtime(self, index=None, generator=None) -> RAGRuntime:
        return RAGRuntime(
            index_backend=index or FakeIndexBackend(),
            generator=generator or RecordingGenerator(),
            top_k=3,
        )

    def build(self, root: Path, runtime: RAGRuntime):
        records = validate_prepared_records(
            [prepared("singleturn", "single"), prepared("multiturn", "multi")]
        )
        adapter = RAGAdapter()
        state = asyncio.run(adapter.build(records, self.context(root), runtime))
        return adapter, records, state

    def test_build_deduplicates_source_and_binds_persistent_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = FakeIndexBackend()
            adapter, records, state = self.build(root, self.runtime(index=index))

            self.assertEqual(adapter.method_id, "rag")
            self.assertEqual(len(index.build_documents), 3)
            self.assertEqual(
                {document["source_example_id"] for document in index.build_documents},
                {"source-1"},
            )
            self.assertEqual(state.metadata["source_count"], 1)
            self.assertEqual(state.metadata["document_count"], 3)
            self.assertTrue(state.path.is_relative_to(root))
            manifest = json.loads(
                (state.path / "state_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["dataset_manifest_sha256"], "a" * 64)
            self.assertEqual(manifest["records_sha256"], state.records_sha256)
            self.assertEqual(manifest["instance_ids"], [r.instance_id for r in records])
            self.assertEqual(manifest["sources"]["source-1"]["document_count"], 3)
            self.assertEqual(set(manifest["backend_artifacts"]), {"index.json"})

    def test_single_and_multiturn_inference_preserve_identity_gt_and_use_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index, generator = FakeIndexBackend(), RecordingGenerator()
            runtime = self.runtime(index=index, generator=generator)
            adapter, records, state = self.build(root, runtime)
            outputs = [
                asyncio.run(adapter.infer(record, self.context(root), state, runtime))
                for record in records
            ]

            self.assertEqual([request.query for request in index.retrieval_requests], [
                "canonical-singleturn-query",
                "canonical-multiturn-query",
            ])
            self.assertEqual(len(generator.calls), 2)
            for record, output in zip(records, outputs):
                self.assertEqual(output["instance_id"], record.instance_id)
                self.assertEqual(output["ground_truth"], list(record.ground_truth))
                self.assertEqual(output["query"], record.query)
                self.assertEqual(output["method_id"], "rag")
                self.assertEqual(output["retrieval_query"], record.query)
                self.assertEqual(output["retrieval_status"], "complete")
                self.assertIn(f"retrieved::{record.query}", output["retrieved_context"][0]["text"])
                self.assertTrue(output["prediction"].startswith("GeneratedFrom("))
            self.assertTrue(all(request.source_example_id == "source-1" for request in index.retrieval_requests))
            self.assertTrue(all(request.user_id == "source-1" for request in index.retrieval_requests))
            self.assertTrue(all(request.top_k == 3 for request in index.retrieval_requests))

    def test_missing_or_failed_retrieval_never_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = FakeIndexBackend()
            runtime = self.runtime(index=index)
            adapter, records, state = self.build(root, runtime)
            with self.assertRaisesRegex(MethodContractError, "requires built index state"):
                asyncio.run(adapter.infer(records[0], self.context(root), None, runtime))
            index.raise_on_retrieve = True
            with self.assertRaisesRegex(RuntimeError, "retrieval failed"):
                asyncio.run(adapter.infer(records[0], self.context(root), state, runtime))

    def test_state_and_source_tampering_fail_closed(self) -> None:
        for target in ("artifact", "manifest", "source"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                runtime = self.runtime()
                adapter, records, state = self.build(root, runtime)
                record = records[0]
                if target == "artifact":
                    path = state.path / "index.json"
                    path.write_bytes(path.read_bytes() + b" ")
                elif target == "manifest":
                    path = state.path / "state_manifest.json"
                    path.write_bytes(path.read_bytes() + b" ")
                else:
                    payload = record.as_mapping()
                    payload["source_example"]["sessions"][0]["dialogue"][0][
                        "message"
                    ] = "tampered"
                    record = validate_prepared_record(payload)
                with self.assertRaisesRegex(MethodContractError, "tamper|differs|mismatch"):
                    asyncio.run(adapter.infer(record, self.context(root), state, runtime))

    def test_state_directory_replacement_cannot_redirect_backend_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index, generator = ReplacingIndexBackend(), RecordingGenerator()
            runtime = self.runtime(index=index, generator=generator)
            adapter, records, state = self.build(root, runtime)
            index.attack_path = state.path

            with self.assertRaisesRegex(MethodContractError, "changed while being used"):
                asyncio.run(adapter.infer(records[0], self.context(root), state, runtime))
            self.assertEqual(generator.calls, [])

    def test_state_manifest_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self.runtime()
            adapter, records, state = self.build(root, runtime)
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
                asyncio.run(adapter.infer(records[0], self.context(root), forged, runtime))

    def test_cross_identity_retrieval_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            index = FakeIndexBackend()
            runtime = self.runtime(index=index)
            adapter, records, state = self.build(root, runtime)
            index.cross_identity = True
            with self.assertRaisesRegex(MethodContractError, "cross-identity"):
                asyncio.run(adapter.infer(records[0], self.context(root), state, runtime))

    def test_state_path_escape_and_overwrite_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = validate_prepared_records([prepared("singleturn", "single")])
            runtime = self.runtime()
            with self.assertRaisesRegex(MethodContractError, "unsafe method state path"):
                asyncio.run(
                    RAGAdapter("../outside").build(records, self.context(root), runtime)
                )
            adapter = RAGAdapter()
            asyncio.run(adapter.build(records, self.context(root), runtime))
            with self.assertRaisesRegex(MethodContractError, "refusing to overwrite"):
                asyncio.run(adapter.build(records, self.context(root), runtime))

    def test_conflicting_duplicate_source_payload_fails_before_backend(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = prepared("singleturn", "single"), prepared("multiturn", "multi")
            second["source_example"]["sessions"][0]["dialogue"][0]["message"] = "conflict"
            index = FakeIndexBackend()
            with self.assertRaisesRegex(MethodContractError, "conflicting payloads"):
                asyncio.run(
                    RAGAdapter().build(
                        validate_prepared_records([first, second]),
                        self.context(root),
                        self.runtime(index=index),
                    )
                )
            self.assertEqual(index.build_documents, [])


if __name__ == "__main__":
    unittest.main()
