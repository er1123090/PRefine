from __future__ import annotations

import asyncio
import copy
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
    MethodStateHandle,
    validate_prepared_record,
    validate_prepared_records,
)
from exp7.methods.preference_memory import (
    PreferenceMemoryAdapter,
    PreferenceMemoryRuntime,
    PreferenceRefinementResult,
)


def source_example(source_id: str = "source-1"):
    return {
        "example_id": source_id,
        "sessions": [
            {
                "api_call": ['GetHotels(city="Seoul")'],
                "dialogue": [
                    {"role": "User", "message": "I prefer Seoul hotels."},
                    {"role": "Assistant", "message": "Noted."},
                ],
            },
            {
                "api_call": ['GetHotels(star="4")', 'GetHotels(parking="True")'],
                "dialogue": [
                    {"role": "User", "message": "Four stars and parking are ideal."}
                ],
            },
        ],
    }


def prepared(turn: str, suffix: str):
    return {
        "dataset_id": "mix600-v1",
        "difficulty": "medium",
        "ground_truth": [f'GetHotels(city="Seoul", turn="{suffix}")'],
        "instance_id": f"mix600-v1:{turn}:medium:source-1:{suffix}:0",
        "query": f"canonical-{turn}-query",
        "source_example": source_example(),
        "source_example_id": "source-1",
        "turn": turn,
    }


class FakeRefiner:
    def __init__(self):
        self.requests = []

    async def __call__(self, request):
        self.requests.append(copy.deepcopy(request))
        return PreferenceRefinementResult(
            preference={
                "api_count": len(request.accumulated_api_calls),
                "session_count": request.session_index,
            },
            evidence=[
                {
                    "dialogue": request.session_dialogue,
                    "session_index": request.session_index,
                }
            ],
            refinement_process=(
                {"draft": request.session_index, "is_valid": True, "step": 1},
            ),
            metadata={"refiner": "fake-v1"},
        )


class RecordingGenerator:
    def __init__(self):
        self.calls = []

    async def __call__(self, record, request):
        self.calls.append((record, request))
        final = request.memory["final_implicit_preference"]
        if len(request.memory_evidence) != 2:
            raise AssertionError("memory evolution evidence was not provided")
        return {
            "prediction": (
                f"GeneratedFromMemory({record['query']},"
                f"sessions={final['session_count']},apis={final['api_count']})"
            ),
            "status": "ok",
        }


class PreferenceMemoryAdapterTests(unittest.TestCase):
    def context(self, root: Path) -> MethodRunContext:
        return MethodRunContext(
            run_dir=root,
            dataset_manifest_sha256="b" * 64,
            model_name="fake-model",
            tools_schema=({"type": "function"},),
        )

    def runtime(self, refiner=None, generator=None) -> PreferenceMemoryRuntime:
        return PreferenceMemoryRuntime(
            refiner=refiner or FakeRefiner(),
            generator=generator or RecordingGenerator(),
        )

    def build(self, root: Path, runtime: PreferenceMemoryRuntime):
        records = validate_prepared_records(
            [prepared("singleturn", "single"), prepared("multiturn", "multi")]
        )
        adapter = PreferenceMemoryAdapter()
        state = asyncio.run(adapter.build(records, self.context(root), runtime))
        return adapter, records, state

    def read_memory(self, state):
        lines = (state.path / "memory.jsonl").read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line]

    def test_build_evolves_once_per_source_session_and_preserves_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            refiner = FakeRefiner()
            adapter, records, state = self.build(root, self.runtime(refiner=refiner))
            memories = self.read_memory(state)

            self.assertEqual(adapter.method_id, "preference_memory")
            self.assertEqual(len(refiner.requests), 2)
            self.assertEqual(refiner.requests[0].previous_preference, {})
            self.assertEqual(
                refiner.requests[1].previous_preference,
                {"api_count": 1, "session_count": 1},
            )
            self.assertEqual(len(memories), 1)
            memory = memories[0]
            self.assertEqual(memory["source_example_id"], "source-1")
            self.assertEqual(memory["total_sessions_processed"], 2)
            self.assertEqual(
                memory["final_accumulated_api_calls"],
                [
                    '[Session 1] GetHotels(city="Seoul")',
                    '[Session 2] GetHotels(star="4")',
                    '[Session 2] GetHotels(parking="True")',
                ],
            )
            self.assertEqual(memory["final_implicit_preference"]["api_count"], 3)
            self.assertEqual(len(memory["preference_evolution_history"]), 2)
            self.assertTrue(state.path.is_relative_to(root))
            self.assertEqual(state.metadata["memory_count"], 1)
            manifest = json.loads(
                (state.path / "state_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["dataset_manifest_sha256"], "b" * 64)
            self.assertEqual(manifest["records_sha256"], state.records_sha256)
            self.assertEqual(manifest["instance_ids"], [r.instance_id for r in records])
            self.assertEqual(set(manifest["sources"]), {"source-1"})
            self.assertEqual(
                {path.name for path in state.path.iterdir()},
                {"memory.jsonl", "state_manifest.json"},
            )
            self.assertFalse(
                any(path.name.startswith(".preference-memory-") for path in state.path.parent.iterdir())
            )

    def test_single_and_multiturn_inference_preserve_identity_gt_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generator = RecordingGenerator()
            runtime = self.runtime(generator=generator)
            adapter, records, state = self.build(root, runtime)
            outputs = [
                asyncio.run(adapter.infer(record, self.context(root), state, runtime))
                for record in records
            ]

            self.assertEqual(len(generator.calls), 2)
            for record, output in zip(records, outputs):
                self.assertEqual(output["instance_id"], record.instance_id)
                self.assertEqual(output["ground_truth"], list(record.ground_truth))
                self.assertEqual(output["query"], record.query)
                self.assertEqual(output["method_id"], "preference_memory")
                self.assertEqual(output["memory_source_example_id"], "source-1")
                self.assertEqual(output["memory_status"], "loaded")
                self.assertEqual(
                    output["preference_memory"]["final_implicit_preference"],
                    {"api_count": 3, "session_count": 2},
                )
                self.assertEqual(len(output["memory_evidence"]), 2)
                self.assertIn(record.query, output["prediction"])

    def test_missing_or_cross_source_memory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self.runtime()
            adapter, records, state = self.build(root, runtime)
            with self.assertRaisesRegex(MethodContractError, "requires built state"):
                asyncio.run(adapter.infer(records[0], self.context(root), None, runtime))
            payload = records[0].as_mapping()
            payload["source_example_id"] = "source-2"
            payload["source_example"] = source_example("source-2")
            cross_source = validate_prepared_record(payload)
            with self.assertRaisesRegex(MethodContractError, "missing preference memory"):
                asyncio.run(
                    adapter.infer(cross_source, self.context(root), state, runtime)
                )

    def test_state_manifest_memory_and_source_tampering_fail_closed(self) -> None:
        for target in ("memory", "manifest", "source"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                runtime = self.runtime()
                adapter, records, state = self.build(root, runtime)
                record = records[0]
                if target == "memory":
                    path = state.path / "memory.jsonl"
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
                asyncio.run(
                    adapter.infer(records[0], self.context(root), forged, runtime)
                )

    def test_unexpected_state_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = self.runtime()
            adapter, records, state = self.build(root, runtime)
            (state.path / "unexpected.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(MethodContractError, "artifacts.*tampered"):
                asyncio.run(adapter.infer(records[0], self.context(root), state, runtime))

    def test_duplicate_and_cross_source_jsonl_rows_are_rejected(self) -> None:
        for corruption in ("duplicate", "cross-source"):
            with self.subTest(corruption=corruption), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                runtime = self.runtime()
                adapter, records, state = self.build(root, runtime)
                memory_path = state.path / "memory.jsonl"
                memory = json.loads(memory_path.read_text(encoding="utf-8"))
                if corruption == "duplicate":
                    rows = [memory, memory]
                else:
                    changed = copy.deepcopy(memory)
                    changed["example_id"] = "other-source"
                    rows = [changed]
                memory_payload = b"".join(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
                    + b"\n"
                    for row in rows
                )
                memory_path.write_bytes(memory_payload)
                manifest_path = state.path / "state_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["memory"].update(
                    {
                        "bytes": len(memory_payload),
                        "count": len(rows),
                        "sha256": hashlib.sha256(memory_payload).hexdigest(),
                    }
                )
                if corruption == "cross-source":
                    canonical = json.dumps(
                        rows[0], ensure_ascii=False, sort_keys=True, separators=(",", ":")
                    ).encode()
                    manifest["sources"]["source-1"]["memory_sha256"] = hashlib.sha256(
                        canonical
                    ).hexdigest()
                manifest_payload = (
                    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
                ).encode()
                manifest_path.write_bytes(manifest_payload)
                forged = MethodStateHandle(
                    method_id=state.method_id,
                    path=state.path,
                    dataset_manifest_sha256=state.dataset_manifest_sha256,
                    records_sha256=state.records_sha256,
                    metadata={
                        **state.metadata,
                        "state_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                    },
                )
                expected = "duplicate" if corruption == "duplicate" else "cross-source"
                with self.assertRaisesRegex(MethodContractError, expected):
                    asyncio.run(
                        adapter.infer(records[0], self.context(root), forged, runtime)
                    )

    def test_state_path_escape_and_overwrite_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = validate_prepared_records([prepared("singleturn", "single")])
            runtime = self.runtime()
            with self.assertRaisesRegex(MethodContractError, "unsafe method state path"):
                asyncio.run(
                    PreferenceMemoryAdapter("../outside").build(
                        records, self.context(root), runtime
                    )
                )
            adapter = PreferenceMemoryAdapter()
            asyncio.run(adapter.build(records, self.context(root), runtime))
            with self.assertRaisesRegex(MethodContractError, "refusing to overwrite"):
                asyncio.run(adapter.build(records, self.context(root), runtime))

    def test_conflicting_duplicate_source_fails_before_refinement(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = prepared("singleturn", "single"), prepared("multiturn", "multi")
            second["source_example"]["sessions"][0]["dialogue"][0]["message"] = "conflict"
            refiner = FakeRefiner()
            with self.assertRaisesRegex(MethodContractError, "conflicting payloads"):
                asyncio.run(
                    PreferenceMemoryAdapter().build(
                        validate_prepared_records([first, second]),
                        self.context(root),
                        self.runtime(refiner=refiner),
                    )
                )
            self.assertEqual(refiner.requests, [])


if __name__ == "__main__":
    unittest.main()
