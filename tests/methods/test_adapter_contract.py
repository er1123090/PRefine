from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


from exp7.methods import (
    MethodContractError,
    MethodRegistry,
    MethodRegistryError,
    MethodRunContext,
    bind_method_state,
    builtin_method_registry,
    canonical_prediction,
    method_state_path,
    validate_prepared_record,
    validate_prepared_records,
)


def prepared(
    instance_id: str = "mix600-v1:singleturn:easy:x:hash:0",
):
    return {
        "dataset_id": "mix600-v1",
        "difficulty": "easy",
        "ground_truth": ['GetHotels(star="4")'],
        "instance_id": instance_id,
        "query": "Find a hotel.",
        "source_example": {"example_id": "example", "sessions": []},
        "source_example_id": "example",
        "turn": "singleturn",
    }


class LocalStatefulAdapter:
    method_id = "rag"

    def build(self, records, context, backend):
        relative = backend.build(records)
        path = method_state_path(context, relative)
        path.mkdir(parents=True)
        return bind_method_state(
            method_id=self.method_id,
            context=context,
            relative_path=relative,
            records=records,
            metadata={"backend": "local"},
        )

    async def infer(self, record, context, state, backend):
        del context
        return canonical_prediction(
            record,
            method_id=self.method_id,
            prediction=await backend.infer(record.as_mapping(), state),
        )


class LocalBackend:
    def __init__(self):
        self.seen = []

    def build(self, records):
        self.seen.extend(record.instance_id for record in records)
        return "method_state/rag"

    async def infer(self, record, state):
        self.seen.append(
            (record["instance_id"], state.dataset_manifest_sha256)
        )
        return {
            "prediction": 'GetHotels(star="4")',
            "status": "ok",
        }


class AdapterContractTests(unittest.TestCase):
    def context(
        self,
        root: Path,
        *,
        model_name: str = "fake-model",
    ) -> MethodRunContext:
        return MethodRunContext(
            run_dir=root,
            dataset_manifest_sha256="a" * 64,
            model_name=model_name,
            tools_schema=({"type": "function"},),
        )

    def test_prepared_record_identity_query_and_gt_are_preserved(self) -> None:
        source = prepared()
        record = validate_prepared_record(source)
        source["ground_truth"].append("MUTATED()")
        self.assertEqual(record.instance_id, prepared()["instance_id"])
        self.assertEqual(record.source_example_id, "example")
        self.assertEqual(record.query, "Find a hotel.")
        self.assertEqual(record.ground_truth, ('GetHotels(star="4")',))
        self.assertEqual(
            record.as_mapping()["ground_truth"],
            ['GetHotels(star="4")'],
        )

    def test_invalid_records_and_duplicate_ids_fail_closed(self) -> None:
        invalid = prepared()
        invalid["ground_truth"] = "GetHotels()"
        with self.assertRaisesRegex(MethodContractError, "ground_truth"):
            validate_prepared_record(invalid)
        with self.assertRaisesRegex(
            MethodContractError,
            "duplicate instance_id",
        ):
            validate_prepared_records([prepared(), prepared()])

    def test_state_handle_is_manifest_and_record_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = self.context(root)
            records = validate_prepared_records([prepared()])
            backend = LocalBackend()
            adapter = LocalStatefulAdapter()
            state = adapter.build(records, context, backend)
            output = asyncio.run(
                adapter.infer(records[0], context, state, backend)
            )
            self.assertEqual(state.method_id, "rag")
            self.assertEqual(
                state.dataset_manifest_sha256,
                "a" * 64,
            )
            self.assertEqual(len(state.records_sha256), 64)
            self.assertEqual(
                output["instance_id"],
                records[0].instance_id,
            )
            self.assertEqual(
                output["ground_truth"],
                list(records[0].ground_truth),
            )
            self.assertEqual(output["method_id"], "rag")
            self.assertEqual(backend.seen[0], records[0].instance_id)

    def test_state_path_escape_and_symlink_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            context = self.context(root)
            with self.assertRaisesRegex(
                MethodContractError,
                "unsafe method state path",
            ):
                method_state_path(context, "../outside")
            with self.assertRaisesRegex(
                MethodContractError,
                "unsafe method state path",
            ):
                method_state_path(context, ".")
            outside = root.parent / f"{root.name}-outside"
            outside.mkdir()
            try:
                (root / "state-link").symlink_to(
                    outside,
                    target_is_directory=True,
                )
                with self.assertRaisesRegex(
                    MethodContractError,
                    "symlinks",
                ):
                    method_state_path(context, "state-link/index")
            finally:
                outside.rmdir()

    def test_prediction_cannot_change_identity_or_gt(self) -> None:
        record = validate_prepared_record(prepared())
        with self.assertRaisesRegex(
            MethodContractError,
            "canonical field 'instance_id'",
        ):
            canonical_prediction(
                record,
                method_id="rag",
                prediction={
                    "instance_id": "other",
                    "prediction": "x",
                },
            )
        with self.assertRaisesRegex(
            MethodContractError,
            "canonical field 'ground_truth'",
        ):
            canonical_prediction(
                record,
                method_id="rag",
                prediction={
                    "ground_truth": [],
                    "prediction": "x",
                },
            )

    def test_registry_rejects_unknown_duplicate_and_fake_methods(self) -> None:
        registry = builtin_method_registry()
        self.assertEqual(
            registry.method_ids,
            (
                "langmem",
                "mem0",
                "preference_memory",
                "rag",
                "vanilla_llm",
            ),
        )
        with self.assertRaisesRegex(
            MethodRegistryError,
            "unknown method adapter",
        ):
            registry.get("unknown")
        with self.assertRaisesRegex(
            MethodRegistryError,
            "duplicate method adapter",
        ):
            registry.register(registry.get("vanilla_llm"))
        broken = type("Broken", (), {"method_id": "broken"})()
        with self.assertRaisesRegex(
            MethodRegistryError,
            "must implement",
        ):
            MethodRegistry((broken,))

    def test_vanilla_adapter_uses_injected_backend_without_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            context = self.context(Path(temporary))
            record = validate_prepared_record(prepared())
            seen = []

            def backend(payload, request):
                seen.append(
                    (payload["instance_id"], request.model_name)
                )
                return 'GetHotels(star="4")'

            adapter = builtin_method_registry().get("vanilla_llm")
            self.assertIsNone(
                adapter.build((record,), context, backend)
            )
            output = asyncio.run(
                adapter.infer(record, context, None, backend)
            )
            self.assertEqual(
                seen,
                [(record.instance_id, "fake-model")],
            )
            self.assertEqual(output["method_id"], "vanilla_llm")
            self.assertEqual(
                output["instance_id"],
                record.instance_id,
            )
            self.assertEqual(
                output["ground_truth"],
                list(record.ground_truth),
            )


if __name__ == "__main__":
    unittest.main()
