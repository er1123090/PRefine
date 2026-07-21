from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exp7.methods.vanilla_llm.runner import (
    InferenceResult,
    ManifestVerificationError,
    _committed_checkpoint_seal_bytes,
    _pending_checkpoint_seal_bytes,
    run_prepared_file,
    run_records,
    verify_prepared_input,
)
from exp7.provenance import admission


def prepared(instance_id: str = "mix600-v1:singleturn:easy:x:hash:0"):
    return {
        "dataset_id": "mix600-v1",
        "difficulty": "easy",
        "ground_truth": ['GetHotels(star="4")'],
        "instance_id": instance_id,
        "legacy_example_id_sub": "example_0",
        "query": "Find a hotel.",
        "source_example": {"example_id": "example", "sessions": []},
        "source_example_id": "example",
        "turn": "singleturn",
    }


def write_prepared_bundle(root: Path, records):
    prepared_path = root / "singleturn/easy.jsonl"
    prepared_path.parent.mkdir(parents=True)
    payload = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ).encode("utf-8")
    prepared_path.write_bytes(payload)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "dataset_id": "mix600-v1",
                "outputs": {
                    "singleturn.easy": {
                        "count": len(records),
                        "path": "singleturn/easy.jsonl",
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return prepared_path, manifest_path


def write_checkpoint_with_seal(path: Path, records) -> None:
    payload = "".join(
        json.dumps(record, ensure_ascii=False) + "\n" for record in records
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.with_name(path.name + ".integrity.json").write_bytes(
        _committed_checkpoint_seal_bytes(payload)
    )


class VanillaRunnerTests(unittest.TestCase):
    def _assert_resume_checkpoint_rejected(
        self,
        *,
        checkpoint_inference,
        checkpoint_model: str = "model-a",
        checkpoint_reasoning_effort: str | None = None,
        resume_model: str = "model-a",
        resume_reasoning_effort: str | None = None,
        mutate=lambda row: None,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [prepared("first"), prepared("second")]
            prepared_path, manifest_path = write_prepared_bundle(root, records)
            checkpoint_row = asyncio.run(
                run_records(
                    [records[0]],
                    inference=checkpoint_inference,
                    model_name=checkpoint_model,
                    tools_schema=[],
                    reasoning_effort=checkpoint_reasoning_effort,
                )
            )[0]
            mutate(checkpoint_row)
            checkpoint = root / "run/checkpoint.jsonl"
            write_checkpoint_with_seal(checkpoint, [checkpoint_row])
            checkpoint_bytes = checkpoint.read_bytes()
            output_path = root / "run/predictions.json"
            calls = []

            def provider(record, request):
                calls.append(record["instance_id"])
                return "GetBanks()"

            with self.assertRaisesRegex(ManifestVerificationError, "checkpoint"):
                run_prepared_file(
                    prepared_path=prepared_path,
                    manifest_path=manifest_path,
                    output_path=output_path,
                    checkpoint_path=checkpoint,
                    tools_schema=[],
                    model_name=resume_model,
                    reasoning_effort=resume_reasoning_effort,
                    inference=provider,
                    resume=True,
                )
            self.assertEqual(calls, [])
            self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
            self.assertFalse(output_path.exists())

    def test_sync_fake_provider_preserves_evaluator_fields(self) -> None:
        seen = []

        def fake(record, request):
            seen.append((record["instance_id"], request.model_name))
            return InferenceResult(
                content='GetHotels(star="4")',
                reasoning_content="selected repeated star rating",
                token_counts={"reasoning_tokens": 7},
            )

        output = asyncio.run(
            run_records(
                [prepared()],
                inference=fake,
                model_name="fake-model",
                tools_schema=[{"type": "function"}],
                reasoning_effort="low",
            )
        )[0]

        self.assertEqual(seen, [(prepared()["instance_id"], "fake-model")])
        self.assertEqual(output["example_id_sub"], "example_0")
        self.assertEqual(output["user_utterance"], "Find a hotel.")
        self.assertEqual(output["reference_ground_truth"], ['GetHotels(star="4")'])
        self.assertEqual(output["llm_output"], 'GetHotels(star="4")')
        self.assertEqual(output["prediction"], output["llm_output"])
        self.assertEqual(output["response"], output["llm_output"])
        self.assertEqual(output["status"], "ok")
        self.assertIsNone(output["error"])
        self.assertEqual(output["reasoning_token_count"], 7)
        self.assertEqual(output["source_example"]["example_id"], "example")
        self.assertEqual(output["example_id"], "example")

    def test_async_fake_provider_and_completed_records(self) -> None:
        calls = []

        async def fake(record, request):
            calls.append(record["instance_id"])
            return {"content": "GetBanks()", "reasoning": "async"}

        first = prepared("first")
        prior = {"instance_id": "first", "status": "ok", "llm_output": "prior"}
        output = asyncio.run(
            run_records(
                [first, prepared("second")],
                inference=fake,
                model_name="fake-model",
                tools_schema=[],
                completed={"first": prior},
            )
        )
        self.assertEqual(calls, ["second"])
        self.assertEqual(output[0], prior)
        self.assertEqual(output[1]["reasoning_content"], "async")

    def test_manifest_tamper_is_rejected_before_inference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            prepared_path, manifest_path = write_prepared_bundle(
                Path(temporary), [prepared()]
            )
            with prepared_path.open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            with self.assertRaisesRegex(
                ManifestVerificationError, "prepared SHA-256 mismatch"
            ):
                verify_prepared_input(prepared_path, manifest_path)

    def test_jsonl_checkpoint_resumes_without_repeating_completed_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [prepared("first"), prepared("second")]
            prepared_path, manifest_path = write_prepared_bundle(root, records)
            checkpoint = root / "run/checkpoint.jsonl"
            first_output = asyncio.run(
                run_records(
                    [records[0]],
                    inference=lambda record, request: "GetBanks()",
                    model_name="fake-model",
                    tools_schema=[],
                )
            )[0]
            write_checkpoint_with_seal(checkpoint, [first_output])
            calls = []

            def fake(record, request):
                calls.append(record["instance_id"])
                return "GetBanks()"

            output_path = root / "run/result.json"
            summary = run_prepared_file(
                prepared_path=prepared_path,
                manifest_path=manifest_path,
                output_path=output_path,
                checkpoint_path=checkpoint,
                tools_schema=[],
                model_name="fake-model",
                inference=fake,
                resume=True,
            )
            final = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(calls, ["second"])
            self.assertEqual([row["instance_id"] for row in final], ["first", "second"])
            self.assertEqual(summary.resumed_count, 1)
            self.assertEqual(summary.inferred_count, 1)
            with self.assertRaises(FileExistsError):
                run_prepared_file(
                    prepared_path=prepared_path,
                    manifest_path=manifest_path,
                    output_path=output_path,
                    checkpoint_path=checkpoint,
                    tools_schema=[],
                    model_name="fake-model",
                    inference=fake,
                    resume=True,
                )

    def test_resume_rejects_checkpoint_identity_query_gt_or_status_tamper(self) -> None:
        mutations = {
            "dataset_id": lambda row: row.update(dataset_id="tampered-dataset"),
            "difficulty": lambda row: row.update(difficulty="hard"),
            "example_id_sub": lambda row: row.update(example_id_sub="tampered"),
            "ground_truth": lambda row: row.update(ground_truth=["GetBanks()"]),
            "instance_id": lambda row: row.update(instance_id="tampered-instance"),
            "query": lambda row: row.update(query="Tampered query."),
            "reference_ground_truth": lambda row: row.update(
                reference_ground_truth=["GetBanks()"]
            ),
            "source_example": lambda row: row.update(
                source_example={"example_id": "tampered", "sessions": []}
            ),
            "source_example_id": lambda row: row.update(
                source_example_id="tampered-source"
            ),
            "status": lambda row: row.update(status="error"),
            "turn": lambda row: row.update(turn="multiturn"),
            "user_utterance": lambda row: row.update(
                user_utterance="Tampered query."
            ),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    records = [prepared("first"), prepared("second")]
                    prepared_path, manifest_path = write_prepared_bundle(root, records)
                    checkpoint_row = asyncio.run(
                        run_records(
                            [records[0]],
                            inference=lambda record, request: "GetBanks()",
                            model_name="fake-model",
                            tools_schema=[],
                        )
                    )[0]
                    mutate(checkpoint_row)
                    checkpoint = root / "run/checkpoint.jsonl"
                    write_checkpoint_with_seal(checkpoint, [checkpoint_row])
                    checkpoint_bytes = checkpoint.read_bytes()
                    output_path = root / "run/predictions.json"
                    calls = []

                    def provider(record, request):
                        calls.append(record["instance_id"])
                        return "GetBanks()"

                    with self.assertRaisesRegex(
                        ManifestVerificationError,
                        "checkpoint",
                    ):
                        run_prepared_file(
                            prepared_path=prepared_path,
                            manifest_path=manifest_path,
                            output_path=output_path,
                            checkpoint_path=checkpoint,
                            tools_schema=[],
                            model_name="fake-model",
                            inference=provider,
                            resume=True,
                        )
                    self.assertEqual(calls, [])
                    self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
                    self.assertFalse(output_path.exists())

    def test_resume_rejects_coordinated_status_error_falsification(self) -> None:
        def provider_error(record, request):
            raise RuntimeError("provider unavailable")

        cases = {
            "ok_to_error": (
                lambda record, request: "GetBanks()",
                lambda row: row.update(status="error", error="forged failure"),
            ),
            "error_to_ok": (
                provider_error,
                lambda row: row.update(status="ok", error=None),
            ),
        }
        for name, (checkpoint_inference, mutate) in cases.items():
            with self.subTest(name=name):
                self._assert_resume_checkpoint_rejected(
                    checkpoint_inference=checkpoint_inference,
                    mutate=mutate,
                )

    def test_resume_rejects_checkpoint_model_name_drift(self) -> None:
        self._assert_resume_checkpoint_rejected(
            checkpoint_inference=lambda record, request: "GetBanks()",
            checkpoint_model="model-a",
            resume_model="model-b",
        )

    def test_resume_rejects_checkpoint_reasoning_effort_drift(self) -> None:
        self._assert_resume_checkpoint_rejected(
            checkpoint_inference=lambda record, request: "GetBanks()",
            checkpoint_reasoning_effort="low",
            resume_reasoning_effort="high",
        )

    def test_resume_rejects_tampered_provider_result_semantics(self) -> None:
        mutations = {
            "prediction_alias": lambda row: row.update(
                prediction="tampered prediction"
            ),
            "reasoning_alias": lambda row: row.update(
                reasoning_content="tampered reasoning"
            ),
            "latency_alias": lambda row: row.update(
                latency_seconds=row["latency"] + 1
            ),
            "reasoning_token_count": lambda row: row.update(
                reasoning_token_count=row["reasoning_token_count"] + 1
            ),
            "token_counts": lambda row: row.update(
                token_counts={"reasoning_tokens": 9}
            ),
            "missing_response": lambda row: row.pop("response"),
        }
        for field, mutate in mutations.items():
            with self.subTest(field=field):
                self._assert_resume_checkpoint_rejected(
                    checkpoint_inference=lambda record, request: InferenceResult(
                        content="GetBanks()",
                        reasoning_content="selected bank lookup",
                        token_counts={"reasoning_tokens": 3},
                    ),
                    mutate=mutate,
                )

    def test_resume_rejects_coordinated_provider_result_forgery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [prepared("first"), prepared("second")]
            prepared_path, manifest_path = write_prepared_bundle(root, records)
            checkpoint = root / "run/checkpoint.jsonl"
            output_path = root / "run/predictions.json"

            def interrupt_after_first(record, request):
                if record["instance_id"] == "second":
                    raise KeyboardInterrupt("simulated process interruption")
                return "GetBanks()"

            with self.assertRaises(KeyboardInterrupt):
                run_prepared_file(
                    prepared_path=prepared_path,
                    manifest_path=manifest_path,
                    output_path=output_path,
                    checkpoint_path=checkpoint,
                    tools_schema=[],
                    model_name="model-a",
                    inference=interrupt_after_first,
                )
            self.assertFalse(output_path.exists())
            checkpoint_rows = [
                json.loads(line)
                for line in checkpoint.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(checkpoint_rows), 1)
            checkpoint_rows[0].update(
                llm_output="forged provider output",
                prediction="forged provider output",
                response="forged provider output",
            )
            checkpoint.write_text(
                json.dumps(checkpoint_rows[0], ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            checkpoint_bytes = checkpoint.read_bytes()
            calls = []

            def provider(record, request):
                calls.append(record["instance_id"])
                return "GetBanks()"

            with self.assertRaisesRegex(
                ManifestVerificationError,
                "checkpoint integrity",
            ):
                run_prepared_file(
                    prepared_path=prepared_path,
                    manifest_path=manifest_path,
                    output_path=output_path,
                    checkpoint_path=checkpoint,
                    tools_schema=[],
                    model_name="model-a",
                    inference=provider,
                    resume=True,
                )
            self.assertEqual(calls, [])
            self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
            self.assertFalse(output_path.exists())

    def test_resume_rejects_missing_or_tampered_committed_seal(self) -> None:
        for mutation in ("missing", "tampered"):
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    records = [prepared("first"), prepared("second")]
                    prepared_path, manifest_path = write_prepared_bundle(
                        root, records
                    )
                    first_output = asyncio.run(
                        run_records(
                            [records[0]],
                            inference=lambda record, request: "GetBanks()",
                            model_name="model-a",
                            tools_schema=[],
                        )
                    )[0]
                    checkpoint = root / "run/checkpoint.jsonl"
                    write_checkpoint_with_seal(checkpoint, [first_output])
                    checkpoint_bytes = checkpoint.read_bytes()
                    seal_path = checkpoint.with_name(
                        checkpoint.name + ".integrity.json"
                    )
                    if mutation == "missing":
                        seal_path.unlink()
                    else:
                        seal = json.loads(seal_path.read_text(encoding="utf-8"))
                        seal["checkpoint_sha256"] = "0" * 64
                        seal_path.write_text(
                            json.dumps(seal, sort_keys=True) + "\n",
                            encoding="utf-8",
                        )
                    calls = []

                    def provider(record, request):
                        calls.append(record["instance_id"])
                        return "GetBanks()"

                    output_path = root / "run/predictions.json"
                    with self.assertRaisesRegex(
                        ManifestVerificationError,
                        "checkpoint integrity",
                    ):
                        run_prepared_file(
                            prepared_path=prepared_path,
                            manifest_path=manifest_path,
                            output_path=output_path,
                            checkpoint_path=checkpoint,
                            tools_schema=[],
                            model_name="model-a",
                            inference=provider,
                            resume=True,
                        )
                    self.assertEqual(calls, [])
                    self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
                    self.assertFalse(output_path.exists())

    def test_resume_recovers_partial_append_from_pending_integrity_seal(self) -> None:
        for stage in ("before_checkpoint_append", "partial_append", "after_append"):
            with self.subTest(stage=stage):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    records = [prepared("first"), prepared("second")]
                    prepared_path, manifest_path = write_prepared_bundle(
                        root, records
                    )
                    first_output = asyncio.run(
                        run_records(
                            [records[0]],
                            inference=lambda record, request: "GetBanks()",
                            model_name="model-a",
                            tools_schema=[],
                        )
                    )[0]
                    appended = (
                        json.dumps(
                            first_output,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    ).encode("utf-8")
                    checkpoint = root / "run/checkpoint.jsonl"
                    checkpoint.parent.mkdir()
                    if stage == "partial_append":
                        checkpoint.write_bytes(appended[: len(appended) // 2])
                    elif stage == "after_append":
                        checkpoint.write_bytes(appended)
                    checkpoint.with_name(
                        checkpoint.name + ".integrity.json"
                    ).write_bytes(_pending_checkpoint_seal_bytes(b"", appended))
                    calls = []

                    def provider(record, request):
                        calls.append(record["instance_id"])
                        return "GetBanks()"

                    output_path = root / "run/predictions.json"
                    summary = run_prepared_file(
                        prepared_path=prepared_path,
                        manifest_path=manifest_path,
                        output_path=output_path,
                        checkpoint_path=checkpoint,
                        tools_schema=[],
                        model_name="model-a",
                        inference=provider,
                        resume=True,
                    )
                    predictions = json.loads(
                        output_path.read_text(encoding="utf-8")
                    )
                    self.assertEqual(calls, ["second"])
                    self.assertEqual(summary.resumed_count, 1)
                    self.assertEqual(
                        [record["instance_id"] for record in predictions],
                        ["first", "second"],
                    )

    def test_resume_reuses_legitimate_provider_error_checkpoint_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [prepared("first"), prepared("second")]
            prepared_path, manifest_path = write_prepared_bundle(root, records)

            def fail(record, request):
                raise RuntimeError("provider unavailable")

            checkpoint_row = asyncio.run(
                run_records(
                    [records[0]],
                    inference=fail,
                    model_name="fake-model",
                    tools_schema=[],
                )
            )[0]
            checkpoint = root / "run/checkpoint.jsonl"
            write_checkpoint_with_seal(checkpoint, [checkpoint_row])
            calls = []

            def provider(record, request):
                calls.append(record["instance_id"])
                return "GetBanks()"

            summary = run_prepared_file(
                prepared_path=prepared_path,
                manifest_path=manifest_path,
                output_path=root / "run/predictions.json",
                checkpoint_path=checkpoint,
                tools_schema=[],
                model_name="fake-model",
                inference=provider,
                resume=True,
            )
            self.assertEqual(calls, ["second"])
            self.assertEqual(summary.resumed_count, 1)
            self.assertEqual(summary.inferred_count, 1)
            self.assertEqual(summary.error_count, 1)
            self.assertEqual(summary.status, "complete_with_provider_errors")

    def test_manifest_and_prepared_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path, manifest_path = write_prepared_bundle(root, [prepared()])
            real_manifest = root / "real-manifest.json"
            manifest_path.rename(real_manifest)
            manifest_path.symlink_to(real_manifest)
            with self.assertRaisesRegex(ManifestVerificationError, "non-symlink"):
                verify_prepared_input(prepared_path, manifest_path)

    def test_manifest_and_prepared_parent_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_root = root / "real"
            prepared_path, manifest_path = write_prepared_bundle(
                real_root, [prepared()]
            )
            alias = root / "alias"
            alias.symlink_to(real_root, target_is_directory=True)
            with self.assertRaisesRegex(
                ManifestVerificationError,
                "parent must be a non-symlink directory",
            ):
                verify_prepared_input(
                    alias / prepared_path.relative_to(real_root),
                    alias / manifest_path.relative_to(real_root),
                )

    def test_prepared_parent_replacement_with_same_leaf_inode_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path, manifest_path = write_prepared_bundle(root, [prepared()])
            parent = prepared_path.parent
            original = parent.with_name("singleturn-original")
            hostile = parent.with_name("singleturn-hostile")
            hostile.mkdir()
            os.link(prepared_path, hostile / prepared_path.name)
            target_stat = os.stat(prepared_path, follow_symlinks=False)
            real_fstat = admission.os.fstat
            leaf_observations = 0
            triggered = False

            def replace_parent_after_leaf_read(descriptor):
                nonlocal leaf_observations, triggered
                value = real_fstat(descriptor)
                if (
                    stat.S_ISREG(value.st_mode)
                    and (value.st_dev, value.st_ino)
                    == (target_stat.st_dev, target_stat.st_ino)
                ):
                    leaf_observations += 1
                    if leaf_observations == 2:
                        parent.rename(original)
                        hostile.rename(parent)
                        triggered = True
                return value

            with mock.patch.object(
                admission.os,
                "fstat",
                side_effect=replace_parent_after_leaf_read,
            ):
                with self.assertRaisesRegex(
                    ManifestVerificationError,
                    "parent changed while being read",
                ):
                    verify_prepared_input(prepared_path, manifest_path)
            self.assertTrue(triggered)

    def test_output_and_checkpoint_parent_symlink_is_rejected_before_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path, manifest_path = write_prepared_bundle(
                root / "inputs", [prepared()]
            )
            outside = root / "outside"
            outside.mkdir()
            output_alias = root / "output-alias"
            output_alias.symlink_to(outside, target_is_directory=True)
            calls = []

            def provider(record, request):
                calls.append(record["instance_id"])
                return "GetBanks()"

            with self.assertRaisesRegex(
                ManifestVerificationError,
                "parent must be a non-symlink directory",
            ):
                run_prepared_file(
                    prepared_path=prepared_path,
                    manifest_path=manifest_path,
                    output_path=output_alias / "predictions.json",
                    checkpoint_path=output_alias / "checkpoint.jsonl",
                    tools_schema=[],
                    model_name="fake-model",
                    inference=provider,
                )
            self.assertEqual(calls, [])
            self.assertEqual(list(outside.iterdir()), [])

    def test_output_parent_replacement_during_provider_cannot_redirect_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path, manifest_path = write_prepared_bundle(
                root / "inputs", [prepared()]
            )
            output_parent = root / "run"
            original_parent = root / "run-original"
            hostile_parent = root / "run-hostile"
            hostile_parent.mkdir()

            def replace_output_parent(record, request):
                output_parent.rename(original_parent)
                hostile_parent.rename(output_parent)
                return "GetBanks()"

            with self.assertRaisesRegex(
                ManifestVerificationError,
                "parent changed while being used",
            ):
                run_prepared_file(
                    prepared_path=prepared_path,
                    manifest_path=manifest_path,
                    output_path=output_parent / "predictions.json",
                    checkpoint_path=output_parent / "checkpoint.jsonl",
                    tools_schema=[],
                    model_name="fake-model",
                    inference=replace_output_parent,
                )
            self.assertEqual(list(output_parent.iterdir()), [])
            self.assertFalse((output_parent / "predictions.json").exists())

    def test_nested_output_parent_replacement_cannot_redirect_final_publish(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path, manifest_path = write_prepared_bundle(
                root / "inputs", [prepared()]
            )
            output_root = root / "run"
            nested_parent = output_root / "predictions"
            original_parent = output_root / "predictions-original"
            hostile_parent = output_root / "predictions-hostile"
            output_root.mkdir()
            hostile_parent.mkdir()

            def replace_nested_parent(record, request):
                nested_parent.rename(original_parent)
                hostile_parent.rename(nested_parent)
                return "GetBanks()"

            with self.assertRaisesRegex(
                ManifestVerificationError,
                "parent changed while being used",
            ):
                run_prepared_file(
                    prepared_path=prepared_path,
                    manifest_path=manifest_path,
                    output_path=nested_parent / "predictions.json",
                    checkpoint_path=nested_parent / "checkpoint.jsonl",
                    tools_schema=[],
                    model_name="fake-model",
                    inference=replace_nested_parent,
                    output_root=output_root,
                    expected_output_root_identity=(
                        os.stat(output_root).st_dev,
                        os.stat(output_root).st_ino,
                    ),
                )
            self.assertEqual(list(nested_parent.iterdir()), [])
            self.assertFalse((nested_parent / "predictions.json").exists())

    def test_expected_output_root_identity_mismatch_fails_before_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path, manifest_path = write_prepared_bundle(
                root / "inputs", [prepared()]
            )
            output_root = root / "run"
            output_root.mkdir()
            identity = os.stat(output_root, follow_symlinks=False)
            calls = []

            with self.assertRaisesRegex(
                ManifestVerificationError,
                "identity differs from parent runner",
            ):
                run_prepared_file(
                    prepared_path=prepared_path,
                    manifest_path=manifest_path,
                    output_path=output_root / "predictions/predictions.json",
                    checkpoint_path=output_root / "predictions/checkpoint.jsonl",
                    tools_schema=[],
                    model_name="fake-model",
                    inference=lambda record, request: calls.append(record),
                    output_root=output_root,
                    expected_output_root_identity=(
                        identity.st_dev,
                        identity.st_ino + 1,
                    ),
                )
            self.assertEqual(calls, [])
            self.assertEqual(list(output_root.iterdir()), [])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path, manifest_path = write_prepared_bundle(root, [prepared()])
            real_prepared = root / "real-prepared.jsonl"
            prepared_path.rename(real_prepared)
            prepared_path.symlink_to(real_prepared)
            with self.assertRaisesRegex(ManifestVerificationError, "non-symlink"):
                verify_prepared_input(prepared_path, manifest_path)

    def test_prepared_mutation_during_read_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path, manifest_path = write_prepared_bundle(root, [prepared()])
            real_fstat = admission.os.fstat
            target = os.stat(prepared_path, follow_symlinks=False)
            leaf_observations = 0

            def mutate_prepared_on_final_fstat(descriptor):
                nonlocal leaf_observations
                value = real_fstat(descriptor)
                if (
                    stat.S_ISREG(value.st_mode)
                    and (value.st_dev, value.st_ino)
                    == (target.st_dev, target.st_ino)
                ):
                    leaf_observations += 1
                if leaf_observations == 2:
                    leaf_observations += 1
                    prepared_path.write_text("{}\n", encoding="utf-8")
                return value

            with mock.patch.object(
                admission.os,
                "fstat",
                side_effect=mutate_prepared_on_final_fstat,
            ):
                with self.assertRaisesRegex(
                    ManifestVerificationError, "changed while being read"
                ):
                    verify_prepared_input(prepared_path, manifest_path)

    def test_provider_error_statuses_are_truthful_and_evaluator_readable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = [prepared("first"), prepared("second")]
            prepared_path, manifest_path = write_prepared_bundle(root, records)

            def mixed_provider(record, request):
                if record["instance_id"] == "first":
                    raise RuntimeError("provider unavailable")
                return "GetBanks()"

            summary = run_prepared_file(
                prepared_path=prepared_path,
                manifest_path=manifest_path,
                output_path=root / "run/predictions.json",
                checkpoint_path=root / "run/checkpoint.jsonl",
                tools_schema=[],
                model_name="fake-model",
                inference=mixed_provider,
            )
            predictions = json.loads(summary.output_path.read_text(encoding="utf-8"))
            self.assertEqual(summary.status, "complete_with_provider_errors")
            self.assertEqual((summary.ok_count, summary.error_count), (1, 1))
            failed = predictions[0]
            self.assertEqual(failed["status"], "error")
            self.assertEqual(failed["error"], "provider unavailable")
            self.assertEqual(failed["ground_truth"], ['GetHotels(star="4")'])
            self.assertEqual(failed["reference_ground_truth"], failed["ground_truth"])
            self.assertIn("API_ERROR", failed["prediction"])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_path, manifest_path = write_prepared_bundle(root, [prepared()])
            summary = run_prepared_file(
                prepared_path=prepared_path,
                manifest_path=manifest_path,
                output_path=root / "run/predictions.json",
                checkpoint_path=root / "run/checkpoint.jsonl",
                tools_schema=[],
                model_name="fake-model",
                inference=lambda record, request: (_ for _ in ()).throw(
                    RuntimeError("provider unavailable")
                ),
            )
            self.assertEqual(summary.status, "all_provider_errors")
            self.assertEqual((summary.ok_count, summary.error_count), (0, 1))

    def test_resume_rejects_checkpoint_with_unknown_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            record = prepared()
            prepared_path, manifest_path = write_prepared_bundle(root, [record])
            checkpoint = root / "run/checkpoint.jsonl"
            write_checkpoint_with_seal(
                checkpoint,
                [{"instance_id": record["instance_id"], "status": "pending"}],
            )
            with self.assertRaisesRegex(
                ManifestVerificationError, "status 'ok' or 'error'"
            ):
                run_prepared_file(
                    prepared_path=prepared_path,
                    manifest_path=manifest_path,
                    output_path=root / "run/predictions.json",
                    checkpoint_path=checkpoint,
                    tools_schema=[],
                    model_name="fake-model",
                    inference=lambda record, request: "not called",
                    resume=True,
                )

    def test_shell_dry_run_has_six_canonical_commands_in_order(self) -> None:
        script = ROOT / "scripts/run_vanilla_llm_mix600_all_difficulties.sh"
        result = subprocess.run(
            [
                "bash",
                str(script),
                "--dry-run",
                "--prepared-root",
                "/tmp/prepared-safe",
                "--output-root",
                "/tmp/output-safe",
                "--model",
                "fake-model",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        labels = [line.split("]", 1)[0] + "]" for line in result.stdout.splitlines()]
        self.assertEqual(
            labels,
            [
                "[singleturn.easy]",
                "[singleturn.medium]",
                "[singleturn.hard]",
                "[multiturn.easy]",
                "[multiturn.medium]",
                "[multiturn.hard]",
            ],
        )
        self.assertEqual(result.stdout.count("scripts/run.py"), 6)
        self.assertEqual(result.stdout.count("schema_easy.json"), 3)
        self.assertEqual(result.stdout.count("schema_all.json"), 3)
        source = script.read_text(encoding="utf-8")
        for forbidden in ("experiments4", "experiment.py", "query_singleturn", "pref_group"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
