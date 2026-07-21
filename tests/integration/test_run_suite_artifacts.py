from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exp7.evaluation import EvaluationError, evaluate_run
from exp7.experiments import load_experiment_config
from exp7.methods.vanilla_llm.runner import (
    _pending_checkpoint_seal_bytes,
    checkpoint_integrity_path,
    run_prepared_file,
    run_records,
)
from exp7.provenance import run_artifacts
from exp7.provenance.admission import AdmittedFile
from exp7.provenance.run_artifacts import RunArtifactError, run_suite


CONDITIONS = tuple(
    (turn, difficulty)
    for turn in ("singleturn", "multiturn")
    for difficulty in ("easy", "medium", "hard")
)
CANONICAL_CONFIG_PATH = ROOT / "configs/experiments/vanilla_mix600.json"


def write_prepared_bundle(root: Path) -> Path:
    prepared_root = root / "prepared"
    outputs = {}
    for index, (turn, difficulty) in enumerate(CONDITIONS):
        relative = f"{turn}/{difficulty}.jsonl"
        path = prepared_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "dataset_id": "mix600-v1",
            "difficulty": difficulty,
            "ground_truth": ["GetBanks()"],
            "instance_id": f"mix600-v1:{turn}:{difficulty}:{index}",
            "query": "Find it.",
            "source_example_id": f"example-{index}",
            "turn": turn,
        }
        payload = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
        path.write_bytes(payload)
        outputs[f"{turn}.{difficulty}"] = {
            "count": 1,
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    (prepared_root / "manifest.json").write_text(
        json.dumps(
            {
                "dataset_id": "mix600-v1",
                "format_version": 1,
                "outputs": outputs,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return prepared_root


def write_experiment_config(root: Path, prepared_root: Path):
    document = json.loads(CANONICAL_CONFIG_PATH.read_text(encoding="utf-8"))
    prepared_relative = prepared_root.relative_to(ROOT).as_posix()
    document["dataset"]["prepared_root"] = prepared_relative
    document["dataset"]["manifest"] = f"{prepared_relative}/manifest.json"
    document["output"]["root"] = root.relative_to(ROOT).as_posix()
    document["runtime"]["python_bin"] = Path(sys.executable).name
    config_path = root / "experiment.json"
    config_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    relative = config_path.relative_to(ROOT).as_posix()
    return load_experiment_config(ROOT, relative)


def successful_fake(calls):
    def run(invocation, log):
        calls.append(invocation.label)
        records = [
            json.loads(line)
            for line in invocation.prepared_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        predictions = [
            {"instance_id": record["instance_id"], "status": "ok"}
            for record in records
        ]
        invocation.output_path.write_text(json.dumps(predictions), encoding="utf-8")
        invocation.checkpoint_path.write_text(
            "".join(json.dumps(record) + "\n" for record in predictions),
            encoding="utf-8",
        )
        log.write(f"fake {invocation.label}\n")
        return 0

    return run


def provider_status_fake(calls, status_for_label):
    def run(invocation, log):
        calls.append(invocation.label)
        records = [
            json.loads(line)
            for line in invocation.prepared_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        status = status_for_label(invocation.label)
        predictions = [
            {
                "error": "provider unavailable" if status == "error" else None,
                "instance_id": record["instance_id"],
                "prediction": "API_ERROR: provider unavailable" if status == "error" else "GetBanks()",
                "status": status,
            }
            for record in records
        ]
        invocation.output_path.write_text(json.dumps(predictions), encoding="utf-8")
        invocation.checkpoint_path.write_text(
            "".join(json.dumps(record) + "\n" for record in predictions),
            encoding="utf-8",
        )
        log.write(f"fake {invocation.label}\n")
        return 0

    return run


class RunSuiteArtifactTests(unittest.TestCase):
    def test_success_writes_one_standard_run_tree_without_fake_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            prepared_root = write_prepared_bundle(temporary_root)
            run_dir = temporary_root / "run"
            calls = []
            invocations = []
            transitions = []
            original = run_artifacts._write_status
            successful = successful_fake(calls)

            def capture(run_dir_arg, status, conditions, **kwargs):
                transitions.append(status)
                return original(run_dir_arg, status, conditions, **kwargs)

            def capture_invocation(invocation, log):
                invocations.append(invocation)
                return successful(invocation, log)

            with mock.patch.object(run_artifacts, "_write_status", side_effect=capture):
                summary = run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    condition_runner=capture_invocation,
                )

            self.assertEqual(summary.status, "complete")
            self.assertEqual(calls, [f"{turn}.{difficulty}" for turn, difficulty in CONDITIONS])
            self.assertEqual(transitions[0], "planned")
            self.assertIn("running", transitions)
            self.assertEqual(transitions[-1], "complete")
            self.assertEqual(
                {path.name for path in run_dir.iterdir()},
                {
                    "dataset_manifest.json",
                    "metrics",
                    "predictions",
                    "resolved_config.json",
                    "run.log",
                    "status.json",
                    "tool_schemas",
                },
            )
            self.assertEqual(
                (run_dir / "dataset_manifest.json").read_bytes(),
                (prepared_root / "manifest.json").read_bytes(),
            )
            resolved = json.loads((run_dir / "resolved_config.json").read_text())
            runtime = resolved["python_runtime"]
            self.assertEqual(runtime["requested"], sys.executable)
            self.assertTrue(Path(runtime["executable"]).is_absolute())
            fingerprint = runtime.pop("fingerprint_sha256")
            self.assertEqual(
                fingerprint,
                hashlib.sha256(
                    json.dumps(
                        runtime,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
            )
            self.assertEqual(resolved["output_policy"], "legacy")
            schemas = resolved["tool_schemas"]
            self.assertEqual(set(schemas), {"singleturn", "multiturn"})
            for turn, filename in (
                ("singleturn", "schema_easy.json"),
                ("multiturn", "schema_all.json"),
            ):
                snapshot = run_dir / schemas[turn]["path"]
                source = ROOT / "configs/schemas/mix600-v1" / filename
                self.assertEqual(snapshot.read_bytes(), source.read_bytes())
                self.assertEqual(
                    schemas[turn]["sha256"],
                    hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                )
            self.assertTrue(
                all(invocation.command[0] == runtime["executable"] for invocation in invocations)
            )
            self.assertTrue(
                all("--tools-schema-sha256" in invocation.command for invocation in invocations)
            )
            metrics_dir = run_dir / "metrics"
            self.assertEqual([path.name for path in metrics_dir.iterdir()], ["status.json"])
            self.assertEqual(
                json.loads((metrics_dir / "status.json").read_text())["status"],
                "pending",
            )
            status = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertTrue(all(item["status"] == "complete" for item in status["conditions"].values()))
            for turn, difficulty in CONDITIONS:
                condition_dir = run_dir / "predictions" / turn / difficulty
                self.assertEqual(
                    {path.name for path in condition_dir.iterdir()},
                    {"checkpoint.jsonl", "predictions.json"},
                )

    def test_configured_suite_copies_exact_config_and_evaluates_all_errors(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".run-artifact-test-",
            dir=ROOT,
        ) as temporary:
            root = Path(temporary)
            prepared_root = write_prepared_bundle(root)
            run_dir = root / "run"
            config = write_experiment_config(root, prepared_root)
            evaluator_observations = []

            def recording_evaluator(*, run_dir, variant):
                evaluator_observations.append(
                    (
                        json.loads((run_dir / "status.json").read_text())["status"],
                        json.loads(
                            (run_dir / "metrics/status.json").read_text()
                        )["status"],
                        variant,
                    )
                )
                return evaluate_run(run_dir=run_dir, variant=variant)

            summary = run_suite(
                repo_root=ROOT,
                prepared_root=prepared_root,
                run_dir=run_dir,
                model=config.provider.model,
                python_bin=config.python_bin,
                reasoning_effort=config.provider.reasoning_effort,
                base_url=config.provider.base_url,
                api_key_env=config.provider.api_key_env,
                extra_args=config.runner_extra_args(),
                experiment_config_bytes=config.config_bytes,
                experiment_config_sha256=config.config_sha256,
                experiment_config_relative_path=config.config_relative_path,
                evaluator_variant=config.evaluator_variant,
                output_policy="config_root",
                evaluation_runner=recording_evaluator,
                condition_runner=provider_status_fake([], lambda label: "error"),
            )

            self.assertEqual(summary.status, "all_provider_errors")
            self.assertEqual(
                evaluator_observations,
                [
                    (
                        "all_provider_errors",
                        "running",
                        "exp6_slot_value_or_v1",
                    )
                ],
            )
            self.assertEqual(
                (run_dir / "experiment_config.json").read_bytes(),
                config.config_bytes,
            )
            resolved = json.loads((run_dir / "resolved_config.json").read_text())
            self.assertEqual(resolved["experiment_config"], "experiment_config.json")
            self.assertEqual(resolved["output_policy"], "config_root")
            self.assertEqual(
                resolved["experiment_config_sha256"],
                config.config_sha256,
            )
            self.assertEqual(
                resolved["evaluator_variant"],
                "exp6_slot_value_or_v1",
            )
            metrics_path = run_dir / "metrics/metrics.json"
            metrics_payload = metrics_path.read_bytes()
            metrics = json.loads(metrics_payload)
            self.assertEqual(
                metrics["provenance"]["experiment_config_sha256"],
                config.config_sha256,
            )
            metrics_status = json.loads(
                (run_dir / "metrics/status.json").read_text()
            )
            self.assertEqual(metrics_status["status"], "complete")
            self.assertEqual(metrics_status["variant"], "exp6_slot_value_or_v1")
            self.assertEqual(metrics_status["provider_status"], "all_provider_errors")
            self.assertEqual(metrics_status["row_count"], 6)
            self.assertEqual(metrics_status["row_error_count"], 6)
            self.assertEqual(
                metrics_status["metrics_sha256"],
                hashlib.sha256(metrics_payload).hexdigest(),
            )
            run_status = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(run_status["status"], "all_provider_errors")
            self.assertEqual(run_status["provider_error_count"], 6)

    def test_evaluation_failure_is_truthful_and_terminal_provider_status_remains(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".run-artifact-test-",
            dir=ROOT,
        ) as temporary:
            root = Path(temporary)
            prepared_root = write_prepared_bundle(root)
            run_dir = root / "run"
            config = write_experiment_config(root, prepared_root)

            def failing_evaluator(*, run_dir, variant):
                self.assertEqual(
                    json.loads((run_dir / "status.json").read_text())["status"],
                    "complete_with_provider_errors",
                )
                self.assertEqual(
                    json.loads(
                        (run_dir / "metrics/status.json").read_text()
                    )["status"],
                    "running",
                )
                raise EvaluationError("forced evaluator failure")

            configured = {
                "experiment_config_bytes": config.config_bytes,
                "experiment_config_sha256": config.config_sha256,
                "experiment_config_relative_path": config.config_relative_path,
                "evaluator_variant": config.evaluator_variant,
                "output_policy": "config_root",
                "evaluation_runner": failing_evaluator,
            }
            with self.assertRaisesRegex(
                RunArtifactError,
                "evaluation failed: forced evaluator failure",
            ):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model=config.provider.model,
                    python_bin=config.python_bin,
                    extra_args=config.runner_extra_args(),
                    condition_runner=provider_status_fake(
                        [],
                        lambda label: (
                            "error" if label == "singleturn.easy" else "ok"
                        ),
                    ),
                    **configured,
                )

            run_status = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(run_status["status"], "complete_with_provider_errors")
            self.assertEqual(run_status["provider_error_count"], 1)
            metrics_status = json.loads(
                (run_dir / "metrics/status.json").read_text()
            )
            self.assertEqual(metrics_status["status"], "failed")
            self.assertEqual(metrics_status["variant"], "exp6_slot_value_or_v1")
            self.assertEqual(
                metrics_status["provider_status"],
                "complete_with_provider_errors",
            )
            self.assertEqual(metrics_status["error"], "forced evaluator failure")
            self.assertFalse((run_dir / "metrics/metrics.json").exists())
            with self.assertRaisesRegex(RunArtifactError, "cannot be resumed"):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model=config.provider.model,
                    python_bin=config.python_bin,
                    extra_args=config.runner_extra_args(),
                    resume=True,
                    condition_runner=successful_fake([]),
                    **configured,
                )

    def test_configured_suite_rejects_sentinel_bytes_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            config_bytes = b'{"sentinel": "not an experiment config"}\n'
            with self.assertRaisesRegex(RunArtifactError, "experiment config"):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=write_prepared_bundle(root),
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    experiment_config_bytes=config_bytes,
                    experiment_config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                    experiment_config_relative_path=(
                        "configs/experiments/vanilla_mix600.json"
                    ),
                    evaluator_variant="exp6_slot_value_or_v1",
                    output_policy="config_root",
                    condition_runner=successful_fake([]),
                )
            self.assertFalse(run_dir.exists())

    def test_explicit_external_output_rejects_symlinked_run_parent(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".run-artifact-test-",
            dir=ROOT,
        ) as temporary:
            root = Path(temporary)
            prepared_root = write_prepared_bundle(root)
            config = write_experiment_config(root, prepared_root)
            outside = root / "outside"
            outside.mkdir()
            alias = root / "external-alias"
            alias.symlink_to(outside, target_is_directory=True)

            with self.assertRaisesRegex(
                RunArtifactError,
                "parent must be a non-symlink directory",
            ):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=alias / "run",
                    model=config.provider.model,
                    python_bin=config.python_bin,
                    reasoning_effort=config.provider.reasoning_effort,
                    base_url=config.provider.base_url,
                    api_key_env=config.provider.api_key_env,
                    extra_args=config.runner_extra_args(),
                    experiment_config_bytes=config.config_bytes,
                    experiment_config_sha256=config.config_sha256,
                    experiment_config_relative_path=config.config_relative_path,
                    evaluator_variant=config.evaluator_variant,
                    output_policy="external_opt_in",
                    condition_runner=successful_fake([]),
                )
            self.assertEqual(list(outside.iterdir()), [])

    def test_configured_suite_rejects_every_semantic_mismatch_before_writes(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".run-artifact-test-",
            dir=ROOT,
        ) as temporary:
            root = Path(temporary)
            prepared_root = write_prepared_bundle(root)
            config = write_experiment_config(root, prepared_root)
            base = {
                "repo_root": ROOT,
                "prepared_root": prepared_root,
                "run_dir": root / "run",
                "model": config.provider.model,
                "python_bin": config.python_bin,
                "reasoning_effort": config.provider.reasoning_effort,
                "base_url": config.provider.base_url,
                "api_key_env": config.provider.api_key_env,
                "extra_args": config.runner_extra_args(),
                "experiment_config_bytes": config.config_bytes,
                "experiment_config_sha256": config.config_sha256,
                "experiment_config_relative_path": config.config_relative_path,
                "evaluator_variant": config.evaluator_variant,
                "output_policy": "config_root",
                "condition_runner": successful_fake([]),
            }
            cases = {
                "prepared_root": root / "other-prepared",
                "model": "other-model",
                "python_bin": "other-python",
                "reasoning_effort": "low",
                "base_url": "https://provider.invalid/v1",
                "api_key_env": "OTHER_API_KEY",
                "extra_args": (*config.runner_extra_args(), "--resume"),
                "evaluator_variant": "other-evaluator",
            }
            for field, value in cases.items():
                arguments = {**base, field: value}
                with self.subTest(field=field):
                    with self.assertRaisesRegex(
                        RunArtifactError,
                        "execution semantics",
                    ):
                        run_suite(**arguments)
                    self.assertFalse((root / "run").exists())

    def test_config_digest_mismatch_fails_before_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            with self.assertRaisesRegex(
                RunArtifactError,
                "SHA-256 does not match",
            ):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=write_prepared_bundle(root),
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    experiment_config_bytes=b"{}\n",
                    experiment_config_sha256="0" * 64,
                    experiment_config_relative_path=(
                        "configs/experiments/vanilla_mix600.json"
                    ),
                    evaluator_variant="exp6_slot_value_or_v1",
                    output_policy="config_root",
                    condition_runner=successful_fake([]),
                )
            self.assertFalse(run_dir.exists())


    def test_failed_run_resumes_only_incomplete_conditions_and_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            prepared_root = write_prepared_bundle(temporary_root)
            run_dir = temporary_root / "run"
            initial_calls = []
            success = successful_fake(initial_calls)

            def fail_second(invocation, log):
                if invocation.label == "singleturn.medium":
                    initial_calls.append(invocation.label)
                    return 9
                return success(invocation, log)

            with self.assertRaisesRegex(RunArtifactError, "exit code 9"):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    condition_runner=fail_second,
                )
            failed = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["conditions"]["singleturn.medium"]["status"], "failed")

            with self.assertRaises(FileExistsError):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    condition_runner=successful_fake([]),
                )

            resumed_calls = []
            summary = run_suite(
                repo_root=ROOT,
                prepared_root=prepared_root,
                run_dir=run_dir,
                model="fake-model",
                python_bin=sys.executable,
                resume=True,
                condition_runner=successful_fake(resumed_calls),
            )
            self.assertEqual(summary.status, "complete")
            self.assertEqual(resumed_calls[0], "singleturn.medium")
            self.assertNotIn("singleturn.easy", resumed_calls)
            with self.assertRaisesRegex(RunArtifactError, "cannot be resumed"):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    resume=True,
                    condition_runner=successful_fake([]),
                )

    def test_suite_resume_recovers_pending_seal_without_reinferring_first_result(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_root = write_prepared_bundle(root)
            run_dir = root / "run"
            initially_inferred = []

            def leave_pending_seal(invocation, log):
                records = [
                    json.loads(line)
                    for line in invocation.prepared_path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line
                ]

                def provider(record, request):
                    initially_inferred.append(record["instance_id"])
                    return "GetBanks()"

                first_result = asyncio.run(
                    run_records(
                        records,
                        inference=provider,
                        model_name="fake-model",
                        tools_schema=[],
                    )
                )[0]
                appended = (
                    json.dumps(
                        first_result,
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                ).encode("utf-8")
                seal_path = checkpoint_integrity_path(
                    invocation.checkpoint_path
                )
                seal_path.write_bytes(
                    _pending_checkpoint_seal_bytes(b"", appended)
                )
                return 9

            with self.assertRaisesRegex(RunArtifactError, "exit code 9"):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    condition_runner=leave_pending_seal,
                )

            checkpoint = (
                run_dir
                / "predictions/singleturn/easy/checkpoint.jsonl"
            )
            seal_path = checkpoint_integrity_path(checkpoint)
            self.assertFalse(checkpoint.exists())
            self.assertTrue(seal_path.is_file())
            self.assertEqual(len(initially_inferred), 1)

            resume_flags = []
            resume_provider_calls = []
            fallback = successful_fake([])

            def recover_condition(invocation, log):
                if invocation.label != "singleturn.easy":
                    return fallback(invocation, log)
                lower_resume = "--resume" in invocation.command
                resume_flags.append(lower_resume)

                def provider(record, request):
                    resume_provider_calls.append(record["instance_id"])
                    return "GetBanks()"

                identity = os.stat(run_dir, follow_symlinks=False)
                run_prepared_file(
                    prepared_path=invocation.prepared_path,
                    manifest_path=prepared_root / "manifest.json",
                    output_path=invocation.output_path,
                    checkpoint_path=invocation.checkpoint_path,
                    tools_schema=[],
                    model_name="fake-model",
                    inference=provider,
                    resume=lower_resume,
                    output_root=run_dir,
                    expected_output_root_identity=(
                        identity.st_dev,
                        identity.st_ino,
                    ),
                )
                return 0

            summary = run_suite(
                repo_root=ROOT,
                prepared_root=prepared_root,
                run_dir=run_dir,
                model="fake-model",
                python_bin=sys.executable,
                resume=True,
                condition_runner=recover_condition,
            )
            self.assertEqual(summary.status, "complete")
            self.assertEqual(resume_flags, [True])
            self.assertEqual(resume_provider_calls, [])
            self.assertTrue(checkpoint.is_file())
            self.assertTrue(seal_path.is_file())

    def test_suite_resume_rejects_symlinked_checkpoint_integrity_seal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_root = write_prepared_bundle(root)
            run_dir = root / "run"
            observed_seal_paths = []

            def leave_seal_only(invocation, log):
                seal_path = checkpoint_integrity_path(
                    invocation.checkpoint_path
                )
                seal_path.write_text("{}\n", encoding="utf-8")
                observed_seal_paths.append(seal_path)
                return 9

            with self.assertRaisesRegex(RunArtifactError, "exit code 9"):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    condition_runner=leave_seal_only,
                )

            seal_path = observed_seal_paths[0]
            outside_seal = root / "outside-integrity.json"
            seal_path.rename(outside_seal)
            seal_path.symlink_to(outside_seal)
            calls = []
            with self.assertRaisesRegex(RunArtifactError, "non-symlink regular file"):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    resume=True,
                    condition_runner=successful_fake(calls),
                )
            self.assertEqual(calls, [])
            self.assertEqual(outside_seal.read_text(encoding="utf-8"), "{}\n")

    def test_provider_error_status_counts_and_partial_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_root = write_prepared_bundle(root)
            run_dir = root / "run"
            initial_calls = []
            first_runner = provider_status_fake(
                initial_calls,
                lambda label: "error" if label == "singleturn.easy" else "ok",
            )

            def fail_second(invocation, log):
                if invocation.label == "singleturn.medium":
                    initial_calls.append(invocation.label)
                    return 9
                return first_runner(invocation, log)

            with self.assertRaisesRegex(RunArtifactError, "exit code 9"):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    condition_runner=fail_second,
                )
            failed = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(
                failed["conditions"]["singleturn.easy"]["status"],
                "all_provider_errors",
            )
            self.assertEqual(failed["provider_error_count"], 1)

            resumed_calls = []
            summary = run_suite(
                repo_root=ROOT,
                prepared_root=prepared_root,
                run_dir=run_dir,
                model="fake-model",
                python_bin=sys.executable,
                resume=True,
                condition_runner=successful_fake(resumed_calls),
            )
            self.assertEqual(summary.status, "complete_with_provider_errors")
            self.assertEqual(summary.prediction_count, 6)
            self.assertEqual(summary.provider_ok_count, 5)
            self.assertEqual(summary.provider_error_count, 1)
            self.assertNotIn("singleturn.easy", resumed_calls)
            status = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(status["status"], "complete_with_provider_errors")
            self.assertEqual(status["provider_error_count"], 1)
            with self.assertRaisesRegex(RunArtifactError, "cannot be resumed"):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    resume=True,
                    condition_runner=successful_fake([]),
                )

    def test_all_provider_errors_is_terminal_and_never_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_root = write_prepared_bundle(root)
            run_dir = root / "run"
            summary = run_suite(
                repo_root=ROOT,
                prepared_root=prepared_root,
                run_dir=run_dir,
                model="fake-model",
                python_bin=sys.executable,
                condition_runner=provider_status_fake(
                    [], lambda label: "error"
                ),
            )
            self.assertEqual(summary.status, "all_provider_errors")
            self.assertEqual(summary.prediction_count, 6)
            self.assertEqual(summary.provider_ok_count, 0)
            self.assertEqual(summary.provider_error_count, 6)
            status = json.loads((run_dir / "status.json").read_text())
            self.assertEqual(status["status"], "all_provider_errors")
            self.assertTrue(
                all(
                    condition["status"] == "all_provider_errors"
                    for condition in status["conditions"].values()
                )
            )
            with self.assertRaisesRegex(RunArtifactError, "cannot be resumed"):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    resume=True,
                    condition_runner=successful_fake([]),
                )

    def test_tampered_prepared_input_fails_before_run_directory_creation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            prepared_root = write_prepared_bundle(temporary_root)
            with (prepared_root / "singleturn/easy.jsonl").open("a", encoding="utf-8") as handle:
                handle.write("{}\n")
            run_dir = temporary_root / "run"
            with self.assertRaisesRegex(RunArtifactError, "invalid prepared condition"):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    condition_runner=successful_fake([]),
                )
            self.assertFalse(run_dir.exists())

    def test_resume_rejects_tampered_tool_schema_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepared_root = write_prepared_bundle(root)
            run_dir = root / "run"
            successful = successful_fake([])

            def fail_second(invocation, log):
                if invocation.label == "singleturn.medium":
                    return 9
                return successful(invocation, log)

            with self.assertRaisesRegex(RunArtifactError, "exit code 9"):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    condition_runner=fail_second,
                )
            snapshot = run_dir / "tool_schemas/singleturn.json"
            snapshot.write_bytes(snapshot.read_bytes() + b"\n")
            with self.assertRaisesRegex(RunArtifactError, "tool schema snapshot"):
                run_suite(
                    repo_root=ROOT,
                    prepared_root=prepared_root,
                    run_dir=run_dir,
                    model="fake-model",
                    python_bin=sys.executable,
                    resume=True,
                    condition_runner=successful_fake([]),
                )

    def test_resume_strictly_decodes_all_canonical_json_artifacts(self) -> None:
        mutations = {
            "resolved_config.json": lambda payload: payload.replace(
                b'  "format_version": 1,',
                b'  "format_version": 1,\n  "format_version": 1,',
                1,
            ),
            "status.json": lambda payload: payload.replace(
                b'"status": "complete"',
                b'"status": "complete", "sta\\u0074us": "complete"',
                1,
            ),
            "metrics/status.json": lambda payload: payload.replace(
                b"{\n",
                b'{\n  "probe": NaN,\n',
                1,
            ),
            "dataset_manifest.json": lambda payload: payload.replace(
                b'"count": 1',
                b'"count": 1, "count": 1',
                1,
            ),
        }
        for relative, mutate in mutations.items():
            with self.subTest(relative=relative):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    prepared_root = write_prepared_bundle(root)
                    run_dir = root / "run"
                    successful = successful_fake([])

                    def fail_second(invocation, log):
                        if invocation.label == "singleturn.medium":
                            return 9
                        return successful(invocation, log)

                    with self.assertRaisesRegex(RunArtifactError, "exit code 9"):
                        run_suite(
                            repo_root=ROOT,
                            prepared_root=prepared_root,
                            run_dir=run_dir,
                            model="fake-model",
                            python_bin=sys.executable,
                            condition_runner=fail_second,
                        )
                    artifact = run_dir / relative
                    artifact.write_bytes(mutate(artifact.read_bytes()))
                    with self.assertRaisesRegex(
                        RunArtifactError,
                        "duplicate object key|non-finite number",
                    ):
                        run_suite(
                            repo_root=ROOT,
                            prepared_root=prepared_root,
                            run_dir=run_dir,
                            model="fake-model",
                            python_bin=sys.executable,
                            resume=True,
                            condition_runner=successful_fake([]),
                        )

    def test_parent_strictly_decodes_canonical_tool_schemas(self) -> None:
        cases = (
            b'[{"function":{"name":"first","na\\u006de":"second"}}]\n',
            b'[{"function":{"score":NaN}}]\n',
            b'[{"function":{"score":Infinity}}]\n',
            b'[{"function":{"score":-Infinity}}]\n',
        )
        for index, payload in enumerate(cases):
            with self.subTest(index=index):
                admitted = AdmittedFile(
                    ROOT / "configs/schemas/mix600-v1/schema_easy.json",
                    payload,
                )
                with mock.patch.object(
                    run_artifacts,
                    "admit_regular_file",
                    return_value=admitted,
                ):
                    with self.assertRaisesRegex(
                        RunArtifactError,
                        "duplicate object key|non-finite number",
                    ):
                        run_artifacts._admit_tool_schemas(ROOT)

    def test_dry_run_has_no_side_effects_and_never_calls_condition_runner(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temporary_root = Path(temporary)
            prepared_root = write_prepared_bundle(temporary_root)
            run_dir = temporary_root / "run"
            calls = []
            output = io.StringIO()
            summary = run_suite(
                repo_root=ROOT,
                prepared_root=prepared_root,
                run_dir=run_dir,
                model="fake-model",
                python_bin=sys.executable,
                dry_run=True,
                condition_runner=successful_fake(calls),
                output=output,
            )
            self.assertEqual(summary.status, "dry-run")
            self.assertEqual(calls, [])
            self.assertFalse(run_dir.exists())
            self.assertEqual(
                [line.split("]", 1)[0] + "]" for line in output.getvalue().splitlines()],
                [f"[{turn}.{difficulty}]" for turn, difficulty in CONDITIONS],
            )


if __name__ == "__main__":
    unittest.main()
