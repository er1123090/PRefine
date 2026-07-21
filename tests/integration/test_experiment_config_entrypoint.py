from __future__ import annotations

import contextlib
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from exp7.experiments import (
    ExperimentConfigError,
    load_experiment_config,
    parse_experiment_config_bytes,
)
from exp7.provenance import admission
from exp7.provenance.admission import FileAdmissionError
from exp7.provenance.run_artifacts import SuiteSummary
import run as run_cli
import run_suite as suite_cli


CONFIG_RELATIVE = "configs/experiments/vanilla_mix600.json"
CONFIG_PATH = ROOT / CONFIG_RELATIVE
EXPECTED_CONDITIONS = (
    ("singleturn", "easy"),
    ("singleturn", "medium"),
    ("singleturn", "hard"),
    ("multiturn", "easy"),
    ("multiturn", "medium"),
    ("multiturn", "hard"),
)


def canonical_document() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def write_config(directory: Path, document: dict) -> str:
    path = directory / "config.json"
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path.relative_to(ROOT).as_posix()


def write_config_bytes(directory: Path, payload: bytes) -> str:
    path = directory / "config.json"
    path.write_bytes(payload)
    return path.relative_to(ROOT).as_posix()


class ExperimentConfigEntrypointTests(unittest.TestCase):
    def test_canonical_config_is_bound_to_exact_bytes_and_six_conditions(self) -> None:
        config = load_experiment_config(ROOT, CONFIG_RELATIVE)

        payload = CONFIG_PATH.read_bytes()
        self.assertEqual(config.config_bytes, payload)
        self.assertEqual(config.config_sha256, hashlib.sha256(payload).hexdigest())
        self.assertEqual(config.config_relative_path, CONFIG_RELATIVE)
        self.assertEqual(config.experiment_id, "vanilla_mix600")
        self.assertEqual(config.method_id, "vanilla_llm")
        self.assertEqual(config.provider.model, "gpt-5-mini")
        self.assertEqual(config.conditions, EXPECTED_CONDITIONS)
        self.assertEqual(config.evaluator_variant, "exp6_slot_value_or_v1")
        self.assertEqual(config.python_bin, "python")
        self.assertEqual(
            config.dataset.prepared_root,
            ROOT / "artifacts/prepared/mix600-v1",
        )
        extra = config.runner_extra_args()
        self.assertIn(config.config_sha256, extra)
        self.assertIn(config.config_relative_path, extra)
        self.assertIn(config.evaluator_variant, extra)

    def test_config_path_rejects_symlink_absolute_historical_and_escape(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".experiment-config-test-",
            dir=ROOT,
        ) as temporary:
            directory = Path(temporary)
            target = directory / "target.json"
            target.write_bytes(CONFIG_PATH.read_bytes())
            link = directory / "link.json"
            link.symlink_to(target)
            relative_link = link.relative_to(ROOT).as_posix()
            with self.assertRaisesRegex(ExperimentConfigError, "symlink"):
                load_experiment_config(ROOT, relative_link)

        invalid_paths = (
            str(CONFIG_PATH),
            "../experiments7/configs/experiments/vanilla_mix600.json",
            "experiments4/config.json",
        )
        for value in invalid_paths:
            with self.subTest(path=value):
                with self.assertRaises(ExperimentConfigError):
                    load_experiment_config(ROOT, value)

    def test_config_mutation_during_admission_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".experiment-config-test-",
            dir=ROOT,
        ) as temporary:
            directory = Path(temporary)
            relative = write_config(directory, canonical_document())
            path = ROOT / relative
            real_fstat = admission.os.fstat
            target = os.stat(path, follow_symlinks=False)
            leaf_observations = 0

            def mutate_on_final_fstat(descriptor):
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
                    path.write_text("{}\n", encoding="utf-8")
                return value

            with mock.patch.object(
                admission.os,
                "fstat",
                side_effect=mutate_on_final_fstat,
            ):
                with self.assertRaisesRegex(
                    ExperimentConfigError,
                    "changed while being read",
                ):
                    load_experiment_config(ROOT, relative)

    def test_admitted_config_bytes_can_be_reparsed_without_reopening_source(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix=".experiment-config-test-",
            dir=ROOT,
        ) as temporary:
            directory = Path(temporary)
            relative = write_config(directory, canonical_document())
            admitted = load_experiment_config(ROOT, relative)
            (ROOT / relative).unlink()
            reparsed = parse_experiment_config_bytes(
                ROOT,
                relative,
                admitted.config_bytes,
                expected_sha256=admitted.config_sha256,
            )
            self.assertEqual(reparsed.config_bytes, admitted.config_bytes)
            self.assertEqual(reparsed.config_sha256, admitted.config_sha256)

    def test_schema_rejects_unknown_or_invalid_values_fail_closed(self) -> None:
        invalid: list[tuple[str, dict, str]] = []

        document = canonical_document()
        document["unexpected"] = True
        invalid.append(("unknown top-level key", document, "unknown keys"))

        document = canonical_document()
        document["provider"]["unexpected"] = True
        invalid.append(("unknown provider key", document, "unknown keys"))

        document = canonical_document()
        document["method"]["method_id"] = "rag"
        invalid.append(("unknown method", document, "unknown method"))

        document = canonical_document()
        document["conditions"] = document["conditions"][:-1]
        invalid.append(("missing condition", document, "all six canonical"))

        document = canonical_document()
        document["conditions"][-1] = deepcopy(document["conditions"][0])
        invalid.append(("duplicate condition", document, "all six canonical"))

        document = canonical_document()
        document["conditions"][0], document["conditions"][1] = (
            document["conditions"][1],
            document["conditions"][0],
        )
        invalid.append(("reordered conditions", document, "in order"))

        document = canonical_document()
        document["provider"]["api_key_env"] = "BAD-NAME"
        invalid.append(("invalid provider env", document, "api_key_env"))

        document = canonical_document()
        document["provider"]["base_url"] = "ftp://provider.invalid"
        invalid.append(("invalid provider URL", document, "HTTP"))

        document = canonical_document()
        document["provider"]["temperature"] = 3
        invalid.append(("invalid temperature", document, "supported range"))

        document = canonical_document()
        document["provider"]["max_tokens"] = 0
        invalid.append(("invalid max tokens", document, "positive integer"))

        document = canonical_document()
        document["schema_version"] = True
        invalid.append(("boolean schema version", document, "schema_version"))

        document = canonical_document()
        document["dataset"]["format_version"] = True
        invalid.append(("boolean dataset version", document, "format_version"))

        document = canonical_document()
        document["provider"]["temperature"] = float("nan")
        invalid.append(("NaN temperature", document, "finite"))

        document = canonical_document()
        document["provider"]["timeout"] = float("inf")
        invalid.append(("infinite timeout", document, "finite"))

        document = canonical_document()
        document["dataset"]["prepared_root"] = "/tmp/prepared"
        invalid.append(("absolute dataset path", document, "repository-relative"))

        document = canonical_document()
        document["dataset"]["prepared_root"] = "experiments6/prepared"
        invalid.append(("historical dataset path", document, "historical"))

        document = canonical_document()
        document["output"]["root"] = "../outside"
        invalid.append(("escaping output path", document, "repository-relative"))

        document = canonical_document()
        document["evaluator"]["variant"] = "new_unverified_evaluator"
        invalid.append(("unknown evaluator", document, "unsupported evaluator"))

        with tempfile.TemporaryDirectory(
            prefix=".experiment-config-test-",
            dir=ROOT,
        ) as temporary:
            directory = Path(temporary)
            for index, (label, value, message) in enumerate(invalid):
                case_directory = directory / str(index)
                case_directory.mkdir()
                relative = write_config(case_directory, value)
                with self.subTest(case=label):
                    with self.assertRaisesRegex(ExperimentConfigError, message):
                        load_experiment_config(ROOT, relative)

    def test_schema_rejects_duplicate_decoded_keys_at_every_depth(self) -> None:
        payload = CONFIG_PATH.read_bytes()
        cases = (
            payload.replace(
                b'  "schema_version": 1,',
                b'  "schema_version": 1,\n  "\\u0073chema_version": 1,',
                1,
            ),
            payload.replace(
                b'    "model": "gpt-5-mini",',
                b'    "model": "gpt-5-mini",\n    "\\u006dodel": "other",',
                1,
            ),
        )
        with tempfile.TemporaryDirectory(
            prefix=".experiment-config-test-",
            dir=ROOT,
        ) as temporary:
            directory = Path(temporary)
            for index, value in enumerate(cases):
                case_directory = directory / str(index)
                case_directory.mkdir()
                relative = write_config_bytes(case_directory, value)
                with self.subTest(index=index):
                    with self.assertRaisesRegex(ExperimentConfigError, "duplicate"):
                        load_experiment_config(ROOT, relative)

    def test_run_py_exact_config_dispatch_uses_only_config_values(self) -> None:
        returned = SuiteSummary("complete", Path("/tmp/fake-run"), 6)
        stdout = io.StringIO()
        with mock.patch.object(
            suite_cli,
            "run_suite",
            return_value=returned,
        ) as invoke:
            with contextlib.redirect_stdout(stdout):
                result = run_cli.main(["--config", CONFIG_RELATIVE])

        self.assertEqual(result, 0)
        invoke.assert_called_once()
        arguments = invoke.call_args.kwargs
        config = load_experiment_config(ROOT, CONFIG_RELATIVE)
        self.assertEqual(arguments["repo_root"], ROOT)
        self.assertEqual(arguments["prepared_root"], config.dataset.prepared_root)
        self.assertEqual(arguments["model"], config.provider.model)
        self.assertEqual(arguments["python_bin"], config.python_bin)
        self.assertEqual(arguments["api_key_env"], config.provider.api_key_env)
        self.assertEqual(arguments["extra_args"], config.runner_extra_args())
        self.assertEqual(arguments["experiment_config_bytes"], config.config_bytes)
        self.assertEqual(
            arguments["experiment_config_sha256"], config.config_sha256
        )
        self.assertEqual(arguments["evaluator_variant"], config.evaluator_variant)
        self.assertEqual(arguments["output_policy"], "config_root")
        self.assertEqual(
            arguments["experiment_config_relative_path"],
            config.config_relative_path,
        )
        self.assertFalse(arguments["resume"])
        self.assertFalse(arguments["dry_run"])
        self.assertEqual(arguments["run_dir"].parent, config.output.root)
        self.assertRegex(
            arguments["run_dir"].name,
            r"^vanilla-mix600-\d{8}T\d{6}Z$",
        )
        self.assertIn('"completed_conditions": 6', stdout.getvalue())

    def test_config_mode_requires_explicit_external_output_opt_in(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "existing-run"
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as rejected:
                    suite_cli.main(
                        [
                            "--config",
                            CONFIG_RELATIVE,
                            "--run-dir",
                            str(run_dir),
                            "--resume",
                            "--dry-run",
                        ]
                    )
            self.assertEqual(rejected.exception.code, 2)

            with mock.patch.object(
                suite_cli,
                "run_suite",
                return_value=SuiteSummary("dry-run", run_dir, 0),
            ) as invoke:
                result = suite_cli.main(
                    [
                        "--config",
                        CONFIG_RELATIVE,
                        "--run-dir",
                        str(run_dir),
                        "--allow-external-output",
                        "--resume",
                        "--dry-run",
                    ]
                )
            self.assertEqual(result, 0)
            self.assertEqual(invoke.call_args.kwargs["run_dir"], run_dir)
            self.assertTrue(invoke.call_args.kwargs["resume"])
            self.assertTrue(invoke.call_args.kwargs["dry_run"])
            self.assertEqual(
                invoke.call_args.kwargs["output_policy"],
                "external_opt_in",
            )

        with self.assertRaisesRegex(ValueError, "explicit --run-dir"):
            suite_cli.main(["--config", CONFIG_RELATIVE, "--resume"])

        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                suite_cli.main(
                    ["--config", CONFIG_RELATIVE, "--model", "hidden-override"]
                )

    def test_config_cli_returns_nonzero_for_all_provider_errors(self) -> None:
        with mock.patch.object(
            suite_cli,
            "run_suite",
            return_value=SuiteSummary(
                "all_provider_errors",
                Path("/tmp/fake-run"),
                6,
                prediction_count=6,
                provider_error_count=6,
            ),
        ):
            self.assertEqual(run_cli.main(["--config", CONFIG_RELATIVE]), 1)

    def test_lower_level_cli_reports_actual_provider_status(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema = root / "schema.json"
            schema.write_text("[]\n", encoding="utf-8")
            output = root / "predictions.json"
            checkpoint = root / "checkpoint.jsonl"
            summary = SimpleNamespace(
                checkpoint_path=checkpoint,
                error_count=1,
                inferred_count=1,
                input_count=1,
                output_path=output,
                resumed_count=0,
                status="all_provider_errors",
            )
            stdout = io.StringIO()
            with mock.patch.object(run_cli, "run_prepared_file", return_value=summary):
                with contextlib.redirect_stdout(stdout):
                    result = run_cli.main(
                        [
                            "--prepared-jsonl",
                            str(root / "prepared.jsonl"),
                            "--manifest",
                            str(root / "manifest.json"),
                            "--tools-schema",
                            str(schema),
                            "--tools-schema-sha256",
                            hashlib.sha256(schema.read_bytes()).hexdigest(),
                            "--output",
                            str(output),
                            "--checkpoint",
                            str(checkpoint),
                            "--model",
                            "fake-model",
                        ]
                    )
            self.assertEqual(result, 0)
            self.assertEqual(json.loads(stdout.getvalue())["status"], "all_provider_errors")

            with self.assertRaisesRegex(ValueError, "tools schema SHA-256"):
                run_cli.main(
                    [
                        "--prepared-jsonl",
                        str(root / "prepared.jsonl"),
                        "--manifest",
                        str(root / "manifest.json"),
                        "--tools-schema",
                        str(schema),
                        "--tools-schema-sha256",
                        "0" * 64,
                        "--output",
                        str(output),
                        "--checkpoint",
                        str(checkpoint),
                        "--model",
                        "fake-model",
                    ]
                )

    def test_lower_level_cli_strictly_decodes_tool_schema_and_rejects_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "predictions.json"
            checkpoint = root / "checkpoint.jsonl"
            summary = SimpleNamespace(
                checkpoint_path=checkpoint,
                error_count=0,
                inferred_count=1,
                input_count=1,
                output_path=output,
                resumed_count=0,
                status="complete",
            )

            def arguments(schema: Path) -> list[str]:
                return [
                    "--prepared-jsonl",
                    str(root / "prepared.jsonl"),
                    "--manifest",
                    str(root / "manifest.json"),
                    "--tools-schema",
                    str(schema),
                    "--tools-schema-sha256",
                    hashlib.sha256(schema.read_bytes()).hexdigest(),
                    "--output",
                    str(output),
                    "--checkpoint",
                    str(checkpoint),
                    "--model",
                    "fake-model",
                ]

            cases = (
                b'[{"function":{"name":"first","na\\u006de":"second"}}]\n',
                b'[{"function":{"score":NaN}}]\n',
                b'[{"function":{"score":Infinity}}]\n',
                b'[{"function":{"score":-Infinity}}]\n',
            )
            with mock.patch.object(run_cli, "run_prepared_file", return_value=summary):
                for index, payload in enumerate(cases):
                    schema = root / f"schema-{index}.json"
                    schema.write_bytes(payload)
                    with self.subTest(index=index):
                        with self.assertRaisesRegex(
                            ValueError,
                            "duplicate object key|non-finite number",
                        ):
                            run_cli.main(arguments(schema))

            target = root / "schema-target.json"
            target.write_text("[]\n", encoding="utf-8")
            link = root / "schema-link.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(FileAdmissionError, "non-symlink regular file"):
                run_cli.main(
                    [
                        "--prepared-jsonl",
                        str(root / "prepared.jsonl"),
                        "--manifest",
                        str(root / "manifest.json"),
                        "--tools-schema",
                        str(link),
                        "--output",
                        str(output),
                        "--checkpoint",
                        str(checkpoint),
                        "--model",
                        "fake-model",
                    ]
                )

            real_parent = root / "schema-parent"
            real_parent.mkdir()
            parent_schema = real_parent / "schema.json"
            parent_schema.write_text("[]\n", encoding="utf-8")
            parent_link = root / "schema-parent-link"
            parent_link.symlink_to(real_parent, target_is_directory=True)
            with self.assertRaisesRegex(
                FileAdmissionError,
                "parent must be a non-symlink directory",
            ):
                with mock.patch.object(
                    run_cli,
                    "run_prepared_file",
                    return_value=summary,
                ):
                    run_cli.main(arguments(parent_link / "schema.json"))

    def test_exact_entrypoint_dry_run_emits_six_commands_without_api_or_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "not-created"
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "scripts/run.py",
                    "--config",
                    CONFIG_RELATIVE,
                    "--dry-run",
                    "--run-dir",
                    str(run_dir),
                    "--allow-external-output",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            labels = [
                line.split("]", 1)[0] + "]"
                for line in result.stdout.splitlines()
            ]
            self.assertEqual(
                labels,
                [f"[{turn}.{difficulty}]" for turn, difficulty in EXPECTED_CONDITIONS],
            )
            self.assertEqual(result.stdout.count("scripts/run.py"), 6)
            self.assertEqual(result.stdout.count("--experiment-config-sha256"), 6)
            self.assertEqual(result.stdout.count("schema_easy.json"), 3)
            self.assertEqual(result.stdout.count("schema_all.json"), 3)
            self.assertFalse(run_dir.exists())

    def test_quickstart_acknowledges_every_explicit_config_run_directory(self) -> None:
        quickstart = (ROOT / "docs/quickstart.md").read_text(encoding="utf-8")
        config_blocks = [
            block
            for block in re.findall(r"```bash\n(.*?)```", quickstart, re.DOTALL)
            if "scripts/run.py" in block
            and "--config" in block
            and "--run-dir" in block
        ]
        self.assertEqual(len(config_blocks), 2)
        for block in config_blocks:
            self.assertIn("--allow-external-output", block)


if __name__ == "__main__":
    unittest.main()
