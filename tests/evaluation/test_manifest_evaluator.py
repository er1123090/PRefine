from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from exp7.evaluation import EvaluationError, evaluate_run
from exp7.provenance import admission


VARIANT = "exp6_slot_value_or_v1"
FIXTURE = ROOT / "tests/fixtures/evaluation/historical_style_parity.json"


def _json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode()


def _build_run(root: Path, fixture: dict | None = None) -> tuple[Path, dict]:
    fixture = copy.deepcopy(
        fixture or json.loads(FIXTURE.read_text(encoding="utf-8"))
    )
    repo_root = root / "repo"
    prepared_root = repo_root / "prepared"
    prepared_path = prepared_root / "singleturn/easy.jsonl"
    prepared_path.parent.mkdir(parents=True)
    prepared_payload = b"".join(_json_bytes(row) for row in fixture["prepared"])
    prepared_path.write_bytes(prepared_payload)

    pref_path = repo_root / "config/pref_list.json"
    pref_path.parent.mkdir(parents=True)
    pref_payload = _json_bytes(fixture["preference_slots"])
    pref_path.write_bytes(pref_payload)
    manifest = {
        "dataset_id": "fixture-v1",
        "format_version": 1,
        "inputs": {
            "pref_list": {
                "bytes": len(pref_payload),
                "path": "config/pref_list.json",
                "sha256": hashlib.sha256(pref_payload).hexdigest(),
            }
        },
        "outputs": {
            "singleturn.easy": {
                "bytes": len(prepared_payload),
                "count": len(fixture["prepared"]),
                "path": "singleturn/easy.jsonl",
                "sha256": hashlib.sha256(prepared_payload).hexdigest(),
            }
        },
    }
    manifest_payload = _json_bytes(manifest)
    manifest_path = prepared_root / "manifest.json"
    manifest_path.write_bytes(manifest_payload)

    run_dir = root / "run"
    prediction_path = run_dir / "predictions/singleturn/easy/predictions.json"
    prediction_path.parent.mkdir(parents=True)
    prediction_path.write_bytes(_json_bytes(fixture["predictions"]))
    run_dir.joinpath("dataset_manifest.json").write_bytes(manifest_payload)
    config = {
        "conditions": ["singleturn.easy"],
        "dataset_manifest": str(manifest_path),
        "dataset_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
        "format_version": 1,
        "method": "vanilla_llm",
        "model": "offline-fixture",
        "prepared_root": str(prepared_root),
        "repo_root": str(repo_root),
        "run_dir": str(run_dir),
    }
    run_dir.joinpath("resolved_config.json").write_bytes(_json_bytes(config))
    return run_dir, fixture


def _metrics(run_dir: Path) -> dict:
    return json.loads((run_dir / "metrics/metrics.json").read_text(encoding="utf-8"))


def _input_paths(run_dir: Path) -> dict[str, Path]:
    config = json.loads((run_dir / "resolved_config.json").read_text())
    return {
        "resolved_config": run_dir / "resolved_config.json",
        "run_manifest": run_dir / "dataset_manifest.json",
        "source_manifest": Path(config["dataset_manifest"]),
        "prepared_jsonl": Path(config["prepared_root"]) / "singleturn/easy.jsonl",
        "predictions": run_dir / "predictions/singleturn/easy/predictions.json",
        "preference_metadata": Path(config["repo_root"]) / "config/pref_list.json",
    }


def _replace_bound_manifest(run_dir: Path, payload: bytes) -> None:
    config_path = run_dir / "resolved_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    Path(config["dataset_manifest"]).write_bytes(payload)
    (run_dir / "dataset_manifest.json").write_bytes(payload)
    config["dataset_manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    config_path.write_bytes(_json_bytes(config))


def _race_fstat(target: Path, attack: str):
    real_fstat = admission.os.fstat
    triggered = {"value": False}

    def adversarial(descriptor):
        value = real_fstat(descriptor)
        if triggered["value"]:
            return value
        try:
            opened = Path(os.readlink(f"/proc/self/fd/{descriptor}"))
        except OSError:
            return value
        if opened != target:
            return value
        triggered["value"] = True
        if attack == "mutation":
            with target.open("ab") as handle:
                handle.write(b" ")
        else:
            replacement = target.with_name(f".{target.name}.replacement")
            replacement.write_bytes(target.read_bytes())
            os.replace(replacement, target)
        return value

    return triggered, adversarial


class ManifestEvaluatorTests(unittest.TestCase):
    def test_historical_style_parity_and_prediction_gt_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, fixture = _build_run(Path(temporary))
            summary = evaluate_run(run_dir=run_dir, variant=VARIANT)
            metrics = _metrics(run_dir)
            condition = metrics["conditions"]["singleturn.easy"]

            self.assertEqual(summary.row_count, 2)
            self.assertEqual(condition["overall"], fixture["expected"]["overall"])
            self.assertEqual(
                condition["preference"]["exact_match"],
                fixture["expected"]["preference_exact_match"],
            )
            self.assertEqual(
                condition["preference"]["slot_value"],
                fixture["expected"]["preference_slot_value"],
            )
            self.assertEqual(
                condition["non_preference"], fixture["expected"]["non_preference"]
            )
            self.assertEqual(metrics["counts"]["parse_failure_count"], 0)
            self.assertTrue(metrics["provenance"]["dataset_manifest_sha256"])
            self.assertTrue(
                metrics["provenance"]["predictions"]["singleturn.easy"]["sha256"]
            )
            self.assertTrue(metrics["evaluator"]["code_sha256"])
            self.assertTrue(metrics["evaluator"]["config_sha256"])

    def test_strict_prediction_join_rejects_drift(self) -> None:
        for drift in ("duplicate", "missing", "extra", "condition"):
            with self.subTest(drift=drift), tempfile.TemporaryDirectory() as temporary:
                fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
                if drift == "duplicate":
                    fixture["predictions"][1]["instance_id"] = fixture["predictions"][0][
                        "instance_id"
                    ]
                elif drift == "missing":
                    fixture["predictions"].pop()
                elif drift == "extra":
                    extra = copy.deepcopy(fixture["predictions"][0])
                    extra["instance_id"] = "fixture:singleturn:easy:extra"
                    fixture["predictions"].append(extra)
                else:
                    fixture["predictions"][0]["difficulty"] = "hard"
                run_dir, _ = _build_run(Path(temporary), fixture)
                with self.assertRaises(EvaluationError):
                    evaluate_run(run_dir=run_dir, variant=VARIANT)
                self.assertFalse((run_dir / "metrics/metrics.json").exists())

    def test_strict_json_rejects_duplicate_keys_at_any_depth(self) -> None:
        cases = ("resolved_config", "nested_manifest", "prediction_status")
        for target in cases:
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                run_dir, _ = _build_run(Path(temporary))
                if target == "resolved_config":
                    path = run_dir / "resolved_config.json"
                    payload = path.read_text(encoding="utf-8").replace(
                        '"method": "vanilla_llm"',
                        '"method": "ignored", "method": "vanilla_llm"',
                        1,
                    ).encode("utf-8")
                    path.write_bytes(payload)
                elif target == "nested_manifest":
                    path = run_dir / "dataset_manifest.json"
                    payload = path.read_text(encoding="utf-8").replace(
                        '"path": "singleturn/easy.jsonl"',
                        '"path": "../ignored.jsonl", '
                        '"path": "singleturn/easy.jsonl"',
                        1,
                    ).encode("utf-8")
                    _replace_bound_manifest(run_dir, payload)
                else:
                    path = run_dir / "predictions/singleturn/easy/predictions.json"
                    payload = path.read_text(encoding="utf-8").replace(
                        '[{"difficulty"',
                        '[{"status": "error", "status": "ok", "difficulty"',
                        1,
                    ).encode("utf-8")
                    path.write_bytes(payload)

                with self.assertRaisesRegex(EvaluationError, "duplicate object key"):
                    evaluate_run(run_dir=run_dir, variant=VARIANT)
                self.assertFalse((run_dir / "metrics/metrics.json").exists())

    def test_strict_json_rejects_nonfinite_numbers(self) -> None:
        cases = {
            "resolved_config": "NaN",
            "manifest": "Infinity",
            "predictions": "-Infinity",
        }
        for target, literal in cases.items():
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                run_dir, _ = _build_run(Path(temporary))
                if target == "resolved_config":
                    path = run_dir / "resolved_config.json"
                    payload = path.read_bytes().replace(
                        b"{", f'{{"strict_probe": {literal}, '.encode("ascii"), 1
                    )
                    path.write_bytes(payload)
                elif target == "manifest":
                    path = run_dir / "dataset_manifest.json"
                    payload = path.read_bytes().replace(
                        b"{", f'{{"strict_probe": {literal}, '.encode("ascii"), 1
                    )
                    _replace_bound_manifest(run_dir, payload)
                else:
                    path = run_dir / "predictions/singleturn/easy/predictions.json"
                    payload = path.read_bytes().replace(
                        b"[{", f'[{{"strict_probe": {literal}, '.encode("ascii"), 1
                    )
                    path.write_bytes(payload)

                with self.assertRaisesRegex(EvaluationError, "non-finite number"):
                    evaluate_run(run_dir=run_dir, variant=VARIANT)
                self.assertFalse((run_dir / "metrics/metrics.json").exists())

    def test_strict_json_rejects_float_overflow(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, _ = _build_run(Path(temporary))
            path = run_dir / "predictions/singleturn/easy/predictions.json"
            payload = path.read_bytes().replace(
                b"[{",
                b'[{"strict_probe": 1e400, ',
                1,
            )
            path.write_bytes(payload)
            with self.assertRaisesRegex(EvaluationError, "non-finite number"):
                evaluate_run(run_dir=run_dir, variant=VARIANT)
            self.assertFalse((run_dir / "metrics/metrics.json").exists())

    def test_strict_json_rejects_trailing_documents_without_metrics(self) -> None:
        for target in ("resolved_config", "predictions"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                run_dir, _ = _build_run(Path(temporary))
                path = (
                    run_dir / "resolved_config.json"
                    if target == "resolved_config"
                    else run_dir / "predictions/singleturn/easy/predictions.json"
                )
                path.write_bytes(path.read_bytes() + b"{}\n")
                with self.assertRaisesRegex(EvaluationError, "cannot decode"):
                    evaluate_run(run_dir=run_dir, variant=VARIANT)
                self.assertFalse((run_dir / "metrics/metrics.json").exists())

    def test_manifest_bound_inputs_reject_tampering(self) -> None:
        for target in ("manifest", "prepared", "preference"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                run_dir, _ = _build_run(Path(temporary))
                config = json.loads((run_dir / "resolved_config.json").read_text())
                if target == "manifest":
                    path = run_dir / "dataset_manifest.json"
                elif target == "prepared":
                    path = Path(config["prepared_root"]) / "singleturn/easy.jsonl"
                else:
                    path = Path(config["repo_root"]) / "config/pref_list.json"
                path.write_bytes(path.read_bytes() + b" ")
                with self.assertRaises(EvaluationError):
                    evaluate_run(run_dir=run_dir, variant=VARIANT)


    def test_all_evaluator_inputs_reject_leaf_symlinks(self) -> None:
        targets = (
            "resolved_config",
            "run_manifest",
            "source_manifest",
            "prepared_jsonl",
            "predictions",
            "preference_metadata",
        )
        for target_name in targets:
            with self.subTest(target=target_name), tempfile.TemporaryDirectory() as temporary:
                run_dir, _ = _build_run(Path(temporary))
                target = _input_paths(run_dir)[target_name]
                real = target.with_name(f"{target.name}.real")
                target.rename(real)
                target.symlink_to(real)
                with self.assertRaisesRegex(EvaluationError, "non-symlink regular file"):
                    evaluate_run(run_dir=run_dir, variant=VARIANT)
                self.assertFalse((run_dir / "metrics/metrics.json").exists())


    def test_run_prepared_and_member_directory_symlinks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, _ = _build_run(root)
            real_run = root / "real-run"
            run_dir.rename(real_run)
            run_dir.symlink_to(real_run, target_is_directory=True)
            with self.assertRaisesRegex(EvaluationError, "non-symlink directory"):
                evaluate_run(run_dir=run_dir, variant=VARIANT)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, _ = _build_run(root)
            config = json.loads((run_dir / "resolved_config.json").read_text())
            prepared_root = Path(config["prepared_root"])
            real_prepared = prepared_root.with_name("prepared-real")
            prepared_root.rename(real_prepared)
            prepared_root.symlink_to(real_prepared, target_is_directory=True)
            with self.assertRaisesRegex(EvaluationError, "non-symlink directory"):
                evaluate_run(run_dir=run_dir, variant=VARIANT)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, _ = _build_run(root)
            predictions = run_dir / "predictions"
            real_predictions = run_dir / "predictions-real"
            predictions.rename(real_predictions)
            predictions.symlink_to(real_predictions, target_is_directory=True)
            with self.assertRaisesRegex(
                EvaluationError, "parent must be a non-symlink directory"
            ):
                evaluate_run(run_dir=run_dir, variant=VARIANT)

    def test_run_ancestor_and_metrics_output_symlinks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_root = root / "real"
            run_dir, _ = _build_run(real_root)
            alias = root / "alias"
            alias.symlink_to(real_root, target_is_directory=True)
            with self.assertRaisesRegex(
                EvaluationError,
                "parent must be a non-symlink directory",
            ):
                evaluate_run(
                    run_dir=alias / run_dir.relative_to(real_root),
                    variant=VARIANT,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir, _ = _build_run(root)
            outside = root / "outside"
            outside.mkdir()
            (run_dir / "metrics").symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(
                EvaluationError,
                "non-symlink directory|parent must be a non-symlink directory",
            ):
                evaluate_run(run_dir=run_dir, variant=VARIANT)
            self.assertEqual(list(outside.iterdir()), [])

    def test_manifest_member_escape_is_rejected_before_admission(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, _ = _build_run(Path(temporary))
            config_path = run_dir / "resolved_config.json"
            config = json.loads(config_path.read_text())
            source_manifest = Path(config["dataset_manifest"])
            manifest = json.loads(source_manifest.read_text())
            manifest["outputs"]["singleturn.easy"]["path"] = "../outside.jsonl"
            payload = _json_bytes(manifest)
            source_manifest.write_bytes(payload)
            (run_dir / "dataset_manifest.json").write_bytes(payload)
            config["dataset_manifest_sha256"] = hashlib.sha256(payload).hexdigest()
            config_path.write_bytes(_json_bytes(config))
            outside = source_manifest.parent.parent / "outside.jsonl"
            outside.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(EvaluationError, "path must be relative"):
                evaluate_run(run_dir=run_dir, variant=VARIANT)
            self.assertFalse((run_dir / "metrics/metrics.json").exists())

    def test_every_input_rejects_in_place_mutation_and_inode_replacement(self) -> None:
        targets = (
            "resolved_config",
            "run_manifest",
            "source_manifest",
            "prepared_jsonl",
            "predictions",
            "preference_metadata",
        )
        for attack in ("mutation", "replacement"):
            for target_name in targets:
                with self.subTest(attack=attack, target=target_name):
                    with tempfile.TemporaryDirectory() as temporary:
                        run_dir, _ = _build_run(Path(temporary))
                        target = _input_paths(run_dir)[target_name]
                        triggered, adversarial = _race_fstat(target, attack)
                        with mock.patch.object(
                            admission.os, "fstat", side_effect=adversarial
                        ):
                            with self.assertRaisesRegex(
                                EvaluationError, "changed while being read"
                            ):
                                evaluate_run(run_dir=run_dir, variant=VARIANT)
                        self.assertTrue(triggered["value"])
                        self.assertFalse((run_dir / "metrics/metrics.json").exists())

    def test_error_row_counts_and_scores_as_empty_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
            fixture["prepared"] = fixture["prepared"][:1]
            fixture["predictions"] = fixture["predictions"][:1]
            fixture["predictions"][0].update(
                {
                    "error": "provider failed",
                    "llm_output": "API_ERROR",
                    "prediction": "API_ERROR",
                    "status": "error",
                }
            )
            run_dir, _ = _build_run(Path(temporary), fixture)
            summary = evaluate_run(run_dir=run_dir, variant=VARIANT)
            condition = _metrics(run_dir)["conditions"]["singleturn.easy"]
            self.assertEqual(summary.parse_failure_count, 1)
            self.assertEqual(summary.row_error_count, 1)
            self.assertEqual(condition["overall"]["tp"], 0)
            self.assertEqual(condition["overall"]["fn"], 2)

    def test_preference_metrics_omitted_without_canonical_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, _ = _build_run(Path(temporary))
            config_path = run_dir / "resolved_config.json"
            config = json.loads(config_path.read_text())
            original = Path(config["dataset_manifest"])
            manifest = json.loads(original.read_text())
            manifest.pop("inputs")
            payload = _json_bytes(manifest)
            original.write_bytes(payload)
            (run_dir / "dataset_manifest.json").write_bytes(payload)
            config["dataset_manifest_sha256"] = hashlib.sha256(payload).hexdigest()
            config_path.write_bytes(_json_bytes(config))

            evaluate_run(run_dir=run_dir, variant=VARIANT)
            aggregate = _metrics(run_dir)["aggregate"]
            self.assertNotIn("preference", aggregate)
            self.assertNotIn("non_preference", aggregate)

    def test_cli_requires_explicit_variant_and_writes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, _ = _build_run(Path(temporary))
            script = ROOT / "scripts/evaluate.py"
            missing = subprocess.run(
                [sys.executable, str(script), "--run-dir", str(run_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(missing.returncode, 2)
            self.assertIn("--variant", missing.stderr)
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--run-dir",
                    str(run_dir),
                    "--variant",
                    VARIANT,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(json.loads(completed.stdout)["status"], "complete")
            self.assertTrue((run_dir / "metrics/metrics.json").is_file())

    def test_unknown_variant_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir, _ = _build_run(Path(temporary))
            with self.assertRaisesRegex(EvaluationError, "unsupported evaluator variant"):
                evaluate_run(run_dir=run_dir, variant="implicit-default")


if __name__ == "__main__":
    unittest.main()
