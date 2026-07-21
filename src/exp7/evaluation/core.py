"""Offline, manifest-bound evaluation for canonical experiment runs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from ..provenance.admission import (
    AdmittedDirectory,
    AdmittedFile,
    FileAdmissionError,
    admit_directory,
    admit_regular_file,
    lexical_absolute,
)
from .parsing import parse_calls


EVALUATOR_VERSION = "1.0.0"
SUPPORTED_VARIANTS = ("exp6_slot_value_or_v1",)


class EvaluationError(RuntimeError):
    """Raised when provenance or strict prediction alignment fails."""


@dataclass(frozen=True)
class EvaluationSummary:
    output_path: Path
    condition_count: int
    row_count: int
    parse_failure_count: int
    row_error_count: int


@dataclass
class _Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "_Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    def as_dict(self) -> dict[str, int | float]:
        precision = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        recall = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        return {
            "f1": f1,
            "fn": self.fn,
            "fp": self.fp,
            "precision": precision,
            "recall": recall,
            "tp": self.tp,
        }


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _admit_file(path: Path, label: str) -> AdmittedFile:
    try:
        return admit_regular_file(path, label=label)
    except FileAdmissionError as exc:
        raise EvaluationError(str(exc)) from exc


def _admit_directory(path: Path, label: str) -> AdmittedDirectory:
    try:
        return admit_directory(path, label=label)
    except FileAdmissionError as exc:
        raise EvaluationError(str(exc)) from exc


def _admit_member(
    root: AdmittedDirectory,
    relative: str,
    label: str,
) -> AdmittedFile:
    _resolve_member(root.path, relative, label)
    try:
        return root.admit_file(relative, label=label)
    except FileAdmissionError as exc:
        raise EvaluationError(str(exc)) from exc


class _StrictJSONError(ValueError):
    """Raised for JSON extensions or ambiguous object decoding."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _StrictJSONError(f"duplicate object key {key!r}")
        value[key] = item
    return value


def _reject_nonfinite_number(value: str) -> Any:
    raise _StrictJSONError(f"non-finite number {value!r}")


def _strict_float(value: str) -> float:
    decoded = float(value)
    if not math.isfinite(decoded):
        raise _StrictJSONError(f"non-finite number {value!r}")
    return decoded


def _decode_json(payload: bytes, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_nonfinite_number,
            parse_float=_strict_float,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _StrictJSONError) as exc:
        raise EvaluationError(f"cannot decode {label}: {exc}") from exc


def _json_records(payload: bytes, label: str) -> list[dict[str, Any]]:
    value = _decode_json(payload, label)
    if isinstance(value, dict):
        for key in ("data", "examples", "items", "predictions"):
            if isinstance(value.get(key), list):
                value = value[key]
                break
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise EvaluationError(f"{label} must contain a JSON array of objects")
    return [dict(row) for row in value]


def _jsonl_records(payload: bytes, label: str) -> list[dict[str, Any]]:
    records = []
    for line_number, raw_line in enumerate(payload.splitlines(), start=1):
        if not raw_line.strip():
            continue
        value = _decode_json(raw_line, f"{label} line {line_number}")
        if not isinstance(value, dict):
            raise EvaluationError(f"{label} line {line_number} is not an object")
        records.append(dict(value))
    return records


def _resolve_member(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise EvaluationError(f"{label} path must be a safe relative path: {relative!r}")
    raw = PurePosixPath(relative)
    if (
        raw.is_absolute()
        or not raw.parts
        or any(part in {"", ".", ".."} for part in raw.parts)
    ):
        raise EvaluationError(f"{label} path must be relative: {relative!r}")
    path = lexical_absolute(root.joinpath(*raw.parts))
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise EvaluationError(f"{label} escapes its artifact root: {relative!r}") from exc
    return path


def _verify_bound_file(
    root: AdmittedDirectory, spec: Mapping[str, Any], label: str
) -> tuple[Path, bytes]:
    path_value, expected_sha = spec.get("path"), spec.get("sha256")
    if not isinstance(path_value, str) or not isinstance(expected_sha, str):
        raise EvaluationError(f"{label} manifest entry lacks path/sha256")
    admitted = _admit_member(root, path_value, label)
    actual_sha = admitted.sha256
    if actual_sha != expected_sha:
        raise EvaluationError(
            f"{label} SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
        )
    expected_bytes = spec.get("bytes")
    if expected_bytes is not None and expected_bytes != len(admitted.payload):
        raise EvaluationError(
            f"{label} byte-size mismatch: expected {expected_bytes}, "
            f"got {len(admitted.payload)}"
        )
    return admitted.path, admitted.payload


def _count(
    ground_truth: Mapping[tuple[str, str], set[str]],
    prediction: Mapping[tuple[str, str], set[str]],
) -> _Counts:
    result = _Counts()
    for key, allowed in ground_truth.items():
        predicted = prediction.get(key, set())
        if predicted & allowed:
            result.tp += 1
        else:
            result.fn += 1
    for key, predicted in prediction.items():
        if key not in ground_truth or not predicted.intersection(ground_truth[key]):
            result.fp += 1
    return result


def _filter_slots(
    values: Mapping[tuple[str, str], set[str]],
    preference_map: Mapping[str, set[str]],
    preference: bool,
) -> dict[tuple[str, str], set[str]]:
    return {
        key: set(items)
        for key, items in values.items()
        if ((key[0] in preference_map and key[1] in preference_map[key[0]]) == preference)
    }


def _prediction_value(row: Mapping[str, Any]) -> tuple[Any, bool]:
    for key in ("prediction", "llm_output", "output", "response"):
        if key in row:
            return row[key], False
    return None, True


def _validate_ids(
    records: Sequence[Mapping[str, Any]], label: str
) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(records):
        instance_id = record.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise EvaluationError(f"{label} row {index} lacks a non-empty instance_id")
        if instance_id in indexed:
            raise EvaluationError(f"duplicate instance_id in {label}: {instance_id}")
        indexed[instance_id] = dict(record)
    return indexed


def _condition_records(
    prepared_root: AdmittedDirectory,
    manifest: Mapping[str, Any],
    label: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or not isinstance(outputs.get(label), dict):
        raise EvaluationError(f"dataset manifest lacks prepared condition {label}")
    spec = outputs[label]
    path, payload = _verify_bound_file(prepared_root, spec, f"prepared {label}")
    records = _jsonl_records(payload, f"prepared {label}")
    if spec.get("count") != len(records):
        raise EvaluationError(
            f"prepared {label} count mismatch: expected {spec.get('count')}, got {len(records)}"
        )
    turn, difficulty = label.split(".", 1)
    dataset_id = manifest.get("dataset_id")
    for index, record in enumerate(records):
        if record.get("turn") != turn or record.get("difficulty") != difficulty:
            raise EvaluationError(f"prepared {label} row {index} has condition drift")
        if dataset_id is not None and record.get("dataset_id") != dataset_id:
            raise EvaluationError(f"prepared {label} row {index} has dataset_id drift")
        if not isinstance(record.get("ground_truth"), list):
            raise EvaluationError(f"prepared {label} row {index} lacks canonical ground_truth")
    _validate_ids(records, f"prepared {label}")
    return records, {"count": len(records), "path": str(path), "sha256": _sha256(payload)}


def _prediction_records(
    run_dir: AdmittedDirectory, label: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    turn, difficulty = label.split(".", 1)
    admitted = _admit_member(
        run_dir,
        f"predictions/{turn}/{difficulty}/predictions.json",
        f"predictions {label}",
    )
    path, payload = admitted.path, admitted.payload
    records = _json_records(payload, f"predictions {label}")
    return records, {
        "count": len(records),
        "path": path.relative_to(run_dir.path).as_posix(),
        "sha256": _sha256(payload),
    }


def _validate_join(
    prepared: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    label: str,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    expected = _validate_ids(prepared, f"prepared {label}")
    actual = _validate_ids(predictions, f"predictions {label}")
    missing, extra = sorted(set(expected) - set(actual)), sorted(set(actual) - set(expected))
    if missing or extra:
        raise EvaluationError(
            f"prediction instance_id set mismatch for {label}: "
            f"missing={missing[:3]}, extra={extra[:3]}"
        )
    joined = []
    for instance_id, canonical in expected.items():
        prediction = actual[instance_id]
        for key in ("dataset_id", "difficulty", "source_example_id", "turn"):
            if key in prediction and prediction[key] != canonical.get(key):
                raise EvaluationError(
                    f"prediction condition mismatch for {instance_id}: {key}="
                    f"{prediction[key]!r}, expected {canonical.get(key)!r}"
                )
        joined.append((canonical, prediction))
    return joined


def _preference_metadata(
    manifest: Mapping[str, Any], repo_root: AdmittedDirectory
) -> tuple[dict[str, set[str]] | None, dict[str, Any] | None]:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or "pref_list" not in inputs:
        return None, None
    spec = inputs["pref_list"]
    if not isinstance(spec, dict):
        raise EvaluationError("pref_list manifest entry must be an object")
    path, payload = _verify_bound_file(repo_root, spec, "preference metadata")
    value = _decode_json(payload, "preference metadata")
    if not isinstance(value, dict) or not all(
        isinstance(domain, str)
        and isinstance(slots, list)
        and all(isinstance(slot, str) for slot in slots)
        for domain, slots in value.items()
    ):
        raise EvaluationError("preference metadata must map domains to slot arrays")
    return (
        {domain: set(slots) for domain, slots in value.items()},
        {"path": str(path), "sha256": _sha256(payload)},
    )


def _evaluate_rows(
    rows: Iterable[tuple[Mapping[str, Any], Mapping[str, Any]]],
    preference_map: Mapping[str, set[str]] | None,
) -> dict[str, Any]:
    overall, pref_counts, nonpref_counts = _Counts(), _Counts(), _Counts()
    row_count = parse_failures = row_errors = pref_exact = 0
    for canonical, prediction_row in rows:
        row_count += 1
        ground_truth = parse_calls(canonical["ground_truth"])
        if not ground_truth.call_count:
            raise EvaluationError(
                f"canonical ground_truth is unparseable: {canonical['instance_id']}"
            )
        prediction_value, missing_prediction = _prediction_value(prediction_row)
        prediction = parse_calls(prediction_value)
        if not prediction.call_count:
            parse_failures += 1
        invalid_prediction = prediction_value is None or not isinstance(
            prediction_value, (str, list, dict)
        )
        if (
            missing_prediction
            or invalid_prediction
            or prediction_row.get("status") == "error"
            or prediction_row.get("error")
        ):
            row_errors += 1
        overall.add(_count(ground_truth.values, prediction.values))
        if preference_map is not None:
            gt_pref = _filter_slots(ground_truth.values, preference_map, True)
            pred_pref = _filter_slots(prediction.values, preference_map, True)
            row_pref = _count(gt_pref, pred_pref)
            pref_counts.add(row_pref)
            pref_exact += int(row_pref.fp == 0 and row_pref.fn == 0)
            nonpref_counts.add(
                _count(
                    _filter_slots(ground_truth.values, preference_map, False),
                    _filter_slots(prediction.values, preference_map, False),
                )
            )
    result: dict[str, Any] = {
        "counts": {"parse_failures": parse_failures, "row_errors": row_errors, "rows": row_count},
        "overall": overall.as_dict(),
    }
    if preference_map is not None:
        result["preference"] = {
            "exact_match": {
                "count": pref_exact,
                "rate": pref_exact / row_count if row_count else 0.0,
                "total": row_count,
            },
            "slot_value": pref_counts.as_dict(),
        }
        result["non_preference"] = nonpref_counts.as_dict()
    return result


def _merge_metrics(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    overall, pref_counts, nonpref_counts = _Counts(), _Counts(), _Counts()
    rows = parse_failures = row_errors = pref_exact = pref_total = 0
    has_preference = bool(values) and all("preference" in value for value in values)
    for value in values:
        counts, metric = value["counts"], value["overall"]
        rows += counts["rows"]
        parse_failures += counts["parse_failures"]
        row_errors += counts["row_errors"]
        overall.add(_Counts(metric["tp"], metric["fp"], metric["fn"]))
        if has_preference:
            pref = value["preference"]
            slot_metric, nonpref = pref["slot_value"], value["non_preference"]
            pref_counts.add(_Counts(slot_metric["tp"], slot_metric["fp"], slot_metric["fn"]))
            nonpref_counts.add(_Counts(nonpref["tp"], nonpref["fp"], nonpref["fn"]))
            pref_exact += pref["exact_match"]["count"]
            pref_total += pref["exact_match"]["total"]
    result: dict[str, Any] = {
        "counts": {"parse_failures": parse_failures, "row_errors": row_errors, "rows": rows},
        "overall": overall.as_dict(),
    }
    if has_preference:
        result["preference"] = {
            "exact_match": {
                "count": pref_exact,
                "rate": pref_exact / pref_total if pref_total else 0.0,
                "total": pref_total,
            },
            "slot_value": pref_counts.as_dict(),
        }
        result["non_preference"] = nonpref_counts.as_dict()
    return result


def _evaluator_code_sha256() -> str:
    root, digest = Path(__file__).resolve().parent, hashlib.sha256()
    for path in sorted(root.glob("*.py"), key=lambda item: item.name):
        payload, label = path.read_bytes(), path.name.encode()
        digest.update(len(label).to_bytes(8, "big"))
        digest.update(label)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def _atomic_write_json(
    root: AdmittedDirectory,
    relative: str,
    value: Mapping[str, Any],
) -> None:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode()
    try:
        root.atomic_write(relative, payload, label="evaluation metrics")
    except FileAdmissionError as exc:
        raise EvaluationError(str(exc)) from exc


def evaluate_run(*, run_dir: Path, variant: str) -> EvaluationSummary:
    """Validate and score every condition named by a canonical run config."""

    if variant not in SUPPORTED_VARIANTS:
        raise EvaluationError(
            f"unsupported evaluator variant {variant!r}; choose one of {SUPPORTED_VARIANTS}"
        )
    with _admit_directory(run_dir, "run directory") as admitted_run:
        return _evaluate_admitted_run(run_dir=admitted_run, variant=variant)


def _evaluate_admitted_run(
    *,
    run_dir: AdmittedDirectory,
    variant: str,
) -> EvaluationSummary:
    config_admitted = _admit_member(
        run_dir, "resolved_config.json", "resolved config"
    )
    config_payload = config_admitted.payload
    config = _decode_json(config_payload, "resolved config")
    if not isinstance(config, dict):
        raise EvaluationError("resolved config must be a JSON object")
    experiment_config_name = config.get("experiment_config")
    experiment_config_sha256 = config.get("experiment_config_sha256")
    experiment_config_admitted = None
    if experiment_config_name is not None or experiment_config_sha256 is not None:
        if (
            experiment_config_name != "experiment_config.json"
            or not isinstance(experiment_config_sha256, str)
            or len(experiment_config_sha256) != 64
            or any(character not in "0123456789abcdef" for character in experiment_config_sha256)
        ):
            raise EvaluationError(
                "resolved config has invalid experiment config provenance"
            )
        if config.get("evaluator_variant") != variant:
            raise EvaluationError(
                "requested evaluator variant differs from experiment config"
            )
        experiment_config_admitted = _admit_member(
            run_dir,
            experiment_config_name,
            "experiment config",
        )
        if _sha256(experiment_config_admitted.payload) != experiment_config_sha256:
            raise EvaluationError(
                "experiment config SHA-256 does not match resolved_config.json"
            )

    run_manifest = _admit_member(
        run_dir, "dataset_manifest.json", "run dataset manifest"
    )
    manifest_payload = run_manifest.payload
    manifest_sha = _sha256(manifest_payload)
    if config.get("dataset_manifest_sha256") != manifest_sha:
        raise EvaluationError("run dataset manifest SHA-256 does not match resolved_config.json")
    manifest = _decode_json(manifest_payload, "run dataset manifest")
    if not isinstance(manifest, dict):
        raise EvaluationError("run dataset manifest must be a JSON object")
    try:
        prepared_path = Path(config["prepared_root"])
        repo_path = Path(config["repo_root"])
        original_manifest = lexical_absolute(Path(config["dataset_manifest"]))
    except (KeyError, TypeError) as exc:
        raise EvaluationError(f"resolved config has invalid artifact paths: {exc}") from exc
    with (
        _admit_directory(prepared_path, "prepared root") as prepared_root,
        _admit_directory(repo_path, "repository root") as repo_root,
    ):
        if original_manifest != prepared_root.path / "manifest.json":
            raise EvaluationError("configured prepared_root does not own dataset_manifest")
        original_admitted = _admit_member(
            prepared_root, "manifest.json", "original dataset manifest"
        )
        if original_admitted.payload != manifest_payload:
            raise EvaluationError("run dataset manifest differs from its configured source")
        return _evaluate_sources(
            run_dir=run_dir,
            prepared_root=prepared_root,
            repo_root=repo_root,
            config=config,
            config_payload=config_payload,
            manifest=manifest,
            manifest_sha=manifest_sha,
            experiment_config_admitted=experiment_config_admitted,
            variant=variant,
        )


def _evaluate_sources(
    *,
    run_dir: AdmittedDirectory,
    prepared_root: AdmittedDirectory,
    repo_root: AdmittedDirectory,
    config: Mapping[str, Any],
    config_payload: bytes,
    manifest: Mapping[str, Any],
    manifest_sha: str,
    experiment_config_admitted: AdmittedFile | None,
    variant: str,
) -> EvaluationSummary:
    raw_conditions = config.get("conditions")
    if (
        not isinstance(raw_conditions, list)
        or not raw_conditions
        or not all(isinstance(label, str) and label.count(".") == 1 for label in raw_conditions)
        or len(raw_conditions) != len(set(raw_conditions))
    ):
        raise EvaluationError("resolved config must contain unique condition labels")
    conditions = list(raw_conditions)
    preference_map, preference_provenance = _preference_metadata(manifest, repo_root)
    condition_metrics, prepared_provenance, prediction_provenance = {}, {}, {}
    seen_prepared_ids: set[str] = set()
    for label in conditions:
        prepared, prepared_info = _condition_records(prepared_root, manifest, label)
        ids = {str(row["instance_id"]) for row in prepared}
        overlap = seen_prepared_ids.intersection(ids)
        if overlap:
            raise EvaluationError(f"cross-condition duplicate instance_id: {sorted(overlap)[0]}")
        seen_prepared_ids.update(ids)
        predictions, prediction_info = _prediction_records(run_dir, label)
        condition_metrics[label] = _evaluate_rows(
            _validate_join(prepared, predictions, label), preference_map
        )
        prepared_provenance[label], prediction_provenance[label] = prepared_info, prediction_info
    aggregate = _merge_metrics(list(condition_metrics.values()))
    evaluator_config = {
        "conditions": conditions,
        "preference_metadata_sha256": preference_provenance["sha256"] if preference_provenance else None,
        "variant": variant,
    }
    document: dict[str, Any] = {
        "aggregate": aggregate,
        "conditions": condition_metrics,
        "counts": {
            "condition_count": len(conditions),
            "parse_failure_count": aggregate["counts"]["parse_failures"],
            "prediction_row_count": aggregate["counts"]["rows"],
            "prepared_row_count": aggregate["counts"]["rows"],
            "row_error_count": aggregate["counts"]["row_errors"],
        },
        "evaluator": {
            "code_sha256": _evaluator_code_sha256(),
            "config_sha256": _sha256(_canonical_json(evaluator_config)),
            "variant": variant,
            "version": EVALUATOR_VERSION,
        },
        "format_version": 1,
        "provenance": {
            "dataset_manifest_sha256": manifest_sha,
            "predictions": prediction_provenance,
            "prepared": prepared_provenance,
            "resolved_config_sha256": _sha256(config_payload),
        },
        "run": {
            "method": config.get("method"),
            "model": config.get("model"),
            "run_id": run_dir.path.name,
        },
    }
    if preference_provenance is not None:
        document["provenance"]["preference_metadata"] = preference_provenance
    if experiment_config_admitted is not None:
        document["provenance"]["experiment_config_sha256"] = _sha256(
            experiment_config_admitted.payload
        )
    output_path = run_dir.path / "metrics" / "metrics.json"
    _atomic_write_json(run_dir, "metrics/metrics.json", document)
    return EvaluationSummary(
        output_path,
        len(conditions),
        aggregate["counts"]["rows"],
        aggregate["counts"]["parse_failures"],
        aggregate["counts"]["row_errors"],
    )
