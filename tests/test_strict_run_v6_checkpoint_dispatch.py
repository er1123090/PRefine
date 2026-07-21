from __future__ import annotations

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_strict_run_v6_checkpoints import SyntheticStrictRun
from tests.test_strict_run_v6_cp45_semantics import CP45Fixture

from strict_run import (
    StrictRunError,
    canonical_json,
    sha256_bytes,
    validate_checkpoint_payload,
)
from strict_run.checkpoint import (
    _SemanticReplayContext,
    _cp3_semantic_inputs,
    _load_json,
    _replay_checkpoint_semantics,
    _require_canonical_checkpoint_chain,
    _require_stored_semantic_replay,
    _validate_stage_chronology,
)
from strict_run.publication import Publication


class StrictRunV6CheckpointDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(
            prefix="exp7-checkpoint-dispatch-v6-"
        )
        self.fixture = SyntheticStrictRun(Path(self.temporary.name))
        self.cp0 = self.fixture.cp0()
        self.cp1 = self.fixture.cp1()
        self.root_fd = os.open(
            self.fixture.run_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )

    def tearDown(self) -> None:
        os.close(self.root_fd)
        self.temporary.cleanup()

    def test_cp3_cp4_cp5_payload_schemas_are_integrated(self) -> None:
        cases = (
            (3, "provenance/nodes.jsonl", "experiments7-cp3-semantic-binding/v6"),
            (4, "copies.jsonl", "experiments7-cp4-semantic-binding/v6"),
            (5, "manifests/source-post.jsonl", "experiments7-cp5-semantic-binding/v6"),
        )
        for checkpoint, path, schema in cases:
            with self.subTest(checkpoint=checkpoint):
                payload = copy.deepcopy(self.cp1.payload)
                evidence = copy.deepcopy(payload["stage_actual_paths"][0])
                evidence["relative_path"] = path
                payload.update(
                    {
                        "checkpoint": checkpoint,
                        "stage_actual_paths": [evidence],
                        "stage_actual_paths_sha256": sha256_bytes(
                            canonical_json([evidence])
                        ),
                        "semantic_bindings": {
                            "schema": schema,
                            "sealed_run_id": payload["sealed_run_id"],
                            "run_root": payload["run_root"],
                            "checkpoint": checkpoint,
                        },
                    }
                )
                payload["semantic_bindings_sha256"] = sha256_bytes(
                    canonical_json(payload["semantic_bindings"])
                )
                validated = validate_checkpoint_payload(
                    payload,
                    expected_sealed_run_id=payload["sealed_run_id"],
                    expected_run_root=payload["run_root"],
                    expected_checkpoint=checkpoint,
                )
                self.assertEqual(validated["checkpoint"], checkpoint)

    def test_dispatch_routes_cp3_cp4_cp5_and_cp6_to_typed_validators(self) -> None:
        source_pre = next(
            publication
            for publication in self.cp0.stage_publications
            if publication.evidence.relative_path == "manifests/source-pre.jsonl"
        )
        placeholder = self.cp1.publication
        context = _SemanticReplayContext(
            checkpoint_publications={
                0: self.cp0.publication,
                1: placeholder,
                2: placeholder,
                3: placeholder,
                4: placeholder,
                5: placeholder,
            },
            stage_publications={
                0: self.cp0.stage_publications,
                1: self.cp1.stage_publications,
                2: self.cp1.stage_publications,
                3: self.cp1.stage_publications,
                4: self.cp1.stage_publications,
                5: self.cp1.stage_publications,
                6: self.cp1.stage_publications,
            },
        )
        registered = list(self.cp0.stage_publications) + list(
            self.cp1.stage_publications
        )
        with self.fixture.transcript_registry(
            registered, self.cp1.publication
        ) as registry:
            with (
                patch(
                    "strict_run.checkpoint._require_canonical_checkpoint_chain"
                ) as chain,
                patch(
                    "strict_run.cp3_semantic.validate_cp3_semantics",
                    return_value={"checkpoint": 3},
                ) as cp3,
            ):
                result = _replay_checkpoint_semantics(
                    self.root_fd,
                    self.cp1.payload["sealed_run_id"],
                    self.cp1.payload["run_root"],
                    3,
                    self.cp1.stage_publications,
                    "3" * 64,
                    "4" * 64,
                    registry,
                    context,
                )
                self.assertEqual(result, {"checkpoint": 3})
                self.assertEqual(chain.call_args.args[3], 2)
                self.assertIs(cp3.call_args.args[6], source_pre)

            with (
                patch(
                    "strict_run.checkpoint._cp3_semantic_inputs",
                    return_value=object(),
                ),
                patch(
                    "strict_run.checkpoint._require_canonical_checkpoint_chain"
                ) as cp4_chain,
                patch(
                    "strict_run.checkpoint._require_stored_semantic_replay"
                ) as cp4_prior_replay,
                patch(
                    "strict_run.cp45_semantic.validate_cp4_semantics",
                    return_value={"checkpoint": 4},
                ) as cp4,
            ):
                result = _replay_checkpoint_semantics(
                    self.root_fd,
                    self.cp1.payload["sealed_run_id"],
                    self.cp1.payload["run_root"],
                    4,
                    self.cp1.stage_publications,
                    "4" * 64,
                    "5" * 64,
                    registry,
                    context,
                )
                self.assertEqual(result, {"checkpoint": 4})
                self.assertEqual(cp4_chain.call_args.args[3], 3)
                self.assertEqual(cp4.call_args.args[3], self.cp1.stage_publications)
                self.assertEqual(cp4_prior_replay.call_args.args[3], 3)

            with (
                patch(
                    "strict_run.checkpoint._cp3_semantic_inputs",
                    return_value=object(),
                ),
                patch(
                    "strict_run.checkpoint._require_canonical_checkpoint_chain"
                ) as cp5_chain,
                patch(
                    "strict_run.checkpoint._require_stored_semantic_replay"
                ) as cp5_prior_replay,
                patch(
                    "strict_run.cp45_semantic.validate_cp5_semantics",
                    return_value={"checkpoint": 5},
                ) as cp5,
            ):
                result = _replay_checkpoint_semantics(
                    self.root_fd,
                    self.cp1.payload["sealed_run_id"],
                    self.cp1.payload["run_root"],
                    5,
                    self.cp1.stage_publications,
                    "5" * 64,
                    "6" * 64,
                    registry,
                    context,
                )
                self.assertEqual(result, {"checkpoint": 5})
                self.assertEqual(cp5_chain.call_args.args[3], 4)
                self.assertEqual(cp5.call_args.args[3], self.cp1.stage_publications)
                self.assertEqual(cp5_prior_replay.call_args.args[3], 3)

            with (
                patch(
                    "strict_run.checkpoint._require_canonical_checkpoint_chain"
                ) as cp6_chain,
                patch(
                    "strict_run.checkpoint.validate_stage_semantics",
                    return_value={"checkpoint": 6},
                ) as cp6,
            ):
                result = _replay_checkpoint_semantics(
                    self.root_fd,
                    self.cp1.payload["sealed_run_id"],
                    self.cp1.payload["run_root"],
                    6,
                    self.cp1.stage_publications,
                    "6" * 64,
                    "7" * 64,
                    registry,
                    context,
                )
                self.assertEqual(result, {"checkpoint": 6})
                self.assertEqual(cp6_chain.call_args.args[3], 5)
                prior = cp6.call_args.kwargs["prior_publications"]
                self.assertEqual(
                    len(prior),
                    sum(
                        len(context.stage_publications[index])
                        for index in range(6)
                    )
                    + 6,
                )

    def test_cp45_typed_context_is_reconstructed_from_exact_prior_stages(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="exp7-cp45-context-v6-")
        self.addCleanup(temporary.cleanup)
        fixture = CP45Fixture(Path(temporary.name))
        context = _SemanticReplayContext(
            checkpoint_publications={
                1: fixture.cp1_checkpoint,
                2: fixture.cp2_checkpoint,
                3: fixture.cp3_checkpoint,
            },
            stage_publications={
                0: (fixture.source_pre, fixture.paper_pre),
                1: (fixture.inventory_publication,),
                3: tuple(fixture.cp3_stage),
            },
        )
        inputs = _cp3_semantic_inputs(context)
        self.assertIs(inputs.checkpoint, fixture.cp3_checkpoint)
        self.assertIs(inputs.cp1_checkpoint, fixture.cp1_checkpoint)
        self.assertIs(inputs.cp2_checkpoint, fixture.cp2_checkpoint)
        self.assertIs(inputs.source_pre, fixture.source_pre)
        self.assertIs(inputs.paper_pre, fixture.paper_pre)
        self.assertIs(inputs.inventory, fixture.inventory_publication)
        self.assertEqual(
            [item.evidence.relative_path for item in inputs.buckets],
            sorted(
                item.evidence.relative_path
                for item in fixture.cp3_stage
                if item.evidence.artifact_type == "regular"
                and item.evidence.relative_path.startswith(
                    "admission/precopy/buckets/"
                )
            ),
        )

    def test_canonical_cp3_replay_rejects_cp45_stored_semantic_shortcut(self) -> None:
        temporary = tempfile.TemporaryDirectory(prefix="exp7-cp3-shortcut-v6-")
        self.addCleanup(temporary.cleanup)
        fixture = CP45Fixture(Path(temporary.name))
        context = _SemanticReplayContext(
            checkpoint_publications={
                1: fixture.cp1_checkpoint,
                2: fixture.cp2_checkpoint,
                3: fixture.cp3_checkpoint,
            },
            stage_publications={
                0: (fixture.source_pre, fixture.paper_pre),
                1: (fixture.inventory_publication,),
                2: (fixture.paper_pre,),
                3: tuple(fixture.cp3_stage),
            },
        )
        root_fd = os.open(
            fixture.run_root,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        try:
            with fixture.registry() as registry, self.assertRaises(StrictRunError):
                _require_stored_semantic_replay(
                    root_fd,
                    fixture.run_root.name,
                    str(fixture.run_root),
                    3,
                    registry,
                    context,
                )
        finally:
            os.close(root_fd)

    def test_generic_stage_checkpoint_chronology_closes_both_edges(self) -> None:
        _validate_stage_chronology(
            self.cp0.stage_publications,
            previous_checkpoint_publication=None,
            current_checkpoint_publication=self.cp0.publication,
        )
        _validate_stage_chronology(
            self.cp1.stage_publications,
            previous_checkpoint_publication=self.cp0.publication,
            current_checkpoint_publication=self.cp1.publication,
        )

        lower_transcript = copy.deepcopy(self.cp1.stage_publications[0].transcript)
        lower_transcript["events"][0]["completed_monotonic_ns"] = (
            self.cp0.publication.transcript["emitted_monotonic_ns"]
        )
        lower = Publication(self.cp1.stage_publications[0].evidence, lower_transcript)
        with self.assertRaises(StrictRunError) as caught:
            _validate_stage_chronology(
                (lower,),
                previous_checkpoint_publication=self.cp0.publication,
                current_checkpoint_publication=self.cp1.publication,
            )
        self.assertEqual(caught.exception.code, "CHECKPOINT_CHRONOLOGY_INVALID")

    def test_canonical_chain_binds_current_cp0_to_cp1(self) -> None:
        context = _SemanticReplayContext(
            checkpoint_publications={
                0: self.cp0.publication,
                1: self.cp1.publication,
            },
            stage_publications={
                0: self.cp0.stage_publications,
                1: self.cp1.stage_publications,
            },
        )
        registered = (
            list(self.cp0.stage_publications)
            + list(self.cp1.stage_publications)
            + [self.cp0.publication, self.cp1.publication]
        )
        with self.fixture.transcript_registry(registered) as registry:
            _require_canonical_checkpoint_chain(
                self.root_fd,
                self.cp1.payload["sealed_run_id"],
                self.cp1.payload["run_root"],
                1,
                registry,
                context,
            )
            forged_context = _SemanticReplayContext(
                checkpoint_publications={
                    0: self.cp1.publication,
                    1: self.cp1.publication,
                },
                stage_publications=context.stage_publications,
            )
            with self.assertRaises(StrictRunError) as caught:
                _require_canonical_checkpoint_chain(
                    self.root_fd,
                    self.cp1.payload["sealed_run_id"],
                    self.cp1.payload["run_root"],
                    1,
                    registry,
                    forged_context,
                )
            self.assertEqual(caught.exception.code, "CHECKPOINT_CHAIN_MISMATCH")

            def forged_cp0_bindings(root_fd: int, relative: str):
                payload, raw = _load_json(root_fd, relative)
                if relative == "checkpoints/cp0.json":
                    payload = copy.deepcopy(payload)
                    payload["bindings"]["reservation_sha256"] = "0" * 64
                return payload, raw

            with (
                patch(
                    "strict_run.checkpoint._load_json",
                    side_effect=forged_cp0_bindings,
                ),
                self.assertRaises(StrictRunError) as caught,
            ):
                _require_canonical_checkpoint_chain(
                    self.root_fd,
                    self.cp1.payload["sealed_run_id"],
                    self.cp1.payload["run_root"],
                    1,
                    registry,
                    context,
                )
            self.assertEqual(caught.exception.code, "CP0_BINDING_MISMATCH")

        upper_transcript = copy.deepcopy(self.cp1.stage_publications[0].transcript)
        upper_transcript["emitted_monotonic_ns"] = self.cp1.publication.transcript[
            "events"
        ][0]["completed_monotonic_ns"]
        upper = Publication(self.cp1.stage_publications[0].evidence, upper_transcript)
        with self.assertRaises(StrictRunError) as caught:
            _validate_stage_chronology(
                (upper,),
                previous_checkpoint_publication=self.cp0.publication,
                current_checkpoint_publication=self.cp1.publication,
            )
        self.assertEqual(caught.exception.code, "CHECKPOINT_CHRONOLOGY_INVALID")


if __name__ == "__main__":
    unittest.main()
