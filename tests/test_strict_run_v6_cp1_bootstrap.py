from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from provenance.strict_v6 import inventory_digest  # noqa: E402
from strict_run import (  # noqa: E402
    ArtifactEvidence,
    canonical_json,
    sha256_bytes,
    validate_checkpoint_payload,
)
from strict_run.cp1_bootstrap import (  # noqa: E402
    CP1_CAPTURE_EXECUTION_SCHEMA,
    CP1_CAPTURE_SCHEMA,
    PreparedCP1,
    bootstrap_cp1,
)
from test_strict_run_v6_checkpoints import RUN_ID, SyntheticStrictRun  # noqa: E402


class CP1BootstrapRegressionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="exp7-cp1-bootstrap-")
        self.fixture = SyntheticStrictRun(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _prepared() -> PreparedCP1:
        structure = tuple(
            {"ordinal": index, "paper_section_id": f"section-{index:02d}"}
            for index in range(51)
        )
        results = tuple(
            {
                "display_value": str(index),
                "numeric_value": index,
                "paper_section_id": f"section-{index % 51:02d}",
                "result_id": f"result:bootstrap:{index:04d}",
            }
            for index in range(1908)
        )
        aliases = tuple(
            {
                "alias_id": f"alias:bootstrap:{index:03d}",
                "target_result_id": f"result:bootstrap:{index:04d}",
            }
            for index in range(86)
        )
        inventory_digest(list(structure), list(results), list(aliases))
        structure_payload = b"".join(canonical_json(row) for row in structure)
        results_payload = b"".join(canonical_json(row) for row in results)
        aliases_payload = b"".join(canonical_json(row) for row in aliases)
        return PreparedCP1(
            structure=structure,
            results=results,
            aliases=aliases,
            lineage_ids=tuple(f"lineage:{index:03d}" for index in range(129)),
            profile_ids=tuple(f"profile:{index:03d}" for index in range(30)),
            layout_sha256="b" * 64,
            source_counts={"fixture": 1908},
            alias_reason_counts={"fixture": 86},
            structure_payload=structure_payload,
            results_payload=results_payload,
            aliases_payload=aliases_payload,
        )

    def test_snapshots_same_parent_inputs_before_creating_sibling_outputs(self) -> None:
        cp0 = self.fixture.cp0()
        cp0_raw = (self.fixture.run_root / "checkpoints" / "cp0.json").read_bytes()
        evidence_by_path = {
            row["relative_path"]: ArtifactEvidence.from_dict(row)
            for row in cp0.payload["stage_actual_paths"]
        }
        source_pre = evidence_by_path["manifests/source-pre.jsonl"]
        paper_pre_evidence = evidence_by_path["manifests/paper-pre.json"]
        paper_pre = {
            "schema": "experiments7-paper-pre/v6",
            "sealed_run_id": RUN_ID,
            "run_root": str(self.fixture.run_root),
            "paper_path": "/fixture/paper.pdf",
            "sha256": "a" * 64,
            "size": 17,
            "descriptor_identity": {},
            "parent_identity": {},
            "provider": {},
        }
        pages = [
            {
                "page": index,
                "layout_text": f"fixture page {index}",
                "layout_text_sha256": sha256_bytes(
                    f"fixture page {index}".encode("utf-8")
                ),
            }
            for index in range(1, 26)
        ]
        capture = {
            "schema": CP1_CAPTURE_SCHEMA,
            "sealed_run_id": RUN_ID,
            "run_root": str(self.fixture.run_root),
            "paper": {"sha256": paper_pre["sha256"], "bytes": paper_pre["size"]},
            "provider": {"writable_directories": []},
            "paper_write_probe": {"denied": True},
            "parser": {"implementation": "fixture"},
            "page_count": len(pages),
            "pages": pages,
        }
        external = self.fixture.external
        capture_path = external / "cp1-protected-layout.json"
        capture_path.write_bytes(canonical_json(capture))
        execution_path = external / "cp1-extraction-execution.json"
        execution_path.write_bytes(
            canonical_json(
                {
                    "schema": CP1_CAPTURE_EXECUTION_SCHEMA,
                    "sealed_run_id": RUN_ID,
                    "run_root": str(self.fixture.run_root),
                    "argv": [],
                    "environment": {},
                    "extractor": {},
                    "provider": {},
                    "expected_paper_sha256": paper_pre["sha256"],
                    "child_stdout_sha256": sha256_bytes(b"fixture"),
                    "child_stdout_bytes": len(b"fixture"),
                    "extraction_sha256": sha256_bytes(canonical_json(capture)),
                    "page_count": len(pages),
                }
            )
        )
        cp0_transcript_path = external / "cp0-transcript.json"
        cp0_transcript_path.write_bytes(canonical_json(cp0.transcript))
        outputs = {
            "task": external / "cp1-task-evidence.json",
            "bundle": external / "cp1-stage-bundle.json",
            "checkpoint": external / "cp1-transcript.json",
            "stage": external / "cp1-stage-transcripts.json",
        }
        with (
            patch(
                "strict_run.cp1_bootstrap._cp0_inputs",
                return_value=(cp0.payload, cp0_raw, source_pre, paper_pre_evidence),
            ),
            patch("strict_run.cp1_bootstrap._load_cp0_paper_pre", return_value=paper_pre),
            patch("strict_run.cp1_bootstrap._load_source_pre_count", return_value=1),
            patch("strict_run.cp1_bootstrap.prepare_cp1", return_value=self._prepared()),
        ):
            result = bootstrap_cp1(
                strict_parent=str(self.fixture.parent),
                sealed_run_id=RUN_ID,
                run_root=str(self.fixture.run_root),
                external_capture=str(capture_path),
                external_capture_execution=str(execution_path),
                external_cp0_transcript=str(cp0_transcript_path),
                external_task_evidence=str(outputs["task"]),
                external_stage_bundle=str(outputs["bundle"]),
                external_cp1_transcript=str(outputs["checkpoint"]),
                external_stage_transcripts=str(outputs["stage"]),
            )
        self.assertEqual(result.checkpoint.payload["checkpoint"], 1)
        self.assertEqual(
            result.checkpoint.payload["semantic_bindings"]["inventory_record_count"], 1908
        )
        validate_checkpoint_payload(
            result.checkpoint.payload,
            expected_sealed_run_id=RUN_ID,
            expected_run_root=str(self.fixture.run_root),
            expected_checkpoint=1,
            policy=self.fixture.policy,
        )
        for path in outputs.values():
            self.assertTrue(path.is_file())
            self.assertEqual(path.stat().st_mode & 0o777, 0o444)
        self.assertEqual(
            set(json.loads(outputs["bundle"].read_bytes())), {"publications"}
        )


if __name__ == "__main__":
    unittest.main()
