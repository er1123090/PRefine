from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from methods.amem.build_memory_batch import (
    build_evolution_request,
    build_metadata_request,
    collect_round,
    initialize,
    paths_for,
    prepare_round,
)
from methods.amem.common import (
    UPSTREAM_COMMIT,
    apply_evolution_result,
    assign_embeddings,
    create_pending_note,
    retrieve_notes,
    turn_note_units,
)
from methods.amem.inference_batch import build_prediction, inference_request
from scripts.run_inference import parser as common_inference_parser


class FakeEmbedder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "window" in lowered or "new memory" in lowered:
                vectors.append([1.0, 0.0])
            elif "aisle" in lowered:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([0.5, 0.5])
        return vectors


def construction_args(root: Path, dataset: Path) -> SimpleNamespace:
    return SimpleNamespace(
        input_path=str(dataset),
        output_dir=str(root),
        model="gpt-5-mini",
        reasoning_effort="minimal",
        embedding_model="all-MiniLM-L6-v2",
        top_k=5,
        max_completion_tokens=1536,
        max_examples=None,
        max_sessions=None,
        api_key=None,
        poll_seconds=1,
        raw_results=None,
        force_init=False,
    )


def batch_result(custom_id: str, content: dict, total: int) -> str:
    return (
        json.dumps(
            {
                "custom_id": custom_id,
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "message": {
                                    "content": json.dumps(content)
                                }
                            }
                        ],
                        "usage": {
                            "prompt_tokens": total - 10,
                            "completion_tokens": 10,
                            "total_tokens": total,
                            "completion_tokens_details": {
                                "reasoning_tokens": 1
                            },
                        },
                    },
                },
            }
        )
        + "\n"
    )


class AmemBatchPayloadTest(unittest.TestCase):
    def test_official_two_stage_requests_fix_requested_backbone(self) -> None:
        metadata = build_metadata_request(
            custom_id="amem-i0000-n0001-m",
            model="gpt-5-mini",
            reasoning_effort="minimal",
            max_completion_tokens=1536,
            content="User: I prefer window seats.",
        )
        evolution = build_evolution_request(
            custom_id="amem-i0000-n0001-e",
            model="gpt-5-mini",
            reasoning_effort="minimal",
            max_completion_tokens=1536,
            pending_note={
                "note_id": "new",
                "content": "User: I prefer window seats.",
                "context": "Seat preference.",
                "keywords": ["window"],
                "tags": ["flight"],
            },
            candidates=[],
        )

        for request in (metadata, evolution):
            self.assertEqual(request["url"], "/v1/chat/completions")
            self.assertEqual(request["body"]["model"], "gpt-5-mini")
            self.assertEqual(
                request["body"]["reasoning_effort"], "minimal"
            )
            self.assertNotIn("temperature", request["body"])
            self.assertTrue(
                request["body"]["response_format"]["json_schema"]["strict"]
            )
        self.assertEqual(
            metadata["body"]["response_format"]["json_schema"]["name"],
            "amem_note_metadata",
        )
        self.assertEqual(
            evolution["body"]["response_format"]["json_schema"]["name"],
            "amem_note_evolution",
        )

    def test_inference_request_fixes_requested_backbone(self) -> None:
        request = inference_request(
            {"sample_id": "amem-00001", "prompt": "predict"},
            model="gpt-5-mini",
            reasoning_effort="minimal",
        )
        self.assertEqual(request["body"]["model"], "gpt-5-mini")
        self.assertEqual(request["body"]["reasoning_effort"], "minimal")
        self.assertEqual(request["body"]["max_completion_tokens"], 2048)
        self.assertNotIn("temperature", request["body"])

    def test_common_amem_retrieval_defaults_match_official_wrapper(self) -> None:
        args = common_inference_parser().parse_args(
            [
                "--method",
                "amem",
                "--turn",
                "single",
                "--pref_type",
                "easy",
                "--model",
                "gpt-5-mini",
            ]
        )
        self.assertEqual(args.memory_top_k, 10)
        self.assertEqual(args.linked_neighbor_limit, 10)


class AmemMemorySemanticsTest(unittest.TestCase):
    def test_dialogue_turns_are_atomic_notes(self) -> None:
        units = turn_note_units(
            {
                "sessions": [
                    {
                        "dialogue_id": "d1",
                        "dialogue": [
                            {
                                "role": "User",
                                "message": "Window, please.",
                            },
                            {
                                "role": "Assistant",
                                "message": "Booked.",
                                "service": [
                                    'BookFlight(seat="window")'
                                ],
                            },
                        ],
                    }
                ]
            }
        )
        self.assertEqual(len(units), 2)
        self.assertEqual(units[0]["content"], "User: Window, please.")
        self.assertIn("Observed service call", units[1]["content"])
        self.assertEqual(units[1]["session_index"], 1)
        self.assertEqual(units[1]["turn_index"], 2)

    def test_official_evolution_is_directed_and_candidate_bounded(self) -> None:
        user = {
            "notes": [
                {
                    "note_id": "old",
                    "note_index": 1,
                    "session_index": 1,
                    "turn_index": 1,
                    "content": "User: I used to choose aisle seats.",
                    "context": "Older seat choice.",
                    "keywords": ["aisle"],
                    "tags": ["flight"],
                    "links": [],
                    "embedding": [0.0, 1.0],
                }
            ]
        }
        pending = create_pending_note(
            {
                "note_index": 2,
                "session_index": 1,
                "turn_index": 2,
                "dialogue_id": "d1",
                "speaker": "User",
                "content": "User: I now choose a window seat.",
            },
            note_id="new",
            metadata_response={
                "context": "Later seat choice.",
                "keywords": ["window"],
                "tags": ["flight"],
            },
            metadata_usage={"total_tokens": 10},
            timestamp="2026-07-27T00:00:00+00:00",
        )
        changed = apply_evolution_result(
            user,
            pending_note=pending,
            candidate_note_ids=["old"],
            response={
                "should_evolve": True,
                "actions": ["strengthen", "update_neighbor"],
                "suggested_connections": ["old", "not-a-candidate"],
                "tags_to_update": ["flight", "preference-change"],
                "new_context_neighborhood": [
                    "Previously aisle; later evidence differs."
                ],
                "new_tags_neighborhood": [
                    ["flight", "historical-preference"]
                ],
            },
            evolution_usage={"total_tokens": 20},
            timestamp="2026-07-27T00:01:00+00:00",
        )
        assign_embeddings(user, changed, embedder=FakeEmbedder())

        old, new = user["notes"]
        self.assertEqual(
            old["content"], "User: I used to choose aisle seats."
        )
        self.assertEqual(
            new["content"], "User: I now choose a window seat."
        )
        self.assertEqual(new["links"], ["old"])
        self.assertEqual(old["links"], [])
        self.assertEqual(
            old["context"], "Previously aisle; later evidence differs."
        )
        self.assertEqual(changed, {"old", "new"})
        self.assertEqual(new["embedding"], [1.0, 0.0])

    def test_retrieval_expands_one_hop_directed_links(self) -> None:
        notes = [
            {
                "note_id": "seed",
                "content": "window seat",
                "context": "",
                "keywords": [],
                "tags": [],
                "links": ["linked"],
                "embedding": [1.0, 0.0],
            },
            {
                "note_id": "linked",
                "content": "aisle seat",
                "context": "",
                "keywords": [],
                "tags": [],
                "links": [],
                "embedding": [0.0, 1.0],
            },
            {
                "note_id": "other",
                "content": "unrelated",
                "context": "",
                "keywords": [],
                "tags": [],
                "links": [],
                "embedding": [-1.0, 0.0],
            },
        ]
        selected = retrieve_notes(
            notes,
            "window please",
            embedder=FakeEmbedder(),
            top_k=1,
            linked_neighbor_limit=10,
        )
        self.assertEqual(
            [note["note_id"] for note in selected], ["seed", "linked"]
        )
        self.assertEqual(selected[1]["retrieval_source"], "linked:seed")


class AmemRoundIntegrationTest(unittest.TestCase):
    def test_two_stage_round_finalizes_one_turn_from_local_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset_path = root / "dataset.json"
            dataset_path.write_text(
                json.dumps(
                    [
                        {
                            "example_id": "u1",
                            "sessions": [
                                {
                                    "dialogue_id": "d1",
                                    "dialogue": [
                                        {
                                            "role": "User",
                                            "message": "I prefer window seats.",
                                        }
                                    ],
                                }
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = construction_args(root / "out", dataset_path)
            state, working = initialize(args)
            self.assertEqual(state["upstream_commit"], UPSTREAM_COMMIT)

            state = prepare_round(args, state, working)
            self.assertEqual(state["phase"], "metadata_prepared")
            metadata_raw = root / "metadata.jsonl"
            metadata_raw.write_text(
                batch_result(
                    "amem-i0000-n0001-m",
                    {
                        "context": "Seat preference.",
                        "keywords": ["window"],
                        "tags": ["flight"],
                    },
                    60,
                ),
                encoding="utf-8",
            )
            args.raw_results = str(metadata_raw)
            state = collect_round(
                args,
                state,
                working,
                embedder=FakeEmbedder(),
            )
            self.assertEqual(state["phase"], "evolution_prepared")

            evolution_manifest = [
                json.loads(line)
                for line in Path(
                    state["active_attempt"]["manifest_path"]
                )
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(
                evolution_manifest[0]["candidate_note_ids"], []
            )
            evolution_raw = root / "evolution.jsonl"
            evolution_raw.write_text(
                batch_result(
                    "amem-i0000-n0001-e",
                    {
                        "should_evolve": False,
                        "actions": [],
                        "suggested_connections": [],
                        "tags_to_update": [],
                        "new_context_neighborhood": [],
                        "new_tags_neighborhood": [],
                    },
                    40,
                ),
                encoding="utf-8",
            )
            args.raw_results = str(evolution_raw)
            final_state = collect_round(
                args,
                state,
                working,
                embedder=FakeEmbedder(),
            )

            self.assertEqual(final_state["phase"], "finalized")
            paths = paths_for(args)
            artifact = [
                json.loads(line)
                for line in paths["artifact"]
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(artifact), 1)
            self.assertEqual(artifact[0]["note_unit"], "dialogue_turn")
            self.assertEqual(artifact[0]["upstream_commit"], UPSTREAM_COMMIT)
            self.assertEqual(len(artifact[0]["notes"]), 1)
            self.assertEqual(
                artifact[0]["construction_usage"]["total_tokens"], 100
            )
            self.assertEqual(len(final_state["batch_history"]), 2)

    def test_prediction_matches_experiment8_evaluator_contract(self) -> None:
        manifest = {
            "example_id": "u1",
            "example_id_sub": "u1_0",
            "utterance": "Book it",
            "reference_ground_truth": ['BookFlight(seat="window")'],
            "retrieved_memories": [{"note_id": "n1"}],
            "retrieved_memory_count": 1,
            "memory_top_k": 10,
            "linked_neighbor_limit": 10,
            "prompt": "prompt",
            "population_index": 0,
            "turn": "single",
            "query": "hint",
            "schema": "easy",
            "pref_type": "medium",
            "condition": "single_medium",
            "original_ex": {"example_id": "u1"},
        }
        prediction, log = build_prediction(
            manifest,
            {
                "error": None,
                "content": 'BookFlight(seat="window")',
                "reasoning": "",
                "usage": {"total_tokens": 5},
            },
            model="gpt-5-mini",
            batch_id="batch-test",
        )
        self.assertEqual(prediction["method"], "amem")
        self.assertEqual(
            prediction["llm_output"], 'BookFlight(seat="window")'
        )
        self.assertEqual(log["status"], "OK")


if __name__ == "__main__":
    unittest.main()
