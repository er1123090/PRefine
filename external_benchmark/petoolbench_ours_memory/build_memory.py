"""Build ours_memory-compatible memory records for normalized PEToolBench rows."""

from __future__ import annotations

import argparse
import json

from common import build_ours_sessions, iter_jsonl, mock_preference, tool_call_to_text, write_jsonl


def build_mock_memory(record):
    sessions = build_ours_sessions(record)
    pref = mock_preference(record)
    accumulated_calls = []
    for idx, session in enumerate(sessions, start=1):
        for call in session.get("api_call", []):
            accumulated_calls.append(f"[Session {idx}] {call}")

    return {
        "example_id": record["example_id"],
        "source_index": record["source_index"],
        "history_type": record["history_type"],
        "memory_mode": "petoolbench_mock",
        "final_implicit_preference": json.dumps(pref, ensure_ascii=False, indent=2),
        "final_accumulated_implicit_preferences": [pref],
        "final_accumulated_api_calls": accumulated_calls,
        "total_sessions_processed": len(sessions),
        "preference_evolution_history": [
            {
                "session_index": len(sessions),
                "memory_mode": "petoolbench_mock",
                "final_preference_at_session": pref,
            }
        ],
        "petoolbench_metadata": {
            "history_length": record.get("history_length"),
            "candidate_tool_count": record.get("candidate_tool_count"),
            "ground_truth": record.get("api_call_ground_truth"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PEToolBench memory JSONL.")
    parser.add_argument("--input", required=True, help="Normalized PEToolBench JSONL.")
    parser.add_argument("--output", required=True, help="Memory JSONL output.")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Build deterministic mock memory records without LLM calls.",
    )
    args = parser.parse_args()

    if not args.mock:
        raise SystemExit(
            "Only --mock is implemented in this adapter. Use the emitted normalized JSONL "
            "as input to a live PreferenceAggregator integration if API-backed memory "
            "generation is needed."
        )

    rows = [build_mock_memory(record) for record in iter_jsonl(args.input)]
    write_jsonl(args.output, rows)
    print(f"memory_records={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()

