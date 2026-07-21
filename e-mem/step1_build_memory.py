from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from common import (
    DEFAULT_EMEM_EMBEDDING_MODEL,
    DEFAULT_EMEM_LLM_MODEL,
    DEFAULT_INDEX_ROOT,
    DEFAULT_INPUT_PATH,
    DEFAULT_MANIFEST_PATH,
    DEFAULT_SUPPORT_JSON_SCHEMA,
    EMem,
    build_pseudo_conversation,
    count_workspace_edus,
    create_emem_config,
    default_run_log_path,
    load_chains_dataset,
    manifest_config_dict,
    now_iso,
    setup_logger,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build EMem indexes for experiments4 1229_dev_6."
    )
    parser.add_argument("--input_path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--index_root", type=str, default=DEFAULT_INDEX_ROOT)
    parser.add_argument("--manifest_path", type=str, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--llm_model", type=str, default=DEFAULT_EMEM_LLM_MODEL)
    parser.add_argument("--llm_base_url", type=str, default=None)
    parser.add_argument("--embedding_model", type=str, default=DEFAULT_EMEM_EMBEDDING_MODEL)
    parser.add_argument("--embedding_base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--embedding_api_key", type=str, default=None)
    parser.add_argument(
        "--support_json_schema",
        type=str,
        choices=["true", "false"],
        default=DEFAULT_SUPPORT_JSON_SCHEMA,
    )
    parser.add_argument("--force_rebuild", action="store_true")
    parser.add_argument("--start_example", type=int, default=0)
    parser.add_argument("--end_example", type=int, default=None)
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--run_log_path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_log_path = args.run_log_path or default_run_log_path(args.manifest_path)
    logger = setup_logger("emem_build", run_log_path)

    df = load_chains_dataset(args.input_path)
    rows = df.to_dict("records")[args.start_example : args.end_example]
    if args.max_examples is not None:
        rows = rows[: args.max_examples]

    os.makedirs(args.index_root, exist_ok=True)
    logger.info(
        "Starting EMem build: dataset_rows=%d selected_examples=%d input=%s index_root=%s manifest=%s llm_model=%s embedding_model=%s llm_base_url=%s embedding_base_url=%s support_json_schema=%s force_rebuild=%s",
        len(df),
        len(rows),
        args.input_path,
        args.index_root,
        args.manifest_path,
        args.llm_model,
        args.embedding_model,
        args.llm_base_url,
        args.embedding_base_url,
        args.support_json_schema,
        args.force_rebuild,
    )

    manifest_rows: List[Dict[str, Any]] = []
    progress_interval = max(1, min(10, len(rows))) if rows else 1

    for row_index, example in enumerate(rows, start=args.start_example):
        example_id = str(example.get("example_id", f"example_{row_index:04d}"))
        workspace_dir = os.path.join(args.index_root, example_id)
        os.makedirs(workspace_dir, exist_ok=True)

        record = {
            "timestamp": now_iso(),
            "example_id": example_id,
            "workspace_dir": workspace_dir,
            "session_count": len(example.get("sessions", [])),
            "edu_count": 0,
            "build_status": "ERROR",
            "error": None,
            "config": manifest_config_dict(
                llm_model=args.llm_model,
                llm_base_url=args.llm_base_url,
                embedding_model=args.embedding_model,
                embedding_base_url=args.embedding_base_url,
                support_json_schema=args.support_json_schema,
                skip_retrieval_ppr=True,
            ),
        }

        try:
            pseudo_conversation = build_pseudo_conversation(example, row_index)
            config = create_emem_config(
                workspace_dir=workspace_dir,
                llm_model=args.llm_model,
                llm_base_url=args.llm_base_url,
                embedding_model=args.embedding_model,
                embedding_base_url=args.embedding_base_url,
                api_key=args.api_key,
                embedding_api_key=args.embedding_api_key,
                force_rebuild=args.force_rebuild,
                skip_retrieval_ppr=True,
                memory_top_k=5,
                support_json_schema=args.support_json_schema,
            )
            runtime = EMem(global_config=config)
            runtime.index_conversation(pseudo_conversation)
            record["edu_count"] = count_workspace_edus(workspace_dir, args.llm_model)
            record["build_status"] = "OK"
        except Exception as exc:
            record["error"] = str(exc)
            logger.exception("Build failed for example_id=%s", example_id)

        manifest_rows.append(record)
        if (
            len(manifest_rows) == len(rows)
            or len(manifest_rows) % progress_interval == 0
            or record["build_status"] != "OK"
        ):
            ok_count = sum(1 for item in manifest_rows if item["build_status"] == "OK")
            error_count = len(manifest_rows) - ok_count
            logger.info(
                "Build progress: %d/%d complete (ok=%d, error=%d)",
                len(manifest_rows),
                len(rows),
                ok_count,
                error_count,
            )

    write_jsonl(args.manifest_path, manifest_rows)
    logger.info(
        "Build finished: written_records=%d manifest=%s index_root=%s",
        len(manifest_rows),
        args.manifest_path,
        args.index_root,
    )
    print(json.dumps({"manifest_path": args.manifest_path, "records": len(manifest_rows)}, indent=2))


if __name__ == "__main__":
    main()
