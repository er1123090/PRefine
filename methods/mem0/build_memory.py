import tqdm
import os
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from mem0 import MemoryClient
except ImportError:
    MemoryClient = None
from utils_mem0 import load_chains_dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.construction_usage import (
    begin_usage_collection,
    end_usage_collection,
    record_response_usage,
    set_usage_session,
)
from src.token_measurement import (
    count_texts_tokens,
    encoding_metadata,
)

# prepare_messages_for_mem0는 세션별 처리를 위해 메인 루프에서 직접 구현하므로 import 제외 가능


def normalize_mem0_memories(response: Any) -> List[Dict[str, Any]]:
    if isinstance(response, list):
        return [item for item in response if isinstance(item, dict)]
    if isinstance(response, dict):
        for key in ("results", "memories", "data"):
            items = response.get(key)
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
    return []


def memory_texts(memories: List[Dict[str, Any]]) -> List[str]:
    rendered = []
    for memory in memories:
        value = (
            memory.get("memory")
            or memory.get("content")
            or memory.get("text")
        )
        if value:
            rendered.append(str(value))
    return rendered


def run_ingestion(
    input_path: str,
    metrics_output: str,
    token_encoding: str = "cl100k_base",
):
    if MemoryClient is None:
        raise RuntimeError("mem0ai is required for Mem0 memory construction")
    memory_client = MemoryClient(api_key=os.environ.get("MEM0_API_KEY"))
    df = load_chains_dataset(input_path)
    output = Path(metrics_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    print(f"Starting ingestion for {len(df)} users...")

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Ingesting Memories"):
        user_data = row.to_dict()
        user_id = str(user_data.get("example_id", "unknown_user"))
        usage_token = begin_usage_collection()
        session_exports = []

        # 1. Reset Memory for this user_id (Clean slate per user)
        # 해당 유저의 이전 기록을 모두 지우고 새로 시작
        try:
            memory_client.delete_all(user_id=user_id)
        except Exception:
            pass

        # 2. Iterate through each session (Dialogue ID level)
        sessions = user_data.get("sessions", [])
        
        for session_index, session in enumerate(sessions, start=1):
            set_usage_session(session_index)
            # dialogue_id = session.get("dialogue_id") # 필요시 메타데이터로 사용
            mem0_messages = []

            # 2-1. Process Dialogue
            for turn in session.get("dialogue", []):
                # Mem0는 role을 소문자(user, assistant)로 권장합니다.
                mem0_messages.append({
                    "role": turn["role"].lower(),
                    "content": turn["message"]
                })

            # 2-2. Process API Calls (dialogue 뒤에 컨텍스트로 추가)
            # dialogue와 api_call을 1번의 add로 넣기 위해 리스트 뒤에 붙입니다.
            api_calls = session.get("api_call", [])
            if api_calls:
                # API 호출 내역을 시스템 메시지나 어시스턴트의 요약 정보로 추가
                mem0_messages.append({
                    "role": "assistant",
                    "content": f"[System Summary] API Calls executed in this session: {str(api_calls)}"
                })

            # 3. Add to memory (Per Session Batch)
            # 세션 단위로 한 번에 add를 실행합니다.
            add_usage = None
            if mem0_messages:
                add_response = memory_client.add(mem0_messages, user_id=user_id)
                add_usage = record_response_usage(
                    add_response,
                    component="mem0_add",
                    provider="mem0",
                    model="managed",
                )

            memory_snapshot_error = None
            memories = []
            try:
                memories = normalize_mem0_memories(
                    memory_client.get_all(user_id=user_id)
                )
            except Exception as exc:
                memory_snapshot_error = f"{type(exc).__name__}: {exc}"

            session_exports.append(
                {
                    "session_index": session_index,
                    "dialogue_id": session.get("dialogue_id"),
                    "local_construction_input_tokens": count_texts_tokens(
                        (
                            f"{message.get('role', '')}: {message.get('content', '')}"
                            for message in mem0_messages
                        ),
                        token_encoding,
                    ),
                    "memory_count_after_session": len(memories),
                    "stored_memory_tokens_after_session": count_texts_tokens(
                        memory_texts(memories),
                        token_encoding,
                    ),
                    "mem0_add_provider_usage": add_usage,
                    "memory_snapshot_error": memory_snapshot_error,
                }
            )

        construction_token_usage = end_usage_collection(usage_token)
        known_input_tokens = [
            item["local_construction_input_tokens"]
            for item in session_exports
            if item["local_construction_input_tokens"] is not None
        ]
        result = {
            "example_id": user_id,
            "method": "mem0",
            "total_sessions_processed": len(sessions),
            "session_exports": session_exports,
            "local_construction_input_tokens": (
                sum(known_input_tokens) if known_input_tokens else None
            ),
            "construction_token_usage": construction_token_usage,
            "token_counts": construction_token_usage["summary"],
            **encoding_metadata(token_encoding),
        }
        with output.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print("Ingestion Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data/minseo/experiment8/data/MPT_v2_mix600.json")
    parser.add_argument(
        "--metrics_output",
        default="/data/minseo/experiment8/outputs/mem0/construction_metrics.jsonl",
    )
    parser.add_argument("--token_encoding", default="cl100k_base")
    args = parser.parse_args()

    run_ingestion(args.input_path, args.metrics_output, args.token_encoding)
