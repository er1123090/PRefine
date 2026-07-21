import tqdm
import os
import argparse
from mem0 import MemoryClient
from utils_mem0 import load_chains_dataset 
# prepare_messages_for_mem0는 세션별 처리를 위해 메인 루프에서 직접 구현하므로 import 제외 가능

# Initialize mem0 Client
memory_client = MemoryClient(api_key=os.environ.get("MEM0_API_KEY"))

def run_ingestion(input_path: str):
    df = load_chains_dataset(input_path)
    print(f"Starting ingestion for {len(df)} users...")

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Ingesting Memories"):
        user_data = row.to_dict()
        user_id = str(user_data.get("example_id", "unknown_user"))

        # 1. Reset Memory for this user_id (Clean slate per user)
        # 해당 유저의 이전 기록을 모두 지우고 새로 시작
        try:
            memory_client.delete_all(user_id=user_id)
        except Exception:
            pass

        # 2. Iterate through each session (Dialogue ID level)
        sessions = user_data.get("sessions", [])
        
        for session in sessions:
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
            if mem0_messages:
                memory_client.add(mem0_messages, user_id=user_id)
    
    print("Ingestion Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data/minseo/experiments4/data/1229_dev_6.json")
    args = parser.parse_args()

    run_ingestion(args.input_path)