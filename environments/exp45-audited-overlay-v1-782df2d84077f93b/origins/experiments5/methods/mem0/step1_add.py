"""
mem0 — Step 1: Ingest session dialogues and API calls into mem0.

Requires:
    MEM0_API_KEY  — API key for the mem0 cloud service
"""

import argparse
import os
import sys

import tqdm
from mem0 import MemoryClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data_utils import load_chains_dataset


def run_ingestion(input_path: str) -> None:
    memory_client = MemoryClient(api_key=os.environ["MEM0_API_KEY"])
    df = load_chains_dataset(input_path)
    print(f"Ingesting {len(df)} users into mem0...")

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Ingesting"):
        user_data = row.to_dict()
        user_id = str(user_data.get("example_id", "unknown"))

        try:
            memory_client.delete_all(user_id=user_id)
        except Exception:
            pass

        for session in user_data.get("sessions", []):
            messages = []

            for turn in session.get("dialogue", []):
                messages.append({
                    "role": turn["role"].lower(),
                    "content": turn["message"],
                })

            api_calls = session.get("api_call", [])
            if api_calls:
                messages.append({
                    "role": "assistant",
                    "content": f"[System Summary] API Calls executed in this session: {api_calls}",
                })

            if messages:
                memory_client.add(messages, user_id=user_id)

    print("Ingestion complete.")


if __name__ == "__main__":
    _ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

    parser = argparse.ArgumentParser(description="Ingest session data into mem0.")
    parser.add_argument("--input_path", default=os.path.join(_ROOT, "data", "dev.json"))
    args = parser.parse_args()

    if not os.environ.get("MEM0_API_KEY"):
        print("[Error] MEM0_API_KEY environment variable is not set.")
        exit(1)

    run_ingestion(args.input_path)
