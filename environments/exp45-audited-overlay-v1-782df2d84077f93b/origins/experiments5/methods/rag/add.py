"""
RAG ingestion: load dialogue sessions and API calls into ChromaDB.

Requires:
    OPENAI_API_KEY  — used for text-embedding-3-small embeddings
"""

import argparse
import json
import os
import sys

import tqdm
import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.data_utils import load_chains_dataset


def run_ingestion(input_path: str, db_path: str, collection_name: str) -> None:
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small",
    )

    client = chromadb.PersistentClient(path=db_path)
    try:
        collection = client.get_or_create_collection(
            name=collection_name, embedding_function=openai_ef
        )
        print(f"Loaded/created collection: {collection_name}")
    except Exception as e:
        print(f"[Error] Failed to initialise collection: {e}")
        return

    df = load_chains_dataset(input_path)
    print(f"Ingesting {len(df)} users into ChromaDB...")

    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Ingesting"):
        user_data = row.to_dict()
        user_id = str(user_data.get("example_id", "unknown"))

        # Clear previous records for this user
        try:
            collection.delete(where={"user_id": user_id})
        except Exception:
            pass

        documents = []
        metadatas = []
        ids = []
        doc_counter = 0

        for session in user_data.get("sessions", []):
            for turn in session.get("dialogue", []):
                role = turn["role"].lower()
                content = turn["message"]
                documents.append(f"{role}: {content}")
                metadatas.append({"user_id": user_id, "type": "dialogue", "role": role,
                                   "session_id": str(doc_counter)})
                ids.append(f"{user_id}_d_{doc_counter}")
                doc_counter += 1

            api_calls = session.get("api_call", [])
            if api_calls:
                doc_text = f"[System Summary] API Calls: {api_calls}"
                documents.append(doc_text)
                metadatas.append({"user_id": user_id, "type": "api_history", "role": "system",
                                   "session_id": str(doc_counter)})
                ids.append(f"{user_id}_a_{doc_counter}")
                doc_counter += 1

        if documents:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)

    print("Ingestion complete.")


if __name__ == "__main__":
    _ROOT = os.path.join(os.path.dirname(__file__), "..", "..")

    parser = argparse.ArgumentParser(description="Ingest dialogue data into ChromaDB for RAG.")
    parser.add_argument("--input_path", default=os.path.join(_ROOT, "data", "dev.json"))
    parser.add_argument("--db_path", default="./chroma_db_rag",
                        help="Directory to persist the ChromaDB vector store.")
    parser.add_argument("--collection_name", default="user_memories")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("[Error] OPENAI_API_KEY is required for embeddings.")
        exit(1)

    run_ingestion(args.input_path, args.db_path, args.collection_name)
