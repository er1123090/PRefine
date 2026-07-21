import tqdm
import os
import argparse
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# ---------------------------------------------------------
# Helper: Load Dataset
# ---------------------------------------------------------
def load_chains_dataset(fpath: str) -> pd.DataFrame:
    try:
        df = pd.read_json(fpath, lines=True)
        return df
    except ValueError:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "dataset" in data:
            return pd.DataFrame(data["dataset"])
        return pd.DataFrame(data)

# ---------------------------------------------------------
# RAG Ingestion Logic
# ---------------------------------------------------------
def run_ingestion(input_path: str, db_path: str, collection_name: str):
    # 1. Initialize ChromaDB Client
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.environ.get("OPENAI_API_KEY"),
        model_name="text-embedding-3-small"
    )
    
    client = chromadb.PersistentClient(path=db_path)
    
    # [FIX] get_or_create_collection 사용하여 없을 시 생성하도록 변경
    try:
        collection = client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)
        print(f"Loaded or Created collection: {collection_name}")
    except Exception as e:
        print(f"Error initializing collection: {e}")
        return

    # 2. Load Data
    df = load_chains_dataset(input_path)
    print(f"Starting RAG ingestion for {len(df)} users...")

    # 3. Process each user
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df), desc="Ingesting to VectorDB"):
        user_data = row.to_dict()
        user_id = str(user_data.get("example_id", "unknown_user"))

        # [Clean Slate] 해당 유저의 기존 데이터 삭제
        try:
            collection.delete(where={"user_id": user_id})
        except Exception:
            pass 

        sessions = user_data.get("sessions", [])
        
        documents = []
        metadatas = []
        ids = []
        
        doc_counter = 0

        for session in sessions:
            # 3-1. Process Dialogue Turns
            for turn in session.get("dialogue", []):
                role = turn["role"].lower()
                content = turn["message"]
                
                documents.append(f"{role}: {content}")
                metadatas.append({
                    "user_id": user_id, 
                    "type": "dialogue", 
                    "role": role,
                    "session_id": str(doc_counter)
                })
                ids.append(f"{user_id}_sess_{doc_counter}")
                doc_counter += 1

            # 3-2. Process API Calls (System Facts)
            api_calls = session.get("api_call", [])
            if api_calls:
                doc_text = f"[System Summary] API Calls executed in this session: {str(api_calls)}"
                documents.append(doc_text)
                metadatas.append({
                    "user_id": user_id, 
                    "type": "api_history", 
                    "role": "system",
                    "session_id": str(doc_counter)
                })
                ids.append(f"{user_id}_sess_{doc_counter}")
                doc_counter += 1

        # 4. Batch Add to Chroma
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
    
    print("Ingestion Complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/data/minseo/experiments4/data/1229_dev_6.json")
    parser.add_argument("--db_path", type=str, default="./chroma_db_rag", help="Path to save vector database")
    parser.add_argument("--collection_name", type=str, default="user_memories")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        print("[Error] OPENAI_API_KEY environment variable is required for embeddings.")
        exit(1)

    run_ingestion(args.input_path, args.db_path, args.collection_name)