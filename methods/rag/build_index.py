import tqdm
import os
import argparse
import json
try:
    import pandas as pd
except ImportError:
    import sys
    from pathlib import Path

    _ROOT = Path(__file__).resolve().parents[2]
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))
    from src.exp4_runtime import tabular as pd

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
def run_ingestion(
    input_path: str,
    db_path: str,
    collection_name: str,
    embedding_model: str = "text-embedding-3-small",
    embedding_api_key: str | None = None,
    embedding_base_url: str | None = None,
):
    try:
        import chromadb
        from chromadb.utils import embedding_functions
    except ImportError as exc:
        raise RuntimeError("chromadb is required for RAG index construction") from exc

    # 1. Initialize ChromaDB Client
    if not embedding_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for embeddings.")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=embedding_api_key,
        api_base=embedding_base_url,
        model_name=embedding_model
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
    parser.add_argument("--input_path", type=str, default="/data/minseo/experiment8/data/MPT_v2_mix600.json")
    parser.add_argument("--db_path", type=str, default="./chroma_db_rag", help="Path to save vector database")
    parser.add_argument("--collection_name", type=str, default="user_memories")
    parser.add_argument("--base_url", type=str, default=None, help="Optional embeddings/base model endpoint override.")
    parser.add_argument("--api_key", type=str, default=None, help="Optional embeddings API key.")
    parser.add_argument("--embedding_base_url", type=str, default=None, help="Optional embedding-only endpoint override.")
    parser.add_argument("--embedding_api_key", type=str, default=None, help="Optional embedding-only API key.")
    parser.add_argument("--embedding_model", type=str, default="text-embedding-3-small")
    args = parser.parse_args()

    embedding_api_key = args.embedding_api_key or args.api_key
    embedding_base_url = args.embedding_base_url or args.base_url

    if not embedding_api_key:
        print("[Error] OPENAI_API_KEY environment variable is required for embeddings.")
        exit(1)

    run_ingestion(
        args.input_path,
        args.db_path,
        args.collection_name,
        embedding_model=args.embedding_model,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
    )
