import json
import glob
import tiktoken
import os

def calculate_avg_tokens():
    log_files = glob.glob('/data/minseo/experiments4/mem0/inference/multiturn/easy/0217*.log')
    print(f"Found files: {log_files}")
    
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Error loading tiktoken: {e}")
        return

    total_tokens = 0
    total_records = 0
    
    for file_path in log_files:
        print(f"Processing {file_path}")
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    retrieved_memories = data.get('retrieved_memories', [])
                    
                    # Convert list to string representation or concatenate
                    # Usually 'retrieved_memories' is a list of strings
                    # Let's verify the content
                    
                    tokens_for_record = 0
                    if isinstance(retrieved_memories, list):
                         for memory in retrieved_memories:
                             if isinstance(memory, str):
                                 tokens_for_record += len(enc.encode(memory))
                    elif isinstance(retrieved_memories, str):
                         tokens_for_record += len(enc.encode(retrieved_memories))
                    
                    total_tokens += tokens_for_record
                    total_records += 1
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line in {file_path}")
                except Exception as e:
                    print(f"Error processing line: {e}")

    if total_records > 0:
        avg_tokens = total_tokens / total_records
        print(f"Total tokens: {total_tokens}")
        print(f"Total records: {total_records}")
        print(f"Average tokens in retrieved_memories: {avg_tokens}")
    else:
        print("No records found.")

if __name__ == "__main__":
    calculate_avg_tokens()
