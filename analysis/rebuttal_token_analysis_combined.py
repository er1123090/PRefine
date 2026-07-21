import json
import glob
import os
import tiktoken
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

def count_tokens(text, enc):
    if not text:
        return 0
    return len(enc.encode(text))

def process_dialogue_data(file_path, enc):
    """
    Simulates accumulating dialogue history (Baseline).
    Returns: dict {session_index: average_token_count}
    """
    print(f"Processing dialogue data from {file_path}...")
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return {}
    
    session_tokens_list = defaultdict(list)
    
    for item in data:
        sessions = item.get('sessions', [])
        accumulated_text = ""
        for i, session in enumerate(sessions):
            dialogue = session.get('dialogue', [])
            session_text = ""
            for turn in dialogue:
                role = turn.get('role', '')
                message = turn.get('message', '')
                session_text += f"{role}: {message}\n"
            
            accumulated_text += session_text
            tokens = count_tokens(accumulated_text, enc)
            session_tokens_list[i + 1].append(tokens)
            
    avg_tokens = {}
    for session_idx, token_counts in session_tokens_list.items():
        if token_counts:
            avg_tokens[session_idx] = np.mean(token_counts)
            
    return avg_tokens

def process_memory_data_per_model(base_path, enc):
    """
    Extracts implicit preference memory tokens per model (Ours).
    Returns: dict {model_name: {session_index: average_token_count}}
    """
    print(f"Processing memory data from {base_path}...")
    model_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    model_avg_tokens = {}

    for model_dir in model_dirs:
        memory_file = os.path.join(base_path, model_dir, '_memory1.jsonl')
        if not os.path.exists(memory_file):
            continue
            
        print(f"Processing {model_dir}...")
        
        # Store counts per session index for this model
        session_tokens_list = defaultdict(list)

        try:
            with open(memory_file, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                        history = item.get('preference_evolution_history', [])
                        
                        for entry in history:
                            session_idx = entry.get('session_index')
                            final_pref = entry.get('final_preference_at_session', {})
                            if final_pref is None:
                                implicit_pref = ""
                            else:
                                implicit_pref = final_pref.get('implicit_pref', "")
                            
                            tokens = count_tokens(implicit_pref, enc)
                            session_tokens_list[session_idx].append(tokens)
                            
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {memory_file}: {e}")
            continue

        # Average for this model
        avg_tokens = {}
        for session_idx, token_counts in session_tokens_list.items():
            if token_counts:
                avg_tokens[session_idx] = np.mean(token_counts)
        
        model_avg_tokens[model_dir] = avg_tokens
            
    return model_avg_tokens

def main():
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception as e:
        print(f"Error loading tiktoken: {e}")
        return

    # 1. Process Baseline (Dialogue)
    dialogue_file = '/data/minseo/experiments5/data/1229_dev_6.json'
    dialogue_avg = process_dialogue_data(dialogue_file, enc)
    
    # 2. Process Ours (Memory) - Per Model
    memory_base_path = '/data/minseo/experiments5/methods/our_memory/inference/1231_MEMORY3_plot'
    memory_per_model = process_memory_data_per_model(memory_base_path, enc)
    
    if not dialogue_avg and not memory_per_model:
        print("No data found.")
        return

    # Calculate global average for Memory (Ours)
    all_memory_sessions = set()
    for m_avg in memory_per_model.values():
        all_memory_sessions.update(m_avg.keys())
    
    memory_agg_avg = {}
    for session_idx in all_memory_sessions:
        vals = []
        for m_avg in memory_per_model.values():
            if session_idx in m_avg:
                vals.append(m_avg[session_idx])
        if vals:
            memory_agg_avg[session_idx] = np.mean(vals)
    
    # Setup Plot
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 7))
    
    # Get all x values
    all_sessions_set = set(dialogue_avg.keys()) | set(memory_agg_avg.keys())
    for m_avg in memory_per_model.values():
        all_sessions_set.update(m_avg.keys())
    all_sessions = sorted(list(all_sessions_set))

    # Plot Dialogue Baseline
    dialogue_vals = [dialogue_avg.get(s, np.nan) for s in all_sessions]
    plt.plot(all_sessions, dialogue_vals, marker='o', label='Accumulated Dialogue Context', color='black', linewidth=3.0, zorder=10)

    # Plot Average Memory (Bold Red)
    memory_vals = [memory_agg_avg.get(s, np.nan) for s in all_sessions]
    plt.plot(all_sessions, memory_vals, marker='s', label='PREFINE (Avg)', color='red', linewidth=4.0, linestyle='--', zorder=11)

    # Use symlog scale to better visualize small values (Ours ~25) alongside large values.
    # linthresh sets the linear scale range around zero. Here, 0-50 will be linear-ish, >50 logarithmic.
    plt.yscale('symlog', linthresh=50)

    # Set custom ticks to make the scale readable
    yticks = [0, 10, 25, 50, 100, 500, 1000, 5000, 10000]
    plt.yticks(yticks, [str(y) for y in yticks])

    plt.xlabel('Session Index', fontsize=12)
    plt.ylabel('Average Token Count (Symlog Scale)', fontsize=12)
    plt.title('Token Growth Comparison: Accumulated Dialogue vs. PREFINE Memory', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xticks(all_sessions) 
    
    output_png = '/data/minseo/experiments5/analysis/token_growth_comparison_combined.png'
    plt.savefig(output_png, dpi=300)
    print(f"Plot saved to {output_png}")

if __name__ == "__main__":
    main()
