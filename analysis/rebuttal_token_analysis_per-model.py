
import json
import os
import glob
import tiktoken
import matplotlib.pyplot as plt
import numpy as np

# Setup tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    if not text:
        return 0
    return len(enc.encode(text))

# --- Task 1: Calculate Session Token Counts from Data ---
data_file_path = '/data/minseo/experiments5/data/1229_dev_6.json'
print(f"Reading data from {data_file_path}...")

with open(data_file_path, 'r') as f:
    data = json.load(f)

# Dictionary to store token counts per session index across all examples
# session_index -> list of cumulative token counts
session_data_tokens = {}

for example in data:
    cumulative_tokens = 0
    sessions = example.get('sessions', [])
    
    for i, session in enumerate(sessions):
        session_idx = i + 1
        
        # Concatenate messages in this session
        session_text = ""
        dialogue = session.get('dialogue', [])
        for turn in dialogue:
            role = turn.get('role', '')
            message = turn.get('message', '')
            session_text += f"{role}: {message}\n"
            
        # Count tokens for this session
        session_tokens = count_tokens(session_text)
        
        # Update cumulative tokens
        cumulative_tokens += session_tokens
        
        if session_idx not in session_data_tokens:
            session_data_tokens[session_idx] = []
        session_data_tokens[session_idx].append(cumulative_tokens)

# Calculate averages
avg_data_tokens = {}
for idx, counts in session_data_tokens.items():
    avg_data_tokens[idx] = np.mean(counts)

print("Average Data Cumulative Token Counts per Session:")
sorted_data_indices = sorted(avg_data_tokens.keys())
for idx in sorted_data_indices:
    print(f"Session {idx}: {avg_data_tokens[idx]:.2f}")


# --- Task 2: Calculate Implicit Preference Token Counts from Memory Logs ---
memory_base_dir = '/data/minseo/experiments5/methods/our_memory/inference/1231_MEMORY3'
print(f"\nScanning memory logs in {memory_base_dir}...")

# Dictionary to store model results: model_name -> { session_index -> list of token counts }
model_memory_tokens = {}

# Find all subdirectories
subdirs = [d for d in glob.glob(os.path.join(memory_base_dir, '*')) if os.path.isdir(d)]

for subdir in subdirs:
    model_name = os.path.basename(subdir)
    memory_file = os.path.join(subdir, '_memory1.jsonl')
    
    if not os.path.exists(memory_file):
        continue
        
    print(f"Processing {model_name}...")
    
    # Store counts for this model
    session_counts = {} # session_index -> list of counts
    
    with open(memory_file, 'r') as f:
        for line in f:
            try:
                record = json.loads(line)
                history = record.get('preference_evolution_history', [])
                
                for entry in history:
                    session_idx = entry.get('session_index')
                    if session_idx is None:
                        continue
                        
                    final_pref = entry.get('final_preference_at_session', {})
                    if final_pref is None:
                         implicit_pref = ""
                    else:
                        implicit_pref = final_pref.get('implicit_pref', "")
                    
                    tokens = count_tokens(implicit_pref)
                    
                    if session_idx not in session_counts:
                        session_counts[session_idx] = []
                    session_counts[session_idx].append(tokens)
            except json.JSONDecodeError:
                print(f"Error decoding JSON line in {memory_file}")
                continue

    # Calculate averages for this model
    avg_counts = {}
    for idx, counts in session_counts.items():
        avg_counts[idx] = np.mean(counts)
        
    model_memory_tokens[model_name] = avg_counts
    
    print(f"  Averages for {model_name}:")
    sorted_indices = sorted(avg_counts.keys())
    for idx in sorted_indices:
        print(f"    Session {idx}: {avg_counts[idx]:.2f}")


# --- Task 3: Plotting ---
output_plot_path = '/data/minseo/experiments5/analysis/token_evolution_plot.png'
print(f"\nGenerating plot at {output_plot_path}...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Subplot 1: Data Dialogue Context Accumulation
x_data = sorted(avg_data_tokens.keys())
y_data = [avg_data_tokens[i] for i in x_data]

ax1.plot(x_data, y_data, marker='o', linestyle='-', color='blue')
ax1.set_title("Dialogue Context Accumulation")
ax1.set_xlabel("Session Index")
ax1.set_ylabel("Average Cumulative Token Count")
ax1.grid(True)
ax1.set_xticks(x_data)

# Subplot 2: Memory Implicit Pref Token Count
colors = plt.cm.tab10(np.linspace(0, 1, len(model_memory_tokens)))

for i, (model_name, avg_counts) in enumerate(model_memory_tokens.items()):
    x_mem = sorted(avg_counts.keys())
    y_mem = [avg_counts[j] for j in x_mem]
    ax2.plot(x_mem, y_mem, marker='s', linestyle='-', label=model_name, color=colors[i])

ax2.set_title("Implicit Preference Token Count Evolution")
ax2.set_xlabel("Session Index")
ax2.set_ylabel("Average Implicit Pref Token Count")
ax2.legend()
ax2.grid(True)
# Determine ticks for ax2 based on all data
all_x_mem = set()
for counts in model_memory_tokens.values():
    all_x_mem.update(counts.keys())
if all_x_mem:
    ax2.set_xticks(sorted(list(all_x_mem)))

plt.tight_layout()
plt.savefig(output_plot_path)
print("Done.")
