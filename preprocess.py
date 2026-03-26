import os
import numpy as np
import pandas as pd
from collections import Counter

# --- CONFIG ---
NORMAL_DIR = "Training_Data_Master"
ATTACK_DIR = "Attack_Data_Master"  # optional attack folder; if missing we still process normal
N_GRAM_SIZE = 6        # sliding window size
MAX_SYSCALL = 350      # ADFA-LD has ~350 unique syscall IDs

def load_traces(folder):
    """Load all syscall trace files from a directory."""
    traces = []
    if not os.path.isdir(folder):
        return traces
    for root, _, files in os.walk(folder):
        for fname in files:
            fpath = os.path.join(root, fname)
            with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
                tokens = f.read().split()
            calls = []
            for token in tokens:
                try:
                    calls.append(int(token))
                except ValueError:
                    continue
            if calls:
                traces.append(calls)
    return traces

def extract_ngram_features(traces, n=N_GRAM_SIZE, max_syscall=MAX_SYSCALL):
    """
    Sliding window n-gram frequency vector.
    Each trace becomes a fixed-length feature vector.
    """
    feature_vectors = []
    for trace in traces:
        freq = Counter()
        for i in range(len(trace) - n + 1):
            gram = tuple(trace[i:i+n])
            freq[gram] += 1
        
        # Flatten to a fixed-size vector (use syscall ID ranges)
        # Simpler approach: use raw frequency of each syscall ID
        vec = np.zeros(max_syscall)
        for syscall in trace:
            if syscall < max_syscall:
                vec[syscall] += 1
        
        # Normalize
        total = vec.sum()
        if total > 0:
            vec = vec / total
        
        feature_vectors.append(vec)
    
    return np.array(feature_vectors)

# --- MAIN ---
print("Loading normal traces...")
normal_traces = load_traces(NORMAL_DIR)
print(f"  Loaded {len(normal_traces)} normal traces")

print("Loading attack traces...")
attack_traces = load_traces(ATTACK_DIR)
print(f"  Loaded {len(attack_traces)} attack traces")

print("Extracting features...")
X_normal = extract_ngram_features(normal_traces)
X_attack  = extract_ngram_features(attack_traces)

# Save processed features
os.makedirs("data/processed", exist_ok=True)
np.save("data/processed/X_normal.npy", X_normal)
  

print("[OK] Features saved to data/processed/")
print(f"   Normal shape: {X_normal.shape}")
