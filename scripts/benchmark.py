import struct
import time
import random
import numpy as np
import sys
import os

# Ensure Python can find your graphite_core.so file in the root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import graphite_core

NUM_QUERIES = 100
K = 5
L_SEARCH = 150  # The beam width for our search

print("📊 Starting Graphite Engine Benchmark Suite")
print("-" * 50)

# 1. Load the ground-truth data from the raw binary file
print("📂 Loading uncompressed vectors.bin for Brute Force comparison...")
with open("vectors.bin", "rb") as f:
    num_vectors, dim = struct.unpack('ii', f.read(8))
    raw_data = f.read()
    # Reshape into a fast numpy array
    all_vectors = np.frombuffer(raw_data, dtype=np.float32).reshape(num_vectors, dim)

print(f"✅ Loaded {num_vectors} vectors of dimension {dim}.")

# 2. Boot your C++ Engine
print("🚀 Booting Graphite Engine (Vamana PQ + SSD)...")
db = graphite_core.GraphiteEngine("pq_data.bin", "vamana_index.bin")

# 3. Generate random test queries
query_indices = random.sample(range(num_vectors), NUM_QUERIES)

total_engine_latency = 0
total_bf_latency = 0
total_hops = 0
total_disk_reads = 0
total_recall = 0.0

print(f"🎯 Running {NUM_QUERIES} random queries...")

for idx in query_indices:
    query_vec = all_vectors[idx]
    
    # --- A. EXACT BRUTE FORCE (The Gold Standard) ---
    # Using perf_counter for high-resolution nanosecond timing
    bf_start = time.perf_counter()
    distances = np.linalg.norm(all_vectors - query_vec, axis=1)
    true_top_k = set(np.argsort(distances)[:K])
    bf_end = time.perf_counter()
    
    # --- B. GRAPHITE ENGINE SEARCH (The Approximation) ---
    engine_start = time.perf_counter()
    result_tuple = db.search(query_vec.tolist(), K, L_SEARCH)
    engine_end = time.perf_counter()
    
    engine_top_k = set(result_tuple[0])
    hops = result_tuple[1]
    disk_reads = result_tuple[2]
    
    # --- C. CALCULATE METRICS ---
    bf_latency_ms = (bf_end - bf_start) * 1000
    engine_latency_ms = (engine_end - engine_start) * 1000
    
    # Recall = How many of the TRUE top K did our engine actually find?
    intersection = len(true_top_k.intersection(engine_top_k))
    recall = (intersection / K) * 100
    
    total_bf_latency += bf_latency_ms
    total_engine_latency += engine_latency_ms
    total_hops += hops
    total_disk_reads += disk_reads
    total_recall += recall

# 4. Final Report
avg_bf_latency = total_bf_latency / NUM_QUERIES
avg_engine_latency = total_engine_latency / NUM_QUERIES
avg_hops = total_hops / NUM_QUERIES
avg_reads = total_disk_reads / NUM_QUERIES
avg_recall = total_recall / NUM_QUERIES

# Prevent division by zero just in case the engine is absurdly fast
speedup = avg_bf_latency / avg_engine_latency if avg_engine_latency > 0 else 0

print("\n" + "=" * 50)
print("🏆 BENCHMARK RESULTS")
print("=" * 50)
print(f"Queries Run        : {NUM_QUERIES}")
print(f"Search Beam (L)    : {L_SEARCH}")
print(f"Target Top (K)     : {K}")
print("-" * 50)
print(f"NumPy Brute Force  : {avg_bf_latency:.2f} ms")
print(f"Graphite Engine    : {avg_engine_latency:.2f} ms")
print(f"Engine Speedup     : {speedup:.1f}x FASTER ⚡")
print("-" * 50)
print(f"Average Recall@{K}   : {avg_recall:.2f}%")
print(f"Average Graph Hops : {avg_hops:.0f}")
print(f"Average SSD Reads  : {avg_reads:.0f}")
print("=" * 50)