from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import time
import graphite_core 

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Graphite Vector DB API")

# --- ADD THIS TO ALLOW WEB BROWSER ACCESS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------
print("🤖 Loading AI Embedding Model...")
ai_model = SentenceTransformer('all-MiniLM-L6-v2')



print("🚀 Booting C++ Graphite Engine...")
db = graphite_core.GraphiteEngine("pq_data.bin", "vamana_index.bin")
print("✅ Engine Ready!")

# --- ADD THIS WARM-UP ROUTINE ---
print("🔥 Warming up AI Model and OS Page Cache...")
start_warmup = time.time()

# 1. Force the AI model to do its heavy first-time memory allocation
warmup_vector = ai_model.encode("Warming up the engine").tolist()

# 2. Force the C++ engine to pull the SSD data into RAM
db.search(warmup_vector, 5, 150)

end_warmup = time.time()
print(f"✅ Server fully warmed up in {round((end_warmup - start_warmup)*1000)}ms. Ready for instant queries!")

# --- LOAD THE ID MAPPING ---
print("📖 Loading CodeAlpaca Mapping...")
id_to_text = {}

# Use <SEP> to split, because code uses | for bitwise OR!
try:
    with open("id_mapping.txt", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("<SEP>") 
            if len(parts) >= 2:
                node_id = int(parts[0])
                text_data = parts[1] 
                id_to_text[node_id] = text_data
    print(f"✅ Loaded {len(id_to_text)} code snippets!")
except FileNotFoundError:
    print("⚠️ Warning: id_mapping.txt not found. Did you run the ingestion script yet?")

# --- API ENDPOINTS ---

class SearchRequest(BaseModel):
    query: str
    k: int = 5
    L: int = 150

@app.get("/health")
def health_check():
    return {"status": "Online", "engine": "Graphite Hybrid (PQ + SSD)", "dataset": "CodeAlpaca 20k"}

@app.post("/search")
def search_database(req: SearchRequest):
    try:
        # A. Start the stopwatch
        start_time = time.time()
        
        # B. AI Translation: English text -> 384-dimensional float array
        vector = ai_model.encode(req.query).tolist()
        
        # C. The Pybind Bridge: Python hands the list to C++ and waits
        # --- UPDATED: Unpack all 3 elements from the C++ tuple ---
        result_tuple = db.search(vector, req.k, req.L)
        
        result_ids = result_tuple[0]
        hops = result_tuple[1]
        disk_reads = result_tuple[2] # <-- NEW: Capturing actual disk reads
        
        # D. Translate IDs to human-readable Code Snippets
        human_readable_results = []
        for node_id in result_ids:
            text_match = id_to_text.get(node_id, f"Unknown Node ID: {node_id}")
            human_readable_results.append({
                "id": node_id,
                "text": text_match
            })
        
        # E. Stop the stopwatch
        end_time = time.time()
        latency_ms = round((end_time - start_time) * 1000, 2)
        
        return {
            "query": req.query,
            "results": human_readable_results,
            "stats": {
                "latency_ms": latency_ms,
                "graph_hops": hops,
                "disk_reads": disk_reads # <-- NEW: Sending integer to frontend
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))