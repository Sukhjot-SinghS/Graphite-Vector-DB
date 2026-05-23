import struct
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import time

DIM = 384

print("📥 Downloading CodeAlpaca 20k (Senior Developer dataset)...")
# Load the famous CodeAlpaca dataset from HuggingFace
dataset = load_dataset("sahil2801/CodeAlpaca-20k", split="train")

texts = []
for row in dataset:
    instruction = row['instruction']
    input_code = row.get('input', '')
    output_code = row['output']
    
    # If there's specific input code provided in the prompt, include it
    if input_code:
        combined_text = f"Q: {instruction} Code: {input_code} | A: {output_code}"
    else:
        combined_text = f"Q: {instruction} | A: {output_code}"
        
    texts.append(combined_text)

NUM_VECTORS = len(texts)
print(f"✅ Loaded {NUM_VECTORS} coding Q&A pairs.")

# 1. Load the AI Model
print(f"🤖 Booting AI Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2') 

# 2. Embed the Text
print(f"⚡ Translating {NUM_VECTORS} coding problems into 384-dimensional math...")
start_time = time.time()
embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype(np.float32)
end_time = time.time()
print(f"✅ Embedded in {round(end_time - start_time, 2)} seconds.")

# 3. Save to Binary File (For builder.cpp)
FILENAME = "vectors.bin"
print(f"💾 Saving {NUM_VECTORS} vectors to {FILENAME}...")
with open(FILENAME, "wb") as f:
    f.write(struct.pack('ii', NUM_VECTORS, DIM))
    embeddings.tofile(f)

# 4. Save the ID Mapping (For main.py)
print("📖 Saving ID mapping text file...")
with open("id_mapping.txt", "w", encoding="utf-8") as f:
    for i, text in enumerate(texts):
        # Clean up newlines so it stays on one line in our text file
        clean_text = text.replace('\n', ' ').replace('\r', '')
        # Using <SEP> as the delimiter because code snippets have lots of pipes (|) in them!
        f.write(f"{i}<SEP>{clean_text}\n")

print("\n🚀 SUCCESS! Your CodeAlpaca dataset is ready.")
print("Next steps:")
print("1. Run ./builder to train K-Means and compress the 20k vectors.")