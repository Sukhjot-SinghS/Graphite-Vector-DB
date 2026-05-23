#pragma once

// --- Vamana Graph Constants ---
const int R = 32;              // Maximum degree (neighbors per node)
const int L_BUILD = 40;        // Search list size during graph construction
const float ALPHA_1 = 1.0f;    // Pass 1: Local neighborhoods
const float ALPHA_2 = 1.2f;    // Pass 2: Long-range highways

// --- Product Quantization Constants ---

const int DIM = 384;           // all-MiniLM-L6-v2 dimension
const int PQ_CHUNKS = 96;      // Number of chunks to split vector into
const int CHUNK_DIM = DIM / PQ_CHUNKS; // Dimensions per chunk (8)
const int PQ_CENTROIDS = 256; // 1 byte per chunk
const int KMEANS_ITERS = 15;   // Training iterations