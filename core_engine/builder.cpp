#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <queue>
#include <set>
#include <random>
#include <numeric>
#include "config.h" 
#include <immintrin.h>
#include <cmath>
#include <cfloat>


using namespace std;


vector<vector<vector<float>>> codebooks(PQ_CHUNKS, vector<vector<float>>(PQ_CENTROIDS, vector<float>(CHUNK_DIM)));
struct Header {
    int num_vectors;
    int dim;
};

// Simple Vector structure
struct node {
    int id;
    vector<float> vec;       // 384 Floats (Original)
    vector<uint8_t> pq_code; // 48 Bytes (The Compressed Code!)
    vector<int> neighbors;
};



void train_pq(const vector<node>& nodes, vector<vector<vector<float>>>& codebooks) {

    // Loop once per chunk (48 total)
    for (int chunk = 0; chunk < PQ_CHUNKS; chunk++) {
        int start_dim = chunk * CHUNK_DIM;  // e.g. chunk 0 → dim 0, chunk 1 → dim 8, etc.

        // ── Step 1: Extract the 8-dim slice from every vector ──────────────────
        // For chunk 5: grab dims [40..47] from every node in the database.
        // Each 'point' here is an 8-dimensional vector.
        vector<vector<float>> points;
        points.reserve(nodes.size());

        for (const auto& node : nodes) {
            vector<float> chunk_slice(
                node.vec.begin() + start_dim,
                node.vec.begin() + start_dim + CHUNK_DIM
            );
            points.push_back(chunk_slice);
        }

        // ── Step 2: Initialize 256 random centroids ────────────────────────────
        // Pick 256 random points from the dataset as starting centroids.
        // (Random init — good enough for this; real systems use K-Means++)
        vector<vector<float>> centroids(PQ_CENTROIDS);
        vector<int> indices(points.size());
        iota(indices.begin(), indices.end(), 0);  // fill [0, 1, 2, ..., N-1]
        shuffle(indices.begin(), indices.end(), mt19937{random_device{}()});

        for (int c = 0; c < PQ_CENTROIDS; c++) {
            centroids[c] = points[indices[c]];
        }

        // ── Step 3: K-Means loop (15 iterations) ───────────────────────────────
        vector<int> assignments(points.size());

        for (int iter = 0; iter < 15; iter++) {

            // --- Assign: each point → nearest centroid ---
            for (int i = 0; i < (int)points.size(); i++) {
                float best_dist = numeric_limits<float>::max();
                int   best_id   = 0;

                for (int c = 0; c < PQ_CENTROIDS; c++) {
                    float dist = 0.0f;
                    for (int d = 0; d < CHUNK_DIM; d++) {
                        float diff = points[i][d] - centroids[c][d];
                        dist += diff * diff;  // squared L2 (no sqrt needed for comparison)
                    }
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_id   = c;
                    }
                }
                assignments[i] = best_id;
            }

            // --- Move: each centroid → mean of its assigned points ---
            vector<vector<float>> new_centroids(PQ_CENTROIDS, vector<float>(CHUNK_DIM, 0.0f));
            vector<int>           counts(PQ_CENTROIDS, 0);

            for (int i = 0; i < (int)points.size(); i++) {
                int c = assignments[i];
                counts[c]++;
                for (int d = 0; d < CHUNK_DIM; d++) {
                    new_centroids[c][d] += points[i][d];
                }
            }

            // Divide by count to get the mean.
            // If a centroid has no points assigned, keep the old centroid (avoid NaN).
            for (int c = 0; c < PQ_CENTROIDS; c++) {
                if (counts[c] > 0) {
                    for (int d = 0; d < CHUNK_DIM; d++) {
                        new_centroids[c][d] /= counts[c];
                    }
                } else {
                    new_centroids[c] = centroids[c];  // keep old centroid
                }
            }

            centroids = move(new_centroids);
        }

        // ── Step 4: Save the 256 trained centroids for this chunk ──────────────
        codebooks[chunk] = centroids;
        // codebooks[chunk][42] is now an 8-float vector: the learned centroid #42
    }
}


void compress_database(vector<node>& nodes, const vector<vector<vector<float>>>& codebooks) {

    // Loop through every node in the database
    for (auto& node : nodes) {

        node.pq_code.clear();
        node.pq_code.reserve(PQ_CHUNKS);

        // Loop through all 48 chunks
        for (int chunk = 0; chunk < PQ_CHUNKS; chunk++) {
            int start_dim = chunk * CHUNK_DIM;

            // Find closest centroid: L2 distance from this chunk's slice
            // to each of the 256 centroids in codebooks[chunk]
            float best_dist = numeric_limits<float>::max();
            uint8_t best_id = 0;

            for (int c = 0; c < PQ_CENTROIDS; c++) {
                float dist = 0.0f;
                for (int d = 0; d < CHUNK_DIM; d++) {
                    float diff = node.vec[start_dim + d] - codebooks[chunk][c][d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_id   = (uint8_t)c;
                }
            }

            // Store just the 1-byte centroid ID instead of 32 bytes of floats
            node.pq_code.push_back(best_id);
        }
        // node.pq_code is now 48 bytes — the compressed representation of this vector
    }
}


// --- HELPER: Load Vectors from Binary File ---
vector<node> load_data(const string& filename, int& dim) {
    ifstream infile(filename, ios::binary);
    if (!infile) {
        cerr << "Error: Could not open " << filename << endl;
        exit(1);
    }

    Header header;
    infile.read((char*)&header, sizeof(Header));
    dim = header.dim;

    vector<node> nodes(header.num_vectors);
    for (int i = 0; i < header.num_vectors; i++) {
        nodes[i].id = i;
        nodes[i].vec.resize(dim);
        infile.read((char*)nodes[i].vec.data(), dim * sizeof(float));
    }
    return nodes;
}

// --- PASS 1 CORE LOGIC: Random Graph Initialization ---
void init_random_graph(vector<node>& nodes) {
    int N = nodes.size();
    
    // Random number generator
    random_device rd;
    mt19937 gen(rd());

    cout << "Initializing PASS_1 Random Graph (R=" << R << ")..." << endl;

    for (int i = 0; i < N; i++) {
        // We use a set to avoid duplicate connections
        set<int> unique_neighbors;
        while ((int)unique_neighbors.size() < R && (int)unique_neighbors.size() < N - 1) {
            int target = gen() % N;
            
            // Don't connect to yourself, and don't duplicate
            if (target != i) {
                unique_neighbors.insert(target);
            }
        }
        // Convert set to vector
        nodes[i].neighbors.assign(unique_neighbors.begin(), unique_neighbors.end());
    }
}
// Compute Euclidean (L2) distance between two float vectors using AVX2
float get_distance_avx(const float* veca, const float* vecb, int dim) {
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    for (; i + 7 < dim; i += 8) {
        __m256 a = _mm256_loadu_ps(veca + i);
        __m256 b = _mm256_loadu_ps(vecb + i);
        __m256 diff = _mm256_sub_ps(a, b);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }
    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    float final_sum = 0.0f;
    for (int j = 0; j < 8; ++j) {
        final_sum += temp[j];
    }
    // Process remaining elements, if any
    for (; i < dim; ++i) {
        float diff = veca[i] - vecb[i];
        final_sum += diff * diff;
    }
    return std::sqrt(final_sum);
}

// --- HELPER: Find Candidates (Greedy Search) ---
vector<int> get_candidates(int target_id, const vector<node>& nodes, int ep, int L) {
    vector<int> candidates;
    vector<bool> visited(nodes.size(), false);
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> pq;
    
    float initial_dist = get_distance_avx(nodes[target_id].vec.data(), nodes[ep].vec.data(),nodes[ep].vec.size());
    pq.push({initial_dist, ep});
    visited[ep] = true;
    
    while(!pq.empty() && (int)candidates.size() < L) {
        int curr = pq.top().second;
        pq.pop();
        candidates.push_back(curr);
        
        for(int neighbor : nodes[curr].neighbors) {
            if(!visited[neighbor]) {
                visited[neighbor] = true;
                float d = get_distance_avx(nodes[target_id].vec.data(), nodes[neighbor].vec.data(),nodes[ep].vec.size());
                pq.push({d, neighbor});
            }
        }
    }
    return candidates;
}
// --- PASS 2 CORE LOGIC: Robust Prune ---
void robust_prune(node& p, vector<int>& candidates, const vector<node>& all_nodes, float alpha, int R) {
    // 1. Add current neighbors to candidates
    for (int neighbor_id : p.neighbors) {
        candidates.push_back(neighbor_id);
    }

    // 2. Remove duplicates and yourself
    sort(candidates.begin(), candidates.end());
    candidates.erase(unique(candidates.begin(), candidates.end()), candidates.end());
    auto it = find(candidates.begin(), candidates.end(), p.id);
    if (it != candidates.end()) candidates.erase(it);

    // 3. Sort candidates by distance to P (Closest first)
    sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        // Compare candidates 'a' and 'b' based on their Euclidean distance (L2) from node 'p' using AVX.
        // Need to use .data() and provide dimension explicitly for get_distance_avx (expects raw pointers).
        return get_distance_avx(p.vec.data(), all_nodes[a].vec.data(), (int)p.vec.size()) 
             < get_distance_avx(p.vec.data(), all_nodes[b].vec.data(), (int)p.vec.size());
    });

    // 4. The Pruning Loop
    p.neighbors.clear(); 
    int idx = 0 ; 
    while (idx< (int)candidates.size() && (int)p.neighbors.size()<R) {
        int new_neighbor = candidates[idx++];
        bool keep = true;
        for (int existing_neighbor : p.neighbors) {
            float dist_p_new = get_distance_avx(p.vec.data(), all_nodes[new_neighbor].vec.data(), (int)p.vec.size());
            float dist_existing_new = get_distance_avx(all_nodes[existing_neighbor].vec.data(), all_nodes[new_neighbor].vec.data(), (int)all_nodes[new_neighbor].vec.size());
   
            // THE VAMANA EQUATION: Cut if a better detour exists
            if (dist_p_new > (alpha * alpha) * dist_existing_new) {
                keep = false;
                break;
            }
        }
        if (keep) p.neighbors.push_back(new_neighbor);
    }
}

// Computes the index of the node whose embedding vector has the minimum sum of distances to all other nodes
int find_medoid(const vector<node>& nodes) {
    int N = nodes.size();
    
    // 1. O(N) - Find the mathematical average of all vectors
    vector<float> centroid(DIM, 0.0f);
    for (const auto& n : nodes) {
        for (int d = 0; d < DIM; d++) {
            centroid[d] += n.vec[d];
        }
    }
    for (int d = 0; d < DIM; d++) {
        centroid[d] /= N;
    }
    
    // 2. O(N) - Find the actual node closest to that average
    int medoid_id = 0;
    float min_dist = get_distance_avx(centroid.data(), nodes[0].vec.data(), DIM);
    
    for (int i = 1; i < N; i++) {
        float dist = get_distance_avx(centroid.data(), nodes[i].vec.data(), DIM);
        if (dist < min_dist) {
            min_dist = dist;
            medoid_id = i;
        }
    }
    
    cout << "🎯 Medoid found: Node " << medoid_id << " (O(N) calculation!)" << endl;
    return medoid_id;
}

// --- THE VAMANA BUILD LOOP ---
void build_vamana(vector<node>& nodes, float alpha, int L, int R,int ep) {
    int N = nodes.size();
    
    cout << " Running Vamana Build (Alpha = " << alpha << ")" << endl;
    
    for (int i = 0; i < N; i++) {
        // 1. Search the graph to find L candidates
        vector<int> candidates = get_candidates(i, nodes, ep, L);
        
        // 2. Prune our own edges using the Alpha parameter
        robust_prune(nodes[i], candidates, nodes, alpha, R);
        
        // 3. Bidirectional Connections (If I add you, you add me)
        for (int neighbor_id : nodes[i].neighbors) {
            nodes[neighbor_id].neighbors.push_back(i);
            vector<int> neighbor_candidates = nodes[neighbor_id].neighbors;
            // Prune the neighbor so they don't exceed the R limit
            robust_prune(nodes[neighbor_id], neighbor_candidates, nodes, alpha, R);
        }
        if (i % 2000 == 0) cout << "  Progress: " << i << "/" << N << " nodes processed..." << endl;
    }
}

// --- SAVING THE GRAPH to binary file ---
void save_graph(const vector<node>& nodes, const string& filename, int medoid_id) {
    ofstream outfile(filename ,ios::binary);
    if (!outfile) {
        cerr << "Error: Could not save graph to " << filename << endl;
        exit(1);
    }
    int num_nodes = nodes.size();
    outfile.write((char*)&num_nodes, sizeof(int));
    outfile.write((char*)&medoid_id, sizeof(int));
    
    vector<long long> offsets(num_nodes);
    
    long long header_size = (2 * sizeof(int)) + (num_nodes * sizeof(long long));
    outfile.seekp(header_size);

    for(int i = 0 ; i<num_nodes;i++){
        offsets[i] = outfile.tellp();
        int degree = nodes[i].neighbors.size();
        outfile.write((char*)&degree , sizeof(int));
        outfile.write((char*)nodes[i].neighbors.data(),degree * sizeof(int));
    }
    
    outfile.seekp(2 * sizeof(int));
    outfile.write((char*)offsets.data(),num_nodes*sizeof(long long));

    cout<<"Graph is now saved to "<<filename<<endl;
}



void save_pq_data(const vector<node>& nodes, const vector<vector<vector<float>>>& codebooks, const string& filename,int medoid_id ) {
    ofstream outfile(filename, ios::binary);
    if (!outfile) {
        cerr << "Error opening " << filename << endl;
        exit(1);
    }

    // 1. Write the Dimensions/Constants as a mini-header
    int num_nodes = nodes.size();
    outfile.write((char*)&num_nodes, sizeof(int));

    outfile.write((char*)&medoid_id, sizeof(int));
    int chunks = PQ_CHUNKS;
    outfile.write((char*)&chunks, sizeof(int));

    // 2. Write the entire Codebook (The Centroids)
    // You need this in RAM later to calculate distances!
    for (int chunk = 0; chunk < PQ_CHUNKS; chunk++) {
        for (int c = 0; c < PQ_CENTROIDS; c++) {
            // Write the 8 floats for this specific centroid
            outfile.write((char*)codebooks[chunk][c].data(), CHUNK_DIM * sizeof(float));
        }
    }

    // 3. Write the Compressed Codes for every node
    for (int i = 0; i < num_nodes; i++) {
        // Write the 48-byte vector of uint8_t
        outfile.write((char*)nodes[i].pq_code.data(), PQ_CHUNKS * sizeof(uint8_t));
    }
    cout << " PQ Data (Codebooks + Compressed Vectors) saved to " << filename << endl;
}



int main() {
    int dim;
    string input_file = "vectors.bin"; // Make sure your path is correct!
    
    cout << "1. Loading Data..." << endl;
    vector<node> nodes = load_data(input_file, dim);
    cout << "   Loaded " << nodes.size() << " vectors (Dim: " << dim << ")" << endl;

    // Vamana Configuration Parameters
    int R = 32; // Max neighbors per node
    int L = 50; // Search window size (Beam width)

    // --- THE VAMANA ALGORITHM ---
    
    // Pass 0: The Primordial Chaos
    init_random_graph(nodes);
    int medoid_id = find_medoid(nodes);
    // Pass 1: Build Local Streets (Alpha = 1.0)
    build_vamana(nodes, 1.0f, L, R, medoid_id);
                
    // Pass 2: Build Long-Range Highways (Alpha = 1.2)
    build_vamana(nodes, 1.2f, L, R,medoid_id);

    // --- SAVE THE RESULTS ---
    save_graph(nodes, "vamana_index.bin",medoid_id);
    cout << " Vamana Index Built Successfully!" << endl;

    // ← ADD THESE TWO
    cout << "2. Training PQ Codebooks..." << endl;
    train_pq(nodes, codebooks);

    cout << "3. Compressing database..." << endl;
    compress_database(nodes, codebooks);

    save_pq_data(nodes, codebooks, "pq_data.bin", medoid_id);
    cout << " Vamana Index & PQ Compression Built Successfully!" << endl;

    return 0;
}