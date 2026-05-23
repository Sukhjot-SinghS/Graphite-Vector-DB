#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
// --- LINUX POSIX HEADERS REPLACING WINDOWS.H ---
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <immintrin.h>
#include<cstdint>
#include <fstream>
#include "config.h"
using namespace std;
struct Header {
    int num_vectors;
    int dim;
};

struct node {
    int id;
    vector<uint8_t> pq_code;
    
};
// The Translation Dictionary
vector<vector<vector<float>>> codebooks;
// --- POSIX DISK-NATIVE GRAPH ACCESS ---
struct DiskGraph {
    int num_nodes;
    int medoid_id;
    long long* offsets; 
    char* base_ptr;    
    
    int fd; // Linux uses file descriptors (integers) instead of Windows HANDLEs
    size_t file_size;

    void map_file(const char* path) {
        // 1. Open the file
        fd = open(path, O_RDONLY);
        if (fd == -1) {
            cerr << "Could not open file!" << endl;
            exit(1);
        }

        // 2. Get the file size (needed for Linux mmap)
        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            cerr << "Could not get file size!" << endl;
            close(fd);
            exit(1);
        }
        file_size = sb.st_size;

        // 3. Map the file directly into memory
        base_ptr = (char*)mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base_ptr == MAP_FAILED) {
            cerr << "Could not map view!" << endl;
            close(fd);
            exit(1);
        }

        // Read the header and jump table
        int* int_ptr = (int*)base_ptr;
        num_nodes = int_ptr[0];
        medoid_id = int_ptr[1];

        offsets = (long long*)(base_ptr + 2*sizeof(int));
    }

    // O(1) Instant Jump to SSD data (Remains EXACTLY the same!)
    pair<int*, int> get_neighbors(int node_id) {
        long long pos = offsets[node_id];
        int* node_start = (int*)(base_ptr + pos);
        int degree = *node_start;
        int* neighbors_ptr = node_start + 1;
        return {neighbors_ptr, degree};
    }
    

    // Cleanup for Linux
    void unmap() {
        munmap(base_ptr, file_size);
        close(fd);
    }
};

float dist_sq(const vector<float>  &veca ,const vector<float> &vecb ){
    __m256 sum_vec = _mm256_setzero_ps();
    int k = veca.size();
    for(int i = 0 ; i<k ;i+=8){
        __m256 a  = _mm256_loadu_ps(veca.data() + i);
        __m256 b  = _mm256_loadu_ps(vecb.data() + i);
        __m256 diff = _mm256_sub_ps(a,b);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
    }
    float final_sum = 0.0f;
    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    for(int i = 0 ; i<8;i++){final_sum += temp[i];}
    return sqrt(final_sum);

}
vector<node> load_pq_data(const string& filename) {
    ifstream infile(filename, ios::binary);
    if (!infile) { cerr << "Error opening " << filename << endl; exit(1); }

    // FIX: Add medoid_id here so the pointer advances correctly!
    int num_nodes, medoid_id, chunks;
    infile.read((char*)&num_nodes, sizeof(int));
    infile.read((char*)&medoid_id, sizeof(int)); 
    infile.read((char*)&chunks, sizeof(int));

    // 1. Load the Codebooks into RAM
    codebooks.assign(PQ_CHUNKS, vector<vector<float>>(PQ_CENTROIDS, vector<float>(CHUNK_DIM)));
    for (int chunk = 0; chunk < PQ_CHUNKS; chunk++) {
        for (int c = 0; c < PQ_CENTROIDS; c++) {
            infile.read((char*)codebooks[chunk][c].data(), CHUNK_DIM * sizeof(float));
        }
    }

    // 2. Load the Compressed Nodes into RAM
    vector<node> nodes(num_nodes);
    for (int i = 0; i < num_nodes; i++) {
        nodes[i].id = i;
        nodes[i].pq_code.resize(PQ_CHUNKS);
        infile.read((char*)nodes[i].pq_code.data(), PQ_CHUNKS * sizeof(uint8_t));
    }
    return nodes;
}


vector<vector<float>> build_distance_table(const vector<float>& query) {
    vector<vector<float>> dist_table(PQ_CHUNKS, vector<float>(PQ_CENTROIDS, 0.0f));
    
    for (int chunk = 0; chunk < PQ_CHUNKS; chunk++) {
        int start_dim = chunk * CHUNK_DIM;
        for (int c = 0; c < PQ_CENTROIDS; c++) {
            float dist = 0.0f;
            for (int d = 0; d < CHUNK_DIM; d++) {
                float diff = query[start_dim + d] - codebooks[chunk][c][d];
                dist += diff * diff;
            }
            dist_table[chunk][c] = dist;
        }
    }
    return dist_table;
}


float get_adc_distance(const vector<uint8_t>& pq_code, const vector<vector<float>>& dist_table) {
    float total_dist = 0.0f;
    for (int chunk = 0; chunk < PQ_CHUNKS; chunk++) {
        // Just look up the answer in the table! Zero math required.
        total_dist += dist_table[chunk][pq_code[chunk]]; 
    }
    return total_dist;
}



// --- THE DISK-NATIVE VAMANA SEARCH ---
// This uses the memory-mapped SSD graph (DiskGraph) to find neighbors instantly
// --- THE HYBRID VAMANA SEARCH (PQ + SSD) ---
vector<int> vamana_disk_search(
    const vector<float>& query_vec,    // The exact floats from Python
    const vector<node>& all_vectors,   // The 48-byte PQ codes in RAM
    DiskGraph& graph,                  // The graph edges mapped to SSD
    int ep, int L, int K, int& hops,int&disc_reads
) {
    vector<int> candidates;
    vector<bool> visited(graph.num_nodes, false);
    
    // Min-heap keeps the closest approximate nodes at the top
    priority_queue<pair<float, int>, vector<pair<float, int>>, greater<pair<float, int>>> pq;
    hops = 0;

    // 1. PRE-COMPUTE: Build the ADC Table for this specific query
    // We do the math against the 256 centroids exactly ONCE.
    vector<vector<float>> dist_table = build_distance_table(query_vec);

    // 2. THE ENTRY POINT: Use the RAM table to approximate distance
    float initial_dist = get_adc_distance(all_vectors[ep].pq_code, dist_table);
    hops++;
    
    pq.push({initial_dist, ep});
    visited[ep] = true;
    
    // 3. THE BEAM SEARCH LOOP
    while(!pq.empty() && (int)candidates.size() < L) {
        int curr_id = pq.top().second;
        pq.pop();
        candidates.push_back(curr_id);
        
        // DISK JUMP: Sniping the connections from the SSD graph (O(1))
        disc_reads++;
        pair<int*, int> p = graph.get_neighbors(curr_id);
        int* neighbors_ptr = p.first;
        int degree = p.second;
        
        for(int i = 0; i < degree; i++) {
            int neighbor_id = neighbors_ptr[i];
            if(!visited[neighbor_id]) {
                visited[neighbor_id] = true;
                
                // THE PQ MAGIC: Zero-math distance approximation!
                // We just look up 48 numbers in our dist_table and add them.
                float d = get_adc_distance(all_vectors[neighbor_id].pq_code, dist_table);
                hops++; 
                pq.push({d, neighbor_id});
            }
        }
    }
    
    // 4. FINAL SORT
    // Sort the expanded candidates by their approximate distances.
    sort(candidates.begin(), candidates.end(), [&](int a, int b) {
        return get_adc_distance(all_vectors[a].pq_code, dist_table) < 
               get_adc_distance(all_vectors[b].pq_code, dist_table);
    });
    
    if ((int)candidates.size() > K) candidates.resize(K);
    return candidates;
}