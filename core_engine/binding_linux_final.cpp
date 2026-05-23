#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // The magic header: Auto-converts Python lists to C++ std::vector
#include <vector>
#include <string>
#include<tuple>
#include <iostream>

// Include your engine block! 
// (Make sure this filename matches exactly what you named your Linux verifier file)
#include "verifier_linux.cpp" 

namespace py = pybind11;
using namespace std;

// --- THE PYTHON-FACING ENGINE CLASS ---
class GraphiteEngine {
private:
    DiskGraph graph;
    vector<node> all_vectors;
    int dim;

public:
    // Constructor: Python calls this ONCE when you initialize the database
    GraphiteEngine(const string& data_file, const string& graph_file) {
        cout << "Starting Graphite Engine..." << endl;
        
        // 1. Load the math into RAM
        all_vectors = load_pq_data(data_file);
        
        // 2. Map the graph to the SSD
        graph.map_file(graph_file.c_str());
        
        cout << "Engine Ready! Loaded " << all_vectors.size() << " vectors into RAM." << endl;
        cout << "Graph mapped to SSD successfully." << endl;
    }

    // Destructor: Safely unmap the file from RAM when Python closes
    ~GraphiteEngine() {
        graph.unmap();
    }

    // The Search Function Python will actually call!
    tuple<vector<int>,int,int> search(const vector<float>& query_vec, int K=5, int L=20) {
        int hops = 0;
        int disc_reads= 0;
        int ep = graph.medoid_id; // Drop the scout at the end of the database
        
        // Call your hardcore C++ AVX2 search logic
        return make_tuple(vamana_disk_search(query_vec, all_vectors, graph, ep, L, K, hops,disc_reads),hops,disc_reads);
    }
};

// --- PYBIND11 MODULE DEFINITION ---
PYBIND11_MODULE(graphite_core, m) {
    m.doc() = "Graphite Vamana Vector Database Core";

    // Bind the C++ class to a Python class
    py::class_<GraphiteEngine>(m, "GraphiteEngine")
        .def(py::init<const string&, const string&>(), py::arg("data_file"), py::arg("graph_file"))
        .def("search", &GraphiteEngine::search, "Run Vamana search on a query vector",
             py::arg("query_vec"), py::arg("K") = 5, py::arg("L") = 20);
}