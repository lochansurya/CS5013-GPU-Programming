// graph_index_client.cu
#pragma once
#include "constants.cuh"
#include "graphT.cuh"
#include "graphIndexT.cuh"
#include <iostream>
#include <string>

//==============
// using namespace declarations
//==============
using std::string;
using std::cout, std::endl, std::cin, std::cerr;


//==============
// main function
//==============
int main(int argc, char **argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <graph_index_file>" << std::endl;
        return EXIT_FAILURE;
    }

    const char *graph_index_file = argv[1];

    // Load graph index
    FreshVamana::GraphIndexT<float> graph_index(graph_index_file);
    if (!graph_index.blob) {
        cerr << "Failed to load graph index from file: " << graph_index_file << std::endl;
        return EXIT_FAILURE;
    }

    cout << "Graph index loaded successfully from " << graph_index_file << endl;
    cout << "Number of nodes: " << graph_index.num_nodes << endl;

    // Example: Print out-degree of the first 5 nodes
    cout << "First 5 nodes out-degrees:" << endl;
    for (size_t i = 0; i < 5 && i < graph_index.num_nodes; ++i) {
        cout << "Node " << i << " out-degree: " << graph_index.entries[i].outDegree << "\n";
    }
    cout << endl;
    // Clean up
    free(graph_index.blob);
    free(graph_index.entries);

    return EXIT_SUCCESS;
    
}