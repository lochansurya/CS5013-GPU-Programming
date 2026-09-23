// graph.cuh
#pragma once
#include "constants.cuh"
#include "globals.cuh"

#include <cstdint>
#include <cstdio>

using uint = unsigned int;

// template <typename T__>
// // T in the following variable names indicates the template type parameter T
// struct GraphT {
//     // This is the actual entry raw binary layout in the graph index(random_graph.bin)
//     using dtype_g = FreshVamana::Consts::dtype_g;
//     using FreshVamana::Consts::R_g;

//     dtype_g *vec_elements = nullptr; // (default) Vector of 128 entries each of data_type =float
//     uint outDegree = R_g; // (default) can be overridden using the custom constructor
//     uint *adjList = nullptr; // out_neighbors indices in the graph index

//     // Default constructor; Keep R_g as fallback out-degree
//     GraphT() = default;

//     // Custom constructor to set a different out-degree than R_g
//     GraphT(uint outDegree_arg) : outDegree(outDegree_arg) {}

//     getNode(uint idx) {
//         return adjList[idx];
//     }

//     // Factory to create a view from raw binary memory
//     __host__ __device__
//     static GraphT from_blob(uint8_t* base_ptr) {
//         GraphT g;
//         g.vec_elements = reinterpret_cast<dtype_g*>(base_ptr);

//         uint offset_vec = FreshVamana::Consts::D_g * sizeof(dtype_g);
//         g.outDegree = *reinterpret_cast<uint*>(base_ptr + offset_vec);

//         uint offset_adj = offset_vec + sizeof(uint);
//         g.adjList = reinterpret_cast<uint*>(base_ptr + offset_adj);
//         return g;
//     }

// };

// template <typename T__>
// struct QueryT{
//     FreshVamana::Consts::queryType type = FreshVamana::Consts::queryType::search_q; // have to
//     overwrite(INITIALIZE) this at the time of instantiation of the query struct
//     FreshVamana::Consts::dtype_g *vec_elements = nullptr; // (default) Vector of 128 entries each
//     of data_type= float
// };

template <typename T__>
struct GraphT {
    uint8_t* d_graph = nullptr;
};

template <typename T__>
void expandGraph(GraphT<T__>& graph, uint min_increase) {
    if (FreshVamana::Globals::d_graph_capacity_g == 0) {
        return;
    }

    const uint grow_by      = static_cast<uint>(min_increase * 3) + 1;
    const uint new_capacity = FreshVamana::Globals::d_graph_capacity_g + grow_by;

    const size_t old_bytes = static_cast<size_t>(FreshVamana::Globals::d_graph_capacity_g) *
                             FreshVamana::Consts::graph_entry_bytes_g;
    const size_t new_bytes =
        static_cast<size_t>(new_capacity) * FreshVamana::Consts::graph_entry_bytes_g;

    uint8_t*    d_new_graph = nullptr;
    cudaError_t err         = cudaMalloc(&d_new_graph, new_bytes);
    if (err != cudaSuccess) {
        return;
    }

    if (graph.d_graph != nullptr && FreshVamana::Globals::d_graph_size_g > 0) {
        cudaMemcpy(d_new_graph, graph.d_graph, old_bytes, cudaMemcpyDeviceToDevice);
        cudaFree(graph.d_graph);
    }

    graph.d_graph                            = d_new_graph;
    FreshVamana::Globals::d_graph_capacity_g = new_capacity;

    printf("\n\n[ expandGraph done ]\n\n");
}
