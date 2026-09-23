#pragma once
#include "constants.cuh"
#include "graphT.cuh"
#include "kernels.cuh"
#include "search.cuh"
#include "timer.h"
#include "utils.cuh"
#include "vamana.cuh"

template <typename T__>
Vamana<T__>::Vamana(std::unique_ptr<GraphT<T__>> graph_arg) {
    graph_ = std::move(graph_arg);

    CPUTimer cputimer;

    cputimer.Start();
    runVamana();
    cputimer.Stop();
    printf("Vamana<T__>::Vamana: %f sec\n", cputimer.Elapsed());
};

template <typename T__>
void Vamana<T__>::insertPoints(T__* d_query_vecs, size_t num_query_vecs) {
    using namespace FreshVamana;
    std::cout << "[ insertPoints ]\n";

    CPUTimer cputimer;
    cputimer.Start();

    insert_list_.addVectors(d_query_vecs, num_query_vecs);

    cputimer.Stop();
}

template <typename T__>
void Vamana<T__>::deletePoints(T__* d_query_vecs, size_t num_d_query_vecs) {
    CPUTimer cputimer;

    // std::cout << "[ deletePoints ]\n";

    cputimer.Start();

    int* d_results = findPointsInGraph(
        graph_->d_graph, FreshVamana::Globals::d_graph_size_g, d_query_vecs, num_d_query_vecs);

    cputimer.Stop();
    // printf("findPointsInGraph(%lu points): %f sec\n", num, cputimer.Elapsed());

    // IMP: doesnt work on the vectors which are not in the graph
    delete_list_.addNodes(reinterpret_cast<uint*>(d_results), num_d_query_vecs);

    cudaFree(d_results);
}

template <typename T__>
[[nodiscard]] uint* Vamana<T__>::searchPoints(T__* d_queryVecs, size_t num) {
    uint*    d_visitedSets;
    uint*    d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex;
    std::cout << "[ searchPoints ]\n";

    CPUTimer cputimer;
    CPUTimer cputimermain;
    cputimermain.Start();

    cputimer.Start();
    gpuErrchk(cudaMalloc(&d_visitedSets,
                         FreshVamana::Globals::d_graph_size_g *
                             FreshVamana::Consts::max_num_parents_per_query * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, FreshVamana::Globals::d_graph_size_g * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeIndex,
                         FreshVamana::Globals::d_graph_size_g *
                             FreshVamana::Consts::reverse_index_entry_bytes_g * sizeof(uint8_t)));
    gpuErrchk(
        cudaMemset(d_visitedSetCount, 0, FreshVamana::Globals::d_graph_size_g * sizeof(uint)));
    cputimer.Stop();
    printf("vamanaInner mallocs: %f sec\n", cputimer.Elapsed());

    cputimer.Start();
    uint* d_worklist = greedySearch<T__>(graph_->d_graph,
                                         delete_list_.data(),
                                         delete_list_.size(),
                                         d_queryVecs,
                                         d_visitedSets,
                                         d_visitedSetCount,
                                         num);
    cputimer.Stop();
    printf("greedySearch: %f sec\n", cputimer.Elapsed());

    cputimermain.Stop();
    printf("Vamana<T__>::search: %f sec\n", cputimermain.Elapsed());
    gpuErrchk(cudaFree(d_visitedSets));
    gpuErrchk(cudaFree(d_visitedSetCount));
    gpuErrchk(cudaFree(d_reverseEdgeIndex));
    return d_worklist;
}

template <typename T__>
[[nodiscard]] T__* Vamana<T__>::searchPoints2(T__* d_queryVecs, size_t num) {
    uint*    d_visitedSets;
    uint*    d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex;
    std::cout << "[ searchPoints2 ]\n";

    CPUTimer cputimer;
    CPUTimer cputimermain;
    cputimermain.Start();

    cputimer.Start();
    gpuErrchk(cudaMalloc(&d_visitedSets,
                         FreshVamana::Globals::d_graph_size_g *
                             FreshVamana::Consts::max_num_parents_per_query * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, FreshVamana::Globals::d_graph_size_g * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeIndex,
                         FreshVamana::Globals::d_graph_size_g *
                             FreshVamana::Consts::reverse_index_entry_bytes_g * sizeof(uint8_t)));
    gpuErrchk(
        cudaMemset(d_visitedSetCount, 0, FreshVamana::Globals::d_graph_size_g * sizeof(uint)));
    cputimer.Stop();
    printf("vamanaInner mallocs: %f sec\n", cputimer.Elapsed());

    cputimer.Start();
    uint* d_worklist = greedySearch<T__>(graph_->d_graph,
                                         delete_list_.data(),
                                         delete_list_.size(),
                                         d_queryVecs,
                                         d_visitedSets,
                                         d_visitedSetCount,
                                         num);
    cputimer.Stop();
    printf("greedySearch: %f sec\n", cputimer.Elapsed());

    cputimer.Start();
    using namespace FreshVamana;
    const size_t L_g       = Consts::L_g;
    const size_t D_g       = Consts::D_g;
    const size_t entrySize = Consts::graph_entry_bytes_g;

    T__* d_worklist_vectors;
    gpuErrchk(cudaMalloc(&d_worklist_vectors, num * L_g * D_g * sizeof(T__)));

    dim3 extract_grid(num * L_g);
    dim3 extract_block(D_g);
    extract_vectors_kernel<<<extract_grid, extract_block>>>(
        graph_->d_graph, d_worklist, d_worklist_vectors, L_g, D_g, entrySize, num);
    gpuErrchk(cudaPeekAtLastError());

    T__* d_final_top_vectors;
    gpuErrchk(cudaMalloc(&d_final_top_vectors, num * L_g * D_g * sizeof(T__)));

    // static_assert(L_g * D_g * sizeof(T__) < 48000, "Shared memory may be insufficient.");

    std::cout << insert_list_.size() << '\n';

    dim3 merge_grid(num);
    dim3 merge_block(256);
    merge_and_rerank_kernel<<<merge_grid, merge_block>>>(d_queryVecs,
                                                         d_worklist_vectors,
                                                         FreshVamana::Globals::d_insert_list_g,
                                                         d_final_top_vectors,
                                                         L_g,
                                                         D_g,
                                                         insert_list_.size(),
                                                         num);
    gpuErrchk(cudaPeekAtLastError());

    cputimer.Stop();
    printf("Vector extraction & merge: %f sec\n", cputimer.Elapsed());
    cputimermain.Stop();

    gpuErrchk(cudaFree(d_worklist));
    gpuErrchk(cudaFree(d_worklist_vectors));

    printf("Vamana<T__>::search (total): %f sec\n", cputimermain.Elapsed());
    gpuErrchk(cudaFree(d_visitedSets));
    gpuErrchk(cudaFree(d_visitedSetCount));
    gpuErrchk(cudaFree(d_reverseEdgeIndex));

    return d_final_top_vectors;
}

template <typename T__>
void Vamana<T__>::patchGraph() {
    using namespace FreshVamana;

    size_t num_new_nodes = insert_list_.size();

    if (num_new_nodes == 0) {
        return;
    }

    std::cout << "[ patchGraph ]\n";
    CPUTimer cputimer_patch;
    cputimer_patch.Start();

    size_t old_graph_size = Globals::d_graph_size_g;
    size_t new_graph_size = old_graph_size + num_new_nodes;

    uint8_t* d_graph_new;
    size_t   new_graph_bytes = new_graph_size * Consts::graph_entry_bytes_g;
    gpuErrchk(cudaMalloc(&d_graph_new, new_graph_bytes));

    if (old_graph_size > 0) {
        size_t old_graph_bytes = old_graph_size * Consts::graph_entry_bytes_g;
        gpuErrchk(
            cudaMemcpy(d_graph_new, graph_->d_graph, old_graph_bytes, cudaMemcpyDeviceToDevice));
    }

    T__* d_new_vecs = insert_list_.data();
    dim3 grid(num_new_nodes);
    dim3 block(256);
    copyNewVectorsToGraph<T__>
        <<<grid, block>>>(d_graph_new, d_new_vecs, old_graph_size, num_new_nodes);

    cudaFree(graph_->d_graph);
    graph_->d_graph         = d_graph_new;
    Globals::d_graph_size_g = new_graph_size;

    uint*    d_visitedSets;
    uint*    d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex;
    float    alpha = 1.5;

    gpuErrchk(cudaMalloc(&d_visitedSets,
                         num_new_nodes * Consts::max_num_parents_per_query * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, num_new_nodes * sizeof(uint)));
    gpuErrchk(cudaMemset(d_visitedSetCount, 0, num_new_nodes * sizeof(uint)));

    size_t reverse_index_bytes =
        new_graph_size * Consts::reverse_index_entry_bytes_g * sizeof(uint8_t);
    gpuErrchk(cudaMalloc(&d_reverseEdgeIndex, reverse_index_bytes));
    gpuErrchk(cudaMemset(d_reverseEdgeIndex, 0, reverse_index_bytes));

    printf("  [patchGraph] Running greedy search for %zu new nodes...\n", num_new_nodes);
    uint* d_worklist = greedySearch<T__>(graph_->d_graph,
                                         delete_list_.data(),
                                         delete_list_.size(),
                                         d_new_vecs,
                                         d_visitedSets,
                                         d_visitedSetCount,
                                         num_new_nodes);
    gpuErrchk(cudaFree(d_worklist));

    printf("  [patchGraph] Computing out-neighbors for new nodes...\n");
    computeOutNeighbors<T__>(graph_->d_graph,
                             d_new_vecs,
                             d_visitedSets,
                             d_visitedSetCount,
                             alpha,
                             d_reverseEdgeIndex,
                             old_graph_size,
                             num_new_nodes);

    printf("  [patchGraph] Patching reverse edges for existing nodes...\n");
    computeReverseEdges<T__>(graph_->d_graph, d_reverseEdgeIndex, alpha);

    gpuErrchk(cudaFree(d_visitedSets));
    gpuErrchk(cudaFree(d_visitedSetCount));
    gpuErrchk(cudaFree(d_reverseEdgeIndex));

    insert_list_.clear();

    cputimer_patch.Stop();
    printf("Vamana<T__>::patchGraph: %f sec\n", cputimer_patch.Elapsed());
}

// PRIVATES
template <typename T__>
[[nodiscard]]
int* Vamana<T__>::findPointsInGraph(const uint8_t* d_graph,
                                    uint           n_nodes,
                                    const dtype_g* d_query_vecs,
                                    uint           n_queries) {
    int* d_results = nullptr;
    cudaMalloc(&d_results, n_queries * sizeof(int));

    // Initialize results to -1
    cudaMemset(d_results, 0xFF, n_queries * sizeof(int));

    dim3 threads(256);
    dim3 blocks((n_nodes + threads.x - 1) / threads.x, n_queries);

    findPointsKernel<<<blocks, threads>>>(d_graph, d_query_vecs, n_nodes, n_queries, d_results);
    cudaDeviceSynchronize();

    return d_results;
}

template <typename T__>
void Vamana<T__>::runVamana() {
    uint*    d_visitedSets;
    uint*    d_visitedSetCount;
    uint8_t* d_reverseEdgeIndex;

    std::cout << "[ runVamana ]\n";

    float alpha = 1.5;
    T__*  d_queryVecs;
    gpuErrchk(
        cudaMalloc(&d_queryVecs,
                   FreshVamana::Globals::d_graph_size_g * FreshVamana::Consts::D_g * sizeof(T__)));

    for (uint i = 0; i < FreshVamana::Globals::d_graph_size_g; i++) {
        T__* src = (T__*)(graph_->d_graph + (i)*FreshVamana::Consts::graph_entry_bytes_g);
        T__* dst = (T__*)(d_queryVecs + i * FreshVamana::Consts::D_g);
        cudaMemcpy(dst, src, FreshVamana::Consts::D_g * sizeof(T__), cudaMemcpyDeviceToDevice);
    }

    CPUTimer cputimer;

    cputimer.Start();
    gpuErrchk(cudaMalloc(&d_visitedSets,
                         FreshVamana::Globals::d_graph_size_g *
                             FreshVamana::Consts::max_num_parents_per_query * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_visitedSetCount, FreshVamana::Globals::d_graph_size_g * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeIndex,
                         FreshVamana::Globals::d_graph_size_g *
                             FreshVamana::Consts::reverse_index_entry_bytes_g * sizeof(uint8_t)));
    gpuErrchk(
        cudaMemset(d_visitedSetCount, 0, FreshVamana::Globals::d_graph_size_g * sizeof(uint)));
    cputimer.Stop();
    printf("vamanaInner mallocs: %f sec\n", cputimer.Elapsed());

    cputimer.Start();
    uint* d_worklist = greedySearch<T__>(graph_->d_graph,
                                         delete_list_.data(),
                                         delete_list_.size(),
                                         d_queryVecs,
                                         d_visitedSets,
                                         d_visitedSetCount,
                                         FreshVamana::Globals::d_graph_size_g);
    gpuErrchk(cudaFree(d_worklist));

    cputimer.Stop();
    printf("greedySearch: %f sec\n", cputimer.Elapsed());

    cputimer.Start();

    computeOutNeighbors<T__>(graph_->d_graph,
                             d_queryVecs,
                             d_visitedSets,
                             d_visitedSetCount,
                             alpha,
                             d_reverseEdgeIndex,
                             0,
                             FreshVamana::Globals::d_graph_size_g);
    cputimer.Stop();
    printf("computeOutNeighbors: %f sec\n", cputimer.Elapsed());

    cputimer.Start();
    computeReverseEdges<T__>(graph_->d_graph, d_reverseEdgeIndex, alpha);
    cputimer.Stop();
    printf("computeReverseEdges: %f sec\n", cputimer.Elapsed());

    cputimer.Start();
    gpuErrchk(cudaFree(d_visitedSets));
    gpuErrchk(cudaFree(d_visitedSetCount));
    gpuErrchk(cudaFree(d_reverseEdgeIndex));
    cputimer.Stop();
}