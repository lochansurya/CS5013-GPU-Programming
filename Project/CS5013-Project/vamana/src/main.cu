#include "graphT.cuh"
#include "utils.cuh"
#include "vamana.cuh"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>

template <typename T__>
void printWorklistVectors(Vamana<T__>& index,
                          uint*        d_worklist,
                          size_t       num_query,
                          size_t       print_count = 5) {
    using namespace FreshVamana;

    const size_t L_g       = Consts::L_g;
    const size_t vecDim    = Consts::D_g;
    const size_t entrySize = Consts::graph_entry_bytes_g;

    // Copy worklist to host
    std::vector<uint> h_worklist(num_query * L_g);
    cudaMemcpy(
        h_worklist.data(), d_worklist, h_worklist.size() * sizeof(uint), cudaMemcpyDeviceToHost);

    printf("=== Worklist Debug (showing up to %zu entries) ===\n", print_count);
    for (size_t i = 0; i < std::min(print_count, h_worklist.size()); ++i) {
        uint node_id = h_worklist[i];
        printf("Worklist[%zu] = %u\n", i, node_id);

        // Load vector from GPU graph
        std::vector<T__> h_vec(vecDim);
        size_t           offset = entrySize * node_id;

        cudaMemcpy(h_vec.data(),
                   index.graph_->d_graph + offset,
                   vecDim * sizeof(T__),
                   cudaMemcpyDeviceToHost);

        // Print first few dimensions for debugging
        printf("  Vec[%u]:", node_id);
        for (size_t j = 0; j < std::min<size_t>(vecDim, 8); ++j) {
            printf(" %.f", static_cast<float>(h_vec[j]));
        }
        printf(" ...\n");
    }
    printf("===============================================\n");
}

/**
 * @brief Prints the final vectors returned by searchPoints.
 * @tparam T__ Vector data type (e.g., float)
 * @param d_final_top_vectors Device pointer to the final vectors (output of searchPoints)
 * @param num_query Total number of queries that were processed
 * @param print_queries How many queries to print results for (default 1)
 * @param print_vecs_per_query How many top vectors to print for each query (default 5)
 */
template <typename T__>
void printFinalVectors(T__*   d_final_top_vectors,
                       size_t num_query,
                       size_t print_queries        = 1,
                       size_t print_vecs_per_query = 5) {
    using namespace FreshVamana;

    const size_t L_g    = Consts::L_g;
    const size_t vecDim = Consts::D_g;

    const size_t queries_to_copy = std::min(print_queries, num_query);
    if (queries_to_copy == 0)
        return;

    const size_t vectors_to_copy = queries_to_copy * L_g;
    if (vectors_to_copy == 0)
        return;

    std::vector<T__> h_final_vectors(vectors_to_copy * vecDim);
    gpuErrchk(cudaMemcpy(h_final_vectors.data(),
                         d_final_top_vectors,
                         h_final_vectors.size() * sizeof(T__),
                         cudaMemcpyDeviceToHost));

    printf("=== Final Vector Results (showing %zu queries) ===\n", queries_to_copy);

    for (size_t q = 0; q < queries_to_copy; ++q) {
        printf("--- Query %zu ---\n", q);
        const size_t vecs_to_print = std::min(print_vecs_per_query, L_g);

        for (size_t i = 0; i < vecs_to_print; ++i) {
            // Get the pointer to the start of the i-th vector for query q
            size_t vector_idx = q * L_g + i;
            T__*   h_vec      = h_final_vectors.data() + vector_idx * vecDim;

            printf("  ResultVec[%zu]:", i);
            // Print first few dimensions
            for (size_t j = 0; j < std::min<size_t>(vecDim, 8); ++j) {
                printf(" %.f", static_cast<float>(h_vec[j]));
            }
            printf(" ...\n");
        }
    }
    printf("==================================================\n");
}

template <typename T__>
void printNodeNeighbors(uint8_t* d_graph, size_t node_num) {
    using namespace FreshVamana::Consts;

    const uint   node_id   = node_num;
    const size_t entrySize = graph_entry_bytes_g;
    const size_t vecDim    = D_g;

    std::vector<uint8_t> h_entry(entrySize);

    cudaMemcpy(h_entry.data(), d_graph + node_id * entrySize, entrySize, cudaMemcpyDeviceToHost);

    printf("%f\n", *((float*)h_entry.data()));

    // T__*    vec    = reinterpret_cast<T__*>(h_entry.data());
    uint  degree    = *(h_entry.data() + vecDim * sizeof(T__));
    uint* neighbors = reinterpret_cast<uint*>(h_entry.data() + vecDim * sizeof(T__) + sizeof(uint));

    printf("Node %u has %u neighbors:\n", node_id, degree);
    for (uint i = 0; i < degree; ++i) {
        printf("  Neighbor[%u] = %u\n", i, neighbors[i]);
    }
}

__global__ void pp() {
    printf("\n\n%d\n\n", FreshVamana::Globals::d_graph_size_g);
}

__global__ void pqv(float* dd) {
    printf("\n\n");

    for (int i = 0; i < 128; i++) {
        printf("%f ", dd[i]);
    }
    printf("\n\n");
}

__global__ void pqpinc(float* dd) {
    dd[0] += 1.0f;
}

__global__ void pqpdec(float* dd) {
    dd[0] -= 1.0f;
}

int main(int argc, char** argv) {
    if (argc != 4) {
        printf("Usage: %s <graph> <basepoints> <output>\n", argv[0]);
        return 1;
    }

    std::string random_graph_bin_path = argv[1];

    std::unique_ptr<GraphT<dtype_g>> graph = initGraph<dtype_g>(random_graph_bin_path);
    // std::cout << graph->h_graph_capacity << " " << graph->h_graph_size << '\n';
    std::cout << FreshVamana::Globals::d_graph_capacity_g << " "
              << FreshVamana::Globals::d_graph_size_g << '\n';

    Vamana<dtype_g> index(std::move(graph));

    // ADDING TEST QUERIES
    dtype_g* d_queryVecs;
    uint     num_vecs = 10;
    gpuErrchk(cudaMalloc(&d_queryVecs, num_vecs * FreshVamana::Consts::D_g * sizeof(dtype_g)));

    for (uint i = 0; i < num_vecs; i++) {
        dtype_g* src =
            (dtype_g*)(index.graph_->d_graph + (i)*FreshVamana::Consts::graph_entry_bytes_g);
        dtype_g* dst = (dtype_g*)(d_queryVecs + i * FreshVamana::Consts::D_g);
        cudaMemcpy(dst, src, FreshVamana::Consts::D_g * sizeof(dtype_g), cudaMemcpyDeviceToDevice);
    }

    // SEARCH TEST
    std::cout << '\n';
    // uint* d_worklist = index.searchPoints(d_queryVecs, 1);
    // printWorklistVectors(index, d_worklist, 1, 5);
    // cudaFree(d_worklist);

    float* d_worklist_nn55 = index.searchPoints2(d_queryVecs, 1);
    printFinalVectors(d_worklist_nn55, 1, 1, 5);
    cudaFree(d_worklist_nn55);

    cudaDeviceSynchronize();

    // pqv<<<1, 1>>>(d_queryVecs);
    pqpinc<<<1, 1>>>(d_queryVecs);
    // pqv<<<1, 1>>>(d_queryVecs);
    std::cout << '\n';

    // printNodeNeighbors<dtype_g>(index.graph_->d_graph, 0);
    // cudaDeviceSynchronize();

    index.insertPoints(d_queryVecs, 1);

    cudaDeviceSynchronize();

    pqpdec<<<1, 1>>>(d_queryVecs);
    // pqv<<<1, 1>>>(d_queryVecs);
    // pqv<<<1, 1>>>((float*)(index.graph_->d_graph + (10000) * graph_entry_bytes_g));
    // printNodeNeighbors<dtype_g>(index.graph_->d_graph,
    // FreshVamana::Globals::d_graph_size_g- 1);
    // printNodeNeighbors<dtype_g>(index.graph_->d_graph, 0);
    // cudaDeviceSynchronize();

    std::cout << '\n';
    // uint* d_worklist4 = index.searchPoints(d_queryVecs, 1);
    // printWorklistVectors(index, d_worklist4, 1, 5);
    // cudaFree(d_worklist4);

    float* d_worklist_nn = index.searchPoints2(d_queryVecs, 1);
    printFinalVectors(d_worklist_nn, 1, 1, 5);
    cudaFree(d_worklist_nn);

    index.patchGraph();
    float* d_worklist_nn33 = index.searchPoints2(d_queryVecs, 1);
    printFinalVectors(d_worklist_nn33, 1, 1, 5);
    cudaFree(d_worklist_nn33);

    // DELETE TEST
// #define delete_test
#if defined(delete_test)

    std::cout << '\n';

    index.deletePoints(
        (dtype_g*)(index.graph_->d_graph + (2) * FreshVamana::Consts::graph_entry_bytes_g), 1);

    // std::cout << isNodeInDeleteList(index.delete_list_.data(), index.delete_list_.size(), 2)
    //           << "\n";

    std::cout << '\n';

    uint* d_worklist2 = index.searchPoints(d_queryVecs, 1);
    printWorklistVectors(index, d_worklist2, 1, 5);
    cudaFree(d_worklist2);

    index.deletePoints(
        (dtype_g*)(index.graph_->d_graph + (6) * FreshVamana::Consts::graph_entry_bytes_g), 1);

    for (int i = 7; i < 4400; i++) {
        index.deletePoints(
            (dtype_g*)(index.graph_->d_graph + (i)*FreshVamana::Consts::graph_entry_bytes_g), 1);
    }

    index.deletePoints(
        (dtype_g*)(index.graph_->d_graph + (4585) * FreshVamana::Consts::graph_entry_bytes_g), 1);
    index.deletePoints(
        (dtype_g*)(index.graph_->d_graph + (9256) * FreshVamana::Consts::graph_entry_bytes_g), 1);

    std::cout << '\n';

    uint* d_worklist3 = index.searchPoints(d_queryVecs, 1);
    printWorklistVectors(index, d_worklist3, 1, 5);
    cudaFree(d_worklist3);

    float* d_worklist_nn22 = index.searchPoints2(d_queryVecs, 1);
    printFinalVectors(d_worklist_nn22, 1, 1, 5);
    cudaFree(d_worklist_nn22);

#endif
    // pp<<<1, 1>>>();

    return 0;
}
