#pragma once
#include "constants.cuh"
#include "graphT.cuh"
#include "kernels.cuh"
#include "search.cuh"
#include "timer.h"
#include "utils.cuh"

#include "delete_list.cuh"
#include "insert_list.cuh"

#include <memory>
#include <vector>

// DONE: design this
// checkout

/*
Binary file data layout

graph is a random graph of size 10000
graph has format

struct node {
    float vec[128];
    uint degree;
    uint neighbors[degree];
} node_t;

basepoints is all the points in the graph
float basepoinst[N][128];
*/

using namespace FreshVamana::Consts;

template <typename T__>
class Vamana {
   public:
    Vamana(std::unique_ptr<GraphT<T__>> graph_arg);

    // Implemented
    void insertPoints(T__* d_queryVecs, size_t num);

    /**
     * @brief Deletes multiple points from the FreshVamana graph given their query vectors.
     *
     * This function takes a buffer of query vectors (`d_queryVecs`) residing on the device
     * and identifies the corresponding node indices in the graph using `findPointsInGraph()`.
     * Once the matching node indices are found, they are added to the internal
     * `delete_list_` for search and deferred deletion from the graph structure.
     *
     * Timing information for the search operation is also printed to standard output.
     *
     * @tparam T__ The data type of each coordinate in the query vector
     *             (e.g., `float`, `double`, etc.).
     *
     * @param[in] d_queryVecs Pointer to the device memory containing `num` query vectors,
     *                        each of dimension `FreshVamana::Consts::D_g`.
     * @param[in] num         Number of query vectors in the buffer (`d_queryVecs`).
     *
     * @see findPointsInGraph
     * @see FreshVamana::Consts::D_g
     * @see FreshVamana::Globals::d_graph_size
     */
    void deletePoints(T__* d_queryVecs, size_t num);

    /**
     * @brief Searches for the nearest neighbors of multiple query vectors in the FreshVamana graph.
     *
     * This function takes a buffer of query vectors (`d_queryVecs`) located on the device
     * and performs a search in the FreshVamana graph to find their approximate nearest neighbors.
     * Each query vector has dimensionality `FreshVamana::Consts::D_g` and element type `T__`.
     *
     * The function returns a pointer to a GPU buffer containing the search results.
     * The buffer size is `num * FreshVamana::Consts::L_g * sizeof(uint)`, where each query
     * has up to `FreshVamana::Consts::L_g` nearest neighbor node indices.
     *
     * @tparam T__ The data type of each coordinate in the query vectors
     *             (e.g., `float`, `double`, etc.).
     *
     * @param[in] d_queryVecs Pointer to the device memory containing `num` query vectors,
     *                        each of dimension `FreshVamana::Consts::D_g`.
     * @param[in] num         Number of query vectors to process.
     *
     * @return A pointer to a device (GPU) buffer of type `uint*`, containing
     *         `num * FreshVamana::Consts::L_g` node indices.
     *         **The caller is responsible for freeing this buffer using `cudaFree()` when done.**
     *
     * @note The function is marked `[[nodiscard]]` to prevent accidental ignoring of
     *       the returned GPU buffer, which must be freed explicitly.
     *
     * @see FreshVamana::Consts::D_g
     * @see FreshVamana::Consts::L_g
     */
    /* NOT WORKING NOT WORKING NOT WORKING NOT WORKING */
    [[nodiscard]] uint* searchPoints(T__* d_queryVecs, size_t num);

    /**
     * @brief Searches for the nearest neighbors of multiple query vectors in the FreshVamana graph.
     *
     * This function takes a buffer of query vectors (`d_queryVecs`) located on the device
     * and performs a search in the FreshVamana graph to find their approximate nearest neighbors.
     * Each query vector has dimensionality `FreshVamana::Consts::D_g` and element type `T__`.
     *
     * The function returns a pointer to a GPU buffer containing the search results.
     * The buffer size is `num * FreshVamana::Consts::L_g * FreshVamana::Consts::D_g * sizeof(T__)`,
     * where each query has up to `FreshVamana::Consts::L_g` nearest neighbor vectors.
     *
     * @tparam T__ The data type of each coordinate in the query vectors
     *             (e.g., `float`, `double`, etc.).
     *
     * @param[in] d_queryVecs Pointer to the device memory containing `num` query vectors,
     *                        each of dimension `FreshVamana::Consts::D_g`.
     * @param[in] num         Number of query vectors to process.
     *
     * @return A pointer to a device (GPU) buffer of type `T__*`, containing
     *         `num * FreshVamana::Consts::L_g` vectors of size `FreshVamana::Consts::D_g *
     *         sizeof(T__)`.
     *         **The caller is responsible for freeing this buffer using `cudaFree()` when done.**
     *
     * @note The function is marked `[[nodiscard]]` to prevent accidental ignoring of
     *       the returned GPU buffer, which must be freed explicitly.
     *
     * @see FreshVamana::Consts::D_g
     * @see FreshVamana::Consts::L_g
     */
    [[nodiscard]] T__* searchPoints2(T__* d_queryVecs, size_t num);

    // TODO: implement this.
    void patchGraph();

   private:
    [[nodiscard]] int* findPointsInGraph(const uint8_t* d_graph,
                                         uint           n_nodes,
                                         const dtype_g* d_query_vecs,
                                         uint           n_queries);

    void runVamana();

   public:
    std::unique_ptr<GraphT<T__>> graph_;

   private:
    InsertList insert_list_;
    DeleteList delete_list_;
};
// DONT REMOVE THIS
#include "vamana_impl.cuh"