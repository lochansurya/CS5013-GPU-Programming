#pragma once

#include "bloom_filter.cuh"
#include "constants.cuh"
#include "graphT.cuh"
#include "utils.cuh"

#include "delete_list.cuh"
#include "kernels.cuh"

// __device__ bool contains(uint* set, uint count, uint el) {
//     for (uint i = 0; i < count; i++) {
//         if (set[i] == el) {
//             return true;
//         }
//     }
//     return false;
// }

__global__ void initializeParents(bool* d_hasParent, uint* d_parents) {
    uint query_id = blockIdx.x;
    uint tid      = threadIdx.x;

    if (tid == 0) {
        d_hasParent[query_id] = true;
        d_parents[query_id]   = FreshVamana::Consts::medoid_g;
    }
}

template <typename T__>
__global__ void initializeWorklist(uint8_t* d_graph,
                                   T__*     d_queryVecs,
                                   uint*    d_worklist,
                                   uint*    d_worklistCount,
                                   T__*     d_worklistDist,
                                   bool*    d_worklistVisited) {
    uint queryID = blockIdx.x;
    uint tid     = threadIdx.x;

    uint worklistOffset = FreshVamana::Consts::L_g * queryID;

    T__* queryVec = d_queryVecs + FreshVamana::Consts::D_g * queryID;
    T__* medoidVec =
        (T__*)(d_graph + FreshVamana::Consts::graph_entry_bytes_g * FreshVamana::Consts::medoid_g);

    if (tid == 0) {
        d_worklist[worklistOffset]        = FreshVamana::Consts::medoid_g;
        d_worklistCount[queryID]          = 1;
        d_worklistVisited[worklistOffset] = true;

        T__ dist = 0;
        for (uint i = 0; i < FreshVamana::Consts::D_g; i++) {
            T__ diff = queryVec[i] - medoidVec[i];
            dist += diff * diff;
        }
        d_worklistDist[worklistOffset] = dist;
    }
}

/* Adds unvisited neighbours of nodes in d_parents to d_neighbors
 *
 * d_graph           - The graph
 * d_hasParent       - Whether there is a node to be visited (a parent) for a query
 * d_parents         - If the query has a parent, then the index of the parent
 * d_bloomFilters    - The bloom filters for checking if a node has been visited
 * d_neighbors       - Array for putting unvisited neighbors into
 * d_neighborsCount  - No. of unvisited neighbors for each query
 * d_visitedSet      - Array of visited points for each query
 * d_visitedSetCount - No. of visited points for each query
 */
template <typename T__>
__global__ void filterNeighbors(uint8_t*    d_graph,
                                const uint* d_delete_list,
                                size_t      d_delete_list_size,
                                bool*       d_hasParent,
                                uint*       d_parents,
                                bool*       d_bloomFilters,
                                uint*       d_neighbors,
                                uint*       d_neighborsCount,
                                uint*       d_visitedSet,
                                uint*       d_visitedSetCount) {
    uint queryID = blockIdx.x;
    uint tid     = threadIdx.x;

    if (!d_hasParent[queryID])
        return;

    bool* bloomFilter =
        d_bloomFilters + (queryID * BF_MEMORY);  // Get the bloom filter for this query
    uint parent = d_parents[queryID];            // Get the parent for this query

    // Get the pointers to the degree and neighbors of the parent
    uint* degreePtr   = (uint*)(d_graph + parent * FreshVamana::Consts::graph_entry_bytes_g +
                              FreshVamana::Consts::D_g * sizeof(T__));
    uint* neighborPtr = degreePtr + 1;

    __syncthreads();

    // Initialize neighbor count to zero, reset hasParent
    if (tid == 0) {
        d_neighborsCount[queryID] = 0;
        d_hasParent[queryID]      = false;

        uint visitedSetIdx = atomicAdd(&d_visitedSetCount[queryID], 1);
        if (visitedSetIdx < FreshVamana::Consts::max_num_parents_per_query) {
            d_visitedSet[FreshVamana::Consts::max_num_parents_per_query * queryID + visitedSetIdx] =
                parent;
        } else {
            printf("Limit hit for visited set: %d\n", queryID);
            atomicSub(&d_visitedSetCount[queryID], 1);
        }
    }

    __syncthreads();

    // Loop over each neighbor
    uint degree = *degreePtr;

    for (uint ii = tid; ii < degree; ii += blockDim.x) {
        uint neighbor = neighborPtr[ii];

        if (isNodeInDeleteList(d_delete_list, d_delete_list_size, neighbor)) {
            continue;
        }

        if (neighbor == queryID)
            continue;

        // Ensure the neighbor has not been visited yet
        if (!bf_check(bloomFilter, neighbor)) {
            bf_set(bloomFilter, neighbor);

            // Add the neighbor to d_neighbors
            uint neighborIdx = atomicAdd(&d_neighborsCount[queryID], 1);
            d_neighbors[(FreshVamana::Consts::R_g + 1) * queryID + neighborIdx] = neighbor;
        }
    }
}

template <typename T__>
__global__ void mergeIntoWorklist(uint* d_worklistCount,
                                  uint* d_worklist,
                                  T__*  d_worklistDist,
                                  bool* d_worklistVisited,
                                  uint* d_neighborsCount,
                                  uint* d_neighbors,
                                  T__*  d_neighborsDist,
                                  bool* d_hasParent,
                                  uint* d_parents,
                                  bool* d_nextIter) {
    uint queryID = blockIdx.x;
    uint tid     = threadIdx.x;

    uint neighborsOffset = queryID * (FreshVamana::Consts::R_g + 1);
    uint worklistOffset  = queryID * FreshVamana::Consts::L_g;

    uint numNeighbors    = d_neighborsCount[queryID];
    uint worklistSize    = d_worklistCount[queryID];
    uint newWorklistSize = min(numNeighbors + worklistSize, FreshVamana::Consts::L_g);

    __shared__ uint sortedPositions[FreshVamana::Consts::R_g + FreshVamana::Consts::L_g + 1];

    uint id;
    T__  dist;
    bool visited;
    uint newPos = FreshVamana::Consts::L_g;

    if (tid < worklistSize) {
        // Fist L threads find new position for worklist elements
        uint before          = lowerBound<T__>(&d_neighborsDist[neighborsOffset],
                                      0,
                                      numNeighbors,
                                      d_worklistDist[worklistOffset + tid]);
        id                   = d_worklist[worklistOffset + tid];
        dist                 = d_worklistDist[worklistOffset + tid];
        visited              = d_worklistVisited[worklistOffset + tid];
        newPos               = before + tid;
        sortedPositions[tid] = newPos;
    } else if (tid >= FreshVamana::Consts::L_g && tid < FreshVamana::Consts::L_g + numNeighbors) {
        // Next R + 1 threads find new position for neighbors
        uint idx             = tid - FreshVamana::Consts::L_g;  // Index into the neighbors array
        uint before          = upperBound<T__>(&d_worklistDist[worklistOffset],
                                      0,
                                      worklistSize,
                                      d_neighborsDist[neighborsOffset + idx]);
        id                   = d_neighbors[neighborsOffset + idx];
        dist                 = d_neighborsDist[neighborsOffset + idx];
        visited              = false;
        newPos               = before + idx;
        sortedPositions[tid] = newPos;
    }

    __syncthreads();
    __threadfence_block();

    if (newPos < newWorklistSize) {
        d_worklist[worklistOffset + newPos]        = id;
        d_worklistDist[worklistOffset + newPos]    = dist;
        d_worklistVisited[worklistOffset + newPos] = visited;
    }

    __syncthreads();
    __threadfence_block();

    if (tid == 0) {
        d_worklistCount[queryID] = newWorklistSize;

        for (uint ii = 0; ii < newWorklistSize; ii++) {
            // uint candidate = d_worklist[worklistOffset + ii];

            // Find the closest unvisited node, set it as the parent for the next iteration, and
            // mark it as visited.
            if (!d_worklistVisited[worklistOffset + ii]) {
                *d_nextIter                            = true;
                d_hasParent[queryID]                   = true;
                d_parents[queryID]                     = d_worklist[worklistOffset + ii];
                d_worklistVisited[worklistOffset + ii] = true;

                break;
            }
        }
    }
}

// Performs greedy search and returns the visited sets
template <typename T__>
[[nodiscard]] uint* greedySearch(uint8_t*    d_graph,
                                 const uint* d_delete_list,
                                 size_t      d_delete_list_size,
                                 T__*        d_queryVecs,
                                 uint*       d_visitedSet /*empty*/,
                                 uint*       d_visitedSetCount /*0*/,
                                 uint        batchSize) {
    bool* d_hasParent;  // 10k
    uint* d_parents;    // 10k uint
    bool* d_bloomFilters;

    gpuErrchk(cudaMalloc(&d_hasParent, batchSize * sizeof(bool)));
    gpuErrchk(cudaMalloc(&d_parents, batchSize * sizeof(uint)));
    size_t allocSize = batchSize * BF_MEMORY * sizeof(bool);

    printf("Allocating %.2f MB (%zu bytes) for d_bloomFilters\n",
           allocSize / (1024.0 * 1024.0),
           allocSize);

    gpuErrchk(cudaMalloc(&d_bloomFilters, batchSize * BF_MEMORY * sizeof(bool)));
    gpuErrchk(cudaMemset(d_bloomFilters, 0, batchSize * BF_MEMORY * sizeof(bool)));

    uint* d_neighbors;         // 10k * (64+1) uint
    uint* d_neighborsCount;    // 10k * 1
    T__*  d_neighborDists;     // 10k * (64+1) T__
    uint* d_neighborsAux;      // 10k * (64+1) uint
    T__*  d_neighborDistsAux;  // 10k * (64+1) T__

    gpuErrchk(cudaMalloc(&d_neighbors, batchSize * (FreshVamana::Consts::R_g + 1) * sizeof(uint)));
    gpuErrchk(
        cudaMemset(d_neighbors, 0, batchSize * (FreshVamana::Consts::R_g + 1) * sizeof(uint)));

    gpuErrchk(cudaMalloc(&d_neighborsCount, batchSize * sizeof(uint)));
    gpuErrchk(cudaMemset(d_neighborsCount, 0, batchSize * sizeof(uint)));

    gpuErrchk(
        cudaMalloc(&d_neighborDists, batchSize * (FreshVamana::Consts::R_g + 1) * sizeof(T__)));
    gpuErrchk(
        cudaMalloc(&d_neighborsAux, batchSize * (FreshVamana::Consts::R_g + 1) * sizeof(uint)));
    gpuErrchk(
        cudaMalloc(&d_neighborDistsAux, batchSize * (FreshVamana::Consts::R_g + 1) * sizeof(T__)));

    // 150 is worklist size
    uint* d_worklist;         // 10k * 150 uint
    uint* d_worklistCount;    // 10k uint
    T__*  d_worklistDist;     // 10k * 150 T__
    bool* d_worklistVisited;  // 10k * 150 bool

    gpuErrchk(cudaMalloc(&d_worklist, batchSize * FreshVamana::Consts::L_g * sizeof(uint)));

    gpuErrchk(cudaMalloc(&d_worklistCount, batchSize * sizeof(uint)));
    gpuErrchk(cudaMemset(d_worklistCount, 0, batchSize * sizeof(uint)));

    gpuErrchk(cudaMalloc(&d_worklistDist, batchSize * FreshVamana::Consts::L_g * sizeof(T__)));
    gpuErrchk(cudaMalloc(&d_worklistVisited, batchSize * FreshVamana::Consts::L_g * sizeof(bool)));

    bool  nextIter;
    bool* d_nextIter;

    gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));

    initializeParents<<<batchSize, 1>>>(d_hasParent, d_parents);
    initializeWorklist<T__><<<batchSize, 1>>>(
        d_graph, d_queryVecs, d_worklist, d_worklistCount, d_worklistDist, d_worklistVisited);
    // cudaDeviceSynchronize();

    int iter = 0;
    do {
        iter++;
        gpuErrchk(cudaMemset(d_nextIter, false, sizeof(bool)));

        filterNeighbors<T__><<<batchSize, FreshVamana::Consts::R_g>>>(d_graph,
                                                                      d_delete_list,
                                                                      d_delete_list_size,
                                                                      d_hasParent,
                                                                      d_parents,
                                                                      d_bloomFilters,
                                                                      d_neighbors,
                                                                      d_neighborsCount,
                                                                      d_visitedSet,
                                                                      d_visitedSetCount);

        // gpuErrchk(cudaDeviceSynchronize());

        computeDists<T__>
            <<<batchSize, FreshVamana::Consts::R_g * 8>>>(d_graph,
                                                          d_neighbors,
                                                          d_neighborsCount,
                                                          d_queryVecs,
                                                          d_neighborDists,
                                                          (FreshVamana::Consts::R_g + 1));
        // gpuErrchk(cudaDeviceSynchronize());

        sortByDistance<T__>
            <<<batchSize, FreshVamana::Consts::R_g, FreshVamana::Consts::R_g * sizeof(uint)>>>(
                d_neighbors,
                d_neighborsCount,
                d_neighborDists,
                d_neighborsAux,
                d_neighborDistsAux,
                FreshVamana::Consts::R_g + 1);
        // gpuErrchk(cudaDeviceSynchronize());

        mergeIntoWorklist<T__>
            <<<batchSize, FreshVamana::Consts::R_g + FreshVamana::Consts::L_g>>>(d_worklistCount,
                                                                                 d_worklist,
                                                                                 d_worklistDist,
                                                                                 d_worklistVisited,

                                                                                 d_neighborsCount,
                                                                                 d_neighbors,
                                                                                 d_neighborDists,

                                                                                 d_hasParent,
                                                                                 d_parents,
                                                                                 d_nextIter);

        gpuErrchk(cudaMemcpy(&nextIter, d_nextIter, sizeof(bool), cudaMemcpyDeviceToHost));
    } while (nextIter);

    gpuErrchk(cudaFree(d_hasParent));
    gpuErrchk(cudaFree(d_parents));
    gpuErrchk(cudaFree(d_bloomFilters));

    gpuErrchk(cudaFree(d_neighbors));
    gpuErrchk(cudaFree(d_neighborsCount));
    gpuErrchk(cudaFree(d_neighborDists));
    gpuErrchk(cudaFree(d_neighborsAux));
    gpuErrchk(cudaFree(d_neighborDistsAux));

    // gpuErrchk(cudaFree(d_worklist));
    gpuErrchk(cudaFree(d_worklistCount));
    gpuErrchk(cudaFree(d_worklistDist));
    gpuErrchk(cudaFree(d_worklistVisited));

    gpuErrchk(cudaFree(d_nextIter));

    printf("Greedy search finished in %d iterations.\n", iter);

    return d_worklist;
}
