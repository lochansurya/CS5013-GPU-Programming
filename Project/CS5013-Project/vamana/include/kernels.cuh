#pragma once
#include "constants.cuh"
#include "globals.cuh"
#include "graphT.cuh"
#include "timer.h"
#include "utils.cuh"

// #include <cublas.h>
#include <cuda/std/limits>

typedef enum : uint8_t { INIT, PRUNED, NEIGHBOR } NodeState;

// TODO: optimize this
template <typename T__>
__global__ void computeDists(uint8_t* d_graph,
                             uint*    d_nodes,
                             uint*    d_node_count,
                             T__*     d_query_vecs,
                             T__*     d_dists,
                             uint     row_size) {
    using namespace FreshVamana;

    uint query_id = blockIdx.x;
    uint tid      = threadIdx.x;

    T__* query_vec = d_query_vecs + Consts::D_g * query_id;  // Pointer to query vector
    uint offset    = row_size * query_id;
    uint num_nodes = d_node_count[query_id];

    // Initialize distances to zero
    for (uint i = tid; i < num_nodes; i += blockDim.x) {
        d_dists[offset + i] = 0;
    }

    __syncthreads();

    // if (queryID == 0 & tid == 0) printf("NumNodes: %d\n", numNodes);

    // Assign 8 threads to each node
    for (uint j = tid / 8; j < num_nodes; j += (blockDim.x + 7) / 8) {
        uint node = d_nodes[offset + j];
        T__* node_vec =
            (T__*)(d_graph + Consts::graph_entry_bytes_g * node);  // Pointer to node vector

        T__ sum = 0;

        // Sum up 8 dimensions in parallel
        for (uint i = tid % 8; i < Consts::D_g; i += 8) {
            T__ diff = node_vec[i] - query_vec[i];
            sum += diff * diff;
        }
        atomicAdd(&d_dists[offset + j], sum);
    }
}

template <typename T__>
__device__ uint lowerBound(T__ arr[], uint lo, uint hi, T__ target) {
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (target > arr[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

template <typename T__>
__device__ uint upperBound(T__ arr[], uint lo, uint hi, T__ target) {
    while (lo < hi) {
        uint mid = (lo + hi) / 2;
        if (target >= arr[mid]) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

template <typename T__>
__global__ void sortByDistance(uint* d_items,
                               uint* d_item_count,
                               T__*  d_dists,
                               uint* d_items_aux,
                               T__*  d_dists_aux,
                               uint  row_size) {
    uint query_id = blockIdx.x;
    uint tid      = threadIdx.x;

    uint numItems = d_item_count[query_id];
    uint offset   = query_id * row_size;

    extern __shared__ uint sorted_positions[];

    for (uint subarray_size = 2; subarray_size < 2 * numItems; subarray_size *= 2) {
        uint subarray_id = tid / subarray_size;
        uint start       = subarray_id * subarray_size;
        uint mid         = min(start + subarray_size / 2, numItems);
        uint end         = min(start + subarray_size, numItems);

        uint before;

        if (tid >= start && tid < mid) {
            // If current thread corresponds to lower half, find the no. of elements before this
            // element from the upper half
            before = lowerBound<T__>(&d_dists[offset + mid], 0, end - mid, d_dists[offset + tid]);
            sorted_positions[tid] = tid + before;
        } else if (tid >= mid && tid < end) {
            // If current thread corresponds to upper half, find the no. of elements before this
            // element from the lower half
            before =
                upperBound<T__>(&d_dists[offset + start], 0, mid - start, d_dists[offset + tid]);
            sorted_positions[tid] = before + (tid - mid + start);
        }

        __syncthreads();
        __threadfence_block();

        // Copy the neigbors to correct positions in auxiliary array
        for (uint i = tid; i < numItems; i += blockDim.x) {
            d_items_aux[offset + sorted_positions[i]] = d_items[offset + i];
            d_dists_aux[offset + sorted_positions[i]] = d_dists[offset + i];
        }

        __syncthreads();
        __threadfence_block();

        // Copy from auxiliary array back into original array
        for (uint i = tid; i < numItems; i += blockDim.x) {
            d_items[offset + i] = d_items_aux[offset + i];
            d_dists[offset + i] = d_dists_aux[offset + i];
        }

        __syncthreads();
        __threadfence_block();
    }
}

template <typename T__>
__global__ void getNeighbors(uint8_t* d_graph,
                             uint     batch_start,
                             uint*    d_neighbors,
                             uint*    d_neighbors_count) {
    using namespace FreshVamana;

    uint queryID         = blockIdx.x;
    uint extendedQueryID = batch_start + queryID;
    uint tid             = threadIdx.x;

    uint* degreePtr   = (uint*)(d_graph + extendedQueryID * Consts::graph_entry_bytes_g +
                              Consts::D_g * sizeof(T__));
    uint* neighborPtr = degreePtr + 1;

    uint degree = *degreePtr;

    if (tid == 0) {
        d_neighbors_count[queryID] = degree;
    }

    // TODO: Test necessity
    __syncthreads();

    // Loop over each neighbor
    for (uint ii = tid; ii < degree; ii += blockDim.x) {
        d_neighbors[(Consts::R_g + 1) * queryID + ii] = neighborPtr[ii];
    }
}

// Could be merged with mergeIntoWorklist (with a dummy array for d_visited))
template <typename T__>
__global__ void mergeIntoVisitedSet(uint* d_visited_set_count,
                                    uint* d_visited_set,
                                    T__*  d_visited_set_dists,
                                    uint* d_neighbors_count,
                                    uint* d_neighbors,
                                    T__*  d_neighbors_dist) {
    using namespace FreshVamana;

    uint query_id = blockIdx.x;
    uint tid      = threadIdx.x;

    uint visited_set_offset = query_id * Consts::max_num_parents_per_query;
    uint neighbors_offset   = query_id * (Consts::R_g + 1);

    uint num_neighbors    = d_neighbors_count[query_id];
    uint visited_set_size = d_visited_set_count[query_id];

    uint new_visited_set_size =
        min(num_neighbors + visited_set_size, Consts::max_num_parents_per_query);

    uint id;
    T__  dist;
    uint new_pos = Consts::max_num_parents_per_query;

    if (tid < visited_set_size) {
        uint before = lowerBound<T__>(&d_neighbors_dist[neighbors_offset],
                                      0,
                                      num_neighbors,
                                      d_visited_set_dists[visited_set_offset + tid]);
        id          = d_visited_set[visited_set_offset + tid];
        dist        = d_visited_set_dists[visited_set_offset + tid];
        new_pos     = before + tid;
    } else if (tid >= Consts::max_num_parents_per_query &&
               tid < Consts::max_num_parents_per_query + num_neighbors) {
        uint idx    = tid - Consts::max_num_parents_per_query;
        uint before = upperBound<T__>(&d_visited_set_dists[visited_set_offset],
                                      0,
                                      visited_set_size,
                                      d_neighbors_dist[neighbors_offset + idx]);
        id          = d_neighbors[neighbors_offset + idx];
        dist        = d_neighbors_dist[neighbors_offset + idx];
        new_pos     = before + idx;
    }

    __syncthreads();

    if (new_pos < new_visited_set_size) {
        d_visited_set[visited_set_offset + new_pos]       = id;
        d_visited_set_dists[visited_set_offset + new_pos] = dist;
    }

    __syncthreads();

    if (tid == 0) {
        d_visited_set_count[query_id] = new_visited_set_size;
    }
}

// Robust Prune
template <typename T__>
__global__ void pruneOutNeighbors(uint8_t*   d_graph,
                                  uint       batch_start,
                                  uint*      d_visited_set,
                                  uint*      d_visited_set_count,
                                  T__*       d_visited_set_dists,
                                  NodeState* d_visited_set_status,
                                  T__*       d_query_vecs,
                                  uint8_t*   d_reverse_edge_index,
                                  float      alpha) {
    using namespace FreshVamana;

    for (uint iter = 1;; iter++) {
        uint query_id          = blockIdx.x;
        uint extended_query_id = batch_start + query_id;
        uint tid               = threadIdx.x;

        uint num_nodes          = d_visited_set_count[query_id];
        uint visited_set_offset = query_id * Consts::max_num_parents_per_query;

        uint* degree_ptr   = (uint*)(d_graph + extended_query_id * Consts::graph_entry_bytes_g +
                                   Consts::D_g * sizeof(T__));
        uint* neighbor_ptr = degree_ptr + 1;

        // Initialization
        if (tid == 0 && iter == 1) {
            // Mark all candidate nodes as INIT
            for (uint i = 0; i < num_nodes; i++) {
                d_visited_set_status[visited_set_offset + i] = INIT;
            }

            // Set degree to zero
            *degree_ptr = 0;
        }

        __syncthreads();

        // Don't add more neighbors if we already have R neighbors
        if (*degree_ptr >= Consts::R_g)
            return;

        __shared__ uint p_star_shared[1];
        *p_star_shared = cuda::std::numeric_limits<uint>::max();

        // Find p_star
        if (tid == 0) {
            for (uint i = 0; i < num_nodes; i++) {
                // Find the closest 'INIT' node (p_star)
                if (d_visited_set_status[visited_set_offset + i] != INIT)
                    continue;

                *p_star_shared = d_visited_set[visited_set_offset + i];

                // Add an edge from the query to p_star
                uint old_degree          = atomicAdd(degree_ptr, 1);
                neighbor_ptr[old_degree] = *p_star_shared;

                // Set it to neighbor
                d_visited_set_status[visited_set_offset + i] = NEIGHBOR;

                // We need to add a reverse edge from p_star to query
                uint* entryPtr = (uint*)&d_reverse_edge_index[*p_star_shared *
                                                              Consts::reverse_index_entry_bytes_g];
                uint  oldLen   = atomicAdd(entryPtr, 1);
                if (oldLen < Consts::max_reverse_index_entries_g) {
                    entryPtr[1 + oldLen] = extended_query_id;
                } else {
                    // printf("Reverse index limit hit: %d\n", queryID);
                    atomicSub(entryPtr, 1);
                }

                // *d_nextIter = true;
                break;
            }
        }

        __syncthreads();

        uint pStar = *p_star_shared;
        if (pStar == cuda::std::numeric_limits<uint>::max()) {
            return;
        }

        // Copy p_star into shared memory
        __shared__ T__ pStarVec[Consts::D_g];
        T__*           vecPtr =
            (T__*)(d_graph + pStar * Consts::graph_entry_bytes_g);  // Pointer to query vector
        for (uint ii = tid; ii < Consts::D_g; ii += blockDim.x) {
            pStarVec[ii] = vecPtr[ii];
        }

        __syncthreads();

        uint laneId        = threadIdx.x & 31;
        uint warpId        = threadIdx.x >> 5;  // warp index within block
        uint warpsPerBlock = blockDim.x >> 5;

        for (uint ii = warpId; ii < num_nodes; ii += warpsPerBlock) {
            if (d_visited_set_status[visited_set_offset + ii] != INIT)
                continue;

            uint       p = d_visited_set[visited_set_offset + ii];
            const T__* pVec =
                reinterpret_cast<const T__*>(d_graph + p * Consts::graph_entry_bytes_g);

            // cooperative distance computation
            float partial = 0.0f;
            for (uint j = laneId; j < Consts::D_g; j += 32) {
                T__ diff = pVec[j] - pStarVec[j];
                partial  = fmaf(diff, diff, partial);
            }

            // warp reduce sum
            for (int offset = 16; offset > 0; offset >>= 1)
                partial += __shfl_down_sync(0xffffffff, partial, offset);

            if (laneId == 0) {
                T__ queryDist = d_visited_set_dists[visited_set_offset + ii];
                if (partial * alpha <= static_cast<float>(queryDist)) {
                    d_visited_set_status[visited_set_offset + ii] = PRUNED;
                }
            }
        }
    }
}

template <typename T__>
void computeOutNeighbors(uint8_t* d_graph,
                         T__*     d_query_vecs,
                         uint*    d_visited_sets,
                         uint*    d_visited_set_count,
                         float    alpha,
                         uint8_t* d_reverse_edge_index,
                         uint     batch_start,
                         uint     batch_size) {
    using namespace FreshVamana;

    bool log = false;

    cudaStream_t stream = 0;  // default stream
    GPUTimer     gputimer(stream, !log);
    // printf("%d\n", batchSize);

    T__*       d_visitedSetDists;
    uint*      d_visitedSetAux;
    T__*       d_visitedSetDistsAux;
    NodeState* d_visitedSetStatus;

    gpuErrchk(cudaMalloc(&d_visitedSetDists,
                         batch_size * Consts::max_num_parents_per_query * sizeof(T__)));
    gpuErrchk(cudaMalloc(&d_visitedSetAux,
                         batch_size * Consts::max_num_parents_per_query * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_visitedSetDistsAux,
                         batch_size * Consts::max_num_parents_per_query * sizeof(T__)));
    gpuErrchk(cudaMalloc(&d_visitedSetStatus,
                         batch_size * Consts::max_num_parents_per_query * sizeof(NodeState)));

    uint* d_neighbors;
    uint* d_neighborsCount;
    T__*  d_neighborsDists;
    uint* d_neighborsAux;
    T__*  d_neighborsDistsAux;

    gpuErrchk(cudaMalloc(&d_neighbors, batch_size * (Consts::R_g + 1) * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_neighborsCount, batch_size * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_neighborsDists, batch_size * (Consts::R_g + 1) * sizeof(T__)));
    gpuErrchk(cudaMalloc(&d_neighborsAux, batch_size * (Consts::R_g + 1) * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_neighborsDistsAux, batch_size * (Consts::R_g + 1) * sizeof(T__)));

    // bool nextIter;
    // bool *d_nextIter;
    // gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));

    gputimer.Start();
    getNeighbors<T__>
        <<<batch_size, Consts::R_g>>>(d_graph, batch_start, d_neighbors, d_neighborsCount);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("getNeighbors GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    computeDists<T__><<<batch_size, Consts::R_g * 8>>>(d_graph,
                                                       d_neighbors,
                                                       d_neighborsCount,
                                                       d_query_vecs,
                                                       d_neighborsDists,
                                                       (FreshVamana::Consts::R_g + 1));
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("computeDists GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    sortByDistance<T__>
        <<<batch_size, Consts::R_g + 1, (Consts::R_g + 1) * sizeof(uint)>>>(d_neighbors,
                                                                            d_neighborsCount,
                                                                            d_neighborsDists,
                                                                            d_neighborsAux,
                                                                            d_neighborsDistsAux,
                                                                            Consts::R_g + 1);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("sortByDistance GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    computeDists<T__><<<batch_size, 1024>>>(d_graph,
                                            d_visited_sets,
                                            d_visited_set_count,
                                            d_query_vecs,
                                            d_visitedSetDists,
                                            Consts::max_num_parents_per_query);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("computeDists GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    sortByDistance<T__>
        <<<batch_size,
           Consts::max_num_parents_per_query,
           Consts::max_num_parents_per_query * sizeof(uint)>>>(d_visited_sets,
                                                               d_visited_set_count,
                                                               d_visitedSetDists,
                                                               d_visitedSetAux,
                                                               d_visitedSetDistsAux,
                                                               Consts::max_num_parents_per_query);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("sortByDistance GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    mergeIntoVisitedSet<T__>
        <<<batch_size, Consts::max_num_parents_per_query + Consts::R_g>>>(d_visited_set_count,
                                                                          d_visited_sets,
                                                                          d_visitedSetDists,
                                                                          d_neighborsCount,
                                                                          d_neighbors,
                                                                          d_neighborsDists);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("mergeIntoVisitedSet GPU time: %f ms\n", gputimer.Elapsed());

    gputimer.Start();
    pruneOutNeighbors<T__><<<batch_size, 32>>>(d_graph,
                                               batch_start,
                                               d_visited_sets,
                                               d_visited_set_count,
                                               d_visitedSetDists,
                                               d_visitedSetStatus,

                                               d_query_vecs,
                                               d_reverse_edge_index,
                                               alpha);
    gputimer.Stop();
    // gpuErrchk(cudaDeviceSynchronize());
    // printf("pruneOutNeighbors GPU time: %f ms\n", gputimer.Elapsed());

    gpuErrchk(cudaFree(d_visitedSetDists));
    gpuErrchk(cudaFree(d_visitedSetAux));
    gpuErrchk(cudaFree(d_visitedSetDistsAux));
    gpuErrchk(cudaFree(d_visitedSetStatus));

    gpuErrchk(cudaFree(d_neighbors));
    gpuErrchk(cudaFree(d_neighborsCount));
    gpuErrchk(cudaFree(d_neighborsDists));
    gpuErrchk(cudaFree(d_neighborsAux));
    gpuErrchk(cudaFree(d_neighborsDistsAux));
    // gpuErrchk(cudaFree(d_nextIter));

    // printf("Out neighbor pruning finished in %d iterations.\n", iter);
}

template <typename T__>
__global__ void loadQueryVecs(uint8_t* d_graph, T__* d_queryVecs) {
    using namespace FreshVamana;

    uint query_id = blockIdx.x;
    uint tid      = threadIdx.x;

    T__* query_vec = (T__*)(d_graph + query_id * Consts::graph_entry_bytes_g);

    for (uint i = tid; i < Consts::D_g; i += blockDim.x) {
        d_queryVecs[query_id * Consts::D_g + i] = query_vec[i];
    }
}

// Convert from the byte-array representation into the usual array-and-length representation
__global__ void parseReverseIndex(uint8_t* d_reverseEdgeIndex,
                                  uint*    d_reverseEdges,
                                  uint*    d_reverseEdgeCount,
                                  uint*    degreeCounts) {
    using namespace FreshVamana;

    uint queryID = blockIdx.x;
    uint tid     = threadIdx.x;

    uint* entryPtr = (uint*)(d_reverseEdgeIndex + queryID * Consts::reverse_index_entry_bytes_g);
    uint  numReverseEdges = *entryPtr;

    if (tid == 0) {
        d_reverseEdgeCount[queryID] = numReverseEdges;
        atomicAdd(&degreeCounts[numReverseEdges], 1);
    }

    for (uint ii = tid; ii < numReverseEdges; ii += blockDim.x) {
        d_reverseEdges[queryID * Consts::max_reverse_index_entries_g + ii] = entryPtr[1 + ii];
    }
}

__global__ void getPrunableQueryIDs(uint* d_reverseEdgeCount,
                                    uint* d_queryIDs,
                                    uint* d_queryCount) {
    using namespace FreshVamana;

    int  tid  = threadIdx.x;
    int  lane = tid % 32;
    uint i    = blockIdx.x * blockDim.x + threadIdx.x;
    uint flag = (i < FreshVamana::Globals::d_graph_size_g) && (d_reverseEdgeCount[i] != 0);
    if (i >= FreshVamana::Globals::d_graph_size_g)
        return;

    // uint flag = (d_reverseEdgeCount[i] != 0);
    uint mask       = __ballot_sync(0xffffffff, flag);
    int  warpActive = __popc(mask);

    uint warpBase = 0;
    if (lane == 0) {
        warpBase = atomicAdd(d_queryCount, warpActive);
    }
    warpBase = __shfl_sync(0xffffffff, warpBase, 0);

    int posInWarp = __popc(mask & ((1u << lane) - 1));
    if (flag)
        d_queryIDs[warpBase + posInWarp] = i;
}

// Could be merged with mergeIntoVisitedSets
template <typename T__>
__global__ void mergeIntoReverseEdges(uint* d_reverseEdgeCount,
                                      uint* d_reverseEdges,
                                      T__*  d_reverseEdgeDists,
                                      uint* d_neighborsCount,
                                      uint* d_neighbors,
                                      T__*  d_neighborsDist) {
    using namespace FreshVamana;

    uint queryID = blockIdx.x;
    uint tid     = threadIdx.x;

    uint reverseEdgeOffset = queryID * Consts::max_reverse_index_entries_g;
    uint neighborsOffset   = queryID * (Consts::R_g + 1);

    uint numNeighbors     = d_neighborsCount[queryID];
    uint reverseEdgeCount = d_reverseEdgeCount[queryID];

    uint newReverseEdgeCount =
        min(numNeighbors + reverseEdgeCount, Consts::max_reverse_index_entries_g);

    uint id;
    T__  dist;
    uint newPos = Consts::max_reverse_index_entries_g;

    if (tid < reverseEdgeCount) {
        uint before = lowerBound<T__>(&d_neighborsDist[neighborsOffset],
                                      0,
                                      numNeighbors,
                                      d_reverseEdgeDists[reverseEdgeOffset + tid]);
        id          = d_reverseEdges[reverseEdgeOffset + tid];
        dist        = d_reverseEdgeDists[reverseEdgeOffset + tid];
        newPos      = before + tid;
    } else if (tid >= Consts::max_reverse_index_entries_g &&
               tid < Consts::max_reverse_index_entries_g + numNeighbors) {
        uint idx    = tid - Consts::max_reverse_index_entries_g;
        uint before = upperBound<T__>(&d_reverseEdgeDists[reverseEdgeOffset],
                                      0,
                                      reverseEdgeCount,
                                      d_neighborsDist[neighborsOffset + idx]);
        id          = d_neighbors[neighborsOffset + idx];
        dist        = d_neighborsDist[neighborsOffset + idx];
        newPos      = before + idx;
    }

    __syncthreads();

    if (newPos < newReverseEdgeCount) {
        d_reverseEdges[reverseEdgeOffset + newPos]     = id;
        d_reverseEdgeDists[reverseEdgeOffset + newPos] = dist;
    }

    __syncthreads();

    if (tid == 0) {
        d_reverseEdgeCount[queryID] = newReverseEdgeCount;
    }
}

template <typename T__>
__global__ void pruneReverseEdges(uint8_t*   d_graph,
                                  uint*      d_queryIDs,
                                  uint*      d_reverseEdges,
                                  uint*      d_reverseEdgeCount,
                                  T__*       d_reverseEdgeDists,
                                  NodeState* d_reverseEdgeStatus,
                                  T__*       d_queryVecs,
                                  float      alpha) {
    using namespace FreshVamana;

    for (uint iter = 1;; iter++) {
        // uint bid = blockIdx.x;
        // uint queryID = d_queryIDs[bid];
        uint queryID = blockIdx.x;
        uint tid     = threadIdx.x;

        uint numNodes          = d_reverseEdgeCount[queryID];
        uint reverseEdgeOffset = queryID * Consts::max_reverse_index_entries_g;

        uint* degreePtr =
            (uint*)(d_graph + queryID * Consts::graph_entry_bytes_g + Consts::D_g * sizeof(T__));
        uint* neighborPtr = degreePtr + 1;

        // printf("%d\n", queryID);

        if (tid == 0 && iter == 1) {
            // Mark all candidate nodes as INIT
            for (uint i = 0; i < numNodes; i++) {
                d_reverseEdgeStatus[reverseEdgeOffset + i] = INIT;
            }

            // Set degree to zero
            *degreePtr = 0;
        }

        __syncthreads();

        // Don't add more neighbors if we already have R neighbors
        if (*degreePtr >= Consts::R_g)
            return;

        __shared__ uint pStarShared[1];
        *pStarShared = cuda::std::numeric_limits<uint>::max();

        // Find p_star
        if (tid == 0) {
            for (uint i = 0; i < numNodes; i++) {
                // Find the closest 'INIT' node (p_star)
                if (d_reverseEdgeStatus[reverseEdgeOffset + i] != INIT)
                    continue;

                *pStarShared = d_reverseEdges[reverseEdgeOffset + i];

                // Add an edge from the query to p_star
                // TODO: Are we sure that oldDegree is always less than R
                uint oldDegree         = atomicAdd(degreePtr, 1);
                neighborPtr[oldDegree] = *pStarShared;

                if (oldDegree > Consts::R_g)
                    printf("Backward degree exceeded R: %d\n", queryID);

                // Set it to neighbor
                d_reverseEdgeStatus[reverseEdgeOffset + i] = NEIGHBOR;
                // *d_nextIter = true;
                break;
            }
        }

        __syncthreads();

        uint pStar = *pStarShared;
        if (pStar == cuda::std::numeric_limits<uint>::max())
            return;

        // Copy p_star into shared memory
        __shared__ T__ pStarVec[Consts::D_g];
        T__*           vecPtr =
            (T__*)(d_graph + pStar * Consts::graph_entry_bytes_g);  // Pointer to query vector
        for (uint ii = tid; ii < Consts::D_g; ii += blockDim.x) {
            pStarVec[ii] = vecPtr[ii];
        }

        __syncthreads();

        uint laneId        = threadIdx.x & 31;
        uint warpId        = threadIdx.x >> 5;  // warp index within block
        uint warpsPerBlock = blockDim.x >> 5;

        for (uint ii = warpId; ii < numNodes; ii += warpsPerBlock) {
            if (d_reverseEdgeStatus[reverseEdgeOffset + ii] != INIT)
                continue;

            uint       p = d_reverseEdges[reverseEdgeOffset + ii];
            const T__* pVec =
                reinterpret_cast<const T__*>(d_graph + p * Consts::graph_entry_bytes_g);

            // cooperative distance computation
            T__ partial = static_cast<T__>(0);
            for (uint j = laneId; j < Consts::D_g; j += 32) {
                T__ diff = pVec[j] - pStarVec[j];
                // TODO: change this?
                partial = fmaf(diff, diff, partial);
            }

            // warp reduce sum
            for (int offset = 16; offset > 0; offset >>= 1)
                partial += __shfl_down_sync(0xffffffff, partial, offset);

            if (laneId == 0) {
                T__ queryDist = d_reverseEdgeDists[reverseEdgeOffset + ii];
                if (partial * alpha <= static_cast<float>(queryDist)) {
                    d_reverseEdgeStatus[reverseEdgeOffset + ii] = PRUNED;
                }
            }
        }
    }
}

template <typename T__>
void computeReverseEdges(uint8_t* d_graph, uint8_t* d_reverseEdgeIndex, float alpha) {
    // TODO: Replace queryVecs with something better. It seems like a good idea to do this pruning
    // in batches
    using namespace FreshVamana;

    T__* d_queryVecs;

    gpuErrchk(
        cudaMalloc(&d_queryVecs, FreshVamana::Globals::d_graph_size_g * Consts::D_g * sizeof(T__)));

    uint*      d_reverseEdges;
    uint*      d_reverseEdgeCount;
    T__*       d_reverseEdgeDists;
    uint*      d_reverseEdgesAux;
    T__*       d_reverseEdgeDistsAux;
    NodeState* d_reverseEdgeStatus;

    gpuErrchk(cudaMalloc(
        &d_reverseEdges,
        FreshVamana::Globals::d_graph_size_g * Consts::max_reverse_index_entries_g * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeCount, FreshVamana::Globals::d_graph_size_g * sizeof(uint)));
    gpuErrchk(cudaMalloc(
        &d_reverseEdgeDists,
        FreshVamana::Globals::d_graph_size_g * Consts::max_reverse_index_entries_g * sizeof(T__)));
    gpuErrchk(cudaMalloc(
        &d_reverseEdgesAux,
        FreshVamana::Globals::d_graph_size_g * Consts::max_reverse_index_entries_g * sizeof(uint)));
    gpuErrchk(cudaMalloc(
        &d_reverseEdgeDistsAux,
        FreshVamana::Globals::d_graph_size_g * Consts::max_reverse_index_entries_g * sizeof(T__)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeStatus,
                         FreshVamana::Globals::d_graph_size_g *
                             Consts::max_reverse_index_entries_g * sizeof(NodeState)));

    uint* d_neighbors;
    uint* d_neighborsCount;
    T__*  d_neighborDists;
    uint* d_neighborsAux;
    T__*  d_neighborDistsAux;

    gpuErrchk(cudaMalloc(&d_neighbors,
                         FreshVamana::Globals::d_graph_size_g * (Consts::R_g + 1) * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_neighborsCount, FreshVamana::Globals::d_graph_size_g * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_neighborDists,
                         FreshVamana::Globals::d_graph_size_g * (Consts::R_g + 1) * sizeof(T__)));
    gpuErrchk(cudaMalloc(&d_neighborsAux,
                         FreshVamana::Globals::d_graph_size_g * (Consts::R_g + 1) * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_neighborDistsAux,
                         FreshVamana::Globals::d_graph_size_g * (Consts::R_g + 1) * sizeof(T__)));

    // bool nextIter;
    // bool *d_nextIter;

    // gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));

    loadQueryVecs<<<FreshVamana::Globals::d_graph_size_g, Consts::D_g>>>(d_graph, d_queryVecs);

    uint* degreeSum;
    gpuErrchk(cudaMalloc(&degreeSum, (Consts::max_reverse_index_entries_g + 1) * sizeof(uint)));
    gpuErrchk(cudaMemset(degreeSum, 0, (Consts::max_reverse_index_entries_g + 1) * sizeof(uint)));

    parseReverseIndex<<<FreshVamana::Globals::d_graph_size_g, 1024>>>(
        d_reverseEdgeIndex, d_reverseEdges, d_reverseEdgeCount, degreeSum);

    // uint h_degreeCounts[MAX_REVERSE_INDEX_ENTRIES + 1];
    // cudaMemcpy(&h_degreeCounts, degreeSum, (MAX_REVERSE_INDEX_ENTRIES + 1) * sizeof(uint),
    // cudaMemcpyDeviceToHost);

    // for (int i = 0; i <= MAX_REVERSE_INDEX_ENTRIES; i++) {
    //     if (h_degreeCounts[i] != 0)
    //         printf("%d:%d ", i, h_degreeCounts[i]);
    // }
    // printf("\n");

    uint  h_queryCount;
    uint *d_queryIDs, *d_queryCount;
    gpuErrchk(cudaMalloc(&d_queryIDs, FreshVamana::Globals::d_graph_size_g * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_queryCount, sizeof(uint)));
    const int numThreads = 32;
    getPrunableQueryIDs<<<(FreshVamana::Globals::d_graph_size_g + numThreads - 1) / numThreads,
                          numThreads>>>(d_reverseEdgeCount, d_queryIDs, d_queryCount);
    cudaMemcpy(&h_queryCount, d_queryCount, sizeof(uint), cudaMemcpyDeviceToHost);

    // printf("%d\n", h_queryCount)

    // Can't use 8*MAX_REVERSE_INDEX_ENTRIES because it exceeds block size limit
    computeDists<T__>
        <<<FreshVamana::Globals::d_graph_size_g, 1024>>>(d_graph,
                                                         d_reverseEdges,
                                                         d_reverseEdgeCount,
                                                         d_queryVecs,
                                                         d_reverseEdgeDists,
                                                         Consts::max_reverse_index_entries_g);

    // sortByDistance<<<N, MAX_REVERSE_INDEX_ENTRIES,
    sortByDistance<T__><<<FreshVamana::Globals::d_graph_size_g,
                          1024,
                          Consts::max_reverse_index_entries_g * sizeof(uint)>>>(
        d_reverseEdges,
        d_reverseEdgeCount,
        d_reverseEdgeDists,
        d_reverseEdgesAux,
        d_reverseEdgeDistsAux,
        Consts::max_reverse_index_entries_g);

    getNeighbors<T__><<<FreshVamana::Globals::d_graph_size_g, Consts::R_g>>>(
        d_graph, 0, d_neighbors, d_neighborsCount);

    computeDists<T__><<<FreshVamana::Globals::d_graph_size_g, Consts::R_g * 8>>>(
        d_graph, d_neighbors, d_neighborsCount, d_queryVecs, d_neighborDists, (Consts::R_g + 1));

    sortByDistance<T__><<<FreshVamana::Globals::d_graph_size_g,
                          Consts::R_g + 1,
                          (Consts::R_g + 1) * sizeof(uint)>>>(d_neighbors,
                                                              d_neighborsCount,
                                                              d_neighborDists,
                                                              d_neighborsAux,
                                                              d_neighborDistsAux,
                                                              Consts::R_g + 1);

    mergeIntoReverseEdges<T__><<<FreshVamana::Globals::d_graph_size_g, 1024>>>(
        d_reverseEdgeCount,
        // mergeIntoReverseEdges<<<N, R+MAX_REVERSE_INDEX_ENTRIES>>>(d_reverseEdgeCount,
        d_reverseEdges,
        d_reverseEdgeDists,
        d_neighborsCount,
        d_neighbors,
        d_neighborDists);

    // uint iter = 0;

    h_queryCount = FreshVamana::Globals::d_graph_size_g;
    pruneReverseEdges<T__><<<h_queryCount, 32>>>(d_graph,
                                                 d_queryIDs,
                                                 d_reverseEdges,
                                                 d_reverseEdgeCount,
                                                 d_reverseEdgeDists,
                                                 d_reverseEdgeStatus,
                                                 d_queryVecs,
                                                 alpha);

    gpuErrchk(cudaFree(d_queryVecs));

    gpuErrchk(cudaFree(d_reverseEdges));
    gpuErrchk(cudaFree(d_reverseEdgeCount));
    gpuErrchk(cudaFree(d_reverseEdgeDists));
    gpuErrchk(cudaFree(d_reverseEdgesAux));
    gpuErrchk(cudaFree(d_reverseEdgeDistsAux));
    gpuErrchk(cudaFree(d_reverseEdgeStatus));

    gpuErrchk(cudaFree(d_neighbors));
    gpuErrchk(cudaFree(d_neighborsCount));
    gpuErrchk(cudaFree(d_neighborDists));
    gpuErrchk(cudaFree(d_neighborsAux));
    gpuErrchk(cudaFree(d_neighborDistsAux));

    // printf("Reverse edge pruning finished in %d iterations.\n", iter);
}

template <typename T__>
__global__ void findPointsKernel(const uint8_t* __restrict__ d_graph,
                                 const T__* __restrict__ d_query_vecs,
                                 uint n_nodes,
                                 uint n_queries,
                                 int* __restrict__ d_results) {
    const uint qid = blockIdx.y;  // query index
    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_nodes || qid >= n_queries)
        return;

    const uint8_t* entry_ptr = d_graph + static_cast<size_t>(tid) * graph_entry_bytes_g;
    const T__*     vec_ptr   = reinterpret_cast<const T__*>(entry_ptr);

    const T__* query_vec = d_query_vecs + static_cast<size_t>(qid) * D_g;

    bool match = true;
#pragma unroll
    for (uint i = 0; i < D_g; ++i) {
        if (vec_ptr[i] != query_vec[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        atomicCAS(&d_results[qid], -1, static_cast<int>(tid));
    }
}

template <typename T__>
__global__ void copyNewVectorsToGraph(uint8_t*   d_graph,
                                      const T__* d_new_vecs,
                                      uint       old_graph_size,
                                      uint       num_new_nodes) {
    using namespace FreshVamana;
    uint new_node_idx = blockIdx.x;
    uint tid          = threadIdx.x;

    if (new_node_idx >= num_new_nodes)
        return;

    uint graph_node_id = old_graph_size + new_node_idx;

    T__* dest_vec_ptr = (T__*)(d_graph + graph_node_id * Consts::graph_entry_bytes_g);

    const T__* src_vec_ptr = d_new_vecs + new_node_idx * Consts::D_g;

    for (uint i = tid; i < Consts::D_g; i += blockDim.x) {
        dest_vec_ptr[i] = src_vec_ptr[i];
    }

    if (tid == 0) {
        uint* degree_ptr = (uint*)(dest_vec_ptr + Consts::D_g);
        *degree_ptr      = 0;
    }
}