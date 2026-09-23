#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>

/**
 * @file common.h
 * @brief Common definitions, constants, and data structures for the Vamana GPU engine.
 */

// Constants
#define MAX_DEGREE 32          ///< Maximum degree of the graph
#define WARP_SIZE 32           ///< CUDA Warp Size
#define VISITED_LIST_SIZE 128  ///< Size of the visited list for greedy search
#define BEAM_WIDTH 128         ///< Beam width for search

/**
 * @brief Macro for checking CUDA errors.
 * 
 * Exits the program if a CUDA call returns an error.
 */
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s at %s:%d\n", \
                    cudaGetErrorString(err), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)

/**
 * @brief Helper function to check fread result.
 * 
 * @param read_count Number of items read.
 * @param expected Expected number of items.
 * @param msg Error message prefix.
 */
inline void check_fread(size_t read_count, size_t expected, const char* msg) {
    if (read_count != expected) {
        fprintf(stderr, "Error reading %s: expected %zu, got %zu\n", msg, expected, read_count);
        exit(1);
    }
}

/**
 * @brief Operation types for the workload.
 */
enum OpType {
    OP_DELETE = 0, ///< Delete a node
    OP_INSERT = 1, ///< Insert a new node
    OP_SEARCH = 2  ///< Search for nearest neighbors
};

/**
 * @brief Represents a single operation in the workload.
 * 
 * @tparam DIM_SZ Dimension of the vectors.
 */
template <int DIM_SZ>
struct Operation {
    int type;            ///< Operation type (OpType)
    int id;              ///< Target ID for Delete, or -1 for Insert/Search
    float vector[DIM_SZ];///< Vector data associated with the operation
};

/**
 * @brief Manages GPU memory pointers and graph state.
 */
struct GraphContext {
    float* d_vectors;      ///< Node vectors [MAX_CAPACITY * DIM]
    int* d_adj;            ///< Adjacency list [MAX_CAPACITY * MAX_DEGREE]
    int* d_delete_mask;    ///< Delete mask [MAX_CAPACITY] (0=Active, 1=Deleted)
    int* d_counters;       ///< Global counters: [0]=Current Max ID, [1]=Freelist Count
    int* d_freelist;       ///< Stack of reusable IDs for deleted nodes
    int dim;               ///< Vector dimension
    int max_capacity;      ///< Maximum capacity of the graph
};

/**
 * @brief Candidate list structure for greedy search.
 * 
 * Maintains a sorted list of the best candidates found so far.
 * 
 * @tparam K Maximum number of candidates to store.
 */
template <int K>
struct CandidateList {
    int ids[K];      ///< Candidate IDs
    float dists[K];  ///< Distances to the query vector
    int size;        ///< Current number of candidates

    /**
     * @brief Initialize the candidate list.
     */
    __device__ void init() {
        size = 0;
        for (int i = 0; i < K; ++i) {
            ids[i] = -1;
            dists[i] = 1e30f;
        }
    }

    /**
     * @brief Insert a candidate into the list if it's better than the worst.
     * 
     * @param id Node ID.
     * @param dist Distance to query.
     * @param limit Maximum number of candidates to keep (can be less than K).
     */
    __device__ void insert(int id, float dist, int limit) {
        // Check if already present
        for (int i = 0; i < size; ++i) {
            if (ids[i] == id) return;
        }

        if (size < limit) {
            ids[size] = id;
            dists[size] = dist;
            size++;
        } else if (dist < dists[size - 1]) {
            ids[size - 1] = id;
            dists[size - 1] = dist;
        } else {
            return;
        }

        // Sort (Bubble sort is efficient for small K)
        for (int i = size - 1; i > 0; --i) {
            if (dists[i] < dists[i - 1]) {
                float td = dists[i]; dists[i] = dists[i - 1]; dists[i - 1] = td;
                int ti = ids[i]; ids[i] = ids[i - 1]; ids[i - 1] = ti;
            } else {
                break;
            }
        }
    }
    
    /**
     * @brief Overload for insert using K as the limit.
     */
    __device__ void insert(int id, float dist) {
        insert(id, dist, K);
    }
    
    /**
     * @brief Check if the list contains a specific ID.
     */
    __device__ bool contains(int id) {
        for(int i=0; i<size; ++i) if(ids[i] == id) return true;
        return false;
    }
};
