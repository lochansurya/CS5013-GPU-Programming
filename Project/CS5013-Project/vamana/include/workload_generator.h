// ============================================================================
// WorkloadGenerator — dynamic workload generator for Vamana ANN graph
// ============================================================================

#pragma once
#include <random>
#include <vector>
#include <unordered_set>
#include <iostream>
#include <algorithm>

template <typename T>
class WorkloadGenerator {
public:
    struct Query {
        enum Type { SEARCH = 0, INSERT = 1, DELETE = 2 } type;

        std::vector<T> vec;      // search or insert vector
        std::vector<T> del_vec;  // delete target vector (exact match)
    };

    WorkloadGenerator(uint D,
                      float insert_ratio = 0.3f,
                      float delete_ratio = 0.1f,
                      float search_ratio = 0.6f)
        : D(D), insR(insert_ratio), delR(delete_ratio), searchR(search_ratio),
          rng(std::random_device{}());
    // -------------------------------------
    // Generate a batch of queries
    // -------------------------------------
    std::vector<Query> generateBatch(size_t batch_size) ;
    // -------------------------------------
    // Utility: convert batch to device arrays for Vamana
    // -------------------------------------
    size_t packSearchQueriesToDevice(const std::vector<Query>& Q,
                                     T** d_queries_out,
                                     cudaStream_t stream = 0);
    size_t packInsertQueriesToDevice(const std::vector<Query>& Q,
                                     T** d_ins_out);

    size_t packDeleteQueriesToDevice(const std::vector<Query>& Q,
                                     T** d_del_out);
private:
    uint D;

    float insR, delR, searchR;
    std::mt19937 rng;

    // active inserted vectors (used to generate valid deletes)
    std::vector<std::vector<T>> activeInserts;

    // ---------------------------------------------------------------------
    // Constructors for query types
    // ---------------------------------------------------------------------
    Query makeSearch() ;
    Query makeInsert() ;

    Query makeDelete() ;
    // Random vector generator
    std::vector<T> randomVector() ;
};


