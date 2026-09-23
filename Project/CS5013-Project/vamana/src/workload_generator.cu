// ============================================================================
// WorkloadGenerator — dynamic workload generator for Vamana ANN graph
// ============================================================================
#include <random>
#include <vector>
#include <unordered_set>
#include <iostream>
#include <algorithm>
#include "workload_generator.h"

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
          rng(std::random_device{}())
    {
        assert(fabs(insert_ratio + delete_ratio + search_ratio - 1.0f) < 1e-6);
    }

    // -------------------------------------
    // Generate a batch of queries
    // -------------------------------------
    std::vector<Query> generateBatch(size_t batch_size) {
        std::vector<Query> out;
        out.reserve(batch_size);

        std::uniform_real_distribution<float> U(0.0f, 1.0f);

        for (size_t i = 0; i < batch_size; i++) {
            float r = U(rng);

            if (r < searchR) {
                out.push_back(makeSearch());
            } else if (r < searchR + insR) {
                auto q = makeInsert();
                activeInserts.push_back(q.vec);    // add to alive set
                out.push_back(q);
            } else {
                if (!activeInserts.empty()) {
                    auto q = makeDelete();
                    out.push_back(q);
                } else {
                    out.push_back(makeSearch()); // fallback
                }
            }
        }

        return out;
    }

    // -------------------------------------
    // Utility: convert batch to device arrays for Vamana
    // -------------------------------------
    size_t packSearchQueriesToDevice(const std::vector<Query>& Q,
                                     T** d_queries_out,
                                     cudaStream_t stream = 0)
    {
        std::vector<T> flat;
        for (auto& q : Q) {
            if (q.type == Query::SEARCH)
                flat.insert(flat.end(), q.vec.begin(), q.vec.end());
        }
        if (flat.empty()) {
            *d_queries_out = nullptr;
            return 0;
        }

        size_t num = flat.size() / D;
        size_t bytes = flat.size() * sizeof(T);

        cudaMalloc(d_queries_out, bytes);
        cudaMemcpyAsync(*d_queries_out, flat.data(), bytes,
                        cudaMemcpyHostToDevice, stream);
        return num;
    }

    size_t packInsertQueriesToDevice(const std::vector<Query>& Q,
                                     T** d_ins_out)
    {
        std::vector<T> flat;
        for (auto& q : Q) {
            if (q.type == Query::INSERT)
                flat.insert(flat.end(), q.vec.begin(), q.vec.end());
        }
        if (flat.empty()) {
            *d_ins_out = nullptr;
            return 0;
        }

        size_t num = flat.size() / D;
        size_t bytes = flat.size() * sizeof(T);

        cudaMalloc(d_ins_out, bytes);
        cudaMemcpy(*d_ins_out, flat.data(), bytes, cudaMemcpyHostToDevice);
        return num;
    }

    size_t packDeleteQueriesToDevice(const std::vector<Query>& Q,
                                     T** d_del_out)
    {
        std::vector<T> flat;
        for (auto& q : Q) {
            if (q.type == Query::DELETE)
                flat.insert(flat.end(), q.del_vec.begin(), q.del_vec.end());
        }
        if (flat.empty()) {
            *d_del_out = nullptr;
            return 0;
        }

        size_t num = flat.size() / D;
        size_t bytes = flat.size() * sizeof(T);

        cudaMalloc(d_del_out, bytes);
        cudaMemcpy(*d_del_out, flat.data(), bytes, cudaMemcpyHostToDevice);

        return num;
    }

private:
    uint D;

    float insR, delR, searchR;
    std::mt19937 rng;

    // active inserted vectors (used to generate valid deletes)
    std::vector<std::vector<T>> activeInserts;

    // ---------------------------------------------------------------------
    // Constructors for query types
    // ---------------------------------------------------------------------
    Query makeSearch() {
        return Query{ Query::SEARCH, randomVector(), {} };
    }

    Query makeInsert() {
        return Query{ Query::INSERT, randomVector(), {} };
    }

    Query makeDelete() {
        std::uniform_int_distribution<size_t> U(0, activeInserts.size() - 1);
        size_t idx = U(rng);

        Query q;
        q.type = Query::DELETE;
        q.del_vec = activeInserts[idx];

        // remove from active-insert set
        activeInserts[idx] = activeInserts.back();
        activeInserts.pop_back();

        return q;
    }

    // Random vector generator
    std::vector<T> randomVector() {
        std::vector<T> v(D);
        std::normal_distribution<float> N(0.0f, 1.0f);
        for (uint i = 0; i < D; i++)
            v[i] = (T)N(rng);
        return v;
    }
};

