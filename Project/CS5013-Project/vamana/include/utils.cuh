#pragma once
#include "constants.cuh"
#include "globals.cuh"
#include "graphT.cuh"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include <math_constants.h>

// uncomment to log the function calls
// #define logfuncs

template <typename Func>
auto callWithLog(const char*   func_name,
                 const char*   file,
                 int           line,
                 const char*   caller,
                 Func&&        func,
                 std::ostream& os = std::cout) {
    constexpr const char* cyan  = "\033[36m";
    constexpr const char* reset = "\033[0m";

    os << cyan << "[" << file << ":" << line << ":" << caller << "]" << reset << " Calling " << cyan
       << func_name << reset << '\n';
    os.flush();

    if constexpr (std::is_void_v<decltype(func())>)
        func();
    else
        return func();
}

#if defined(logfuncs)
#define CALL_WITH_LOG(fn_call, ...) \
    callWithLog(#fn_call, __FILE__, __LINE__, __func__, [&]() { return fn_call; }, ##__VA_ARGS__)
#else
#define CALL_WITH_LOG(fn_call, ...) fn_call
#endif

template <typename... Args>
void logHostError(const char*        file,
                  int                line,
                  const char*        caller,
                  std::ostream&      os,
                  const std::string& msg,
                  Args&&... args) {
    constexpr const char* red   = "\033[31m";
    constexpr const char* reset = "\033[0m";

    os << red << "[ERROR] [" << file << ":" << line << ":" << caller << "] " << reset;

    std::ostringstream oss;
    oss << msg;
    ((oss << ' ' << std::forward<Args>(args)), ...);

    os << oss.str() << '\n';
    os.flush();
}

#define LOG_HOST_ERROR(stream, msg, ...) \
    logHostError(__FILE__, __LINE__, __func__, stream, msg, ##__VA_ARGS__)

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}
#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

//==================================================================================================

[[nodiscard]] inline bool readBin(const std::filesystem::path& bin_path,
                                  uint8_t*                     buffer,
                                  size_t                       number_of_entries,
                                  size_t                       entry_size) {
    FILE* bin_file = fopen(bin_path.c_str(), "rb");
    if (!bin_file) {
        LOG_HOST_ERROR(std::cerr, "Failed to open bin file ", bin_path.c_str());
        // printf("Could not open bin file.\n");
        return false;
    }

    // constexpr size_t                      IO_BUFFER_SIZE = 4 * 1024 * 1024;
    // static thread_local std::vector<char> io_buffer(IO_BUFFER_SIZE);
    // setvbuf(bin_file, io_buffer.data(), _IOFBF, IO_BUFFER_SIZE);

    const size_t read_count = fread(buffer, entry_size, number_of_entries, bin_file);
    fclose(bin_file);

    if (read_count != number_of_entries) {
        LOG_HOST_ERROR(std::cerr, "Incomplete read:", read_count, "of", number_of_entries);
        return false;
    }
    return true;
}

using namespace FreshVamana::Consts;

/*
The returned graph should have d_graph correctly populated
*/
// don't mark noexcept
template <typename T__>
[[nodiscard]] inline std::unique_ptr<GraphT<T__>> initGraph(
    const std::filesystem::path& graph_bin_path) {
    using GraphType = GraphT<T__>;

    auto graph = std::make_unique<GraphType>();

    FreshVamana::Globals::d_graph_capacity_g = rg_bin_size_g * 1.3;
    FreshVamana::Globals::d_graph_size_g     = rg_bin_size_g;

    uint8_t* h_graph = (uint8_t*)std::calloc(FreshVamana::Globals::d_graph_capacity_g,
                                             FreshVamana::Consts::graph_entry_bytes_g);
    if (!readBin(graph_bin_path,
                 h_graph,
                 FreshVamana::Globals::d_graph_size_g,
                 FreshVamana::Consts::graph_entry_bytes_g)) {
        return nullptr;
    }

    gpuErrchk(cudaMalloc(
        &graph->d_graph,
        FreshVamana::Globals::d_graph_capacity_g * FreshVamana::Consts::graph_entry_bytes_g));

    gpuErrchk(cudaMemcpy(graph->d_graph,
                         h_graph,
                         rg_bin_size_g * FreshVamana::Consts::graph_entry_bytes_g,
                         cudaMemcpyHostToDevice));

    free(h_graph);
    std::cout << "[ initGraph ] Graph initialized: "
              << "N=" << FreshVamana::Globals::d_graph_size_g << ", D=" << FreshVamana::Consts::D_g
              << ", R=" << FreshVamana::Consts::R_g
              << ", EntrySize=" << FreshVamana::Consts::graph_entry_bytes_g << " bytes.\n";

    return graph;
}

template <typename T__>
__device__ inline float l2_distance_sq(const T__* vec1, const T__* vec2, const size_t vecDim) {
    float dist = 0.0f;
    for (size_t i = 0; i < vecDim; ++i) {
        float diff = static_cast<float>(vec1[i]) - static_cast<float>(vec2[i]);
        dist += diff * diff;
    }
    return dist;
}

/**
 * @brief Kernel 1: Extracts vectors from the graph using node IDs from a worklist.
 *
 * Each block processes one vector.
 * Grid: (num_queries * L_g)
 * Block: (vecDim)
 */
template <typename T__>
__global__ void extract_vectors_kernel(const uint8_t* d_graph,
                                       const uint*    d_worklist,
                                       T__*           d_worklist_vectors,
                                       const size_t   L_g,
                                       const size_t   vecDim,
                                       const size_t   entrySize,
                                       const size_t   num_queries) {
    const size_t global_worklist_idx = blockIdx.x;
    const size_t dim_idx             = threadIdx.x;

    if (global_worklist_idx >= num_queries * L_g || dim_idx >= vecDim) {
        return;
    }

    const uint node_id = d_worklist[global_worklist_idx];

    const T__* src_vec_start = (const T__*)(d_graph + node_id * entrySize);

    T__* dest_vec_start = d_worklist_vectors + global_worklist_idx * vecDim;

    dest_vec_start[dim_idx] = src_vec_start[dim_idx];
}

/**
 * @brief Kernel 2: Merges worklist vectors and insert_list vectors, finding the
 * top L_g for each query. (Refactored to use global/local memory)
 *
 * Each block processes one query.
 * Grid: (num_queries)
 * Block: (e.g., 256) - Note: All work is done by t_idx=0.
 */
template <typename T__>
__global__ void merge_and_rerank_kernel(const T__*   d_queryVecs,
                                        const T__*   d_worklist_vectors,
                                        const T__*   d_insert_list_g,
                                        T__*         d_final_top_vectors,
                                        const size_t L_g,
                                        const size_t D_g,
                                        const size_t insert_list_size,
                                        const size_t num_queries) {
    const size_t q_idx = blockIdx.x;
    const size_t t_idx = threadIdx.x;

    if (q_idx >= num_queries) {
        return;
    }

    if (t_idx == 0) {
        T__* g_top_vectors = d_final_top_vectors + q_idx * L_g * D_g;

        float l_top_dists[FreshVamana::Consts::L_g];
        for (size_t k = 0; k < L_g; ++k) {
            l_top_dists[k] = CUDART_INF_F;
        }

        for (size_t c_idx = 0; c_idx < L_g; ++c_idx) {
            const T__* cand_vec_ptr = d_worklist_vectors + (q_idx * L_g + c_idx) * D_g;
            float      dist         = l2_distance_sq(d_queryVecs + q_idx * D_g, cand_vec_ptr, D_g);

            float  max_dist = -1.0f;
            size_t max_k    = 0;
            for (size_t k = 0; k < L_g; ++k) {
                if (l_top_dists[k] > max_dist) {
                    max_dist = l_top_dists[k];
                    max_k    = k;
                }
            }

            if (dist < max_dist) {
                l_top_dists[max_k] = dist;
                T__* g_dest        = g_top_vectors + max_k * D_g;
                for (size_t d = 0; d < D_g; ++d) {
                    g_dest[d] = cand_vec_ptr[d];
                }
            }
        }

        for (size_t c_idx = 0; c_idx < insert_list_size; ++c_idx) {
            const T__* cand_vec_ptr = d_insert_list_g + c_idx * D_g;
            float      dist         = l2_distance_sq(d_queryVecs + q_idx * D_g, cand_vec_ptr, D_g);

            float  max_dist = -1.0f;
            size_t max_k    = 0;
            for (size_t k = 0; k < L_g; ++k) {
                if (l_top_dists[k] > max_dist) {
                    max_dist = l_top_dists[k];
                    max_k    = k;
                }
            }

            if (dist < max_dist) {
                l_top_dists[max_k] = dist;
                T__* g_dest        = g_top_vectors + max_k * D_g;
                for (size_t d = 0; d < D_g; ++d) {
                    g_dest[d] = cand_vec_ptr[d];
                }
            }
        }

        for (size_t i = 0; i < L_g - 1; ++i) {
            for (size_t j = 0; j < L_g - i - 1; ++j) {
                if (l_top_dists[j] > l_top_dists[j + 1]) {
                    float temp_dist    = l_top_dists[j];
                    l_top_dists[j]     = l_top_dists[j + 1];
                    l_top_dists[j + 1] = temp_dist;

                    T__* vec_j  = g_top_vectors + j * D_g;
                    T__* vec_j1 = g_top_vectors + (j + 1) * D_g;
                    for (size_t d = 0; d < D_g; ++d) {
                        T__ temp_val = vec_j[d];
                        vec_j[d]     = vec_j1[d];
                        vec_j1[d]    = temp_val;
                    }
                }
            }
        }
    }
}