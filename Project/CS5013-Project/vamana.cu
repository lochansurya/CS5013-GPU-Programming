// vamana_fixed.cu
// Features:
// - Implements Recall@10 (1-NN in Top-10) calculation via external 'calculate_recall' utility.
// - Preserves GPU vector search (searchPoints2 returns float*) and maps to IDs on Host.
// - Integrates external ground truth computation via 'compute_groundtruth'.
// - Includes saveGraph and patchGraph functionality.
// - FIX: Reordered all Kernels to top of file to fix "identifier undefined" errors.
// - FIX: Added missing flags (--data_type, --dist_fn) to compute_groundtruth command.
// - FIX: Solved Race Condition in mergeIntoWorklist kernel.
// - FIX: WorkloadGenerator updates baseVectors_ on insert to prevent 0% recall.
// - FIX: Fixed file IO crash by removing redundant temp file read/write in loadBaseVectorsFromFvecs.
// - FIX: Corrected variable names in pruneOutNeighbors/pruneReverseEdges (__shared__ scoping).
// - FIX: Added all requested kernels from user snippets.
// - FIX: Cleanup of intermediate bin files after recall calculation.

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>
#include <set> 
#include <chrono>
#include <typeinfo>
#include <unistd.h> // for sync()
#include <omp.h>    // For parallelizing if needed

// CUDA and std headers used across files
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std; 

// ------------------------------- Utils --------------------------------

using uint = unsigned int;

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit((int)code);
    }
}
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

struct CPUTimer {
    std::chrono::time_point<std::chrono::high_resolution_clock> start_time;
    void Start() { start_time = std::chrono::high_resolution_clock::now(); }
    void Stop() {}
    double Elapsed() const {
        auto end_time = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double>(end_time - start_time).count();
    }
};

struct GPUTimer {
    GPUTimer(cudaStream_t, bool) {}
    void   Start() {}
    void   Stop() {}
    double Elapsed() const { return 0.0; }
};

// ------------------------------- FVecs Utility -------------------------------

struct FVecsHeader {
    int dim;  
    FVecsHeader() : dim(128) {}              
    explicit FVecsHeader(int d) : dim(d) {} 
};

struct FVecsPoint {
    std::vector<float> values;  
};

struct FVecs {
    FVecsHeader             header;
    std::vector<FVecsPoint> points;

    FVecs() = default;
    FVecs(const FVecsHeader& h, const std::vector<FVecsPoint>& p);
    ~FVecs() = default;

    bool   load_from_fvecs_file(const std::string& file_path);
    bool   load_first_n_from_fvecs_file_fast(size_t n, const std::string& file_path);
    bool   load_from_fvecs_file_fast(const std::string& file_path);
    void   show_first_n_fvecs(unsigned int num_vectors = 1) const;
    bool   load_from_binary(const std::string& file_path);
    bool   save_to_binary(const std::string& output_file_path) const;
    bool   is_empty() const;
    size_t size() const;
    void   clear();
};

// FVecs Implementation
FVecs::FVecs(const FVecsHeader& h, const std::vector<FVecsPoint>& p) : header(h), points(p) {}

bool FVecs::load_from_fvecs_file(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) return false;
    while (file.peek() != EOF) {
        unsigned int dim = 0;
        file.read(reinterpret_cast<char*>(&dim), sizeof(unsigned int));
        if (file.eof()) break;
        FVecsPoint point;
        point.values.resize(dim);
        file.read(reinterpret_cast<char*>(point.values.data()), dim * sizeof(float));
        if (file.eof()) break;
        points.push_back(std::move(point));
    }
    if (!points.empty()) header.dim = static_cast<int>(points.front().values.size());
    return true;
}

bool FVecs::load_from_fvecs_file_fast(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) return false;
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    int dim = 0;
    if (!file.read(reinterpret_cast<char*>(&dim), sizeof(int))) return false;
    this->header.dim = dim;
    if (this->header.dim <= 0) return false;
    const size_t file_record_size = 1 * sizeof(int) + this->header.dim * sizeof(float);
    if (file_size < file_record_size) return false;
    size_t total_num_records = file_size / file_record_size;
    std::vector<char> buffer(file_size);
    file.seekg(0, std::ios::beg);
    if (!file.read(buffer.data(), file_size)) return false;
    points.clear();
    points.reserve(total_num_records);
    const char* buf_ptr = buffer.data();
    for (size_t i = 0; i < total_num_records; ++i) {
        int d = *(reinterpret_cast<const int*>(buf_ptr));
        if (d != dim) return false;
        buf_ptr += sizeof(int);
        FVecsPoint point;
        point.values.resize(dim);
        memcpy(point.values.data(), buf_ptr, dim * sizeof(float));
        buf_ptr += dim * sizeof(float); 
        points.push_back(std::move(point));
    }
    return true;
}

bool FVecs::load_first_n_from_fvecs_file_fast(size_t n, const std::string& file_path) {
    return load_from_fvecs_file_fast(file_path); 
}

bool FVecs::save_to_binary(const std::string& output_file_path) const {
    std::ofstream file(output_file_path, std::ios::binary);
    if (!file.is_open()) return false;
    unsigned int num_points = static_cast<unsigned int>(this->size());
    unsigned int dim        = static_cast<unsigned int>(header.dim);
    file.write(reinterpret_cast<const char*>(&num_points), sizeof(unsigned int));
    file.write(reinterpret_cast<const char*>(&dim), sizeof(unsigned int));
    for (const auto& point : points) {
        file.write(reinterpret_cast<const char*>(point.values.data()), dim * sizeof(float));
    }
    file.close(); 
    return true;
}

bool FVecs::load_from_binary(const std::string& file_path) {
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) return false;
    unsigned int num_points_ui = 0;
    unsigned int dim_ui        = 0;
    if (!file.read(reinterpret_cast<char*>(&num_points_ui), sizeof(unsigned int))) return false;
    if (!file.read(reinterpret_cast<char*>(&dim_ui), sizeof(unsigned int))) return false;
    size_t num_points = num_points_ui;
    unsigned int dim = dim_ui;
    if (!file || num_points == 0 || dim == 0) return false;
    points.clear();
    points.reserve(num_points);
    header.dim = dim;
    for (unsigned int i = 0; i < num_points; ++i) {
        FVecsPoint point;
        point.values.resize(dim);
        file.read(reinterpret_cast<char*>(point.values.data()), dim * sizeof(float));
        if (!file) return false;
        points.push_back(point);
    }
    return true;
}

bool FVecs::is_empty() const { return points.empty(); }
size_t FVecs::size() const { return points.size(); }
void FVecs::clear() { points.clear(); header.dim = 0; }
void FVecs::show_first_n_fvecs(unsigned int num_vectors) const {
    std::cout << "Dim: " << header.dim << " N: " << points.size() << '\n';
}

// ------------------------------- Constants ------------------------------

namespace FreshVamana {
namespace Consts {
constexpr uint D_g           = 128;
constexpr uint R_g           = 64;
constexpr uint rg_bin_size_g = 10000;
constexpr uint L_g           = 199;
constexpr uint medoid_g                  = 5000;
constexpr uint max_num_parents_per_query = 600;
using dtype_g = float;
constexpr uint graph_entry_bytes_g = D_g * sizeof(dtype_g) + sizeof(uint) + R_g * sizeof(uint);
constexpr uint max_reverse_index_entries_g = 500;
constexpr uint reverse_index_entry_words_g = (max_reverse_index_entries_g + 1);
constexpr uint reverse_index_entry_bytes_g = reverse_index_entry_words_g * sizeof(uint);
enum class queryType : std::int8_t { undefined_q = -1, insert_q = 0, delete_q = 1, search_q = 2 };
}  
} 

namespace FreshVamana {
namespace DeviceConsts {
__constant__ const uint D_g                         = Consts::D_g;
__constant__ const uint R_g                         = Consts::R_g;
__constant__ const uint L_g                         = Consts::L_g;
__constant__ const uint max_num_parents_per_query   = Consts::max_num_parents_per_query;
__constant__ const uint graph_entry_bytes_g         = Consts::graph_entry_bytes_g;
__constant__ const uint max_reverse_index_entries_g = Consts::max_reverse_index_entries_g;
__constant__ const uint reverse_index_entry_words_g = Consts::reverse_index_entry_words_g;
} 
} 

// ------------------------------- Globals -----------------------
namespace FreshVamana {
namespace Globals {
__device__ __managed__ uint d_graph_capacity_g = 0;
__device__ __managed__ uint d_graph_size_g     = 0;
uint* d_delete_list_g = nullptr;
FreshVamana::Consts::dtype_g* d_insert_list_g = nullptr;
uint8_t* d_bloom_bits_g = nullptr;
}  
} 

// ------------------------------- Bloom Filter --------------------------
using namespace FreshVamana::Consts;
#define BF_ENTRIES 39988U
constexpr uint BF_MEMORY = (BF_ENTRIES & 0xFFFFFFFC) + sizeof(uint);
__device__ uint bf_hashFn1(uint x) {
    uint64_t hash = 0xcbf29ce4ULL;
    hash = (hash ^ (x & 0xff)) * 0x01000193ULL;
    hash = (hash ^ ((x >> 8) & 0xff)) * 0x01000193ULL;
    hash = (hash ^ ((x >> 16) & 0xff)) * 0x01000193ULL;
    hash = (hash ^ ((x >> 24) & 0xff)) * 0x01000193ULL;
    return (uint)(hash % BF_ENTRIES);
}
__device__ uint bf_hashFn2(uint x) {
    uint64_t hash = 0x84222325ULL;
    hash = (hash ^ (x & 0xff)) * 0x1B3ULL;
    hash = (hash ^ ((x >> 8) & 0xff)) * 0x1B3ULL;
    hash = (hash ^ ((x >> 16) & 0xff)) * 0x1B3ULL;
    hash = (hash ^ ((x >> 24) & 0xff)) * 0x1B3ULL;
    return (uint)(hash % BF_ENTRIES);
}
__device__ bool bf_check(bool* bf, uint x) { return bf[bf_hashFn1(x)] && bf[bf_hashFn2(x)]; }
__device__ void bf_set(bool* bf, uint x) { bf[bf_hashFn1(x)] = true; bf[bf_hashFn2(x)] = true; }

// ------------------------------- Delete List --------------------------
__global__ void checkIfNodeDeletedKernel(const uint* d_delete_list, uint size, uint node_id, bool* d_found) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        if (d_delete_list[tid] == node_id) *d_found = true;
    }
}

class DeleteList {
   public:
    DeleteList(uint initial_capacity = 5000u) {
        growth_factor_ = 1.3f; capacity_ = std::max(1u, initial_capacity); size_ = 0;
        gpuErrchk(cudaMalloc(&FreshVamana::Globals::d_delete_list_g, capacity_ * sizeof(uint)));
    }
    ~DeleteList() { cudaFree(FreshVamana::Globals::d_delete_list_g); FreshVamana::Globals::d_delete_list_g = nullptr; }
    uint size() const noexcept { return size_; }
    uint capacity() const noexcept { return capacity_; }
    void addNode(uint node_id) {
        if (size_ >= capacity_) expandDeleteList();
        gpuErrchk(cudaMemcpy(FreshVamana::Globals::d_delete_list_g + size_, &node_id, sizeof(uint), cudaMemcpyHostToDevice));
        ++size_;
    }
    void addNodes(uint* d_node_ids, size_t num_nodes) {
        if (num_nodes == 0) return;
        if (size_ + num_nodes >= capacity_) expandDeleteList(num_nodes);
        gpuErrchk(cudaMemcpy(FreshVamana::Globals::d_delete_list_g + size_, d_node_ids, num_nodes * sizeof(uint), cudaMemcpyDeviceToDevice));
        size_ = static_cast<uint>(size_ + num_nodes);
    }
    uint* data() noexcept { return FreshVamana::Globals::d_delete_list_g; }
    const uint* data() const noexcept { return FreshVamana::Globals::d_delete_list_g; }
    void clear() noexcept { capacity_ = 0; size_ = 0; }
   private:
    void expandDeleteList(size_t extra = 1) {
        const uint new_capacity = static_cast<uint>(std::max<uint>(1u, static_cast<uint>(capacity_ * growth_factor_ + extra)));
        uint* new_buffer = nullptr;
        gpuErrchk(cudaMalloc(&new_buffer, new_capacity * sizeof(uint)));
        if (FreshVamana::Globals::d_delete_list_g && capacity_ > 0) {
            gpuErrchk(cudaMemcpy(new_buffer, FreshVamana::Globals::d_delete_list_g, capacity_ * sizeof(uint), cudaMemcpyDeviceToDevice));
            cudaFree(FreshVamana::Globals::d_delete_list_g);
        }
        FreshVamana::Globals::d_delete_list_g = new_buffer;
        capacity_ = new_capacity;
    }
    float growth_factor_ = 1.3f; uint capacity_ = 0; uint size_ = 0;
};

__host__ inline bool isNodeInDeleteList(uint* d_delete_list, size_t size, uint node_id) {
    if (size == 0) return false;
    bool  h_found = false; bool* d_found = nullptr;
    cudaMalloc(&d_found, sizeof(bool));
    cudaMemcpy(d_found, &h_found, sizeof(bool), cudaMemcpyHostToDevice);
    checkIfNodeDeletedKernel<<<1, 1>>>(d_delete_list, (uint)size, node_id, d_found);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_found);
    return h_found;
}
__device__ inline bool isNodeInDeleteList_dev(const uint* d_delete_list, size_t size, uint node_id) {
    for (size_t i = 0; i < size; ++i) if (d_delete_list[i] == node_id) return true;
    return false;
}

// ------------------------------- Insert List --------------------------
constexpr uint insert_entry_bytes_g = FreshVamana::Consts::D_g * sizeof(FreshVamana::Consts::dtype_g);

class InsertList {
   public:
    InsertList(uint initial_capacity = 5000u) {
        growth_factor_ = 1.3f; capacity_ = std::max(1u, initial_capacity); size_ = 0;
        gpuErrchk(cudaMalloc(&FreshVamana::Globals::d_insert_list_g, capacity_ * insert_entry_bytes_g));
    }
    ~InsertList() { cudaFree(FreshVamana::Globals::d_insert_list_g); FreshVamana::Globals::d_insert_list_g = nullptr; }
    uint size() const noexcept { return size_; }
    uint capacity() const noexcept { return capacity_; }
    void addVectors(FreshVamana::Consts::dtype_g* d_vectors, size_t num_vectors) {
        if (num_vectors == 0) return;
        if (size_ + num_vectors >= capacity_) expandInsertList(num_vectors);
        FreshVamana::Consts::dtype_g* dst = FreshVamana::Globals::d_insert_list_g + static_cast<size_t>(size_) * FreshVamana::Consts::D_g;
        gpuErrchk(cudaMemcpy(dst, d_vectors, num_vectors * insert_entry_bytes_g, cudaMemcpyDeviceToDevice));
        size_ = static_cast<uint>(size_ + num_vectors);
    }
    FreshVamana::Consts::dtype_g* data() noexcept { return FreshVamana::Globals::d_insert_list_g; }
    const FreshVamana::Consts::dtype_g* data() const noexcept { return FreshVamana::Globals::d_insert_list_g; }
    void clear() noexcept { capacity_ = 0; size_ = 0; }
   private:
    void expandInsertList(size_t extra = 1) {
        const uint new_capacity = static_cast<uint>(std::max<uint>(1u, static_cast<uint>(capacity_ * growth_factor_ + extra)));
        FreshVamana::Consts::dtype_g* new_buffer = nullptr;
        gpuErrchk(cudaMalloc(&new_buffer, static_cast<size_t>(new_capacity) * insert_entry_bytes_g));
        if (FreshVamana::Globals::d_insert_list_g && capacity_ > 0) {
            gpuErrchk(cudaMemcpy(new_buffer, FreshVamana::Globals::d_insert_list_g, size_ * insert_entry_bytes_g, cudaMemcpyDeviceToDevice));
            cudaFree(FreshVamana::Globals::d_insert_list_g);
        }
        FreshVamana::Globals::d_insert_list_g = new_buffer;
        capacity_ = new_capacity;
    }
    float growth_factor_ = 1.3f; uint capacity_ = 0; uint size_ = 0;
};

// ------------------------------- Graph Init Util -----------------------
template <typename T__>
[[nodiscard]] inline std::unique_ptr<uint8_t[]> readBinToHost(const std::filesystem::path& graph_bin_path, size_t number_of_entries, size_t entry_size) {
    FILE* bin_file = fopen(graph_bin_path.c_str(), "rb");
    if (!bin_file) { fprintf(stderr, "Failed to open bin file %s\n", graph_bin_path.c_str()); return nullptr; }
    const size_t read_count = number_of_entries;
    std::unique_ptr<uint8_t[]> buffer(new uint8_t[number_of_entries * entry_size]);
    size_t r = fread(buffer.get(), entry_size, read_count, bin_file);
    fclose(bin_file);
    if (r != read_count) { fprintf(stderr, "Incomplete read: %zu of %zu\n", r, read_count); return nullptr; }
    return buffer;
}

struct GraphT { uint8_t* d_graph = nullptr; uint h_graph_capacity = 0; uint h_graph_size = 0; };

template <typename T__>
[[nodiscard]] inline std::unique_ptr<GraphT> initGraph(const std::filesystem::path& graph_bin_path) {
    using namespace FreshVamana::Consts;
    auto graph = std::make_unique<GraphT>();
    FreshVamana::Globals::d_graph_capacity_g = rg_bin_size_g * 13 / 10;
    FreshVamana::Globals::d_graph_size_g     = rg_bin_size_g;
    size_t alloc_entries = FreshVamana::Globals::d_graph_capacity_g;
    size_t entry_bytes   = graph_entry_bytes_g;
    uint8_t* h_graph = (uint8_t*)std::calloc(FreshVamana::Globals::d_graph_capacity_g, entry_bytes);
    if (!h_graph) { fprintf(stderr, "calloc failed\n"); return nullptr; }
    std::unique_ptr<uint8_t[]> readbuf = readBinToHost<T__>(graph_bin_path, FreshVamana::Globals::d_graph_size_g, entry_bytes);
    if (!readbuf) { free(h_graph); return nullptr; }
    memcpy(h_graph, readbuf.get(), (size_t)FreshVamana::Globals::d_graph_size_g * entry_bytes);
    gpuErrchk(cudaMalloc(&graph->d_graph, alloc_entries * entry_bytes));
    gpuErrchk(cudaMemset(graph->d_graph, 0, alloc_entries * entry_bytes));
    gpuErrchk(cudaMemcpy(graph->d_graph, h_graph, FreshVamana::Globals::d_graph_size_g * entry_bytes, cudaMemcpyHostToDevice));
    free(h_graph);
    graph->h_graph_capacity = static_cast<int>(FreshVamana::Globals::d_graph_capacity_g);
    graph->h_graph_size     = static_cast<int>(FreshVamana::Globals::d_graph_size_g);
    std::cout << "[ initGraph ] Graph initialized: N=" << FreshVamana::Globals::d_graph_size_g << ", D=" << D_g << ", R=" << R_g << "\n";
    return graph;
}

// ------------------------------- All Device Kernels (Defined BEFORE usage) ------------------

// 1. computeDists
template <typename T__>
__global__ void computeDists(uint8_t* d_graph, uint* d_nodes, uint* d_node_count, T__* d_query_vecs, T__* d_dists, uint row_size) {
    const uint D_g                 = FreshVamana::Consts::D_g;
    const uint graph_entry_bytes_g = FreshVamana::Consts::graph_entry_bytes_g;
    uint query_id = blockIdx.x;
    uint tid      = threadIdx.x;
    const T__* query_vec = d_query_vecs + D_g * query_id;
    uint       offset    = row_size * query_id;
    uint       num_nodes = 0;
    if (d_node_count) num_nodes = d_node_count[query_id]; else return;
    for (uint i = tid; i < num_nodes; i += blockDim.x) d_dists[offset + i] = 0;
    __syncthreads();
    for (uint j = tid / 8; j < num_nodes; j += (blockDim.x + 7) / 8) {
        uint node = d_nodes[offset + j];
        if (node >= FreshVamana::Globals::d_graph_size_g) continue;
        const T__* node_vec = reinterpret_cast<const T__*>(d_graph + node * graph_entry_bytes_g);
        T__ sum = 0;
        for (uint i = tid % 8; i < D_g; i += 8) {
            T__ diff = node_vec[i] - query_vec[i];
            sum += diff * diff;
        }
        atomicAdd(&d_dists[offset + j], sum);
    }
}

// 2. Helper Math
template <typename T__>
__device__ uint lowerBound(T__ arr[], uint lo, uint hi, T__ target) {
    while (lo < hi) { uint mid = (lo + hi) / 2; if (target > arr[mid]) lo = mid + 1; else hi = mid; }
    return lo;
}

template <typename T__>
__device__ uint upperBound(T__ arr[], uint lo, uint hi, T__ target) {
    while (lo < hi) { uint mid = (lo + hi) / 2; if (target >= arr[mid]) lo = mid + 1; else hi = mid; }
    return lo;
}

// 3. sortByDistance
template <typename T__>
__global__ void sortByDistance(uint* d_items, uint* d_item_count, T__* d_dists, uint* d_items_aux, T__* d_dists_aux, uint row_size) {
    uint query_id = blockIdx.x;
    uint tid      = threadIdx.x;
    uint numItems = d_item_count[query_id];
    uint offset   = query_id * row_size;
    extern __shared__ uint shared_pos[]; 
    if (numItems == 0) return;
    for (uint subarray_size = 2; subarray_size < 2 * numItems; subarray_size *= 2) {
        uint subarray_id = tid / subarray_size;
        uint start       = subarray_id * subarray_size;
        uint mid = (start + subarray_size / 2 <= numItems) ? start + subarray_size / 2 : numItems;
        uint end = (start + subarray_size <= numItems) ? start + subarray_size : numItems;
        uint before = 0;
        if (tid >= start && tid < mid) {
            if (mid < end) { before = lowerBound<T__>(&d_dists[offset + mid], 0, end - mid, d_dists[offset + tid]); shared_pos[tid - start] = tid + before; }
            else { shared_pos[tid - start] = tid; }
        } else if (tid >= mid && tid < end) {
            before = upperBound<T__>(&d_dists[offset + start], 0, mid - start, d_dists[offset + tid]); shared_pos[tid - start] = before + (tid - mid + start);
        }
        __syncthreads(); __threadfence_block();
        for (uint i = tid; i < numItems; i += blockDim.x) { uint pos = shared_pos[i]; if (pos < row_size) { d_items_aux[offset + pos] = d_items[offset + i]; d_dists_aux[offset + pos] = d_dists[offset + i]; } }
        __syncthreads(); __threadfence_block();
        for (uint i = tid; i < numItems; i += blockDim.x) { d_items[offset + i] = d_items_aux[offset + i]; d_dists[offset + i] = d_dists_aux[offset + i]; }
        __syncthreads();
    }
}

// 4. getNeighbors
template <typename T__>
__global__ void getNeighbors(uint8_t* d_graph, uint batch_start, uint* d_neighbors, uint* d_neighbors_count) {
    const uint D_g                 = FreshVamana::Consts::D_g;
    const uint R_g                 = FreshVamana::Consts::R_g;
    const uint graph_entry_bytes_g = FreshVamana::Consts::graph_entry_bytes_g;
    uint queryID     = blockIdx.x;
    uint extendedQueryID = batch_start + queryID;
    uint tid             = threadIdx.x;
    uint* degreePtr   = reinterpret_cast<uint*>(d_graph + extendedQueryID * graph_entry_bytes_g + D_g * sizeof(T__));
    uint* neighborPtr = degreePtr + 1;
    if (extendedQueryID >= FreshVamana::Globals::d_graph_size_g) return;
    uint degree = *degreePtr;
    if (degree > R_g) degree = R_g; 
    if (tid == 0) d_neighbors_count[queryID] = degree;
    __syncthreads();
    for (uint ii = tid; ii < degree; ii += blockDim.x) d_neighbors[(R_g + 1) * queryID + ii] = neighborPtr[ii];
}

// 5. mergeIntoVisitedSet
template <typename T__>
__global__ void mergeIntoVisitedSet(uint* d_visited_set_count, uint* d_visited_set, T__* d_visited_set_dists, uint* d_neighbors_count, uint* d_neighbors, T__* d_neighbors_dist) {
    const uint max_num_parents_per_query = FreshVamana::Consts::max_num_parents_per_query;
    const uint R_g                       = FreshVamana::Consts::R_g;
    uint query_id = blockIdx.x;
    uint tid      = threadIdx.x;
    uint visited_set_offset = query_id * max_num_parents_per_query;
    uint neighbors_offset   = query_id * (R_g + 1);
    uint num_neighbors    = d_neighbors_count[query_id];
    uint visited_set_size = d_visited_set_count[query_id];
    uint new_visited_set_size = min(num_neighbors + visited_set_size, max_num_parents_per_query);
    uint id      = UINT_MAX;
    T__  dist    = static_cast<T__>(0);
    uint new_pos = max_num_parents_per_query;
    if (tid < visited_set_size) {
        uint before = lowerBound<T__>(&d_neighbors_dist[neighbors_offset], 0, num_neighbors, d_visited_set_dists[visited_set_offset + tid]);
        id          = d_visited_set[visited_set_offset + tid];
        dist        = d_visited_set_dists[visited_set_offset + tid];
        new_pos     = before + tid;
    } else if (tid >= max_num_parents_per_query && tid < max_num_parents_per_query + num_neighbors) {
        uint idx    = tid - max_num_parents_per_query;
        uint before = upperBound<T__>(&d_visited_set_dists[visited_set_offset], 0, visited_set_size, d_neighbors_dist[neighbors_offset + idx]);
        id          = d_neighbors[neighbors_offset + idx];
        dist        = d_neighbors_dist[neighbors_offset + idx];
        new_pos     = before + idx;
    }
    __syncthreads();
    if (new_pos < new_visited_set_size && new_pos < max_num_parents_per_query) {
        d_visited_set[visited_set_offset + new_pos]       = id;
        d_visited_set_dists[visited_set_offset + new_pos] = dist;
    }
    __syncthreads();
    if (tid == 0) d_visited_set_count[query_id] = new_visited_set_size;
}

typedef enum : uint8_t { NODE_INIT = 0, NODE_PRUNED = 1, NODE_NEIGHBOR = 2 } NodeStateLocal;

// 6. pruneOutNeighbors
// Moved __shared__ declaration to top to fix scoping issues.
// Standardized variable name to pStarShared.
template <typename T__>
__global__ void pruneOutNeighbors(uint8_t* d_graph, uint batch_start, uint* d_visited_set, uint* d_visited_set_count, T__* d_visited_set_dists, NodeStateLocal* d_visited_set_status, T__* d_query_vecs, uint* d_reverse_edge_index, float alpha) {
    const uint D_g                         = FreshVamana::Consts::D_g;
    const uint R_g                         = FreshVamana::Consts::R_g;
    const uint max_num_parents_per_query   = FreshVamana::Consts::max_num_parents_per_query;
    const uint graph_entry_bytes_g         = FreshVamana::Consts::graph_entry_bytes_g;
    const uint reverse_index_entry_words_g = FreshVamana::Consts::reverse_index_entry_words_g;
    const uint max_reverse_index_entries_g = FreshVamana::Consts::max_reverse_index_entries_g;
    
    // Declaration moved to top of kernel
    __shared__ uint pStarShared[1];

    for (uint iter = 1;; ++iter) {
        uint query_id          = blockIdx.x;
        uint extended_query_id = batch_start + query_id;
        uint tid               = threadIdx.x;
        uint num_nodes          = d_visited_set_count[query_id];
        uint visited_set_offset = query_id * max_num_parents_per_query;
        if (extended_query_id >= FreshVamana::Globals::d_graph_size_g) return;
        uint* degree_ptr = reinterpret_cast<uint*>(d_graph + extended_query_id * graph_entry_bytes_g + D_g * sizeof(T__));
        uint* neighborPtr = degree_ptr + 1;
        if (tid == 0 && iter == 1) {
            for (uint i = 0; i < num_nodes && i < max_num_parents_per_query; ++i) d_visited_set_status[visited_set_offset + i] = NODE_INIT;
            atomicExch(degree_ptr, 0u);
        }
        __syncthreads();
        if (*degree_ptr >= R_g) return;
        
        if (tid == 0) pStarShared[0] = UINT_MAX;
        __syncthreads();
        if (tid == 0) {
            for (uint i = 0; i < num_nodes && i < max_num_parents_per_query; ++i) {
                if (d_visited_set_status[visited_set_offset + i] != NODE_INIT) continue;
                pStarShared[0] = d_visited_set[visited_set_offset + i];
                uint old_degree = atomicAdd(degree_ptr, 1u);
                if (old_degree < R_g) {
                    neighborPtr[old_degree] = pStarShared[0];
                } else {
                    atomicSub(degree_ptr, 1u);
                    pStarShared[0] = UINT_MAX;
                    break;
                }
                d_visited_set_status[visited_set_offset + i] = NODE_NEIGHBOR;
                uint node_idx = pStarShared[0];
                if (node_idx < FreshVamana::Globals::d_graph_size_g) {
                    uint* entryPtr = d_reverse_edge_index + node_idx * reverse_index_entry_words_g;
                    uint oldLen = atomicAdd(&entryPtr[0], 1u);
                    if (oldLen < max_reverse_index_entries_g) {
                        entryPtr[1 + oldLen] = extended_query_id;
                    } else {
                        atomicSub(&entryPtr[0], 1u);
                    }
                } else {
                    atomicSub(degree_ptr, 1u);
                    pStarShared[0] = UINT_MAX;
                    break;
                }
            }
        }
        __syncthreads();
        uint pStar = pStarShared[0];
        if (pStar == UINT_MAX) return;
        __shared__ T__ pStarVec[D_g];
        const T__* vecPtr = reinterpret_cast<const T__*>(d_graph + pStar * graph_entry_bytes_g);
        for (uint ii = tid; ii < D_g; ii += blockDim.x) { pStarVec[ii] = vecPtr[ii]; }
        __syncthreads();
        uint laneId        = threadIdx.x & 31;
        uint warpId        = threadIdx.x >> 5;
        uint warpsPerBlock = (blockDim.x + 31) / 32;
        for (uint ii = warpId; ii < num_nodes; ii += warpsPerBlock) {
            if (ii >= max_num_parents_per_query) continue;
            if (d_visited_set_status[visited_set_offset + ii] != NODE_INIT) continue;
            uint p = d_visited_set[visited_set_offset + ii];
            if (p >= FreshVamana::Globals::d_graph_size_g) continue;
            const T__* pVec = reinterpret_cast<const T__*>(d_graph + p * graph_entry_bytes_g);
            float partial = 0.0f;
            for (uint j = laneId; j < D_g; j += 32) {
                float diff = static_cast<float>(pVec[j]) - static_cast<float>(pStarVec[j]);
                partial    = fmaf(diff, diff, partial);
            }
            for (int offset = 16; offset > 0; offset >>= 1) partial += __shfl_down_sync(0xffffffff, partial, offset);
            if (laneId == 0) {
                T__ queryDist = d_visited_set_dists[visited_set_offset + ii];
                if (partial * alpha <= static_cast<float>(queryDist)) {
                    d_visited_set_status[visited_set_offset + ii] = NODE_PRUNED;
                }
            }
        }
    }
}

// 7. findPointsKernel
template <typename T__>
__global__ void findPointsKernel(const uint8_t* __restrict__ d_graph, const T__* __restrict__ d_query_vecs, uint n_nodes, uint n_queries, int* __restrict__ d_results) {
    const uint D_g                 = FreshVamana::Consts::D_g;
    const uint graph_entry_bytes_g = FreshVamana::Consts::graph_entry_bytes_g;
    const uint qid = blockIdx.y;
    const uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_nodes || qid >= n_queries) return;
    const uint8_t* entry_ptr = d_graph + static_cast<size_t>(tid) * graph_entry_bytes_g;
    const T__* vec_ptr   = reinterpret_cast<const T__*>(entry_ptr);
    const T__* query_vec = d_query_vecs + static_cast<size_t>(qid) * D_g;
    bool match = true;
    #pragma unroll
    for (uint i = 0; i < D_g; ++i) {
        if (vec_ptr[i] != query_vec[i]) { match = false; break; }
    }
    if (match) atomicCAS(&d_results[qid], -1, static_cast<int>(tid));
}

// 8. copyNewVectorsToGraph
template <typename T__>
__global__ void copyNewVectorsToGraph(uint8_t* d_graph, const T__* d_new_vecs, uint old_graph_size, uint num_new_nodes) {
    const uint D_g                 = FreshVamana::Consts::D_g;
    const uint R_g                 = FreshVamana::Consts::R_g;
    const uint graph_entry_bytes_g = FreshVamana::Consts::graph_entry_bytes_g;
    uint new_node_idx = blockIdx.x;
    uint tid          = threadIdx.x;
    if (new_node_idx >= num_new_nodes) return;
    uint graph_node_id = old_graph_size + new_node_idx;
    if (graph_node_id >= FreshVamana::Globals::d_graph_capacity_g) return;
    T__* dest_vec_ptr = reinterpret_cast<T__*>(d_graph + graph_node_id * graph_entry_bytes_g);
    const T__* src_vec_ptr  = d_new_vecs + new_node_idx * D_g;
    for (uint i = tid; i < D_g; i += blockDim.x) dest_vec_ptr[i] = src_vec_ptr[i];
    if (tid == 0) {
        uint* degree_ptr = reinterpret_cast<uint*>(reinterpret_cast<char*>(dest_vec_ptr) + D_g * sizeof(T__));
        *degree_ptr = 0u;
        uint* nbptr = degree_ptr + 1;
        for (uint k = 0; k < R_g; ++k) nbptr[k] = UINT_MAX;
    }
}

// 9. loadQueryVecs
template <typename T__>
__global__ void loadQueryVecs(uint8_t* d_graph, T__* d_queryVecs) {
    const uint D_g                 = FreshVamana::Consts::D_g;
    const uint graph_entry_bytes_g = FreshVamana::Consts::graph_entry_bytes_g;
    uint query_id = blockIdx.x;
    uint tid      = threadIdx.x;
    if (query_id >= FreshVamana::Globals::d_graph_size_g) return;
    const T__* query_vec = reinterpret_cast<const T__*>(d_graph + query_id * graph_entry_bytes_g);
    for (uint i = tid; i < D_g; i += blockDim.x) {
        d_queryVecs[query_id * D_g + i] = query_vec[i];
    }
}

// 10. parseReverseIndex
__global__ void parseReverseIndex(uint* d_reverseEdgeIndex, uint* d_reverseEdges, uint* d_reverseEdgeCount, uint* degreeCounts) {
    using namespace FreshVamana::DeviceConsts;
    uint queryID = blockIdx.x;
    uint tid     = threadIdx.x;
    uint* entryPtr = d_reverseEdgeIndex + queryID * FreshVamana::DeviceConsts::reverse_index_entry_words_g;
    uint numReverseEdges = entryPtr[0];
    if (tid == 0) {
        d_reverseEdgeCount[queryID] = numReverseEdges;
        atomicAdd(&degreeCounts[numReverseEdges], 1u);
    }
    for (uint ii = tid; ii < numReverseEdges; ii += blockDim.x) {
        if (ii < FreshVamana::DeviceConsts::max_reverse_index_entries_g)
            d_reverseEdges[queryID * FreshVamana::DeviceConsts::max_reverse_index_entries_g + ii] = entryPtr[1 + ii];
    }
}

// 11. getPrunableQueryIDs
__global__ void getPrunableQueryIDs(uint* d_reverseEdgeCount, uint* d_queryIDs, uint* d_queryCount) {
    int  tid  = threadIdx.x;
    int  lane = tid % 32;
    uint i    = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= FreshVamana::Globals::d_graph_size_g) return;
    uint flag       = (d_reverseEdgeCount[i] != 0);
    uint mask       = __ballot_sync(0xffffffff, flag);
    int  warpActive = __popc(mask);
    uint warpBase = 0;
    if (lane == 0) warpBase = atomicAdd(d_queryCount, warpActive);
    warpBase = __shfl_sync(0xffffffff, warpBase, 0);
    int posInWarp = __popc(mask & ((1u << lane) - 1));
    if (flag) d_queryIDs[warpBase + posInWarp] = i;
}

// 12. mergeIntoReverseEdges
template <typename T__>
__global__ void mergeIntoReverseEdges(uint* d_reverseEdgeCount, uint* d_reverseEdges, T__* d_reverseEdgeDists, uint* d_neighborsCount, uint* d_neighbors, T__* d_neighborsDist) {
    const uint R_g                         = FreshVamana::Consts::R_g;
    const uint max_reverse_index_entries_g = FreshVamana::Consts::max_reverse_index_entries_g;
    uint queryID = blockIdx.x;
    uint tid     = threadIdx.x;
    uint reverseEdgeOffset = queryID * max_reverse_index_entries_g;
    uint neighborsOffset   = queryID * (R_g + 1);
    uint numNeighbors     = d_neighborsCount[queryID];
    uint reverseEdgeCount = d_reverseEdgeCount[queryID];
    uint newReverseEdgeCount = min(numNeighbors + reverseEdgeCount, max_reverse_index_entries_g);
    uint id      = UINT_MAX;
    T__  dist    = static_cast<T__>(0);
    uint newPos = max_reverse_index_entries_g;
    if (tid < reverseEdgeCount) {
        uint before = lowerBound<T__>(&d_neighborsDist[neighborsOffset], 0, numNeighbors, d_reverseEdgeDists[reverseEdgeOffset + tid]);
        id          = d_reverseEdges[reverseEdgeOffset + tid];
        dist        = d_reverseEdgeDists[reverseEdgeOffset + tid];
        newPos      = before + tid;
    } else if (tid >= max_reverse_index_entries_g && tid < max_reverse_index_entries_g + numNeighbors) {
        uint idx    = tid - max_reverse_index_entries_g;
        uint before = upperBound<T__>(&d_reverseEdgeDists[reverseEdgeOffset], 0, reverseEdgeCount, d_neighborsDist[neighborsOffset + idx]);
        id          = d_neighbors[neighborsOffset + idx];
        dist        = d_neighborsDist[neighborsOffset + idx];
        newPos      = before + idx;
    }
    __syncthreads();
    if (newPos < newReverseEdgeCount && newPos < max_reverse_index_entries_g) {
        d_reverseEdges[reverseEdgeOffset + newPos]     = id;
        d_reverseEdgeDists[reverseEdgeOffset + newPos] = dist;
    }
    __syncthreads();
    if (tid == 0) {
        d_reverseEdgeCount[queryID] = newReverseEdgeCount;
    }
}

// 13. pruneReverseEdges
// Moved __shared__ declaration to top to fix scoping issues.
// Standardized variable name to pStarShared.
template <typename T__>
__global__ void pruneReverseEdges(uint8_t* d_graph, uint* d_queryIDs, uint* d_reverseEdges, uint* d_reverseEdgeCount, T__* d_reverseEdgeDists, NodeStateLocal* d_reverseEdgeStatus, T__* d_queryVecs, float alpha) {
    const uint D_g                         = FreshVamana::Consts::D_g;
    const uint R_g                         = FreshVamana::Consts::R_g;
    const uint graph_entry_bytes_g         = FreshVamana::Consts::graph_entry_bytes_g;
    const uint max_reverse_index_entries_g = FreshVamana::Consts::max_reverse_index_entries_g;
    
    // Declaration moved to top of kernel
    __shared__ uint pStarShared[1];

    uint queryID = blockIdx.x;
    uint tid     = threadIdx.x;
    uint numNodes          = d_reverseEdgeCount[queryID];
    uint reverseEdgeOffset = queryID * max_reverse_index_entries_g;
    uint* degreePtr = reinterpret_cast<uint*>(d_graph + queryID * graph_entry_bytes_g + D_g * sizeof(T__));
    uint* neighborPtr = degreePtr + 1;
    if (tid == 0) {
        for (uint i = 0; i < numNodes && i < max_reverse_index_entries_g; ++i) d_reverseEdgeStatus[reverseEdgeOffset + i] = NODE_INIT;
        atomicExch(degreePtr, 0u);
    }
    __syncthreads();
    if (*degreePtr >= R_g) return;
    
    if (tid == 0) pStarShared[0] = UINT_MAX;
    __syncthreads();
    if (tid == 0) {
        for (uint i = 0; i < numNodes && i < max_reverse_index_entries_g; ++i) {
            if (d_reverseEdgeStatus[reverseEdgeOffset + i] != NODE_INIT) continue;
            pStarShared[0] = d_reverseEdges[reverseEdgeOffset + i];
            uint oldDegree = atomicAdd(degreePtr, 1u);
            if (oldDegree < R_g) {
                neighborPtr[oldDegree] = pStarShared[0];
            } else {
                atomicSub(degreePtr, 1u);
                pStarShared[0] = UINT_MAX;
                break;
            }
            d_reverseEdgeStatus[reverseEdgeOffset + i] = NODE_NEIGHBOR;
            break;
        }
    }
    __syncthreads();
    uint pStar = pStarShared[0];
    if (pStar == UINT_MAX) return;
    __shared__ T__ pStarVec[D_g];
    const T__* vecPtr = reinterpret_cast<const T__*>(d_graph + pStar * graph_entry_bytes_g);
    for (uint ii = tid; ii < D_g; ii += blockDim.x) { pStarVec[ii] = vecPtr[ii]; }
    __syncthreads();
    uint laneId        = threadIdx.x & 31;
    uint warpId        = threadIdx.x >> 5;
    uint warpsPerBlock = (blockDim.x + 31) / 32;
    for (uint ii = warpId; ii < numNodes; ii += warpsPerBlock) {
        if (ii >= max_reverse_index_entries_g) continue;
        if (d_reverseEdgeStatus[reverseEdgeOffset + ii] != NODE_INIT) continue;
        uint p = d_reverseEdges[reverseEdgeOffset + ii];
        if (p >= FreshVamana::Globals::d_graph_size_g) continue;
        const T__* pVec = reinterpret_cast<const T__*>(d_graph + p * graph_entry_bytes_g);
        float partial = 0.0f;
        for (uint j = laneId; j < D_g; j += 32) {
            float diff = static_cast<float>(pVec[j]) - static_cast<float>(pStarVec[j]);
            partial    = fmaf(diff, diff, partial);
        }
        for (int offset = 16; offset > 0; offset >>= 1) partial += __shfl_down_sync(0xffffffff, partial, offset);
        if (laneId == 0) {
            T__ queryDist = d_reverseEdgeDists[reverseEdgeOffset + ii];
            if (partial * alpha <= static_cast<float>(queryDist)) {
                d_reverseEdgeStatus[reverseEdgeOffset + ii] = NODE_PRUNED;
            }
        }
    }
}

// 14. initializeParents
__global__ void initializeParents(bool* d_hasParent, uint* d_parents) {
    uint query_id = blockIdx.x;
    if (threadIdx.x == 0) {
        d_hasParent[query_id] = true;
        d_parents[query_id]   = FreshVamana::Consts::medoid_g;
    }
}

// 15. initializeWorklist
template <typename T__>
__global__ void initializeWorklist(uint8_t* d_graph, T__* d_queryVecs, uint* d_worklist, uint* d_worklistCount, T__* d_worklistDist, bool* d_worklistVisited) {
    uint queryID        = blockIdx.x;
    uint tid            = threadIdx.x;
    uint worklistOffset = FreshVamana::Consts::L_g * queryID;
    const T__* queryVec  = d_queryVecs + FreshVamana::Consts::D_g * queryID;
    const T__* medoidVec = reinterpret_cast<const T__*>(d_graph + FreshVamana::Consts::graph_entry_bytes_g * FreshVamana::Consts::medoid_g);
    if (tid == 0) {
        d_worklist[worklistOffset]        = FreshVamana::Consts::medoid_g;
        d_worklistCount[queryID]          = 1;
        d_worklistVisited[worklistOffset] = true;
        T__ dist                          = 0;
        for (uint i = 0; i < FreshVamana::Consts::D_g; ++i) {
            T__ diff = queryVec[i] - medoidVec[i];
            dist += diff * diff;
        }
        d_worklistDist[worklistOffset] = dist;
    }
}

// 16. filterNeighbors
template <typename T__>
__global__ void filterNeighbors(uint8_t* d_graph, const uint* d_delete_list, size_t d_delete_list_size, bool* d_hasParent, uint* d_parents, bool* d_bloomFilters, uint* d_neighbors, uint* d_neighborsCount, uint* d_visitedSet, uint* d_visitedSetCount) {
    uint queryID = blockIdx.x;
    uint tid     = threadIdx.x;
    if (!d_hasParent[queryID]) return;
    bool* bloomFilter = d_bloomFilters + (queryID * BF_MEMORY);
    uint  parent      = d_parents[queryID];
    if (parent >= FreshVamana::Globals::d_graph_size_g) {
        if (tid == 0) d_neighborsCount[queryID] = 0;
        return;
    }
    uint* degreePtr = reinterpret_cast<uint*>(d_graph + parent * FreshVamana::Consts::graph_entry_bytes_g + FreshVamana::Consts::D_g * sizeof(T__));
    uint* neighborPtr = degreePtr + 1;
    __syncthreads();
    if (tid == 0) {
        d_neighborsCount[queryID] = 0;
        d_hasParent[queryID]      = false;
        uint visitedSetIdx        = atomicAdd(&d_visitedSetCount[queryID], 1u);
        if (visitedSetIdx < FreshVamana::Consts::max_num_parents_per_query) {
            d_visitedSet[FreshVamana::Consts::max_num_parents_per_query * queryID + visitedSetIdx] = parent;
        } else {
            atomicSub(&d_visitedSetCount[queryID], 1u);
        }
    }
    __syncthreads();
    uint degree = *degreePtr;
    if (degree > FreshVamana::Consts::R_g) degree = FreshVamana::Consts::R_g;
    for (uint ii = tid; ii < degree; ii += blockDim.x) {
        uint neighbor = neighborPtr[ii];
        if (d_delete_list_size && isNodeInDeleteList_dev(d_delete_list, (uint)d_delete_list_size, neighbor)) continue;
        if (neighbor == queryID) continue;
        if (!bf_check(bloomFilter, neighbor)) {
            bf_set(bloomFilter, neighbor);
            uint neighborIdx = atomicAdd(&d_neighborsCount[queryID], 1u);
            if (neighborIdx < (FreshVamana::Consts::R_g + 1))
                d_neighbors[(FreshVamana::Consts::R_g + 1) * queryID + neighborIdx] = neighbor;
            else
                atomicSub(&d_neighborsCount[queryID], 1u);
        }
    }
}

// 17. mergeIntoWorklist
template <typename T__>
__global__ void mergeIntoWorklist(uint* d_worklistCount, uint* d_worklist, T__* d_worklistDist, bool* d_worklistVisited, uint* d_neighborsCount, uint* d_neighbors, T__* d_neighborsDist, bool* d_hasParent, uint* d_parents, bool* d_nextIter) {
    uint queryID = blockIdx.x;
    uint tid     = threadIdx.x;
    uint neighborsOffset = queryID * (FreshVamana::Consts::R_g + 1);
    uint worklistOffset  = queryID * FreshVamana::Consts::L_g;
    uint numNeighbors    = d_neighborsCount[queryID];
    uint worklistSize    = d_worklistCount[queryID];
    uint newWorklistSize = min(numNeighbors + worklistSize, FreshVamana::DeviceConsts::L_g);
    extern __shared__ uint sortedPositions[]; 
    uint id      = UINT_MAX;
    T__  dist    = static_cast<T__>(0);
    uint newPos = FreshVamana::Consts::L_g;
    if (tid < worklistSize) {
        uint before          = lowerBound<T__>(&d_neighborsDist[neighborsOffset], 0, numNeighbors, d_worklistDist[worklistOffset + tid]);
        id                   = d_worklist[worklistOffset + tid];
        dist                 = d_worklistDist[worklistOffset + tid];
        newPos               = before + tid;
        sortedPositions[tid] = newPos;
    } else if (tid >= FreshVamana::Consts::L_g && tid < FreshVamana::Consts::L_g + numNeighbors) {
        uint idx             = tid - FreshVamana::Consts::L_g;
        uint before          = upperBound<T__>(&d_worklistDist[worklistOffset], 0, worklistSize, d_neighborsDist[neighborsOffset + idx]);
        id                   = d_neighbors[neighborsOffset + idx];
        dist                 = d_neighborsDist[neighborsOffset + idx];
        newPos               = before + idx;
        sortedPositions[tid] = newPos;
    }
    __syncthreads(); __threadfence_block();

    // RACE CONDITION FIX: Read visited status before overwriting
    bool is_visited = false;
    if(tid < worklistSize) is_visited = d_worklistVisited[worklistOffset + tid];
    __syncthreads(); // Ensure all reads complete

    if (newPos < newWorklistSize && newPos < FreshVamana::Consts::L_g) {
        d_worklist[worklistOffset + newPos]        = id;
        d_worklistDist[worklistOffset + newPos]    = dist;
        // If this item came from old worklist (tid < worklistSize), restore its status
        if(tid < worklistSize) d_worklistVisited[worklistOffset + newPos] = is_visited;
        else d_worklistVisited[worklistOffset + newPos] = false; // New items from neighbors
    }
    __syncthreads(); __threadfence_block();

    if (tid == 0) {
        d_worklistCount[queryID] = newWorklistSize;
        for (uint ii = 0; ii < newWorklistSize; ++ii) {
            if (!d_worklistVisited[worklistOffset + ii]) {
                *d_nextIter                        = true;
                d_hasParent[queryID]                   = true;
                d_parents[queryID]                 = d_worklist[worklistOffset + ii];
                d_worklistVisited[worklistOffset + ii] = true;
                break;
            }
        }
    }
}

// 18. overwriteVectorsKernel
template <typename T__>
__global__ void overwriteVectorsKernel(uint8_t* d_graph, const T__* d_new_vecs, uint num_nodes) {
    const uint D_g                 = FreshVamana::Consts::D_g;
    const uint graph_entry_bytes_g = FreshVamana::Consts::graph_entry_bytes_g;
    uint node_idx = blockIdx.x;
    uint tid      = threadIdx.x;
    if (node_idx >= num_nodes) return;
    T__* dest_vec_ptr = reinterpret_cast<T__*>(d_graph + node_idx * graph_entry_bytes_g);
    const T__* src_vec_ptr  = d_new_vecs + node_idx * D_g;
    for (uint i = tid; i < D_g; i += blockDim.x) {
        dest_vec_ptr[i] = src_vec_ptr[i];
    }
}

// 19. extract_vectors_kernel
template <typename T__>
__global__ void extract_vectors_kernel(const uint8_t* d_graph, const uint* d_worklist, T__* d_worklist_vectors, const size_t L_g, const size_t vecDim, const size_t entrySize, const size_t num_queries) {
    const size_t global_worklist_idx = blockIdx.x;
    const size_t dim_idx             = threadIdx.x;
    if (global_worklist_idx >= num_queries * L_g || dim_idx >= vecDim) return;
    const uint node_id = d_worklist[global_worklist_idx];
    if (node_id >= FreshVamana::Globals::d_graph_size_g) return;
    const T__* src_vec_start  = reinterpret_cast<const T__*>(d_graph + node_id * entrySize);
    T__* dest_vec_start = d_worklist_vectors + global_worklist_idx * vecDim;
    dest_vec_start[dim_idx]   = src_vec_start[dim_idx];
}

// 20. merge_and_rerank_kernel
template <typename T__>
__global__ void merge_and_rerank_kernel(const T__* d_queryVecs, const T__* d_worklist_vectors, const T__* d_insert_list_g, T__* d_final_top_vectors, const size_t L_g_param, const size_t D_g_param, const size_t insert_list_size, const size_t num_queries) {
    using namespace FreshVamana::DeviceConsts;
    const size_t q_idx = blockIdx.x;
    const size_t t_idx = threadIdx.x;
    if (q_idx >= num_queries) return;
    if (t_idx == 0) {
        T__* g_top_vectors = d_final_top_vectors + q_idx * FreshVamana::DeviceConsts::L_g * FreshVamana::DeviceConsts::D_g;
        float l_top_dists[FreshVamana::DeviceConsts::L_g];
        for (size_t k = 0; k < FreshVamana::DeviceConsts::L_g; ++k) l_top_dists[k] = 1e30f;
        const size_t L_g_run = L_g_param;
        const size_t D_g_run = D_g_param;
        for (size_t c_idx = 0; c_idx < L_g_run; ++c_idx) {
            const T__* cand_vec_ptr = d_worklist_vectors + (q_idx * L_g_run + c_idx) * D_g_run;
            float      dist         = 0.0f;
            for (size_t d = 0; d < D_g_run; ++d) {
                float diff = static_cast<float>(d_queryVecs[q_idx * D_g_run + d]) - static_cast<float>(cand_vec_ptr[d]);
                dist += diff * diff;
            }
            float  max_dist = -1.0f; size_t max_k = 0;
            for (size_t k = 0; k < L_g_run; ++k) if (l_top_dists[k] > max_dist) { max_dist = l_top_dists[k]; max_k = k; }
            if (dist < max_dist) {
                l_top_dists[max_k] = dist;
                T__* g_dest        = g_top_vectors + max_k * D_g_run;
                for (size_t d = 0; d < D_g_run; ++d) g_dest[d] = cand_vec_ptr[d];
            }
        }
        for (size_t c_idx = 0; c_idx < insert_list_size; ++c_idx) {
            const T__* cand_vec_ptr = d_insert_list_g + c_idx * D_g_run;
            float      dist         = 0.0f;
            for (size_t d = 0; d < D_g_run; ++d) {
                float diff = static_cast<float>(d_queryVecs[q_idx * D_g_run + d]) - static_cast<float>(cand_vec_ptr[d]);
                dist += diff * diff;
            }
            float  max_dist = -1.0f; size_t max_k = 0;
            for (size_t k = 0; k < L_g_run; ++k) if (l_top_dists[k] > max_dist) { max_dist = l_top_dists[k]; max_k = k; }
            if (dist < max_dist) {
                l_top_dists[max_k] = dist;
                T__* g_dest        = g_top_vectors + max_k * D_g_run;
                for (size_t d = 0; d < D_g_run; ++d) g_dest[d] = cand_vec_ptr[d];
            }
        }
        for (size_t i = 0; i < L_g_run - 1; ++i) {
            for (size_t j = 0; j < L_g_run - i - 1; ++j) {
                if (l_top_dists[j] > l_top_dists[j + 1]) {
                    float temp_dist    = l_top_dists[j];
                    l_top_dists[j]     = l_top_dists[j + 1];
                    l_top_dists[j + 1] = temp_dist;
                    T__* vec_j         = g_top_vectors + j * D_g_run;
                    T__* vec_j1        = g_top_vectors + (j + 1) * D_g_run;
                    for (size_t d = 0; d < D_g_run; ++d) {
                        T__ tmp   = vec_j[d];
                        vec_j[d]  = vec_j1[d];
                        vec_j1[d] = tmp;
                    }
                }
            }
        }
    }
}

// 21. checkIfNodeInsertedKernel (Added as requested)
__global__ void checkIfNodeInsertedKernel(const FreshVamana::Consts::dtype_g* d_insert_list,
                                          uint                                insert_size,
                                          const FreshVamana::Consts::dtype_g* d_query_vecs,
                                          uint                                query_size,
                                          unsigned int* d_found) {
    using namespace FreshVamana::Consts;
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= insert_size) return;
    if (atomicAdd(d_found, 0u) != 0u) return;

    const dtype_g* insert_vec = d_insert_list + idx * D_g;

    for (uint q = 0; q < query_size; ++q) {
        const dtype_g* query_vec = d_query_vecs + q * D_g;
        bool same = true;
        for (uint j = 0; j < D_g; ++j) {
            dtype_g a = insert_vec[j];
            dtype_g b = query_vec[j];
            if (fabsf(a - b) > 1e-6f) {
                same = false;
                break;
            }
        }
        if (same) {
            atomicExch(d_found, 1u);
            return;
        }
    }
}

// ------------------------------- Host Helper Functions (Defined BEFORE Vamana) -----------------------

template <typename T__>
[[nodiscard]] uint* greedySearch(uint8_t* d_graph, const uint* d_delete_list, size_t d_delete_list_size, T__* d_queryVecs, uint* d_visitedSet, uint* d_visitedSetCount, uint batchSize) {
    using namespace FreshVamana::Consts;
    bool* d_hasParent;
    uint* d_parents;
    bool* d_bloomFilters;
    gpuErrchk(cudaMalloc(&d_hasParent, batchSize * sizeof(bool)));
    gpuErrchk(cudaMalloc(&d_parents, batchSize * sizeof(uint)));
    size_t allocSize = (size_t)batchSize * BF_MEMORY * sizeof(bool);
    gpuErrchk(cudaMalloc(&d_bloomFilters, allocSize));
    gpuErrchk(cudaMemset(d_bloomFilters, 0, allocSize));

    uint* d_neighbors;
    uint* d_neighborsCount;
    T__* d_neighborDists;
    uint* d_neighborsAux;
    T__* d_neighborDistsAux;

    gpuErrchk(cudaMalloc(&d_neighbors, (size_t)batchSize * (R_g + 1) * sizeof(uint)));
    gpuErrchk(cudaMemset(d_neighbors, 0, (size_t)batchSize * (R_g + 1) * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_neighborsCount, (size_t)batchSize * sizeof(uint)));
    gpuErrchk(cudaMemset(d_neighborsCount, 0, (size_t)batchSize * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_neighborDists, (size_t)batchSize * (R_g + 1) * sizeof(T__)));
    gpuErrchk(cudaMalloc(&d_neighborsAux, (size_t)batchSize * (R_g + 1) * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_neighborDistsAux, (size_t)batchSize * (R_g + 1) * sizeof(T__)));

    uint* d_worklist;
    uint* d_worklistCount;
    T__* d_worklistDist;
    bool* d_worklistVisited;

    gpuErrchk(cudaMalloc(&d_worklist, (size_t)batchSize * L_g * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_worklistCount, (size_t)batchSize * sizeof(uint)));
    gpuErrchk(cudaMemset(d_worklistCount, 0, (size_t)batchSize * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_worklistDist, (size_t)batchSize * L_g * sizeof(T__)));
    gpuErrchk(cudaMalloc(&d_worklistVisited, (size_t)batchSize * L_g * sizeof(bool)));

    bool* d_nextIter;
    gpuErrchk(cudaMalloc(&d_nextIter, sizeof(bool)));

    initializeParents<<<batchSize, 1>>>(d_hasParent, d_parents);
    gpuErrchk(cudaPeekAtLastError());
    initializeWorklist<T__><<<batchSize, 1>>>(
        d_graph, d_queryVecs, d_worklist, d_worklistCount, d_worklistDist, d_worklistVisited);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    int  iter         = 0;
    bool nextIterHost = false;
    do {
        ++iter;
        gpuErrchk(cudaMemset(d_nextIter, 0, sizeof(bool)));
        filterNeighbors<T__><<<batchSize, R_g>>>(d_graph, d_delete_list, d_delete_list_size, d_hasParent, d_parents, d_bloomFilters, d_neighbors, d_neighborsCount, d_visitedSet, d_visitedSetCount);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        computeDists<T__><<<batchSize, R_g * 8>>>(
            d_graph, d_neighbors, d_neighborsCount, d_queryVecs, d_neighborDists, (R_g + 1));
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        sortByDistance<T__><<<batchSize, R_g, R_g * sizeof(uint)>>>(d_neighbors, d_neighborsCount, d_neighborDists, d_neighborsAux, d_neighborDistsAux, R_g + 1);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        mergeIntoWorklist<T__><<<batchSize, R_g + L_g, (R_g + L_g) * sizeof(uint)>>>(d_worklistCount, d_worklist, d_worklistDist, d_worklistVisited, d_neighborsCount, d_neighbors, d_neighborDists, d_hasParent, d_parents, d_nextIter);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaMemcpy(&nextIterHost, d_nextIter, sizeof(bool), cudaMemcpyDeviceToHost));
    } while (nextIterHost);

    gpuErrchk(cudaFree(d_hasParent)); gpuErrchk(cudaFree(d_parents)); gpuErrchk(cudaFree(d_bloomFilters));
    gpuErrchk(cudaFree(d_neighbors)); gpuErrchk(cudaFree(d_neighborsCount)); gpuErrchk(cudaFree(d_neighborDists));
    gpuErrchk(cudaFree(d_neighborsAux)); gpuErrchk(cudaFree(d_neighborDistsAux));
    gpuErrchk(cudaFree(d_worklistCount)); gpuErrchk(cudaFree(d_worklistDist)); gpuErrchk(cudaFree(d_worklistVisited)); gpuErrchk(cudaFree(d_nextIter));
    return d_worklist;
}

template <typename T__>
void computeOutNeighbors(uint8_t* d_graph, T__* d_query_vecs, uint* d_visited_sets, uint* d_visited_set_count, float alpha, uint* d_reverse_edge_index, uint batch_start, uint batch_size) {
    using namespace FreshVamana;
    T__* d_visitedSetDists; uint* d_visitedSetAux; T__* d_visitedSetDistsAux; NodeStateLocal* d_visitedSetStatus;
    gpuErrchk(cudaMalloc(&d_visitedSetDists, (size_t)batch_size * Consts::max_num_parents_per_query * sizeof(T__)));
    gpuErrchk(cudaMalloc(&d_visitedSetAux, (size_t)batch_size * Consts::max_num_parents_per_query * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_visitedSetDistsAux, (size_t)batch_size * Consts::max_num_parents_per_query * sizeof(T__)));
    gpuErrchk(cudaMalloc(&d_visitedSetStatus, (size_t)batch_size * Consts::max_num_parents_per_query * sizeof(NodeStateLocal)));

    uint* d_neighbors; uint* d_neighborsCount; T__* d_neighborsDists; uint* d_neighborsAux; T__* d_neighborsDistsAux;
    gpuErrchk(cudaMalloc(&d_neighbors, (size_t)batch_size * (Consts::R_g + 1) * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_neighborsCount, (size_t)batch_size * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_neighborsDists, (size_t)batch_size * (Consts::R_g + 1) * sizeof(T__)));
    gpuErrchk(cudaMalloc(&d_neighborsAux, (size_t)batch_size * (Consts::R_g + 1) * sizeof(uint)));
    gpuErrchk(
        cudaMalloc(&d_neighborsDistsAux, (size_t)batch_size * (Consts::R_g + 1) * sizeof(T__)));

    getNeighbors<T__><<<batch_size, Consts::R_g>>>(d_graph, batch_start, d_neighbors, d_neighborsCount);
    gpuErrchk(cudaPeekAtLastError());
    computeDists<T__><<<batch_size, Consts::R_g * 8>>>(d_graph, d_neighbors, d_neighborsCount, d_query_vecs, d_neighborsDists, (Consts::R_g + 1));
    gpuErrchk(cudaPeekAtLastError());
    sortByDistance<T__><<<batch_size, Consts::R_g + 1, (Consts::R_g + 1) * sizeof(uint)>>>(d_neighbors, d_neighborsCount, d_neighborsDists, d_neighborsAux, d_neighborsDistsAux, Consts::R_g + 1);
    gpuErrchk(cudaPeekAtLastError());
    computeDists<T__><<<batch_size, 1024>>>(d_graph, d_visited_sets, d_visited_set_count, d_query_vecs, d_visitedSetDists, Consts::max_num_parents_per_query);
    gpuErrchk(cudaPeekAtLastError());
    sortByDistance<T__><<<batch_size, Consts::max_num_parents_per_query, Consts::max_num_parents_per_query * sizeof(uint)>>>(d_visited_sets, d_visited_set_count, d_visitedSetDists, d_visitedSetAux, d_visitedSetDistsAux, Consts::max_num_parents_per_query);
    gpuErrchk(cudaPeekAtLastError());
    mergeIntoVisitedSet<T__><<<batch_size, Consts::max_num_parents_per_query + Consts::R_g>>>(d_visited_set_count, d_visited_sets, d_visitedSetDists, d_neighborsCount, d_neighbors, d_neighborsDists);
    gpuErrchk(cudaPeekAtLastError());
    pruneOutNeighbors<T__><<<batch_size, 32>>>(d_graph, batch_start, d_visited_sets, d_visited_set_count, d_visitedSetDists, d_visitedSetStatus, d_query_vecs, d_reverse_edge_index, alpha);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaFree(d_visitedSetDists)); gpuErrchk(cudaFree(d_visitedSetAux)); gpuErrchk(cudaFree(d_visitedSetDistsAux)); gpuErrchk(cudaFree(d_visitedSetStatus));
    gpuErrchk(cudaFree(d_neighbors)); gpuErrchk(cudaFree(d_neighborsCount)); gpuErrchk(cudaFree(d_neighborsDists)); gpuErrchk(cudaFree(d_neighborsAux)); gpuErrchk(cudaFree(d_neighborsDistsAux));
}

template <typename T__>
void computeReverseEdges(uint8_t* d_graph, uint* d_reverseEdgeIndex, float alpha) {
    using namespace FreshVamana;
    const uint R_g = FreshVamana::Consts::R_g;
    const uint D_g = FreshVamana::Consts::D_g;
    const uint max_reverse_index_entries_g = FreshVamana::Consts::max_reverse_index_entries_g;
    const uint reverse_index_entry_words_g = FreshVamana::Consts::reverse_index_entry_words_g;
    T__* d_queryVecs;
    gpuErrchk(cudaMalloc(&d_queryVecs, FreshVamana::Globals::d_graph_size_g * D_g * sizeof(T__)));
    uint* d_reverseEdges; uint* d_reverseEdgeCount; T__* d_reverseEdgeDists; uint* d_reverseEdgesAux; T__* d_reverseEdgeDistsAux; NodeStateLocal* d_reverseEdgeStatus;
    gpuErrchk(cudaMalloc(&d_reverseEdges, FreshVamana::Globals::d_graph_size_g * max_reverse_index_entries_g * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeCount, FreshVamana::Globals::d_graph_size_g * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeDists, FreshVamana::Globals::d_graph_size_g * max_reverse_index_entries_g * sizeof(T__)));
    gpuErrchk(cudaMalloc(&d_reverseEdgesAux, FreshVamana::Globals::d_graph_size_g * max_reverse_index_entries_g * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeDistsAux, FreshVamana::Globals::d_graph_size_g * max_reverse_index_entries_g * sizeof(T__)));
    gpuErrchk(cudaMalloc(&d_reverseEdgeStatus, FreshVamana::Globals::d_graph_size_g * max_reverse_index_entries_g * sizeof(NodeStateLocal)));
    uint* d_neighbors; uint* d_neighborsCount; T__* d_neighborDists; uint* d_neighborsAux; T__* d_neighborDistsAux;
    gpuErrchk(cudaMalloc(&d_neighbors, FreshVamana::Globals::d_graph_size_g * (R_g + 1) * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_neighborsCount, FreshVamana::Globals::d_graph_size_g * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_neighborDists, FreshVamana::Globals::d_graph_size_g * (R_g + 1) * sizeof(T__)));
    gpuErrchk(cudaMalloc(&d_neighborsAux, FreshVamana::Globals::d_graph_size_g * (R_g + 1) * sizeof(uint)));
    gpuErrchk(cudaMalloc(&d_neighborDistsAux, FreshVamana::Globals::d_graph_size_g * (R_g + 1) * sizeof(T__)));

    loadQueryVecs<T__><<<FreshVamana::Globals::d_graph_size_g, D_g>>>(d_graph, d_queryVecs);
    uint* degreeSum; gpuErrchk(cudaMalloc(&degreeSum, (max_reverse_index_entries_g + 1) * sizeof(uint))); gpuErrchk(cudaMemset(degreeSum, 0, (max_reverse_index_entries_g + 1) * sizeof(uint)));
    parseReverseIndex<<<FreshVamana::Globals::d_graph_size_g, 1024>>>(d_reverseEdgeIndex, d_reverseEdges, d_reverseEdgeCount, degreeSum);
    uint* d_queryIDs; uint* d_queryCount; gpuErrchk(cudaMalloc(&d_queryIDs, FreshVamana::Globals::d_graph_size_g * sizeof(uint))); gpuErrchk(cudaMalloc(&d_queryCount, sizeof(uint))); gpuErrchk(cudaMemset(d_queryCount, 0, sizeof(uint)));
    const int numThreads = 32;
    getPrunableQueryIDs<<<(FreshVamana::Globals::d_graph_size_g + numThreads - 1) / numThreads, numThreads>>>(d_reverseEdgeCount, d_queryIDs, d_queryCount);
    computeDists<T__><<<FreshVamana::Globals::d_graph_size_g, 1024>>>(d_graph, d_reverseEdges, d_reverseEdgeCount, d_queryVecs, d_reverseEdgeDists, max_reverse_index_entries_g);
    sortByDistance<T__><<<FreshVamana::Globals::d_graph_size_g, 1024, reverse_index_entry_words_g * sizeof(uint)>>>(d_reverseEdges, d_reverseEdgeCount, d_reverseEdgeDists, d_reverseEdgesAux, d_reverseEdgeDistsAux, max_reverse_index_entries_g);
    getNeighbors<T__><<<FreshVamana::Globals::d_graph_size_g, R_g>>>(d_graph, 0, d_neighbors, d_neighborsCount);
    computeDists<T__><<<FreshVamana::Globals::d_graph_size_g, R_g * 8>>>(d_graph, d_neighbors, d_neighborsCount, d_queryVecs, d_neighborDists, (R_g + 1));
    sortByDistance<T__><<<FreshVamana::Globals::d_graph_size_g, (R_g + 1), (R_g + 1) * sizeof(uint)>>>(d_neighbors, d_neighborsCount, d_neighborDists, d_neighborsAux, d_neighborDistsAux, (R_g + 1));
    mergeIntoReverseEdges<T__><<<FreshVamana::Globals::d_graph_size_g, 1024>>>(d_reverseEdgeCount, d_reverseEdges, d_reverseEdgeDists, d_neighborsCount, d_neighbors, d_neighborDists);
    pruneReverseEdges<T__><<<FreshVamana::Globals::d_graph_size_g, 32>>>(d_graph, d_queryIDs, d_reverseEdges, d_reverseEdgeCount, d_reverseEdgeDists, d_reverseEdgeStatus, d_queryVecs, alpha);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaFree(d_queryVecs)); gpuErrchk(cudaFree(d_reverseEdges)); gpuErrchk(cudaFree(d_reverseEdgeCount)); gpuErrchk(cudaFree(d_reverseEdgeDists));
    gpuErrchk(cudaFree(d_reverseEdgesAux)); gpuErrchk(cudaFree(d_reverseEdgeDistsAux)); gpuErrchk(cudaFree(d_reverseEdgeStatus));
    gpuErrchk(cudaFree(d_neighbors)); gpuErrchk(cudaFree(d_neighborsCount)); gpuErrchk(cudaFree(d_neighborDists));
    gpuErrchk(cudaFree(d_neighborsAux)); gpuErrchk(cudaFree(d_neighborDistsAux)); gpuErrchk(cudaFree(degreeSum)); gpuErrchk(cudaFree(d_queryIDs)); gpuErrchk(cudaFree(d_queryCount));
}

// ------------------------------- Vamana Class -----------------------

template <typename T__>
class Vamana {
   public:
    explicit Vamana(std::unique_ptr<GraphT> graph_arg) {
        graph_ = std::move(graph_arg);
        CPUTimer cputimer; cputimer.Start();
        runVamana();
        cputimer.Stop();
        printf("Vamana<T__>::Vamana: %f sec\n", cputimer.Elapsed());
    }

    void insertPoints(T__* d_query_vecs, size_t num_query_vecs) {
        std::cout << "[ insertPoints ]\n";
        CPUTimer cputimer; cputimer.Start();
        insert_list_.addVectors(d_query_vecs, num_query_vecs);
        cputimer.Stop();
    }

    void deletePoints(T__* d_query_vecs, size_t num_d_query_vecs) {
        CPUTimer cputimer; cputimer.Start();
        int* d_results = findPointsInGraph(graph_->d_graph, FreshVamana::Globals::d_graph_size_g, d_query_vecs, (uint)num_d_query_vecs);
        cputimer.Stop();
        delete_list_.addNodes(reinterpret_cast<uint*>(d_results), num_d_query_vecs);
        cudaFree(d_results);
    }

    [[nodiscard]] uint* searchPoints(T__* d_queryVecs, size_t num) {
        using namespace FreshVamana;
        uint* d_visitedSets;
        uint* d_visitedSetCount;
        uint* d_reverseEdgeIndex_uint;
        
        gpuErrchk(cudaMalloc(&d_visitedSets, Globals::d_graph_size_g * Consts::max_num_parents_per_query * sizeof(uint)));
        gpuErrchk(cudaMalloc(&d_visitedSetCount, Globals::d_graph_size_g * sizeof(uint)));
        gpuErrchk(cudaMalloc(&d_reverseEdgeIndex_uint, Globals::d_graph_size_g * Consts::reverse_index_entry_words_g * sizeof(uint)));
        gpuErrchk(cudaMemset(d_visitedSetCount, 0, Globals::d_graph_size_g * sizeof(uint)));

        uint* d_worklist = greedySearch<T__>(graph_->d_graph, delete_list_.data(), delete_list_.size(), d_queryVecs, d_visitedSets, d_visitedSetCount, (uint)num);

        gpuErrchk(cudaFree(d_visitedSets));
        gpuErrchk(cudaFree(d_visitedSetCount));
        gpuErrchk(cudaFree(d_reverseEdgeIndex_uint));
        return d_worklist;
    }

    [[nodiscard]] T__* searchPoints2(T__* d_queryVecs, size_t num) {
        uint* d_visitedSets;
        uint* d_visitedSetCount;
        uint* d_reverseEdgeIndex_uint;
        std::cout << "[ searchPoints2 ]\n";
        CPUTimer cputimer; cputimer.Start();

        gpuErrchk(cudaMalloc(&d_visitedSets, FreshVamana::Globals::d_graph_size_g * FreshVamana::Consts::max_num_parents_per_query * sizeof(uint)));
        gpuErrchk(cudaMalloc(&d_visitedSetCount, FreshVamana::Globals::d_graph_size_g * sizeof(uint)));
        gpuErrchk(cudaMalloc(&d_reverseEdgeIndex_uint, FreshVamana::Globals::d_graph_size_g * FreshVamana::Consts::reverse_index_entry_words_g * sizeof(uint)));
        gpuErrchk(cudaMemset(d_visitedSetCount, 0, FreshVamana::Globals::d_graph_size_g * sizeof(uint)));
        printf("vamanaInner mallocs: (searchPoints2) done\n");

        uint* d_worklist = greedySearch<T__>(graph_->d_graph, delete_list_.data(), delete_list_.size(), d_queryVecs, d_visitedSets, d_visitedSetCount, (uint)num);
        printf("greedySearch done\n");

        const size_t L_g       = FreshVamana::Consts::L_g;
        const size_t D_g       = FreshVamana::Consts::D_g;
        const size_t entrySize = FreshVamana::Consts::graph_entry_bytes_g;

        T__* d_worklist_vectors;
        gpuErrchk(cudaMalloc(&d_worklist_vectors, num * L_g * D_g * sizeof(T__)));

        dim3 extract_grid((uint)(num * L_g));
        dim3 extract_block((uint)D_g);
        extract_vectors_kernel<<<extract_grid, extract_block>>>(graph_->d_graph, d_worklist, d_worklist_vectors, L_g, D_g, entrySize, num);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        T__* d_final_top_vectors;
        gpuErrchk(cudaMalloc(&d_final_top_vectors, num * L_g * D_g * sizeof(T__)));

        dim3 merge_grid((uint)num);
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
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaFree(d_worklist));
        gpuErrchk(cudaFree(d_worklist_vectors));

        printf("Vamana<T__>::search (total) done\n");
        gpuErrchk(cudaFree(d_visitedSets));
        gpuErrchk(cudaFree(d_visitedSetCount));
        gpuErrchk(cudaFree(d_reverseEdgeIndex_uint));

        return d_final_top_vectors;
    }

    void saveGraph(const std::string& output_path) {
        using namespace FreshVamana;
        const size_t N           = Globals::d_graph_size_g;
        const size_t entry_bytes = Consts::graph_entry_bytes_g;
        const size_t total_bytes = N * entry_bytes;
        if (N == 0) return;
        uint8_t* h_graph = (uint8_t*)malloc(total_bytes);
        if (!h_graph) return;
        gpuErrchk(cudaMemcpy(h_graph, graph_->d_graph, total_bytes, cudaMemcpyDeviceToHost));
        FILE* bin_file = fopen(output_path.c_str(), "wb");
        if (bin_file) {
            fwrite(h_graph, 1, total_bytes, bin_file);
            fclose(bin_file);
            printf("[saveGraph] Saved N=%zu to %s.\n", N, output_path.c_str());
        }
        free(h_graph);
    }

    void patchGraph() {
        using namespace FreshVamana;
        size_t num_new_nodes = insert_list_.size();
        if (num_new_nodes == 0) return;
        std::cout << "[ patchGraph ]\n";
        CPUTimer cputimer_patch; cputimer_patch.Start();
        size_t   old_graph_size = FreshVamana::Globals::d_graph_size_g;
        size_t   new_graph_size = old_graph_size + num_new_nodes;
        uint8_t* d_graph_new;
        size_t   new_graph_bytes = new_graph_size * FreshVamana::Consts::graph_entry_bytes_g;
        gpuErrchk(cudaMalloc(&d_graph_new, new_graph_bytes));

        if (old_graph_size > 0) {
            size_t old_graph_bytes = old_graph_size * FreshVamana::Consts::graph_entry_bytes_g;
            gpuErrchk(cudaMemcpy(d_graph_new, graph_->d_graph, old_graph_bytes, cudaMemcpyDeviceToDevice));
        }

        T__* d_new_vecs = insert_list_.data();
        dim3 grid((uint)num_new_nodes);
        dim3 block(256);
        copyNewVectorsToGraph<T__><<<grid, block>>>(d_graph_new, d_new_vecs, (uint)old_graph_size, (uint)num_new_nodes);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        cudaFree(graph_->d_graph);
        graph_->d_graph         = d_graph_new;
        Globals::d_graph_size_g = static_cast<uint>(new_graph_size);

        uint* d_visitedSets;
        uint* d_visitedSetCount;
        gpuErrchk(cudaMalloc(&d_visitedSets, (size_t)num_new_nodes * Consts::max_num_parents_per_query * sizeof(uint)));
        gpuErrchk(cudaMalloc(&d_visitedSetCount, (size_t)num_new_nodes * sizeof(uint)));
        gpuErrchk(cudaMemset(d_visitedSetCount, 0, (size_t)num_new_nodes * sizeof(uint)));

        uint* d_reverseEdgeIndex;
        gpuErrchk(cudaMalloc(&d_reverseEdgeIndex, (size_t)Globals::d_graph_size_g * Consts::reverse_index_entry_words_g * sizeof(uint)));
        gpuErrchk(cudaMemset(d_reverseEdgeIndex, 0, (size_t)Globals::d_graph_size_g * Consts::reverse_index_entry_words_g * sizeof(uint)));

        uint* d_worklist = greedySearch<T__>(graph_->d_graph, delete_list_.data(), delete_list_.size(), d_new_vecs, d_visitedSets, d_visitedSetCount, (uint)num_new_nodes);
        gpuErrchk(cudaFree(d_worklist));

        computeOutNeighbors<T__>(graph_->d_graph, d_new_vecs, d_visitedSets, d_visitedSetCount, 1.5f, d_reverseEdgeIndex, (uint)old_graph_size, (uint)num_new_nodes);
        computeReverseEdges<T__>(graph_->d_graph, d_reverseEdgeIndex, 1.5f);

        gpuErrchk(cudaFree(d_visitedSets));
        gpuErrchk(cudaFree(d_visitedSetCount));
        gpuErrchk(cudaFree(d_reverseEdgeIndex));
        insert_list_.clear();
        cputimer_patch.Stop();
        printf("Vamana<T__>::patchGraph: %f sec\n", cputimer_patch.Elapsed());
    }

    // Helper to copy host vectors into device graph
    void populateGraphVectors(const std::vector<std::vector<T__>>& base_vecs) {
        using namespace FreshVamana::Consts;
        size_t num_vecs = base_vecs.size();
        size_t limit = std::min(num_vecs, (size_t)FreshVamana::Globals::d_graph_size_g);
        if (limit == 0) return;

        // Flatten host vectors for batch copy
        std::vector<T__> flat_vecs;
        flat_vecs.reserve(limit * D_g);
        for(size_t i=0; i<limit; ++i) {
            flat_vecs.insert(flat_vecs.end(), base_vecs[i].begin(), base_vecs[i].end());
        }

        T__* d_flat_vecs;
        gpuErrchk(cudaMalloc(&d_flat_vecs, flat_vecs.size() * sizeof(T__)));
        gpuErrchk(cudaMemcpy(d_flat_vecs, flat_vecs.data(), flat_vecs.size() * sizeof(T__), cudaMemcpyHostToDevice));

        dim3 block(256);
        dim3 grid((uint)limit);
        
        overwriteVectorsKernel<<<grid, block>>>(graph_->d_graph, d_flat_vecs, (uint)limit);
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());

        gpuErrchk(cudaFree(d_flat_vecs));
        printf("[populateGraphVectors] Overwrote %zu vectors in graph from ground truth source.\n", limit);
    }

   private:
    std::unique_ptr<GraphT> graph_;
    InsertList              insert_list_;
    DeleteList              delete_list_;

    // helper: findPointsInGraph
    [[nodiscard]] int* findPointsInGraph(const uint8_t* d_graph,
                                         uint           n_nodes,
                                         const T__* d_query_vecs,
                                         uint           n_queries) {
        int* d_results = nullptr;
        cudaMalloc(&d_results, n_queries * sizeof(int));
        cudaMemset(d_results, 0xFF, n_queries * sizeof(int));
        dim3 threads(256);
        dim3 blocks((n_nodes + threads.x - 1) / threads.x, n_queries);
        findPointsKernel<T__>
            <<<blocks, threads>>>(d_graph, d_query_vecs, n_nodes, n_queries, d_results);
        cudaDeviceSynchronize();
        return d_results;
    }

    void runVamana() {
        using namespace FreshVamana;
        uint* d_visitedSets;
        uint* d_visitedSetCount;
        uint* d_reverseEdgeIndex;
        std::cout << "[ runVamana ]\n";

        T__* d_queryVecs;
        gpuErrchk(cudaMalloc(&d_queryVecs, Globals::d_graph_size_g * Consts::D_g * sizeof(T__)));
        // Copy existing vectors from graph into d_queryVecs device buffer
        // Perform device-to-device copy of each row (entry) into the contiguous d_queryVecs buffer.
        // Each graph entry starts at graph_->d_graph + i * graph_entry_bytes_g and its vector is
        // the first D_g elements.
        for (uint i = 0; i < Globals::d_graph_size_g; ++i) {
            // source pointer: device memory for the i-th entry vector (first D_g elements)
            const T__* src = reinterpret_cast<const T__*>(
                graph_->d_graph + static_cast<size_t>(i) * Consts::graph_entry_bytes_g);
            // destination pointer: contiguous d_queryVecs row
            T__* dst = d_queryVecs + static_cast<size_t>(i) * Consts::D_g;
            // copy D_g values (device-to-device)
            gpuErrchk(cudaMemcpy(dst, src, Consts::D_g * sizeof(T__), cudaMemcpyDeviceToDevice));
        }

        gpuErrchk(
            cudaMalloc(&d_visitedSets,
                       Globals::d_graph_size_g * Consts::max_num_parents_per_query * sizeof(uint)));
        gpuErrchk(cudaMalloc(&d_visitedSetCount, Globals::d_graph_size_g * sizeof(uint)));
        gpuErrchk(cudaMalloc(
            &d_reverseEdgeIndex,
            Globals::d_graph_size_g * Consts::reverse_index_entry_words_g * sizeof(uint)));
        gpuErrchk(cudaMemset(d_visitedSetCount, 0, Globals::d_graph_size_g * sizeof(uint)));
        gpuErrchk(cudaMemset(
            d_reverseEdgeIndex,
            0,
            Globals::d_graph_size_g * Consts::reverse_index_entry_words_g * sizeof(uint)));

        uint* d_worklist = greedySearch<T__>(graph_->d_graph,
                                             delete_list_.data(),
                                             delete_list_.size(),
                                             d_queryVecs,
                                             d_visitedSets,
                                             d_visitedSetCount,
                                             Globals::d_graph_size_g);
        gpuErrchk(cudaFree(d_worklist));

        computeOutNeighbors<T__>(graph_->d_graph,
                                 d_queryVecs,
                                 d_visitedSets,
                                 d_visitedSetCount,
                                 1.5f,
                                 d_reverseEdgeIndex,
                                 0,
                                 Globals::d_graph_size_g);

        computeReverseEdges<T__>(graph_->d_graph, d_reverseEdgeIndex, 1.5f);

        gpuErrchk(cudaFree(d_visitedSets));
        gpuErrchk(cudaFree(d_visitedSetCount));
        gpuErrchk(cudaFree(d_reverseEdgeIndex));
        gpuErrchk(cudaFree(d_queryVecs));
    }
};

// ============================================================================
// WorkloadGenerator
// ============================================================================

template <typename T>
class WorkloadGenerator {
   public:
    struct Query {
        enum Type : int8_t { SEARCH = 0, INSERT = 1, DELETE = 2 } type;
        std::vector<T> vec;      // search or insert vector
        std::vector<T> del_vec;  // delete target vector (exact match)
    };

    struct GTData {
        std::vector<unsigned> ids;
        std::vector<float> dists;
        unsigned K;
    };

    std::vector<std::vector<T>> baseVectors_;

    WorkloadGenerator(uint  D, float insert_ratio = 0.3f, float delete_ratio = 0.1f, float search_ratio = 0.6f)
        : D(D), insR(insert_ratio), delR(delete_ratio), searchR(search_ratio), rng(std::random_device{}()) {
        assert(std::fabs(insert_ratio + delete_ratio + search_ratio - 1.0f) < 1e-6);
    }

    void loadBaseVectors(const std::string& path) {
        FVecs fvecs_data;
        if (!fvecs_data.load_from_binary(path)) throw std::runtime_error("Failed to load vectors from BIN file: " + path);
        if (fvecs_data.header.dim != D) throw std::runtime_error("Dimension mismatch. Expected " + std::to_string(D));
        baseVectors_.clear();
        baseVectors_.reserve(fvecs_data.size());
        for (const auto& point : fvecs_data.points) baseVectors_.push_back(point.values);
        printf("[WorkloadGenerator] Loaded %zu vectors (Dim: %u) from %s.\n", baseVectors_.size(), D, path.c_str());
    }

    void loadBaseVectorsFromFvecs(const std::string& fvecs_path, const std::string& temp_bin_path) {
        FVecs fvecs_data;
        if (!fvecs_data.load_from_fvecs_file_fast(fvecs_path)) throw std::runtime_error("Failed to load vectors from FVECS file.");
        if (fvecs_data.header.dim != D) throw std::runtime_error("Dimension mismatch.");
        
        // Optimization: Load directly into memory to avoid file IO errors
        baseVectors_.clear();
        baseVectors_.reserve(fvecs_data.size());
        for (const auto& point : fvecs_data.points) {
            baseVectors_.push_back(point.values);
        }
        printf("[WorkloadGenerator] Loaded %zu vectors (Dim: %u) from %s (Direct conversion).\n", baseVectors_.size(), D, fvecs_path.c_str());
        
        // Legacy argument kept for compatibility, but unused
        (void)temp_bin_path; 
    }

    std::vector<Query> generateBatch(size_t batch_size) {
        std::vector<Query> out;
        out.reserve(batch_size);
        std::uniform_real_distribution<float> U(0.0f, 1.0f);
        for (size_t i = 0; i < batch_size; i++) {
            float r = U(rng);
            if (r < searchR) {
                out.push_back(this->makeSearch());
            } else if (r < searchR + insR) {
                auto q = this->makeInsert();
                activeInserts.push_back(q.vec);
                // FIX: Add inserted vector to baseVectors_ so GT calculation is correct
                baseVectors_.push_back(q.vec);
                out.push_back(q);
            } else {
                if (!activeInserts.empty()) {
                    auto q = this->makeDelete();
                    out.push_back(q);
                } else {
                    out.push_back(this->makeSearch());
                }
            }
        }
        return out;
    }

    // Helper to save vectors to binary file (N, D, flat_data)
    void saveVectorsToBinary(const std::vector<std::vector<T>>& vecs, const std::string& path) {
        std::ofstream f(path, std::ios::binary);
        if (!f.is_open()) throw std::runtime_error("Cannot open " + path);
        unsigned int n = (unsigned int)vecs.size();
        unsigned int d = (n > 0) ? (unsigned int)vecs[0].size() : 0;
        f.write((char*)&n, 4);
        f.write((char*)&d, 4);
        for (const auto& v : vecs) {
            f.write((char*)v.data(), d * sizeof(T));
        }
        f.close();
    }

    // Computes GT using external binary and returns the GT filename
    std::string computeGroundTruthExternal(const std::vector<Query>& batch, int iter) {
        // 1. Filter search queries
        std::vector<std::vector<T>> search_queries;
        for (const auto& q : batch) {
            if (q.type == Query::SEARCH) search_queries.push_back(q.vec);
        }
        if (search_queries.empty()) return "";

        // 2. Define unique filenames
        std::string base_file = "base_" + std::to_string(iter) + ".bin";
        std::string query_file = "query_" + std::to_string(iter) + ".bin";
        std::string gt_file = "gt_" + std::to_string(iter) + ".bin";

        // 3. Save files
        saveVectorsToBinary(baseVectors_, base_file);
        saveVectorsToBinary(search_queries, query_file);

        // 4. Call external command
        // Ensure flush
        sync();
        
        std::string cmd = "compute_groundtruth --data_type float --dist_fn l2 --base_file " + base_file + 
                          " --query_file " + query_file + " --gt_file " + gt_file + " --K 100 > /dev/null";
        
        int ret = std::system(cmd.c_str());
        if (ret != 0) {
            std::cerr << "External GT computation failed for iter " << iter << "\n";
            return "";
        }
        
        return gt_file;
    }
    
    // Calls external recall calculator
    void calculateRecallExternal(const std::string& gt_file, const std::string& res_file, int k) {
        std::string cmd = "calculate_recall " + gt_file + " " + res_file + " " + std::to_string(k);
        int ret = std::system(cmd.c_str());
        if (ret != 0) {
             std::cerr << "External recall calculation failed.\n";
        }
    }

    int findID(const std::vector<T>& vec) {
        for(size_t i=0; i<baseVectors_.size(); ++i) {
            bool match = true;
            for(size_t j=0; j<vec.size(); ++j) {
                if(std::abs(baseVectors_[i][j] - vec[j]) > 1e-4) {
                    match = false;
                    break;
                }
            }
            if(match) return (int)i;
        }
        return -1; 
    }

    size_t packSearchQueriesToDevice(const std::vector<Query>& Q, T** d_queries_out, cudaStream_t stream = 0) {
        std::vector<T> flat;
        for (auto& q : Q) if (q.type == Query::SEARCH) flat.insert(flat.end(), q.vec.begin(), q.vec.end());
        if (flat.empty()) { *d_queries_out = nullptr; return 0; }
        size_t bytes = flat.size() * sizeof(T);
        cudaMalloc(d_queries_out, bytes);
        cudaMemcpyAsync(*d_queries_out, flat.data(), bytes, cudaMemcpyHostToDevice, stream);
        return flat.size() / D;
    }

    size_t packInsertQueriesToDevice(const std::vector<Query>& Q, T** d_ins_out) {
        std::vector<T> flat;
        for (auto& q : Q) if (q.type == Query::INSERT) flat.insert(flat.end(), q.vec.begin(), q.vec.end());
        if (flat.empty()) { *d_ins_out = nullptr; return 0; }
        size_t bytes = flat.size() * sizeof(T);
        cudaMalloc(d_ins_out, bytes);
        cudaMemcpy(*d_ins_out, flat.data(), bytes, cudaMemcpyHostToDevice);
        return flat.size() / D;
    }

    size_t packDeleteQueriesToDevice(const std::vector<Query>& Q, T** d_del_out) {
        std::vector<T> flat;
        for (auto& q : Q) if (q.type == Query::DELETE) flat.insert(flat.end(), q.del_vec.begin(), q.del_vec.end());
        if (flat.empty()) { *d_del_out = nullptr; return 0; }
        size_t bytes = flat.size() * sizeof(T);
        cudaMalloc(d_del_out, bytes);
        cudaMemcpy(*d_del_out, flat.data(), bytes, cudaMemcpyHostToDevice);
        return flat.size() / D;
    }

   private:
    uint D;
    float insR, delR, searchR;
    std::mt19937 rng;
    std::vector<std::vector<T>> activeInserts;

    bool saveToBinary(const std::vector<std::vector<T>>& vecs, const std::string& path) {
        std::ofstream file(path, std::ios::binary);
        if (!file.is_open()) return false;
        unsigned int num = static_cast<unsigned int>(vecs.size());
        unsigned int dim = 0;
        if (num > 0) dim = static_cast<unsigned int>(vecs[0].size());
        file.write(reinterpret_cast<const char*>(&num), sizeof(unsigned int));
        file.write(reinterpret_cast<const char*>(&dim), sizeof(unsigned int));
        for (const auto& v : vecs) file.write(reinterpret_cast<const char*>(v.data()), dim * sizeof(T));
        file.close(); // Ensure flush
        return true;
    }

    Query randomVector_from_base() {
        if (baseVectors_.empty()) return {Query::SEARCH, std::vector<T>(D, 0), {}};
        std::uniform_int_distribution<size_t> U(0, baseVectors_.size() - 1);
        return {Query::SEARCH, baseVectors_[U(rng)], {}};
    }
    Query makeSearch() { return randomVector_from_base(); }
    Query makeInsert() { return randomVector_from_base(); }
    Query makeDelete() {
        if (activeInserts.empty()) return {Query::DELETE, {}, {}};
        std::uniform_int_distribution<size_t> U(0, this->activeInserts.size() - 1);
        size_t idx = U(this->rng);
        Query q; q.type = Query::DELETE; q.del_vec = this->activeInserts[idx];
        this->activeInserts[idx] = this->activeInserts.back();
        this->activeInserts.pop_back();
        return q;
    }
};

// ============================================================================
// Main Driver with Vector Overwrite Fix and Valid Recall@10 Utility
// ============================================================================
// Note: calculate_recall utility: <ground_truth_bin> <our_results_bin> <r>
// We write a helper to save the results to binary for the utility.

// Helper to save uint32 results for the utility
// Expected format likely: [num_queries][dim][data...]
bool saveIdsToBinary(const std::vector<unsigned>& ids, size_t n_queries, size_t top_k, const std::string& path) {
    std::ofstream file(path, std::ios::binary);
    if (!file.is_open()) return false;
    int num = (int)n_queries;
    int dim = (int)top_k;
    file.write((char*)&num, sizeof(int));
    file.write((char*)&dim, sizeof(int));
    // Write data. Since ids is vector<unsigned>, cast to char* is fine.
    file.write((char*)ids.data(), ids.size() * sizeof(unsigned));
    return true;
}

// NEW: Helper function to save vectors for GT computation
// This simplifies main() and avoids complex lambdas
template <typename T>
void saveVecsToBinary(const std::vector<std::vector<T>>& vecs, const std::string& path) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) return;
    unsigned int n = (unsigned int)vecs.size();
    unsigned int d = (n > 0) ? (unsigned int)vecs[0].size() : 0;
    f.write((char*)&n, 4);
    f.write((char*)&d, 4);
    for(const auto& v : vecs) {
        f.write((char*)v.data(), d * sizeof(T));
    }
    f.close();
}

int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        printf("Usage: %s <random_graph_path> <basepoints_path> [queries_path]\n", argv[0]);
        return 1;
    }

    std::string random_graph_bin_path = argv[1];
    std::string basepoints_source_path = argv[2];

    std::unique_ptr<GraphT> graph = initGraph<float>(random_graph_bin_path);
    if (!graph) return 1;

    Vamana<float> index(std::move(graph));
    WorkloadGenerator<float> gen(FreshVamana::Consts::D_g, 0.30f, 0.10f, 0.60f);

    try {
        if (basepoints_source_path.size() >= 6 &&
            basepoints_source_path.substr(basepoints_source_path.size() - 6) == ".fvecs") {
            gen.loadBaseVectorsFromFvecs(basepoints_source_path, "temp_init_base.bin");
        } else {
            gen.loadBaseVectors(basepoints_source_path);
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error loading base vectors: %s\n", e.what());
        return 1;
    }

    const int NUM_ITER = 200;    
    const int BATCH    = 2000;  
    const std::string GT_EXEC_PATH = "compute_groundtruth"; 
    const std::string RECALL_EXEC_PATH = "calculate_recall"; // User provided utility name

    for (int iter = 0; iter < NUM_ITER; iter++) {
        auto batch = gen.generateBatch(BATCH);
        
        // On iter 0, fix the graph vectors
        if (iter == 0) {
             index.populateGraphVectors(gen.baseVectors_);
        }

        float *d_search = nullptr, *d_insert = nullptr, *d_delete = nullptr;
        size_t nS = gen.packSearchQueriesToDevice(batch, &d_search);
        size_t nI = gen.packInsertQueriesToDevice(batch, &d_insert);
        size_t nD = gen.packDeleteQueriesToDevice(batch, &d_delete);

        if (nS > 0) {
            // Compute Ground Truth via External Tool
            string gt_file = gen.computeGroundTruthExternal(batch, iter);
            if (gt_file.empty()) {
                cerr << "GT computation failed for iter " << iter << endl;
            } else {
                // Proceed with Vamana Search
                float* d_top_k_vecs = index.searchPoints2(d_search, nS);
                
                const uint dim = FreshVamana::Consts::D_g;
                const uint L_g = FreshVamana::Consts::L_g;
                size_t total_floats = nS * L_g * dim;
                std::vector<float> h_results_vecs(total_floats);
                gpuErrchk(cudaMemcpy(h_results_vecs.data(), d_top_k_vecs, total_floats * sizeof(float), cudaMemcpyDeviceToHost));
                cudaFree(d_top_k_vecs);

                std::vector<unsigned> h_results_ids(nS * L_g);
                
                #pragma omp parallel for
                for(size_t i=0; i<nS; ++i) {
                    for(size_t k=0; k<L_g; ++k) {
                        std::vector<float> vec(dim);
                        size_t base_idx = (i * L_g + k) * dim;
                        for(size_t d=0; d<dim; ++d) vec[d] = h_results_vecs[base_idx + d];
                        int id = gen.findID(vec);
                        h_results_ids[i*L_g + k] = (id != -1) ? (unsigned)id : UINT_MAX;
                    }
                }

                // Save Results to Bin for External Recall Tool
                string res_file = "res_" + to_string(iter) + ".bin";
                saveIdsToBinary(h_results_ids, nS, L_g, res_file);
                
                // Force flush
                sync();

                // Call External Recall Utility
                cout << "[Main] Iter " << iter << ": ";
                fflush(stdout);
                gen.calculateRecallExternal(gt_file, res_file, 10);

                // Cleanup intermediate bin files
                std::remove(gt_file.c_str());
                std::remove(res_file.c_str());
                std::string base_file = "base_" + std::to_string(iter) + ".bin";
                std::string query_file = "query_" + std::to_string(iter) + ".bin";
                std::remove(base_file.c_str());
                std::remove(query_file.c_str());
            }
        }

        if (nI > 0) index.insertPoints(d_insert, nI);
        if (nD > 0) index.deletePoints(d_delete, nD);

        if(d_search) cudaFree(d_search);
        if(d_insert) cudaFree(d_insert);
        if(d_delete) cudaFree(d_delete);

        if (iter % 10 == 0 && iter > 0) {
            printf("\n[Main] Consolidating graph via patchGraph() ...\n");
            index.patchGraph();
        }
    }

    printf("\nDynamic workload finished.\n");
    index.saveGraph("vamana_dynamic_graph.bin");

    return 0;
}