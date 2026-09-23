#include "common.h"
#include "device_funcs.cu"
#include <algorithm>
#include <map>
#include <string>

/**
 * @file engine.cu
 * @brief Core Vamana engine implementation for search, insert, and delete operations.
 */

// -------------------------------------------------------------------------
// Kernels
// -------------------------------------------------------------------------

/**
 * @brief Greedy search device function.
 * 
 * Performs a greedy search on the graph to find the nearest neighbors.
 * 
 * @tparam DIM_SZ Dimension of the vectors.
 * @param ctx Graph context.
 * @param query_vec Query vector.
 * @param start_node Starting node ID.
 * @param best_nodes_out Output array for best node IDs.
 * @param k_results Number of results to return.
 * @param beam_width Beam width for the search.
 */
template <int DIM_SZ>
__device__ void greedy_search(
    const GraphContext ctx,
    const float* query_vec,
    int start_node,
    int* best_nodes_out,
    int k_results,
    int beam_width
) {
    CandidateList<BEAM_WIDTH> candidates;
    candidates.init();
    
    int visited[VISITED_LIST_SIZE]; 
    int visited_count = 0;
    
    float start_dist = distance<DIM_SZ>(query_vec, &ctx.d_vectors[start_node * DIM_SZ]);
    if (ctx.d_delete_mask[start_node]) start_dist = 1e30f;
    
    candidates.insert(start_node, start_dist, beam_width);
    
    int iter = 0;
    while (iter < 200) {
        iter++;
        int next_node = -1;
        
        // Find closest unvisited node
        for (int i = 0; i < candidates.size; ++i) {
            int id = candidates.ids[i];
            bool is_visited = false;
            for (int k=0; k<visited_count; ++k) {
                if (visited[k] == id) { is_visited = true; break; }
            }
            if (!is_visited) {
                next_node = id;
                break;
            }
        }
        
        if (next_node == -1) break;
        
        if (visited_count < VISITED_LIST_SIZE) visited[visited_count++] = next_node;
        else break;
        
        int* neighbors = &ctx.d_adj[next_node * MAX_DEGREE];
        for (int i = 0; i < MAX_DEGREE; ++i) {
            int nbr = neighbors[i];
            if (nbr == -1) break;
            if (ctx.d_delete_mask[nbr]) continue;
            if (candidates.contains(nbr)) continue;
            
            float d = distance<DIM_SZ>(query_vec, &ctx.d_vectors[nbr * DIM_SZ]);
            candidates.insert(nbr, d, beam_width);
        }
    }
    
    for(int i=0; i<k_results; ++i) {
        best_nodes_out[i] = (i < candidates.size) ? candidates.ids[i] : -1;
    }
}

/**
 * @brief Kernel for batch search operations.
 * 
 * @tparam DIM_SZ Dimension of the vectors.
 * @param ctx Graph context.
 * @param d_queries Pointer to query vectors.
 * @param d_results Pointer to output results.
 * @param num_queries Number of queries in the batch.
 * @param start_node Starting node for search.
 * @param beam_width Beam width.
 */
template <int DIM_SZ>
__global__ void searchKernel(
    GraphContext ctx,
    const float* d_queries,
    int* d_results,
    int num_queries,
    int start_node,
    int beam_width
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) return;

    const float* query = &d_queries[tid * DIM_SZ];
    int best_nodes[10];

    greedy_search<DIM_SZ>(ctx, query, start_node, best_nodes, 10, beam_width);
    
    for(int i=0; i<10; ++i) {
        d_results[tid * 10 + i] = best_nodes[i];
    }
}

/**
 * @brief Kernel for batch delete operations.
 * 
 * Marks nodes as deleted and adds them to the freelist.
 * 
 * @param ctx Graph context.
 * @param d_delete_ids Pointer to IDs to delete.
 * @param num_deletes Number of deletions.
 */
__global__ void deleteKernel(
    GraphContext ctx,
    const int* d_delete_ids,
    int num_deletes
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_deletes) return;
    
    int target = d_delete_ids[tid];
    if (target >= 0 && target < ctx.max_capacity) {
        ctx.d_delete_mask[target] = 1; // Mark deleted
        
        // Push to freelist (Atomic)
        int idx = atomicAdd(&ctx.d_counters[1], 1);
        ctx.d_freelist[idx] = target;
    }
}

/**
 * @brief Kernel to prepare for batch insertions.
 * 
 * Assigns IDs (new or reused) and copies vectors to GPU memory.
 * 
 * @tparam DIM_SZ Dimension of the vectors.
 * @param ctx Graph context.
 * @param d_new_vecs Pointer to new vectors.
 * @param d_assigned_ids Output array for assigned IDs.
 * @param num_inserts Number of insertions.
 */
template <int DIM_SZ>
__global__ void insertPrepareKernel(
    GraphContext ctx,
    const float* d_new_vecs,
    int* d_assigned_ids,
    int num_inserts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_inserts) return;

    // Get ID from freelist or increment max_id
    int my_id = -1;
    
    // Try freelist first
    int free_count = ctx.d_counters[1];
    if (free_count > 0) {
        int idx = atomicSub(&ctx.d_counters[1], 1);
        if (idx > 0) { // Valid slot
             my_id = ctx.d_freelist[idx - 1];
             // Reset delete mask
             ctx.d_delete_mask[my_id] = 0;
        }
    }
    
    if (my_id == -1) {
        my_id = atomicAdd(&ctx.d_counters[0], 1);
    }
    
    d_assigned_ids[tid] = my_id;
    
    // Copy Vector
    for (int i = 0; i < DIM_SZ; ++i) {
        ctx.d_vectors[my_id * DIM_SZ + i] = d_new_vecs[tid * DIM_SZ + i];
    }
    
    // Init Adjacency
    for (int i = 0; i < MAX_DEGREE; ++i) {
        ctx.d_adj[my_id * MAX_DEGREE + i] = -1;
    }
}

/**
 * @brief Kernel to link inserted nodes.
 * 
 * Finds a neighbor for the new node and adds a bidirectional link.
 * 
 * @tparam DIM_SZ Dimension of the vectors.
 * @param ctx Graph context.
 * @param d_assigned_ids Array of assigned IDs.
 * @param num_inserts Number of insertions.
 * @param start_node Starting node for search.
 */
template <int DIM_SZ>
__global__ void insertLinkKernel(
    GraphContext ctx,
    const int* d_assigned_ids,
    int num_inserts,
    int start_node
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_inserts) return;

    int my_id = d_assigned_ids[tid];
    if (my_id == -1) return;

    const float* my_vec = &ctx.d_vectors[my_id * DIM_SZ];
    
    int best_node;
    greedy_search<DIM_SZ>(ctx, my_vec, start_node, &best_node, 1, BEAM_WIDTH);
    
    // Link Bidirectionally
    if (best_node != -1) {
        ctx.d_adj[my_id * MAX_DEGREE] = best_node;
        
        int* neighbor_adj = &ctx.d_adj[best_node * MAX_DEGREE];
        for (int i = 0; i < MAX_DEGREE; ++i) {
            if (atomicCAS(&neighbor_adj[i], -1, my_id) == -1) break;
        }
    }
}

/**
 * @brief Brute force search kernel for ground truth verification.
 * 
 * Performs an exhaustive linear scan to find the exact nearest neighbors.
 * 
 * @tparam DIM_SZ Dimension of the vectors.
 * @param ctx Graph context.
 * @param d_queries Pointer to query vectors.
 * @param d_gt_results Pointer to output ground truth results.
 * @param num_queries Number of queries.
 * @param max_id Current maximum ID in the graph.
 */
template <int DIM_SZ>
__global__ void bruteForceKernel(
    GraphContext ctx,
    const float* d_queries,
    int* d_gt_results,
    int num_queries,
    int max_id
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_queries) return;

    const float* query = &d_queries[tid * DIM_SZ];
    
    // Find Top-10
    CandidateList<10> topk;
    topk.init();
    
    for (int i = 0; i < max_id; ++i) {
        if (ctx.d_delete_mask[i]) continue;
        float d = distance<DIM_SZ>(query, &ctx.d_vectors[i * DIM_SZ]);
        topk.insert(i, d);
    }
    
    for(int i=0; i<10; ++i) {
        d_gt_results[tid * 10 + i] = (i < topk.size) ? topk.ids[i] : -1;
    }
}

// -------------------------------------------------------------------------
// Host Engine Class
// -------------------------------------------------------------------------

// Visualization Data
struct QueryInfo {
    int q_id;
    int batch_id; // Added batch_id
    float recall;
    std::vector<float> query_vec;
    std::vector<int> found_ids;
    std::vector<int> gt_ids;
};

/**
 * @brief Host-side Vamana engine.
 * 
 * Manages the graph data structures, GPU memory, and coordinates
 * the execution of search, insert, and delete operations.
 * 
 * @tparam DIM_SZ Dimension of the vectors.
 * @tparam MAX_N Maximum capacity of the graph.
 */
template <int DIM_SZ, int MAX_N>
class VamanaEngine {
    GraphContext ctx;
    int current_max_id;
    
    // Host storage for reset
    std::vector<float> h_init_vectors;
    std::vector<int> h_init_adj;
    int h_init_num_nodes;
    
    // Host workload
    std::vector<Operation<DIM_SZ>> h_workload;
    
public:
    VamanaEngine() {
        ctx.dim = DIM_SZ;
        ctx.max_capacity = MAX_N;
        
        CHECK_CUDA(cudaMalloc(&ctx.d_vectors, MAX_N * DIM_SZ * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&ctx.d_adj, MAX_N * MAX_DEGREE * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&ctx.d_delete_mask, MAX_N * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&ctx.d_counters, 2 * sizeof(int)));
        CHECK_CUDA(cudaMalloc(&ctx.d_freelist, MAX_N * sizeof(int)));
        
        CHECK_CUDA(cudaMemset(ctx.d_delete_mask, 0, MAX_N * sizeof(int)));
        CHECK_CUDA(cudaMemset(ctx.d_counters, 0, 2 * sizeof(int)));
    }
    
    /**
     * @brief Loads graph and workload data from files.
     * 
     * @param graph_file Path to the initial graph file.
     * @param workload_file Path to the workload file.
     */
    void load_data(const char* graph_file, const char* workload_file) {
        // 1. Load Graph
        FILE* f = fopen(graph_file, "rb");
        if (!f) { perror("fopen graph"); exit(1); }
        
        int num_nodes, dim, max_degree;
        check_fread(fread(&num_nodes, sizeof(int), 1, f), 1, "num_nodes");
        check_fread(fread(&dim, sizeof(int), 1, f), 1, "dim");
        check_fread(fread(&max_degree, sizeof(int), 1, f), 1, "max_degree");
        
        if (dim != DIM_SZ) { fprintf(stderr, "Dim mismatch\n"); exit(1); }
        
        h_init_num_nodes = num_nodes;
        h_init_vectors.resize(num_nodes * DIM_SZ);
        h_init_adj.resize(num_nodes * MAX_DEGREE);
        
        check_fread(fread(h_init_vectors.data(), sizeof(float), num_nodes * DIM_SZ, f), num_nodes * DIM_SZ, "vecs");
        check_fread(fread(h_init_adj.data(), sizeof(int), num_nodes * MAX_DEGREE, f), num_nodes * MAX_DEGREE, "adj");
        fclose(f);
        printf("Loaded graph: %d nodes\n", num_nodes);
        
        // 2. Load Workload
        f = fopen(workload_file, "rb");
        if (!f) { perror("fopen workload"); exit(1); }
        
        h_workload.clear();
        while (true) {
            Operation<DIM_SZ> op;
            if (fread(&op.type, sizeof(int), 1, f) != 1) break;
            size_t r1 = fread(&op.id, sizeof(int), 1, f);
            size_t r2 = fread(op.vector, sizeof(float), DIM_SZ, f);
            if (r1 != 1 || r2 != DIM_SZ) break;
            h_workload.push_back(op);
        }
        fclose(f);
        printf("Loaded %zu operations\n", h_workload.size());
    }
    
    /**
     * @brief Resets the GPU state to the initial graph.
     * 
     * Restores vectors, adjacency lists, and counters to their initial state
     * to allow for fair benchmarking of different parameters.
     */
    void reset_gpu_state() {
        CHECK_CUDA(cudaMemcpy(ctx.d_vectors, h_init_vectors.data(), h_init_num_nodes * DIM_SZ * sizeof(float), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(ctx.d_adj, h_init_adj.data(), h_init_num_nodes * MAX_DEGREE * sizeof(int), cudaMemcpyHostToDevice));
        
        current_max_id = h_init_num_nodes;
        CHECK_CUDA(cudaMemcpy(&ctx.d_counters[0], &current_max_id, sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(ctx.d_counters + 1, 0, sizeof(int))); // Reset freelist count
        CHECK_CUDA(cudaMemset(ctx.d_delete_mask, 0, ctx.max_capacity * sizeof(int)));
    }
    
    struct BenchmarkResult {
        float qps;
        float recall;
    };

    /**
     * @brief Processes the loaded workload in batches.
     * 
     * Executes deletes, inserts, and searches on the GPU.
     * 
     * @param batch_size Number of operations per batch.
     * @param output_dir Directory to save visualization/metric outputs (optional).
     * @param beam_width Beam width for the search.
     * @return BenchmarkResult containing QPS and Recall@10.
     */
    BenchmarkResult process_workload(int batch_size, const char* output_dir, int beam_width) {
        // Use h_workload
        const auto& ops = h_workload;
        
        // Metrics
        double total_recall_10 = 0;
        long long total_queries = 0;
        float total_search_time_ms = 0;
        int batch_idx = 1;
        
        // SIFT Visualization Data
        std::vector<QueryInfo> query_stats;
        
        for (size_t i = 0; i < ops.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, ops.size());
            std::vector<Operation<DIM_SZ>> batch(ops.begin() + i, ops.begin() + end);
            
            // Sort: Delete -> Insert -> Search
            std::vector<Operation<DIM_SZ>> deletes, inserts, searches;
            for(auto& op : batch) {
                if (op.type == OP_DELETE) deletes.push_back(op);
                else if (op.type == OP_INSERT) inserts.push_back(op);
                else searches.push_back(op);
            }
            
            // 1. Execute Deletes
            if (!deletes.empty()) {
                std::vector<int> ids;
                for(auto& op : deletes) ids.push_back(op.id);
                int* d_ids;
                CHECK_CUDA(cudaMalloc(&d_ids, ids.size() * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(d_ids, ids.data(), ids.size() * sizeof(int), cudaMemcpyHostToDevice));
                deleteKernel<<<1, 256>>>(ctx, d_ids, ids.size());
                CHECK_CUDA(cudaDeviceSynchronize());
                cudaFree(d_ids);
            }
            
            // 2. Execute Inserts
            if (!inserts.empty()) {
                std::vector<float> vecs;
                for(auto& op : inserts) vecs.insert(vecs.end(), op.vector, op.vector + DIM_SZ);
                float* d_vecs; int* d_ids;
                CHECK_CUDA(cudaMalloc(&d_vecs, vecs.size() * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_ids, inserts.size() * sizeof(int)));
                CHECK_CUDA(cudaMemcpy(d_vecs, vecs.data(), vecs.size() * sizeof(float), cudaMemcpyHostToDevice));
                
                insertPrepareKernel<DIM_SZ><<<1, 256>>>(ctx, d_vecs, d_ids, inserts.size());
                CHECK_CUDA(cudaDeviceSynchronize());
                insertLinkKernel<DIM_SZ><<<1, 256>>>(ctx, d_ids, inserts.size(), 0);
                CHECK_CUDA(cudaDeviceSynchronize());
                
                cudaFree(d_vecs); cudaFree(d_ids);
            }
            
            // 3. Execute Searches
            if (!searches.empty()) {
                std::vector<float> vecs;
                for(auto& op : searches) vecs.insert(vecs.end(), op.vector, op.vector + DIM_SZ);
                float* d_queries; int* d_results;
                CHECK_CUDA(cudaMalloc(&d_queries, vecs.size() * sizeof(float)));
                CHECK_CUDA(cudaMalloc(&d_results, searches.size() * 10 * sizeof(int))); // Store Top-10
                CHECK_CUDA(cudaMemcpy(d_queries, vecs.data(), vecs.size() * sizeof(float), cudaMemcpyHostToDevice));
                
                cudaEvent_t start, stop;
                cudaEventCreate(&start); cudaEventCreate(&stop);
                cudaEventRecord(start);
                
                // searchKernel: Core approximate nearest neighbor (ANN) search kernel.
                // Finds the 10 nearest neighbors for each query vector within the graph.
                // Parameters '0' and 'beam_width' control the search start node and beam width.
                searchKernel<DIM_SZ><<<(searches.size()+255)/256, 256>>>(ctx, d_queries, d_results, searches.size(), 0, beam_width);
                
                cudaEventRecord(stop);
                CHECK_CUDA(cudaDeviceSynchronize());
                float ms; cudaEventElapsedTime(&ms, start, stop);
                total_search_time_ms += ms;
                total_queries += searches.size();
                
                // Verification: Evaluate the accuracy of the 'searchKernel'.
                // Compare approximate results against a brute-force search which guarantees
                // finding the true nearest neighbors.
                int* d_gt;
                CHECK_CUDA(cudaMalloc(&d_gt, searches.size() * 10 * sizeof(int)));
                
                // Update max_id: Retrieve the current maximum ID present in the data structure.
                // Crucial for the brute-force search to know the range of IDs to consider.
                CHECK_CUDA(cudaMemcpy(&current_max_id, &ctx.d_counters[0], sizeof(int), cudaMemcpyDeviceToHost));
                
                // bruteForceKernel: Performs an exhaustive search for the 10 nearest neighbors
                // for each query vector by comparing it against all active vectors in the dataset.
                // This provides the ground truth (gt) nearest neighbors.
                bruteForceKernel<DIM_SZ><<<(searches.size()+255)/256, 256>>>(ctx, d_queries, d_gt, searches.size(), current_max_id);
                CHECK_CUDA(cudaDeviceSynchronize());
                
                std::vector<int> h_res(searches.size() * 10);
                std::vector<int> h_gt(searches.size() * 10);
                CHECK_CUDA(cudaMemcpy(h_res.data(), d_results, searches.size() * 10 * sizeof(int), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_gt.data(), d_gt, searches.size() * 10 * sizeof(int), cudaMemcpyDeviceToHost));
                
                for(int k=0; k<searches.size(); ++k) {
                    std::vector<int> found_list;
                    std::vector<int> gt_list;
                    int matches = 0;
                    
                    for(int j=0; j<10; ++j) {
                        found_list.push_back(h_res[k*10 + j]);
                        gt_list.push_back(h_gt[k*10 + j]);
                    }
                    
                    // Recall@10: Intersection of Found@10 and GT@10
                    for(int f_id : found_list) {
                        if (f_id == -1) continue;
                        for(int g_id : gt_list) {
                            if (f_id == g_id) {
                                matches++;
                                break;
                            }
                        }
                    }
                    
                    float recall = (float)matches / 10.0f;
                    total_recall_10 += recall;
                    
                    // Store stats for Visualization (SIFT and Toy)
                    QueryInfo qi;
                    qi.q_id = i + k;
                    qi.batch_id = batch_idx; // Store current batch index
                    qi.recall = recall;
                    qi.query_vec.assign(searches[k].vector, searches[k].vector + DIM_SZ);
                    qi.found_ids = found_list;
                    qi.gt_ids = gt_list;
                    query_stats.push_back(qi);
                }
                
                cudaFree(d_queries); cudaFree(d_results); cudaFree(d_gt);
            }
            
            // Snapshot for Toy 2D
            if (DIM_SZ == 2) {
                save_snapshot(batch_idx, output_dir);
            }
            
            batch_idx++;
        }
        
        float avg_qps = total_queries / (total_search_time_ms / 1000.0f);
        float avg_recall = total_recall_10 / total_queries;
        
        // Save Metrics (only if output_dir provided)
        if (output_dir) {
            char metrics_path[256];
            sprintf(metrics_path, "%s/metrics.json", output_dir);
            FILE* fm = fopen(metrics_path, "w");
            if (fm) {
                fprintf(fm, "{\"qps\": %.2f, \"recall_10\": %.4f}\n", avg_qps, avg_recall);
                fclose(fm);
            }
            export_query_data(query_stats, output_dir);
        }
        
        return {avg_qps, avg_recall};
    }
    
    /**
     * @brief Runs an automated benchmark for a range of L values.
     * 
     * @param graph_file Path to the graph file.
     * @param workload_file Path to the workload file.
     * @param batch_size Batch size for operations.
     * @param output_dir Output directory.
     */
    void run_automated_benchmark(const char* graph_file, const char* workload_file, int batch_size, const char* output_dir) {
        // Load data once
        load_data(graph_file, workload_file);
        
        printf("Running Automated Benchmark...\n");
        
        std::vector<int> L_values = {10, 20, 32, 64, 100};
        
        printf("\n%-8s %-8s %-16s %-8s\n", "L", "Time", "QPS", "10-r@10");
        printf("%-8s %-8s %-16s %-8s\n", "---", "----", "---", "-------");
        
        for(int L : L_values) {
            // Reset graph for fair comparison
            reset_gpu_state();
            
            // Run workload
            BenchmarkResult res = process_workload(batch_size, output_dir, L);
            
            // Print Row
            // Approximate Time = TotalQueries / QPS
            float time_sec = h_workload.size() / res.qps; 
            
            printf("%-8d %-8.2f %-16.2f %-8.2f\n", L, time_sec, res.qps, res.recall);
        }
    }
    
    void save_snapshot(int step, const char* output_dir) {
        // Update max_id from GPU to ensure we capture all nodes
        CHECK_CUDA(cudaMemcpy(&current_max_id, &ctx.d_counters[0], sizeof(int), cudaMemcpyDeviceToHost));

        // Only for 2D
        char fname[256];
        sprintf(fname, "%s/snapshot_%03d.bin", output_dir, step);
        FILE* f = fopen(fname, "wb");
        if (!f) return;
        
        int num_nodes = current_max_id;
        fwrite(&num_nodes, sizeof(int), 1, f);
        
        std::vector<float> vecs(num_nodes * DIM_SZ);
        std::vector<int> adj(num_nodes * MAX_DEGREE);
        std::vector<int> mask(num_nodes);
        
        CHECK_CUDA(cudaMemcpy(vecs.data(), ctx.d_vectors, num_nodes * DIM_SZ * sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(adj.data(), ctx.d_adj, num_nodes * MAX_DEGREE * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(mask.data(), ctx.d_delete_mask, num_nodes * sizeof(int), cudaMemcpyDeviceToHost));
        
        fwrite(vecs.data(), sizeof(float), num_nodes * DIM_SZ, f);
        fwrite(adj.data(), sizeof(int), num_nodes * MAX_DEGREE, f);
        fwrite(mask.data(), sizeof(int), num_nodes, f);
        fclose(f);
    }
    
    void export_query_data(std::vector<QueryInfo>& stats, const char* output_dir) {
        // Sort by recall
        std::sort(stats.begin(), stats.end(), [](const QueryInfo& a, const QueryInfo& b) {
            return a.recall > b.recall;
        });
        
        if (stats.empty()) return;
        
        char path[256];
        sprintf(path, "%s/query_data.bin", output_dir);
        FILE* f = fopen(path, "wb");
        if (!f) { perror("fopen vis"); return; }
        
        // For Toy (DIM=2), export ALL queries. For SIFT, export Top/Bottom.
        int count = stats.size();
        if (DIM_SZ == 128) {
             count = std::min((size_t)10, stats.size());
        }
        
        fwrite(&count, sizeof(int), 1, f);
        
        if (DIM_SZ == 128) {
            // Write Top 5
            for(int i=0; i<5 && i<stats.size(); ++i) write_query_info(f, stats[i]);
            // Write Bottom 5
            for(int i=0; i<5 && i<stats.size(); ++i) {
                write_query_info(f, stats[stats.size() - 1 - i]);
            }
        } else {
            // Write All
            for(const auto& q : stats) write_query_info(f, q);
        }
        
        fclose(f);
        // printf("Exported visualization data to %s\n", path);
    }
    
    void write_query_info(FILE* f, const QueryInfo& q) {
        fwrite(&q.q_id, sizeof(int), 1, f);
        fwrite(&q.batch_id, sizeof(int), 1, f); // Export batch_id
        fwrite(&q.recall, sizeof(float), 1, f);
        fwrite(q.query_vec.data(), sizeof(float), DIM_SZ, f);
        int found_count = q.found_ids.size();
        fwrite(&found_count, sizeof(int), 1, f);
        fwrite(q.found_ids.data(), sizeof(int), found_count, f);
        int gt_count = q.gt_ids.size();
        fwrite(&gt_count, sizeof(int), 1, f);
        fwrite(q.gt_ids.data(), sizeof(int), gt_count, f);
        
        // Also dump the vector of the found node and GT nodes for plotting
        // We need to fetch them from GPU... this is slow but okay for just 10 queries at end
        // Actually we can't easily fetch random vectors here without keeping CPU copy.
        // Let's just assume python can look them up from initial_graph.bin if they are old,
        // but for new nodes it's hard.
        // BETTER: Write the actual vectors here.
        // Fetch found vecs (Top 10)
        std::vector<float> temp(DIM_SZ);
        for(int id : q.found_ids) {
            if (id != -1) {
                CHECK_CUDA(cudaMemcpy(temp.data(), &ctx.d_vectors[id * DIM_SZ], DIM_SZ * sizeof(float), cudaMemcpyDeviceToHost));
                fwrite(temp.data(), sizeof(float), DIM_SZ, f);
            } else {
                // Write dummy
                std::vector<float> dummy(DIM_SZ, 0.0f);
                fwrite(dummy.data(), sizeof(float), DIM_SZ, f);
            }
        }
        
        // Fetch GT vecs
        for(int id : q.gt_ids) {
            if (id != -1) {
                 CHECK_CUDA(cudaMemcpy(temp.data(), &ctx.d_vectors[id * DIM_SZ], DIM_SZ * sizeof(float), cudaMemcpyDeviceToHost));
                 fwrite(temp.data(), sizeof(float), DIM_SZ, f);
            }
        }
    }
};

int main(int argc, char** argv) {
    if (argc < 5) {
        printf("Usage: %s <initial_graph> <workload> <batch_size> <output_dir>\n", argv[0]);
        return 1;
    }
    
    const char* graph_file = argv[1];
    const char* workload_file = argv[2];
    int batch_size = atoi(argv[3]);
    const char* output_dir = argv[4];
    
#if DIM == 128
    VamanaEngine<128, 20000> engine;
#else
    VamanaEngine<2, 1000> engine;
#endif

    // Default to Automated Benchmark
    engine.run_automated_benchmark(graph_file, workload_file, batch_size, output_dir);
    
    return 0;
}
