#include "common.h"
#include "device_funcs.cu"
#include <random>
#include <fstream>
#include <algorithm>
#include <vector>

/**
 * @file build_graph.cu
 * @brief CUDA implementation of Vamana graph construction.
 *
 * Implements random initialization, greedy search, and robust pruning
 * to construct a high-quality approximate nearest neighbor graph.
 */

// -------------------------------------------------------------------------
// Kernels
// -------------------------------------------------------------------------

/**
 * @brief Initializes the graph with random neighbors.
 *
 * @param ctx Graph context.
 * @param num_nodes Number of nodes in the graph.
 * @param R Target degree for random initialization.
 * @param seed Random seed.
 */
__global__ void randomInitKernel(GraphContext ctx, int num_nodes, int R, unsigned long long seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    // Simple LCG for random numbers
    unsigned long long state = seed + tid;

    int* adj = &ctx.d_adj[tid * MAX_DEGREE];
    int count = 0;

    // Fill with -1
    for(int i=0; i<MAX_DEGREE; ++i) adj[i] = -1;

    while (count < R) {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        int target = (state >> 32) % num_nodes;

        if (target != tid) {
            // Check duplicates
            bool exists = false;
            for (int i = 0; i < count; ++i) {
                if (adj[i] == target) { exists = true; break; }
            }
            if (!exists) {
                adj[count++] = target;
            }
        }
    }
}

/**
 * @brief Greedy search for construction (collects full candidate list).
 *
 * @tparam DIM_SZ Dimension of the vectors.
 * @param ctx Graph context.
 * @param query_vec Query vector.
 * @param start_node Starting node ID.
 * @param candidates Output candidate list.
 * @param beam_width Beam width.
 */
template <int DIM_SZ>
__device__ void greedy_search_build(
    const GraphContext ctx,
    const float* query_vec,
    int start_node,
    CandidateList<BEAM_WIDTH>& candidates,
    int beam_width
) {
    candidates.init();

    int visited[VISITED_LIST_SIZE];
    int visited_count = 0;

    float start_dist = distance<DIM_SZ>(query_vec, &ctx.d_vectors[start_node * DIM_SZ]);
    candidates.insert(start_node, start_dist, beam_width);

    int iter = 0;
    while (iter < 200) {
        iter++;
        int next_node = -1;

        // Find closest unvisited
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

            // No candidates.contains(nbr) check here, as per Vamana construction
            // we want to explore all neighbors and let pruning handle duplicates/redundancy.

            float d = distance<DIM_SZ>(query_vec, &ctx.d_vectors[nbr * DIM_SZ]);
            candidates.insert(nbr, d, beam_width);
        }
    }
}

/**
 * @brief Kernel to perform search for all nodes to find candidates.
 *
 * @tparam DIM_SZ Dimension of the vectors.
 * @param ctx Graph context.
 * @param num_nodes Number of nodes.
 * @param start_node Starting node (medoid).
 * @param L Beam width / candidate list size.
 * @param d_candidates Output array for candidates [num_nodes * L].
 * @param d_dists Output array for distances [num_nodes * L].
 */
template <int DIM_SZ>
__global__ void searchCandidatesKernel(
    GraphContext ctx,
    int num_nodes,
    int start_node,
    int L,
    int* d_candidates,
    float* d_dists
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    const float* query = &ctx.d_vectors[tid * DIM_SZ];
    CandidateList<BEAM_WIDTH> candidates; // Max L=128 (BEAM_WIDTH)

    greedy_search_build<DIM_SZ>(ctx, query, start_node, candidates, L);

    // Store results
    for(int i=0; i<L; ++i) {
        if (i < candidates.size) {
            d_candidates[tid * L + i] = candidates.ids[i];
            d_dists[tid * L + i] = candidates.dists[i];
        } else {
            d_candidates[tid * L + i] = -1;
            d_dists[tid * L + i] = 1e30f;
        }
    }
}

/**
 * @brief Kernel to perform robust pruning.
 *
 * @tparam DIM_SZ Dimension of the vectors.
 * @param ctx Graph context.
 * @param num_nodes Number of nodes.
 * @param R Target degree (max degree).
 * @param alpha Pruning parameter.
 * @param d_candidates Input candidates from search.
 * @param d_dists Input distances from search.
 * @param L Number of candidates per node.
 */
template <int DIM_SZ>
__global__ void robustPruneKernel(
    GraphContext ctx,
    int num_nodes,
    int R,
    float alpha,
    const int* d_candidates,
    const float* d_dists,
    int L
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_nodes) return;

    // Collect all candidates: existing neighbors + search results
    // Max candidates = R + L. Assume R=32, L=128 -> 160.

    int cand_ids[200];
    float cand_dists[200];
    int cand_count = 0;

    const float* my_vec = &ctx.d_vectors[tid * DIM_SZ];

    // 1. Add Search Candidates
    for(int i=0; i<L; ++i) {
        int id = d_candidates[tid * L + i];
        if (id != -1 && id != tid) {
            cand_ids[cand_count] = id;
            cand_dists[cand_count] = d_dists[tid * L + i];
            cand_count++;
        }
    }

    // 2. Add Existing Neighbors
    int* adj = &ctx.d_adj[tid * MAX_DEGREE];
    for(int i=0; i<MAX_DEGREE; ++i) {
        int nbr = adj[i];
        if (nbr != -1 && nbr != tid) {
            // Check duplicate
            bool dup = false;
            for(int j=0; j<cand_count; ++j) if (cand_ids[j] == nbr) { dup = true; break; }
            if (!dup) {
                cand_ids[cand_count] = nbr;
                cand_dists[cand_count] = distance<DIM_SZ>(my_vec, &ctx.d_vectors[nbr * DIM_SZ]);
                cand_count++;
            }
        }
    }

    // 3. Sort by distance (Bubble sort)
    for(int i=0; i<cand_count-1; ++i) {
        for(int j=0; j<cand_count-i-1; ++j) {
            if (cand_dists[j] > cand_dists[j+1]) {
                float td = cand_dists[j]; cand_dists[j] = cand_dists[j+1]; cand_dists[j+1] = td;
                int ti = cand_ids[j]; cand_ids[j] = cand_ids[j+1]; cand_ids[j+1] = ti;
            }
        }
    }

    // 4. Robust Prune
    int new_adj[MAX_DEGREE];
    int new_count = 0;

    // Initialize new_adj with -1
    for(int i=0; i<MAX_DEGREE; ++i) new_adj[i] = -1;

    bool kept[200];
    for(int i=0; i<cand_count; ++i) kept[i] = true;

    for(int i=0; i<cand_count && new_count < R; ++i) {
        if (!kept[i]) continue;

        int p_star = cand_ids[i];
        new_adj[new_count++] = p_star;

        // Prune remaining
        for(int j=i+1; j<cand_count; ++j) {
            if (!kept[j]) continue;

            int p_prime = cand_ids[j];
            float dist_p_star_prime = distance<DIM_SZ>(&ctx.d_vectors[p_star * DIM_SZ], &ctx.d_vectors[p_prime * DIM_SZ]);

            if (alpha * dist_p_star_prime < cand_dists[j]) {
                kept[j] = false;
            }
        }
    }

    // Write back
    for(int i=0; i<MAX_DEGREE; ++i) {
        adj[i] = (i < new_count) ? new_adj[i] : -1;
    }
}

// -------------------------------------------------------------------------
// Host Code
// -------------------------------------------------------------------------

/**
 * @brief Host function to coordinate graph construction.
 * 
 * Loads data, initializes the graph, and runs the Vamana optimization loop.
 * 
 * @tparam DIM_SZ Dimension of the vectors.
 * @param input_file Path to input vectors.
 * @param output_file Path to save the built graph.
 * @param R Target degree (max degree).
 * @param L Beam width for construction.
 * @param alpha Pruning parameter.
 */
template <int DIM_SZ>
void build_graph(const char* input_file, const char* output_file, int R, int L, float alpha) {
    // 1. Load Data
    std::ifstream in(input_file, std::ios::binary);
    if (!in) { std::cerr << "Error opening input file" << std::endl; exit(1); }

    // Input file format: [N (int)] [DIM (int)] [Vectors (N*DIM floats)]
    // This matches the temporary file created by gen_sift_workload.py.

    int num_nodes, dim;
    in.read((char*)&num_nodes, sizeof(int));
    in.read((char*)&dim, sizeof(int));

    if (dim != DIM_SZ) { std::cerr << "Dim mismatch" << std::endl; exit(1); }

    std::vector<float> h_vectors(num_nodes * DIM_SZ);
    in.read((char*)h_vectors.data(), num_nodes * DIM_SZ * sizeof(float));
    in.close();

    printf("Loaded %d vectors (dim=%d)\n", num_nodes, dim);

    // 2. Setup GPU
    GraphContext ctx;
    ctx.dim = DIM_SZ;
    ctx.max_capacity = num_nodes;

    CHECK_CUDA(cudaMalloc(&ctx.d_vectors, num_nodes * DIM_SZ * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&ctx.d_adj, num_nodes * MAX_DEGREE * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&ctx.d_delete_mask, num_nodes * sizeof(int))); // Not used but needed for struct
    CHECK_CUDA(cudaMemset(ctx.d_delete_mask, 0, num_nodes * sizeof(int)));

    CHECK_CUDA(cudaMemcpy(ctx.d_vectors, h_vectors.data(), num_nodes * DIM_SZ * sizeof(float), cudaMemcpyHostToDevice));

    // 3. Init Random Graph
    printf("Initializing random graph...\n");
    randomInitKernel<<< (num_nodes+255)/256, 256 >>>(ctx, num_nodes, R, 1234ULL);
    CHECK_CUDA(cudaDeviceSynchronize());

    // 4. Vamana Optimization
    int* d_candidates;
    float* d_dists;
    CHECK_CUDA(cudaMalloc(&d_candidates, num_nodes * L * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_dists, num_nodes * L * sizeof(float)));
    
    // Calculate centroid on host to find medoid approximation
    std::vector<float> centroid(DIM_SZ, 0.0f);
    for(int i=0; i<num_nodes; ++i) {
        for(int d=0; d<DIM_SZ; ++d) centroid[d] += h_vectors[i*DIM_SZ + d];
    }
    for(int d=0; d<DIM_SZ; ++d) centroid[d] /= num_nodes;
    
    int medoid = 0;
    float min_dist = 1e30f;
    for(int i=0; i<num_nodes; ++i) {
        float d = 0;
        for(int k=0; k<DIM_SZ; ++k) {
            float diff = h_vectors[i*DIM_SZ + k] - centroid[k];
            d += diff*diff;
        }
        if (d < min_dist) { min_dist = d; medoid = i; }
    }
    printf("Medoid: %d\n", medoid);
    
    for (int pass = 0; pass < 2; ++pass) {
        printf("Pass %d/2...\n", pass+1);
        
        // Search
        searchCandidatesKernel<DIM_SZ><<< (num_nodes+255)/256, 256 >>>(ctx, num_nodes, medoid, L, d_candidates, d_dists);
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // Prune
        robustPruneKernel<DIM_SZ><<< (num_nodes+255)/256, 256 >>>(ctx, num_nodes, R, alpha, d_candidates, d_dists, L);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    // 5. Save Graph
    std::vector<int> h_adj(num_nodes * MAX_DEGREE);
    CHECK_CUDA(cudaMemcpy(h_adj.data(), ctx.d_adj, num_nodes * MAX_DEGREE * sizeof(int), cudaMemcpyDeviceToHost));
    
    std::ofstream out(output_file, std::ios::binary);
    // Write format expected by engine: [N] [DIM] [MAX_DEGREE] [VECS] [ADJ]
    out.write((char*)&num_nodes, sizeof(int));
    out.write((char*)&dim, sizeof(int));
    int max_degree = MAX_DEGREE;
    out.write((char*)&max_degree, sizeof(int));
    out.write((char*)h_vectors.data(), num_nodes * DIM_SZ * sizeof(float));
    out.write((char*)h_adj.data(), num_nodes * MAX_DEGREE * sizeof(int));
    out.close();
    
    printf("Saved graph to %s\n", output_file);
    
    cudaFree(d_candidates);
    cudaFree(d_dists);
    cudaFree(ctx.d_vectors);
    cudaFree(ctx.d_adj);
    cudaFree(ctx.d_delete_mask);
}

int main(int argc, char** argv) {
    if (argc < 6) {
        printf("Usage: %s <input_vectors> <output_graph> <R> <L> <alpha>\n", argv[0]);
        return 1;
    }
    
    const char* input_file = argv[1];
    const char* output_file = argv[2];
    int R = atoi(argv[3]);
    int L = atoi(argv[4]);
    float alpha = atof(argv[5]);
    
#if DIM == 128
    build_graph<128>(input_file, output_file, R, L, alpha);
#else
    build_graph<2>(input_file, output_file, R, L, alpha);
#endif

    return 0;
}
