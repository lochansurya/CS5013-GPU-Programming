#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <string.h>
#include <iostream>
#include <cstdlib>

// ---------------------------- Arrays struct & CSV APIs(Destination-First) ----------------------------
struct arrays{
    uint32_t *arr;
    uint32_t *offsets;
    size_t num_arrays;
    size_t total_len;
};

typedef struct arrays Arrays;

Arrays* read_from_csv_file_uint32(const char* input_csv_file_path);
void write_to_csv_file_uint32(const char *output_file_path, const Arrays *arrays);
void free_arrays(Arrays *arrays);

Arrays* read_from_csv_file_uint32(const char *input_csv_file_path){
    Arrays* arrays = (Arrays*)malloc(sizeof(Arrays));
    FILE *fp_in = fopen(input_csv_file_path, "r");
    if(!fp_in){
        printf("File Opening Error\n");
        return NULL;
    }

    const size_t N = 1 << 15; // >= 30K
    const size_t L = 1 << 7;  // 128

    arrays->arr = (uint32_t*)malloc(N * L * sizeof(uint32_t));
    arrays->offsets = (uint32_t*)malloc((N + 1) * sizeof(uint32_t));
    if(!arrays->arr || !arrays->offsets){
        printf("Memory allocation failed!\n");
        free(arrays->arr);
        free(arrays->offsets);
        free(arrays);
        fclose(fp_in);
        return NULL;
    }

    arrays->num_arrays = 0;
    arrays->total_len = 0;
    arrays->offsets[0] = 0;

    char *lineptr = NULL;
    size_t n = 0;
    while(getline(&lineptr, &n, fp_in) != -1){
        char *token = strtok(lineptr, ",");
        if(!token) continue;
        size_t len = atoi(token);
        size_t base = arrays->total_len;
        arrays->offsets[arrays->num_arrays] = base;
        for(size_t i = 0; i < len; ++i){
            token = strtok(NULL, ",");
            arrays->arr[base + i] = token ? (uint32_t)strtoul(token, NULL, 10): 0;
        }
        arrays->total_len += len;
        arrays->num_arrays++;
    }
    arrays->offsets[arrays->num_arrays] = arrays->total_len;

    free(lineptr);
    fclose(fp_in);
    return arrays;
}

void write_to_csv_file_uint32(const char *output_csv_file_path, const Arrays *arrays){
    FILE *fp_out = fopen(output_csv_file_path, "w");
    if(!fp_out){
        printf("File Opening Error\n");
        return ;
    }

    for(uint32_t i = 0; i < arrays->num_arrays; ++i){
        size_t start = arrays->offsets[i];
        size_t end   = arrays->offsets[i+1];
        size_t arr_len = end - start;

        fprintf(fp_out, "%zu", arr_len);
        for(size_t j = start; j < end; ++j){
            fprintf(fp_out, ",%u", arrays->arr[j]);
        }
        fprintf(fp_out, "\r\n");
    }
    fclose(fp_out);
}

void free_arrays(Arrays *arrays){
    if(!arrays) return;
    free(arrays->arr);
    free(arrays->offsets);
    free(arrays);
}

// ---------------------------- Block-per-array Odd-Even Sort Kernel with Early Exit ----------------------------
__global__ void block_per_array_oddeven_sort(uint32_t *d_arr, uint32_t *d_offsets, size_t num_arrays) {
    int array_idx = blockIdx.x;
    if (array_idx >= (int)num_arrays) return;

    size_t start = d_offsets[array_idx];
    size_t end   = d_offsets[array_idx + 1];
    size_t len   = end - start;
    if (len <= 1) return;

    extern __shared__ uint32_t s_mem[];
    uint32_t *local = s_mem;
    __shared__ int isSorted;

    // Cooperative load
    for (size_t i = threadIdx.x; i < len; i += blockDim.x) {
        local[i] = d_arr[start + i];
    }
    __syncthreads();

    for (size_t phase = 0; phase < len; ++phase) {
        if (threadIdx.x == 0) isSorted = 1; // assume sorted
        __syncthreads();

        size_t parity = phase & 1;
        for (size_t i = 2 * threadIdx.x + parity; i + 1 < len; i += 2 * blockDim.x) {
            if (local[i] > local[i + 1]) {
                uint32_t tmp = local[i];
                local[i]     = local[i + 1];
                local[i + 1] = tmp;
                atomicExch(&isSorted, 0); // mark unsorted
            }
        }
        __syncthreads();

        if (isSorted) break; // early exit
    }

    // Write back
    for (size_t i = threadIdx.x; i < len; i += blockDim.x) {
        d_arr[start + i] = local[i];
    }
}

// ---------------------------- Solver ----------------------------
extern "C" void solver(Arrays *arrays, int max_array_len){
    if(!arrays || arrays->num_arrays == 0) return;

    uint32_t *d_arr = nullptr;
    uint32_t *d_offsets = nullptr;
    size_t N = arrays->total_len;
    size_t num_arrays = arrays->num_arrays;

    cudaMalloc(&d_arr, N * sizeof(uint32_t));
    cudaMalloc(&d_offsets, (num_arrays + 1) * sizeof(uint32_t));

    cudaMemcpy(d_arr, arrays->arr, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, arrays->offsets, (num_arrays + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    int threads_per_block = (max_array_len + 1) / 2;
    if (threads_per_block > 1024) threads_per_block = 1024;
    if (threads_per_block < 1) threads_per_block = 1;

    int blocks_per_grid = (int)num_arrays;

    printf("Threads per block = %d\n", threads_per_block);
    printf("Blocks per grid   = %d\n", blocks_per_grid);

    size_t shared_size = (size_t)max_array_len * sizeof(uint32_t);

    block_per_array_oddeven_sort<<<blocks_per_grid, threads_per_block, shared_size>>>(d_arr, d_offsets, num_arrays);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA sync error: %s\n", cudaGetErrorString(err));
    }

    cudaMemcpy(arrays->arr, d_arr, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_offsets);
}

// ---------------------------- Main ----------------------------
int main(int argc, char *argv[]){
    if(argc != 4){
        std::cerr << "Usage: " << argv[0] << " <input_csv_file> -o <output_csv_file>\n";
        return 1;
    }

    const char *input_file = argv[1];
    const char *flag_o     = argv[2];
    const char *output_file= argv[3];

    if(strcmp(flag_o, "-o") != 0){
        std::cerr << "Invalid flags. Expected -o.\n";
        return 1;
    }

    Arrays *arrays = read_from_csv_file_uint32(input_file);
    if(!arrays){
        std::cerr << "Failed to read input CSV file.\n";
        return 1;
    }

    int max_array_len = 0;
    for(size_t i = 0; i < arrays->num_arrays; i++){
        int len = arrays->offsets[i+1] - arrays->offsets[i];
        if(len > max_array_len) max_array_len = len;
    }

    solver(arrays, max_array_len);
    write_to_csv_file_uint32(output_file, arrays);
    free_arrays(arrays);

    std::cout << "Sorting completed. Output written to " << output_file << "\n";
    return 0;
}
