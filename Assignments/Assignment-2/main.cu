#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <string.h>
#include <iostream>
#include <cstdlib>

#define FULL_MASK 0xffffffff

const unsigned int WARP_SIZE = 32;

//-----------------------------CUDA Timers-----------------------------
struct cudaTimers{
    cudaEvent_t start;
    cudaEvent_t stop;
};
typedef struct cudaTimers CudaTimers;

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

// ---------------------------- Warp-per-array Odd-Even Sort Kernel ----------------------------
__global__ void warp_per_array_oddeven_sort(uint32_t *d_arr, uint32_t *d_offsets, size_t num_arrays) {
    unsigned int tid_x   = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int warp_id = tid_x / WARP_SIZE;
    if (warp_id >= num_arrays) return;

    unsigned int lane = threadIdx.x % WARP_SIZE;
    unsigned int mask = __activemask();

    size_t start = d_offsets[warp_id];
    size_t end = d_offsets[warp_id + 1];
    size_t len   = end - start;
    if (len <= 1 || len > 128) return; // arrays must fit in shared buffer

    // Shared memory buffer: 128 elements per warp, up to 4 warps per block
    __shared__ uint32_t shmem[128 * 4];
    uint32_t *local = &shmem[(threadIdx.x / WARP_SIZE) * 128];

    // Load array into shared memory (strided) 
    for (size_t i = lane; i < len; i += WARP_SIZE) {
        local[i] = d_arr[start + i];
    }
    // __syncwarp(mask);

    // Odd-even sort 
    for (size_t pass = 0; pass < len; ++pass) {
        int swap_flag = 0;

        for (size_t i = lane; i + 1 < len; i += WARP_SIZE) {
            if ((i % 2) == (pass % 2)) {
                uint32_t a = local[i];
                uint32_t b = local[i + 1];
                if (a > b) {
                    local[i]     = b;
                    local[i + 1] = a;
                    swap_flag    = 1;
                }
            }
        }

        // __syncwarp(mask);

        // if no swaps in this pass, array is sorted
        // if (__all_sync(mask, swap_flag)) {
        //     printf("early exit..,\n");
        //     break;       
        // }
    }

    // Write back to global memory (strided) 
    for (size_t i = lane; i < len; i += WARP_SIZE) {
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

    unsigned int num_warps_per_block = 4;       // 4 warps = 128 threads
    const unsigned int num_threads_per_warp = WARP_SIZE;
    unsigned int num_threads_per_block = num_warps_per_block * num_threads_per_warp;
    unsigned int num_warps = (unsigned int)num_arrays;
    unsigned int num_blocks_per_grid = (num_warps + num_warps_per_block - 1) / num_warps_per_block;

    printf("Threads per block = %d\n", num_threads_per_block);
    printf("Blocks per grid   = %d\n", num_blocks_per_grid);

    CudaTimers timers;
    cudaEventCreate(&timers.start);
    cudaEventCreate(&timers.stop);
    cudaEventRecord(timers.start);

    warp_per_array_oddeven_sort<<<num_blocks_per_grid, num_threads_per_block>>>(d_arr, d_offsets, num_arrays);

    cudaEventRecord(timers.stop);
    cudaEventSynchronize(timers.stop);

    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, timers.start, timers.stop);
    printf("Elapsed time: %.4f ms\n", elapsed_time);

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