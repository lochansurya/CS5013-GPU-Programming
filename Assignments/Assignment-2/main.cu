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
        fprintf(fp_out, "\n");
    }
    fclose(fp_out);
}

void free_arrays(Arrays *arrays){
    if(!arrays) return;
    free(arrays->arr);
    free(arrays->offsets);
    free(arrays);
}

// ---------------------------- Thread-per-array Radix Kernel (fixed) ----------------------------
__global__ void thread_per_array_radix_sort(uint32_t *d_arr, uint32_t *d_offsets, size_t num_arrays, int max_array_len){
    int array_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(array_idx >= num_arrays) return;

    size_t start = d_offsets[array_idx];
    size_t end   = d_offsets[array_idx + 1];
    size_t len   = end - start;
    if(len == 0) return;

    extern __shared__ uint32_t s_mem[]; // only for local array
    uint32_t* local = s_mem + threadIdx.x * max_array_len;

    uint32_t temp[128]; // temp in registers / local memory

    for(size_t i = 0; i < len; i++)
        local[i] = d_arr[start + i];

    for(int pass = 0; pass < 4; pass++){
        int count[256] = {0};
        for(size_t i = 0; i < len; i++){
            int digit = (local[i] >> (pass*8)) & 0xFF;
            count[digit]++;
        }
        int sum = 0;
        for(int i = 0; i < 256; i++){
            int c = count[i];
            count[i] = sum;
            sum += c;
        }
        for(size_t i = 0; i < len; i++){
            int digit = (local[i] >> (pass*8)) & 0xFF;
            temp[count[digit]++] = local[i];
        }
        for(size_t i = 0; i < len; i++)
            local[i] = temp[i];
    }

    for(size_t i = 0; i < len; i++)
        d_arr[start + i] = local[i];
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

    int num_threads_per_block = 48*1024 / (max_array_len * sizeof(uint32_t));
    int num_blocks_per_grid = (num_arrays + num_threads_per_block - 1) / num_threads_per_block;

    // Allocate double shared memory: local + temp
    size_t shared_size_in_num_bytes = num_threads_per_block * max_array_len * sizeof(uint32_t);

    thread_per_array_radix_sort<<<num_blocks_per_grid, num_threads_per_block, shared_size_in_num_bytes>>>(d_arr, d_offsets, num_arrays, max_array_len);
    cudaDeviceSynchronize();

    cudaMemcpy(arrays->arr, d_arr, N * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cudaFree(d_arr);
    cudaFree(d_offsets);
}

// ---------------------------- Main ----------------------------
int main(int argc, char *argv[]){
    if(argc != 6){
        std::cerr << "Usage: " << argv[0] << " <input_csv_file> -L <length_upper_bound> -o <output_csv_file>\n";
        return 1;
    }

    const char *input_file = argv[1];
    const char *flag_L = argv[2];
    const char *length_str = argv[3];
    const char *flag_o = argv[4];
    const char *output_file = argv[5];

    if(strcmp(flag_L, "-L") != 0 || strcmp(flag_o, "-o") != 0){
        std::cerr << "Invalid flags.\n";
        return 1;
    }

    int max_array_len = std::atoi(length_str);
    if(max_array_len <= 0){
        std::cerr << "Invalid length upper bound.\n";
        return 1;
    }

    Arrays *arrays = read_from_csv_file_uint32(input_file);
    if(!arrays){
        std::cerr << "Failed to read input CSV file.\n";
        return 1;
    }

    solver(arrays, max_array_len);
    write_to_csv_file_uint32(output_file, arrays);
    free_arrays(arrays);

    std::cout << "Sorting completed. Output written to " << output_file << "\n";
    return 0;
}
