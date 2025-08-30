//Modify the previous (basic version) kernel to use Shared Memory for fetching the operands from 
//Global Memory.
//The tile sizes should be dynamically configurable (not #defines) via command-line arguments.
//Increase the tile size in steps till the maximum possible value and observe the corresponding change in execution time.
//You can tune and choose optimal kernel launch parameters (block sizes) and assign them in the code.
#include "matrix.h"
#include "matrix_csv.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <cstdint>

// ================= NEW: Instrumentation Counters =================
__device__ unsigned long long d_gmem_reads = 0;
__device__ unsigned long long d_gmem_writes = 0;
__device__ unsigned long long d_smem_reads = 0;
__device__ unsigned long long d_smem_writes = 0;

// CUDA kernel: Tiled matrix multiplication using shared memory
__global__ void matrix_multiplication_tiled_dkernel(
    int32_t* d_C, int32_t* d_A, int32_t* d_B,
    unsigned int M, unsigned int N, unsigned int K, unsigned int tile_width)
{
    extern __shared__ int32_t shm[];
    int32_t* TILE_A = shm;
    int32_t* TILE_B = shm + tile_width * (tile_width + 1); // +1 to avoid bank conflicts

    unsigned int bid_x = blockIdx.x * blockDim.x;
    unsigned int bid_y = blockIdx.y * blockDim.y;
    unsigned int tid_x = bid_x + threadIdx.x;
    unsigned int tid_y = bid_y + threadIdx.y;

    unsigned int row_in_tile = threadIdx.y;
    unsigned int col_in_tile = threadIdx.x;

    int32_t tmp = 0;
    unsigned int num_tiles = (N + tile_width - 1) / tile_width;

    for (unsigned int phase = 0; phase < num_tiles; ++phase)
    {
        unsigned int effective_col_inside_tile = phase * tile_width + col_in_tile;

        if (tid_y < M && effective_col_inside_tile < N) {
            TILE_A[row_in_tile * tile_width + col_in_tile] =
                d_A[tid_y * N + effective_col_inside_tile];
            atomicAdd(&d_gmem_reads, 1ULL);
        } else {
            TILE_A[row_in_tile * tile_width + col_in_tile] = 0;
        }
        atomicAdd(&d_smem_writes, 1ULL);

        unsigned int effective_row_inside_tile = phase * tile_width + row_in_tile;

        if (tid_x < K && effective_row_inside_tile < N) {
            TILE_B[row_in_tile * tile_width + col_in_tile] =
                d_B[effective_row_inside_tile * K + tid_x];
            atomicAdd(&d_gmem_reads, 1ULL);
        } else {
            TILE_B[row_in_tile * tile_width + col_in_tile] = 0;
        }
        atomicAdd(&d_smem_writes, 1ULL);

        __syncthreads();  // To Avoid Data Race

        for (unsigned int k = 0; k < tile_width; ++k) {
            tmp += TILE_A[row_in_tile * tile_width + k] *
                   TILE_B[k * tile_width + col_in_tile];
            atomicAdd(&d_smem_reads, 2ULL); // one read from TILE_A + one from TILE_B
        }

        __syncthreads(); // To make the right update visible
    }

    if (tid_y < M && tid_x < K) {
        d_C[tid_y * K + tid_x] = tmp;
        atomicAdd(&d_gmem_writes, 1ULL);
    }
}

// Host-callable function
extern "C" void solve(int32_t* d_C, int32_t* d_A, int32_t* d_B,
                      int M, int N, int K, unsigned int tile_width,
                      unsigned int *kernel_time)  
{
    dim3 block(tile_width, tile_width);
    dim3 grid((K + tile_width - 1) / tile_width,
              (M + tile_width - 1) / tile_width);
    size_t shared_size_in_bytes =
        2 * tile_width * (tile_width + 1) * sizeof(int32_t);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Reset counters before launch
    cudaMemcpyToSymbol(d_gmem_reads, 0, sizeof(unsigned long long));
    cudaMemcpyToSymbol(d_gmem_writes, 0, sizeof(unsigned long long));
    cudaMemcpyToSymbol(d_smem_reads, 0, sizeof(unsigned long long));
    cudaMemcpyToSymbol(d_smem_writes, 0, sizeof(unsigned long long));

    cudaEventRecord(start);
    matrix_multiplication_tiled_dkernel<<<grid, block, shared_size_in_bytes>>>(
        d_C, d_A, d_B, M, N, K, tile_width);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    *kernel_time = ms * 1000.0f;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Copy counters back to host
    unsigned long long h_gmem_reads, h_gmem_writes, h_smem_reads, h_smem_writes;
    cudaMemcpyFromSymbol(&h_gmem_reads, d_gmem_reads, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&h_gmem_writes, d_gmem_writes, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&h_smem_reads, d_smem_reads, sizeof(unsigned long long));
    cudaMemcpyFromSymbol(&h_smem_writes, d_smem_writes, sizeof(unsigned long long));

    // Print results to stdout only
    printf("Kernel Execution Time: %u microseconds\n", *kernel_time);
    printf("Global Memory Reads : %llu\n", h_gmem_reads);
    printf("Global Memory Writes: %llu\n", h_gmem_writes);
    printf("Shared Memory Reads : %llu\n", h_smem_reads);
    printf("Shared Memory Writes: %llu\n", h_smem_writes);
}



int main(int argc, char* argv[])
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <TILE_WIDTH> <matrix_a.csv> <matrix_b.csv>\n", argv[0]);
        return 1;
    }

    unsigned int tile_width = atoi(argv[1]);
    const char* matrix_A_file_path = argv[2];
    const char* matrix_B_file_path = argv[3];

    // Initialize matrices
    Matrix A;
    Matrix B;
    
    matrix_read_from_csv_int32(&A, matrix_A_file_path);
    matrix_read_from_csv_int32(&B, matrix_B_file_path);

    if (A.num_cols != B.num_rows) {
        fprintf(stderr, "Error: Incompatible matrix dimensions\n");
        free(A.elements);
        free(B.elements);
        return 1;
    }

    // Allocate C matrix
    Matrix C;
    C.num_rows = A.num_rows;
    C.num_cols = B.num_cols;
    C.elements = (int32_t*)malloc(C.num_rows * C.num_cols * sizeof(int32_t));

    // Allocate device memory
    int32_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.num_rows * A.num_cols * sizeof(int32_t));
    cudaMalloc(&d_B, B.num_rows * B.num_cols * sizeof(int32_t));
    cudaMalloc(&d_C, C.num_rows * C.num_cols * sizeof(int32_t));

    // Copy host data to device
    cudaMemcpy(d_A, A.elements, A.num_rows * A.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.elements, B.num_rows * B.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    unsigned int kernel_time_us = 0.0f;
    solve(d_C, d_A, d_B, A.num_rows, A.num_cols, B.num_cols, tile_width, &kernel_time_us);

    // Copy result back to host
    cudaMemcpy(C.elements, d_C, C.num_rows * C.num_cols * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Write output matrix
    matrix_write_to_csv_int32(&C, "output_2_CS25MTECH11015.csv");

    unsigned int num_elements = C.num_rows * C.num_cols;
    printf("Product Matrix of size %u stored as output_2_CS25MTECH11015.csv\n", num_elements);
    printf("Kernel execution time: %u microseconds\n", kernel_time_us);

    // Write stats to output.txt
    FILE* fout = fopen("output_2_CS25MTECH11015.txt", "w");
    if (fout) {
        fprintf(fout, "%u\n", num_elements);
        fprintf(fout, "%u\n", kernel_time_us);
        fclose(fout);
    }

    // Free device memory
    free(A.elements);
    free(B.elements);
    free(C.elements);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
