#include "matrix.h"
#include "matrix_csv.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>

// CUDA kernel: Tiled matrix multiplication using shared memory
__global__ void matrix_multiplication_tiled_dkernel(
    int32_t* d_C, int32_t* d_A, int32_t* d_B,
    unsigned int M, unsigned int N, unsigned int K, unsigned int tile_width,
    unsigned long long* gmem_reads, unsigned long long* gmem_writes,
    unsigned long long* smem_reads, unsigned long long* smem_writes)
{
    extern __shared__ int32_t shm[];
    int32_t* TILE_A = shm;
    int32_t* TILE_B = shm + tile_width * (tile_width + 1);

    unsigned int bid_x = blockIdx.x * blockDim.x ;
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
        int32_t a_val = (tid_y < M && effective_col_inside_tile < N) ? 
                          d_A[tid_y * N + effective_col_inside_tile] : 0;

        // global read
        if (tid_y < M && effective_col_inside_tile < N)
            atomicAdd(gmem_reads, 1ULL);

        TILE_A[row_in_tile * tile_width + col_in_tile] = a_val;
        atomicAdd(smem_writes, 1ULL);  // SHM write

        unsigned int effective_row_inside_tile = phase * tile_width + row_in_tile;
        int32_t b_val = (tid_x < K && effective_row_inside_tile < N) ?
                          d_B[effective_row_inside_tile * K + tid_x] : 0;

        if (tid_x < K && effective_row_inside_tile < N)
            atomicAdd(gmem_reads, 1ULL);  // global read

        TILE_B[row_in_tile * tile_width + col_in_tile] = b_val;
        atomicAdd(smem_writes, 1ULL);  // SHM write

        __syncthreads();

        for (unsigned int k = 0; k < tile_width; ++k) {
            int a_reg = TILE_A[row_in_tile * tile_width + k];
            int b_reg = TILE_B[k * tile_width + col_in_tile];
            atomicAdd(smem_reads, 2ULL);  // two SHM reads
            tmp += a_reg * b_reg;
        }

        __syncthreads();
    }

    if (tid_y < M && tid_x < K) {
        d_C[tid_y * K + tid_x] = tmp;
        atomicAdd(gmem_writes, 1ULL);  // global write
    }
}

// Host-callable function
extern "C" void solve(int32_t* d_C, int32_t* d_A, int32_t* d_B,
                      int M, int N, int K, unsigned int tile_width,
                      unsigned int *kernel_time,
                      unsigned long long* h_greads,
                      unsigned long long* h_gwrites,
                      unsigned long long* h_sreads,
                      unsigned long long* h_swrites)  
{
    dim3 block(tile_width, tile_width);
    dim3 grid((K + tile_width - 1) / tile_width, (M + tile_width - 1) / tile_width);
    size_t shared_size_in_bytes = 2 * tile_width * (tile_width + 1) * sizeof(int32_t);

    unsigned long long *d_greads, *d_gwrites, *d_sreads, *d_swrites;
    cudaMalloc(&d_greads, sizeof(unsigned long long));
    cudaMalloc(&d_gwrites, sizeof(unsigned long long));
    cudaMalloc(&d_sreads, sizeof(unsigned long long));
    cudaMalloc(&d_swrites, sizeof(unsigned long long));
    cudaMemset(d_greads, 0, sizeof(unsigned long long));
    cudaMemset(d_gwrites, 0, sizeof(unsigned long long));
    cudaMemset(d_sreads, 0, sizeof(unsigned long long));
    cudaMemset(d_swrites, 0, sizeof(unsigned long long));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrix_multiplication_tiled_dkernel<<<grid, block, shared_size_in_bytes>>>(
        d_C, d_A, d_B, M, N, K, tile_width,
        d_greads, d_gwrites, d_sreads, d_swrites);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    *kernel_time = ms * 1000.0f; // microseconds

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(h_greads, d_greads, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_gwrites, d_gwrites, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sreads, d_sreads, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_swrites, d_swrites, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(d_greads);
    cudaFree(d_gwrites);
    cudaFree(d_sreads);
    cudaFree(d_swrites);
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

    Matrix A, B;
    matrix_read_from_csv_int32(&A, matrix_A_file_path);
    matrix_read_from_csv_int32(&B, matrix_B_file_path);

    if (A.num_cols != B.num_rows) {
        fprintf(stderr, "Error: Incompatible matrix dimensions\n");
        free(A.elements);
        free(B.elements);
        return 1;
    }

    Matrix C;
    C.num_rows = A.num_rows;
    C.num_cols = B.num_cols;
    C.elements = (int32_t*)malloc(C.num_rows * C.num_cols * sizeof(int32_t));

    int32_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.num_rows * A.num_cols * sizeof(int32_t));
    cudaMalloc(&d_B, B.num_rows * B.num_cols * sizeof(int32_t));
    cudaMalloc(&d_C, C.num_rows * C.num_cols * sizeof(int32_t));

    cudaMemcpy(d_A, A.elements, A.num_rows * A.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.elements, B.num_rows * B.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);

    unsigned int kernel_time_us = 0;
    unsigned long long greads=0, gwrites=0, sreads=0, swrites=0;

    solve(d_C, d_A, d_B, A.num_rows, A.num_cols, B.num_cols, tile_width,
          &kernel_time_us, &greads, &gwrites, &sreads, &swrites);

    cudaMemcpy(C.elements, d_C, C.num_rows * C.num_cols * sizeof(int32_t), cudaMemcpyDeviceToHost);
    matrix_write_to_csv_int32(&C, "output_2_CS25MTECH11015.csv");

    unsigned int num_elements = C.num_rows * C.num_cols;
    printf("Product Matrix of size %u stored as output_2_CS25MTECH11015.csv\n", num_elements);
    printf("Kernel execution time: %u microseconds\n", kernel_time_us);
    printf("Global Reads: %llu, Global Writes: %llu\n", greads, gwrites);
    printf("Shared Reads: %llu, Shared Writes: %llu\n", sreads, swrites);

    FILE* fout = fopen("output_2_CS25MTECH11015.txt", "w");
    if (fout) {
        fprintf(fout, "%u\n", num_elements);
        fprintf(fout, "%u\n", kernel_time_us);
        fprintf(fout, "Global Reads: %llu\n", greads);
        fprintf(fout, "Global Writes: %llu\n", gwrites);
        fprintf(fout, "Shared Reads: %llu\n", sreads);
        fprintf(fout, "Shared Writes: %llu\n", swrites);
        fclose(fout);
    }

    free(A.elements);
    free(B.elements);
    free(C.elements);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

