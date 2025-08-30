#include "matrix.h"
#include "matrix_csv.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

// Kernel: Transpose input -> output using shared memory tiling (+1 padding to avoid bank conflicts)
__global__
void matrix_transpose_tiled_dkernel(int32_t *d_output, int32_t *d_input, 
                                    unsigned int num_rows, 
                                    unsigned int num_cols,
                                    unsigned int shared_size_in_bytes, 
                                    unsigned int tile_width) {
    // dynamic shared memory
    extern __shared__ int32_t shm[];
    // stride with +1 padding in the x dimension to avoid bank conflicts
    const unsigned int stride = tile_width + 1;
    int32_t *TILE = shm;

    // -------------------------
    // 1) block origin in global coords (top-left corner of this block in the input matrix)
    // -------------------------
    unsigned int block_start_col = blockIdx.x * blockDim.x; // starting col in input for this block
    unsigned int block_start_row = blockIdx.y * blockDim.y; // starting row in input for this block

    // -------------------------
    // 2) local thread coords
    // -------------------------
    unsigned int local_col = threadIdx.x; // [0..blockDim.x-1]
    unsigned int local_row = threadIdx.y; // [0..blockDim.y-1]

    // -------------------------
    // 3) global thread coords in the input matrix
    // -------------------------
    unsigned int global_col_in = block_start_col + local_col;
    unsigned int global_row_in = block_start_row + local_row;

    // -------------------------
    // 4) load from input into shared memory
    // -------------------------
    if (global_row_in < num_rows && global_col_in < num_cols) {
        unsigned int idx_in = global_row_in * num_cols + global_col_in;
        TILE[local_row * stride + local_col] = d_input[idx_in];
    } else {
        TILE[local_row * stride + local_col] = 0; // padding for out-of-bounds
    }

    __syncthreads();

    // -------------------------
    // 5) compute block origin in the output matrix
    //    (notice: blockIdx.x <-> blockIdx.y swapped)
    // -------------------------
    unsigned int block_start_col_out = blockIdx.y * blockDim.y; // becomes columns in output
    unsigned int block_start_row_out = blockIdx.x * blockDim.x; // becomes rows in output

    // -------------------------
    // 6) compute global coords in the output matrix
    // -------------------------
    unsigned int global_col_out = block_start_col_out + local_col; // [0 .. num_rows-1]
    unsigned int global_row_out = block_start_row_out + local_row; // [0 .. num_cols-1]

    // -------------------------
    // 7) store from shared memory transposed
    //    output is [num_cols x num_rows]
    // -------------------------
    if (global_row_out < num_cols && global_col_out < num_rows) {
        unsigned int idx_out = global_row_out * num_rows + global_col_out; // row-major: out_cols = num_rows
        d_output[idx_out] = TILE[local_col * stride + local_row];
    }
}

// Host-callable entry point
extern "C"
void solve(int32_t* d_A, int32_t* d_A_T,
           unsigned int /*M*/, unsigned int /*N*/,
           unsigned int num_rows,
           unsigned int num_cols,
           unsigned int shared_size_in_bytes,
           unsigned int tile_width,
           unsigned int *kernel_time_us) {
    // each block handles a tile_width x tile_width tile
    dim3 dimBlock(tile_width, tile_width, 1);
    dim3 dimGrid((num_cols + tile_width - 1) / tile_width,
                 (num_rows + tile_width - 1) / tile_width,
                 1);

    // shared memory size passed via 3rd kernel launch param
    size_t shm_bytes = shared_size_in_bytes; // Dynamic Device Memory allocation: Shared Memory Size Configuration at kernel-launch-time

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrix_transpose_tiled_dkernel<<<dimGrid, dimBlock, shm_bytes>>>(
        d_A_T, d_A, num_rows, num_cols, shared_size_in_bytes, tile_width);
    cudaEventRecord(stop);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    *kernel_time_us = (unsigned int)(ms * 1000.0f);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Example host main (using Matrix helpers)
int main(int argc, char** argv) {
    // Expect: tile_width, input.csv
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <tile_width> <matrix_a.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // tile width
    unsigned int tile_width = (unsigned int)atoi(argv[1]);
    if (tile_width == 0) {
        fprintf(stderr, "Error: tile_width must be a positive integer.\n");
        return EXIT_FAILURE;
    }

    // input matrix csv filepath
    const char *matrix_A_csv_file_path = argv[2];

    Matrix A;
    matrix_read_from_csv_int32(&A, matrix_A_csv_file_path);

    printf("Shape(A) = (%u, %u)\n", A.num_rows, A.num_cols);

    // output matrix (transpose)
    Matrix A_T;
    A_T.num_cols = A.num_rows;
    A_T.num_rows = A.num_cols;
    A_T.elements = (int32_t*)malloc((size_t)A_T.num_cols * (size_t)A_T.num_rows * sizeof(int32_t));
    if (!A_T.elements) {
        fprintf(stderr, "malloc failed for A_T.elements\n");
        return EXIT_FAILURE;
    }

    int32_t *d_A = nullptr, *d_A_T = nullptr;
    size_t bytes_A   = (size_t)A.num_cols   * (size_t)A.num_rows   * sizeof(int32_t);
    size_t bytes_A_T = (size_t)A_T.num_cols * (size_t)A_T.num_rows * sizeof(int32_t);

    cudaError_t err_A = cudaMalloc(&d_A, bytes_A);
    if (err_A != cudaSuccess){
        printf("cudaMalloc Failed for d_A: %s\n", cudaGetErrorString(err_A));
        free(A_T.elements);
        return EXIT_FAILURE;
    }

    cudaError_t err_A_T = cudaMalloc(&d_A_T, bytes_A_T);
    if (err_A_T != cudaSuccess){
        printf("cudaMalloc Failed for d_A_T: %s\n", cudaGetErrorString(err_A_T));
        cudaFree(d_A);
        free(A_T.elements);
        return EXIT_FAILURE;
    }

    cudaError_t err_H2D = cudaMemcpy(d_A, A.elements, bytes_A, cudaMemcpyHostToDevice);
    if (err_H2D != cudaSuccess){
        printf("CUDA memcpy H2D Error: %s\n", cudaGetErrorString(err_H2D));
        cudaFree(d_A);
        cudaFree(d_A_T);
        free(A_T.elements);
        return EXIT_FAILURE;
    }

    // Dynamic shared memory size: tile_width * (tile_width + 1) to account for padded stride
    unsigned int shared_size_in_bytes = tile_width * (tile_width + 1) * (unsigned int)sizeof(int32_t);

    unsigned int kernel_time_us = 0;
    // Call the C wrapper (M, N kept in signature for compatibility but unused)
    solve(  d_A, d_A_T,
            A.num_rows, A.num_cols,
            A.num_rows, A.num_cols,
            shared_size_in_bytes,
            tile_width,
            &kernel_time_us);

    cudaError_t err_D2H = cudaMemcpy(A_T.elements, d_A_T, bytes_A_T, cudaMemcpyDeviceToHost);
    if (err_D2H != cudaSuccess){
        printf("CUDA memcpy D2H Error: %s\n", cudaGetErrorString(err_D2H));
        cudaFree(d_A);
        cudaFree(d_A_T);
        free(A_T.elements);
        return EXIT_FAILURE;
    }

    printf("Product Matrix of size (%u, %u) stored as output_4_CS25MTECH11015.csv...\n", A_T.num_rows, A_T.num_cols);
    matrix_write_to_csv_int32(&A_T, "public_test_cases/output_4_CS25MTECH11015.csv");
    printf("Kernel execution time: %u microseconds\n", kernel_time_us);
    // Write stats to output.txt
    FILE* fout = fopen("public_test_cases/output_4_CS25MTECH11015.txt", "w");
    if (fout) {
        unsigned int num_elements = A_T.num_rows * A_T.num_cols;
        fprintf(fout, "%u\n", num_elements);
        fprintf(fout, "%u\n", kernel_time_us);
        fclose(fout);
    }
    cudaFree(d_A);
    cudaFree(d_A_T);
    free(A.elements);
    free(A_T.elements);

    return EXIT_SUCCESS;
}
