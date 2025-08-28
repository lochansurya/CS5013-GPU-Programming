#include "matrix.h"
#include "matrix_csv.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel: Transpose input -> output using shared memory tiling
__global__
void matrix_transpose_tiled_dkernel(uint32_t *d_output, uint32_t *d_input,
                                    unsigned int num_rows, unsigned int num_cols,
                                    unsigned int tile_width) {
    extern __shared__ uint32_t TILE[];

    // -------------------------
    // 1) block origin in global coords
    // -------------------------
    unsigned int block_start_col = blockIdx.x * blockDim.x;
    unsigned int block_start_row = blockIdx.y * blockDim.y;

    // -------------------------
    // 2) local thread coords
    // -------------------------
    unsigned int local_col = threadIdx.x;
    unsigned int local_row = threadIdx.y;

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
        TILE[local_row * tile_width + local_col] = d_input[idx_in];
    } else {
        TILE[local_row * tile_width + local_col] = 0;
    }

    __syncthreads();

    // -------------------------
    // 5) compute block origin in output
    // -------------------------
    unsigned int block_start_col_out = blockIdx.y * blockDim.y;
    unsigned int block_start_row_out = blockIdx.x * blockDim.x;

    // -------------------------
    // 6) compute global coords in output
    // -------------------------
    unsigned int global_col_out = block_start_col_out + local_col;
    unsigned int global_row_out = block_start_row_out + local_row;

    // -------------------------
    // 7) store from shared memory transposed
    // -------------------------
    if (global_row_out < num_cols && global_col_out < num_rows) {
        unsigned int idx_out = global_row_out * num_rows + global_col_out;
        d_output[idx_out] = TILE[local_col * tile_width + local_row];
    }
}

// Host-callable entry point
extern "C"
void solve(const uint32_t* d_A, uint32_t* d_A_T,
           unsigned int M, unsigned int N,
           unsigned int tile_width) {
    dim3 dimBlock(tile_width, tile_width, 1);
    dim3 dimGrid((N + tile_width - 1) / tile_width,
                 (M + tile_width - 1) / tile_width, 1);

    size_t shm_size = tile_width * tile_width * sizeof(uint32_t);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrix_transpose_tiled_dkernel<<<dimGrid, dimBlock, shm_size>>>(d_A_T, (uint32_t*)d_A,
                                                                    M, N, tile_width);
    cudaEventRecord(stop);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel elapsed time: %f us \n", ms * 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <tile_width> <matrix_a.csv> <matrix_a_t.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    unsigned int tile_width = (unsigned int)atoi(argv[1]);
    const char *matrix_A_csv_file_path = argv[2];
    const char *matrix_A_T_csv_file_path = argv[3];

    Matrix A;
    matrix_read_from_csv_uint32(&A, matrix_A_csv_file_path);

    Matrix A_T;
    A_T.num_cols = A.num_rows;
    A_T.num_rows = A.num_cols;
    A_T.elements = (uint32_t*)malloc(A_T.num_cols * A_T.num_rows * sizeof(uint32_t));

    uint32_t *d_A, *d_A_T;
    cudaError_t err_A = cudaMalloc(&d_A, A.num_cols * A.num_rows * sizeof(uint32_t));
    if(err_A != cudaSuccess){
        printf("cudaMalloc Failed for d_A\n");
        return EXIT_FAILURE;
    }
    cudaError_t err_A_T = cudaMalloc(&d_A_T, A_T.num_cols * A_T.num_rows * sizeof(uint32_t));
    if(err_A_T != cudaSuccess){
        printf("cudaMalloc Failed for d_A_T\n");
        return EXIT_FAILURE;
    }

    cudaError_t err_H2D= cudaMemcpy(d_A, A.elements,
                            A.num_cols * A.num_rows * sizeof(uint32_t),
                            cudaMemcpyHostToDevice);
    if( err_H2D != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err_H2D));
        return EXIT_FAILURE;
    }

    // Call the wrapper
    solve(d_A, d_A_T, A.num_rows, A.num_cols, tile_width);

    cudaError_t err_D2H = cudaMemcpy(A_T.elements, d_A_T,
                            A_T.num_cols * A_T.num_rows * sizeof(uint32_t),
                            cudaMemcpyDeviceToHost);
    if( err_D2H != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err_D2H));
        return EXIT_FAILURE;
    }

    printf("==============================\n");
    print_matrix_uint32(&A_T);
    printf("==============================\n\n");

    printf("==============================\n");
    printf("Writing the transposed matrix to public_test_cases/matrix_A_T.csv...\n");
    matrix_write_to_csv_uint32(&A_T, matrix_A_T_csv_file_path);
    printf("==============================\n");

    cudaFree(d_A);
    cudaFree(d_A_T);
    free(A.elements);
    free(A_T.elements);

    return EXIT_SUCCESS;
}
