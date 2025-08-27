#include "matrix.h"
#include "matrix_csv.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 16   // tile width and height (for shared memory)

typedef struct {
    unsigned int width;   // number of columns
    unsigned int height;  // number of rows
    int32_t *elements;      // linear row-major array: row * width + col
} Matrix;

// Kernel: Transpose input -> output using shared memory tiling
__global__
void matrix_transpose_tiled_dkernel(int32_t *output, Matrix *input, unsigned int tile_width) {
    // shared memory tile with +1 padding on x to avoid bank conflicts
    __shared__ int32_t tile[tile_width][tile_width + 1];

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
    if (global_row_in < input.height && global_col_in < input.width) {
        unsigned int idx_in = global_row_in * input.width + global_col_in;
        tile[local_row][local_col] = input.elements[idx_in];
    } else {
        tile[local_row][local_col] = 0.0f; // padding for out-of-bounds
    }

    __syncthreads();

    // -------------------------
    // 5) compute block origin in the output matrix
    //    (notice: blockIdx.x <-> blockIdx.y swapped)
    // -------------------------
    unsigned int block_start_col_out = blockIdx.y * blockDim.y;
    unsigned int block_start_row_out = blockIdx.x * blockDim.x;

    // -------------------------
    // 6) compute global coords in the output matrix
    // -------------------------
    unsigned int global_col_out = block_start_col_out + local_col;
    unsigned int global_row_out = block_start_row_out + local_row;

    // -------------------------
    // 7) store from shared memory transposed
    // -------------------------
    if (global_row_out < output.height && global_col_out < output.width) {
        unsigned int idx_out = global_row_out * output.width + global_col_out;
        output.elements[idx_out] = tile[local_col][local_row];
    }
}

// Host-callable entry point
extern "C"

void solve(const int32_t* d_A, int32_t* d_A_T,
           unsigned int M, unsigned int N,
           unsigned int num_rows,
           unsigned int num_cols) {
    dim3 dimBlock(N, 1, 1);
    dim3 dimGrid(M, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrix_transpose_naive_dkernel<<<dimGrid, dimBlock>>>(d_A_T, d_A, num_rows, num_cols);
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

// Example host main (using Matrix helpers)
int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <matrix_a.csv> <matrix_b.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // input matrix csv filepath
    const char *matrix_A_csv_file_path = argv[1];
    Matrix A = matrix_read_from_csv_int32(matrix_A_csv_file_path);

    // ouput matrix_csv filepath
    const char *matrix_A_T_csv_file_path = argv[2];
    Matrix A_T;
    A_T.width = A.height;
    A_T.height = A.width;
    A_T.elements = (int32_t*)malloc(A_T.width * A_T.height * sizeof(int32_t));

    int32_t *d_A, *d_A_T;
    cudaError_t err_A = cudaMalloc(&d_A, A.width * A.height * sizeof(int32_t));
    
    if(err_A != cudaSuccess){
        printf("cudaMalloc Failed for d_A\n");
        return EXIT_FAILURE;
    }
    
    cudaError_t err_A_T = cudaMalloc(&d_A_T, A_T.width * A_T.height * sizeof(int32_t));

    if(err_A_T = cudaSuccess){
        printf("cudaMalloc Failed for d_A_T\n");
        return EXIT_FAILURE;
    }

    cudaError_t err_H2D= cudaMemcpy(d_A, A.elements,
                            A.width * A.height * sizeof(int32_t),
                            cudaMemcpyHostToDevice);

    if( err_H2D != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err_H2D));
        return EXIT_FAILURE;
    }

    // Call the C wrapper
    solve(d_A, d_A_T, A.height, A.width);

    cudaError_t err_D2H = cudaMemcpy(A_T.elements, d_A_T,
                            A_T.width * A_T.height * sizeof(int32_t),
                            cudaMemcpyDeviceToHost);

    if( err_D2H != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err_D2H));
        return EXIT_FAILURE;
    }

    matrix_write_to_csv_int32(&A_T, matrix_A_T_csv_file_path);

    cudaFree(d_A);
    cudaFree(d_A_T);
    free(A.elements);
    free(A_T.elements);

    return EXIT_SUCCESS;
}
