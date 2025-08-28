#include "matrix_csv.h"
#include "matrix.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel: transpose using 1D grid/block
__global__
void matrix_transpose_naive_dkernel(int32_t* output, const int32_t* input,
                                    unsigned int num_rows, unsigned int num_cols) {
    // 1) Global thread index
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 2) Total elements in the matrix
    unsigned int total_elements = num_rows * num_cols;

    // 3) Check bounds
    if(tid < total_elements) {
        // 4) Compute global row and column in input
        unsigned int global_row_in = tid / num_cols;
        unsigned int global_col_in = tid % num_cols;

        // 5) Compute global row and column in output (transposed)
        unsigned int global_row_out = global_col_in;
        unsigned int global_col_out = global_row_in;

        // 6) Write transposed value
        output[global_row_out * num_rows + global_col_out] = input[global_row_in * num_cols + global_col_in];
    }
}

// Host-callable entry point
extern "C"
void solve(int32_t* d_A, int32_t* d_A_T,
           unsigned int M, unsigned int N,
           unsigned int num_rows,
           unsigned int num_cols) {

    // 1D block and grid
    unsigned int threadsPerBlock = 256; // can vary
    unsigned int numBlocks = (num_rows * num_cols + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrix_transpose_naive_dkernel<<<numBlocks, threadsPerBlock>>>(d_A_T, d_A, num_rows, num_cols);
    cudaEventRecord(stop);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %f microseconds\n", 1000.f*ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <matrix_a.csv> <matrix_a_T.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    const char *matrix_A_csv_file_path = argv[1];

    Matrix A;
    matrix_read_from_csv_uint32(&A, matrix_A_csv_file_path);

    Matrix A_T;
    A_T.num_rows = A.num_cols;
    A_T.num_cols = A.num_rows;
    A_T.elements = (int32_t*)malloc(A_T.num_rows * A_T.num_cols * sizeof(int32_t));

    int32_t *d_A, *d_A_T;
    if(cudaMalloc(&d_A, A.num_rows * A.num_cols * sizeof(int32_t)) != cudaSuccess ||
       cudaMalloc(&d_A_T, A_T.num_rows * A_T.num_cols * sizeof(int32_t)) != cudaSuccess) {
        printf("cudaMalloc failed\n");
        return EXIT_FAILURE;
    }

    if(cudaMemcpy(d_A, A.elements, A.num_rows * A.num_cols * sizeof(int32_t),
                  cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("CUDA H2D copy failed\n");
        return EXIT_FAILURE;
    }

    solve(d_A, d_A_T, A.num_rows, A.num_cols, A.num_rows, A.num_cols);

    if(cudaMemcpy(A_T.elements, d_A_T, A_T.num_rows * A_T.num_cols * sizeof(int32_t),
                  cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("CUDA D2H copy failed\n");
        return EXIT_FAILURE;
    }

    // printf("=================================\n");
    // printf("Printing the Transposed Matrix...\n");
    // print_matrix_uint32(&A_T);
    // printf("=================================\n\n");

    printf("Product Matrix of size (%u, %u) stored as output_3_CS25MTECH11015.csv...\n", A_T.num_rows, A_T.num_cols);
    matrix_write_to_csv_uint32(&A_T, "output_3_CS25MTECH11015.csv");


    cudaFree(d_A);
    cudaFree(d_A_T);
    free(A.elements);
    free(A_T.elements);

    return EXIT_SUCCESS;
}
