#include "matrix_csv.h"
#include "matrix.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel: Transpose input -> output using raw pointers
__global__
void matrix_transpose_naive_dkernel(int32_t* output, int32_t* input,
                            unsigned int num_rows, unsigned int num_cols) {
    // 1) block origin in input
    unsigned int block_start_col = blockIdx.x * blockDim.x;
    unsigned int block_start_row = blockIdx.y * blockDim.y;

    // 2) local thread coords
    unsigned int tid_x = threadIdx.x;
    unsigned int tid_y = threadIdx.y;

    // 3) global input coords
    unsigned int global_col_in = block_start_col + tid_x;
    unsigned int global_row_in = block_start_row + tid_y;

    if (global_col_in < num_cols && global_row_in < num_rows) {
        // 4) global output coords
        unsigned int global_col_out = global_row_in;
        unsigned int global_row_out = global_col_out; 

        // 5) transpose write
        output[global_row_out * num_rows + global_col_out] = input[global_row_in * num_cols + global_col_in];
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
