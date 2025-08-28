#include "matrix.h"
#include "matrix_csv.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>

// Regular kernel: each thread computes one element
__global__ void matrix_multiplication_regular_dkernel(
    int32_t* d_C, const int32_t* d_A, const int32_t* d_B, int M, int N, int K)
{
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row >= M || col >= K) return;

    int32_t tmp = 0;
    for(unsigned int k = 0; k < N; ++k)
        tmp += d_A[row * N + k] * d_B[k * K + col];

    d_C[row * K + col] = tmp;
}

// Thread coarsened kernel
__global__ void matrix_multiplication_coarsened_dkernel(
    int32_t* d_C, const int32_t* d_A, const int32_t* d_B, int M, int N, int K,
    unsigned int work_per_thread_row, unsigned int work_per_thread_col)
{   
    unsigned int bid_x = blockIdx.x * blockDim.x;
    unsigned int bid_y = blockIdx.y * blockDim.y;

    unsigned int tid_x = bid_x + threadIdx.x;
    unsigned int tid_y = bid_y + threadIdx.y;

    for(unsigned int i = 0; i < work_per_thread_row; ++i){
        unsigned int row = tid_y + i * blockDim.y * gridDim.y;
        if(row >= M) break;

        for(unsigned int j = 0; j < work_per_thread_col; ++j){
            unsigned int col = tid_x + j * blockDim.x * gridDim.x;
            if(col >= K) break;

            int32_t tmp = 0;
            for(unsigned int k = 0; k < N; ++k)
                tmp += d_A[row * N + k] * d_B[k * K + col];

            d_C[row * K + col] = tmp;
        }
    }
}

extern "C" void solve(int32_t* d_C, const int32_t* d_A, const int32_t* d_B,
                      unsigned int grid_x,
                      unsigned int grid_y,
                      unsigned int block_x,
                      unsigned int block_y,
                      int M, int N, int K)
{
    dim3 num_threads_per_block(block_x, block_y, 1);
    dim3 num_blocks_per_grid(grid_x, grid_y, 1);

    unsigned int total_threads_x = grid_x * block_x;
    unsigned int total_threads_y = grid_y * block_y;

    unsigned int work_per_thread_row = (M + total_threads_y - 1) / total_threads_y;
    unsigned int work_per_thread_col = (K + total_threads_x - 1) / total_threads_x;

    // Timing events
    cudaEvent_t start_regular, stop_regular;
    cudaEvent_t start_coarse, stop_coarse;
    cudaEventCreate(&start_regular);
    cudaEventCreate(&stop_regular);
    cudaEventCreate(&start_coarse);
    cudaEventCreate(&stop_coarse);

    if(work_per_thread_row == 1 && work_per_thread_col == 1){
        // Regular kernel timing
        cudaEventRecord(start_regular);
        matrix_multiplication_regular_dkernel<<<num_blocks_per_grid, num_threads_per_block>>>(
            d_C, d_A, d_B, M, N, K);
        cudaEventRecord(stop_regular);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }

        cudaEventSynchronize(stop_regular);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_regular, stop_regular);
        printf("Kernel execution time: %f microseconds\n", ms * 1000.0f);

    } else {
        // Coarsened kernel timing
        cudaEventRecord(start_coarse);
        matrix_multiplication_coarsened_dkernel<<<num_blocks_per_grid, num_threads_per_block>>>(
            d_C, d_A, d_B, M, N, K, work_per_thread_row, work_per_thread_col);
        cudaEventRecord(stop_coarse);

        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }

        cudaEventSynchronize(stop_coarse);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_coarse, stop_coarse);
        printf("Kernel execution time: %f microseconds\n", ms * 1000.0f);
    }

    // Destroy events
    cudaEventDestroy(start_regular);
    cudaEventDestroy(stop_regular);
    cudaEventDestroy(start_coarse);
    cudaEventDestroy(stop_coarse);
}


// Host test code
int main(int argc, char* argv[]) {
    if(argc < 7){
        fprintf(stderr, "Usage: ./matmul_2d <X1> <Y1> <X2> <Y2> <path to matrix_a.csv> <path_to_matrix_b.csv>\n");
        return 1;
    }
    
    unsigned int num_blocks_per_grid_x = atoi(argv[1]);
    unsigned int num_blocks_per_grid_y = atoi(argv[2]);
    unsigned int num_threads_per_block_x = atoi(argv[3]);
    unsigned int num_threads_per_block_y = atoi(argv[4]);
    const char *matrix_A_file_path = argv[5];
    const char *matrix_B_file_path = argv[6];

    Matrix A = {0, 0, NULL};
    Matrix B = {0, 0, NULL};
    Matrix C = {0, 0, NULL};

    matrix_read_from_csv_uint32(&A, matrix_A_file_path);
    matrix_read_from_csv_uint32(&B, matrix_B_file_path);

    if(A.num_cols != B.num_rows){
        printf("Wrong Shapes of the Input Matrices\n");
        return 0;
    }else{
        printf("Shape(A) = (%u, %u)\n", A.num_rows, A.num_cols);
        printf("Shape(B) = (%u, %u)\n", B.num_rows, B.num_cols);
    }

    C.num_rows = A.num_rows;
    C.num_cols = B.num_cols;
    printf("shape(C) = (%u, %u)\n", C.num_rows, C.num_cols);
    C.elements = (int32_t*)malloc(C.num_rows * C.num_cols * sizeof(int32_t));

    int32_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.num_rows * A.num_cols * sizeof(int32_t));
    cudaMalloc(&d_B, B.num_rows * B.num_cols * sizeof(int32_t));
    cudaMalloc(&d_C, C.num_rows * C.num_cols * sizeof(int32_t));

    cudaMemcpy(d_A, A.elements, A.num_rows * A.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.elements, B.num_rows * B.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);

    solve(d_C, d_A, d_B, num_blocks_per_grid_x, num_blocks_per_grid_y,
          num_threads_per_block_x, num_threads_per_block_y,
          A.num_rows, A.num_cols, B.num_cols);

    cudaMemcpy(C.elements, d_C, C.num_rows * C.num_cols * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // printf("====================\n");
    // printf("Printing the matrix...\n");
    // print_matrix_uint32(&C);
    // printf("====================\n");

    //Write the output matrix to the csv file
    printf("Product Matrix of size (%u, %u) stored as matrix_c.csv public_test_cases/output_1_2_CS25MTECH11015.csv...\n", C.num_rows, C.num_cols);
    matrix_write_to_csv_uint32(&C, "public_test_cases/output_1_2_CS25MTECH11015.csv");


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A.elements);
    free(B.elements);
    free(C.elements);

    return 0;
}
