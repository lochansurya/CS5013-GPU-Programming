#include "matrix.h"
#include "matrix_csv.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>

// Unified kernel: supports thread coarsening
__global__ void matrix_multiplication_dkernel(
    int32_t* d_C, const int32_t* d_A, const int32_t* d_B,
    int M, int N, int K,
    unsigned int work_per_thread_row, unsigned int work_per_thread_col)
{
    unsigned int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    for(unsigned int i = 0; i < work_per_thread_row; ++i){
        unsigned int row = tid_y + i * blockDim.y * gridDim.y;
        if(row >= (unsigned int)M) break;

        for(unsigned int j = 0; j < work_per_thread_col; ++j){
            unsigned int col = tid_x + j * blockDim.x * gridDim.x;
            if(col >= (unsigned int)K) break;

            int32_t tmp = 0;
            for(unsigned int k = 0; k < (unsigned int)N; ++k)
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
                      int M, int N, int K,
                    unsigned int *kernel_time_us)
{
    dim3 num_threads_per_block(block_x, block_y, 1);
    dim3 num_blocks_per_grid(grid_x, grid_y, 1);

    unsigned int total_threads_x = grid_x * block_x;
    unsigned int total_threads_y = grid_y * block_y;

    // Ensure at least 1
    unsigned int work_per_thread_row = (M + total_threads_y - 1) / total_threads_y;
    if(work_per_thread_row == 0) work_per_thread_row = 1;
    unsigned int work_per_thread_col = (K + total_threads_x - 1) / total_threads_x;
    if(work_per_thread_col == 0) work_per_thread_col = 1;

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrix_multiplication_dkernel<<<num_blocks_per_grid, num_threads_per_block>>>(
        d_C, d_A, d_B, M, N, K, work_per_thread_row, work_per_thread_col);
    cudaEventRecord(stop);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    *kernel_time_us = (unsigned int)(ms * 1000.f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
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

    matrix_read_from_csv_int32(&A, matrix_A_file_path);
    matrix_read_from_csv_int32(&B, matrix_B_file_path);

    if(A.num_cols != B.num_rows){
        printf("Wrong Shapes of the Input Matrices\n");
        return 0;
    } else {
        printf("Shape(A) = (%u, %u)\n", A.num_rows, A.num_cols);
        printf("Shape(B) = (%u, %u)\n", B.num_rows, B.num_cols);
    }

    C.num_rows = A.num_rows;
    C.num_cols = B.num_cols;
    printf("Shape(C) = (%u, %u)\n", C.num_rows, C.num_cols);
    C.elements = (int32_t*)malloc(C.num_rows * C.num_cols * sizeof(int32_t));
    if(!C.elements){
        perror("Host memory allocation failed");
        return 1;
    }

    int32_t *d_A, *d_B, *d_C;
    if(cudaMalloc(&d_A, A.num_rows * A.num_cols * sizeof(int32_t)) != cudaSuccess ||
       cudaMalloc(&d_B, B.num_rows * B.num_cols * sizeof(int32_t)) != cudaSuccess ||
       cudaMalloc(&d_C, C.num_rows * C.num_cols * sizeof(int32_t)) != cudaSuccess){
        perror("Device memory allocation failed");
        return 1;
    }

    cudaMemcpy(d_A, A.elements, A.num_rows * A.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.elements, B.num_rows * B.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);
    
    unsigned int kernel_time_us = 0;
    solve(d_C, d_A, d_B, num_blocks_per_grid_x, num_blocks_per_grid_y,
          num_threads_per_block_x, num_threads_per_block_y,
          A.num_rows, A.num_cols, B.num_cols, &kernel_time_us);

    cudaMemcpy(C.elements, d_C, C.num_rows * C.num_cols * sizeof(int32_t), cudaMemcpyDeviceToHost);


    printf("Kernel execution time: %u microseconds\n", kernel_time_us);
    printf("Matrix of size %u stored as output_1_2_CS25MTECH11015.csv\n",
           C.num_rows*C.num_cols);
    matrix_write_to_csv_int32(&C, "output_1_2_CS25MTECH11015.csv");

    // writing to the output to the .txt file
    FILE* fout = fopen("output_1_2_CS25MTECH11015.txt", "w");
    if (fout) {
        unsigned int num_elements = C.num_rows * C.num_cols;
        fprintf(fout, "Line1: %u\n", num_elements);
        fprintf(fout, "Line2: %u\n", kernel_time_us);
        fclose(fout);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    free(A.elements);
    free(B.elements);
    free(C.elements);

    return 0;
}
