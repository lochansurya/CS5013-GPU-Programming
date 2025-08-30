// naive matrix multiplication using 1D grid dimensions
#include "matrix.h"
#include "matrix_csv.h"
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matmul_1d_dkernel(int32_t* C, const int32_t* A, const int32_t* B,
                                  int M, int N, int K, int work_per_thread){
    // 1D thread id
    unsigned int bid = blockIdx.x * blockDim.x;
    unsigned int tid = bid + threadIdx.x;
    unsigned int total = M * K;

    // starting output index for this thread
    unsigned int start = tid * work_per_thread;

    // loop over multiple outputs per thread
    for(int w = 0; w < work_per_thread; ++w){
        unsigned int idx = start + w;
        if (idx >= total) break;

        unsigned int row = idx / K;
        unsigned int col = idx % K;

        int32_t Cvalue = 0;
        for (unsigned int k = 0; k < (unsigned int)N; ++k) {
            Cvalue += A[row * N + k] * B[k * K + col];
        }
        C[row * K + col] = Cvalue;
    }
}

// Host-callable function using raw device pointers and explicit thread/block dims
extern "C" void solve(int32_t* d_C, const int32_t* d_A, const int32_t* d_B,
                      unsigned int grid_x,
                      unsigned int block_x,
                      int M, int N, int K,
                      unsigned int *kernel_time)   // <---- added parameter
{
    dim3 num_threads_per_block(block_x, 1, 1);
    dim3 num_blocks_per_grid(grid_x, 1, 1);

    int total_outputs = M * K;
    int total_threads = grid_x * block_x;
    int work_per_thread = (total_outputs + total_threads - 1) / total_threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_1d_dkernel<<<num_blocks_per_grid, num_threads_per_block>>>(d_C, d_A, d_B, M, N, K, work_per_thread);
    cudaEventRecord(stop);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    *kernel_time = ms * 1000.0f;   // <---- microseconds

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char* argv[]) {
    if(argc < 5){
        fprintf(stderr, "Usage: ./matmul_1d <grid_x> <block_x> <path to matrix_a.csv> <path_to_matrix_b.csv>\n");
        return EXIT_FAILURE;
    }
    
    unsigned int num_blocks_per_grid_x = atoi(argv[1]);
    unsigned int num_threads_per_block_x = atoi(argv[2]);
    const char *matrix_A_file_path = argv[3];
    const char *matrix_B_file_path = argv[4];

    Matrix A = {0, 0, NULL};
    Matrix B = {0, 0, NULL};
    Matrix C = {0, 0, NULL};

    matrix_read_from_csv_int32(&A, matrix_A_file_path);
    matrix_read_from_csv_int32(&B, matrix_B_file_path);

    if(A.num_cols != B.num_rows){
        printf("Wrong Shapes of the Input Matrices\n");
        return 0;
    }else{
        printf("Shape(A) = (%u, %u)\n", A.num_rows, A.num_cols);
        printf("Shape(B) = (%u, %u)\n", B.num_rows, B.num_cols);
    }

    C.num_rows = A.num_rows;
    C.num_cols = B.num_cols;
    C.elements = (int32_t*)malloc(C.num_rows * C.num_cols * sizeof(int32_t));

    int32_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.num_rows * A.num_cols * sizeof(int32_t));
    cudaMalloc(&d_B, B.num_rows * B.num_cols * sizeof(int32_t));
    cudaMalloc(&d_C, C.num_rows * C.num_cols * sizeof(int32_t));

    cudaMemcpy(d_A, A.elements, A.num_rows * A.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.elements, B.num_rows * B.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);

    unsigned int kernel_time_us = 0.0f;
    solve(d_C, d_A, d_B, num_blocks_per_grid_x, num_threads_per_block_x,
          A.num_rows, A.num_cols, B.num_cols, &kernel_time_us);

    cudaMemcpy(C.elements, d_C, C.num_rows * C.num_cols * sizeof(int32_t), cudaMemcpyDeviceToHost);

    matrix_write_to_csv_int32(&C, "public_test_cases/output_1_1_CS25MTECH11015.csv");

    unsigned int num_elements = C.num_rows * C.num_cols;
    printf("Product Matrix of size %u stored as output_1_1_CS25MTECH11015.csv\n", num_elements);
    printf("Kernel execution time: %u microseconds\n", kernel_time_us);

    FILE* fout = fopen("public_test_cases/output_1_1_CS25MTECH11015.txt", "w");
    if (fout) {
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

    return EXIT_SUCCESS;
}
