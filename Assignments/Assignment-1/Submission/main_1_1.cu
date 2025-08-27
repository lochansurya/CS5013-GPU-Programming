// naive matrix multiplication using 1D grid dimensions
#include "matrix.h"
#include "matrix_csv.h"
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void matmul_1d_dkernel(uint32_t* C, const uint32_t* A, const uint32_t* B,
                                                 int M, int N, int K){
    // 1D thread id
    unsigned int bid = blockIdx.x * blockDim.x;
    unsigned int tid = bid + threadIdx.x;
    unsigned int total = M * K;
    if (tid >= total) return;

    unsigned int row = tid / K;
    unsigned int col = tid % K;

    uint32_t Cvalue = 0;
    for (unsigned int k = 0; k < (unsigned int)N; ++k) {
        Cvalue += A[row * N + k] * B[k * K + col];
    }
    C[row * K + col] = Cvalue;
}

// Host-callable function using raw device pointers and explicit thread/block dims
extern "C" void solve(uint32_t* d_C, const uint32_t* d_A, const uint32_t* d_B,
                      unsigned int grid_x,
                      unsigned int block_x,
                      int M, int N, int K)
{
    dim3 num_threads_per_block(block_x, 1, 1);
    dim3 num_blocks_per_grid(grid_x, 1, 1);

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //launch the CUDA kernel
    matmul_1d_dkernel<<<num_blocks_per_grid, num_threads_per_block>>>(d_C, d_A, d_B, M, N, K);
    cudaEventRecord(stop);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel elapsed time: %f us \n", ms * 1000.0f);

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

    // Initialize matrices
    Matrix A = {0, 0, NULL};
    Matrix B = {0, 0, NULL};
    Matrix C = {0, 0, NULL};

    // Read matrices into the Matrix buffers
    matrix_read_from_csv_int32(&A, matrix_A_file_path);
    matrix_read_from_csv_int32(&B, matrix_B_file_path);

    //check the shapes of the matrices read
    if(A.num_cols != B.num_rows){
        printf("Wrong Shapes of the Input Matrices\n");
        return 0;
    }else{
        printf("Shape(A) = (%u, %u)\n", A.num_rows, A.num_cols);
        printf("Shape(B) = (%u, %u)\n", B.num_rows, B.num_cols);
    }

    // Allocate C matrix
    C.num_rows = A.num_rows;
    C.num_cols = B.num_cols;
    printf("shape(C) = (%u, %u)", C.num_rows, C.num_cols);
    C.elements = (uint32_t*)malloc(C.num_rows * C.num_cols * sizeof(uint32_t));

    // Allocate device memory
    uint32_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.num_rows * A.num_cols * sizeof(uint32_t));
    cudaMalloc(&d_B, B.num_rows * B.num_cols * sizeof(uint32_t));
    cudaMalloc(&d_C, C.num_rows * C.num_cols * sizeof(uint32_t));

    // Copy host data to device
    cudaMemcpy(d_A, A.elements, A.num_rows * A.num_cols * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.elements, B.num_rows * B.num_cols * sizeof(uint32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    solve(d_C, d_A, d_B, num_blocks_per_grid_x, num_threads_per_block_x,
          A.num_rows, A.num_cols, B.num_cols);

    // Copy result back to host
    cudaMemcpy(C.elements, d_C, C.num_rows * C.num_cols * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    //print matrix
    print_matrix(&C);

    //Write the output matrix matrix_c.csv
    matrix_write_to_csv_int32(&C, "public_test_cases/matrix_c.csv");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(A.elements);
    free(B.elements);
    free(C.elements);

    return EXIT_SUCCESS;
}
