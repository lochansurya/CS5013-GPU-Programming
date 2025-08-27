#include "matrix.h"
#include "matrix_csv.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>

// CUDA kernel for matrix multiplication: C = A * B
// Destination/Result-First API
__global__ void matrix_multiplication_dkernel(
    uint32_t* C, const uint32_t* A, const uint32_t* B, int num_rows, int N, int num_cols) 
{   
    // Compute the Position of the Block in the Grid
    unsigned int bid_x = blockIdx.x * blockDim.x;
    unsigned int bid_y = blockIdx.y * blockDim.y;

    // Compute the Position of the Thread inside the block
    unsigned int tid_x = bid_x + threadIdx.x;
    unsigned int tid_y = bid_y + threadIdx.y;

    // Compute the row and the column you want to read from
    unsigned int row = tid_y;
    unsigned int col = tid_x;

    if(row < num_rows && col < num_cols) {
        uint32_t Cvalue = 0; //acuumulator
        for(unsigned int k = 0; k < N; ++k) {
            Cvalue += A[row * N + k] * B[k * num_cols + col];
        }
        C[row * num_cols + col] = Cvalue;
    }
}

// Host-callable function using Matrix structs and explicit thread/block dims
extern "C" void solve(uint32_t* d_C, const uint32_t* d_A, const uint32_t* d_B,
                      unsigned int grid_x,
                      unsigned int grid_y,
                      unsigned int block_x,
                      unsigned int block_y,
                      int M, int N, int K)
{
    dim3 num_threads_per_block(block_x, block_y, 1);
    dim3 num_blocks_per_grid(grid_x, grid_y, 1);

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //launch the CUDA kernel
    matrix_multiplication_dkernel<<<num_blocks_per_grid, num_threads_per_block>>>(d_C, d_A, d_B, M, N, K);
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


// ----------------------
// Test in host code
// ----------------------
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

    // Initialize matrices
    Matrix A = {0, 0, NULL};
    Matrix B = {0, 0, NULL};
    Matrix C = {0, 0, NULL};

    // Read matrices into the Matrix buffers, from the CLI input CSV filepaths
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
    solve(d_C, d_A, d_B, num_blocks_per_grid_x, num_blocks_per_grid_y, num_threads_per_block_x, num_threads_per_block_y, A.num_rows, A.num_cols, B.num_cols);

    // Copy result back to host
    cudaMemcpy(C.elements, d_C, C.num_rows * C.num_cols * sizeof(uint32_t), cudaMemcpyDeviceToHost);

    //print matrix
    print_matrix(&C);

    //Write the output matrix matrix_c.csv
    matrix_write_to_csv_int32(&C, "public_test_cases/matrix_c.csv");

    // have to check if the matrix multiplication is correct by running a diff between the output_matrix_mul.csv and matrix_c.csv under the public_test_cases directory
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(A.elements);
    free(B.elements);
    free(C.elements);

    return 0;
}
