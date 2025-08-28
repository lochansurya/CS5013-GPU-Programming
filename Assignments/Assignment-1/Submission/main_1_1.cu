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
        if (idx >= total) return;

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
                      int M, int N, int K)
{
    dim3 num_threads_per_block(block_x, 1, 1);
    dim3 num_blocks_per_grid(grid_x, 1, 1);

    // compute work_per_thread based on matrix size and launch config
    int total_outputs = M * K;
    int total_threads = grid_x * block_x;
    int work_per_thread = (total_outputs + total_threads - 1) / total_threads; // ceil division

    // timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    //launch the CUDA kernel
    matmul_1d_dkernel<<<num_blocks_per_grid, num_threads_per_block>>>(d_C, d_A, d_B, M, N, K, work_per_thread);
    cudaEventRecord(stop);

	// CUDA Event Handling for Profiling; SIGNAL-based; Interrupt-based;
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %f microseconds \n", ms * 1000.0f);

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
    matrix_read_from_csv_uint32(&A, matrix_A_file_path);
    matrix_read_from_csv_uint32(&B, matrix_B_file_path);

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
    printf("shape(C=AB) = (%u, %u)\n", C.num_rows, C.num_cols);
    C.elements = (int32_t*)malloc(C.num_rows * C.num_cols * sizeof(int32_t));

    // Allocate device memory
    int32_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.num_rows * A.num_cols * sizeof(int32_t));
    cudaMalloc(&d_B, B.num_rows * B.num_cols * sizeof(int32_t));
    cudaMalloc(&d_C, C.num_rows * C.num_cols * sizeof(int32_t));

    // Copy host data to device
    cudaMemcpy(d_A, A.elements, A.num_rows * A.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.elements, B.num_rows * B.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);

    // Launch kernel
    solve(d_C, d_A, d_B, num_blocks_per_grid_x, num_threads_per_block_x,
          A.num_rows, A.num_cols, B.num_cols);

    // Copy result back to host
    cudaMemcpy(C.elements, d_C, C.num_rows * C.num_cols * sizeof(int32_t), cudaMemcpyDeviceToHost);

    //print matrix
    // printf("==============\n");
    // printf("Printing the output matrix...\n");
    // print_matrix_uint32(&C);
    // printf("==============\n");

    //Write the output matrix matrix_c.csv
    printf("Product Matrix of size (%u, %u) stored as output_1_1_CS25MTECH11015.csv...\n", C.num_rows, C.num_cols);
    matrix_write_to_csv_uint32(&C, "output_1_1_CS25MTECH11015.csv");
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
