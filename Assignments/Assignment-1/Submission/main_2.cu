//Modify the previous (basic version) kernel to use Shared Memory for fetching the operands from //Global Memory.
//The tile sizes should be dynamically configurable (not #defines) via command-line arguments.
//Increase the tile size in steps till the maximum possible value and observe the corresponding change in execution time.
//You can tune and choose optimal kernel launch parameters (block sizes) and assign them in the code.
#include "matrix.h"
#include "matrix_csv.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdint>

// CUDA kernel: Tiled matrix multiplication using shared memory
__global__ void matrix_multiplication_tiled_dkernel(
    int32_t* C, int32_t* A, int32_t* B,
    unsigned int M, unsigned int N, unsigned int K, unsigned int tile_width)
{
    extern __shared__ int32_t shm[]; // kernel-launch-time-configurable; Dynamic Memory Allocation of the Shared Memory
    int32_t* TILE_A = shm;
    int32_t* TILE_B = shm + tile_width * tile_width;

    // Block Coords in the Global Memory
    unsigned int bid_x = blockIdx.x * blockDim.x ;
    unsigned int bid_y = blockIdx.y * blockDim.y;

    // Thread Coords in the Global Memory
    unsigned int tid_x = bid_x+ threadIdx.x;
    unsigned int tid_y = bid_y + threadIdx.y;

    // Row and Col in the Global Memory Array
    unsigned int row_in_tile = threadIdx.y;
    unsigned int col_in_tile = threadIdx.x;

    int32_t tmp = 0; // accumulator
    unsigned int num_tiles = (N + tile_width - 1) / tile_width;

    for (unsigned int phase = 0; phase < num_tiles; ++phase)
    {
        unsigned int effective_col_inside_tile = phase * tile_width + col_in_tile;
        TILE_A[row_in_tile * tile_width + col_in_tile] =
            (tid_y < M && effective_col_inside_tile < N) ? A[tid_y * N + effective_col_inside_tile] : 0; // Halo Cells

        unsigned int effective_row_inside_tile = phase * tile_width + row_in_tile;
        TILE_B[row_in_tile * tile_width + col_in_tile] =
            (tid_x < K && effective_row_inside_tile < N) ? B[effective_row_inside_tile * K + tid_x] : 0; // Halo Cells

        __syncthreads(); // Barrier for Race Condition Avoidance between threads

        for (unsigned int k = 0; k < tile_width; ++k)
            tmp += TILE_A[row_in_tile * tile_width + k] *
                   TILE_B[k * tile_width + col_in_tile];

        __syncthreads();
    }

    if (tid_y < M && tid_x < K)
        C[tid_y * K + tid_x] = tmp;
}

// Host-callable function
extern "C" void solve(int32_t* d_C, int32_t* d_A, int32_t* d_B,
                      int M, int N, int K, unsigned int tile_width)
{
    dim3 block(tile_width, tile_width);
    dim3 grid((K + tile_width - 1) / tile_width, (M + tile_width - 1) / tile_width);
    size_t shared_size_in_bytes = 2 * tile_width * tile_width * sizeof(int32_t); // for the 2 different tiles A, B;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matrix_multiplication_tiled_dkernel<<<grid, block, shared_size_in_bytes>>>(
        d_C, d_A, d_B, M, N, K, tile_width);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %f microseconds\n", ms * 1000.0f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}


int main(int argc, char* argv[])
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <TILE_WIDTH> <matrix_a.csv> <matrix_b.csv>\n", argv[0]);
        return 1;
    }

    unsigned int tile_width = atoi(argv[1]);
    const char* matrix_A_file_path = argv[2];
    const char* matrix_B_file_path = argv[3];

    Matrix A;
    Matrix B;
    
    matrix_read_from_csv_uint32(&A, matrix_A_file_path);
    matrix_read_from_csv_uint32(&B, matrix_B_file_path);

    if (A.num_cols != B.num_rows) {
        fprintf(stderr, "Error: Incompatible matrix dimensions\n");
        free(A.elements);
        free(B.elements);
        return 1;
    }

    Matrix C;
    C.num_rows = A.num_rows;
    C.num_cols = B.num_cols;
    C.elements = (int32_t*)malloc(C.num_rows * C.num_cols * sizeof(int32_t));

    int32_t *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, A.num_rows * A.num_cols * sizeof(int32_t));
    cudaMalloc(&d_B, B.num_rows * B.num_cols * sizeof(int32_t));
    cudaMalloc(&d_C, C.num_rows * C.num_cols * sizeof(int32_t));

    cudaMemcpy(d_A, A.elements, A.num_rows * A.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B.elements, B.num_rows * B.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);

    solve(d_C, d_A, d_B, A.num_rows, A.num_cols, B.num_cols, tile_width);

    cudaMemcpy(C.elements, d_C, C.num_rows * C.num_cols * sizeof(int32_t), cudaMemcpyDeviceToHost);


    // printf("====================\n");
    // printf("Printing the matrix...\n");
    // print_matrix_uint32(&C);
    // printf("====================\n");

    printf("Product Matrix of size (%u, %u) stored as matrix_c.csv output_2_CS25MTECH11015.csv...\n", C.num_rows, C.num_cols);
    matrix_write_to_csv_uint32(&C, "output_2_CS25MTECH11015.csv");


    free(A.elements);
    free(B.elements);
    free(C.elements);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
