#include "matrix.h"
#include "matrix_csv.h"
#include <stdio.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>

// 1D coarsened transpose kernel
__global__ void transpose_1d_dkernel(int32_t* A_T, const int32_t* A,
                                     unsigned int num_rows, unsigned int num_cols,
                                     int work_per_thread)
{
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = num_rows * num_cols;

    for(int w = 0; w < work_per_thread; ++w){
        unsigned int idx = tid + w * gridDim.x * blockDim.x;
        if(idx >= total) break;

        unsigned int row = idx / num_cols;
        unsigned int col = idx % num_cols;

        A_T[col * num_rows + row] = A[row * num_cols + col];
    }
}

// Host-callable function
extern "C" void solve(int32_t* d_A_T, const int32_t* d_A,
                      unsigned int grid_x, unsigned int block_x,
                      unsigned int num_rows, unsigned int num_cols,
                      unsigned int* kernel_time)
{
    int total_threads = grid_x * block_x;
    int work_per_thread = (num_rows * num_cols + total_threads - 1) / total_threads;
    if(work_per_thread == 0) work_per_thread = 1;

    dim3 threads(block_x, 1, 1);
    dim3 blocks(grid_x, 1, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    transpose_1d_dkernel<<<blocks, threads>>>(d_A_T, d_A, num_rows, num_cols, work_per_thread);
    cudaEventRecord(stop);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess){
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    *kernel_time = (unsigned int)(ms * 1000.0f); // microseconds

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Host test code
int main(int argc, char* argv[]){
    if(argc < 4){
        fprintf(stderr, "Usage: %s <grid_x> <block_x> <path_to_matrix_a.csv>\n", argv[0]);
        return EXIT_FAILURE;
    }

    unsigned int grid_x = atoi(argv[1]);
    unsigned int block_x = atoi(argv[2]);
    const char* matrix_A_file = argv[3];

    Matrix A = {0, 0, NULL};
    Matrix A_T = {0, 0, NULL};

    matrix_read_from_csv_int32(&A, matrix_A_file);

    printf("Shape(A) = (%u, %u)\n", A.num_rows, A.num_cols);


    A_T.num_rows = A.num_cols;
    A_T.num_cols = A.num_rows;
    A_T.elements = (int32_t*)malloc(A_T.num_rows * A_T.num_cols * sizeof(int32_t));

    int32_t *d_A, *d_A_T;
    cudaMalloc(&d_A, A.num_rows * A.num_cols * sizeof(int32_t));
    cudaMalloc(&d_A_T, A_T.num_rows * A_T.num_cols * sizeof(int32_t));

    cudaMemcpy(d_A, A.elements, A.num_rows * A.num_cols * sizeof(int32_t), cudaMemcpyHostToDevice);

    unsigned int kernel_time_us = 0;
    solve(d_A_T, d_A, grid_x, block_x, A.num_rows, A.num_cols, &kernel_time_us);

    cudaMemcpy(A_T.elements, d_A_T, A_T.num_rows * A_T.num_cols * sizeof(int32_t), cudaMemcpyDeviceToHost);

    matrix_write_to_csv_int32(&A_T, "public_test_cases/output_3_CS25MTECH11015.csv");

    unsigned int num_elements = A_T.num_rows * A_T.num_cols;
    printf("Transposed matrix of size %u stored as output_3_CS25MTECH11015.csv\n", num_elements);
    printf("Kernel execution time: %.u microseconds\n", kernel_time_us);

    // write the stats to the output
    FILE* fout = fopen("output_3_CS25MTECH11015.txt", "w");
    if(fout){
        unsigned int num_elements = A_T.num_rows * A_T.num_cols;
        fprintf(fout, "%u\n", num_elements);
        fprintf(fout, "%u\n", kernel_time_us);
        fclose(fout);
    }

    cudaFree(d_A);
    cudaFree(d_A_T);
    free(A.elements);
    free(A_T.elements);

    return EXIT_SUCCESS;
}
