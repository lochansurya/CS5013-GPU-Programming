#include "matrix_csv.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 16   // tile width and height (for shared memory)

typedef struct {
    unsigned int width;   // number of columns
    unsigned int height;  // number of rows
    int32_t *elements;      // linear row-major array: row * width + col
} Matrix;

// Kernel: Transpose input -> output using shared memory tiling
__global__
void transpose_tiled_kernel(Matrix input, Matrix output) {
    // shared memory tile with +1 padding on x to avoid bank conflicts
    __shared__ int32_t tile[TILE_DIM][TILE_DIM + 1];

    // -------------------------
    // 1) block origin in global coords (top-left corner of this block in the input matrix)
    // -------------------------
    unsigned int block_start_col = blockIdx.x * blockDim.x; // starting col in input for this block
    unsigned int block_start_row = blockIdx.y * blockDim.y; // starting row in input for this block

    // -------------------------
    // 2) local thread coords
    // -------------------------
    unsigned int local_col = threadIdx.x; // [0..blockDim.x-1]
    unsigned int local_row = threadIdx.y; // [0..blockDim.y-1]

    // -------------------------
    // 3) global thread coords in the input matrix
    // -------------------------
    unsigned int global_col_in = block_start_col + local_col;
    unsigned int global_row_in = block_start_row + local_row;

    // -------------------------
    // 4) load from input into shared memory
    // -------------------------
    if (global_row_in < input.height && global_col_in < input.width) {
        unsigned int idx_in = global_row_in * input.width + global_col_in;
        tile[local_row][local_col] = input.elements[idx_in];
    } else {
        tile[local_row][local_col] = 0.0f; // padding for out-of-bounds
    }

    __syncthreads();

    // -------------------------
    // 5) compute block origin in the output matrix
    //    (notice: blockIdx.x <-> blockIdx.y swapped)
    // -------------------------
    unsigned int block_start_col_out = blockIdx.y * blockDim.y;
    unsigned int block_start_row_out = blockIdx.x * blockDim.x;

    // -------------------------
    // 6) compute global coords in the output matrix
    // -------------------------
    unsigned int global_col_out = block_start_col_out + local_col;
    unsigned int global_row_out = block_start_row_out + local_row;

    // -------------------------
    // 7) store from shared memory transposed
    // -------------------------
    if (global_row_out < output.height && global_col_out < output.width) {
        unsigned int idx_out = global_row_out * output.width + global_col_out;
        output.elements[idx_out] = tile[local_col][local_row];
    }
}

int main(void) {
    const unsigned int M = 32; // rows
    const unsigned int N = 32; // cols

    // host allocation
    int32_t *h_in  = (int32_t*)malloc(M * N * sizeof(int32_t));
    int32_t *h_out = (int32_t*)malloc(M * N * sizeof(int32_t));

    // initialize input matrix with row*width+col
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            h_in[r * N + c] = (int32_t)(r * N + c);
        }
    }

    // device allocation
    Matrix d_in, d_out;
    d_in.width = N; d_in.height = M;
    d_out.width = M; d_out.height = N;

    cudaMalloc((void**)&d_in.elements, M * N * sizeof(int32_t));
    cudaMalloc((void**)&d_out.elements, M * N * sizeof(int32_t));

    cudaMemcpy(d_in.elements, h_in, M * N * sizeof(int32_t), cudaMemcpyHostToDevice);

    // launch transpose kernel
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (M + dimBlock.y - 1) / dimBlock.y);

    transpose_tiled_kernel<<<dimGrid, dimBlock>>>(d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out.elements, M * N * sizeof(int32_t), cudaMemcpyDeviceToHost);

    // Verify transpose
    bool correct = true;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            int32_t expected = h_in[r * N + c];
            int32_t got = h_out[c * M + r];
            if (expected != got) {
                correct = false;
                printf("Mismatch at input(%d,%d) -> output(%d,%d): %f != %f\n",
                       r, c, c, r, expected, got);
                goto end;
            }
        }
    }

end:
    if (correct) printf("Matrix transpose PASSED ✅\n");
    else         printf("Matrix transpose FAILED ❌\n");

    free(h_in); free(h_out);
    cudaFree(d_in.elements);
    cudaFree(d_out.elements);

    return 0;
}

