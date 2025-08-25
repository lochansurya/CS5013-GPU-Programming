
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 16   // tile width and height (for shared memory)

typedef struct {
    unsigned int width;   // number of columns
    unsigned int height;  // number of rows
    float *elements;      // linear row-major array: row * width + col
} Matrix;

// Kernel: Transpose input -> output naive 
__global__
void matrix_transpose_kernel(Matrix input, Matrix output) {
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

    
}
   int main(void) {
    const unsigned int M = 32; // rows
    const unsigned int N = 32; // cols

    // host allocation
    float *h_in  = (float*)malloc(M * N * sizeof(float));
    float *h_out = (float*)malloc(M * N * sizeof(float));

    // initialize input matrix with row*width+col
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            h_in[r * N + c] = (float)(r * N + c);
        }
    }

    // device allocation
    Matrix d_in, d_out;
    d_in.width = N; d_in.height = M;
    d_out.width = M; d_out.height = N;

    cudaMalloc((void**)&d_in.elements, M * N * sizeof(float));
    cudaMalloc((void**)&d_out.elements, M * N * sizeof(float));

    cudaMemcpy(d_in.elements, h_in, M * N * sizeof(float), cudaMemcpyHostToDevice);

    // launch transpose kernel
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x,
                 (M + dimBlock.y - 1) / dimBlock.y);

    matrix_transpose_kernel<<<dimGrid, dimBlock>>>(d_in, d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out.elements, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify transpose
    bool correct = true;
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < N; ++c) {
            float expected = h_in[r * N + c];
            float got = h_out[c * M + r];
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

