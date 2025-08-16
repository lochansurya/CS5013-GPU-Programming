#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TILE_HEIGHT 16
#define TILE_WIDTH 16

typedef struct {
    unsigned int width;
    unsigned int height;
    float *elements;
}Matrix;

__global__ void matmatmul_tiled_kernel(Matrix A, Matrix B, Matrix C){
    __shared__ float TILE_A[TILE_HEIGHT][TILE_WIDTH];
    __shared__ float TILE_B[TILE_HEIGHT][TILE_WIDTH]; 
    
    // Compute the Block Coordinates in the Grid
    unsigned int bid_x = blockIdx.x * blockDim.x;
    unsigned int bid_y = blockIdx.y * blockDim.y;

    //Compute the Thread coordinates in the Grid
    unsigned int tid_x = bid_x + threadIdx.x;
    unsigned int tid_y = bid_y + threadIdx.y;

    //Compute the row and the column in the Global Input Matrices
    unsigned int row = tid_y;
    unsigned int col = tid_x;
    
    //Compute the row and the column inside the tile matrix
    unsigned int row_in_tile = threadIdx.y;
    unsigned int col_in_tile = threadIdx.x;

    //Declare the temporary variable to store the partial result ofthe matrix multiplication
    float tmp = 0.0f;

    unsigned int num_tiles = (A.width + TILE_WIDTH -1) / TILE_WIDTH;

    for(unsigned int phase = 0; phase < num_tiles; ++phase){
        // Load in TILE_A data elements to work on, from the global memory i.e, A, to the shared memory, i.e, TILE_A
        unsigned int effective_col_inside_tile = phase * TILE_WIDTH + col_in_tile; //w.r.t the Global Matrix A
        if(row < A.height && (effective_col_inside_tile < A.width))
            TILE_A[row_in_tile][col_in_tile] = A.elements[row * A.width + effective_col_inside_tile]; 
        else 
            TILE_A[row_in_tile][col_in_tile] = 0.0f;
        // Load in TILE_B data elements to work on, from the global memory i.e, B, to the shared memory, i.e, TILE_B

        unsigned int effective_row_inside_tile = phase * TILE_HEIGHT + row_in_tile; //w.r.t the Global Matrix B
        if(col < B.width && (effective_row_inside_tile < B.height))
            TILE_B[row_in_tile][col_in_tile] = B.elements[effective_row_inside_tile * B.width + col]; 
        else 
            TILE_B[row_in_tile][col_in_tile] = 0.0f;
        __syncthreads();

        for(unsigned int k = 0; k < TILE_WIDTH; ++k){
            tmp += TILE_A[row_in_tile][k] * TILE_B[k][col_in_tile];
        }

        __syncthreads();
   }
   
    //Write the partial result to the global input array C
    if(row < C.height && col < C.width)
        C.elements[row * C.width + col] = tmp;
}
int main(void){
// Matrix dimensions
    const int M = 32;  // Rows of A and C
    const int K = 32;  // Cols of A / Rows of B
    const int N = 32;  // Cols of B and C

    // Host allocation
    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    // Initialize matrices
    for(int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for(int i = 0; i < K * N; i++) h_B[i] = 1.0f;
    for(int i = 0; i < M * N; i++) h_C[i] = 0.0f;

    // Device allocation
    Matrix d_A, d_B, d_C;
    d_A.width = K; d_A.height = M;
    d_B.width = N; d_B.height = K;
    d_C.width = N; d_C.height = M;

    cudaMalloc((void**)&d_A.elements, M * K * sizeof(float));
    cudaMalloc((void**)&d_B.elements, K * N * sizeof(float));
    cudaMalloc((void**)&d_C.elements, M * N * sizeof(float));

    // Copy data from host → device
    cudaMemcpy(d_A.elements, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B.elements, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // Kernel launch config
    dim3 dimBlock(TILE_WIDTH, TILE_HEIGHT);
    dim3 dimGrid((N + TILE_WIDTH - 1) / TILE_WIDTH,
                 (M + TILE_HEIGHT - 1) / TILE_HEIGHT);

    // Launch kernel
    matmatmul_tiled_kernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(h_C, d_C.elements, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify result (since A and B are filled with 1, result should be = K)
    bool correct = true;
    for(int i = 0; i < M * N; i++){
        if(h_C[i] != (float)K){
            correct = false;
            printf("Mismatch at index %d: %f != %d\n", i, h_C[i], K);
            break;
        }
    }

    if(correct) printf("Matrix multiplication PASSED ✅\n");
    else        printf("Matrix multiplication FAILED ❌\n");

    // Free memory
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

    return 0;
}
