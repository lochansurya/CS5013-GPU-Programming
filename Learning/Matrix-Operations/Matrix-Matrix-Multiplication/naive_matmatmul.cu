#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel for matrix multiplication: C = A * B
__global__ void matrix_multiplication_kernel(
    const float* A, const float* B, float* C, int M, int N, int K) 
{
    // Compute global thread coordinates
    dim3 thread_coords;
    thread_coords.x = blockIdx.x * blockDim.x + threadIdx.x; // column
    thread_coords.y = blockIdx.y * blockDim.y + threadIdx.y; // row

    unsigned int row = thread_coords.y;
    unsigned int col = thread_coords.x;

    if(row < M && col < K) {
        float Cvalue = 0.0f;
        for(unsigned int k = 0; k < N; ++k) {
            Cvalue += A[row * N + k] * B[k * K + col];
        }
        C[row * K + col] = Cvalue;
    }
}

// Host-callable function
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);

    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
}

// ----------------------
// Optional: Test in host code
// ----------------------
int main() {
    int M = 4, N = 4, K = 4;

    float h_A[M*N] = {0.f}, h_B[N*K] = {0.f}, h_C[M*K] = {0.f};
    for(unsigned int i = 0; i < M; i++) {
        for(unsigned int j = 0; j < N; ++j){
            if(i == j) h_A[i * N + j] = 1.0f;
        }
    }
    for(unsigned int i = 0; i < N; i++) {
        for(unsigned int j = 0; j < K; ++j){
            if(i == j) h_B[i * K + j] = 1.0f;
        }
    }
        float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*N*sizeof(float));
    cudaMalloc(&d_B, N*K*sizeof(float));
    cudaMalloc(&d_C, M*K*sizeof(float));

    cudaMemcpy(d_A, h_A, M*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N*K*sizeof(float), cudaMemcpyHostToDevice);

    solve(d_A, d_B, d_C, M, N, K);

    cudaMemcpy(h_C, d_C, M*K*sizeof(float), cudaMemcpyDeviceToHost);

    printf("C =\n");
    for(int i = 0; i < M; i++) {
        for(int j = 0; j < K; j++) {
            printf("%6.1f ", h_C[i*K + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}

