#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to square each element
__global__ void d_square_array(int *d_arr, int N) {
    // Thread Organization: SM = Grid => Block => (Warp) => Thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_arr[idx] *= d_arr[idx];
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <N>\n", argv[0]);
        return 1;
    }

    const int N = atoi(argv[1]);
    int h_arr[N];  // Host array
    //int d_arr[N]; // device array

    // to allocate and deallocate the memory on the device, we need to use dynamic storage i.e. There is gonna be a corresponding free function API for the device. 
    int *d_arr;
    size_t arr_size = N * sizeof(int);
        
    // Initialize host array and/or Pre-Process the input to the GPU
    for (int i = 0; i < N; i++) {
        h_arr[i] = i + 1;  // 1, 2, 3, ...
    }


    // Configuration of the Device
    
    // Allocate memory on device
    cudaMalloc((void**)&d_arr, arr_size); // This void** cast is required; the API signature;
    // This is a pointer to a pointer, because on the device, it is an array so, that is a pointer. The function (API) needs to mutate the store directly so, it expects a pointer by default. without it, the developers have to move data across caller and callee.

    cudaMemcpy(d_arr, h_arr, arr_size, cudaMemcpyHostToDevice);
    // Copy array from host to device: HOST => DEVICE(GPU)
    // source: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html#group__CUDART__MEMORY_1gc263dbe6574220cc776b45438fc351e8
    // API: __host__ cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind )
    // Copies data between host and device.
    /* Parameters
        dst
            - Destination memory address
        src
            - Source memory address
        count
            - Size in bytes to copy
        kind
            - Type of transfer
        Returns
            cudaSuccess, cudaErrorInvalidValue, cudaErrorInvalidMemcpyDirection
    */


    // Kernel launch parameters
    // Thread Organization: SM = Grid => Block => (Warp) => Thread
    int num_threads_per_block = 256;
    int num_blocks_per_grid = (N + num_threads_per_block - 1) / num_threads_per_block;
    //making sure you have enough blocks to cover all N elements, even when N is not perfectly divisible by threads_per_block.
    
    // Launch kernel
    d_square_array<<<num_blocks_per_grid, num_threads_per_block>>>(d_arr, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_arr, d_arr, arr_size, cudaMemcpyDeviceToHost);

    // Print squared array; This runs on the Host(CPU)
    printf("Squared array:\n");
    for (int i = 0; i < N; i++) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_arr);

    return 0;
}

