#include <cuda_runtime.h>
#include <stdio.h>

/* __global__ keyword identifier to identify the function as a cuda kernel*/
/* d for device(GPU) in the kernel name prefix*/

__global__ void d_hello_kernel(){
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
    printf("%d: Hello, World from GPU!\n", id);
}
int main(void){
    d_hello_kernel<<<2,3>>>();
    
    // check for launch errors
    //source: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g3f51e3575c2178246db0a94a430e0038
    // __host__ __device__ cudaError_t cudaDeviceSynchronize ( void )
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // wait for the device to finish and flush device-side printf output
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}

