#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void 
d_vec_add_kernel(int* a, int* b, int* c, int num_elems){
    // calculate the index using the threadId and blockId
    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index < num_elems){
        //vector add
        c[index] = a[index] + b[index];
    }
}


int
main(void){
    int n = 1 << 16; // number of elements in the array.

    // Host Memory Allocation
    int *h_a, *h_b, *h_c;
    size_t num_bytes = n * sizeof(int);
    h_a = (int *)malloc(num_bytes);
    h_b = (int *)malloc(num_bytes);
    h_c = (int *)malloc(num_bytes);

    memset(h_c, 0, num_bytes);
    for(int i = 0; i < n; ++i){
        h_a[i] = i;
        h_b[i] = i+1;
    }

    // Device Memory Allocation
    int *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, num_bytes);
    cudaMalloc(&d_b, num_bytes);
    cudaMalloc(&d_c, num_bytes);

    // Set the grid (num_blocks_per_grid, num_threads_per_block, 1)
    unsigned int num_threads_per_block = 1 << 10 ;
    unsigned int num_blocks_per_grid = (int)ceil((float) n/ num_threads_per_block);

    printf("Grid Size is %d\n", num_blocks_per_grid);

    cudaMemcpy(d_a, h_a, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, num_bytes, cudaMemcpyHostToDevice);


    d_vec_add_kernel<<<num_blocks_per_grid, num_threads_per_block>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, num_bytes, cudaMemcpyDeviceToHost);

    // check the result
    printf("Result Vector: \n[");
    for(unsigned int i = 0; i < n; ++i){
        printf("%d ", h_c[i]);
    }
    printf("]\n");

    

    
    return 0;
}
