#include "delete_list_kernels.cuh"

__global__ void checkIfNodeDeletedKernel(const uint* d_delete_list,
                                         uint        size,
                                         uint        node_id,
                                         bool*       d_found) {
    uint tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        if (d_delete_list[tid] == node_id)
            *d_found = true;
    }
}
