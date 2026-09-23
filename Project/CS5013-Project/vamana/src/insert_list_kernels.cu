#include "constants.cuh"
#include "insert_list_kernels.cuh"

#include <cmath>

// Kernel that checks if any query vector matches any vector in the insert list.
// Sets *d_found = 1 if found.
__global__ void checkIfNodeInsertedKernel(const FreshVamana::Consts::dtype_g* d_insert_list,
                                          uint                                insert_size,
                                          const FreshVamana::Consts::dtype_g* d_query_vecs,
                                          uint                                query_size,
                                          unsigned int*                       d_found) {
    using namespace FreshVamana::Consts;

    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= insert_size)
        return;

    if (atomicAdd(d_found, 0u) != 0u)  // already found
        return;

    const dtype_g* insert_vec = d_insert_list + idx * D_g;

    for (uint q = 0; q < query_size; ++q) {
        const dtype_g* query_vec = d_query_vecs + q * D_g;

        bool same = true;
        for (uint j = 0; j < D_g; ++j) {
            dtype_g a = insert_vec[j];
            dtype_g b = query_vec[j];

            if constexpr (std::is_floating_point_v<dtype_g>) {
                if (fabsf(a - b) > 1e-6f) {
                    same = false;
                    break;
                }
            } else {
                if (a != b) {
                    same = false;
                    break;
                }
            }
        }

        if (same) {
            atomicExch(d_found, 1u);
            return;
        }
    }
}
