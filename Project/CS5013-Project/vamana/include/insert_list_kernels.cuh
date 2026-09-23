#pragma once
#include "constants.cuh"

// Kernel that checks if any query vector matches any vector in the insert list.
// Sets *d_found = 1 if found.
__global__ void checkIfNodeInsertedKernel(const FreshVamana::Consts::dtype_g* d_insert_list,
                                          uint                                insert_size,
                                          const FreshVamana::Consts::dtype_g* d_query_vecs,
                                          uint                                query_size,
                                          unsigned int*                       d_found);