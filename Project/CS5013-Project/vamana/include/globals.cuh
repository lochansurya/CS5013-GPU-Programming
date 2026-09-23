#pragma once
#include "constants.cuh"

namespace FreshVamana::Globals {

extern __device__ __managed__ uint d_graph_capacity_g;
extern __device__ __managed__ uint d_graph_size_g;

extern uint*                         d_delete_list_g;
extern FreshVamana::Consts::dtype_g* d_insert_list_g;

inline uint8_t* d_bloom_bits_g = nullptr;

}  // namespace FreshVamana::Globals
