#pragma once
#include "common.h"

/**
 * @file device_funcs.cu
 * @brief Device functions for distance calculation.
 */

/**
 * @brief Calculates the Euclidean distance squared between two vectors.
 *
 * @tparam DIM_SZ Dimension of the vectors.
 * @param a Pointer to the first vector.
 * @param b Pointer to the second vector.
 * @return float Euclidean distance squared.
 */
template <int DIM_SZ>
__device__ inline float distance(const float* a, const float* b) {
    float dist = 0.0f;

    // Unrolled loop for better performance on fixed dimensions
    #pragma unroll
    for (int i = 0; i < DIM_SZ; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }

    return dist;
}

// Specialization for 128D (Unrolled for performance)
template <>
__device__ inline float distance<128>(const float* a, const float* b) {
    float dist = 0.0f;
    #pragma unroll
    for (int i = 0; i < 128; ++i) {
        float diff = a[i] - b[i];
        dist += diff * diff;
    }
    return dist;
}

// Specialization for 2D
template <>
__device__ inline float distance<2>(const float* a, const float* b) {
    float d0 = a[0] - b[0];
    float d1 = a[1] - b[1];
    return d0*d0 + d1*d1;
}
