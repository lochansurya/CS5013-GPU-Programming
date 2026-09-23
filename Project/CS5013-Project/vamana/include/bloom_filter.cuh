#pragma once
#include <cstdint>

using uint = unsigned int;

// Vishank: reduced the size of bloom filter to fit in gpu
#define BF_ENTRIES 39988U  // per query, max entries in BF, (prime number)
constexpr uint BF_MEMORY =
    (BF_ENTRIES & 0xFFFFFFFC) + sizeof(uint);  // 4-byte mem aligned size for actual allocation

__device__ uint bf_hashFn1(uint x) {
    // FNV-1a hash
    uint64_t hash = 0xcbf29ce4;
    hash          = (hash ^ (x & 0xff)) * 0x01000193;
    hash          = (hash ^ ((x >> 8) & 0xff)) * 0x01000193;
    hash          = (hash ^ ((x >> 16) & 0xff)) * 0x01000193;
    hash          = (hash ^ ((x >> 24) & 0xff)) * 0x01000193;

    return hash % BF_ENTRIES;
}

__device__ uint bf_hashFn2(uint x) {
    // FNV-1a hash
    uint64_t hash = 0x84222325;
    hash          = (hash ^ (x & 0xff)) * 0x1B3;
    hash          = (hash ^ ((x >> 8) & 0xff)) * 0x1B3;
    hash          = (hash ^ ((x >> 16) & 0xff)) * 0x1B3;
    hash          = (hash ^ ((x >> 24) & 0xff)) * 0x1B3;
    return hash % BF_ENTRIES;
}

__device__ bool bf_check(bool* bf, uint x) {
    return bf[bf_hashFn1(x)] && bf[bf_hashFn2(x)];
}

__device__ void bf_set(bool* bf, uint x) {
    bf[bf_hashFn1(x)] = true;
    bf[bf_hashFn2(x)] = true;
}
