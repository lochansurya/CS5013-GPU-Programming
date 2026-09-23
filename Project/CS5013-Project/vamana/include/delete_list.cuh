#pragma once
#include "constants.cuh"
#include "delete_list_kernels.cuh"
#include "utils.cuh"

#include "globals.cuh"

// TODO: Maybe use a bloom filter?
class DeleteList {
   public:
    DeleteList(uint initial_capacity = 5000u) {
        growth_factor_ = 1.3f;
        capacity_      = std::max(1u, initial_capacity);
        size_          = 0;
        gpuErrchk(cudaMalloc(&FreshVamana::Globals::d_delete_list_g, capacity_ * sizeof(uint)));
    }

    ~DeleteList() {
        cudaFree(FreshVamana::Globals::d_delete_list_g);
        FreshVamana::Globals::d_delete_list_g = nullptr;
    }

    uint size() const noexcept { return size_; }
    uint capacity() const noexcept { return capacity_; }

    void addNode(uint node_id) {
        if (size_ >= capacity_)
            expandDeleteList();

        gpuErrchk(cudaMemcpy(FreshVamana::Globals::d_delete_list_g + size_,
                             &node_id,
                             sizeof(uint),
                             cudaMemcpyHostToDevice));
        ++size_;
    }

    void addNodes(uint* d_node_ids, size_t num_nodes) {
        if (num_nodes == 0)
            return;

        if (size_ + num_nodes >= capacity_)
            expandDeleteList(num_nodes);

        gpuErrchk(cudaMemcpy(FreshVamana::Globals::d_delete_list_g + size_,
                             d_node_ids,
                             num_nodes * sizeof(uint),
                             cudaMemcpyDeviceToDevice));
        ++size_;
    }

    uint*       data() noexcept { return FreshVamana::Globals::d_delete_list_g; }
    const uint* data() const noexcept { return FreshVamana::Globals::d_delete_list_g; }

    void clear() noexcept {
        // cudaFree(FreshVamana::Globals::d_delete_list_g);
        // FreshVamana::Globals::d_delete_list_g = nullptr;
        capacity_ = 0;
        size_     = 0;
    }

   private:
    void expandDeleteList(size_t extra = 1) {
        const uint new_capacity = static_cast<uint>(
            std::max<uint>(1u, static_cast<uint>(capacity_ * growth_factor_ + extra)));
        uint* new_buffer = nullptr;
        gpuErrchk(cudaMalloc(&new_buffer, new_capacity * sizeof(uint)));

        if (FreshVamana::Globals::d_delete_list_g && capacity_ > 0) {
            gpuErrchk(cudaMemcpy(new_buffer,
                                 FreshVamana::Globals::d_delete_list_g,
                                 capacity_ * sizeof(uint),
                                 cudaMemcpyDeviceToDevice));
            cudaFree(FreshVamana::Globals::d_delete_list_g);
        }

        FreshVamana::Globals::d_delete_list_g = new_buffer;
        capacity_                             = new_capacity;
    }

   private:
    // uint* d_delete_list_ = nullptr;
    float growth_factor_ = 1.3f;
    uint  capacity_      = 0;
    uint  size_          = 0;
};

__host__ inline bool isNodeInDeleteList(uint* d_delete_list, size_t size, uint node_id) {
    if (size == 0)
        return false;

    bool  h_found = false;
    bool* d_found = nullptr;

    cudaMalloc(&d_found, sizeof(bool));
    cudaMemcpy(d_found, &h_found, sizeof(bool), cudaMemcpyHostToDevice);

    const uint threads = 256;
    const uint blocks  = (size + threads - 1) / threads;

    checkIfNodeDeletedKernel<<<blocks, threads>>>(d_delete_list, size, node_id, d_found);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_found, d_found, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(d_found);

    return h_found;
}

__device__ inline bool isNodeInDeleteList(const uint* d_delete_list, size_t size, uint node_id) {
    for (size_t i = 0; i < size; ++i) {
        if (d_delete_list[i] == node_id)
            return true;
    }
    return false;
}
