#pragma once
#include "constants.cuh"
#include "insert_list_kernels.cuh"
#include "utils.cuh"

#include "globals.cuh"

// FreshVamana::Consts::D_g * sizeof(FreshVamana::Consts::dtype_g) = insert list element size
constexpr uint insert_entry_bytes_g =
    FreshVamana::Consts::D_g * sizeof(FreshVamana::Consts::dtype_g);

class InsertList {
   public:
    InsertList(uint initial_capacity = 5000u) {
        growth_factor_ = 1.3f;
        capacity_      = std::max(1u, initial_capacity);
        size_          = 0;
        using namespace FreshVamana;
        gpuErrchk(cudaMalloc(&Globals::d_insert_list_g, capacity_ * insert_entry_bytes_g));
    }

    ~InsertList() {
        using namespace FreshVamana;
        cudaFree(Globals::d_insert_list_g);
        Globals::d_insert_list_g = nullptr;
    }

    uint size() const noexcept { return size_; }
    uint capacity() const noexcept { return capacity_; }

    void addVectors(FreshVamana::Consts::dtype_g* d_vectors, size_t num_vectors) {
        if (num_vectors == 0)
            return;

        if (size_ + num_vectors >= capacity_)
            expandInsertList(num_vectors);

        using namespace FreshVamana;

        FreshVamana::Consts::dtype_g* dst =
            Globals::d_insert_list_g + static_cast<size_t>(size_) * FreshVamana::Consts::D_g;
        const size_t bytes = num_vectors * insert_entry_bytes_g;
        gpuErrchk(cudaMemcpy(dst, d_vectors, bytes, cudaMemcpyDeviceToDevice));
        size_ = static_cast<uint>(size_ + num_vectors);
        // printf("%i\n", size_);
    }

    FreshVamana::Consts::dtype_g* data() noexcept { return FreshVamana::Globals::d_insert_list_g; }
    const FreshVamana::Consts::dtype_g* data() const noexcept {
        return FreshVamana::Globals::d_insert_list_g;
    }

    void clear() noexcept {
        using namespace FreshVamana;
        // cudaFree(FreshVamana::Globals::d_insert_list_g);
        // FreshVamana::Globals::d_insert_list_g = nullptr;
        capacity_ = 0;
        size_     = 0;
    }

   private:
    void expandInsertList(size_t extra = 1) {
        using namespace FreshVamana;

        const uint new_capacity = static_cast<uint>(
            std::max<uint>(1u, static_cast<uint>(capacity_ * growth_factor_ + extra)));
        Consts::dtype_g* new_buffer = nullptr;
        gpuErrchk(
            cudaMalloc(&new_buffer, static_cast<size_t>(new_capacity) * insert_entry_bytes_g));

        if (Globals::d_insert_list_g && capacity_ > 0) {
            gpuErrchk(cudaMemcpy(new_buffer,
                                 Globals::d_insert_list_g,
                                 size_ * insert_entry_bytes_g,
                                 cudaMemcpyDeviceToDevice));
            cudaFree(Globals::d_insert_list_g);
        }

        Globals::d_insert_list_g = new_buffer;
        capacity_                = new_capacity;
    }

   private:
    float growth_factor_ = 1.3f;
    uint  capacity_      = 0;
    uint  size_          = 0;
};
