#ifndef UTILS_H_
#define UTILS_H_

#include <cuda_runtime.h>
#include <stdio.h>
#include <cstdlib>

#define gpuErrchk(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// wrap each API call with the gpuErrchk macro, which will process the return status of the API call
// it wraps

// gpuAssert can be modified to raise an exception rather than call exit() in a more sophisticated
// application if it were required.

// sample usage:
// gpuErrchk(cudaMalloc(&d_a, sizeof(int)*1000000000000000));

/******************************************************************************************************/

// To check for errors in kernel launches, do the following in the cuda code

// kernel<<<1,122222>>>(a);
// gpuErrchk( cudaPeekAtLastError() ); // or gpuErrchk( cudaGetLastError() );
// gpuErrchk( cudaDeviceSynchronize() );

#include <iostream>

template <typename Func>
auto callWithLog(const char* func_name, const char* file, int line, const char* caller,
                 Func&& func) {
    constexpr const char* cyan  = "\033[36m";
    constexpr const char* reset = "\033[0m";

    std::cout << cyan << "[" << file << ":" << line << ":" << caller << "]" << reset << " Calling "
              << cyan << func_name << reset << '\n';
    std::flush(std::cout);

    if constexpr (std::is_void_v<decltype(func())>)
        func();
    else
        return func();
}

#define logfuncs

#if defined(logfuncs)
#define CALL_WITH_LOG(fn_call) \
    callWithLog(#fn_call, __FILE__, __LINE__, __func__, [&]() { return fn_call; })
#else
#define CALL_WITH_LOG(fn_call) fn_call
#endif

#endif
