cmake_minimum_required(VERSION 3.18)
project(fresh_vamana LANGUAGES CXX CUDA)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)
# file(GLOB_RECURSE KERNEL_SRC_FILES cuda_kernels/*.cu)
# file(GLOB_RECURSE TEST_FILES test/*.cpp)

add_executable(fresh_vamana ${SRC_FILES})

set_target_properties(fresh_vamana PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "80;86;89"
    )

target_include_directories(fresh_vamana PRIVATE
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/include
    )

target_compile_options(fresh_vamana PRIVATE 
    -Wall 
    -Wextra
    -Wno-unused-parameter
    -Wno-deprecated-gpu-targets # for 6.0 compute p100
    -Xcompiler -Wno-unused-variable
    -Xcudafe --diag_suppress=550
    )

set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3 -rdc=true")

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")


target_link_libraries(fresh_vamana PRIVATE 
    ${CUDAToolkit_LIBRARIES}
    )

# -----------------------------
# Testing setup
# -----------------------------
# enable_testing()
# find_package(GTest REQUIRED)

# add_executable(test_exec ${TEST_FILES} ${KERNEL_SRC_FILES})

# set_target_properties(test_exec PROPERTIES
#     CUDA_STANDARD 17
#     CUDA_STANDARD_REQUIRED ON
#     CUDA_SEPARABLE_COMPILATION ON
#     )

# target_include_directories(test_exec PRIVATE
#     ${CUDAToolkit_INCLUDE_DIRS}
#     ${CMAKE_CURRENT_SOURCE_DIR}/include
#     ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernels
#     )

# target_link_libraries(test_exec PRIVATE GTest::gtest_main)

# include(GoogleTest)
# gtest_discover_tests(test_exec)
