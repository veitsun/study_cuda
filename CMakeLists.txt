#cmake_minimum_required(VERSION 3.18)
cmake_minimum_required(VERSION 3.23)
project(kernel_gemm CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES "native")
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6/" CACHE PATH "CUDA Toolkit Root Directory")

find_package(CUDA REQUIRED)
# find_package(cublas REQUIRED HINTS ${CUDA_TOOLKIT_ROOT_DIR})
find_library(CUBLAS_LIBRARY NAMES cublas PATHS ${CUDA_TOOLKIT_ROOT_DIR}/lib64 NO_DEFAULT_PATH)
if (NOT CUBLAS_LIBRARY)
    message(FATAL_ERROR "Could not find cuBLAS library")
else()
    message(STATUS "Found cuBLAS library: ${CUBLAS_LIBRARY}")
endif()


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CUDA_ARCHITECTURES 60)
include_directories(include)
add_subdirectory(src)
set(CMAKE_CUDA_COMPILER "nvcc")


