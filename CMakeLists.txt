#cmake_minimum_required(VERSION 3.18)
cmake_minimum_required(VERSION 3.23)
project(kernel_gemm CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDA REQUIRED)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CUDA_ARCHITECTURES 60)
include_directories(include)
add_subdirectory(src)
 

