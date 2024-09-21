
# 定义宏
macro(add_multiple_executables)
    foreach(executable_name ${ARGN})
        add_executable(${executable_name} ${executable_name}.cu)
    endforeach()
endmacro()

# 调用宏
add_multiple_executables(
    VectorAdd
    VectorMatrix
    VectorMatrix_Outer
    helloWorld
)


# # AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/src/*.cpp )
# # AUX_SOURCE_DIRECTORY(${PROJECT_SOURCE_DIR}/src/*.cu HELLOlib_src)

# # add_library(CMatrixData STATIC $(HELLOlib_src))
# # add_library(CGemmWithC STATIC CGemmWithC.cpp)

# # add_library(cuda_gemm)


# # add


# # Collect all .cpp and .cu files in the src directory
# file(GLOB HELLOlib_src_cpp ${PROJECT_SOURCE_DIR}/src/*.cpp)
# file(GLOB HELLOlib_src_cu ${PROJECT_SOURCE_DIR}/src/*.cu)

# # Combine both lists into HELLOlib_src
# set(HELLOlib_src ${HELLOlib_src_cpp} ${HELLOlib_src_cu})

# # Create a static library from the collected source files
# add_library(SUNWEI SHARED  ${HELLOlib_src})

# # add_executable(main VectorAdd.cu)

# # target_link_libraries(main PRIVATE SUNWEI)

# add_executable(VectorAdd VectorAdd.cu)
# add_executable(VectorMatrix VectorMatrix.cu)
# add_executable(VectorMatrixOuter VectorMatrix_Outer.cu)
# add_executable(helloWorld helloWorld.cu)

