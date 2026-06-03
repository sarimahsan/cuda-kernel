cmake_minimum_required(VERSION 3.18)
project(CUDAKernels LANGUAGES CXX CUDA)

# Enable C++17 and CUDA 17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -------------------------------------------------------------
# 01-basic-cuda executables
# -------------------------------------------------------------
add_executable(vector_addition 
    01-basic-cuda/01-vector-addition/vector_addition.cu
)

add_executable(matrix_multiplication 
    01-basic-cuda/02-matrix-multiplication/matrix_multiplication.cu
)

# -------------------------------------------------------------
# 02-shared-memory executables
# -------------------------------------------------------------
add_executable(parallel_reduction 
    02-shared-memory/01-parallel-reduction/parallel_reduction.cu
)

add_executable(tiled_matrix_multiplication 
    02-shared-memory/02-tiled-matrix-multiplication/tiled_matrix_multiplication.cu
)

add_executable(softmax 
    02-shared-memory/03-softmax/softmax.cu
)
