cmake_minimum_required(VERSION 3.18)
project(CUDAKernels LANGUAGES CXX CUDA)

# C++ and CUDA Language Standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optimization and Architecture Flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")

# Place all compiled binaries in a unified bin directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# -------------------------------------------------------------
# Module 01: Basic CUDA Kernels
# -------------------------------------------------------------
add_executable(vector_addition 
    01-basic-cuda/01-vector-addition/vector_addition.cu
)

add_executable(matrix_multiplication 
    01-basic-cuda/02-matrix-multiplication/matrix_multiplication.cu
)

# -------------------------------------------------------------
# Module 02: Shared Memory & Reductions
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

add_executable(warp_softmax 
    02-shared-memory/04-warp-softmax/warp_softmax.cu
)

# -------------------------------------------------------------
# Module 03: Transformer & Attention Building Blocks
# -------------------------------------------------------------
add_executable(flashattn
    03-attention/01-flashattention/flashattn.cu
)

add_executable(rmsnorm
    03-attention/02-rmsnorm/rmsnorm.cu
)

add_executable(rope
    03-attention/03-rope/rope.cu
)

add_executable(swiglu
    03-attention/04-swiglu/swiglu.cu
)

add_executable(swiglu_half2
    03-attention/04-swiglu/swiglu_half2.cu
)

add_executable(swiglu_fast
    03-attention/04-swiglu/swiglu_fast.cu
)
