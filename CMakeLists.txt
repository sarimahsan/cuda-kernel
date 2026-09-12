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
# Module 01: CUDA Basics & Hardware Model
# -------------------------------------------------------------
add_executable(hello_world 
    01-cuda-basics/01-hello-world/hello_world.cu
)

add_executable(device_query 
    01-cuda-basics/02-device-query/device_query.cu
)

add_executable(vector_addition 
    01-cuda-basics/03-vector-addition/vector_addition.cu
)

add_executable(vector_addition_vectorized 
    01-cuda-basics/04-vector-addition-vectorized/vector_addition_vectorized.cu
)

add_executable(matrix_multiplication 
    01-cuda-basics/05-matrix-multiplication/matrix_multiplication.cu
)

# -------------------------------------------------------------
# Module 02: Shared Memory & Warp-Level Reductions
# -------------------------------------------------------------
add_executable(parallel_reduction 
    02-shared-memory-and-warp/01-parallel-reduction/parallel_reduction.cu
)

add_executable(tiled_matrix_multiplication 
    02-shared-memory-and-warp/02-tiled-matrix-multiplication/tiled_matrix_multiplication.cu
)

add_executable(warp_primitives 
    02-shared-memory-and-warp/03-warp-primitives/warp_primitives.cu
)

add_executable(softmax 
    02-shared-memory-and-warp/04-softmax-shared/softmax.cu
)

add_executable(warp_softmax 
    02-shared-memory-and-warp/05-warp-softmax/warp_softmax.cu
)

# -------------------------------------------------------------
# Module 03: Fused Elementwise Activations & SwiGLU
# -------------------------------------------------------------
add_executable(silu 
    03-fused-activations/01-silu/silu.cu
)

add_executable(swiglu 
    03-fused-activations/02-swiglu/swiglu.cu
)

add_executable(swiglu_half2 
    03-fused-activations/02-swiglu/swiglu_half2.cu
)

add_executable(swiglu_fast 
    03-fused-activations/02-swiglu/swiglu_fast.cu
)

# -------------------------------------------------------------
# Module 04: Transformer Building Blocks
# -------------------------------------------------------------
add_executable(rmsnorm 
    04-transformer-blocks/01-rmsnorm/rmsnorm.cu
)

add_executable(rope 
    04-transformer-blocks/02-rope/rope.cu
)

add_executable(flashattn 
    04-transformer-blocks/03-flashattention/flashattn.cu
)
