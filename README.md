# CUDA Kernels

A collection of CUDA kernel implementations for learning GPU programming fundamentals, including thread hierarchy, memory management, and parallel algorithm optimization.

## Project Structure

```text
.
├── 01-basic-cuda/
│   ├── 01-vector-addition/
│   │   └── vector_addition.cu             # Basic vector addition on GPU
│   └── 02-matrix-multiplication/
│       └── matrix_multiplication.cu        # Simple matrix multiplication using global memory
├── 02-shared-memory/
│   ├── 01-parallel-reduction/
│   │   └── parallel_reduction.cu          # Block sum reduction using shared memory
│   ├── 02-tiled-matrix-multiplication/
│   │   └── tiled_matrix_multiplication.cu # Tiled matrix multiplication using shared memory
│   └── 03-softmax/
│       └── softmax.cu                     # Fast Softmax implementation using shared memory
├── CMakeLists.txt                         # Cross-platform CMake configuration
└── README.md                              # This documentation
```

## Compilation and Execution

There are two primary ways to compile these CUDA kernels: using **CMake** or compiling with **nvcc** directly.

### Option 1: Using CMake (Recommended)

1. Create a build directory and configure:
   ```bash
   mkdir build
   cd build
   cmake ..
   ```
2. Build the project:
   - **Linux / macOS**:
     ```bash
     make
     ```
   - **Windows**:
     ```bash
     cmake --build . --config Release
     ```

This will compile all five executable targets in the build directory.

### Option 2: Compiling Directly with nvcc

You can compile individual kernels directly using the NVIDIA CUDA Compiler (`nvcc`):

#### Basic CUDA Programs
```bash
# Vector Addition
nvcc 01-basic-cuda/01-vector-addition/vector_addition.cu -o vector_addition

# Matrix Multiplication
nvcc 01-basic-cuda/02-matrix-multiplication/matrix_multiplication.cu -o matrix_multiplication
```

#### Shared Memory Kernels
```bash
# Parallel Reduction
nvcc 02-shared-memory/01-parallel-reduction/parallel_reduction.cu -o parallel_reduction

# Tiled Matrix Multiplication
nvcc 02-shared-memory/02-tiled-matrix-multiplication/tiled_matrix_multiplication.cu -o tiled_matrix_multiplication

# Softmax
nvcc 02-shared-memory/03-softmax/softmax.cu -o softmax
```
