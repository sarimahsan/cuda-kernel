# CUDA Kernels

A collection of CUDA kernel implementations written while learning GPU programming, memory optimization techniques, and core operations used in modern architectures like Transformers.

---

## Structure

```
cuda-kernels/
├── 01-basic-cuda/
│   ├── 01-vector-addition/
│   │   └── vector_addition.cu          # Vector addition using 1D grid
│   └── 02-matrix-multiplication/
│       └── matrix_multiplication.cu    # Naive matrix multiplication
│
├── 02-shared-memory/
│   ├── 01-parallel-reduction/
│   │   └── parallel_reduction.cu       # Sum reduction with shared memory
│   ├── 02-tiled-matrix-multiplication/
│   │   └── tiled_matrix_multiplication.cu # Tiled matrix multiplication using shared memory
│   └── 03-softmax/
│       └── softmax.cu                  # Numerically stable softmax in shared memory
│
├── 03-attention/
│   ├── 01-flashattention/
│   │   └── flashattn.cu                # FlashAttention-v1 forward pass with online softmax
│   ├── 02-rmsnorm/
│   │   └── rmsnorm.cu                  # RMSNorm kernel
│   └── 03-rope/
│       └── rope.cu                     # Rotary Position Embedding (RoPE) in FP16
│
├── CMakeLists.txt
└── README.md
```

---

## Key Concepts & Mathematics

### 1. Vector Addition & Matrix Multiplication
- **Vector Addition**: Demonstrates basic 1D thread indexing:
  $$i = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}$$
  $$C[i] = A[i] + B[i]$$
- **Matrix Multiplication**: 2D grid mapping $(i, j)$ computing row-column dot products:
  $$C_{i, j} = \sum_{k=0}^{N-1} A_{i, k} \cdot B_{k, j}$$

---

### 2. Shared Memory Optimizations
- **Parallel Reduction**: Uses `__syncthreads()` and tree-based reduction in shared memory to sum an array of size $N$ in $O(\log N)$ parallel steps per block.
- **Tiled Matrix Multiplication**: Loads $T \times T$ sub-matrices into shared memory (`__shared__ float sA[T][T]`) to reduce slow global memory accesses by a factor of $T$.
- **Softmax**: Computes numerical max subtraction and exponent sum in block-shared memory:
  $$m = \max_i(x_i), \quad \operatorname{Softmax}(x_i) = \frac{\exp(x_i - m)}{\sum_j \exp(x_j - m)}$$

---

### 3. Transformer Building Blocks

#### Rotary Position Embedding (RoPE)
Applies a position-dependent rotation to queries/keys. For position $s$ and dimension index pair $i \in \{0, \dots, D/2 - 1\}$:

$$\theta_i = \text{base}^{-\frac{2i}{D}}$$
$$\alpha_{s, i} = s \cdot \theta_i$$

$$y_i = x_i \cos(\alpha_{s, i}) - x_{i + \frac{D}{2}} \sin(\alpha_{s, i})$$
$$y_{i + \frac{D}{2}} = x_i \sin(\alpha_{s, i}) + x_{i + \frac{D}{2}} \cos(\alpha_{s, i})$$

#### Root Mean Square Normalization (RMSNorm)
Scales input tokens by their root mean square without centering:

$$\operatorname{RMS}(\mathbf{x}) = \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2 + \epsilon}$$
$$y_i = \frac{x_i}{\operatorname{RMS}(\mathbf{x})} \cdot \gamma_i$$

#### FlashAttention (Forward Pass)
Computes multi-head attention in tiles without writing intermediate $N \times N$ score matrices to GPU DRAM:

$$\mathbf{O} = \operatorname{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}$$

Uses online softmax rescaling between tiles in SRAM:

$$\alpha = \exp(m_{\text{old}} - m_{\text{new}})$$
$$l_{\text{new}} = \alpha \cdot l_{\text{old}} + \sum \exp(S - m_{\text{new}})$$
$$\mathbf{O}_{\text{new}} = \alpha \cdot \mathbf{O}_{\text{old}} + \sum \exp(S - m_{\text{new}})\mathbf{V}$$

---

## Compilation

### Using CMake

```bash
mkdir -p build && cd build
cmake ..
cmake --build .
```

### Using NVCC directly

```bash
# Example: RoPE
nvcc -O3 03-attention/03-rope/rope.cu -o rope
./rope

# Example: FlashAttention
nvcc -O3 03-attention/01-flashattention/flashattn.cu -o flashattn
./flashattn
```
