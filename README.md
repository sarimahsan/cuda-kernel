# CUDA Kernel Engineering: From Hello World to SwiGLU & Transformers

A pedagogical, production-grade repository of CUDA kernels written to build an intuitive, mathematically grounded understanding of GPU architectures, memory hierarchies, warp-level collectives, and high-performance fused operations used in modern Large Language Models (LLMs).

---

## Pedagogical Flow & Architecture

The repository is structured as a progressive curriculum divided into four sequential milestones:

```
[01-cuda-basics]
  ├── 01-hello-world/                   # SIMT execution model, grid/block indexing, printf, sync
  ├── 02-device-query/                  # Hardware introspection, SMs, warp limits, bandwidth formula
  ├── 03-vector-addition/               # 1D grids, memory management, host/device verification
  ├── 04-vector-addition-vectorized/   # float4 128-bit memory transactions, grid-stride loops
  └── 05-matrix-multiplication/         # 2D indexing, memory bandwidth bottleneck, arithmetic intensity
        │
        ▼
[02-shared-memory-and-warp]
  ├── 01-parallel-reduction/            # Shared memory (SRAM), tree reduction, bank conflicts
  ├── 02-tiled-matrix-multiplication/   # Block-level data reuse, T×T shared memory staging
  ├── 03-warp-primitives/               # Warp shuffles (__shfl_sync, __shfl_down_sync), zero-SRAM reduce
  ├── 04-softmax-shared/                # Numerically stable multi-pass block softmax
  └── 05-warp-softmax/                  # 1-warp-per-row, float4 vectorization, warp shuffles
        │
        ▼
[03-fused-activations]  <-- The Capstone of Elementwise Fusion
  ├── 01-silu/                          # Fast math intrinsics (__expf, __frcp_rn), float4 vectorization
  └── 02-swiglu/                        # Fused Swish-Gated Linear Unit
        ├── swiglu.cu                   # Vectorized FP32 (float4)
        ├── swiglu_half2.cu             # Vectorized FP16 (half2, __hmul2)
        └── swiglu_fast.cu              # 128-bit uint4 (8x FP16) + hardware reciprocal showdown
        │
        ▼
[04-transformer-blocks]
  ├── 01-rmsnorm/                       # Root Mean Square Normalization with warp reductions
  ├── 02-rope/                          # Rotary Position Embedding (RoPE) in FP16
  └── 03-flashattention/                # FlashAttention-v1 forward pass with online softmax in SRAM
```

---

## Repository Structure

```
cuda-kernels/
├── 01-cuda-basics/
│   ├── 01-hello-world/
│   │   └── hello_world.cu
│   ├── 02-device-query/
│   │   └── device_query.cu
│   ├── 03-vector-addition/
│   │   └── vector_addition.cu
│   ├── 04-vector-addition-vectorized/
│   │   └── vector_addition_vectorized.cu
│   └── 05-matrix-multiplication/
│       └── matrix_multiplication.cu
│
├── 02-shared-memory-and-warp/
│   ├── 01-parallel-reduction/
│   │   └── parallel_reduction.cu
│   ├── 02-tiled-matrix-multiplication/
│   │   └── tiled_matrix_multiplication.cu
│   ├── 03-warp-primitives/
│   │   └── warp_primitives.cu
│   ├── 04-softmax-shared/
│   │   └── softmax.cu
│   └── 05-warp-softmax/
│       └── warp_softmax.cu
│
├── 03-fused-activations/
│   ├── 01-silu/
│   │   └── silu.cu
│   └── 02-swiglu/
│       ├── swiglu.cu
│       ├── swiglu_half2.cu
│       └── swiglu_fast.cu
│
├── 04-transformer-blocks/
│   ├── 01-rmsnorm/
│   │   └── rmsnorm.cu
│   ├── 02-rope/
│   │   └── rope.cu
│   └── 03-flashattention/
│       └── flashattn.cu
│
├── CMakeLists.txt
└── README.md
```

---

## Detailed Mathematical & Architectural Concepts

### Module 01: CUDA Basics & Execution Model

#### 1. Hello World (`hello_world.cu`)
Introduces the Single Instruction, Multiple Threads (SIMT) programming model. A kernel is launched with dimensions `<<<gridDim, blockDim>>>`:
- Each thread is assigned coordinates $(\text{blockIdx.x}, \text{threadIdx.x})$.
- The unique global linear thread index is:
  $$\text{tid}_{\text{global}} = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}$$
- Because device execution is asynchronous relative to the host CPU, `cudaDeviceSynchronize()` flushes device buffers (such as device `printf`) to the host terminal.

#### 2. Device Query (`device_query.cu`)
Queries device parameters using `cudaGetDeviceProperties`. The theoretical peak DRAM memory bandwidth is computed by:
$$\text{Bandwidth}_{\text{peak}} = \frac{\text{Bus Width (bits)} \times \text{Memory Clock (Hz)} \times 2 \text{ (DDR)}}{8 \times 10^9} \text{ GB/s}$$

#### 3. Vector Addition (`vector_addition.cu`)
Computes elementwise addition across arrays $A, B \in \mathbb{R}^N$:
$$C[i] = A[i] + B[i], \quad i \in \{0, 1, \dots, N-1\}$$
Employs boundary checks $(i < N)$ to prevent buffer overruns when $N$ is not divisible by `blockDim.x`.

#### 4. Vectorized Memory Access (`vector_addition_vectorized.cu`)
A standard load instruction (`LDG.E.32`) reads 4 bytes per thread. Modern NVIDIA memory controllers process memory in 32-byte, 64-byte, or 128-byte transactions. By casting pointers to `float4`, each thread reads 16 bytes (128 bits) simultaneously using `LDG.E.128`:
$$\text{Data Moved Per Thread} = 4 \times 32\text{ bits} = 128\text{ bits}$$
Grid-stride loops are introduced to allow a fixed grid size to process arbitrary problem lengths without reallocating thread blocks:
$$i \leftarrow i + \text{gridDim.x} \cdot \text{blockDim.x}$$

#### 5. Naive Matrix Multiplication (`matrix_multiplication.cu`)
Computes matrix product $C = A \cdot B$ where $A \in \mathbb{R}^{M \times K}, B \in \mathbb{R}^{K \times N}, C \in \mathbb{R}^{M \times N}$:
$$C_{i, j} = \sum_{k=0}^{K-1} A_{i, k} \cdot B_{k, j}$$
Each thread $(i, j)$ performs $K$ global memory reads from row $i$ of $A$ and column $j$ of $B$. The arithmetic intensity is:
$$\text{Arithmetic Intensity} = \frac{2 M N K \text{ FLOPs}}{(M K + K N + M N) \times 4 \text{ Bytes}} \approx \frac{N}{6} \text{ FLOPs/Byte (for } M=N=K)$$
Without caching or shared memory, this naive kernel is severely bound by DRAM latency and redundant memory traffic.

---

### Module 02: Shared Memory & Warp-Level Reductions

#### 1. Parallel Reduction (`parallel_reduction.cu`)
Sums an array of $N$ elements using on-chip shared memory (`__shared__`). Rather than an interleaved loop stride that introduces warp divergence, it uses sequential addressing:
```cpp
for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
}
```
This reduces $N$ elements to block-level sums in $O(\log_2(\text{blockDim.x}))$ steps without shared memory bank conflicts.

#### 2. Tiled Matrix Multiplication (`tiled_matrix_multiplication.cu`)
Divides matrices into $T \times T$ tiles loaded cooperatively into shared memory:
$$\text{sA}[T][T], \quad \text{sB}[T][T]$$
Each element loaded into shared memory is reused $T$ times by the other threads within the block. This reduces global DRAM traffic by a factor of $T$ (typically $16\times$ or $32\times$).

#### 3. Warp Primitives & Register Shuffles (`warp_primitives.cu`)
A warp consists of 32 threads executing in SIMT lockstep. Register shuffle instructions exchange 32-bit registers directly between lanes within the warp without touching shared memory:
- `__shfl_sync(mask, val, src_lane)`: Broadcasts `val` from `src_lane` to all active lanes.
- `__shfl_down_sync(mask, val, delta)`: Shifts register values down by `delta` lanes.
- `__shfl_xor_sync(mask, val, lane_mask)`: Butterfly exchange for reductions.
- `__ballot_sync(mask, predicate)`: Evaluates a condition across all 32 threads and returns a 32-bit bitmask.

#### 4. Shared Memory Softmax (`softmax.cu`)
Computes numerically stable softmax along an array:
$$m = \max_{j} x_j$$
$$S_i = \exp(x_i - m)$$
$$\operatorname{Softmax}(x_i) = \frac{S_i}{\sum_{j} S_j}$$
Subtracting $m$ prevents numerical overflow during exponentiation ($\exp(x)$ overflows standard FP32 at $x \approx 88.7$).

#### 5. Warp-Level Softmax (`warp_softmax.cu`)
Dedicates exactly one warp (32 threads) to each row of a matrix. It combines 128-bit vectorized memory reads (`float4`) with `warp_reduce_max` and `warp_reduce_sum` via `__shfl_xor_sync`. Because all communications take place inside the register file, shared memory allocations and bank conflicts are eliminated.

---

### Module 03: Fused Elementwise Activations & SwiGLU

#### 1. SiLU (Swish) Fused Activation (`silu.cu`)
The Sigmoid Linear Unit (SiLU), also known as Swish-1, is defined as:
$$\operatorname{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$
To maximize hardware instruction throughput, division and exponentiation are evaluated using hardware intrinsics:
```cpp
__device__ __forceinline__ float silu(float x) {
    return x * __frcp_rn(1.0f + __expf(-x));
}
```
where `__expf` maps directly to the Special Function Unit (SFU) and `__frcp_rn` computes a single-cycle reciprocal with round-to-nearest.

#### 2. SwiGLU Fused Kernel (`swiglu.cu`, `swiglu_half2.cu`, `swiglu_fast.cu`)
SwiGLU is the gated activation function introduced by Shazeer (2020) and utilized in state-of-the-art LLMs (e.g., LLaMA 1/2/3, Mistral, Gemma, PaLM). Given input $x$ and gate $g$:
$$\operatorname{SwiGLU}(x, g) = \operatorname{SiLU}(x) \odot g = \left( \frac{x}{1 + e^{-x}} \right) \cdot g$$

The repository provides a complete optimization ladder:
1. **FP32 Vectorized (`swiglu.cu`)**: Uses `float4` loads to process 4 floats per thread.
2. **FP16 Native Vectorized (`swiglu_half2.cu`)**: Uses native `half2` instructions (`__hmul2`, `__hadd2`) to process 2 FP16 elements per 32-bit transaction.
3. **128-bit Vectorized FP16 (`swiglu_fast.cu`)**: Casts pointers to `float4` (or `uint4`), packaging 8 FP16 `half` elements per thread into a single 128-bit bus transaction. Features a systematic benchmark showdown comparing 8 distinct kernel configurations:
   - Scalar baseline (16-bit loads)
   - `half2` across thread-block sizes (128, 256, 512)
   - Pure FP16 native math (`h2exp`, `__h2div`) vs Hybrid FP32 compute (`__frcp_rn`)
   - Multiple elements per thread ($2\times$ and $4\times$ `half2`)
   - Full 128-bit vectorization (8 halfs per thread) reaching memory-bus saturation.

---

### Module 04: Transformer Building Blocks

#### 1. Root Mean Square Normalization (`rmsnorm.cu`)
RMSNorm simplifies LayerNorm by eliminating mean-centering:
$$\operatorname{RMS}(\mathbf{x}) = \sqrt{\frac{1}{D} \sum_{i=1}^D x_i^2 + \epsilon}$$
$$y_i = \frac{x_i}{\operatorname{RMS}(\mathbf{x})} \cdot \gamma_i$$
Reduces computational overhead while matching transformer performance.

#### 2. Rotary Position Embedding (`rope.cu`)
Applies position-dependent rotation to query and key representations. For token position $s$ and head dimension index $i \in \{0, \dots, D/2 - 1\}$:
$$\theta_i = \text{base}^{-\frac{2i}{D}}, \quad \alpha_{s, i} = s \cdot \theta_i$$
$$\begin{bmatrix} y_i \\ y_{i + D/2} \end{bmatrix} = \begin{bmatrix} \cos(\alpha_{s, i}) & -\sin(\alpha_{s, i}) \\ \sin(\alpha_{s, i}) & \cos(\alpha_{s, i}) \end{bmatrix} \begin{bmatrix} x_i \\ x_{i + D/2} \end{bmatrix}$$

#### 3. FlashAttention Forward Pass (`flashattn.cu`)
Standard attention evaluates:
$$\mathbf{O} = \operatorname{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d}}\right)\mathbf{V}$$
Materializing the intermediate $N \times N$ attention matrix incurs $O(N^2)$ memory reads/writes to high-latency HBM. FlashAttention tiles $Q$ into blocks of size $B_r \times d$ and $K, V$ into blocks of size $B_c \times d$ in SRAM, employing an online softmax rescaling algorithm:
$$\alpha = \exp(m_{\text{old}} - m_{\text{new}})$$
$$l_{\text{new}} = \alpha \cdot l_{\text{old}} + \sum \exp(S - m_{\text{new}})$$
$$\mathbf{O}_{\text{new}} = \alpha \cdot \mathbf{O}_{\text{old}} + \sum \exp(S - m_{\text{new}})\mathbf{V}$$
$$\mathbf{O}_{\text{final}} = \frac{\mathbf{O}_{\text{new}}}{l_{\text{new}}}$$

---

## Compilation & Execution

### Building with CMake

```bash
mkdir -p build && cd build
cmake ..
cmake --build . --config Release
```

The compiled binaries will be output to `build/bin/`.

### Building Directly with NVCC

```bash
# Module 01: CUDA Basics
nvcc -O3 01-cuda-basics/01-hello-world/hello_world.cu -o hello_world
nvcc -O3 01-cuda-basics/02-device-query/device_query.cu -o device_query
nvcc -O3 01-cuda-basics/03-vector-addition/vector_addition.cu -o vector_addition
nvcc -O3 01-cuda-basics/04-vector-addition-vectorized/vector_addition_vectorized.cu -o vector_addition_vectorized
nvcc -O3 01-cuda-basics/05-matrix-multiplication/matrix_multiplication.cu -o matrix_multiplication

# Module 02: Shared Memory & Warp
nvcc -O3 02-shared-memory-and-warp/01-parallel-reduction/parallel_reduction.cu -o parallel_reduction
nvcc -O3 02-shared-memory-and-warp/02-tiled-matrix-multiplication/tiled_matrix_multiplication.cu -o tiled_matrix_multiplication
nvcc -O3 02-shared-memory-and-warp/03-warp-primitives/warp_primitives.cu -o warp_primitives
nvcc -O3 02-shared-memory-and-warp/04-softmax-shared/softmax.cu -o softmax
nvcc -O3 02-shared-memory-and-warp/05-warp-softmax/warp_softmax.cu -o warp_softmax

# Module 03: Fused Activations & SwiGLU
nvcc -O3 03-fused-activations/01-silu/silu.cu -o silu
nvcc -O3 03-fused-activations/02-swiglu/swiglu.cu -o swiglu
nvcc -O3 -arch=sm_80 03-fused-activations/02-swiglu/swiglu_half2.cu -o swiglu_half2
nvcc -O3 -arch=sm_80 03-fused-activations/02-swiglu/swiglu_fast.cu -o swiglu_fast

# Module 04: Transformer Building Blocks
nvcc -O3 04-transformer-blocks/01-rmsnorm/rmsnorm.cu -o rmsnorm
nvcc -O3 -arch=sm_80 04-transformer-blocks/02-rope/rope.cu -o rope
nvcc -O3 -arch=sm_80 04-transformer-blocks/03-flashattention/flashattn.cu -o flashattn
```
