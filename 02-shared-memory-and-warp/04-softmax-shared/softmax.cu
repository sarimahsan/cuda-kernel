#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(err) << "\n";             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/**
 * Numerically Stable Softmax in Shared Memory:
 * Computes:
 *   m = max_i(x_i)
 *   Softmax(x_i) = exp(x_i - m) / sum_j exp(x_j - m)
 * 
 * Subtracting the max element prevents floating-point overflow during exponentiation.
 */
__global__ void softmax_kernel(const float* __restrict__ x,
                               float* __restrict__ y,
                               int N) {

    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float val = (idx < N) ? x[idx] : -INFINITY;

    // -------------------------------------------------------------
    // 1. Reduce MAX in block
    // -------------------------------------------------------------
    shared[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = shared[0];

    // -------------------------------------------------------------
    // 2. Compute exp(x - max) and SUM reduction
    // -------------------------------------------------------------
    float exp_val = (idx < N) ? expf(val - max_val) : 0.0f;

    shared[tid] = exp_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }

    float sum = shared[0];

    // -------------------------------------------------------------
    // 3. Normalize probabilities
    // -------------------------------------------------------------
    if (idx < N) {
        y[idx] = exp_val / sum;
    }
}

// Host reference function
void softmax_cpu_reference(const std::vector<float>& x,
                           std::vector<float>& y) {

    float max_val = -1e30f;
    for (float v : x) max_val = std::max(max_val, v);

    float sum = 0.0f;
    for (float v : x) sum += std::exp(v - max_val);

    for (size_t i = 0; i < x.size(); i++) {
        y[i] = std::exp(x[i] - max_val) / sum;
    }
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);

    std::cout << "====================================================\n";
    std::cout << "  Module 02: Softmax with Shared Memory Reduction   \n";
    std::cout << "====================================================\n";
    std::cout << "Vector Size: " << N << " elements\n\n";

    std::vector<float> h_x(N), h_y(N), h_out(N);

    // Initialize input
    for (int i = 0; i < N; i++) {
        h_x[i] = sinf((float)i) * 2.0f;
    }

    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc(&d_x, size));
    CHECK_CUDA(cudaMalloc(&d_y, size));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), size, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    softmax_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_x, d_y, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, size, cudaMemcpyDeviceToHost));

    // CPU reference verification
    softmax_cpu_reference(h_x, h_out);

    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        max_err = fmaxf(max_err, fabsf(h_y[i] - h_out[i]));
    }

    std::cout << "Max error against CPU reference: " << max_err << "\n";
    std::cout << "Status: " << (max_err < 1e-5f ? "PASSED" : "FAILED") << "\n\n";

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
