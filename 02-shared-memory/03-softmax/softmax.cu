#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cmath>

#define CHECK_CUDA(call) \
    if((call) != cudaSuccess) { \
        std::cerr << "CUDA error\n"; \
        exit(1); \
    }

__global__ void softmax_kernel(const float* __restrict__ x,
                               float* __restrict__ y,
                               int N) {

    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    float val = (idx < N) ? x[idx] : -INFINITY;

    // -----------------------------
    // 1. Reduce MAX in block
    // -----------------------------
    shared[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        }
        __syncthreads();
    }

    float max_val = shared[0];

    // -----------------------------
    // 2. Compute exp(x - max) + SUM
    // -----------------------------
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

    // -----------------------------
    // 3. Normalize
    // -----------------------------
    if (idx < N) {
        y[idx] = exp_val / sum;
    }
}


// -----------------------------
// Host function
// -----------------------------
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


// -----------------------------
// Main
// -----------------------------
int main() {

    int N = 1024;
    size_t size = N * sizeof(float);

    std::vector<float> h_x(N), h_y(N), h_out(N);

    // initialize input
    for (int i = 0; i < N; i++) {
        h_x[i] = sin(i) * 2.0f; // sample data
    }

    float *d_x, *d_y;

    CHECK_CUDA(cudaMalloc(&d_x, size));
    CHECK_CUDA(cudaMalloc(&d_y, size));

    CHECK_CUDA(cudaMemcpy(d_x, h_x.data(), size, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    softmax_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_x, d_y, N);

    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_y.data(), d_y, size, cudaMemcpyDeviceToHost));

    // CPU reference
    softmax_cpu_reference(h_x, h_out);

    // compare
    float max_err = 0;
    for (int i = 0; i < N; i++) {
        max_err = fmax(max_err, fabs(h_y[i] - h_out[i]));
    }

    std::cout << "Max error: " << max_err << std::endl;

    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}
