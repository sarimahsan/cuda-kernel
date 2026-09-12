#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/**
 * Mathematical definition of Sigmoid Linear Unit (SiLU / Swish):
 *   SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
 * 
 * In hardware, we can compute 1 / (1 + exp(-x)) using fast math intrinsics:
 *   __frcp_rn(1.0f + __expf(-x))
 */
__device__ __forceinline__ float silu_device(float x) {
    return x * __frcp_rn(1.0f + __expf(-x));
}

/**
 * 1. Scalar SiLU Kernel (FP32)
 */
__global__ void silu_scalar_kernel(const float *x, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = silu_device(x[idx]);
    }
}

/**
 * 2. Vectorized 128-bit SiLU Kernel using float4 (FP32)
 * Dispatches 4 floats per thread using 128-bit LDG.E.128 transactions.
 */
__global__ void silu_vectorized_kernel(const float4 *x, float4 *out, int n_vec4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < n_vec4; i += stride) {
        float4 xv = x[i];
        float4 res;
        res.x = silu_device(xv.x);
        res.y = silu_device(xv.y);
        res.z = silu_device(xv.z);
        res.w = silu_device(xv.w);
        out[i] = res;
    }
}

float cpu_silu(float x) {
    return x / (1.0f + expf(-x));
}

int main() {
    const int N = 1 << 22; // ~4.19M elements
    size_t bytes = N * sizeof(float);

    printf("====================================================\n");
    printf("     Module 03: SiLU (Swish) Fused Activation       \n");
    printf("====================================================\n");
    printf("Array size: %d elements (%.2f MB)\n\n", N, (double)bytes / (1024.0 * 1024.0));

    float *h_x = (float*)malloc(bytes);
    float *h_out_scalar = (float*)malloc(bytes);
    float *h_out_vec = (float*)malloc(bytes);

    for (int i = 0; i < N; ++i) {
        h_x[i] = ((float)rand() / RAND_MAX) * 6.0f - 3.0f;
    }

    float *d_x, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_x, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocksScalar = (N + blockSize - 1) / blockSize;
    int N4 = N / 4;
    int numBlocksVec = (N4 + blockSize - 1) / blockSize;
    if (numBlocksVec > 65535) numBlocksVec = 65535;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int iters = 100;

    // --- Benchmark 1: Scalar ---
    silu_scalar_kernel<<<numBlocksScalar, blockSize>>>(d_x, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        silu_scalar_kernel<<<numBlocksScalar, blockSize>>>(d_x, d_out, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msScalar = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msScalar, start, stop));
    msScalar /= iters;
    CUDA_CHECK(cudaMemcpy(h_out_scalar, d_out, bytes, cudaMemcpyDeviceToHost));

    // --- Benchmark 2: Vectorized float4 ---
    silu_vectorized_kernel<<<numBlocksVec, blockSize>>>((const float4*)d_x, (float4*)d_out, N4);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        silu_vectorized_kernel<<<numBlocksVec, blockSize>>>((const float4*)d_x, (float4*)d_out, N4);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msVec = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msVec, start, stop));
    msVec /= iters;
    CUDA_CHECK(cudaMemcpy(h_out_vec, d_out, bytes, cudaMemcpyDeviceToHost));

    // Bandwidth: 1 read + 1 write = 2 * bytes
    double totalBytes = 2.0 * bytes;
    double bwScalar = (totalBytes / (msScalar / 1000.0)) / 1e9;
    double bwVec = (totalBytes / (msVec / 1000.0)) / 1e9;

    printf("--- Performance Comparison ---\n");
    printf("1. Scalar SiLU (32-bit loads)  : %7.4f ms | Bandwidth: %6.2f GB/s\n", msScalar, bwScalar);
    printf("2. float4 SiLU (128-bit loads) : %7.4f ms | Bandwidth: %6.2f GB/s (Speedup: %.2fx)\n\n",
           msVec, bwVec, msScalar / msVec);

    // Verify against CPU
    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i) {
        float expected = cpu_silu(h_x[i]);
        float diff = fabsf(h_out_vec[i] - expected);
        if (diff > maxErr) maxErr = diff;
    }
    printf("Verification against CPU: Max Absolute Error = %e (%s)\n\n",
           maxErr, (maxErr < 1e-4f) ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_out));
    free(h_x);
    free(h_out_scalar);
    free(h_out_vec);

    return 0;
}
