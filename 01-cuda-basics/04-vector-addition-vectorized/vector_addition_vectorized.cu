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
 * 1. Naive Scalar Vector Addition Kernel
 * Each thread loads and stores 1 float (4 bytes) per transaction.
 */
__global__ void vectorAddScalar(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * 2. Vectorized 128-bit (float4) Vector Addition Kernel
 * Each thread loads 4 floats (16 bytes = 128 bits) simultaneously
 * using hardware vector instruction LDG.E.128, which maximizes
 * memory bus utilization and reduces instruction dispatch overhead.
 * 
 * Grid-stride loop handles arbitrary sizes cleanly.
 */
__global__ void vectorAddVectorized4(const float4 *A, const float4 *B, float4 *C, int N4) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N4; i += stride) {
        float4 a = A[i];
        float4 b = B[i];
        float4 c;
        c.x = a.x + b.x;
        c.y = a.y + b.y;
        c.z = a.z + b.z;
        c.w = a.w + b.w;
        C[i] = c;
    }
}

int main() {
    int N = 1 << 24; // 16,777,216 elements (~67.1 MB per buffer)
    size_t bytes = N * sizeof(float);

    printf("====================================================\n");
    printf("  Module 01: Vectorized Memory Access (float4)      \n");
    printf("====================================================\n");
    printf("Problem size: %d elements (Total data moved per run: %.2f MB)\n\n",
           N, (3.0 * bytes) / (1024.0 * 1024.0));

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_scalar = (float*)malloc(bytes);
    float *h_C_vec = (float*)malloc(bytes);

    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.5f;
        h_B[i] = 2.5f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // CUDA timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    int warmup = 5;
    int iters = 20;

    // --- Benchmark 1: Scalar Kernel ---
    for (int i = 0; i < warmup; ++i) {
        vectorAddScalar<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        vectorAddScalar<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msScalar = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msScalar, start, stop));
    msScalar /= iters;
    CUDA_CHECK(cudaMemcpy(h_C_scalar, d_C, bytes, cudaMemcpyDeviceToHost));

    // --- Benchmark 2: Vectorized float4 Kernel ---
    int N4 = N / 4;
    int numBlocksVec = (N4 + blockSize - 1) / blockSize;
    if (numBlocksVec > 65535) numBlocksVec = 65535; // Cap for grid-stride

    for (int i = 0; i < warmup; ++i) {
        vectorAddVectorized4<<<numBlocksVec, blockSize>>>((const float4*)d_A, (const float4*)d_B, (float4*)d_C, N4);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        vectorAddVectorized4<<<numBlocksVec, blockSize>>>((const float4*)d_A, (const float4*)d_B, (float4*)d_C, N4);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msVec = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&msVec, start, stop));
    msVec /= iters;
    CUDA_CHECK(cudaMemcpy(h_C_vec, d_C, bytes, cudaMemcpyDeviceToHost));

    // Effective Bandwidth (GB/s): 2 reads + 1 write = 3 * N * sizeof(float)
    double totalBytesMoved = 3.0 * bytes;
    double bwScalar = (totalBytesMoved / (msScalar / 1000.0)) / 1e9;
    double bwVec = (totalBytesMoved / (msVec / 1000.0)) / 1e9;

    printf("--- Performance Showdown ---\n");
    printf("1. Scalar kernel (32-bit loads)  : %7.3f ms | Bandwidth: %6.2f GB/s\n", msScalar, bwScalar);
    printf("2. float4 kernel (128-bit loads) : %7.3f ms | Bandwidth: %6.2f GB/s (Speedup: %.2fx)\n\n",
           msVec, bwVec, msScalar / msVec);

    // Verify consistency
    float maxDiff = 0.0f;
    for (int i = 0; i < N; ++i) {
        float diff = fabsf(h_C_scalar[i] - h_C_vec[i]);
        if (diff > maxDiff) maxDiff = diff;
    }
    printf("Verification check: Max diff between Scalar and Vectorized = %e (%s)\n\n",
           maxDiff, (maxDiff == 0.0f) ? "IDENTICAL" : "MISMATCH");

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C_scalar);
    free(h_C_vec);

    return 0;
}
