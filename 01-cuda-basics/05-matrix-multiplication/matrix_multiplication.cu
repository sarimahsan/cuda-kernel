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
 * Naive Matrix Multiplication Kernel (Row-Major):
 * C = A * B
 * Matrix Dimensions: A (M x K), B (K x N), C (M x N)
 * 
 * Thread mapping:
 * row = blockIdx.y * blockDim.y + threadIdx.y
 * col = blockIdx.x * blockDim.x + threadIdx.x
 * 
 * Notice:
 * Each thread performs K memory reads from A and K memory reads from B,
 * making naive matmul heavily memory-bandwidth bound.
 */
__global__ void matMulNaive(const float *A, const float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int M = 512;
    int K = 512;
    int N = 512;

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    printf("====================================================\n");
    printf("     Module 01: Naive 2D Matrix Multiplication      \n");
    printf("====================================================\n");
    printf("Matrix Dimensions: %d x %d x %d\n\n", M, K, N);

    // Host allocations
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);

    for (int i = 0; i < M * K; ++i) h_A[i] = 1.0f / (float)(i % 13 + 1);
    for (int i = 0; i < K * N; ++i) h_B[i] = 1.0f / (float)(i % 17 + 1);

    // Device allocations
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeA));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeB));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    printf("Launch Grid: (%d, %d), Block: (%d, %d)\n", gridDim.x, gridDim.y, blockDim.x, blockDim.y);

    // Warmup & Timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    matMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaEventRecord(start));
    int iters = 10;
    for (int i = 0; i < iters; ++i) {
        matMulNaive<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    ms /= iters;

    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost));

    // Calculate GFLOPS
    // 2 * M * N * K operations (1 multiply + 1 accumulate per element)
    double ops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (ops / (ms / 1000.0)) / 1e9;

    printf("Average execution time: %.3f ms\n", ms);
    printf("Compute Throughput: %.2f GFLOPS\n\n", gflops);

    // Verify sample calculation against CPU
    float cpuSum = 0.0f;
    for (int k = 0; k < K; ++k) {
        cpuSum += h_A[0 * K + k] * h_B[k * N + 0];
    }
    printf("Spot check C[0, 0]: GPU = %f, CPU = %f (Diff = %e)\n\n",
           h_C[0], cpuSum, fabsf(h_C[0] - cpuSum));

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
