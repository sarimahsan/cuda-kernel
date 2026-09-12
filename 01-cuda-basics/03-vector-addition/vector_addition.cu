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
 * Standard 1D Vector Addition Kernel:
 * C[i] = A[i] + B[i]
 * 
 * Thread mapping:
 * i = blockIdx.x * blockDim.x + threadIdx.x
 */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1 << 20; // 1,048,576 elements (~1M floats)
    size_t bytes = N * sizeof(float);

    printf("====================================================\n");
    printf("        Module 01: Vector Addition (1D Grid)        \n");
    printf("====================================================\n");
    printf("Vector size: %d elements (%.2f MB per vector)\n\n", N, (double)bytes / (1024.0 * 1024.0));

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = sinf((float)i);
        h_B[i] = cosf((float)i);
    }

    // Allocate device memory
    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy host memory to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Kernel launch config: %d blocks, %d threads per block\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Numerical verification
    float maxErr = 0.0f;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        float diff = fabsf(h_C[i] - expected);
        if (diff > maxErr) maxErr = diff;
    }

    printf("Verification: Max absolute difference against CPU = %e\n", maxErr);
    printf("Status: %s\n\n", (maxErr < 1e-5f) ? "PASSED" : "FAILED");

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
