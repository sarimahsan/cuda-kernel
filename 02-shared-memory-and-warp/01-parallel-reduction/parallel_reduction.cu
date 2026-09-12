#include <stdio.h>
#include <stdlib.h>
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
 * Tree-based Parallel Sum Reduction using Shared Memory:
 * Reduces an array of size N down to gridDim.x block sums.
 * 
 * Key optimization principles:
 * 1. Cooperative loading from global memory to shared memory (__shared__)
 * 2. Sequential addressing loop (stride >>= 1) to eliminate warp divergence
 *    and shared memory bank conflicts.
 * 3. __syncthreads() barrier synchronization between reduction tree levels.
 */
__global__ void reduceSum(const int *input, int *output, int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data from global memory into shared memory
    sdata[tid] = (i < n) ? input[i] : 0;
    __syncthreads();

    // Tree reduction in shared memory (sequential addressing)
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // Thread 0 writes the block's sum to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    int n = 1 << 20; // 1,048,576 elements
    size_t size = n * sizeof(int);

    printf("====================================================\n");
    printf("     Module 02: Parallel Reduction (Shared Memory)  \n");
    printf("====================================================\n");
    printf("Array size: %d elements\n\n", n);

    int *h_input = (int*)malloc(size);
    for (int i = 0; i < n; ++i) h_input[i] = 1; // Expected sum = n

    int *d_input, *d_output;
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    CUDA_CHECK(cudaMalloc((void**)&d_input, size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, gridSize * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    printf("Executing reduction across %d blocks (%d threads/block)...\n", gridSize, blockSize);
    reduceSum<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int *h_output = (int*)malloc(gridSize * sizeof(int));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost));

    long long totalSum = 0;
    for (int i = 0; i < gridSize; ++i) {
        totalSum += h_output[i];
    }

    printf("GPU Computed Sum: %lld | Expected Sum: %d\n", totalSum, n);
    printf("Status: %s\n\n", (totalSum == n) ? "PASSED" : "FAILED");

    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
