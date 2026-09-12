#include <stdio.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

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
 * Warp-level tree reduction for SUM using __shfl_down_sync.
 * Transfers data directly between thread register files across lanes
 * with zero shared memory latency and zero bank conflicts.
 */
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val; // Lane 0 holds the total sum
}

/**
 * Warp-level tree reduction for MAX using __shfl_down_sync.
 */
__device__ __forceinline__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(FULL_MASK, val, offset));
    }
    return val; // Lane 0 holds the maximum value
}

/**
 * Kernel demonstrating warp primitives:
 * 1. __shfl_sync (broadcast)
 * 2. __shfl_down_sync (tree reduction)
 * 3. __shfl_xor_sync (butterfly shuffle)
 * 4. __ballot_sync (warp predicate voting)
 */
__global__ void warpPrimitivesDemoKernel(float *d_sum, float *d_max) {
    int laneId = threadIdx.x % WARP_SIZE;
    int warpId = threadIdx.x / WARP_SIZE;

    // Give each lane a value: lane 0 has 1.0, lane 1 has 2.0, ..., lane 31 has 32.0
    float myVal = (float)(laneId + 1);

    // 1. Warp Tree Reductions
    float sumVal = warpReduceSum(myVal);
    float maxVal = warpReduceMax(myVal);

    // 2. Broadcast lane 0's sum to all other 31 lanes using __shfl_sync
    float broadcastSum = __shfl_sync(FULL_MASK, sumVal, 0);

    // 3. Butterfly exchange with immediate neighbor using __shfl_xor_sync (distance 1)
    float neighborVal = __shfl_xor_sync(FULL_MASK, myVal, 1);

    // 4. Warp vote predicate: check which threads have laneId >= 16
    unsigned int ballotMask = __ballot_sync(FULL_MASK, laneId >= 16);

    // Print demo output from Warp 0
    if (warpId == 0) {
        if (laneId == 0) {
            printf("[Warp 0, Lane 0] Sum of all 32 lanes: %.1f (Expected: 528.0)\n", sumVal);
            printf("[Warp 0, Lane 0] Max of all 32 lanes: %.1f (Expected: 32.0)\n", maxVal);
            printf("[Warp 0, Lane 0] Ballot mask for (laneId >= 16): 0x%08x (Expected: 0xffff0000)\n", ballotMask);
            *d_sum = sumVal;
            *d_max = maxVal;
        }

        // Print neighbor exchange for lanes 0 and 1
        if (laneId == 0 || laneId == 1) {
            printf("[Warp 0, Lane %d] Original val: %.1f, Neighbor val via __shfl_xor_sync: %.1f\n",
                   laneId, myVal, neighborVal);
        }

        // Verify broadcast on lane 15
        if (laneId == 15) {
            printf("[Warp 0, Lane 15] Received broadcast sum via __shfl_sync: %.1f\n", broadcastSum);
        }
    }
}

int main() {
    printf("====================================================\n");
    printf("     Module 02: Warp-Level Primitives & Shuffles    \n");
    printf("====================================================\n\n");

    float *d_sum, *d_max;
    CUDA_CHECK(cudaMalloc((void**)&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_max, sizeof(float)));

    // Launch 1 block with 1 warp (32 threads)
    warpPrimitivesDemoKernel<<<1, 32>>>(d_sum, d_max);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float h_sum = 0.0f, h_max = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_max, d_max, sizeof(float), cudaMemcpyDeviceToHost));

    printf("\nHost Validation: Sum = %.1f, Max = %.1f\n", h_sum, h_max);
    printf("Status: %s\n\n", (h_sum == 528.0f && h_max == 32.0f) ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_max));

    return 0;
}
