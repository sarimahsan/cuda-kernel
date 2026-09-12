#include <stdio.h>
#include <cuda_runtime.h>

/**
 * Error-checking macro for CUDA API calls.
 */
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
 * Hello World CUDA Kernel:
 * Executed by every thread in the launched grid.
 * 
 * Each thread has unique coordinates:
 *  - blockIdx.x  : ID of the block within the grid
 *  - threadIdx.x : ID of the thread within its block
 *  - blockDim.x  : Number of threads per block
 *  - gridDim.x   : Number of blocks in the grid
 * 
 * Global unique 1D thread ID:
 *   global_tid = blockIdx.x * blockDim.x + threadIdx.x
 */
__global__ void helloWorldKernel() {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int bdim = blockDim.x;
    int gdim = gridDim.x;
    int global_tid = bid * bdim + tid;

    printf("Hello from GPU! [Grid: %d blocks, BlockDim: %d threads] -> Block: %d, Thread: %d, Global Thread ID: %d\n",
           gdim, bdim, bid, tid, global_tid);
}

int main() {
    printf("====================================================\n");
    printf("         Module 01: Hello World from CUDA           \n");
    printf("====================================================\n\n");

    // Launch configuration: 2 blocks, each containing 4 threads (Total = 8 threads)
    int blocksPerGrid = 2;
    int threadsPerBlock = 4;

    printf("Host: Launching kernel with %d blocks of %d threads (%d threads total)...\n\n",
           blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);

    // Launch the kernel on the device
    helloWorldKernel<<<blocksPerGrid, threadsPerBlock>>>();

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());

    // Kernel launches are asynchronous on the host. We must synchronize to
    // ensure device printf outputs are flushed to the host terminal before exiting.
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("\nHost: Kernel execution complete and synchronized successfully.\n");
    return 0;
}
