#include <stdio.h>
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

int main() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("====================================================\n");
    printf("         Module 01: CUDA Device Query Tool          \n");
    printf("====================================================\n\n");

    if (err != cudaSuccess || deviceCount == 0) {
        printf("No CUDA-capable device detected or driver issue encountered: %s\n",
               cudaGetErrorString(err));
        return 0;
    }

    printf("Detected %d CUDA-capable device(s)\n\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        printf("----------------------------------------------------\n");
        printf("Device ID: %d\n", dev);
        printf("Device Name: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Total Global Memory: %.2f GB (%zu bytes)\n",
               (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
               prop.totalGlobalMem);
        printf("Streaming Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
        printf("Warp Size: %d threads\n", prop.warpSize);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Shared Memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
        printf("Shared Memory per SM: %.2f KB\n", prop.sharedMemPerMultiprocessor / 1024.0);
        printf("Registers per Block (32-bit): %d\n", prop.regsPerBlock);
        printf("Registers per SM (32-bit): %d\n", prop.regsPerMultiprocessor);
        printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);
        printf("Memory Clock Rate: %.2f MHz\n", prop.memoryClockRate / 1000.0);

        // Theoretical peak memory bandwidth calculation:
        // Bandwidth = Memory Clock (Hz) * 2 (DDR) * (Bus Width / 8 bytes)
        double peakBandwidthGBs = 2.0 * (prop.memoryClockRate * 1e3) * (prop.memoryBusWidth / 8.0) / 1e9;
        printf("Theoretical Peak Memory Bandwidth: %.2f GB/s\n", peakBandwidthGBs);
        printf("L2 Cache Size: %.2f MB\n", prop.l2CacheSize / (1024.0 * 1024.0));
        printf("Concurrent Kernels Supported: %s\n", prop.concurrentKernels ? "Yes" : "No");
        printf("----------------------------------------------------\n\n");
    }

    return 0;
}
