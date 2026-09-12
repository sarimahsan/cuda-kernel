#include <cuda_runtime.h>
#include <iostream>
#include <cmath>

#define TILE 16   

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__      \
                      << ": " << cudaGetErrorString(err) << "\n";             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/**
 * Tiled Matrix Multiplication using Shared Memory:
 * C = A * B
 * 
 * Each block of (TILE x TILE) threads cooperatively loads two (TILE x TILE)
 * tiles of matrices A and B from global memory into fast shared memory (__shared__).
 * 
 * This reduces global memory traffic by a factor of TILE (16x),
 * transforming memory-bound matmul into a much higher throughput operation.
 */
__global__ void matMulTiled(const float *A, const float *B, float *C, int N)
{
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float value = 0.0f;

    // Loop over all tiles required to compute C[row, col]
    for (int t = 0; t < (N + TILE - 1) / TILE; t++)
    {
        // Collaboratively load tile from A into shared memory
        if (row < N && (t * TILE + threadIdx.x) < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        // Collaboratively load tile from B into shared memory
        if (col < N && (t * TILE + threadIdx.y) < N)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  

        // Multiply the two tiles in shared memory
        #pragma unroll
        for (int k = 0; k < TILE; k++)
        {
            value += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads(); 
    }

    if (row < N && col < N)
    {
        C[row * N + col] = value;
    }
}

int main()
{
    int N = 1024;
    size_t size = N * N * sizeof(float);

    std::cout << "====================================================\n";
    std::cout << " Module 02: Tiled Matrix Multiplication (Shared Mem)\n";
    std::cout << "====================================================\n";
    std::cout << "Matrix Dimensions: " << N << " x " << N << " (Tile: " << TILE << "x" << TILE << ")\n\n";

    float *hA = new float[N * N];
    float *hB = new float[N * N];
    float *hC = new float[N * N];

    for (int i = 0; i < N * N; i++) {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, size));
    CUDA_CHECK(cudaMalloc(&dB, size));
    CUDA_CHECK(cudaMalloc(&dC, size));

    CUDA_CHECK(cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE, TILE);
    dim3 dimGrid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    matMulTiled<<<dimGrid, dimBlock>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost));

    std::cout << "Spot check C[0, 0] = " << hC[0] << " (Expected: " << (float)(N * 2.0f) << ")\n";
    std::cout << "Spot check C[N-1, N-1] = " << hC[N * N - 1] << "\n";
    std::cout << "Status: " << ((fabs(hC[0] - (float)(N * 2.0f)) < 1e-3) ? "PASSED" : "FAILED") << "\n\n";

    delete[] hA;
    delete[] hB;
    delete[] hC;
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return 0;
}
