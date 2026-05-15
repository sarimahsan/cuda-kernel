%%writefile tile.cu
#include <cuda_runtime.h>
#include <iostream>

#define TILE 16   

__global__ void matMulTiled(float *A, float *B, float *C, int N)
{
    // Shared memory for tiles
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float value = 0.0f;

    // Loop over tiles
    for (int t = 0; t < (N + TILE - 1) / TILE; t++)
    {

        if (row < N && (t * TILE + threadIdx.x) < N)
            sA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE + threadIdx.x];
        else
            sA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < N && (t * TILE + threadIdx.y) < N)
            sB[threadIdx.y][threadIdx.x] = B[(t * TILE + threadIdx.y) * N + col];
        else
            sB[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();  

        // Multiply tile
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

    float *hA, *hB, *hC;
    float *dA, *dB, *dC;

    hA = new float[N*N];
    hB = new float[N*N];
    hC = new float[N*N];

    // Initialize matrices
    for (int i = 0; i < N*N; i++)
    {
        hA[i] = 1.0f;
        hB[i] = 2.0f;
    }

    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    dim3 threads(TILE, TILE);
    dim3 blocks(N / TILE, N / TILE);

    matMulTiled<<<blocks, threads>>>(dA, dB, dC, N);

    cudaDeviceSynchronize();

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    std::cout << "C[0] = " << hC[0] << std::endl;

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    delete[] hA;
    delete[] hB;
    delete[] hC;

    return 0;
}