#include <stdio.h>
#include <cuda_runtime.h>

#define N 3   

// CUDA kernel
__global__ void matMul(int *A, int *B, int *C, int width) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {

        int sum = 0;

        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }

        C[row * width + col] = sum;
    }
}

int main() {

    int size = N * N * sizeof(int);

    int h_A[N * N] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };

    int h_B[N * N] = {
        9, 8, 7,
        6, 5, 4,
        3, 2, 1
    };

    int h_C[N * N];
    int *d_A, *d_B, *d_C;

    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // CPU to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    matMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    cudaDeviceSynchronize();

    //  GPU BACK TO CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result Matrix C:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_C[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
