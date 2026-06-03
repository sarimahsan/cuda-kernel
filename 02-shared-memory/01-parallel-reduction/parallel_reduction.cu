#include <cuda.h>
#include <iostream>

__global__ void reduceSum(int *input, int *output, int n) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n)
        sdata[tid] = input[i];
    else
        sdata[tid] = 0;

    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
int main() {
    int n = 1024;
    int size = n * sizeof(int);

    int *h_input = new int[n];
    int *h_output;

    for (int i = 0; i < n; i++)
        h_input[i] = 1;

    int *d_input, *d_output;

    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;

    reduceSum<<<gridSize, blockSize, blockSize * sizeof(int)>>>(d_input, d_output, n);

    h_output = new int[gridSize];
    cudaMemcpy(h_output, d_output, gridSize * sizeof(int), cudaMemcpyDeviceToHost);

    int sum = 0;
    for (int i = 0; i < gridSize; i++)
        sum += h_output[i];

    std::cout << "Sum = " << sum << std::endl;
}
