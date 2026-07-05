#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d (%s) \"%s\"\n", file, line, err, cudaGetErrorName(err), func);
        exit(EXIT_FAILURE);
    }
}

__global__ void rmsnorm_kernel(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    float* __restrict__ out,
    int hidden_dim,
    float eps
) {
    extern __shared__ float sdata[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float* x_row = x + row * hidden_dim;
    float* out_row = out + row * hidden_dim;

    float local_sum = 0.0f;
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        float val = x_row[i];
        local_sum += val * val;
    }
    sdata[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    __shared__ float rms;
    if (tid == 0) {
        float mean_sq = sdata[0] / hidden_dim;
        rms = rsqrtf(mean_sq + eps);
    }
    __syncthreads();
    for (int i = tid; i < hidden_dim; i += blockDim.x) {
        out_row[i] = x_row[i] * rms * gamma[i];
    }
}

int main() {
    int rows = 2;
    int hidden_dim = 4;
    float eps = 1e-6f;

    float h_x[] = {1.0f, 2.0f, 3.0f, 4.0f,
                   5.0f, 6.0f, 7.0f, 8.0f};
    float h_gamma[] = {1.0f, 1.0f, 1.0f, 1.0f};

    float h_out[rows * hidden_dim];

    float *d_x, *d_gamma, *d_out;

    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x, rows * hidden_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_gamma, hidden_dim * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out, rows * hidden_dim * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x, rows * hidden_dim * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_gamma, h_gamma, hidden_dim * sizeof(float), cudaMemcpyHostToDevice));

    int threadsPerBlock = hidden_dim;
    if (threadsPerBlock > 256) threadsPerBlock = 256;
    if (threadsPerBlock == 0) threadsPerBlock = 1;

    dim3 dimGrid(rows);
    dim3 dimBlock(threadsPerBlock);
    size_t smem_size = threadsPerBlock * sizeof(float);

    printf("Launching kernel with %d blocks and %d threads per block, shared memory size %zu bytes.\n",
           dimGrid.x, dimBlock.x, smem_size);

    rmsnorm_kernel<<<dimGrid, dimBlock, smem_size>>>(d_x, d_gamma, d_out, hidden_dim, eps);
    CHECK_CUDA_ERROR(cudaGetLastError());

    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_out, d_out, rows * hidden_dim * sizeof(float), cudaMemcpyDeviceToHost));

    printf("Output (h_out):\n");
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < hidden_dim; ++j) {
            printf("%f ", h_out[i * hidden_dim + j]);
        }
        printf("\n");
    }

    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_gamma));
    CHECK_CUDA_ERROR(cudaFree(d_out));

    printf("Program finished successfully.\n");

    return 0;
}