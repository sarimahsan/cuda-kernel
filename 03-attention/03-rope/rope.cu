#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

__global__ void rope_kernel(
    const half* __restrict__ x,
    half* __restrict__ y,
    int B,
    int S,
    int H,
    int D,
    float theta_base
) {
    int half_d = D / 2;
    int total_pairs = B * S * H * half_d;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_pairs) return;

    int i = idx % half_d;
    int temp = idx / half_d;
    int h = temp % H;
    temp = temp / H;
    int s = temp % S;
    int b = temp / S;

    float exponent = (float)(2 * i) / (float)D;
    float freq = 1.0f / powf(theta_base, exponent);
    float angle = (float)s * freq;

    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    int base_offset = ((b * S + s) * H + h) * D;
    int idx1 = base_offset + i;
    int idx2 = base_offset + i + half_d;

    float x1 = __half2float(x[idx1]);
    float x2 = __half2float(x[idx2]);

    float y1 = x1 * cos_val - x2 * sin_val;
    float y2 = x1 * sin_val + x2 * cos_val;

    y[idx1] = __float2half(y1);
    y[idx2] = __float2half(y2);
}

int main() {
    int B = 1;
    int S = 4;
    int H = 2;
    int D = 4;
    float theta_base = 10000.0f;

    int total_elements = B * S * H * D;
    size_t bytes = total_elements * sizeof(half);

    half* h_x = (half*)malloc(bytes);
    half* h_y = (half*)malloc(bytes);

    for (int i = 0; i < total_elements; ++i) {
        h_x[i] = __float2half((float)(i + 1));
    }

    half *d_x, *d_y;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_x, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_y, bytes));

    CHECK_CUDA_ERROR(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));

    int half_d = D / 2;
    int total_pairs = B * S * H * half_d;
    int threadsPerBlock = 256;
    int blocks = (total_pairs + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching kernel with %d blocks and %d threads per block.\n", blocks, threadsPerBlock);

    rope_kernel<<<blocks, threadsPerBlock>>>(d_x, d_y, B, S, H, D, theta_base);
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    CHECK_CUDA_ERROR(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    printf("Output (h_y):\n");
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S; ++s) {
            for (int h = 0; h < H; ++h) {
                printf("Batch %d, Seq %d, Head %d: ", b, s, h);
                int base = ((b * S + s) * H + h) * D;
                for (int d = 0; d < D; ++d) {
                    printf("%.4f ", __half2float(h_y[base + d]));
                }
                printf("\n");
            }
        }
    }

    CHECK_CUDA_ERROR(cudaFree(d_x));
    CHECK_CUDA_ERROR(cudaFree(d_y));
    free(h_x);
    free(h_y);

    printf("Program finished successfully.\n");
    return 0;
}
