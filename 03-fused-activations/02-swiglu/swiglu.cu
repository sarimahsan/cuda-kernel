#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while (0)

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}

__global__ void swiglu_kernel_f32(
    const float4* __restrict__ x,
    const float4* __restrict__ g,
    float4* __restrict__ out,
    int n_vec4
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec4) {
        float4 xv = x[idx];
        float4 gv = g[idx];
        float4 ov;
        ov.x = silu(xv.x) * gv.x;
        ov.y = silu(xv.y) * gv.y;
        ov.z = silu(xv.z) * gv.z;
        ov.w = silu(xv.w) * gv.w;
        out[idx] = ov;
    }
}

__global__ void swiglu_kernel_f32_tail(
    const float* __restrict__ x,
    const float* __restrict__ g,
    float* __restrict__ out,
    int start,
    int n
) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = silu(x[idx]) * g[idx];
    }
}

void swiglu_launch(const float* x, const float* g, float* out, int n, cudaStream_t stream = 0) {
    int n_vec4 = n / 4;
    int tail_start = n_vec4 * 4;
    int tail_n = n - tail_start;

    if (n_vec4 > 0) {
        int threads = 256;
        int blocks = (n_vec4 + threads - 1) / threads;
        swiglu_kernel_f32<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const float4*>(x),
            reinterpret_cast<const float4*>(g),
            reinterpret_cast<float4*>(out),
            n_vec4
        );
    }
    if (tail_n > 0) {
        int threads = tail_n;
        int blocks = 1;
        swiglu_kernel_f32_tail<<<blocks, threads, 0, stream>>>(x, g, out, tail_start, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

float cpu_silu(float x) {
    return x / (1.0f + expf(-x));
}

int main() {
    const int N = 1 << 20;
    const int N_test = N + 3;

    size_t bytes = N_test * sizeof(float);

    float *h_x = (float*)malloc(bytes);
    float *h_g = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N_test; i++) {
        h_x[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        h_g[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        h_ref[i] = cpu_silu(h_x[i]) * h_g[i];
    }

    float *d_x, *d_g, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_g, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, h_g, bytes, cudaMemcpyHostToDevice));

    swiglu_launch(d_x, d_g, d_out, N_test);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iters = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        swiglu_launch(d_x, d_g, d_out, N_test);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    double max_abs_err = 0.0;
    int first_bad = -1;
    for (int i = 0; i < N_test; i++) {
        double err = fabs((double)h_out[i] - (double)h_ref[i]);
        if (err > max_abs_err) max_abs_err = err;
        if (err > 1e-3 && first_bad < 0) first_bad = i;
    }

    printf("====================================================\n");
    printf("     Module 03: Vectorized SwiGLU (FP32 float4)     \n");
    printf("====================================================\n\n");
    printf("N = %d elements\n", N_test);
    printf("Avg kernel time: %.4f ms\n", avg_ms);
    double bytes_moved = 3.0 * N_test * sizeof(float);
    double gbps = (bytes_moved / 1e9) / (avg_ms / 1e3);
    printf("Effective bandwidth: %.2f GB/s\n", gbps);
    printf("Max abs error vs CPU ref: %e\n", max_abs_err);
    if (first_bad >= 0) {
        printf("FAIL: first mismatch at idx %d (got %f, expected %f)\n",
               first_bad, h_out[first_bad], h_ref[first_bad]);
    } else {
        printf("Verification: PASSED (output matches CPU reference within tolerance)\n\n");
    }

    free(h_x); free(h_g); free(h_out); free(h_ref);
    cudaFree(d_x); cudaFree(d_g); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
