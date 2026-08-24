#include <cuda_runtime.h>
#include <cuda_fp16.h>
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

// SiLU computed in fp32 internally for accuracy, then cast back to half.
__device__ __forceinline__ half silu_h(half x) {
    float xf = __half2float(x);
    float sf = xf / (1.0f + __expf(-xf));
    return __float2half_rn(sf);
}

__device__ __forceinline__ half2 silu2(half2 x) {
    float2 xf = __half22float2(x);
    float2 sf;
    sf.x = xf.x / (1.0f + __expf(-xf.x));
    sf.y = xf.y / (1.0f + __expf(-xf.y));
    return __float22half2_rn(sf);
}

// Vectorized: one thread handles one half2 (2 elements) as a single 32-bit
// transaction -> same coalescing win float4 gave the fp32 kernel.
__global__ void swiglu_kernel_h2(
    const half2* __restrict__ x,
    const half2* __restrict__ g,
    half2* __restrict__ out,
    int n_vec2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec2) {
        half2 xv = x[idx];
        half2 gv = g[idx];
        half2 sv = silu2(xv);
        out[idx] = __hmul2(sv, gv);
    }
}

__global__ void swiglu_kernel_h_tail(
    const half* __restrict__ x,
    const half* __restrict__ g,
    half* __restrict__ out,
    int start,
    int n
) {
    int idx = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __hmul(silu_h(x[idx]), g[idx]);
    }
}

void swiglu_launch_h2(const half* x, const half* g, half* out, int n, cudaStream_t stream = 0) {
    int n_vec2 = n / 2;
    int tail_start = n_vec2 * 2;
    int tail_n = n - tail_start;

    if (n_vec2 > 0) {
        int threads = 1024;
        int blocks = (n_vec2 + threads - 1) / threads;
        swiglu_kernel_h2<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<const half2*>(x),
            reinterpret_cast<const half2*>(g),
            reinterpret_cast<half2*>(out),
            n_vec2
        );
    }
    if (tail_n > 0) {
        int threads = tail_n;
        int blocks = 1;
        swiglu_kernel_h_tail<<<blocks, threads, 0, stream>>>(x, g, out, tail_start, n);
    }
    CUDA_CHECK(cudaGetLastError());
}

float cpu_silu(float x) {
    return x / (1.0f + expf(-x));
}

int main() {
    const int N = 1 << 20;
    const int N_test = N + 3;

    size_t bytes_h = N_test * sizeof(half);
    size_t bytes_f = N_test * sizeof(float);

    float *h_xf = (float*)malloc(bytes_f);
    float *h_gf = (float*)malloc(bytes_f);
    half  *h_x  = (half*)malloc(bytes_h);
    half  *h_g  = (half*)malloc(bytes_h);
    half  *h_out = (half*)malloc(bytes_h);
    float *h_ref = (float*)malloc(bytes_f);

    srand(42);
    for (int i = 0; i < N_test; i++) {
        h_xf[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        h_gf[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        h_x[i] = __float2half_rn(h_xf[i]);
        h_g[i] = __float2half_rn(h_gf[i]);
        h_ref[i] = cpu_silu(__half2float(h_x[i])) * __half2float(h_g[i]);
    }

    half *d_x, *d_g, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, bytes_h));
    CUDA_CHECK(cudaMalloc(&d_g, bytes_h));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_h));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes_h, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, h_g, bytes_h, cudaMemcpyHostToDevice));

    swiglu_launch_h2(d_x, d_g, d_out, N_test);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int iters = 100;
    cudaEventRecord(start);
    for (int i = 0; i < iters; i++) {
        swiglu_launch_h2(d_x, d_g, d_out, N_test);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    float avg_ms = ms / iters;

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes_h, cudaMemcpyDeviceToHost));

    double max_abs_err = 0.0;
    int first_bad = -1;
    for (int i = 0; i < N_test; i++) {
        double got = __half2float(h_out[i]);
        double err = fabs(got - (double)h_ref[i]);
        if (err > max_abs_err) max_abs_err = err;
        if (err > 5e-3 && first_bad < 0) first_bad = i;
    }

    printf("N = %d elements (fp16)\n", N_test);
    printf("Avg kernel time: %.4f ms\n", avg_ms);
    double bytes_moved = 3.0 * N_test * sizeof(half);
    double gbps = (bytes_moved / 1e9) / (avg_ms / 1e3);
    printf("Effective bandwidth: %.2f GB/s\n", gbps);
    printf("Max abs error vs fp32 ref: %e\n", max_abs_err);
    if (first_bad >= 0) {
        printf("FAIL: first mismatch at idx %d (got %f, expected %f)\n",
               first_bad, __half2float(h_out[first_bad]), h_ref[first_bad]);
    } else {
        printf("PASS: kernel output matches reference within fp16 tolerance.\n");
    }

    free(h_xf); free(h_gf); free(h_x); free(h_g); free(h_out); free(h_ref);
    cudaFree(d_x); cudaFree(d_g); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
