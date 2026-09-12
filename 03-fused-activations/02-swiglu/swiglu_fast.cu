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

// ─────────────────────────────────────────────────────────────
// Mathematical & Intrinsics Helpers
// ─────────────────────────────────────────────────────────────

// Helper 1: Hybrid FP32 compute (promotes to float2, uses __frcp_rn, casts to half2)
__device__ __forceinline__ half2 hybrid_silu2(half2 x) {
    float2 xf = __half22float2(x);
    float2 sf;
    sf.x = xf.x * __frcp_rn(1.0f + __expf(-xf.x));
    sf.y = xf.y * __frcp_rn(1.0f + __expf(-xf.y));
    return __float22half2_rn(sf);
}

// Helper 2: Pure FP16 native math (uses h2exp & __h2div with zero float conversions)
__device__ __forceinline__ half2 native_fp16_silu2(half2 x) {
    const half2 one = __float2half2_rn(1.0f);
    half2 neg_x = __hneg2(x);
    half2 exp_neg_x = h2exp(neg_x);
    half2 denom = __hadd2(one, exp_neg_x);
    return __h2div(x, denom);
}

// ─────────────────────────────────────────────────────────────
// 1. BASELINE KERNEL: Scalar FP16 (16-bit loads)
// ─────────────────────────────────────────────────────────────
__global__ void swiglu_scalar_baseline(
    const half* __restrict__ x,
    const half* __restrict__ g,
    half* __restrict__ out,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float xf = __half2float(x[idx]);
        float gf = __half2float(g[idx]);
        float silu = xf / (1.0f + expf(-xf));
        out[idx] = __float2half(silu * gf);
    }
}

// ─────────────────────────────────────────────────────────────
// 2. KERNEL: half2 Standard (1x half2 per thread)
// ─────────────────────────────────────────────────────────────
__global__ void swiglu_half2_hybrid(
    const half2* __restrict__ x,
    const half2* __restrict__ g,
    half2* __restrict__ out,
    int n_vec2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec2) {
        half2 xv = x[idx];
        half2 gv = g[idx];
        out[idx] = __hmul2(hybrid_silu2(xv), gv);
    }
}

__global__ void swiglu_half2_pure_fp16(
    const half2* __restrict__ x,
    const half2* __restrict__ g,
    half2* __restrict__ out,
    int n_vec2
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec2) {
        half2 xv = x[idx];
        half2 gv = g[idx];
        out[idx] = __hmul2(native_fp16_silu2(xv), gv);
    }
}

// ─────────────────────────────────────────────────────────────
// 3. KERNEL: half2 with 2x Elements Per Thread (4 halfs = 2 half2)
// ─────────────────────────────────────────────────────────────
__global__ void swiglu_half2_2x(
    const half2* __restrict__ x,
    const half2* __restrict__ g,
    half2* __restrict__ out,
    int n_vec2
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n_vec2) {
        half2 x0 = x[idx];
        half2 x1 = x[idx + 1];
        half2 g0 = g[idx];
        half2 g1 = g[idx + 1];

        out[idx]     = __hmul2(hybrid_silu2(x0), g0);
        out[idx + 1] = __hmul2(hybrid_silu2(x1), g1);
    } else if (idx < n_vec2) {
        out[idx] = __hmul2(hybrid_silu2(x[idx]), g[idx]);
    }
}

// ─────────────────────────────────────────────────────────────
// 4. KERNEL: half2 with 4x Elements Per Thread (8 halfs = 4 half2)
// ─────────────────────────────────────────────────────────────
__global__ void swiglu_half2_4x(
    const half2* __restrict__ x,
    const half2* __restrict__ g,
    half2* __restrict__ out,
    int n_vec2
) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (idx + k < n_vec2) {
            half2 xv = x[idx + k];
            half2 gv = g[idx + k];
            out[idx + k] = __hmul2(hybrid_silu2(xv), gv);
        }
    }
}

// ─────────────────────────────────────────────────────────────
// 5. KERNEL: 128-bit Vectorized (8 halfs / float4) at 256 threads
// ─────────────────────────────────────────────────────────────
__global__ void swiglu_128bit_256threads(
    const float4* __restrict__ x,
    const float4* __restrict__ g,
    float4* __restrict__ out,
    int n_vec8
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec8) {
        float4 rx = x[idx];
        float4 rg = g[idx];

        half2 x0 = *reinterpret_cast<half2*>(&rx.x);
        half2 x1 = *reinterpret_cast<half2*>(&rx.y);
        half2 x2 = *reinterpret_cast<half2*>(&rx.z);
        half2 x3 = *reinterpret_cast<half2*>(&rx.w);

        half2 g0 = *reinterpret_cast<half2*>(&rg.x);
        half2 g1 = *reinterpret_cast<half2*>(&rg.y);
        half2 g2 = *reinterpret_cast<half2*>(&rg.z);
        half2 g3 = *reinterpret_cast<half2*>(&rg.w);

        float4 rout;
        *reinterpret_cast<half2*>(&rout.x) = __hmul2(hybrid_silu2(x0), g0);
        *reinterpret_cast<half2*>(&rout.y) = __hmul2(hybrid_silu2(x1), g1);
        *reinterpret_cast<half2*>(&rout.z) = __hmul2(hybrid_silu2(x2), g2);
        *reinterpret_cast<half2*>(&rout.w) = __hmul2(hybrid_silu2(x3), g3);

        out[idx] = rout;
    }
}

// ─────────────────────────────────────────────────────────────
// Benchmark & Verification Suite
// ─────────────────────────────────────────────────────────────
float cpu_silu(float x) {
    return x / (1.0f + expf(-x));
}

int main() {
    printf("=========================================================================\n");
    printf("           SwiGLU Systematic Optimization Benchmark Study (FP16)        \n");
    printf("=========================================================================\n\n");

    const int N = 16 * 1024 * 1024; // 16M elements (~32MB per input tensor)
    size_t bytes_h = N * sizeof(half);
    size_t bytes_f = N * sizeof(float);

    half* h_x   = (half*)malloc(bytes_h);
    half* h_g   = (half*)malloc(bytes_h);
    half* h_out = (half*)malloc(bytes_h);
    float* h_ref = (float*)malloc(bytes_f);

    srand(42);
    for (int i = 0; i < N; i++) {
        float xf = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        float gf = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
        h_x[i] = __float2half_rn(xf);
        h_g[i] = __float2half_rn(gf);
        h_ref[i] = cpu_silu(xf) * gf;
    }

    half *d_x, *d_g, *d_out;
    CUDA_CHECK(cudaMalloc(&d_x, bytes_h));
    CUDA_CHECK(cudaMalloc(&d_g, bytes_h));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_h));

    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes_h, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_g, h_g, bytes_h, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int iters = 200;
    double bytes_moved = 3.0 * N * sizeof(half); // 2 reads (x, g) + 1 write (out)

    int n_vec2 = N / 2;
    int n_vec8 = N / 8;

    auto benchmark_kernel = [&](const char* name, auto launch_fn) -> float {
        // Warm-up
        launch_fn();
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            launch_fn();
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        return ms / iters;
    };

    // ─────────────────────────────────────────────
    // Run Sweeps
    // ─────────────────────────────────────────────

    // 1. Baseline
    float t_baseline = benchmark_kernel("Scalar Baseline", [&]() {
        swiglu_scalar_baseline<<<(N + 255) / 256, 256>>>(d_x, d_g, d_out, N);
    });

    // 2. Block Size Sweeps (half2 hybrid)
    float t_h2_128 = benchmark_kernel("half2 (128 threads)", [&]() {
        swiglu_half2_hybrid<<<(n_vec2 + 127) / 128, 128>>>(
            reinterpret_cast<const half2*>(d_x),
            reinterpret_cast<const half2*>(d_g),
            reinterpret_cast<half2*>(d_out),
            n_vec2
        );
    });

    float t_h2_256 = benchmark_kernel("half2 (256 threads)", [&]() {
        swiglu_half2_hybrid<<<(n_vec2 + 255) / 256, 256>>>(
            reinterpret_cast<const half2*>(d_x),
            reinterpret_cast<const half2*>(d_g),
            reinterpret_cast<half2*>(d_out),
            n_vec2
        );
    });

    float t_h2_512 = benchmark_kernel("half2 (512 threads)", [&]() {
        swiglu_half2_hybrid<<<(n_vec2 + 511) / 512, 512>>>(
            reinterpret_cast<const half2*>(d_x),
            reinterpret_cast<const half2*>(d_g),
            reinterpret_cast<half2*>(d_out),
            n_vec2
        );
    });

    // 3. Pure FP16 Native vs Hybrid
    float t_h2_pure_fp16 = benchmark_kernel("half2 (Pure FP16 h2exp)", [&]() {
        swiglu_half2_pure_fp16<<<(n_vec2 + 255) / 256, 256>>>(
            reinterpret_cast<const half2*>(d_x),
            reinterpret_cast<const half2*>(d_g),
            reinterpret_cast<half2*>(d_out),
            n_vec2
        );
    });

    // 4. Element Granularity Sweeps (256 threads)
    float t_h2_2x = benchmark_kernel("half2 + 2x elements/thread (256 th)", [&]() {
        swiglu_half2_2x<<<(n_vec2 / 2 + 255) / 256, 256>>>(
            reinterpret_cast<const half2*>(d_x),
            reinterpret_cast<const half2*>(d_g),
            reinterpret_cast<half2*>(d_out),
            n_vec2
        );
    });

    float t_h2_4x = benchmark_kernel("half2 + 4x elements/thread (256 th)", [&]() {
        swiglu_half2_4x<<<(n_vec2 / 4 + 255) / 256, 256>>>(
            reinterpret_cast<const half2*>(d_x),
            reinterpret_cast<const half2*>(d_g),
            reinterpret_cast<half2*>(d_out),
            n_vec2
        );
    });

    // 5. 128-bit Vectorized (float4 @ 256 threads)
    float t_128b_256 = benchmark_kernel("128-bit float4 (256 threads)", [&]() {
        swiglu_128bit_256threads<<<(n_vec8 + 255) / 256, 256>>>(
            reinterpret_cast<const float4*>(d_x),
            reinterpret_cast<const float4*>(d_g),
            reinterpret_cast<float4*>(d_out),
            n_vec8
        );
    });

    // ─────────────────────────────────────────────
    // Verification
    // ─────────────────────────────────────────────
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes_h, cudaMemcpyDeviceToHost));
    double max_abs_err = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = fabs((double)__half2float(h_out[i]) - (double)h_ref[i]);
        if (diff > max_abs_err) max_abs_err = diff;
    }

    // ─────────────────────────────────────────────
    // Print Formatted Report Table
    // ─────────────────────────────────────────────
    auto print_row = [&](const char* label, float time_ms) {
        double gbps = (bytes_moved / 1e9) / (time_ms / 1e3);
        double speedup = t_baseline / time_ms;
        printf("| %-45s | %9.4f | %16.2f | %7.2fx |\n", label, time_ms, gbps, speedup);
    };

    printf("Dataset: %d FP16 elements (Total IO: %.2f MB per iteration)\n\n",
           N, (double)bytes_moved / (1024.0 * 1024.0));

    printf("| Kernel Configuration                          | Time (ms) | Bandwidth (GB/s) | Speedup |\n");
    printf("| :-------------------------------------------- | :-------- | :--------------- | :------ |\n");
    print_row("1. Baseline Scalar (16-bit)", t_baseline);
    print_row("2. half2 (Block Size: 128 threads)", t_h2_128);
    print_row("3. half2 (Block Size: 256 threads)", t_h2_256);
    print_row("4. half2 (Block Size: 512 threads)", t_h2_512);
    print_row("5. half2 (Pure FP16 Math with h2exp)", t_h2_pure_fp16);
    print_row("6. half2 (2x elements/thread @ 256 th)", t_h2_2x);
    print_row("7. half2 (4x elements/thread @ 256 th)", t_h2_4x);
    print_row("8. float4 (128-bit @ 256 th)", t_128b_256);
    printf("\n");

    printf("Verification vs CPU FP32 Ref: Max Absolute Error = %.6e -> [%s]\n",
           max_abs_err, max_abs_err < 5e-3 ? "PASS" : "FAIL");

    // Cleanup
    free(h_x); free(h_g); free(h_out); free(h_ref);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_g));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
