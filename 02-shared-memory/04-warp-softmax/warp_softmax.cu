#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
void check(cudaError_t err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d (%s) \"%s\"\n", file, line, err, cudaGetErrorName(err), func);
        exit(EXIT_FAILURE);
    }
}

// Warp-level reduce max using register shuffles
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

// Warp-level reduce sum using register shuffles
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// Highly optimized Warp-Level Softmax: One warp (32 threads) processes one entire row
__global__ void warp_softmax_f32(
    const float* __restrict__ input,
    float* __restrict__ output,
    int rows,
    int cols
) {
    int lane_id = threadIdx.x % WARP_SIZE;
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;

    if (warp_id >= rows) return;

    const float* row_in = input + warp_id * cols;
    float* row_out = output + warp_id * cols;

    // 1. Compute Max with vectorized float4 loads
    float local_max = -INFINITY;
    int cols_vec4 = cols / 4;
    const float4* row_in_vec4 = reinterpret_cast<const float4*>(row_in);

    for (int i = lane_id; i < cols_vec4; i += WARP_SIZE) {
        float4 v = row_in_vec4[i];
        local_max = fmaxf(local_max, fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w)));
    }
    for (int i = cols_vec4 * 4 + lane_id; i < cols; i += WARP_SIZE) {
        local_max = fmaxf(local_max, row_in[i]);
    }

    float row_max = warp_reduce_max(local_max);

    // 2. Compute Exp and Sum
    float local_sum = 0.0f;
    for (int i = lane_id; i < cols_vec4; i += WARP_SIZE) {
        float4 v = row_in_vec4[i];
        local_sum += expf(v.x - row_max) + expf(v.y - row_max) + 
                     expf(v.z - row_max) + expf(v.w - row_max);
    }
    for (int i = cols_vec4 * 4 + lane_id; i < cols; i += WARP_SIZE) {
        local_sum += expf(row_in[i] - row_max);
    }

    float row_sum = warp_reduce_sum(local_sum);
    float inv_sum = 1.0f / row_sum;

    // 3. Write back normalized probabilities
    float4* row_out_vec4 = reinterpret_cast<float4*>(row_out);
    for (int i = lane_id; i < cols_vec4; i += WARP_SIZE) {
        float4 v = row_in_vec4[i];
        float4 res;
        res.x = expf(v.x - row_max) * inv_sum;
        res.y = expf(v.y - row_max) * inv_sum;
        res.z = expf(v.z - row_max) * inv_sum;
        res.w = expf(v.w - row_max) * inv_sum;
        row_out_vec4[i] = res;
    }
    for (int i = cols_vec4 * 4 + lane_id; i < cols; i += WARP_SIZE) {
        row_out[i] = expf(row_in[i] - row_max) * inv_sum;
    }
}

void cpu_softmax(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* r_in = in + r * cols;
        float* r_out = out + r * cols;

        float max_v = -INFINITY;
        for (int c = 0; c < cols; ++c) {
            if (r_in[c] > max_v) max_v = r_in[c];
        }

        float sum = 0.0f;
        for (int c = 0; c < cols; ++c) {
            r_out[c] = expf(r_in[c] - max_v);
            sum += r_out[c];
        }

        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; ++c) {
            r_out[c] *= inv_sum;
        }
    }
}

int main() {
    int rows = 4096;
    int cols = 1024;
    size_t num_elements = (size_t)rows * cols;
    size_t bytes = num_elements * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out_gpu = (float*)malloc(bytes);
    float* h_out_cpu = (float*)malloc(bytes);

    srand(42);
    for (size_t i = 0; i < num_elements; ++i) {
        h_in[i] = ((float)rand() / RAND_MAX) * 4.0f - 2.0f;
    }

    cpu_softmax(h_in, h_out_cpu, rows, cols);

    float *d_in, *d_out;
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_in, bytes));
    CHECK_CUDA_ERROR(cudaMalloc((void**)&d_out, bytes));
    CHECK_CUDA_ERROR(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    int threads_per_block = 256;
    int warps_per_block = threads_per_block / WARP_SIZE;
    int num_blocks = (rows + warps_per_block - 1) / warps_per_block;

    // Warm-up
    warp_softmax_f32<<<num_blocks, threads_per_block>>>(d_in, d_out, rows, cols);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    int iters = 100;
    CHECK_CUDA_ERROR(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        warp_softmax_f32<<<num_blocks, threads_per_block>>>(d_in, d_out, rows, cols);
    }
    CHECK_CUDA_ERROR(cudaEventRecord(stop));
    CHECK_CUDA_ERROR(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&ms, start, stop));
    float avg_ms = ms / iters;

    CHECK_CUDA_ERROR(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost));

    double max_err = 0.0;
    for (size_t i = 0; i < num_elements; ++i) {
        double err = fabs((double)h_out_gpu[i] - (double)h_out_cpu[i]);
        if (err > max_err) max_err = err;
    }

    double bytes_transferred = 2.0 * bytes; // 1 read + 1 write
    double gbps = (bytes_transferred / (avg_ms / 1000.0)) / 1e9;

    printf("Tensor size: [%d, %d] (%zu elements)\n", rows, cols, num_elements);
    printf("Avg kernel time: %.4f ms\n", avg_ms);
    printf("Effective bandwidth: %.2f GB/s\n", gbps);
    printf("Max absolute error vs CPU: %e\n", max_err);
    printf("Verification: %s\n", max_err < 1e-5 ? "PASS" : "FAIL");

    CHECK_CUDA_ERROR(cudaFree(d_in));
    CHECK_CUDA_ERROR(cudaFree(d_out));
    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    free(h_in);
    free(h_out_gpu);
    free(h_out_cpu);

    return 0;
}
