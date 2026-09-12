/*
 * flashattn.cu
 *
 * Complete Flash Attention v1 forward pass in CUDA.
 * Features:
 *   - Tiled Q/K/V loads into shared memory
 *   - Online (streaming) softmax — no N×N matrix materialised
 *   - Multi-head, batched  (B, H, N, d)
 *   - Optional causal (autoregressive) masking
 *   - Log-sum-exp saved for backward pass
 *   - Simple CPU reference + correctness check
 *
 * Build:
 *   nvcc -O3 -arch=sm_80 flashattn.cu -o flashattn
 * Run:
 *   ./flashattn            # runs self-test
 *   ./flashattn causal     # runs self-test with causal mask
 *
 * Tile-size guide (shared mem budget ~48 KB on Ampere):
 *   d=64  → Br=64, Bc=64  : (64+128)*64*4 = 49152 B  (tight — works on sm_80+)
 *   d=64  → Br=64, Bc=32  : (64+ 64)*64*4 = 32768 B  (safe everywhere)
 *   d=128 → Br=32, Bc=32  : (32+ 64)*128*4= 49152 B
 */

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <float.h>

// ─────────────────────────────────────────────────────────────
// Compile-time tile parameters — adjust for your d and GPU SRAM
// ─────────────────────────────────────────────────────────────
#define Br   64          // rows of Q tile  (= threads per block)
#define Bc   64          // rows of K/V tile
#define D_MAX 64         // head dimension (must match runtime d)

// ─────────────────────────────────────────────────────────────
// Convenience macro
// ─────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                   \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

// ═════════════════════════════════════════════════════════════
//  KERNEL
//
//  Grid  : (ceil(N/Br), H, B)     — one CTA per (query tile, head, batch)
//  Block : Br threads              — one thread per query row in the tile
//
//  Shared memory layout (dynamic):
//    [ Qi : Br×d ][ Kj : Bc×d ][ Vj : Bc×d ]
// ═════════════════════════════════════════════════════════════
__global__ void flash_attn_forward_kernel(
    const float* __restrict__ Q,    // [B, H, N, d]
    const float* __restrict__ K,    // [B, H, N, d]
    const float* __restrict__ V,    // [B, H, N, d]
          float* __restrict__ O,    // [B, H, N, d]
          float* __restrict__ L,    // [B, H, N]   log-sum-exp (for bwd)
    int N,                          // sequence length
    int d,                          // head dimension
    bool causal)                    // apply causal mask?
{
    // ── identify which (batch, head, query-tile) this CTA owns ──
    int tile_i = blockIdx.x;        // which Qi tile
    int head   = blockIdx.y;
    int batch  = blockIdx.z;
    int tx     = threadIdx.x;       // 0 … Br-1

    int i_start = tile_i * Br;      // global row index of first query in tile
    int global_q_row = i_start + tx;

    // base pointer for this (batch, head) slice
    int bh_offset = (batch * gridDim.y + head) * N * d;

    const float* Qbh = Q + bh_offset;
    const float* Kbh = K + bh_offset;
    const float* Vbh = V + bh_offset;
          float* Obh = O + bh_offset;
          float* Lbh = L + (batch * gridDim.y + head) * N;

    // ── shared memory ─────────────────────────────────────────
    extern __shared__ float smem[];
    float* Qs = smem;               // Br × d
    float* Ks = Qs + Br * d;       // Bc × d
    float* Vs = Ks + Bc * d;       // Bc × d

    // ── load Qi tile into shared memory ──────────────────────
    if (global_q_row < N) {
        for (int c = 0; c < d; ++c)
            Qs[tx * d + c] = Qbh[global_q_row * d + c];
    } else {
        for (int c = 0; c < d; ++c)
            Qs[tx * d + c] = 0.f;
    }
    __syncthreads();

    // ── per-thread accumulators (live in registers) ───────────
    float mi = -FLT_MAX;           // running row-max
    float li = 0.f;                // running normaliser
    float oi[D_MAX] = {};          // output accumulator (zero-initialised)

    float scale = 1.f / sqrtf((float)d);

    // ── inner loop: sweep over all KV tiles ──────────────────
    int num_kv_tiles = (N + Bc - 1) / Bc;

    for (int tj = 0; tj < num_kv_tiles; ++tj) {
        int j_start = tj * Bc;

        if (causal && j_start > i_start + Br - 1)
            break;

        for (int row = tx; row < Bc; row += Br) {
            int global_kv_row = j_start + row;
            if (global_kv_row < N) {
                for (int c = 0; c < d; ++c) {
                    Ks[row * d + c] = Kbh[global_kv_row * d + c];
                    Vs[row * d + c] = Vbh[global_kv_row * d + c];
                }
            } else {
                for (int c = 0; c < d; ++c) {
                    Ks[row * d + c] = 0.f;
                    Vs[row * d + c] = 0.f;
                }
            }
        }
        __syncthreads();

        float sij[Bc];
        float row_max = -FLT_MAX;

        for (int k = 0; k < Bc; ++k) {
            int global_kv_row = j_start + k;

            if (causal && global_kv_row > global_q_row) {
                sij[k] = -FLT_MAX;
                continue;
            }
            if (global_kv_row >= N) {
                sij[k] = -FLT_MAX;
                continue;
            }

            float dot = 0.f;
            for (int c = 0; c < d; ++c)
                dot += Qs[tx * d + c] * Ks[k * d + c];
            sij[k] = dot * scale;
            row_max = fmaxf(row_max, sij[k]);
        }

        // ── online softmax update ─────────────────────────────
        float m_new = fmaxf(mi, row_max);
        float alpha = (mi == -FLT_MAX) ? 0.f : expf(mi - m_new);

        float l_new = alpha * li;

        for (int c = 0; c < d; ++c)
            oi[c] *= alpha;

        for (int k = 0; k < Bc; ++k) {
            if (sij[k] == -FLT_MAX) continue;
            float p = expf(sij[k] - m_new);
            l_new += p;
            for (int c = 0; c < d; ++c)
                oi[c] += p * Vs[k * d + c];
        }

        mi = m_new;
        li = l_new;

        __syncthreads();
    }

    // ── write final output ────────────────────────────────────
    if (global_q_row < N) {
        float inv_l = (li == 0.f) ? 0.f : 1.f / li;
        for (int c = 0; c < d; ++c)
            Obh[global_q_row * d + c] = oi[c] * inv_l;

        Lbh[global_q_row] = mi + logf(li + 1e-8f);
    }
}

// ═════════════════════════════════════════════════════════════
//  HOST LAUNCHER
// ═════════════════════════════════════════════════════════════
void flash_attention_forward(
    const float* Q,
    const float* K,
    const float* V,
          float* O,
          float* L,
    int B, int H, int N, int d,
    bool causal = false)
{
    assert(d <= D_MAX && "Increase D_MAX to match head dimension");
    assert(d == Bc    && "This kernel assumes d == Bc for simplicity");

    size_t smem_bytes = (size_t)(Br + 2 * Bc) * d * sizeof(float);

    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    if (smem_bytes > prop.sharedMemPerBlock) {
        fprintf(stderr,
            "ERROR: smem needed %zu B > device max %zu B\n"
            "       Reduce Br or Bc.\n",
            smem_bytes, prop.sharedMemPerBlock);
        exit(EXIT_FAILURE);
    }

    dim3 grid((N + Br - 1) / Br, H, B);
    dim3 block(Br);

    flash_attn_forward_kernel<<<grid, block, smem_bytes>>>(
        Q, K, V, O, L, N, d, causal);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ═════════════════════════════════════════════════════════════
//  CPU REFERENCE
// ═════════════════════════════════════════════════════════════
void cpu_attention_reference(
    const float* Q,
    const float* K,
    const float* V,
          float* O,
    int B, int H, int N, int d,
    bool causal = false)
{
    float scale = 1.f / sqrtf((float)d);

    for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
        int base = (b * H + h) * N * d;

        for (int i = 0; i < N; ++i) {
            float row_max = -FLT_MAX;
            float scores[1024];
            for (int j = 0; j < N; ++j) {
                if (causal && j > i) { scores[j] = -FLT_MAX; continue; }
                float dot = 0.f;
                for (int c = 0; c < d; ++c)
                    dot += Q[base + i*d+c] * K[base + j*d+c];
                scores[j] = dot * scale;
                if (scores[j] > row_max) row_max = scores[j];
            }

            float sum = 0.f;
            for (int j = 0; j < N; ++j) {
                if (scores[j] == -FLT_MAX) { scores[j] = 0.f; continue; }
                scores[j] = expf(scores[j] - row_max);
                sum += scores[j];
            }
            for (int j = 0; j < N; ++j) scores[j] /= sum;

            for (int c = 0; c < d; ++c) {
                float acc = 0.f;
                for (int j = 0; j < N; ++j)
                    acc += scores[j] * V[base + j*d+c];
                O[base + i*d+c] = acc;
            }
        }
    }
}

static void fill_random(float* p, int n) {
    for (int i = 0; i < n; ++i)
        p[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
}

int main(int argc, char** argv)
{
    bool causal = (argc > 1 && strcmp(argv[1], "causal") == 0);

    printf("====================================================\n");
    printf("     Module 04: FlashAttention-v1 (Forward Pass)    \n");
    printf("====================================================\n");
    printf("Tile params: Br=%d, Bc=%d, d=%d, causal=%s\n\n",
           Br, Bc, D_MAX, causal ? "yes" : "no");

    int B = 2;          // batch size
    int H = 4;          // number of heads
    int N = 256;        // sequence length
    int d = D_MAX;      // head dimension

    assert(N % Br == 0 && "For this test keep N a multiple of Br");

    size_t qkv_sz = (size_t)B * H * N * d * sizeof(float);
    size_t l_sz   = (size_t)B * H * N     * sizeof(float);

    float* hQ  = (float*)malloc(qkv_sz);
    float* hK  = (float*)malloc(qkv_sz);
    float* hV  = (float*)malloc(qkv_sz);
    float* hO  = (float*)calloc(B*H*N*d, sizeof(float));
    float* hL  = (float*)calloc(B*H*N,   sizeof(float));
    float* hO_ref = (float*)calloc(B*H*N*d, sizeof(float));

    srand(42);
    fill_random(hQ, B*H*N*d);
    fill_random(hK, B*H*N*d);
    fill_random(hV, B*H*N*d);

    printf("Running CPU reference... ");
    fflush(stdout);
    cpu_attention_reference(hQ, hK, hV, hO_ref, B, H, N, d, causal);
    printf("done.\n");

    float *dQ, *dK, *dV, *dO, *dL;
    CUDA_CHECK(cudaMalloc(&dQ, qkv_sz));
    CUDA_CHECK(cudaMalloc(&dK, qkv_sz));
    CUDA_CHECK(cudaMalloc(&dV, qkv_sz));
    CUDA_CHECK(cudaMalloc(&dO, qkv_sz));
    CUDA_CHECK(cudaMalloc(&dL, l_sz));

    CUDA_CHECK(cudaMemcpy(dQ, hQ, qkv_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dK, hK, qkv_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dV, hV, qkv_sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dO, 0, qkv_sz));
    CUDA_CHECK(cudaMemset(dL, 0, l_sz));

    printf("Running GPU Flash Attention... ");
    fflush(stdout);

    // Warm-up
    flash_attention_forward(dQ, dK, dV, dO, dL, B, H, N, d, causal);

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));

    int REPS = 10;
    for (int r = 0; r < REPS; ++r)
        flash_attention_forward(dQ, dK, dV, dO, dL, B, H, N, d, causal);

    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    printf("done.  avg %.3f ms per call\n", ms / REPS);

    CUDA_CHECK(cudaMemcpy(hO, dO, qkv_sz, cudaMemcpyDeviceToHost));

    int   n_elem  = B * H * N * d;
    float max_err = 0.f, sum_err = 0.f;
    for (int i = 0; i < n_elem; ++i) {
        float diff = fabsf(hO[i] - hO_ref[i]);
        if (diff > max_err) max_err = diff;
        sum_err += diff;
    }
    float mean_err = sum_err / n_elem;

    printf("\nCorrectness (vs CPU reference):\n");
    printf("  max |error| = %.6e\n", max_err);
    printf("  mean|error| = %.6e\n", mean_err);
    printf("Status: %s\n\n", max_err < 1e-4f ? "PASSED" : "FAILED");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dO); cudaFree(dL);
    free(hQ); free(hK); free(hV);
    free(hO); free(hL); free(hO_ref);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    return 0;
}
