/*
 * flash_attention.cu
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
 *   nvcc -O3 -arch=sm_80 flash_attention.cu -o flash_attn
 * Run:
 *   ./flash_attn            # runs self-test
 *   ./flash_attn causal     # runs self-test with causal mask
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
// Each thread keeps its output row in registers (D_MAX floats).
// Increase D_MAX for d=128, but watch register pressure.

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
    // Guard: rows past N are out of bounds — pad with 0
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

        // ── causal shortcut: entire tile is in the future ─────
        // For causal attention the query at row i only attends to
        // keys at rows j ≤ i.  If the smallest j in this tile
        // is already > the largest i in our query tile, skip it.
        if (causal && j_start > i_start + Br - 1)
            break;

        // ── cooperative load of Kj, Vj into shared memory ────
        // Distribute rows among threads: thread tx loads rows
        // tx, tx+Br, tx+2*Br, … of the Bc-row tile.
        for (int row = tx; row < Bc; row += Br) {
            int global_kv_row = j_start + row;
            if (global_kv_row < N) {
                for (int c = 0; c < d; ++c) {
                    Ks[row * d + c] = Kbh[global_kv_row * d + c];
                    Vs[row * d + c] = Vbh[global_kv_row * d + c];
                }
            } else {
                // pad out-of-bounds rows so masked scores → -inf
                for (int c = 0; c < d; ++c) {
                    Ks[row * d + c] = 0.f;
                    Vs[row * d + c] = 0.f;
                }
            }
        }
        __syncthreads();

        // ── compute score row Sij[0..Bc) for this thread ─────
        float sij[Bc];
        float row_max = -FLT_MAX;

        for (int k = 0; k < Bc; ++k) {
            int global_kv_row = j_start + k;

            // causal mask: query i should not attend to key j > i
            if (causal && global_kv_row > global_q_row) {
                sij[k] = -FLT_MAX;
                continue;
            }
            // out-of-bounds key
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
        //
        //  Given previous state (mi, li, oi) and new block scores sij:
        //
        //    m_new = max(mi, row_max(sij))
        //    alpha = exp(mi - m_new)               ← rescale old state
        //    p_k   = exp(sij[k] - m_new)           ← unnorm softmax weights
        //    l_new = alpha * li + Σ_k p_k
        //    O_new = alpha * oi + Σ_k p_k * Vj[k]
        //
        //  Final output: O / l_new  (done after all tiles)

        float m_new = fmaxf(mi, row_max);
        float alpha = (mi == -FLT_MAX) ? 0.f : expf(mi - m_new);

        float l_new = alpha * li;

        // rescale old output accumulator
        for (int c = 0; c < d; ++c)
            oi[c] *= alpha;

        // accumulate new softmax-weighted V rows
        for (int k = 0; k < Bc; ++k) {
            if (sij[k] == -FLT_MAX) continue;          // masked
            float p = expf(sij[k] - m_new);
            l_new += p;
            for (int c = 0; c < d; ++c)
                oi[c] += p * Vs[k * d + c];
        }

        mi = m_new;
        li = l_new;

        __syncthreads();    // protect shared mem before next tile's write
    }

    // ── write final output ────────────────────────────────────
    if (global_q_row < N) {
        float inv_l = (li == 0.f) ? 0.f : 1.f / li;
        for (int c = 0; c < d; ++c)
            Obh[global_q_row * d + c] = oi[c] * inv_l;

        // save log-sum-exp for backward pass:  L[i] = m + log(l)
        Lbh[global_q_row] = mi + logf(li + 1e-8f);
    }
}

// ═════════════════════════════════════════════════════════════
//  HOST LAUNCHER
// ═════════════════════════════════════════════════════════════
void flash_attention_forward(
    const float* Q,     // device ptr [B, H, N, d]
    const float* K,
    const float* V,
          float* O,     // device ptr [B, H, N, d]
          float* L,     // device ptr [B, H, N]
    int B, int H, int N, int d,
    bool causal = false)
{
    assert(d <= D_MAX && "Increase D_MAX to match head dimension");
    assert(d == Bc    && "This kernel assumes d == Bc for simplicity; "
                         "change Bc or template on d for other sizes");

    // shared memory: Qi(Br×d) + Kj(Bc×d) + Vj(Bc×d)
    size_t smem_bytes = (size_t)(Br + 2 * Bc) * d * sizeof(float);

    // Verify smem fits
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
//  CPU REFERENCE  (slow but obviously correct)
// ═════════════════════════════════════════════════════════════
void cpu_attention_reference(
    const float* Q,     // host ptr [B, H, N, d]
    const float* K,
    const float* V,
          float* O,     // host ptr [B, H, N, d]
    int B, int H, int N, int d,
    bool causal = false)
{
    float scale = 1.f / sqrtf((float)d);

    for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h) {
        int base = (b * H + h) * N * d;

        for (int i = 0; i < N; ++i) {
            // compute scores
            float row_max = -FLT_MAX;
            float scores[1024];   // assume N ≤ 1024 for the test
            for (int j = 0; j < N; ++j) {
                if (causal && j > i) { scores[j] = -FLT_MAX; continue; }
                float dot = 0.f;
                for (int c = 0; c < d; ++c)
                    dot += Q[base + i*d+c] * K[base + j*d+c];
                scores[j] = dot * scale;
                if (scores[j] > row_max) row_max = scores[j];
            }

            // softmax
            float sum = 0.f;
            for (int j = 0; j < N; ++j) {
                if (scores[j] == -FLT_MAX) { scores[j] = 0.f; continue; }
                scores[j] = expf(scores[j] - row_max);
                sum += scores[j];
            }
            for (int j = 0; j < N; ++j) scores[j] /= sum;

            // weighted sum
            for (int c = 0; c < d; ++c) {
                float acc = 0.f;
                for (int j = 0; j < N; ++j)
                    acc += scores[j] * V[base + j*d+c];
                O[base + i*d+c] = acc;
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════
//  SELF-TEST
// ═════════════════════════════════════════════════════════════
static void fill_random(float* p, int n) {
    for (int i = 0; i < n; ++i)
        p[i] = ((float)rand() / RAND_MAX) * 2.f - 1.f;
}

int main(int argc, char** argv)
{
    bool causal = (argc > 1 && strcmp(argv[1], "causal") == 0);

    printf("Flash Attention forward  —  tiling + online softmax\n");
    printf("  Br=%d  Bc=%d  d=%d  causal=%s\n\n",
           Br, Bc, D_MAX, causal ? "yes" : "no");

    // ── problem dimensions ────────────────────────────────────
    int B = 2;          // batch size
    int H = 4;          // number of heads
    int N = 256;        // sequence length (must be multiple of Br for this test)
    int d = D_MAX;      // head dimension

    assert(N % Br == 0 && "For this test keep N a multiple of Br");

    size_t qkv_sz = (size_t)B * H * N * d * sizeof(float);
    size_t l_sz   = (size_t)B * H * N     * sizeof(float);

    // ── host allocation + random init ────────────────────────
    float* hQ  = (float*)malloc(qkv_sz);
    float* hK  = (float*)malloc(qkv_sz);
    float* hV  = (float*)malloc(qkv_sz);
    float* hO  = (float*)calloc(B*H*N*d, sizeof(float));   // GPU output
    float* hL  = (float*)calloc(B*H*N,   sizeof(float));
    float* hO_ref = (float*)calloc(B*H*N*d, sizeof(float)); // CPU reference

    srand(42);
    fill_random(hQ, B*H*N*d);
    fill_random(hK, B*H*N*d);
    fill_random(hV, B*H*N*d);

    // ── CPU reference ─────────────────────────────────────────
    printf("Running CPU reference... ");
    fflush(stdout);
    cpu_attention_reference(hQ, hK, hV, hO_ref, B, H, N, d, causal);
    printf("done.\n");

    // ── device allocation + copy ──────────────────────────────
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

    // ── GPU kernel ────────────────────────────────────────────
    printf("Running GPU Flash Attention... ");
    fflush(stdout);

    // Warm-up
    flash_attention_forward(dQ, dK, dV, dO, dL, B, H, N, d, causal);

    // Timed run
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

    // ── copy result back + correctness check ──────────────────
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
    printf("  %s\n", max_err < 1e-4f ? "PASS" : "FAIL — check tile params");

    // ── cleanup ───────────────────────────────────────────────
    cudaFree(dQ); cudaFree(dK); cudaFree(dV);
    cudaFree(dO); cudaFree(dL);
    free(hQ); free(hK); free(hV);
    free(hO); free(hL); free(hO_ref);

    cudaEventDestroy(t0);
    cudaEventDestroy(t1);

    return 0;
}
