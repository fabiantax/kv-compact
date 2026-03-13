// KV Cache Compaction — ROCm/HIP Accelerated Kernels
//
// GPU-accelerated matrix multiply for the attention scoring hotspot in
// Step 1 (Section 3.1) of the compaction algorithm from:
//   "Fast KV Compaction via Attention Matching" (Zweiger et al., 2026)
//   https://arxiv.org/abs/2602.16284
//
// The dominant cost is computing Q_ref @ K^T / sqrt(d_k) for all heads,
// which is an (n_q × d_k) * (T × d_k)^T matmul. On CPU this is O(n_q * T * d_k)
// per head per layer; GPU acceleration helps when T is large (>1K tokens).
//
// Target: AMD Ryzen AI 395 Pro Max with integrated RDNA 3.5 GPU (8060S)
// The integrated GPU shares system memory, so no explicit D↔H copies needed
// for unified memory allocations (hipMallocManaged).

#include <hip/hip_runtime.h>
#include "kv-compact-accel.h"

#include <cstdio>
#include <cstdlib>

// ============================================================================
// HIP Kernels
// ============================================================================

// Matrix multiply: C = A * B^T
// A is (m x k), B is (n x k), C is (m x n)
// Each thread computes one element of C
__global__ void kernel_mat_mul_ABt(
        const float * __restrict__ A,
        const float * __restrict__ B,
        float * __restrict__ C,
        int m, int n, int k, float scale) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[col * k + l];
        }
        C[row * n + col] = sum * scale;
    }
}

// Tiled matrix multiply for better cache utilization
// Uses shared memory tiles for both A and B.
// Tiles are padded to TILE_SIZE+1 columns to avoid shared memory bank conflicts
// on RDNA 3.5 (32 banks, 4-byte stride → consecutive columns hit same bank).
#define TILE_SIZE 16

__global__ void kernel_mat_mul_ABt_tiled(
        const float * __restrict__ A,
        const float * __restrict__ B,
        float * __restrict__ C,
        int m, int n, int k, float scale) {
    __shared__ float tile_A[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float tile_B[TILE_SIZE][TILE_SIZE + 1];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (k + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_col = t * TILE_SIZE + threadIdx.y;

        tile_A[threadIdx.y][threadIdx.x] = (row < m && a_col < k) ? A[row * k + a_col] : 0.0f;
        tile_B[threadIdx.y][threadIdx.x] = (col < n && b_col < k) ? B[col * k + b_col] : 0.0f;

        __syncthreads();

        for (int l = 0; l < TILE_SIZE; l++) {
            sum += tile_A[threadIdx.y][l] * tile_B[l][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < m && col < n) {
        C[row * n + col] = sum * scale;
    }
}

// ============================================================================
// Strided kernel for interleaved multi-head layout
// ============================================================================

// Per-head matmul on interleaved data:
//   Q_ref: [n_q × n_embd_k], K_all: [T × n_embd_k]
//   For head h, Q_h row i starts at Q_ref[i * n_embd_k + h * d_k]
//   For head h, K_h row j starts at K_all[j * n_embd_k + h * d_k]
//   scores_out[h * n_q * T + qi * T + ki] = dot(Q_h[qi], K_h[ki]) * scale
//
// Grid: (ceil(T/16), ceil(n_q/16), n_head_kv)
__global__ void kernel_score_all_heads(
        const float * __restrict__ Q_ref,
        const float * __restrict__ K_all,
        float * __restrict__ scores_out,
        int n_q, int T, int n_head_kv, int d_k,
        int n_embd_k, float scale) {
    int ki  = blockIdx.x * TILE_SIZE + threadIdx.x;
    int qi  = blockIdx.y * TILE_SIZE + threadIdx.y;
    int h   = blockIdx.z;

    if (qi < n_q && ki < T) {
        const float * q_row = Q_ref + qi * n_embd_k + h * d_k;
        const float * k_row = K_all + ki * n_embd_k + h * d_k;
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) {
            dot += q_row[d] * k_row[d];
        }
        scores_out[(size_t)h * n_q * T + qi * T + ki] = dot * scale;
    }
}

// ============================================================================
// Value refit target kernel: Y_h = attn_weights_h @ V_h (interleaved V)
// ============================================================================

// Computes Y[h][qi * d_v + d] = sum_ki( W[h][qi * T + ki] * V[ki * n_embd_v + h * d_v + d] )
// for all heads in a single dispatch.
//
// W (attn_weights): [n_head_kv][n_q × T] — contiguous per-head blocks
// V_all:            [T × n_embd_v]         — interleaved, stride = n_embd_v
// Y_out:            [n_head_kv][n_q × d_v] — contiguous per-head blocks
//
// Grid: (ceil(d_v/16), ceil(n_q/16), n_head_kv)
__global__ void kernel_refit_target_all_heads(
        const float * __restrict__ W,       // [n_head_kv * n_q * T]
        const float * __restrict__ V_all,   // [T * n_embd_v]
        float * __restrict__ Y_out,         // [n_head_kv * n_q * d_v]
        int n_q, int T, int n_head_kv, int d_v, int n_embd_v) {
    int d   = blockIdx.x * TILE_SIZE + threadIdx.x;  // column in d_v
    int qi  = blockIdx.y * TILE_SIZE + threadIdx.y;  // row in n_q
    int h   = blockIdx.z;                            // head

    if (qi < n_q && d < d_v) {
        const float * w_row = W + (size_t)h * n_q * T + qi * T;
        float sum = 0.0f;
        for (int ki = 0; ki < T; ki++) {
            sum += w_row[ki] * V_all[ki * n_embd_v + h * d_v + d];
        }
        Y_out[(size_t)h * n_q * d_v + qi * d_v + d] = sum;
    }
}

// ============================================================================
// Persistent buffer pool
// ============================================================================

// Avoids per-call hipMallocManaged/hipFree overhead. Buffers grow as needed
// and are reused across calls. On APU, managed memory is backed by the same
// physical LPDDR5X so the "allocation" is just a virtual address reservation.
struct hip_buffer_pool {
    float * q_buf;       // Q or W (attn_weights) input
    float * k_buf;       // K or V input
    float * scores_buf;  // scores or Y output
    float * aux_buf;     // auxiliary (e.g. W for value refit when q_buf holds Q)
    size_t  q_cap;       // allocated capacity in bytes
    size_t  k_cap;
    size_t  scores_cap;
    size_t  aux_cap;
};

static hip_buffer_pool g_pool = {nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0};

// Ensure pool buffer is at least `needed` bytes. Only reallocates if current
// capacity is insufficient.
static hipError_t pool_ensure(float ** buf, size_t * cap, size_t needed) {
    if (*cap >= needed) return hipSuccess;
    if (*buf) hipFree(*buf);
    hipError_t err = hipMallocManaged(buf, needed);
    *cap = (err == hipSuccess) ? needed : 0;
    return err;
}

// ============================================================================
// Public API (C-linkage)
// ============================================================================

static bool hip_initialized = false;
static bool hip_available = false;

static void ensure_hip_init() {
    if (hip_initialized) return;
    hip_initialized = true;

    int device_count = 0;
    hipError_t err = hipGetDeviceCount(&device_count);
    if (err != hipSuccess || device_count == 0) {
        hip_available = false;
        return;
    }

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, 0);
    fprintf(stderr, "[kv-compact] HIP device: %s (CUs: %d, clock: %d MHz)\n",
            props.name, props.multiProcessorCount, props.clockRate / 1000);

    hip_available = true;
}

extern "C" {

int kv_compact_hip_available(void) {
    ensure_hip_init();
    return hip_available ? 1 : 0;
}

int kv_compact_hip_mat_mul_ABt_scaled(
        const float * A, const float * B, float * C,
        int m, int n, int k, float scale) {
    ensure_hip_init();
    if (!hip_available) return -1;

    size_t size_A = (size_t)m * k * sizeof(float);
    size_t size_B = (size_t)n * k * sizeof(float);
    size_t size_C = (size_t)m * n * sizeof(float);

    // Use persistent pool buffers instead of per-call alloc
    if (pool_ensure(&g_pool.q_buf, &g_pool.q_cap, size_A) != hipSuccess) return -1;
    if (pool_ensure(&g_pool.k_buf, &g_pool.k_cap, size_B) != hipSuccess) return -1;
    if (pool_ensure(&g_pool.scores_buf, &g_pool.scores_cap, size_C) != hipSuccess) return -1;

    memcpy(g_pool.q_buf, A, size_A);
    memcpy(g_pool.k_buf, B, size_B);

    // Use tiled kernel for larger problems, simple kernel for small ones
    if (m >= TILE_SIZE && n >= TILE_SIZE) {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((n + TILE_SIZE - 1) / TILE_SIZE,
                  (m + TILE_SIZE - 1) / TILE_SIZE);
        kernel_mat_mul_ABt_tiled<<<grid, block>>>(
            g_pool.q_buf, g_pool.k_buf, g_pool.scores_buf, m, n, k, scale);
    } else {
        dim3 block(16, 16);
        dim3 grid((n + 15) / 16, (m + 15) / 16);
        kernel_mat_mul_ABt<<<grid, block>>>(
            g_pool.q_buf, g_pool.k_buf, g_pool.scores_buf, m, n, k, scale);
    }

    hipDeviceSynchronize();
    memcpy(C, g_pool.scores_buf, size_C);

    return 0;
}

int kv_compact_hip_score_all_heads(
        const float * Q_ref, const float * K_all, float ** scores_out,
        int n_q, int T, int n_head_kv, int d_k, float scale) {
    ensure_hip_init();
    if (!hip_available) return -1;

    int n_embd_k = n_head_kv * d_k;
    size_t q_bytes = (size_t)n_q * n_embd_k * sizeof(float);
    size_t k_bytes = (size_t)T * n_embd_k * sizeof(float);
    size_t scores_bytes = (size_t)n_head_kv * n_q * T * sizeof(float);

    // Grow pool buffers if needed (no-op if already large enough)
    if (pool_ensure(&g_pool.q_buf, &g_pool.q_cap, q_bytes) != hipSuccess) return -1;
    if (pool_ensure(&g_pool.k_buf, &g_pool.k_cap, k_bytes) != hipSuccess) return -1;
    if (pool_ensure(&g_pool.scores_buf, &g_pool.scores_cap, scores_bytes) != hipSuccess) return -1;

    // Copy interleaved Q and K into managed memory (zero-copy on APU)
    memcpy(g_pool.q_buf, Q_ref, q_bytes);
    memcpy(g_pool.k_buf, K_all, k_bytes);

    // Launch one 3D grid: (T, n_q, n_head_kv)
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid((T + TILE_SIZE - 1) / TILE_SIZE,
              (n_q + TILE_SIZE - 1) / TILE_SIZE,
              n_head_kv);
    kernel_score_all_heads<<<grid, block>>>(
        g_pool.q_buf, g_pool.k_buf, g_pool.scores_buf,
        n_q, T, n_head_kv, d_k, n_embd_k, scale);

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) return -1;

    // Copy results into caller's per-head score buffers
    for (int h = 0; h < n_head_kv; h++) {
        memcpy(scores_out[h],
               g_pool.scores_buf + (size_t)h * n_q * T,
               (size_t)n_q * T * sizeof(float));
    }

    return 0;
}

int kv_compact_hip_refit_target_all_heads(
        const float * const * W_per_head,
        const float * V_all,
        float ** Y_out,
        int n_q, int T, int n_head_kv, int d_v) {
    ensure_hip_init();
    if (!hip_available) return -1;

    int n_embd_v = n_head_kv * d_v;
    size_t w_bytes = (size_t)n_head_kv * n_q * T * sizeof(float);
    size_t v_bytes = (size_t)T * n_embd_v * sizeof(float);
    size_t y_bytes = (size_t)n_head_kv * n_q * d_v * sizeof(float);

    // Grow pool buffers as needed
    if (pool_ensure(&g_pool.aux_buf, &g_pool.aux_cap, w_bytes) != hipSuccess) return -1;
    if (pool_ensure(&g_pool.k_buf, &g_pool.k_cap, v_bytes) != hipSuccess) return -1;
    if (pool_ensure(&g_pool.scores_buf, &g_pool.scores_cap, y_bytes) != hipSuccess) return -1;

    // Pack per-head W arrays into contiguous buffer
    for (int h = 0; h < n_head_kv; h++) {
        memcpy(g_pool.aux_buf + (size_t)h * n_q * T,
               W_per_head[h],
               (size_t)n_q * T * sizeof(float));
    }

    // Copy interleaved V (zero-copy on APU)
    memcpy(g_pool.k_buf, V_all, v_bytes);

    // Launch one 3D grid: (d_v, n_q, n_head_kv)
    dim3 block(TILE_SIZE, TILE_SIZE, 1);
    dim3 grid((d_v + TILE_SIZE - 1) / TILE_SIZE,
              (n_q + TILE_SIZE - 1) / TILE_SIZE,
              n_head_kv);
    kernel_refit_target_all_heads<<<grid, block>>>(
        g_pool.aux_buf, g_pool.k_buf, g_pool.scores_buf,
        n_q, T, n_head_kv, d_v, n_embd_v);

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) return -1;

    // Copy results into caller's per-head Y buffers
    for (int h = 0; h < n_head_kv; h++) {
        memcpy(Y_out[h],
               g_pool.scores_buf + (size_t)h * n_q * d_v,
               (size_t)n_q * d_v * sizeof(float));
    }

    return 0;
}

void kv_compact_hip_pool_free(void) {
    if (g_pool.q_buf)      { hipFree(g_pool.q_buf);      g_pool.q_buf = nullptr;      g_pool.q_cap = 0; }
    if (g_pool.k_buf)      { hipFree(g_pool.k_buf);      g_pool.k_buf = nullptr;      g_pool.k_cap = 0; }
    if (g_pool.scores_buf) { hipFree(g_pool.scores_buf); g_pool.scores_buf = nullptr; g_pool.scores_cap = 0; }
    if (g_pool.aux_buf)    { hipFree(g_pool.aux_buf);    g_pool.aux_buf = nullptr;    g_pool.aux_cap = 0; }
}

} // extern "C"
