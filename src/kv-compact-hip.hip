// KV Cache Compaction — ROCm/HIP Accelerated Kernels
//
// GPU-accelerated matrix operations for the compaction algorithm from:
//   "Fast KV Compaction via Attention Matching" (Zweiger et al., 2026)
//   https://arxiv.org/abs/2602.16284
//
// Two acceleration tiers:
//   1. rocBLAS strided-batched GEMM — processes all heads in a single library
//      call, leveraging Tensile's auto-tuned kernels for the target GPU.
//   2. Custom HIP kernels — fallback for small matrices or if rocBLAS init fails.
//
// The interleaved Q/K/V layout maps directly to strided-batched GEMM strides:
//   Q_ref[n_q × n_embd_k] → batch of n_head_kv matrices (n_q × d_k), stride = d_k
//   K_all[T   × n_embd_k] → batch of n_head_kv matrices (T   × d_k), stride = d_k
//   Row-major (n_q × T) output == column-major (T × n_q) — no reshuffling needed.
//
// Target: AMD Ryzen AI 395 Pro Max — 8060S RDNA 3.5 iGPU (gfx1151)
// APU unified memory: hipMallocManaged() has near-zero overhead.

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include "kv-compact-accel.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ============================================================================
// Custom HIP Kernels (fallback for small matrices / rocBLAS unavailable)
// ============================================================================

// Matrix multiply: C = A * B^T
// A is (m x k), B is (n x k), C is (m x n)
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

// Tiled matrix multiply with shared memory (bank-conflict-free on RDNA 3.5)
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

// Per-head scoring kernel (fallback when rocBLAS unavailable)
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

// Per-head value refit kernel (fallback when rocBLAS unavailable)
__global__ void kernel_refit_target_all_heads(
        const float * __restrict__ W,
        const float * __restrict__ V_all,
        float * __restrict__ Y_out,
        int n_q, int T, int n_head_kv, int d_v, int n_embd_v) {
    int d   = blockIdx.x * TILE_SIZE + threadIdx.x;
    int qi  = blockIdx.y * TILE_SIZE + threadIdx.y;
    int h   = blockIdx.z;

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

struct hip_buffer_pool {
    float * q_buf;       // Q or W (attn_weights) input
    float * k_buf;       // K or V input
    float * scores_buf;  // scores or Y output
    float * aux_buf;     // auxiliary (W packing for value refit)
    size_t  q_cap;
    size_t  k_cap;
    size_t  scores_cap;
    size_t  aux_cap;
};

static hip_buffer_pool g_pool = {nullptr, nullptr, nullptr, nullptr, 0, 0, 0, 0};

static hipError_t pool_ensure(float ** buf, size_t * cap, size_t needed) {
    if (*cap >= needed) return hipSuccess;
    if (*buf) hipFree(*buf);
    hipError_t err = hipMallocManaged(buf, needed);
    *cap = (err == hipSuccess) ? needed : 0;
    return err;
}

// ============================================================================
// rocBLAS handle (lazy-initialized, lives for process lifetime)
// ============================================================================

static rocblas_handle g_rocblas_handle = nullptr;
static bool g_rocblas_available = false;

static void ensure_rocblas_init() {
    if (g_rocblas_handle) return;  // already initialized (success or fail)

    // Disable rocBLAS's internal hipBLASLt path on gfx11 — it regresses for FP32
    // batched GEMM. Standard Tensile kernels are faster. Must be set BEFORE
    // rocblas_create_handle, which reads the env var during initialization.
    setenv("ROCBLAS_USE_HIPBLASLT", "0", 0);  // don't override if already set

    rocblas_status status = rocblas_create_handle(&g_rocblas_handle);
    if (status != rocblas_status_success) {
        fprintf(stderr, "[kv-compact] rocBLAS init failed (status %d), using custom kernels\n", status);
        g_rocblas_handle = nullptr;
        g_rocblas_available = false;
        return;
    }

    g_rocblas_available = true;
    fprintf(stderr, "[kv-compact] rocBLAS initialized — strided-batched GEMM enabled\n");
}

// Minimum total elements (across all heads) to prefer rocBLAS over custom kernels.
// Below this threshold, kernel launch overhead dominates.
// Empirically: rocBLAS wins when batch * m * n * k > ~32K elements.
#define ROCBLAS_MIN_ELEMENTS 32768

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

    // Initialize rocBLAS eagerly so first scoring call isn't slow
    ensure_rocblas_init();
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

    if (pool_ensure(&g_pool.q_buf, &g_pool.q_cap, size_A) != hipSuccess) return -1;
    if (pool_ensure(&g_pool.k_buf, &g_pool.k_cap, size_B) != hipSuccess) return -1;
    if (pool_ensure(&g_pool.scores_buf, &g_pool.scores_cap, size_C) != hipSuccess) return -1;

    memcpy(g_pool.q_buf, A, size_A);
    memcpy(g_pool.k_buf, B, size_B);

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

// ---------------------------------------------------------------------------
// Multi-head attention scoring: scores_h = Q_h @ K_h^T * scale
//
// rocBLAS path: single strided-batched GEMM over all heads.
//
// Row-major → column-major mapping for C = A * B^T:
//   In col-major terms:  C^T(T × n_q) = K^T(T × d_k) * Q^T(d_k × n_q)
//   rocblas_sgemm(T, N, T, n_q, d_k, scale, K, n_embd_k, Q, n_embd_k, 0, S, T)
//
// The interleaved layout provides natural strides:
//   stride_K = stride_Q = d_k  (head-to-head offset within a row)
//   stride_S = n_q * T         (contiguous per-head score blocks)
// ---------------------------------------------------------------------------
int kv_compact_hip_score_all_heads(
        const float * Q_ref, const float * K_all, float ** scores_out,
        int n_q, int T, int n_head_kv, int d_k, float scale) {
    ensure_hip_init();
    if (!hip_available) return -1;

    int n_embd_k = n_head_kv * d_k;
    size_t q_bytes = (size_t)n_q * n_embd_k * sizeof(float);
    size_t k_bytes = (size_t)T * n_embd_k * sizeof(float);
    size_t scores_bytes = (size_t)n_head_kv * n_q * T * sizeof(float);

    if (pool_ensure(&g_pool.q_buf, &g_pool.q_cap, q_bytes) != hipSuccess) return -1;
    if (pool_ensure(&g_pool.k_buf, &g_pool.k_cap, k_bytes) != hipSuccess) return -1;
    if (pool_ensure(&g_pool.scores_buf, &g_pool.scores_cap, scores_bytes) != hipSuccess) return -1;

    memcpy(g_pool.q_buf, Q_ref, q_bytes);
    memcpy(g_pool.k_buf, K_all, k_bytes);

    int64_t total_elements = (int64_t)n_head_kv * n_q * T * d_k;
    bool use_rocblas = g_rocblas_available && total_elements >= ROCBLAS_MIN_ELEMENTS;

    if (use_rocblas) {
        // rocBLAS strided-batched GEMM: all heads in one call
        //
        // Column-major interpretation of row-major data:
        //   Q_h row-major (n_q × d_k) → col-major (d_k × n_q), ld = n_embd_k
        //   K_h row-major (T   × d_k) → col-major (d_k × T),   ld = n_embd_k
        //
        // We want scores_h = Q_h * K_h^T (row-major)
        // ⟹ scores_h^T = K_h * Q_h^T (col-major result)
        // ⟹ gemm(T, N, T, n_q, d_k, α, K_cm, ld_k, Q_cm, ld_q, β, S_cm, T)
        //
        // K_cm is (d_k × T), transposed → (T × d_k): rows of K_h ✓
        // Q_cm is (d_k × n_q), no transpose → Q_h^T: cols are query rows ✓
        // S_cm is (T × n_q) = scores_h^T, which in row-major IS scores_h ✓
        float zero = 0.0f;
        rocblas_status status = rocblas_sgemm_strided_batched(
            g_rocblas_handle,
            rocblas_operation_transpose,    // op(A) = K^T → K rows
            rocblas_operation_none,         // op(B) = Q^T (already transposed in col-major)
            T,                             // m: rows of result
            n_q,                           // n: cols of result
            d_k,                           // k: inner dimension
            &scale,                        // alpha
            g_pool.k_buf,                  // A = K (col-major d_k × T per head)
            n_embd_k,                      // lda (row stride of interleaved K)
            (rocblas_stride)d_k,           // stride_a (head-to-head offset)
            g_pool.q_buf,                  // B = Q (col-major d_k × n_q per head)
            n_embd_k,                      // ldb (row stride of interleaved Q)
            (rocblas_stride)d_k,           // stride_b (head-to-head offset)
            &zero,                         // beta
            g_pool.scores_buf,             // C = scores (T × n_q per head, col-major)
            T,                             // ldc
            (rocblas_stride)n_q * T,       // stride_c (contiguous per-head blocks)
            n_head_kv);                    // batch_count

        if (status != rocblas_status_success) {
            fprintf(stderr, "[kv-compact] rocBLAS sgemm_strided_batched failed (status %d), "
                    "falling back to custom kernel\n", status);
            use_rocblas = false;
        }
    }

    if (!use_rocblas) {
        // Fallback: custom kernel, one thread per element
        dim3 block(TILE_SIZE, TILE_SIZE, 1);
        dim3 grid((T + TILE_SIZE - 1) / TILE_SIZE,
                  (n_q + TILE_SIZE - 1) / TILE_SIZE,
                  n_head_kv);
        kernel_score_all_heads<<<grid, block>>>(
            g_pool.q_buf, g_pool.k_buf, g_pool.scores_buf,
            n_q, T, n_head_kv, d_k, n_embd_k, scale);
    }

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) return -1;

    for (int h = 0; h < n_head_kv; h++) {
        memcpy(scores_out[h],
               g_pool.scores_buf + (size_t)h * n_q * T,
               (size_t)n_q * T * sizeof(float));
    }

    return 0;
}

// ---------------------------------------------------------------------------
// Multi-head value refit target: Y_h = W_h @ V_h
//
// rocBLAS path: single strided-batched GEMM.
//
// Row-major → column-major mapping for C = W * V (where V is interleaved):
//   Y_h^T(d_v × n_q) = V_h^T(d_v × T) * W_h^T(T × n_q)
//   rocblas_sgemm(N, N, d_v, n_q, T, 1.0, V_cm, n_embd_v, W_cm, T, 0, Y_cm, d_v)
//
// Strides:
//   stride_V = d_v           (interleaved V heads)
//   stride_W = n_q * T       (contiguous W blocks, already packed)
//   stride_Y = n_q * d_v     (contiguous output blocks)
// ---------------------------------------------------------------------------
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

    if (pool_ensure(&g_pool.aux_buf, &g_pool.aux_cap, w_bytes) != hipSuccess) return -1;
    if (pool_ensure(&g_pool.k_buf, &g_pool.k_cap, v_bytes) != hipSuccess) return -1;
    if (pool_ensure(&g_pool.scores_buf, &g_pool.scores_cap, y_bytes) != hipSuccess) return -1;

    // Pack per-head W arrays into contiguous buffer for batched GEMM
    for (int h = 0; h < n_head_kv; h++) {
        memcpy(g_pool.aux_buf + (size_t)h * n_q * T,
               W_per_head[h],
               (size_t)n_q * T * sizeof(float));
    }

    memcpy(g_pool.k_buf, V_all, v_bytes);

    int64_t total_elements = (int64_t)n_head_kv * n_q * T * d_v;
    bool use_rocblas = g_rocblas_available && total_elements >= ROCBLAS_MIN_ELEMENTS;

    if (use_rocblas) {
        // rocBLAS strided-batched GEMM: all heads in one call
        //
        // We want Y_h = W_h * V_h (row-major), so:
        //   Y_h^T = V_h^T * W_h^T (col-major)
        //
        // V_h row-major (T × d_v) → col-major (d_v × T), ld = n_embd_v
        // W_h row-major (n_q × T) → col-major (T × n_q), ld = T
        // Y_h row-major (n_q × d_v) → col-major (d_v × n_q), ld = d_v
        //
        // gemm(N, N, d_v, n_q, T):
        //   op(A) = V_cm (d_v × T), op(B) = W_cm (T × n_q) → C = (d_v × n_q)
        float one = 1.0f, zero = 0.0f;
        rocblas_status status = rocblas_sgemm_strided_batched(
            g_rocblas_handle,
            rocblas_operation_none,         // op(A) = V^T (already in col-major)
            rocblas_operation_none,         // op(B) = W^T (already in col-major)
            d_v,                           // m: rows of result
            n_q,                           // n: cols of result
            T,                             // k: inner dimension
            &one,                          // alpha
            g_pool.k_buf,                  // A = V (col-major d_v × T per head)
            n_embd_v,                      // lda (row stride of interleaved V)
            (rocblas_stride)d_v,           // stride_a (head-to-head offset)
            g_pool.aux_buf,                // B = W (col-major T × n_q per head)
            T,                             // ldb
            (rocblas_stride)n_q * T,       // stride_b (contiguous W blocks)
            &zero,                         // beta
            g_pool.scores_buf,             // C = Y (d_v × n_q per head, col-major)
            d_v,                           // ldc
            (rocblas_stride)n_q * d_v,     // stride_c
            n_head_kv);                    // batch_count

        if (status != rocblas_status_success) {
            fprintf(stderr, "[kv-compact] rocBLAS value refit failed (status %d), "
                    "falling back to custom kernel\n", status);
            use_rocblas = false;
        }
    }

    if (!use_rocblas) {
        // Fallback: custom kernel
        dim3 block(TILE_SIZE, TILE_SIZE, 1);
        dim3 grid((d_v + TILE_SIZE - 1) / TILE_SIZE,
                  (n_q + TILE_SIZE - 1) / TILE_SIZE,
                  n_head_kv);
        kernel_refit_target_all_heads<<<grid, block>>>(
            g_pool.aux_buf, g_pool.k_buf, g_pool.scores_buf,
            n_q, T, n_head_kv, d_v, n_embd_v);
    }

    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) return -1;

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

    if (g_rocblas_handle) {
        rocblas_destroy_handle(g_rocblas_handle);
        g_rocblas_handle = nullptr;
        g_rocblas_available = false;
    }
}

} // extern "C"
