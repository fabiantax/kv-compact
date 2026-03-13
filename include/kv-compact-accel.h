// KV Cache Compaction — Accelerated Compute Interface
//
// Provides a unified interface for GPU-accelerated or CPU-fallback
// matrix operations used by the compaction algorithm.
//
// When KV_COMPACT_HIP is defined, routes to ROCm/HIP kernels.
// Otherwise, provides inline CPU stubs that return "not available".
//
// Usage in exactly ONE .cpp file:
//   #define KV_COMPACT_ACCEL_IMPL
//   #include "kv-compact-accel.h"

#pragma once

#ifdef KV_COMPACT_HIP
// ---- HIP is compiled in: declare extern functions (defined in .hip file) ----

#ifdef __cplusplus
extern "C" {
#endif

int kv_compact_hip_available(void);

int kv_compact_hip_mat_mul_ABt_scaled(
        const float * A, const float * B, float * C,
        int m, int n, int k, float scale);

// GPU-accelerated multi-head attention scoring with persistent buffer pool.
//
// Computes scores[h][qi * T + ki] = Q_ref[qi, h, :] . K[ki, h, :] * scale
// for all heads in a single call. Q_ref and K_all use interleaved layout:
//   Q_ref: [n_q × n_embd_k] where n_embd_k = n_head_kv * d_k
//   K_all: [T   × n_embd_k]
//   scores_out: pre-allocated [n_head_kv][n_q * T] (array of per-head buffers)
//
// Uses a persistent GPU buffer pool — no per-call hipMalloc/hipFree.
// Returns 0 on success, -1 if HIP not available.
int kv_compact_hip_score_all_heads(
        const float * Q_ref, const float * K_all, float ** scores_out,
        int n_q, int T, int n_head_kv, int d_k, float scale);

// Release persistent GPU buffer pool (call at shutdown or when done).
void kv_compact_hip_pool_free(void);

#ifdef __cplusplus
}
#endif

#else // !KV_COMPACT_HIP
// ---- No HIP: provide inline stub implementations ----

#ifdef KV_COMPACT_ACCEL_IMPL
// Exactly one translation unit defines these (via KV_COMPACT_ACCEL_IMPL)

inline int kv_compact_hip_available(void) { return 0; }

inline int kv_compact_hip_mat_mul_ABt_scaled(
        const float * A, const float * B, float * C,
        int m, int n, int k, float scale) {
    (void)A; (void)B; (void)C; (void)m; (void)n; (void)k; (void)scale;
    return -1;
}

inline int kv_compact_hip_score_all_heads(
        const float * Q_ref, const float * K_all, float ** scores_out,
        int n_q, int T, int n_head_kv, int d_k, float scale) {
    (void)Q_ref; (void)K_all; (void)scores_out;
    (void)n_q; (void)T; (void)n_head_kv; (void)d_k; (void)scale;
    return -1;
}

inline void kv_compact_hip_pool_free(void) {}

#else
// Other translation units just get declarations
#ifdef __cplusplus
extern "C" {
#endif

int kv_compact_hip_available(void);
int kv_compact_hip_mat_mul_ABt_scaled(
        const float * A, const float * B, float * C,
        int m, int n, int k, float scale);
int kv_compact_hip_score_all_heads(
        const float * Q_ref, const float * K_all, float ** scores_out,
        int n_q, int T, int n_head_kv, int d_k, float scale);
void kv_compact_hip_pool_free(void);

#ifdef __cplusplus
}
#endif
#endif // KV_COMPACT_ACCEL_IMPL

#endif // KV_COMPACT_HIP
