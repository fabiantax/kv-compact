// KV Cache Compaction — Accelerated Compute Interface
//
// Provides a unified interface for GPU-accelerated or CPU-fallback
// matrix operations used by the compaction algorithm.
//
// When KV_COMPACT_HIP is defined, routes to ROCm/HIP kernels.
// Otherwise, uses the CPU implementations from kv-compact-math.h.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// Check if HIP acceleration is available at runtime
// Returns 1 if HIP device is present and initialized, 0 otherwise
int kv_compact_hip_available(void);

// GPU-accelerated: C = A * B^T * scale
// A is (m x k), B is (n x k), C is (m x n)
// Returns 0 on success, -1 if HIP not available
int kv_compact_hip_mat_mul_ABt_scaled(
        const float * A, const float * B, float * C,
        int m, int n, int k, float scale);

#ifdef __cplusplus
}
#endif

#ifndef KV_COMPACT_HIP
// CPU-only stub implementations when HIP is not compiled in
#ifdef KV_COMPACT_ACCEL_IMPL

static int kv_compact_hip_available(void) { return 0; }

static int kv_compact_hip_mat_mul_ABt_scaled(
        const float * A, const float * B, float * C,
        int m, int n, int k, float scale) {
    (void)A; (void)B; (void)C; (void)m; (void)n; (void)k; (void)scale;
    return -1;  // not available
}

#endif // KV_COMPACT_ACCEL_IMPL
#endif // !KV_COMPACT_HIP
