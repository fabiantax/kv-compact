// KV Cache Compaction — C Library API
//
// Programmatic interface for KV cache compaction via Attention Matching.
// Designed for integration into serving frameworks that manage KV caches
// at runtime (e.g., auto-compact when context grows too large).
//
// US-8: "Expose compaction as a library API (not just a CLI tool)"
//
// Usage:
//   kv_compact_params params = kv_compact_params_default();
//   params.target_ratio = 0.5f;  // keep 50%
//
//   kv_compact_result result = {};
//   int rc = kv_compact(K, V, Q_ref, T, n_q, n_head_kv, d_k, d_v, &params, &result);
//
//   // result.selected_indices[0..result.t-1]  — which positions were kept
//   // result.beta[h][0..result.t-1]           — per-head attention biases
//   // result.C_v[h][0..result.t*d_v-1]        — per-head refitted values
//   // result.stats                             — quality and timing metrics
//
//   kv_compact_result_free(&result);

#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Parameters
// ============================================================================

typedef struct kv_compact_params {
    float  target_ratio;       // fraction of tokens to keep (0.0, 1.0)
    int    target_count;       // explicit target count (overrides ratio if > 0)
    int    use_sensitivity;    // weight key selection by per-head sensitivity
    float  ridge;              // ridge regularization for least-squares (default: 1e-6)
    int    nnls_max_iter;      // max NNLS iterations (default: 200)
} kv_compact_params;

// ============================================================================
// Result
// ============================================================================

typedef struct kv_compact_stats {
    float  avg_cosine_sim;     // average cosine similarity (compacted vs original)
    float  avg_mse;            // average MSE across heads
    float  avg_agreement;      // fraction of queries with matching argmax
    double elapsed_ms;         // total compaction wall-clock time
    double scoring_ms;         // time for attention scoring phase
    double nnls_ms;            // time for NNLS + least-squares phase
} kv_compact_stats;

typedef struct kv_compact_result {
    int     t;                  // number of tokens kept
    int     n_head_kv;          // number of KV heads

    int   * selected_indices;   // [t] which original positions were kept (sorted)

    // Per-head arrays: beta[h] points to t floats, C_v[h] points to t*d_v floats
    float ** beta;              // [n_head_kv] pointers to per-head beta arrays
    float ** C_v;               // [n_head_kv] pointers to per-head C_v arrays

    kv_compact_stats stats;     // quality and timing metrics
} kv_compact_result;

// ============================================================================
// API functions
// ============================================================================

// Return default parameters with sensible values
kv_compact_params kv_compact_params_default(void);

// Compact KV cache for all heads within a single layer (or fused multi-layer).
//
//   K_all:      [T × n_head_kv × d_k] row-major, all heads concatenated per token
//   V_all:      [T × n_head_kv × d_v] row-major, all heads concatenated per token
//   Q_ref_all:  [n_q × n_head_kv × d_k] reference queries, all heads concatenated
//   T:          number of tokens (cache positions)
//   n_q:        number of reference queries
//   n_head_kv:  number of KV heads
//   d_k:        key dimension per head
//   d_v:        value dimension per head
//   params:     compaction parameters (NULL for defaults)
//   result:     output (caller must call kv_compact_result_free when done)
//
// Returns 0 on success, non-zero on error.
int kv_compact(
    const float * K_all,
    const float * V_all,
    const float * Q_ref_all,
    int T, int n_q, int n_head_kv, int d_k, int d_v,
    const kv_compact_params * params,
    kv_compact_result * result);

// Free all memory allocated by kv_compact()
void kv_compact_result_free(kv_compact_result * result);

#ifdef __cplusplus
}
#endif
