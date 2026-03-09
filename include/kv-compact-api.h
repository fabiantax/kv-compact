// KV Cache Compaction API — drop-in replacement for context shift
//
// Provides a single function that compacts a sequence's KV cache in-place
// using attention matching (Zweiger et al., 2026).
//
// Usage:
//   #include "kv-compact-api.h"
//
//   // When context is full, instead of discarding tokens:
//   kv_compact_params params = kv_compact_params_default();
//   int new_size = kv_compact_sequence(ctx, seq_id, params);

#pragma once

#include "llama.h"

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

struct kv_compact_params {
    float compact_ratio;   // fraction of tokens to keep (0.0-1.0), default 0.5
    int   n_keep;          // first n positions to always keep (sink tokens), default 0
    int   n_ref_queries;   // reference queries for importance scoring (0 = auto: last quarter)
};

// Return sensible defaults
static inline kv_compact_params kv_compact_params_default(void) {
    kv_compact_params p;
    p.compact_ratio  = 0.5f;
    p.n_keep         = 0;
    p.n_ref_queries  = 0;
    return p;
}

// Compact a sequence's KV cache in-place using attention matching.
//
// Steps:
//   1. Save KV state via llama_state_seq_get_data()
//   2. Parse → compute importance → select top-t keys
//   3. Per-layer, per-head NNLS bias fitting + least-squares value refitting
//   4. Build compacted state → clear → reload via llama_state_seq_set_data()
//
// Returns:
//   > 0: new number of tokens in cache after compaction
//   -1:  error (state too small, parse failure, etc.)
//
// The caller is responsible for updating any token bookkeeping (e.g., prompt
// token arrays in the server) to reflect the compacted positions.
int kv_compact_sequence(
    llama_context * ctx,
    llama_seq_id    seq_id,
    kv_compact_params params);

#ifdef __cplusplus
}
#endif
