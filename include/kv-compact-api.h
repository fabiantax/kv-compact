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

// ---- Bandwidth-aware ratio suggestion ----
//
// On bandwidth-limited hardware (APUs, mobile GPUs), KV cache reads during
// decode dominate memory bandwidth.  Compacting the cache allows either:
//   (a) more parallel agents at the same tg/s, or
//   (b) higher tg/s per agent with the same memory.
//
// This function suggests a compact_ratio given hardware constraints.
//
// Parameters:
//   model            - loaded model (used to read layer/head/embd dimensions)
//   ctx_size         - context size per sequence (n_ctx)
//   mem_budget_mb    - total memory budget for KV caches (in MB)
//   n_parallel       - number of parallel sequences (agents/slots)
//
// Returns:
//   Suggested compact_ratio in (0, 1].  Returns 1.0 if no compaction needed.
//   Returns -1.0 on error (null model).
//
// Example: Strix Halo with 8 GB KV budget, 8 parallel agents, 4096 ctx:
//   float ratio = kv_compact_suggest_ratio(model, 4096, 8192, 8);
//   // → might return 0.25 (keep 25%, 4x compression)
float kv_compact_suggest_ratio(
    const llama_model * model,
    int ctx_size,
    float mem_budget_mb,
    int n_parallel);

// ---- Compaction parameters ----

struct kv_compact_params {
    float compact_ratio;   // fraction of tokens to keep (0.0-1.0), default 0.5
    int   n_keep;          // first n positions to always keep (sink tokens), default 0
    int   n_ref_queries;   // reference queries for importance scoring (0 = auto: last quarter)

    // Output: which original positions were kept (caller-provided buffer).
    // If non-NULL, the API fills kept_positions[0..new_size-1] with the
    // original cell positions that were selected (sorted, before renumbering).
    // The buffer must have room for at least (int)(n_ctx * compact_ratio) entries.
    // Set to NULL if you don't need this information.
    int32_t * kept_positions;
    int       kept_positions_cap;  // capacity of kept_positions buffer
};

// Return sensible defaults
static inline kv_compact_params kv_compact_params_default(void) {
    kv_compact_params p;
    p.compact_ratio      = 0.5f;
    p.n_keep             = 0;
    p.n_ref_queries      = 0;
    p.kept_positions     = NULL;
    p.kept_positions_cap = 0;
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
// If params.kept_positions is non-NULL, fills it with the original positions
// that were kept (useful for server token bookkeeping).
int kv_compact_sequence(
    llama_context * ctx,
    llama_seq_id    seq_id,
    kv_compact_params params);

#ifdef __cplusplus
}
#endif
