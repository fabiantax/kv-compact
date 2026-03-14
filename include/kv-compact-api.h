// KV Cache Compaction — C Library API
//
// Programmatic interface for KV cache compaction via Attention Matching.
// Implements the algorithm from:
//
//   "Fast KV Compaction via Attention Matching" (Zweiger et al., 2026)
//   https://arxiv.org/abs/2602.16284
//
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
// Layer filter — determines which layers have compactable KV caches
// ============================================================================

// Callback to decide whether a layer should be compacted.
// Returns non-zero (true) if the layer should be compacted, 0 to skip.
//
// For standard transformers: return true for all layers.
// For hybrid architectures (e.g., Qwen 3.5 with 30 DeltaNet + 10 attention):
//   return true only for softmax attention layers that have KV caches.
//
// layer_idx: 0-based layer index
// n_layers:  total number of layers in the model
// user_data: opaque pointer passed through from params
typedef int (*kv_layer_filter_fn)(int layer_idx, int n_layers, void * user_data);

// Built-in filter: compact all layers (default behavior)
static inline int kv_layer_filter_all(int layer_idx, int n_layers, void * user_data) {
    (void)layer_idx; (void)n_layers; (void)user_data;
    return 1;
}

// Built-in filter: compact every Nth layer (for hybrid models with periodic attention)
// Pass the attention interval as user_data: (void *)(intptr_t)interval
// Example: Qwen 3.5 has full_attention_interval=4, so attention layers are 3,7,11,...
//   filter_fn = kv_layer_filter_periodic
//   filter_user_data = (void *)(intptr_t)4
static inline int kv_layer_filter_periodic(int layer_idx, int n_layers, void * user_data) {
    (void)n_layers;
    int interval = (int)(intptr_t)user_data;
    if (interval <= 0) return 1;
    return ((layer_idx + 1) % interval == 0) ? 1 : 0;
}

// Built-in filter: compact only layers listed in an explicit array
// user_data must point to a kv_layer_list struct
typedef struct kv_layer_list {
    const int * layers;   // sorted array of layer indices to compact
    int         n_layers; // number of entries in the array
} kv_layer_list;

static inline int kv_layer_filter_explicit(int layer_idx, int n_layers, void * user_data) {
    (void)n_layers;
    const kv_layer_list * list = (const kv_layer_list *)user_data;
    if (!list || !list->layers) return 1;
    for (int i = 0; i < list->n_layers; i++) {
        if (list->layers[i] == layer_idx) return 1;
        if (list->layers[i] > layer_idx) break;  // sorted, early exit
    }
    return 0;
}

// ============================================================================
// Reasoning token classification (R-KV — Feature A4)
// ============================================================================

// Token types for reasoning-aware compression.
// Reasoning models (Qwen 3.5, DeepSeek-R1) produce long chains of thinking
// tokens that are highly redundant in the KV cache.  R-KV classifies each
// cached position and applies differential importance weighting so thinking
// tokens compress more aggressively while answer tokens are preserved.
typedef enum {
    KV_TOKEN_ANSWER     = 0,   // regular / answer token (default, preserved)
    KV_TOKEN_THINKING   = 1,   // reasoning / thinking token (compress aggressively)
    KV_TOKEN_TRANSITION = 2    // boundary between thinking and answer
} kv_token_type;

// Callback to classify a cached token position.
// Called once per position during key selection.
//
//   token_pos: 0-based position in the KV cache [0, T)
//   user_data: opaque pointer passed through from kv_compact_reasoning
//
// Returns a kv_token_type value.  Positions not classified default to ANSWER.
typedef int (*kv_token_classifier_fn)(int token_pos, void * user_data);

// Reasoning-aware compression configuration.
// Attach to kv_compact_params.reasoning to enable.
typedef struct kv_compact_reasoning {
    kv_token_classifier_fn  classifier;     // NULL = disabled
    void *                  classifier_data;

    // Importance multipliers applied during key selection.
    // Lower values → more aggressive compression of that token class.
    float  thinking_weight;       // default 0.3  — thinking tokens 3.3x less important
    float  transition_weight;     // default 0.7  — boundary tokens moderately important
    // Answer tokens are always weight 1.0 (the reference scale).

    int    token_offset;          // internal: added to pos before calling classifier
                                  // (set automatically by chunked compaction)
} kv_compact_reasoning;

// ============================================================================
// Parameters
// ============================================================================

typedef struct kv_compact_params {
    float  target_ratio;       // fraction of tokens to keep (0.0, 1.0) — compression ratio
    int    target_count;       // explicit target count (overrides ratio if > 0)
    int    use_sensitivity;    // weight key selection by per-head sensitivity (Section 4)
    float  ridge;              // ridge regularization for value LS solve (Section 3.3)
    int    nnls_max_iter;      // max NNLS iterations for beta solve (Section 3.2)
    int    refine_rounds;      // iterative refinement rounds (0=disabled, default: 0)
                               // each round evaluates per-key reconstruction error,
                               // swaps worst selected keys with best unused, then
                               // re-runs NNLS + LS. Typically 2-3 rounds.
    int    use_diversity;      // diversity-aware key selection (0=disabled, default: 0)
                               // down-weights keys similar to already-selected ones,
                               // reducing redundancy in the compacted cache.
    float  diversity_strength; // how strongly to penalize similar keys (default: 0.5)
                               // 0.0 = no effect, 1.0 = full suppression of duplicates
    int    n_shared_prefix;    // number of shared prefix tokens to always keep (default: 0)
                               // these positions are never evicted, supporting shared
                               // prompt KV caches across multiple agents.
    int    use_cheap_qref;     // use K-vector proxy for Q_ref (0=disabled, default: 0)
                               // when enabled, Q_ref_all can be NULL and reference
                               // queries are generated from K vectors automatically.
    int    skip_beta;          // skip NNLS beta solve, set beta=0 (default: 1)
                               // the LS value refit alone achieves equal or better quality
                               // while eliminating the O(n_q*t^2) NNLS bottleneck.
                               // at 2048 tokens this is 6.75x faster with 0.9999 vs 0.9994 cos sim.

    int    score_method;       // key importance aggregation across queries (default: 1 = RMS)
                               //   0 = max: max_i softmax[i,j] — fast, original default
                               //   1 = rms: sqrt(mean_i(softmax[i,j]^2)) — paper default (Section 3.1)
                               //   2 = mean: mean_i softmax[i,j]
                               // The paper (Section 3.1) found RMS "more robust than mean or max".

    int    use_omp;            // use OMP key selection (0=disabled, default: 0)
                               // when enabled, uses Orthogonal Matching Pursuit (Algorithm 1)
                               // to greedily select keys that best approximate the partition
                               // function. This is the paper's best method (AM-OMP) but slower
                               // than highest-attention selection.
    int    omp_k_choice;       // OMP: keys to select per iteration (default: 1)
                               //   1 = standard OMP (best quality, slowest)
                               //   4 = fast OMP (paper's AM-OMP-fast, 4-8x speedup)
    int    omp_refit_interval; // OMP: NNLS refit interval (default: 1)
                               //   1 = refit every iteration (standard)
                               //   2 = refit every other iteration (paper's fast variant)

    int    nnls_method;        // NNLS solver method (default: 1 = PGD)
                               //   0 = Lawson-Hanson active-set (exact, finite convergence)
                               //   1 = projected gradient descent (paper's Algorithm 3)
    int    nnls_pgd_iters;     // PGD iterations for NNLS refinement (default: 0)
                               //   0 = clamped LS only (paper's default, fast)
                               //  >0 = additional PGD refinement steps

    int    chunk_size;         // chunk size for chunked compaction (default: 0 = auto)
                               //   0 = auto: target t_chunk ≤ 256 (chunk = ceil(256/ratio))
                               //             e.g., ratio=0.5 → chunk=512, ratio=0.2 → chunk=1280
                               //             no chunking when t ≤ 256 (T*ratio ≤ 256)
                               //  -1 = disabled: never chunk (may OOM at large T)
                               //  >0 = explicit chunk size (tokens per chunk)
                               // Chunked compaction splits the T tokens into segments,
                               // compacts each independently, then merges results.
                               // This bounds LS memory to O(chunk_t^2) per chunk instead
                               // of O(t^2) total, enabling 1M+ token contexts.
                               // Quality impact is minimal since attention is mostly local.

    int    n_threads;          // number of threads for parallel chunk processing (default: 0)
                               //   0 = auto (use all available cores via OpenMP)
                               //   1 = single-threaded (no OpenMP overhead)
                               //  >1 = explicit thread count
                               // Only affects chunked compaction. Each chunk is processed
                               // independently in parallel. Speedup is near-linear with cores.

    // Layer filter for hybrid architectures (e.g., Qwen 3.5 DeltaNet + attention)
    // When non-NULL, only layers where filter returns non-zero are compacted.
    // Layers that are filtered out are passed through unchanged.
    // Default: NULL (compact all layers — equivalent to kv_layer_filter_all)
    kv_layer_filter_fn  layer_filter;
    void *              layer_filter_data;

    // Reasoning-aware compression (R-KV — Feature A4)
    // When non-NULL, applies differential importance weighting by token type.
    // Thinking tokens are compressed more aggressively; answer tokens are preserved.
    // Default: NULL (all tokens treated equally)
    kv_compact_reasoning * reasoning;
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

// Apply multiple rounds of compaction to progressively shrink the cache.
//
// Each round compacts the output of the previous round. After round i,
// the K data has beta folded in and V data is replaced by C_v, which
// becomes the input for round i+1.
//
// This is useful for very long conversations where the cache grows
// incrementally and needs periodic compaction.
//
//   n_rounds:       number of compaction rounds to apply
//   per_round_stats: if non-NULL, array of n_rounds stats structs filled in
//
// The final result contains the cumulative compaction. selected_indices
// refer to the ORIGINAL positions (before any compaction).
//
// Returns 0 on success, non-zero on error.
int kv_compact_multi_round(
    const float * K_all,
    const float * V_all,
    const float * Q_ref_all,
    int T, int n_q, int n_head_kv, int d_k, int d_v,
    const kv_compact_params * params,
    int n_rounds,
    kv_compact_result * result,
    kv_compact_stats * per_round_stats);

// Free all memory allocated by kv_compact()
void kv_compact_result_free(kv_compact_result * result);

// Count how many layers pass the filter (utility for callers doing multi-layer compaction)
static inline int kv_compact_count_layers(
        kv_layer_filter_fn filter, void * filter_data, int n_layers) {
    if (!filter) return n_layers;
    int count = 0;
    for (int l = 0; l < n_layers; l++) {
        if (filter(l, n_layers, filter_data)) count++;
    }
    return count;
}

// Check if a specific layer should be compacted
static inline int kv_compact_should_compact_layer(
        const kv_compact_params * params, int layer_idx, int n_layers) {
    if (!params || !params->layer_filter) return 1;
    return params->layer_filter(layer_idx, n_layers, params->layer_filter_data);
}

#ifdef __cplusplus
}
#endif
