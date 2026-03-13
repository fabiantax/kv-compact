// KV Cache Compaction — C Library API implementation
//
// Implements the compaction pipeline from:
//   "Fast KV Compaction via Attention Matching" (Zweiger et al., 2026)
//   https://arxiv.org/abs/2602.16284
//
// Wraps the header-only math library in a C-compatible interface with
// memory management and quality metrics computation.
//
// Pipeline per call to kv_compact():
//   1. Key selection (Section 3.1) — score by max attention, select top-t
//      Optional: diversity penalty, shared prefix preservation
//   2. Beta solve (Section 3.2, Eq. 4) — NNLS mass matching per head
//   3. Value refit (Section 3.3, Eq. 6) — least-squares C_v per head
//   4. Optional: iterative refinement — swap worst keys, re-run steps 2-3
//   5. Quality metrics — cosine similarity, MSE, agreement rate

#include "kv-compact-api.h"
#include "kv-compact-math.h"

// Pull in GPU acceleration (or CPU stubs when KV_COMPACT_HIP is not defined)
#define KV_COMPACT_ACCEL_IMPL
#include "kv-compact-accel.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

using clock_type = std::chrono::high_resolution_clock;

static double elapsed_ms(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

// ============================================================================
// Cheap Q_ref generation from K vectors
// ============================================================================

// Generate proxy reference queries from K vectors (our extension).
//
// The paper (Section 2.2) generates Q_ref via "repeat-prefill" — feeding
// the context a second time and capturing query activations. This is
// expensive (~40% of compaction time). As a cheaper alternative, we sample
// K vectors as proxy queries, exploiting the observation that K and Q live
// in similar subspaces. This sacrifices some quality for ~10x speedup.
// Samples are quadratically biased toward recent tokens (more likely to be
// queried during generation).
static void generate_cheap_qref(const float * K_all, int T, int n_head_kv,
                                 int d_k, int n_q_out,
                                 std::vector<float> & Q_out) {
    int n_embd_k = n_head_kv * d_k;
    Q_out.resize((size_t)n_q_out * n_embd_k);

    // Sample evenly-spaced positions, biased toward recent tokens
    // (more recent = more likely to be queried in practice)
    for (int qi = 0; qi < n_q_out; qi++) {
        // Quadratic spacing: more samples from the end
        float frac = (float)(qi + 1) / (float)(n_q_out + 1);
        frac = frac * frac;  // bias toward end
        int pos = (int)(frac * (T - 1));
        if (pos >= T) pos = T - 1;
        if (pos < 0) pos = 0;

        memcpy(Q_out.data() + qi * n_embd_k,
               K_all + pos * n_embd_k,
               n_embd_k * sizeof(float));
    }
}

// ============================================================================
// Diversity-aware key selection
// ============================================================================

// Select top-t keys with diversity penalty (our extension beyond the paper).
//
// The paper's "Highest Attention Keys" (Section 3.1) selects purely by
// importance score. At extreme compression (>20x), this can select
// redundant keys that cover the same attention mass. Diversity-aware
// selection applies a greedy cosine-similarity penalty: after selecting
// each key, penalize remaining keys that are similar to those already
// chosen. This reduces wasted budget slots and improves quality at
// high compression ratios.
//
// importance: [T] global importance scores
// K_all:      [T × n_embd_k] key vectors (for similarity computation)
// T:          number of positions
// t:          target count
// n_embd_k:   total K embedding size
// strength:   diversity penalty strength (0=none, 1=full)
// n_prefix:   number of prefix positions to always include first
static std::vector<int> select_keys_diverse(
        const float * importance, const float * K_all,
        int T, int t, int n_embd_k, float strength, int n_prefix) {

    std::vector<int> selected;
    selected.reserve(t);
    std::vector<bool> used(T, false);
    std::vector<float> penalty(T, 0.0f);

    // Force-include shared prefix positions first
    int prefix_count = n_prefix < t ? n_prefix : t;
    for (int j = 0; j < prefix_count; j++) {
        selected.push_back(j);
        used[j] = true;
    }

    // Greedy selection with diversity penalty
    while ((int)selected.size() < t) {
        int best = -1;
        float best_score = -1e30f;

        for (int j = 0; j < T; j++) {
            if (used[j]) continue;
            float score = importance[j] * (1.0f - strength * penalty[j]);
            if (score > best_score) {
                best_score = score;
                best = j;
            }
        }

        if (best < 0) break;

        selected.push_back(best);
        used[best] = true;

        // Update penalty: compute cosine similarity of newly selected key
        // with all remaining candidates
        const float * k_sel = K_all + best * n_embd_k;
        float norm_sel = 0.0f;
        for (int d = 0; d < n_embd_k; d++) norm_sel += k_sel[d] * k_sel[d];
        norm_sel = sqrtf(norm_sel) + 1e-8f;

        for (int j = 0; j < T; j++) {
            if (used[j]) continue;
            const float * k_j = K_all + j * n_embd_k;
            float dot = 0.0f, norm_j = 0.0f;
            for (int d = 0; d < n_embd_k; d++) {
                dot += k_sel[d] * k_j[d];
                norm_j += k_j[d] * k_j[d];
            }
            float cos_sim = dot / (norm_sel * sqrtf(norm_j) + 1e-8f);
            // Track max similarity to any selected key
            if (cos_sim > penalty[j]) penalty[j] = cos_sim;
        }
    }

    std::sort(selected.begin(), selected.end());
    return selected;
}

// ============================================================================
// Quality metrics computation
// ============================================================================

static void compute_quality_metrics(
        const float * K_all, const float * V_all, const float * Q_ref_all,
        int T, int n_q, int n_head_kv, int d_k, int d_v,
        const kv_compact_result * result,
        kv_compact_stats * stats) {

    float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);
    int t = result->t;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    float sum_cos = 0.0f, sum_mse = 0.0f;
    int agree_count = 0, total_count = 0;

    // Sample up to 4 heads and 32 queries for efficiency
    int max_heads = n_head_kv < 4 ? n_head_kv : 4;
    int max_q = n_q < 32 ? n_q : 32;

    for (int h = 0; h < max_heads; h++) {
        for (int qi = 0; qi < max_q; qi++) {
            const float * q = Q_ref_all + qi * n_embd_k + h * d_k;

            // Original output
            std::vector<float> orig_scores(T);
            for (int j = 0; j < T; j++) {
                float dot = 0.0f;
                const float * k = K_all + j * n_embd_k + h * d_k;
                for (int d = 0; d < d_k; d++) dot += q[d] * k[d];
                orig_scores[j] = dot * inv_sqrt_dk;
            }
            softmax_rows(orig_scores.data(), 1, T);

            std::vector<float> orig_out(d_v, 0.0f);
            for (int j = 0; j < T; j++) {
                const float * v = V_all + j * n_embd_v + h * d_v;
                for (int d = 0; d < d_v; d++)
                    orig_out[d] += orig_scores[j] * v[d];
            }

            // Compacted output
            std::vector<float> comp_scores(t);
            for (int j = 0; j < t; j++) {
                float dot = 0.0f;
                const float * k = K_all + result->selected_indices[j] * n_embd_k + h * d_k;
                for (int d = 0; d < d_k; d++) dot += q[d] * k[d];
                comp_scores[j] = dot * inv_sqrt_dk + result->beta[h][j];
            }
            softmax_rows(comp_scores.data(), 1, t);

            std::vector<float> comp_out(d_v, 0.0f);
            for (int j = 0; j < t; j++) {
                for (int d = 0; d < d_v; d++)
                    comp_out[d] += comp_scores[j] * result->C_v[h][j * d_v + d];
            }

            // Cosine similarity
            float dot_p = 0.0f, no = 0.0f, nc = 0.0f;
            for (int d = 0; d < d_v; d++) {
                dot_p += orig_out[d] * comp_out[d];
                no += orig_out[d] * orig_out[d];
                nc += comp_out[d] * comp_out[d];
            }
            sum_cos += dot_p / (sqrtf(no * nc) + 1e-8f);

            // MSE
            float mse = 0.0f;
            for (int d = 0; d < d_v; d++) {
                float diff = orig_out[d] - comp_out[d];
                mse += diff * diff;
            }
            sum_mse += mse / d_v;

            // Agreement (argmax)
            int am_orig = 0, am_comp = 0;
            for (int d = 1; d < d_v; d++) {
                if (orig_out[d] > orig_out[am_orig]) am_orig = d;
                if (comp_out[d] > comp_out[am_comp]) am_comp = d;
            }
            if (am_orig == am_comp) agree_count++;
            total_count++;
        }
    }

    stats->avg_cosine_sim = sum_cos / total_count;
    stats->avg_mse = sum_mse / total_count;
    stats->avg_agreement = (float) agree_count / total_count;
}

// ============================================================================
// Public API
// ============================================================================

extern "C" {

kv_compact_params kv_compact_params_default(void) {
    kv_compact_params p;
    p.target_ratio = 0.5f;
    p.target_count = 0;
    p.use_sensitivity = 0;
    p.ridge = 1e-6f;
    p.nnls_max_iter = 200;
    p.refine_rounds = 0;
    p.use_diversity = 0;
    p.diversity_strength = 0.5f;
    p.n_shared_prefix = 0;
    p.use_cheap_qref = 0;
    p.skip_beta = 1;
    p.layer_filter = NULL;
    p.layer_filter_data = NULL;
    return p;
}

int kv_compact(
    const float * K_all,
    const float * V_all,
    const float * Q_ref_all,
    int T, int n_q, int n_head_kv, int d_k, int d_v,
    const kv_compact_params * params,
    kv_compact_result * result) {

    // Validate inputs
    if (!K_all || !V_all || !result) return -1;
    if (T <= 0 || n_head_kv <= 0 || d_k <= 0 || d_v <= 0) return -2;

    // Apply defaults if no params
    kv_compact_params p = params ? *params : kv_compact_params_default();

    int n_embd_k = n_head_kv * d_k;

    // Handle cheap Q_ref: generate proxy queries from K if Q_ref_all is NULL
    std::vector<float> cheap_qref;
    const float * Q_ref_used = Q_ref_all;
    int n_q_used = n_q;

    if (p.use_cheap_qref || !Q_ref_all) {
        // Auto-size: use min(T/2, 64) reference queries
        n_q_used = T / 2;
        if (n_q_used > 64) n_q_used = 64;
        if (n_q_used < 4) n_q_used = 4;
        if (n_q_used > T) n_q_used = T;

        generate_cheap_qref(K_all, T, n_head_kv, d_k, n_q_used, cheap_qref);
        Q_ref_used = cheap_qref.data();
    } else {
        if (n_q <= 0) return -2;
    }

    // Determine target size
    int t;
    if (p.target_count > 0) {
        t = p.target_count < T ? p.target_count : T;
    } else {
        t = (int)(T * p.target_ratio);
        if (t < 1) t = 1;
        if (t > T) t = T;
    }

    // Ensure shared prefix fits within target
    if (p.n_shared_prefix > t) {
        // If more prefix tokens than target, keep all prefix
        t = p.n_shared_prefix < T ? p.n_shared_prefix : T;
    }

    auto t_total = clock_type::now();

    // ---- Step 1: Attention scoring and key selection ----
    auto t_scoring = clock_type::now();

    int n_embd_v = n_head_kv * d_v;
    float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);

    // Per-head precomputed data
    struct head_cache {
        std::vector<float> scores;
        std::vector<float> exp_scores;
        std::vector<float> row_sums;
        std::vector<float> attn_weights;
    };
    std::vector<std::vector<head_cache>> hcache(1);  // single "layer"
    hcache[0].resize(n_head_kv);

    // Global importance for key selection
    std::vector<float> global_importance(T, 0.0f);

    // Optional sensitivity-weighted selection
    std::vector<std::vector<float>> per_head_imp;
    std::vector<float> sensitivities;

    // Allocate per-head score buffers
    for (int h = 0; h < n_head_kv; h++) {
        auto & hc = hcache[0][h];
        hc.scores.resize(n_q_used * T);
        hc.attn_weights.resize(n_q_used * T);
        if (!p.skip_beta) {
            hc.exp_scores.resize(n_q_used * T);
            hc.row_sums.resize(n_q_used);
        }
    }

    // Try GPU-accelerated scoring for all heads at once.
    // Falls through to CPU path if HIP is not compiled in or no device found.
    bool gpu_scored = false;
    if (kv_compact_hip_available()) {
        // Build array of per-head score buffer pointers
        std::vector<float *> score_ptrs(n_head_kv);
        for (int h = 0; h < n_head_kv; h++) {
            score_ptrs[h] = hcache[0][h].scores.data();
        }
        int rc = kv_compact_hip_score_all_heads(
            Q_ref_used, K_all, score_ptrs.data(),
            n_q_used, T, n_head_kv, d_k, inv_sqrt_dk);
        gpu_scored = (rc == 0);
    }

    // Temporary contiguous buffers for per-head extraction (reused across heads)
    std::vector<float> q_buf(n_q_used * d_k);
    std::vector<float> k_buf((size_t)T * d_k);
    std::vector<float> v_buf((size_t)T * d_v);

    if (!gpu_scored) {
        // CPU fallback: extract per-head contiguous data + batched matmul
        for (int h = 0; h < n_head_kv; h++) {
            auto & hc = hcache[0][h];
            const int head_offset = h * d_k;
            extract_head_contiguous(Q_ref_used, q_buf.data(), n_q_used, d_k, n_embd_k, head_offset);
            extract_head_contiguous(K_all, k_buf.data(), T, d_k, n_embd_k, head_offset);
            mat_mul_ABt(q_buf.data(), k_buf.data(), hc.scores.data(), n_q_used, T, d_k);
            for (int i = 0; i < n_q_used * T; i++) hc.scores[i] *= inv_sqrt_dk;
        }
    }

    // Fused post-processing: softmax + per-key importance
    // When skip_beta, use lighter version (no exp_scores/row_sums needed)
    for (int h = 0; h < n_head_kv; h++) {
        auto & hc = hcache[0][h];
        std::vector<float> head_imp(T);

        if (p.skip_beta) {
            softmax_importance_fused(hc.scores.data(), hc.attn_weights.data(),
                                     head_imp.data(), n_q_used, T);
        } else {
            exp_softmax_importance_fused(hc.scores.data(), hc.exp_scores.data(),
                                         hc.row_sums.data(), hc.attn_weights.data(),
                                         head_imp.data(), n_q_used, T);
        }

        if (p.use_sensitivity) {
            float sens = compute_head_sensitivity(hc.attn_weights.data(), n_q_used, T);
            per_head_imp.push_back(head_imp);
            sensitivities.push_back(sens);
        } else {
            for (int j = 0; j < T; j++) {
                if (head_imp[j] > global_importance[j])
                    global_importance[j] = head_imp[j];
            }
        }
    }

    if (p.use_sensitivity) {
        accumulate_weighted_importance(per_head_imp, sensitivities,
                                       T, n_head_kv, global_importance.data());
    }

    // Key selection: diversity-aware or standard
    std::vector<int> selected;
    if (p.use_diversity) {
        selected = select_keys_diverse(global_importance.data(), K_all,
                                       T, t, n_embd_k, p.diversity_strength,
                                       p.n_shared_prefix);
    } else {
        // Standard top-t selection, with shared prefix forced in
        if (p.n_shared_prefix > 0) {
            std::vector<bool> is_prefix(T, false);
            int prefix_count = p.n_shared_prefix < t ? p.n_shared_prefix : t;
            for (int j = 0; j < prefix_count; j++) is_prefix[j] = true;

            // Select remaining from non-prefix by importance
            std::vector<int> candidates;
            for (int j = prefix_count; j < T; j++) candidates.push_back(j);
            int remaining = t - prefix_count;
            std::partial_sort(candidates.begin(), candidates.begin() + remaining,
                              candidates.end(),
                              [&](int a, int b) { return global_importance[a] > global_importance[b]; });

            selected.reserve(t);
            for (int j = 0; j < prefix_count; j++) selected.push_back(j);
            for (int j = 0; j < remaining; j++) selected.push_back(candidates[j]);
            std::sort(selected.begin(), selected.end());
        } else {
            std::vector<int> indices(T);
            std::iota(indices.begin(), indices.end(), 0);
            std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                              [&](int a, int b) { return global_importance[a] > global_importance[b]; });

            selected.assign(indices.begin(), indices.begin() + t);
            std::sort(selected.begin(), selected.end());
        }
    }

    double scoring_time = elapsed_ms(t_scoring);

    // ---- Step 2-3: Per-head NNLS + least-squares ----
    auto t_nnls = clock_type::now();

    std::vector<std::vector<float>> beta_vecs(n_head_kv);
    std::vector<std::vector<float>> cv_vecs(n_head_kv);

    // Precompute Y_h = attn_weights_h @ V_h for all heads.
    // Y_h only depends on attn_weights (fixed from scoring) and V_all,
    // so it can be computed once and reused in the per-head LS solve.
    std::vector<std::vector<float>> Y_vecs(n_head_kv);
    for (int h = 0; h < n_head_kv; h++) {
        Y_vecs[h].resize(n_q_used * d_v, 0.0f);
    }

    bool gpu_refit = false;
    if (kv_compact_hip_available()) {
        std::vector<const float *> w_ptrs(n_head_kv);
        std::vector<float *> y_ptrs(n_head_kv);
        for (int h = 0; h < n_head_kv; h++) {
            w_ptrs[h] = hcache[0][h].attn_weights.data();
            y_ptrs[h] = Y_vecs[h].data();
        }
        int rc = kv_compact_hip_refit_target_all_heads(
            w_ptrs.data(), V_all, y_ptrs.data(),
            n_q_used, T, n_head_kv, d_v);
        gpu_refit = (rc == 0);
    }

    if (!gpu_refit) {
        // CPU fallback: extract per-head V contiguous + batched matmul
        for (int h = 0; h < n_head_kv; h++) {
            extract_head_contiguous(V_all, v_buf.data(), T, d_v, n_embd_v, h * d_v);
            mat_mul_AB(hcache[0][h].attn_weights.data(), v_buf.data(),
                       Y_vecs[h].data(), n_q_used, T, d_v);
        }
    }

    for (int h = 0; h < n_head_kv; h++) {
        const auto & hc = hcache[0][h];
        beta_vecs[h].resize(t);
        cv_vecs[h].resize(t * d_v);

        if (p.skip_beta) {
            // Skip NNLS: beta = 0, go straight to LS value refit
            std::fill(beta_vecs[h].begin(), beta_vecs[h].end(), 0.0f);
        } else {
            // NNLS for beta (Section 3.2)
            std::vector<float> M(n_q_used * t);
            for (int qi = 0; qi < n_q_used; qi++) {
                for (int j = 0; j < t; j++) {
                    M[qi * t + j] = hc.exp_scores[qi * T + selected[j]];
                }
            }

            std::vector<float> w(t);
            nnls_solve(M.data(), hc.row_sums.data(), w.data(), n_q_used, t, p.nnls_max_iter);

            for (int j = 0; j < t; j++) {
                beta_vecs[h][j] = logf(std::max(1e-12f, w[j]));
            }
        }

        // Least squares for C_v
        std::vector<float> X(n_q_used * t);
        for (int qi = 0; qi < n_q_used; qi++) {
            for (int j = 0; j < t; j++) {
                X[qi * t + j] = hc.scores[qi * T + selected[j]] + beta_vecs[h][j];
            }
        }
        softmax_rows(X.data(), n_q_used, t);

        least_squares_solve(X.data(), Y_vecs[h].data(), cv_vecs[h].data(),
                           n_q_used, t, d_v, p.ridge);
    }

    // ---- Iterative refinement: swap worst selected keys with best unused ----
    if (p.refine_rounds > 0 && t < T) {
        // Build set of unused indices
        std::vector<bool> is_selected(T, false);
        for (int j = 0; j < t; j++) is_selected[selected[j]] = true;

        auto compute_key_error = [&](int h, int key_slot) -> float {
            const auto & hc = hcache[0][h];
            float total_err = 0.0f;
            for (int qi = 0; qi < n_q_used; qi++) {
                std::vector<float> comp_s(t);
                for (int j = 0; j < t; j++) {
                    comp_s[j] = hc.scores[qi * T + selected[j]] + beta_vecs[h][j];
                }
                softmax_rows(comp_s.data(), 1, t);
                float weight = comp_s[key_slot];
                total_err += weight * weight;
            }
            return total_err / n_q_used;
        };

        for (int refine = 0; refine < p.refine_rounds; refine++) {
            std::vector<float> key_value(t, 0.0f);
            for (int h = 0; h < n_head_kv; h++) {
                for (int j = 0; j < t; j++) {
                    key_value[j] += compute_key_error(h, j);
                }
            }

            int worst_slot = 0;
            for (int j = 1; j < t; j++) {
                if (key_value[j] < key_value[worst_slot]) worst_slot = j;
            }

            int best_unused = -1;
            float best_imp = -1.0f;
            for (int j = 0; j < T; j++) {
                if (!is_selected[j] && global_importance[j] > best_imp) {
                    best_imp = global_importance[j];
                    best_unused = j;
                }
            }

            if (best_unused < 0) break;
            if (best_imp <= global_importance[selected[worst_slot]] * 0.5f) break;

            is_selected[selected[worst_slot]] = false;
            is_selected[best_unused] = true;
            selected[worst_slot] = best_unused;
            std::sort(selected.begin(), selected.end());

            // Y_vecs[h] is unchanged (attn_weights and V_all are fixed) —
            // reuse the precomputed values from before the per-head loop.
            for (int h = 0; h < n_head_kv; h++) {
                const auto & hc = hcache[0][h];

                if (p.skip_beta) {
                    std::fill(beta_vecs[h].begin(), beta_vecs[h].end(), 0.0f);
                } else {
                    std::vector<float> M(n_q_used * t);
                    for (int qi = 0; qi < n_q_used; qi++) {
                        for (int j = 0; j < t; j++) {
                            M[qi * t + j] = hc.exp_scores[qi * T + selected[j]];
                        }
                    }

                    std::vector<float> w(t);
                    nnls_solve(M.data(), hc.row_sums.data(), w.data(), n_q_used, t, p.nnls_max_iter);

                    for (int j = 0; j < t; j++) {
                        beta_vecs[h][j] = logf(std::max(1e-12f, w[j]));
                    }
                }

                std::vector<float> X(n_q_used * t);
                for (int qi = 0; qi < n_q_used; qi++) {
                    for (int j = 0; j < t; j++) {
                        X[qi * t + j] = hc.scores[qi * T + selected[j]] + beta_vecs[h][j];
                    }
                }
                softmax_rows(X.data(), n_q_used, t);

                least_squares_solve(X.data(), Y_vecs[h].data(), cv_vecs[h].data(),
                                   n_q_used, t, d_v, p.ridge);
            }
        }
    }

    double nnls_time = elapsed_ms(t_nnls);

    // ---- Populate result ----
    result->t = t;
    result->n_head_kv = n_head_kv;

    result->selected_indices = (int *) malloc(t * sizeof(int));
    memcpy(result->selected_indices, selected.data(), t * sizeof(int));

    result->beta = (float **) malloc(n_head_kv * sizeof(float *));
    result->C_v  = (float **) malloc(n_head_kv * sizeof(float *));

    for (int h = 0; h < n_head_kv; h++) {
        result->beta[h] = (float *) malloc(t * sizeof(float));
        memcpy(result->beta[h], beta_vecs[h].data(), t * sizeof(float));

        result->C_v[h] = (float *) malloc(t * d_v * sizeof(float));
        memcpy(result->C_v[h], cv_vecs[h].data(), t * d_v * sizeof(float));
    }

    // Compute quality metrics (using the effective Q_ref)
    compute_quality_metrics(K_all, V_all, Q_ref_used,
                           T, n_q_used, n_head_kv, d_k, d_v, result,
                           &result->stats);

    result->stats.elapsed_ms = elapsed_ms(t_total);
    result->stats.scoring_ms = scoring_time;
    result->stats.nnls_ms = nnls_time;

    return 0;
}

int kv_compact_multi_round(
    const float * K_all,
    const float * V_all,
    const float * Q_ref_all,
    int T, int n_q, int n_head_kv, int d_k, int d_v,
    const kv_compact_params * params,
    int n_rounds,
    kv_compact_result * result,
    kv_compact_stats * per_round_stats) {

    if (!K_all || !V_all || !result) return -1;
    if (T <= 0 || n_head_kv <= 0 || d_k <= 0 || d_v <= 0) return -2;
    if (n_rounds < 1) return -3;

    // Check: Q_ref_all is required unless cheap_qref is enabled
    kv_compact_params p_check = params ? *params : kv_compact_params_default();
    if (!Q_ref_all && !p_check.use_cheap_qref) return -1;
    if (Q_ref_all && n_q <= 0 && !p_check.use_cheap_qref) return -2;

    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    // Track original indices through rounds
    std::vector<int> original_indices(T);
    std::iota(original_indices.begin(), original_indices.end(), 0);

    // Current KV data (starts as copy, modified each round)
    std::vector<float> cur_K(K_all, K_all + (size_t)T * n_embd_k);
    std::vector<float> cur_V(V_all, V_all + (size_t)T * n_embd_v);
    int cur_T = T;

    // Handle Q_ref: use provided or generate cheap proxy
    std::vector<float> cheap_qref_mr;
    const float * Q_ref_mr = Q_ref_all;
    int n_q_mr = n_q;

    if (p_check.use_cheap_qref || !Q_ref_all) {
        n_q_mr = T / 2;
        if (n_q_mr > 64) n_q_mr = 64;
        if (n_q_mr < 4) n_q_mr = 4;
        if (n_q_mr > T) n_q_mr = T;
        generate_cheap_qref(K_all, T, n_head_kv, d_k, n_q_mr, cheap_qref_mr);
        Q_ref_mr = cheap_qref_mr.data();
    }

    for (int round = 0; round < n_rounds; round++) {
        int n_q_round = n_q_mr < cur_T ? n_q_mr : cur_T;

        kv_compact_result round_result = {};
        int rc = kv_compact(cur_K.data(), cur_V.data(), Q_ref_mr,
                           cur_T, n_q_round, n_head_kv, d_k, d_v,
                           params, &round_result);
        if (rc != 0) {
            kv_compact_result_free(&round_result);
            return rc;
        }

        if (per_round_stats) {
            per_round_stats[round] = round_result.stats;
        }

        int new_T = round_result.t;

        // Map selected indices back to original positions
        std::vector<int> new_original_indices(new_T);
        for (int j = 0; j < new_T; j++) {
            new_original_indices[j] = original_indices[round_result.selected_indices[j]];
        }

        // Build new K (original, without beta) and new V (C_v) for next round.
        // We do NOT fold beta into K between rounds — the compaction math
        // already accounts for beta during NNLS/LS. Folding it in would
        // distort the K vectors for subsequent rounds.
        std::vector<float> new_K(new_T * n_embd_k);
        std::vector<float> new_V(new_T * n_embd_v);

        for (int j = 0; j < new_T; j++) {
            int src_idx = round_result.selected_indices[j];
            // Keep original K (no beta modification)
            memcpy(new_K.data() + j * n_embd_k,
                   cur_K.data() + src_idx * n_embd_k,
                   n_embd_k * sizeof(float));
            // V from C_v (per head)
            for (int h = 0; h < n_head_kv; h++) {
                memcpy(new_V.data() + j * n_embd_v + h * d_v,
                       round_result.C_v[h] + j * d_v,
                       d_v * sizeof(float));
            }
        }

        // If this is the last round, populate the final result
        if (round == n_rounds - 1) {
            result->t = new_T;
            result->n_head_kv = n_head_kv;

            result->selected_indices = (int *) malloc(new_T * sizeof(int));
            memcpy(result->selected_indices, new_original_indices.data(),
                   new_T * sizeof(int));

            result->beta = (float **) malloc(n_head_kv * sizeof(float *));
            result->C_v  = (float **) malloc(n_head_kv * sizeof(float *));

            for (int h = 0; h < n_head_kv; h++) {
                // Store the last round's beta (to be applied to original K)
                result->beta[h] = (float *) malloc(new_T * sizeof(float));
                memcpy(result->beta[h], round_result.beta[h],
                       new_T * sizeof(float));

                result->C_v[h] = (float *) malloc(new_T * d_v * sizeof(float));
                memcpy(result->C_v[h], round_result.C_v[h],
                       new_T * d_v * sizeof(float));
            }

            result->stats = round_result.stats;
        }

        kv_compact_result_free(&round_result);

        // Update for next round
        cur_K = std::move(new_K);
        cur_V = std::move(new_V);
        original_indices = std::move(new_original_indices);
        cur_T = new_T;
    }

    // Compute final quality metrics against the ORIGINAL data
    compute_quality_metrics(K_all, V_all, Q_ref_mr,
                           T, n_q_mr, n_head_kv, d_k, d_v, result,
                           &result->stats);

    return 0;
}

void kv_compact_result_free(kv_compact_result * result) {
    if (!result) return;

    free(result->selected_indices);
    result->selected_indices = NULL;

    if (result->beta) {
        for (int h = 0; h < result->n_head_kv; h++) {
            free(result->beta[h]);
        }
        free(result->beta);
        result->beta = NULL;
    }

    if (result->C_v) {
        for (int h = 0; h < result->n_head_kv; h++) {
            free(result->C_v[h]);
        }
        free(result->C_v);
        result->C_v = NULL;
    }

    result->t = 0;
    result->n_head_kv = 0;
}

} // extern "C"
