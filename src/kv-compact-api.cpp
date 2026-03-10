// KV Cache Compaction — C Library API implementation
//
// Wraps the header-only math library in a C-compatible interface with
// memory management and quality metrics computation.

#include "kv-compact-api.h"
#include "kv-compact-math.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>

using clock_type = std::chrono::high_resolution_clock;

static double elapsed_ms(clock_type::time_point t0) {
    return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
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
    if (!K_all || !V_all || !Q_ref_all || !result) return -1;
    if (T <= 0 || n_q <= 0 || n_head_kv <= 0 || d_k <= 0 || d_v <= 0) return -2;

    // Apply defaults if no params
    kv_compact_params p = params ? *params : kv_compact_params_default();

    // Determine target size
    int t;
    if (p.target_count > 0) {
        t = p.target_count < T ? p.target_count : T;
    } else {
        t = (int)(T * p.target_ratio);
        if (t < 1) t = 1;
        if (t > T) t = T;
    }

    auto t_total = clock_type::now();

    // ---- Step 1: Attention scoring and key selection ----
    auto t_scoring = clock_type::now();

    int n_embd_k = n_head_kv * d_k;
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

    for (int h = 0; h < n_head_kv; h++) {
        auto & hc = hcache[0][h];
        hc.scores.resize(n_q * T);
        hc.exp_scores.resize(n_q * T);
        hc.row_sums.resize(n_q);
        hc.attn_weights.resize(n_q * T);

        // Compute scores
        for (int qi = 0; qi < n_q; qi++) {
            const float * q_row = Q_ref_all + qi * n_embd_k + h * d_k;
            for (int ki = 0; ki < T; ki++) {
                const float * k_row = K_all + ki * n_embd_k + h * d_k;
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++) dot += q_row[d] * k_row[d];
                hc.scores[qi * T + ki] = dot * inv_sqrt_dk;
            }
        }

        memcpy(hc.exp_scores.data(), hc.scores.data(), n_q * T * sizeof(float));
        exp_rows_stable(hc.exp_scores.data(), hc.row_sums.data(), n_q, T);

        memcpy(hc.attn_weights.data(), hc.scores.data(), n_q * T * sizeof(float));
        softmax_rows(hc.attn_weights.data(), n_q, T);

        // Per-key importance
        std::vector<float> head_imp(T);
        for (int j = 0; j < T; j++) {
            float max_w = 0.0f;
            for (int qi = 0; qi < n_q; qi++) {
                float w = hc.attn_weights[qi * T + j];
                if (w > max_w) max_w = w;
            }
            head_imp[j] = max_w;
        }

        if (p.use_sensitivity) {
            float sens = compute_head_sensitivity(hc.attn_weights.data(), n_q, T);
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

    // Select top-t
    std::vector<int> indices(T);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                      [&](int a, int b) { return global_importance[a] > global_importance[b]; });

    std::vector<int> selected(indices.begin(), indices.begin() + t);
    std::sort(selected.begin(), selected.end());

    double scoring_time = elapsed_ms(t_scoring);

    // ---- Step 2-3: Per-head NNLS + least-squares ----
    auto t_nnls = clock_type::now();

    std::vector<std::vector<float>> beta_vecs(n_head_kv);
    std::vector<std::vector<float>> cv_vecs(n_head_kv);

    for (int h = 0; h < n_head_kv; h++) {
        const auto & hc = hcache[0][h];
        beta_vecs[h].resize(t);
        cv_vecs[h].resize(t * d_v);

        // NNLS for beta
        std::vector<float> M(n_q * t);
        for (int qi = 0; qi < n_q; qi++) {
            for (int j = 0; j < t; j++) {
                M[qi * t + j] = hc.exp_scores[qi * T + selected[j]];
            }
        }

        std::vector<float> w(t);
        nnls_solve(M.data(), hc.row_sums.data(), w.data(), n_q, t, p.nnls_max_iter);

        for (int j = 0; j < t; j++) {
            beta_vecs[h][j] = logf(std::max(1e-12f, w[j]));
        }

        // Least squares for C_v
        std::vector<float> X(n_q * t);
        for (int qi = 0; qi < n_q; qi++) {
            for (int j = 0; j < t; j++) {
                X[qi * t + j] = hc.scores[qi * T + selected[j]] + beta_vecs[h][j];
            }
        }
        softmax_rows(X.data(), n_q, t);

        std::vector<float> Y(n_q * d_v, 0.0f);
        for (int qi = 0; qi < n_q; qi++) {
            for (int ki = 0; ki < T; ki++) {
                float w_ij = hc.attn_weights[qi * T + ki];
                const float * v_row = V_all + ki * n_embd_v + h * d_v;
                for (int d = 0; d < d_v; d++)
                    Y[qi * d_v + d] += w_ij * v_row[d];
            }
        }

        least_squares_solve(X.data(), Y.data(), cv_vecs[h].data(),
                           n_q, t, d_v, p.ridge);
    }

    // ---- Iterative refinement: swap worst selected keys with best unused ----
    if (p.refine_rounds > 0 && t < T) {
        // Build set of unused indices
        std::vector<bool> is_selected(T, false);
        for (int j = 0; j < t; j++) is_selected[selected[j]] = true;

        // Helper lambda to compute per-key reconstruction error for head h
        // Returns the total MSE contribution of removing key j from the selection
        auto compute_key_error = [&](int h, int key_slot) -> float {
            const auto & hc = hcache[0][h];
            float total_err = 0.0f;

            // Compute compacted attention output for each query
            for (int qi = 0; qi < n_q; qi++) {
                // Compacted scores with beta
                std::vector<float> comp_s(t);
                for (int j = 0; j < t; j++) {
                    comp_s[j] = hc.scores[qi * T + selected[j]] + beta_vecs[h][j];
                }
                softmax_rows(comp_s.data(), 1, t);

                // How much does this key contribute to the output?
                float weight = comp_s[key_slot];
                total_err += weight * weight;  // squared attention weight = proxy for impact
            }
            return total_err / n_q;
        };

        for (int refine = 0; refine < p.refine_rounds; refine++) {
            // For each head, find the selected key with the LOWEST attention mass
            // (contributes least), averaged across heads
            std::vector<float> key_value(t, 0.0f);
            for (int h = 0; h < n_head_kv; h++) {
                for (int j = 0; j < t; j++) {
                    key_value[j] += compute_key_error(h, j);
                }
            }

            // Find worst selected key (lowest value)
            int worst_slot = 0;
            for (int j = 1; j < t; j++) {
                if (key_value[j] < key_value[worst_slot]) worst_slot = j;
            }

            // Find best unused key (highest global importance)
            int best_unused = -1;
            float best_imp = -1.0f;
            for (int j = 0; j < T; j++) {
                if (!is_selected[j] && global_importance[j] > best_imp) {
                    best_imp = global_importance[j];
                    best_unused = j;
                }
            }

            if (best_unused < 0) break;  // no unused keys to swap

            // Only swap if the unused key has higher importance than worst selected
            if (best_imp <= global_importance[selected[worst_slot]] * 0.5f) break;

            // Perform swap
            is_selected[selected[worst_slot]] = false;
            is_selected[best_unused] = true;
            selected[worst_slot] = best_unused;
            std::sort(selected.begin(), selected.end());

            // Re-run NNLS + LS for all heads with new selection
            for (int h = 0; h < n_head_kv; h++) {
                const auto & hc = hcache[0][h];

                std::vector<float> M(n_q * t);
                for (int qi = 0; qi < n_q; qi++) {
                    for (int j = 0; j < t; j++) {
                        M[qi * t + j] = hc.exp_scores[qi * T + selected[j]];
                    }
                }

                std::vector<float> w(t);
                nnls_solve(M.data(), hc.row_sums.data(), w.data(), n_q, t, p.nnls_max_iter);

                for (int j = 0; j < t; j++) {
                    beta_vecs[h][j] = logf(std::max(1e-12f, w[j]));
                }

                std::vector<float> X(n_q * t);
                for (int qi = 0; qi < n_q; qi++) {
                    for (int j = 0; j < t; j++) {
                        X[qi * t + j] = hc.scores[qi * T + selected[j]] + beta_vecs[h][j];
                    }
                }
                softmax_rows(X.data(), n_q, t);

                std::vector<float> Y(n_q * d_v, 0.0f);
                for (int qi = 0; qi < n_q; qi++) {
                    for (int ki = 0; ki < T; ki++) {
                        float w_ij = hc.attn_weights[qi * T + ki];
                        const float * v_row = V_all + ki * n_embd_v + h * d_v;
                        for (int d = 0; d < d_v; d++)
                            Y[qi * d_v + d] += w_ij * v_row[d];
                    }
                }

                least_squares_solve(X.data(), Y.data(), cv_vecs[h].data(),
                                   n_q, t, d_v, p.ridge);
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

    // Compute quality metrics
    compute_quality_metrics(K_all, V_all, Q_ref_all,
                           T, n_q, n_head_kv, d_k, d_v, result,
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

    if (!K_all || !V_all || !Q_ref_all || !result) return -1;
    if (T <= 0 || n_q <= 0 || n_head_kv <= 0 || d_k <= 0 || d_v <= 0) return -2;
    if (n_rounds < 1) return -3;

    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    // Track original indices through rounds
    std::vector<int> original_indices(T);
    std::iota(original_indices.begin(), original_indices.end(), 0);

    // Current KV data (starts as copy, modified each round)
    std::vector<float> cur_K(K_all, K_all + (size_t)T * n_embd_k);
    std::vector<float> cur_V(V_all, V_all + (size_t)T * n_embd_v);
    int cur_T = T;

    // Q_ref stays the same (from original context)
    // But we need to limit n_q to current T
    int cur_n_q = n_q;

    for (int round = 0; round < n_rounds; round++) {
        // Use original Q_ref for all rounds
        // This avoids using beta-distorted K as proxy queries
        int n_q_round = n_q < cur_T ? n_q : cur_T;

        kv_compact_result round_result = {};
        int rc = kv_compact(cur_K.data(), cur_V.data(), Q_ref_all,
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
    // result->beta from last round + result->C_v vs original K_all/V_all
    // Use the original indices to find the right K rows
    compute_quality_metrics(K_all, V_all, Q_ref_all,
                           T, n_q, n_head_kv, d_k, d_v, result,
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
