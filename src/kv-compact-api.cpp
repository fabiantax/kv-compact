// KV Cache Compaction API — implementation
//
// Wraps the compaction pipeline (importance scoring, key selection, NNLS,
// least-squares value refitting) into a single callable function.

#include "kv-compact-api.h"
#include "kv-compact-math.h"
#include "kv-compact-state.h"

#include "llama.h"
#include "log.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

int kv_compact_sequence(
    llama_context * ctx,
    llama_seq_id    seq_id,
    kv_compact_params params)
{
    if (!ctx) return -1;
    if (params.compact_ratio <= 0.0f || params.compact_ratio >= 1.0f) return -1;

    const llama_model * model = llama_get_model(ctx);
    if (!model) return -1;

    const int n_layer   = llama_model_n_layer(model);
    const int n_head_kv = llama_model_n_head_kv(model);

    // Determine rope type for M-RoPE support
    const enum llama_rope_type rope_type = llama_model_rope_type(model);
    const uint32_t n_pos_per_embd = (rope_type == LLAMA_ROPE_TYPE_MROPE ||
                                     rope_type == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Step 1: Save state
    const size_t state_size = llama_state_seq_get_size(ctx, seq_id);
    if (state_size == 0) return -1;

    std::vector<uint8_t> state_buf(state_size);
    const size_t saved = llama_state_seq_get_data(ctx, state_buf.data(), state_buf.size(), seq_id);
    if (saved == 0) return -1;

    // Step 2: Parse state
    parsed_kv_state kv_state;
    if (!kv_state.parse(state_buf.data(), saved, n_pos_per_embd)) {
        return -1;
    }

    if (kv_state.n_stream == 0 || kv_state.streams[0].cell_count == 0) return -1;

    const auto & sd = kv_state.streams[0];
    if (sd.n_layer == 0) return -1;

    const int T = (int)sd.cell_count;
    if (T < 4) return -1; // too few tokens to compact

    const int n_embd_k_gqa = sd.layers[0].n_embd_k_gqa();
    const int n_embd_v_gqa = sd.layers[0].n_embd_v_gqa_computed();
    const int d_k = n_embd_k_gqa / n_head_kv;
    const int d_v = n_embd_v_gqa / n_head_kv;

    if (d_k == 0 || d_v == 0) return -1;

    // Compute target size
    int t = std::max(2, (int)(T * params.compact_ratio));

    // Ensure we keep at least n_keep tokens
    if (params.n_keep > 0 && t < params.n_keep) {
        t = params.n_keep;
    }
    if (t >= T) return T; // nothing to compact

    // Reference queries: last quarter of context
    int n_ref = params.n_ref_queries;
    if (n_ref <= 0) {
        n_ref = std::max(16, T / 4);
    }
    n_ref = std::min(n_ref, T);
    const int ref_start = T - n_ref;

    const float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    // Step 3: Global importance scoring across all layers × heads
    std::vector<float> global_importance(T, 0.0f);

    // Cache per-layer per-head precomputed data
    struct head_cache {
        std::vector<float> scores;       // [n_ref, T]
        std::vector<float> exp_scores;   // [n_ref, T]
        std::vector<float> row_sums;     // [n_ref]
        std::vector<float> attn_weights; // [n_ref, T]
    };
    std::vector<std::vector<head_cache>> lh_cache(sd.n_layer);

    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];
        lh_cache[l].resize(n_head_kv);

        for (int h = 0; h < n_head_kv; h++) {
            auto & hc = lh_cache[l][h];
            hc.scores.resize(n_ref * T);
            hc.exp_scores.resize(n_ref * T);
            hc.row_sums.resize(n_ref);
            hc.attn_weights.resize(n_ref * T);

            // Compute scores: Q_ref_h @ K_h^T / sqrt(d_k)
            for (int qi = 0; qi < n_ref; qi++) {
                const float * q_row = ld.K.data() + (ref_start + qi) * n_embd_k_gqa + h * d_k;
                for (int ki = 0; ki < T; ki++) {
                    const float * k_row = ld.K.data() + ki * n_embd_k_gqa + h * d_k;
                    float dot = 0.0f;
                    for (int d = 0; d < d_k; d++) {
                        dot += q_row[d] * k_row[d];
                    }
                    hc.scores[qi * T + ki] = dot * inv_sqrt_dk;
                }
            }

            // exp + row sums (for NNLS)
            memcpy(hc.exp_scores.data(), hc.scores.data(), n_ref * T * sizeof(float));
            exp_rows_stable(hc.exp_scores.data(), hc.row_sums.data(), n_ref, T);

            // softmax (for importance + value fitting)
            memcpy(hc.attn_weights.data(), hc.scores.data(), n_ref * T * sizeof(float));
            softmax_rows(hc.attn_weights.data(), n_ref, T);

            // Per-key max attention across queries → global importance
            for (int j = 0; j < T; j++) {
                float max_w = 0.0f;
                for (int qi = 0; qi < n_ref; qi++) {
                    float w = hc.attn_weights[qi * T + j];
                    if (w > max_w) max_w = w;
                }
                if (max_w > global_importance[j]) {
                    global_importance[j] = max_w;
                }
            }
        }
    }

    // Step 4: Select top-t positions globally
    // Force-keep the first n_keep positions (sink tokens)
    if (params.n_keep > 0) {
        for (int i = 0; i < std::min(params.n_keep, T); i++) {
            global_importance[i] = 1e30f; // ensure they're always selected
        }
    }

    std::vector<int> indices(T);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                      [&](int a, int b) { return global_importance[a] > global_importance[b]; });

    std::vector<int> selected(indices.begin(), indices.begin() + t);
    std::sort(selected.begin(), selected.end());

    // Step 5: Per-layer, per-head NNLS (beta) + least-squares (C_v)
    std::vector<std::vector<std::vector<float>>> cv_all(sd.n_layer);

    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];
        cv_all[l].resize(n_head_kv);

        for (int h = 0; h < n_head_kv; h++) {
            const auto & hc = lh_cache[l][h];
            auto & cv = cv_all[l][h];
            cv.resize(t * d_v);

            // NNLS for beta
            std::vector<float> M(n_ref * t);
            for (int qi = 0; qi < n_ref; qi++) {
                for (int j = 0; j < t; j++) {
                    M[qi * t + j] = hc.exp_scores[qi * T + selected[j]];
                }
            }

            std::vector<float> w(t);
            nnls_solve(M.data(), hc.row_sums.data(), w.data(), n_ref, t);

            std::vector<float> beta(t);
            for (int j = 0; j < t; j++) {
                beta[j] = logf(std::max(1e-12f, w[j]));
            }

            // Least squares for C_v
            std::vector<float> X(n_ref * t);
            for (int qi = 0; qi < n_ref; qi++) {
                for (int j = 0; j < t; j++) {
                    X[qi * t + j] = hc.scores[qi * T + selected[j]] + beta[j];
                }
            }
            softmax_rows(X.data(), n_ref, t);

            // Y = original attention output: attn_weights @ V_head [n_ref, d_v]
            std::vector<float> Y(n_ref * d_v, 0.0f);
            for (int qi = 0; qi < n_ref; qi++) {
                for (int ki = 0; ki < T; ki++) {
                    float w_ij = hc.attn_weights[qi * T + ki];
                    const float * v_row = ld.V.data() + ki * n_embd_v_gqa + h * d_v;
                    for (int d = 0; d < d_v; d++) {
                        Y[qi * d_v + d] += w_ij * v_row[d];
                    }
                }
            }

            least_squares_solve(X.data(), Y.data(), cv.data(), n_ref, t, d_v);
        }
    }

    // Step 6: Build compacted state and write back
    auto compacted_buf = build_compacted_state(
        kv_state, selected, cv_all, n_head_kv, d_k, d_v, n_pos_per_embd);

    if (compacted_buf.empty()) return -1;

    // Clear and reload
    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, seq_id, -1, -1);

    size_t loaded = llama_state_seq_set_data(ctx, compacted_buf.data(), compacted_buf.size(), seq_id);
    if (loaded == 0) return -1;

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    LOG_INF("kv_compact: seq %d compacted %d → %d tokens (%.1fx) in %.1f ms\n",
            seq_id, T, t, (float)T / t, ms);

    return t;
}
