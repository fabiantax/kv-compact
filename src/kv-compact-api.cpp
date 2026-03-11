// KV Cache Compaction API — implementation
//
// Implements "Fast KV Compaction via Attention Matching" (Zweiger et al., 2026)
// https://arxiv.org/abs/2602.16284
//
// Uses attention-matching importance scoring to select which KV positions to
// keep, then refits the compacted values via least-squares optimization.
//
// Pipeline (paper Section 3, algorithms.md §2):
//   importance scoring → NNLS bias fitting → least-squares value refitting
//   → state rebuild via llama_state_seq_set_data()
//
// All models (pure attention and hybrid SSM+attention) go through the same
// refitting pipeline. Hybrid models additionally save/restore recurrent state.
//
// NOT IMPLEMENTED (paper ideas — see algorithms.md §12 for full list):
//   - Iterative compaction (paper §6.1): apply compaction multiple times as
//     context grows. Paper shows 6 consecutive compressions without quality loss.
//     Callers can invoke kv_compact_sequence() repeatedly, but no built-in
//     trigger or scheduling logic exists yet.
//   - Beta injection during inference (paper §4.1): betas are computed but
//     discarded — llama.cpp has no attention bias hook. See refit_head_values().
//   - Non-uniform per-head budgets (paper §6.2): all heads use same target t.
//   - OMP key selection (paper §5.2): uses simpler max-attention scoring instead.

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

// ============================================================================
// Bandwidth-aware ratio suggestion
// ============================================================================

float kv_compact_suggest_ratio(
    const llama_model * model,
    int ctx_size,
    float mem_budget_mb,
    int n_parallel,
    enum ggml_type type_k,
    enum ggml_type type_v)
{
    if (!model) return -1.0f;

    const int n_layer   = llama_model_n_layer(model);
    const int n_embd    = llama_model_n_embd(model);
    const int n_head    = llama_model_n_head(model);
    const int n_head_kv = llama_model_n_head_kv(model);

    // Compute bytes per element from ggml types
    // ggml_type_size returns bytes per block, ggml_blck_size returns elements per block
    const float bpe_k = (float)ggml_type_size(type_k) / (float)ggml_blck_size(type_k);
    const float bpe_v = (float)ggml_type_size(type_v) / (float)ggml_blck_size(type_v);

    float ratio = compute_suggest_ratio(n_layer, n_embd, n_head, n_head_kv,
                                        ctx_size, mem_budget_mb, n_parallel,
                                        bpe_k, bpe_v);

    if (ratio > 0.0f && ratio < 1.0f) {
        const int d_head = n_embd / n_head;
        const int n_embd_kv_gqa = d_head * n_head_kv;
        const float bytes_per_token_per_layer =
            n_embd_kv_gqa * bpe_k + n_embd_kv_gqa * bpe_v;
        const float kv_bytes_per_seq = bytes_per_token_per_layer * n_layer * ctx_size;
        const float total_kv_bytes = kv_bytes_per_seq * n_parallel;

        LOG_INF("kv_compact_suggest_ratio: KV cache type K=%s V=%s (%.2f + %.2f bpe), "
                "%.1f MB/seq × %d parallel = %.1f MB total, budget = %.1f MB → ratio = %.2f\n",
                ggml_type_name(type_k), ggml_type_name(type_v), bpe_k, bpe_v,
                kv_bytes_per_seq / (1024.0f * 1024.0f), n_parallel,
                total_kv_bytes / (1024.0f * 1024.0f), mem_budget_mb, ratio);
    }

    return ratio;
}

// ============================================================================
// Importance scoring
// ============================================================================

// Score importance of each KV position using attention matching (paper §3.1).
// Per-key importance = max softmax attention weight across all reference queries
// and all layers/heads (algorithms.md §3.3). Uses K vectors from the last
// n_ref positions as proxy queries (simplest reference query strategy, paper §4).
//
// NOT IMPLEMENTED (paper §7.2, algorithms.md §12.4): better reference queries.
// Current approach uses trailing K vectors as query proxies. The paper suggests
// true repeat-prefill (running context through the model twice) or learned
// representative query sets for higher-quality importance estimates.
static int score_importance(
    const parsed_kv_state::stream_data & sd,
    int n_head_kv,
    int n_ref,
    std::vector<float> & global_importance)
{
    const int T = (int)sd.cell_count;
    if (n_head_kv == 0) return 0;
    const int n_embd_k_gqa = sd.layers[0].n_embd_k_gqa();
    const int d_k = n_embd_k_gqa / n_head_kv;
    if (d_k == 0) return 0;
    const float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
    const int ref_start = std::max(0, T - n_ref);

    global_importance.assign(T, 0.0f);
    int scored_layers = 0;

    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];
        if (ld.K.empty()) continue;
        scored_layers++;

        for (int h = 0; h < n_head_kv; h++) {
            std::vector<float> attn_weights(n_ref * T);

            for (int qi = 0; qi < n_ref; qi++) {
                const float * q_row = ld.K.data() + (ref_start + qi) * n_embd_k_gqa + h * d_k;
                for (int ki = 0; ki < T; ki++) {
                    const float * k_row = ld.K.data() + ki * n_embd_k_gqa + h * d_k;
                    float dot = 0.0f;
                    for (int d = 0; d < d_k; d++) {
                        dot += q_row[d] * k_row[d];
                    }
                    attn_weights[qi * T + ki] = dot * inv_sqrt_dk;
                }
            }

            softmax_rows(attn_weights.data(), n_ref, T);

            for (int j = 0; j < T; j++) {
                float max_w = 0.0f;
                for (int qi = 0; qi < n_ref; qi++) {
                    float w = attn_weights[qi * T + j];
                    if (w > max_w) max_w = w;
                }
                if (max_w > global_importance[j]) {
                    global_importance[j] = max_w;
                }
            }
        }
    }

    return scored_layers;
}

int kv_compact_sequence(
    llama_context * ctx,
    llama_seq_id    seq_id,
    kv_compact_params params)
{
    if (!ctx) return -1;
    if (params.compact_ratio <= 0.0f || params.compact_ratio >= 1.0f) return -1;

    const llama_model * model = llama_get_model(ctx);
    if (!model) return -1;

    const int n_head_kv = llama_model_n_head_kv(model);
    const enum llama_rope_type rope_type = llama_model_rope_type(model);
    const uint32_t n_pos_per_embd = (rope_type == LLAMA_ROPE_TYPE_MROPE ||
                                     rope_type == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;

    auto t_start = std::chrono::high_resolution_clock::now();

    // Step 1: Save full state for importance scoring
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
    if (T < 4) return -1;

    if (n_head_kv == 0) return -1;

    const int n_embd_k_gqa = sd.layers[0].n_embd_k_gqa();
    const int n_embd_v_gqa = sd.layers[0].n_embd_v_gqa_computed();
    const int d_k = n_embd_k_gqa / n_head_kv;
    const int d_v = n_embd_v_gqa / n_head_kv;
    if (d_k == 0 || d_v == 0) return -1;

    int t = std::max(2, (int)(T * params.compact_ratio));
    if (params.n_keep > 0 && t < params.n_keep) t = params.n_keep;
    if (t >= T) return T;

    int n_ref = params.n_ref_queries;
    if (n_ref <= 0) n_ref = std::max(16, T / 4);
    n_ref = std::min(n_ref, T);

    // Step 3: Importance scoring (paper §3.1, algorithms.md §3)
    std::vector<float> global_importance;
    int scored_layers = score_importance(sd, n_head_kv, n_ref, global_importance);
    if (scored_layers == 0) return -1;

    // Step 4: Select top-t positions (paper §3.1, algorithms.md §3.4)
    if (params.n_keep > 0) {
        for (int i = 0; i < std::min(params.n_keep, T); i++) {
            global_importance[i] = 1e30f;
        }
    }

    std::vector<int> indices(T);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                      [&](int a, int b) { return global_importance[a] > global_importance[b]; });

    std::vector<int> selected(indices.begin(), indices.begin() + t);
    std::sort(selected.begin(), selected.end());

    // Fill kept_positions output if caller provided a buffer
    if (params.kept_positions && params.kept_positions_cap > 0) {
        int n_out = std::min(t, params.kept_positions_cap);
        for (int i = 0; i < n_out; i++) {
            params.kept_positions[i] = (int32_t)sd.cells[selected[i]].pos;
        }
    }

    // Step 5: Per-layer, per-head NNLS bias + least-squares value refitting
    //         (paper §3.2-3.3, algorithms.md §4-5)
    //         C_v is fitted with un-biased softmax so it works at inference time
    //         (betas are not stored in the state format — see algorithms.md §12.1).
    const int ref_start_refit = std::max(0, T - n_ref);

    std::vector<std::vector<std::vector<float>>> cv_all(sd.n_layer);
    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];
        cv_all[l].resize(n_head_kv);

        if (ld.K.empty()) {
            // SSM layer with no K data — fill with zeros
            for (int h = 0; h < n_head_kv; h++) {
                cv_all[l][h].assign(t * d_v, 0.0f);
            }
            continue;
        }

        for (int h = 0; h < n_head_kv; h++) {
            auto rr = refit_head_values(
                ld.K.data(), ld.V.data(),
                T, n_embd_k_gqa, n_embd_v_gqa,
                h, d_k, d_v,
                n_ref, ref_start_refit,
                selected,
                false /* use_beta_for_cv: false since betas aren't stored */);

            cv_all[l][h] = std::move(rr.C_v);
        }
    }

    // Step 6: Renumber selected cell positions to [0..t-1] so the server can
    //         continue generating from position t without conflicts.
    parsed_kv_state compacted_state = kv_state;
    for (uint32_t s = 0; s < compacted_state.n_stream; s++) {
        auto & csd = compacted_state.streams[s];
        for (int j = 0; j < t; j++) {
            csd.cells[selected[j]].pos = (int32_t)j;
        }
    }

    // Step 7: Handle hybrid recurrent state if present
    llama_memory_t mem = llama_get_memory(ctx);
    const bool is_hybrid = llama_model_is_hybrid(model);

    std::vector<uint8_t> recr_buf;
    size_t recr_saved = 0;

    if (is_hybrid) {
        LOG_INF("kv_compact: seq %d using hybrid path (save recurrent + rebuild KV)\n", seq_id);

        // Save recurrent state separately via PARTIAL_ONLY (skips KV, saves only recurrent)
        const size_t recr_size = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        recr_buf.resize(recr_size);
        if (recr_size > 0) {
            recr_saved = llama_state_seq_get_data_ext(ctx, recr_buf.data(), recr_buf.size(),
                                                       seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (recr_saved == 0) {
                LOG_ERR("kv_compact: seq %d failed to save recurrent state\n", seq_id);
                return -1;
            }
        }

        // Update recurrent state's tail position to t-1 so it matches the
        // compacted KV positions. Recurrent state format starts with:
        //   [cell_count:u32] [pos:i32] [n_seq_id:u32] ...
        if (!compacted_state.trailing_data.empty() && compacted_state.trailing_data.size() >= 8) {
            uint32_t recr_cell_count = 0;
            memcpy(&recr_cell_count, compacted_state.trailing_data.data(), sizeof(uint32_t));
            if (recr_cell_count > 0) {
                int32_t new_tail_pos = t - 1;
                memcpy(compacted_state.trailing_data.data() + 4, &new_tail_pos, sizeof(int32_t));
            }
        }
    }

    // Step 8: Build compacted state buffer with refitted values and restore
    auto compacted_buf = build_compacted_state(
        compacted_state, selected, cv_all, n_head_kv, d_k, d_v, n_pos_per_embd);

    if (compacted_buf.empty()) return -1;

    llama_memory_seq_rm(mem, seq_id, -1, -1);

    size_t loaded = llama_state_seq_set_data(ctx, compacted_buf.data(),
                                              compacted_buf.size(), seq_id);
    if (loaded == 0) {
        LOG_ERR("kv_compact: seq %d compacted state restore failed, restoring original\n", seq_id);
        // Restore original state from the buffer we saved in step 1
        size_t restored = llama_state_seq_set_data(ctx, state_buf.data(), saved, seq_id);
        if (restored == 0) {
            LOG_ERR("kv_compact: seq %d original state restore also failed!\n", seq_id);
        }
        if (recr_saved > 0) {
            llama_state_seq_set_data_ext(ctx, recr_buf.data(), recr_buf.size(),
                                          seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }
        return -1;
    }

    // For hybrid models, restore recurrent state from clean PARTIAL_ONLY save
    if (recr_saved > 0) {
        size_t recr_loaded = llama_state_seq_set_data_ext(ctx, recr_buf.data(), recr_buf.size(),
                                                           seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        if (recr_loaded == 0) {
            LOG_ERR("kv_compact: seq %d failed to restore recurrent state (non-fatal)\n", seq_id);
        }
    }

    int result = t;

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    LOG_INF("kv_compact: seq %d compacted %d → %d tokens (%.1fx) in %.1f ms (%d attn layers scored)\n",
            seq_id, T, result, (float)T / result, ms, scored_layers);

    return result;
}
