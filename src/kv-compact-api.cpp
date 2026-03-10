// KV Cache Compaction API — implementation
//
// Uses attention-matching importance scoring to select which KV positions to
// keep, then removes unimportant positions.
//
// For pure attention models: removes positions directly via llama_memory_seq_rm.
// For hybrid SSM+attention models: saves recurrent state separately, rebuilds
// compacted KV state, restores both independently.

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

// Score importance of each KV position using attention matching.
static int score_importance(
    const parsed_kv_state::stream_data & sd,
    int n_head_kv,
    int n_ref,
    std::vector<float> & global_importance)
{
    const int T = (int)sd.cell_count;
    const int n_embd_k_gqa = sd.layers[0].n_embd_k_gqa();
    const int d_k = n_embd_k_gqa / n_head_kv;
    const float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
    const int ref_start = T - n_ref;

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

    const int n_embd_k_gqa = sd.layers[0].n_embd_k_gqa();
    const int n_embd_v_gqa = sd.layers[0].n_embd_v_gqa_computed();
    const int d_k = n_embd_k_gqa / n_head_kv;
    const int d_v = n_embd_v_gqa / n_head_kv;
    if (d_k == 0) return -1;

    int t = std::max(2, (int)(T * params.compact_ratio));
    if (params.n_keep > 0 && t < params.n_keep) t = params.n_keep;
    if (t >= T) return T;

    int n_ref = params.n_ref_queries;
    if (n_ref <= 0) n_ref = std::max(16, T / 4);
    n_ref = std::min(n_ref, T);

    // Step 3: Importance scoring
    std::vector<float> global_importance;
    int scored_layers = score_importance(sd, n_head_kv, n_ref, global_importance);
    if (scored_layers == 0) return -1;

    // Step 4: Select top-t positions
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

    std::vector<bool> keep(T, false);
    for (int i = 0; i < t; i++) keep[indices[i]] = true;

    // Step 5: Detect model type and choose compaction strategy
    llama_memory_t mem = llama_get_memory(ctx);
    const bool is_hybrid = llama_model_is_hybrid(model);

    int result;

    if (!is_hybrid) {
        // Pure attention model: try direct KV eviction via llama_memory_seq_rm
        bool direct_ok = true;
        for (int i = T - 1; i >= 0; i--) {
            if (!keep[i]) {
                llama_pos pos = sd.cells[i].pos;
                if (!llama_memory_seq_rm(mem, seq_id, pos, pos + 1)) {
                    direct_ok = false;
                    break;
                }
            }
        }

        if (!direct_ok) {
            // Shouldn't happen for pure attention models, but handle gracefully
            LOG_ERR("kv_compact: seq %d direct eviction failed on non-hybrid model\n", seq_id);
            return -1;
        }
        result = t;
    } else {
        // Hybrid SSM+attention model: rebuild KV state, preserve recurrent state
        LOG_INF("kv_compact: seq %d using hybrid path (save recurrent + rebuild KV)\n", seq_id);

        // 1. Save recurrent state separately via PARTIAL_ONLY (skips KV, saves only recurrent)
        const size_t recr_size = llama_state_seq_get_size_ext(ctx, seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        std::vector<uint8_t> recr_buf(recr_size);
        size_t recr_saved = 0;
        if (recr_size > 0) {
            recr_saved = llama_state_seq_get_data_ext(ctx, recr_buf.data(), recr_buf.size(),
                                                       seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (recr_saved == 0) {
                LOG_ERR("kv_compact: seq %d failed to save recurrent state\n", seq_id);
                return -1;
            }
        }

        // 2. Build compacted V values for selected positions
        std::vector<std::vector<std::vector<float>>> cv_all(sd.n_layer);
        for (uint32_t l = 0; l < sd.n_layer; l++) {
            const auto & ld = sd.layers[l];
            cv_all[l].resize(n_head_kv);
            for (int h = 0; h < n_head_kv; h++) {
                auto & cv = cv_all[l][h];
                cv.resize(t * d_v);
                for (int j = 0; j < t; j++) {
                    int orig_idx = selected[j];
                    const float * v_row = ld.V.data() + orig_idx * n_embd_v_gqa + h * d_v;
                    memcpy(cv.data() + j * d_v, v_row, d_v * sizeof(float));
                }
            }
        }

        // 3. Renumber selected cell positions to [0..t-1] to avoid position
        //    conflicts when the server continues generating from position t.
        //    Also update the recurrent state's tail position to match.
        parsed_kv_state compacted_state = kv_state;
        for (uint32_t s = 0; s < compacted_state.n_stream; s++) {
            auto & csd = compacted_state.streams[s];
            // Renumber only the selected cells to [0, 1, 2, ..., t-1]
            for (int j = 0; j < t; j++) {
                csd.cells[selected[j]].pos = (int32_t)j;
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

        // 4. Build full compacted state buffer (KV + recurrent trailing data)
        auto compacted_buf = build_compacted_state(
            compacted_state, selected, cv_all, n_head_kv, d_k, d_v, n_pos_per_embd);

        if (compacted_buf.empty()) return -1;

        // 5. Clear everything and restore compacted state in one step
        llama_memory_seq_rm(mem, seq_id, -1, -1);

        size_t loaded = llama_state_seq_set_data(ctx, compacted_buf.data(),
                                                  compacted_buf.size(), seq_id);
        if (loaded == 0) {
            // Full state restore failed — try to recover by restoring recurrent only
            LOG_ERR("kv_compact: seq %d full state restore failed, attempting recurrent recovery\n", seq_id);
            if (recr_saved > 0) {
                llama_state_seq_set_data_ext(ctx, recr_buf.data(), recr_buf.size(),
                                              seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            }
            return -1;
        }

        // 6. Restore recurrent state from clean PARTIAL_ONLY save to ensure consistency.
        //    This overwrites the recurrent data from trailing_data with a fresh copy
        //    saved via the proper llama.cpp API, eliminating any format alignment issues.
        if (recr_saved > 0) {
            size_t recr_loaded = llama_state_seq_set_data_ext(ctx, recr_buf.data(), recr_buf.size(),
                                                               seq_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            if (recr_loaded == 0) {
                LOG_ERR("kv_compact: seq %d failed to restore recurrent state (non-fatal)\n", seq_id);
                // Continue anyway — the trailing_data restore might be sufficient
            }
        }

        result = t;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    LOG_INF("kv_compact: seq %d compacted %d → %d tokens (%.1fx) in %.1f ms (%d attn layers scored)\n",
            seq_id, T, result, (float)T / result, ms, scored_layers);

    return result;
}
