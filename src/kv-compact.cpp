// KV Cache Compaction via Attention Matching
//
// Implements the "Highest Attention Keys" variant from:
//   "Fast KV Compaction via Attention Matching" (Zweiger et al., 2026)
//   https://arxiv.org/abs/2602.16284
//
// Algorithm:
//   1. Prefill context to fill KV cache
//   2. Save state → parse → extract per-layer K/V
//   3. For each layer: global key selection + per-head NNLS + C_v refitting
//   4. Build compacted state buffer → write back → generate
//   5. Compare generation quality: full cache vs eviction vs attention matching

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include "ggml.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "kv-compact-math.h"
#include "kv-compact-state.h"
#include "kv-compact-optimized.h"

// ============================================================================
// Helpers
// ============================================================================

static std::string generate_tokens(llama_context * ctx, llama_model * model,
                                   const llama_vocab * vocab,
                                   common_params & params,
                                   llama_pos start_pos, int n_gen) {
    std::string output;
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_sampler * smpl = common_sampler_init(model, params.sampling);

    for (int i = 0; i < n_gen; i++) {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;

        output += common_token_to_piece(vocab, id);
        common_sampler_accept(smpl, id, true);

        common_batch_clear(batch);
        common_batch_add(batch, id, start_pos + i, {0}, true);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("Failed to decode during generation\n");
            break;
        }
    }

    common_sampler_free(smpl);
    llama_batch_free(batch);
    return output;
}

// ============================================================================
// Main
// ============================================================================

static void print_usage(int argc, char ** argv) {
    (void) argc;
    LOG("\nKV Cache Compaction via Attention Matching\n\n");
    LOG("Usage: %s [options]\n\n", argv[0]);
    LOG("  -m  MODEL         path to model file\n");
    LOG("  -p  PROMPT        input context to compact\n");
    LOG("  -f  FILE          read context from file\n");
    LOG("  -c  N             context size (default: 2048)\n");
    LOG("  --compact-ratio R compaction ratio (default: 0.2, meaning keep 20%%)\n");
    LOG("  --n-ref-queries N reference queries (default: 0 = last quarter of context)\n");
    LOG("  -n  N             tokens to generate after compaction (default: 128)\n");
    LOG("  --no-writeback    skip write-back (demo mode: compute quality metrics only)\n");
    LOG("  --no-eviction     skip token eviction baseline\n");
    LOG("\n");
    LOG("Streaming compaction options (Phase 1):\n");
    LOG("  --pin-prefix N    pin first N tokens (system prompt, default: 256)\n");
    LOG("  --recent-window N keep last N tokens uncompactable (default: 512)\n");
    LOG("  --trigger N       compact when cache exceeds N tokens (default: 8192)\n");
    LOG("  --budget N        target budget after compaction (default: 4096)\n");
    LOG("\n");
    LOG("Sublinear optimization (Phase 2):\n");
    LOG("  --optimized       enable sublinear optimizations (O(n log n) instead of O(n²))\n");
    LOG("  --method M        compaction method: baseline|l2|hybrid (default: baseline)\n");
    LOG("  --early-stop      enable early stopping in NNLS (reduces iterations)\n");
    LOG("  --layer-budget    enable layer-wise budget allocation\n");
    LOG("\n");
}

int main(int argc, char ** argv) {
    common_params params;

    // Custom parameters
    float compact_ratio = 0.2f;
    int   n_ref_queries = 0;   // 0 = auto (last quarter)
    bool  do_writeback  = true;
    bool  do_eviction   = true;

    // Streaming compaction parameters (Phase 1)
    int stream_pin_prefix = 256;
    int stream_recent_window = 512;
    int stream_trigger = 8192;
    int stream_budget = 4096;

    // Sublinear optimization parameters (Phase 2)
    bool use_optimized = false;
    std::string compaction_method = "baseline";  // baseline, l2, hybrid
    bool enable_early_stop = false;
    bool enable_layer_budget = false;

    // Parse custom args FIRST (before common_params_parse which rejects unknowns)
    // Build a filtered argv for llama.cpp's parser
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--compact-ratio") == 0 && i + 1 < argc) {
            compact_ratio = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--n-ref-queries") == 0 && i + 1 < argc) {
            n_ref_queries = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-writeback") == 0) {
            do_writeback = false;
        } else if (strcmp(argv[i], "--no-eviction") == 0) {
            do_eviction = false;
        } else if (strcmp(argv[i], "--pin-prefix") == 0 && i + 1 < argc) {
            stream_pin_prefix = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--recent-window") == 0 && i + 1 < argc) {
            stream_recent_window = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--trigger") == 0 && i + 1 < argc) {
            stream_trigger = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--budget") == 0 && i + 1 < argc) {
            stream_budget = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--optimized") == 0) {
            use_optimized = true;
        } else if (strcmp(argv[i], "--method") == 0 && i + 1 < argc) {
            compaction_method = argv[++i];
        } else if (strcmp(argv[i], "--early-stop") == 0) {
            enable_early_stop = true;
        } else if (strcmp(argv[i], "--layer-budget") == 0) {
            enable_layer_budget = true;
        } else {
            // Pass through to llama.cpp parser
            filtered_argv.push_back(argv[i]);
        }
    }

    int filtered_argc = (int)filtered_argv.size();

    // Parse standard params (llama.cpp args only)
    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMPLETION, print_usage)) {
        return 1;
    }

    if (compact_ratio <= 0.0f || compact_ratio >= 1.0f) {
        LOG_ERR("compact-ratio must be between 0 and 1 (exclusive)\n");
        return 1;
    }

    // Validate streaming parameters
    if (stream_pin_prefix < 0) {
        LOG_ERR("pin-prefix must be non-negative\n");
        return 1;
    }
    if (stream_recent_window < 0) {
        LOG_ERR("recent-window must be non-negative\n");
        return 1;
    }
    if (stream_trigger <= stream_budget) {
        LOG_ERR("trigger must be greater than budget\n");
        return 1;
    }
    if (stream_pin_prefix + stream_recent_window >= stream_budget) {
        LOG_ERR("pin-prefix + recent-window must be less than budget\n");
        return 1;
    }

    // Validate optimization parameters
    if (compaction_method != "baseline" && compaction_method != "l2" && compaction_method != "hybrid") {
        LOG_ERR("Invalid method: %s (must be baseline, l2, or hybrid)\n", compaction_method.c_str());
        return 1;
    }

    // Log streaming config if any streaming flag was used
    bool use_streaming = (stream_trigger != 8192 || stream_budget != 4096 ||
                          stream_pin_prefix != 256 || stream_recent_window != 512);

    // Log optimization config if any optimization flag was used
    bool use_optimizations = (use_optimized || compaction_method != "baseline" ||
                              enable_early_stop || enable_layer_budget);

    if (use_streaming) {
        LOG_INF("Streaming mode ENABLED: budget=%d, trigger=%d, pin=%d, recent=%d\n",
                stream_budget, stream_trigger, stream_pin_prefix, stream_recent_window);
    }

    if (use_optimizations) {
        LOG_INF("Optimization mode ENABLED: method=%s, early_stop=%s, layer_budget=%s\n",
                compaction_method.c_str(),
                enable_early_stop ? "ON" : "OFF",
                enable_layer_budget ? "ON" : "OFF");
    }

    // Construct streaming_config
    streaming_config stream_cfg;
    stream_cfg.budget = stream_budget;
    stream_cfg.trigger = stream_trigger;
    stream_cfg.pin_prefix = stream_pin_prefix;
    stream_cfg.recent_window = stream_recent_window;
    stream_cfg.select_mode = KEY_SELECT_MAX_ATTN;
    stream_cfg.fit_mode = BETA_FIT_NNLS;
    stream_cfg.n_alt_rounds = 2;

    if (use_streaming && !stream_cfg.is_valid()) {
        LOG_ERR("Invalid streaming configuration\n");
        return 1;
    }

    common_init();

    LOG_INF("=== KV Cache Compaction via Attention Matching ===\n");
    LOG_INF("Compaction ratio: keep %.1f%% of cache\n", compact_ratio * 100.0f);

    // ---- Initialize ----
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_context * ctx   = llama_init->context();
    llama_model   * model = llama_init->model();

    if (!ctx) {
        LOG_ERR("Failed to create context\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_ctx     = llama_n_ctx(ctx);
    const int n_layer   = llama_model_n_layer(model);
    const int n_head    = llama_model_n_head(model);
    const int n_head_kv = llama_model_n_head_kv(model);
    const int n_embd    = llama_model_n_embd(model);
    const enum llama_rope_type rope_type = llama_model_rope_type(model);
    const uint32_t n_pos_per_embd = (rope_type == LLAMA_ROPE_TYPE_MROPE ||
                                     rope_type == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;
    LOG_INF("Model: %d layers, %d heads (%d KV), n_embd=%d, context=%d, rope_type=%d\n",
            n_layer, n_head, n_head_kv, n_embd, n_ctx, (int)rope_type);

    // ---- Tokenize ----
    std::string prompt = params.prompt;
    if (prompt.empty()) {
        LOG_ERR("No input prompt. Use -p or -f.\n");
        return 1;
    }

    std::vector<llama_token> tokens = common_tokenize(vocab, prompt, true, false);
    const int n_tokens = (int) tokens.size();

    if (n_tokens < 16) {
        LOG_ERR("Input too short (%d tokens). Need >= 16.\n", n_tokens);
        return 1;
    }

    const int t = std::max(1, (int)(n_tokens * compact_ratio));
    LOG_INF("Input: %d tokens → compact to %d (%.1fx compression)\n",
            n_tokens, t, (float) n_tokens / t);

    // Auto-set reference queries: last quarter of context
    if (n_ref_queries <= 0) {
        n_ref_queries = std::max(16, n_tokens / 4);
    }
    n_ref_queries = std::min(n_ref_queries, n_tokens);
    LOG_INF("Reference queries: %d (from last quarter of context)\n", n_ref_queries);

    // ============================================================
    // Phase 1: Prefill
    // ============================================================
    LOG_INF("\n--- Phase 1: Prefill ---\n");

    llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        common_batch_add(batch, tokens[i], i, {0}, (i == n_tokens - 1));
    }

    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("Prefill failed\n");
        llama_batch_free(batch);
        return 1;
    }
    LOG_INF("Prefill complete: %d tokens in KV cache\n", n_tokens);

    // Save full state
    const size_t state_size = llama_state_seq_get_size(ctx, 0);
    std::vector<uint8_t> full_state(state_size);
    const size_t saved = llama_state_seq_get_data(ctx, full_state.data(), full_state.size(), 0);
    if (saved == 0) {
        LOG_ERR("Failed to save state\n");
        llama_batch_free(batch);
        return 1;
    }
    LOG_INF("State saved: %.2f MB\n", saved / (1024.0 * 1024.0));

    // ============================================================
    // Phase 2: Generate with full cache (reference output)
    // ============================================================
    LOG_INF("\n--- Phase 2: Full cache generation (reference) ---\n");

    const int n_gen = std::min(params.n_predict > 0 ? params.n_predict : 128, n_ctx - n_tokens);
    std::string full_output = generate_tokens(ctx, model, vocab, params, n_tokens, n_gen);
    LOG_INF("Full output:\n%s\n", full_output.c_str());

    llama_memory_t mem = llama_get_memory(ctx);

    // ============================================================
    // Phase 3: Token eviction baseline
    // ============================================================
    std::string evict_output;
    if (do_eviction) {
        LOG_INF("\n--- Phase 3: Token eviction baseline ---\n");

        // Restore original state
        llama_memory_seq_rm(mem, 0, -1, -1);
        llama_state_seq_set_data(ctx, full_state.data(), full_state.size(), 0);

        // Heuristic: keep sinks + recent + uniformly sampled middle
        const int n_sink = std::min(4, t / 4);
        const int n_recent = std::min(t / 2, n_tokens);
        const int n_middle = t - n_sink - n_recent;

        std::vector<bool> keep(n_tokens, false);
        for (int i = 0; i < n_sink && i < n_tokens; i++) keep[i] = true;
        for (int i = std::max(0, n_tokens - n_recent); i < n_tokens; i++) keep[i] = true;

        if (n_middle > 0 && n_tokens > n_sink + n_recent) {
            const int mid_start = n_sink;
            const int mid_end = n_tokens - n_recent;
            const float step = (float)(mid_end - mid_start) / (float)n_middle;
            for (int i = 0; i < n_middle; i++) {
                int idx = mid_start + (int)(i * step);
                if (idx < mid_end) keep[idx] = true;
            }
        }

        int n_kept = 0;
        for (int i = 0; i < n_tokens; i++) if (keep[i]) n_kept++;

        for (int i = n_tokens - 1; i >= 0; i--) {
            if (!keep[i]) llama_memory_seq_rm(mem, 0, i, i + 1);
        }

        LOG_INF("Eviction: keeping %d / %d tokens\n", n_kept, n_tokens);

        llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
        evict_output = generate_tokens(ctx, model, vocab, params, pos_max + 1, n_gen);
        LOG_INF("Eviction output:\n%s\n", evict_output.c_str());
    }

    // ============================================================
    // Phase 4: Attention Matching compaction (all layers × heads)
    // ============================================================
    LOG_INF("\n--- Phase 4: Attention Matching compaction ---\n");

    // Parse the saved state
    parsed_kv_state kv_state;
    if (!kv_state.parse(full_state.data(), saved, n_pos_per_embd)) {
        LOG_ERR("Failed to parse state buffer\n");
        llama_batch_free(batch);
        return 1;
    }

    LOG_INF("Parsed state: %u streams\n", kv_state.n_stream);

    // We compact stream 0 (seq_id=0)
    const auto & sd = kv_state.streams[0];
    LOG_INF("Stream 0: %u cells, %u layers, v_trans=%u\n",
            sd.cell_count, sd.n_layer, sd.v_trans);

    // Validate dimensions
    if (sd.n_layer == 0 || sd.cell_count == 0) {
        LOG_ERR("Empty state\n");
        llama_batch_free(batch);
        return 1;
    }

    const int n_embd_k_gqa = sd.layers[0].n_embd_k_gqa();
    const int n_embd_v_gqa = sd.layers[0].n_embd_v_gqa_computed();
    // Use parsed state dimensions (handles GQA and hybrid models correctly)
    const int d_k = n_embd_k_gqa / n_head_kv;
    const int d_v = n_embd_v_gqa / n_head_kv;
    LOG_INF("Dimensions: n_embd_k_gqa=%d, n_embd_v_gqa=%d, d_k=%d, d_v=%d (from state)\n",
            n_embd_k_gqa, n_embd_v_gqa, d_k, d_v);

    // Reference queries: use K from last quarter of context (all heads)
    const int ref_start = (int)sd.cell_count - n_ref_queries;
    // Q_ref_all: [n_ref_queries, n_embd_k_gqa]
    // We use K vectors as proxy queries (Q and K share similar structure)

    auto t_start = std::chrono::high_resolution_clock::now();

    // Compact each layer independently, but with shared key selection per layer
    std::vector<int> shared_selected;  // will be set by first layer, may differ per layer
    std::vector<std::vector<std::vector<float>>> cv_all(sd.n_layer);

    // For the state writer, we need a single shared selection across ALL layers
    // (because cell positions must be consistent across layers in the state format)
    // Strategy: compute importance per layer, aggregate, then select globally

    const float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);
    const int T = (int) sd.cell_count;

    // Store per-layer precomputed data for reuse in NNLS/LSQ steps
    struct layer_head_cache {
        std::vector<float> scores;       // [n_q, T]
        std::vector<float> exp_scores;   // [n_q, T]
        std::vector<float> row_sums;     // [n_q]
        std::vector<float> attn_weights; // [n_q, T]
    };

    // ---- Step 1: Key importance scoring and selection ----
    // Global importance: max across all layers and heads
    std::vector<float> global_importance(sd.cell_count, 0.0f);

    // Cache only needed for baseline and NNLS/LSQ phases
    std::vector<std::vector<layer_head_cache>> lh_cache(sd.n_layer);

    if (use_optimized && (compaction_method == "l2" || compaction_method == "hybrid")) {
        // ---- OPTIMIZED: L2 importance estimation ----
        LOG_INF("Using L2 importance estimation (method=%s)...\n", compaction_method.c_str());

        for (uint32_t l = 0; l < sd.n_layer; l++) {
            const auto & ld = sd.layers[l];

            for (int h = 0; h < n_head_kv; h++) {
                // Use L2 norm + query dot product for importance
                auto importance = kvcompact::optimized::FastImportanceEstimator::estimate_importance_l2(
                    ld.K.data() + h * d_k,  // keys for this head (strided)
                    ld.K.data() + ref_start * n_embd_k_gqa + h * d_k,  // queries for this head
                    T, n_ref_queries, d_k
                );

                // NOTE: keys/queries are stored with stride n_embd_k_gqa, not d_k.
                // Recompute with correct strided access:
                for (int j = 0; j < T; j++) {
                    const float * k_row = ld.K.data() + j * n_embd_k_gqa + h * d_k;

                    // L2 norm of key
                    float key_norm = 0.0f;
                    for (int d = 0; d < d_k; d++) {
                        key_norm += k_row[d] * k_row[d];
                    }
                    key_norm = sqrtf(key_norm);

                    // Mean absolute dot product with reference queries
                    float query_importance = 0.0f;
                    for (int qi = 0; qi < n_ref_queries; qi++) {
                        const float * q_row = ld.K.data() + (ref_start + qi) * n_embd_k_gqa + h * d_k;
                        float dot = 0.0f;
                        for (int d = 0; d < d_k; d++) {
                            dot += q_row[d] * k_row[d];
                        }
                        query_importance += fabsf(dot);
                    }

                    float imp = key_norm * (query_importance / n_ref_queries);
                    if (imp > global_importance[j]) {
                        global_importance[j] = imp;
                    }
                }
            }

            if ((l + 1) % 8 == 0 || l + 1 == sd.n_layer) {
                LOG_INF("  L2-scored %u / %u layers\n", l + 1, sd.n_layer);
            }
        }

        // We still need the attention cache for NNLS/LSQ phases
        LOG_INF("Computing attention cache for NNLS/LSQ...\n");
        for (uint32_t l = 0; l < sd.n_layer; l++) {
            const auto & ld = sd.layers[l];
            lh_cache[l].resize(n_head_kv);

            for (int h = 0; h < n_head_kv; h++) {
                auto & hc = lh_cache[l][h];
                hc.scores.resize(n_ref_queries * T);
                hc.exp_scores.resize(n_ref_queries * T);
                hc.row_sums.resize(n_ref_queries);
                hc.attn_weights.resize(n_ref_queries * T);

                for (int qi = 0; qi < n_ref_queries; qi++) {
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

                memcpy(hc.exp_scores.data(), hc.scores.data(), n_ref_queries * T * sizeof(float));
                exp_rows_stable(hc.exp_scores.data(), hc.row_sums.data(), n_ref_queries, T);

                memcpy(hc.attn_weights.data(), hc.scores.data(), n_ref_queries * T * sizeof(float));
                softmax_rows(hc.attn_weights.data(), n_ref_queries, T);
            }
        }
    } else {
        // ---- BASELINE: Full attention scoring ----
        LOG_INF("Computing global key importance across %u layers × %d heads...\n",
                sd.n_layer, n_head_kv);

        for (uint32_t l = 0; l < sd.n_layer; l++) {
            const auto & ld = sd.layers[l];
            lh_cache[l].resize(n_head_kv);

            for (int h = 0; h < n_head_kv; h++) {
                auto & hc = lh_cache[l][h];
                hc.scores.resize(n_ref_queries * T);
                hc.exp_scores.resize(n_ref_queries * T);
                hc.row_sums.resize(n_ref_queries);
                hc.attn_weights.resize(n_ref_queries * T);

                for (int qi = 0; qi < n_ref_queries; qi++) {
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

                memcpy(hc.exp_scores.data(), hc.scores.data(), n_ref_queries * T * sizeof(float));
                exp_rows_stable(hc.exp_scores.data(), hc.row_sums.data(), n_ref_queries, T);

                memcpy(hc.attn_weights.data(), hc.scores.data(), n_ref_queries * T * sizeof(float));
                softmax_rows(hc.attn_weights.data(), n_ref_queries, T);

                // Per-key max attention across queries → global importance
                for (int j = 0; j < T; j++) {
                    float max_w = 0.0f;
                    for (int qi = 0; qi < n_ref_queries; qi++) {
                        float w = hc.attn_weights[qi * T + j];
                        if (w > max_w) max_w = w;
                    }
                    if (max_w > global_importance[j]) {
                        global_importance[j] = max_w;
                    }
                }
            }

            if ((l + 1) % 8 == 0 || l + 1 == sd.n_layer) {
                LOG_INF("  Scored %u / %u layers\n", l + 1, sd.n_layer);
            }
        }
    }

    // ---- Select top-t globally ----
    {
        std::vector<int> indices(T);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                          [&](int a, int b) { return global_importance[a] > global_importance[b]; });

        shared_selected.assign(indices.begin(), indices.begin() + t);
        std::sort(shared_selected.begin(), shared_selected.end());
    }

    LOG_INF("Selected %d / %d positions globally\n", t, T);

    // ---- Layer-wise budget allocation (optional) ----
    std::vector<int> layer_budgets(sd.n_layer, t);  // default: same budget for all layers
    if (use_optimized && enable_layer_budget) {
        LOG_INF("Using layer-wise budget allocation...\n");
        auto sensitivities = kvcompact::optimized::LayerWiseBudgetAllocator::get_default_sensitivities(sd.n_layer);
        layer_budgets = kvcompact::optimized::LayerWiseBudgetAllocator::allocate_budgets(t, sensitivities, sd.n_layer);

        // Clamp budgets to shared_selected size (can't exceed global selection)
        for (uint32_t l = 0; l < sd.n_layer; l++) {
            layer_budgets[l] = std::min(layer_budgets[l], t);
            layer_budgets[l] = std::max(layer_budgets[l], 1);
        }

        LOG_INF("  Budget range: %d - %d (global t=%d)\n",
                *std::min_element(layer_budgets.begin(), layer_budgets.end()),
                *std::max_element(layer_budgets.begin(), layer_budgets.end()), t);
    }

    // ---- Step 2+3: Per-layer, per-head NNLS (beta) and least-squares (C_v) ----
    std::vector<std::vector<std::vector<float>>> beta_all(sd.n_layer);
    int total_nnls_iters = 0;
    int total_nnls_calls = 0;

    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];
        cv_all[l].resize(n_head_kv);
        beta_all[l].resize(n_head_kv);

        // Layer budget: use fewer tokens for less sensitive layers
        const int t_layer = layer_budgets[l];

        for (int h = 0; h < n_head_kv; h++) {
            const auto & hc = lh_cache[l][h];

            auto & beta = beta_all[l][h];
            auto & cv   = cv_all[l][h];
            beta.resize(t);   // always t for state consistency
            cv.resize(t * d_v);

            // Step 2: NNLS for beta (using layer budget for NNLS solve)
            std::vector<float> M(n_ref_queries * t_layer);
            for (int qi = 0; qi < n_ref_queries; qi++) {
                for (int j = 0; j < t_layer; j++) {
                    M[qi * t_layer + j] = hc.exp_scores[qi * T + shared_selected[j]];
                }
            }

            std::vector<float> w(t_layer);

            if (use_optimized && enable_early_stop) {
                // Early-stop NNLS: converge faster
                int iters = nnls_solve_early_stop(M.data(), hc.row_sums.data(), w.data(),
                                                   n_ref_queries, t_layer, 200, 1e-4f);
                total_nnls_iters += iters;
            } else {
                nnls_solve(M.data(), hc.row_sums.data(), w.data(), n_ref_queries, t_layer);
                total_nnls_iters += 200;
            }
            total_nnls_calls++;

            for (int j = 0; j < t_layer; j++) {
                beta[j] = logf(std::max(1e-12f, w[j]));
            }
            // Zero out beta for positions beyond layer budget
            for (int j = t_layer; j < t; j++) {
                beta[j] = 0.0f;
            }

            // Step 3: Least squares for C_v
            std::vector<float> X(n_ref_queries * t);
            for (int qi = 0; qi < n_ref_queries; qi++) {
                for (int j = 0; j < t; j++) {
                    X[qi * t + j] = hc.scores[qi * T + shared_selected[j]] + beta[j];
                }
            }
            softmax_rows(X.data(), n_ref_queries, t);

            // Y = original attention output: attn_weights @ V_head  [n_q, d_v]
            std::vector<float> Y(n_ref_queries * d_v, 0.0f);
            for (int qi = 0; qi < n_ref_queries; qi++) {
                for (int ki = 0; ki < T; ki++) {
                    float w_ij = hc.attn_weights[qi * T + ki];
                    const float * v_row = ld.V.data() + ki * n_embd_v_gqa + h * d_v;
                    for (int d = 0; d < d_v; d++) {
                        Y[qi * d_v + d] += w_ij * v_row[d];
                    }
                }
            }

            least_squares_solve(X.data(), Y.data(), cv.data(), n_ref_queries, t, d_v);
        }

        if ((l + 1) % 8 == 0 || l + 1 == sd.n_layer) {
            LOG_INF("  Compacted %u / %u layers\n", l + 1, sd.n_layer);
        }
    }

    if (use_optimized && enable_early_stop && total_nnls_calls > 0) {
        LOG_INF("NNLS early-stop: avg %.1f iters/call (vs 200 max), %d calls\n",
                (float)total_nnls_iters / total_nnls_calls, total_nnls_calls);
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double compact_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    LOG_INF("Compaction took %.1f ms (%.1f ms/layer)\n", compact_ms, compact_ms / sd.n_layer);

    // ============================================================
    // Phase 5: Quality metrics (sample layers/heads)
    // ============================================================
    LOG_INF("\n--- Phase 5: Quality metrics ---\n");

    // Evaluate on 3 representative layers
    int eval_layers[] = { 0, (int)sd.n_layer / 2, (int)sd.n_layer - 1 };
    for (int li = 0; li < 3; li++) {
        int l = eval_layers[li];
        if (l < 0 || l >= (int)sd.n_layer) continue;

        const auto & ld = sd.layers[l];
        const auto & hc = lh_cache[l][0]; // head 0

        // Test query: last K vector, head 0
        const float * q_test = ld.K.data() + (T - 1) * n_embd_k_gqa;

        // Original output
        std::vector<float> orig_scores(T);
        for (int j = 0; j < T; j++) {
            float dot = 0.0f;
            const float * k_row = ld.K.data() + j * n_embd_k_gqa;
            for (int d = 0; d < d_k; d++) dot += q_test[d] * k_row[d];
            orig_scores[j] = dot * inv_sqrt_dk;
        }
        softmax_rows(orig_scores.data(), 1, T);

        std::vector<float> orig_out(d_v, 0.0f);
        for (int j = 0; j < T; j++) {
            const float * v_row = ld.V.data() + j * n_embd_v_gqa;
            for (int d = 0; d < d_v; d++) {
                orig_out[d] += orig_scores[j] * v_row[d];
            }
        }

        // Compacted output (with C_v + beta)
        std::vector<float> comp_scores(t);
        for (int j = 0; j < t; j++) {
            float dot = 0.0f;
            const float * k_row = ld.K.data() + shared_selected[j] * n_embd_k_gqa;
            for (int d = 0; d < d_k; d++) dot += q_test[d] * k_row[d];
            comp_scores[j] = dot * inv_sqrt_dk + beta_all[l][0][j];
        }
        softmax_rows(comp_scores.data(), 1, t);

        std::vector<float> comp_out(d_v, 0.0f);
        for (int j = 0; j < t; j++) {
            for (int d = 0; d < d_v; d++) {
                comp_out[d] += comp_scores[j] * cv_all[l][0][j * d_v + d];
            }
        }

        // Cosine similarity
        float dot_p = 0.0f, n_o = 0.0f, n_c = 0.0f;
        for (int d = 0; d < d_v; d++) {
            dot_p += orig_out[d] * comp_out[d];
            n_o += orig_out[d] * orig_out[d];
            n_c += comp_out[d] * comp_out[d];
        }
        float cos_sim = dot_p / (sqrtf(n_o * n_c) + 1e-8f);
        float rel_err = sqrtf((n_o + n_c - 2 * dot_p) / (n_o + 1e-8f));

        LOG_INF("  Layer %2d head 0: cos_sim=%.6f rel_err=%.6f\n", l, cos_sim, rel_err);
    }

    // ============================================================
    // Phase 6: Write back compacted state and generate
    // ============================================================
    if (do_writeback) {
        LOG_INF("\n--- Phase 6: Write-back and generation ---\n");

        // Build compacted state buffer
        auto compacted_buf = build_compacted_state(
            kv_state, shared_selected, cv_all, n_head_kv, d_k, d_v, n_pos_per_embd);

        LOG_INF("Compacted state: %.2f MB (was %.2f MB, %.1fx smaller)\n",
                compacted_buf.size() / (1024.0 * 1024.0),
                saved / (1024.0 * 1024.0),
                (double)saved / compacted_buf.size());

        // Clear cache and load compacted state
        llama_memory_seq_rm(mem, 0, -1, -1);
        size_t loaded = llama_state_seq_set_data(ctx, compacted_buf.data(), compacted_buf.size(), 0);
        if (loaded == 0) {
            LOG_ERR("Failed to load compacted state!\n");
            LOG_ERR("State buffer size: %zu bytes\n", compacted_buf.size());
        } else {
            LOG_INF("Loaded compacted state: %zu bytes\n", loaded);

            // Compute averaged beta across heads and layers for mask injection
            // beta_avg[j] = mean over all layers and heads of beta_all[l][h][j]
            std::vector<float> beta_avg(t, 0.0f);
            for (uint32_t l = 0; l < sd.n_layer; l++) {
                for (int h = 0; h < n_head_kv; h++) {
                    for (int j = 0; j < t; j++) {
                        beta_avg[j] += beta_all[l][h][j];
                    }
                }
            }
            const float inv_count = 1.0f / (sd.n_layer * n_head_kv);
            float beta_min = 0.0f, beta_max = 0.0f, beta_mean = 0.0f;
            for (int j = 0; j < t; j++) {
                beta_avg[j] *= inv_count;
                beta_mean += beta_avg[j];
                if (beta_avg[j] < beta_min) beta_min = beta_avg[j];
                if (beta_avg[j] > beta_max) beta_max = beta_avg[j];
            }
            beta_mean /= t;
            LOG_INF("Beta bias: mean=%.4f min=%.4f max=%.4f (averaged across %u layers × %d heads)\n",
                    beta_mean, beta_min, beta_max, sd.n_layer, n_head_kv);

            // Inject beta into the attention mask via per-cell bias
            llama_memory_set_attn_bias(mem, 0, beta_avg.data(), t);
            LOG_INF("Set attention bias for %d compacted cells\n", t);

            // Generate with compacted cache
            // Position after the max position in selected cells
            llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
            LOG_INF("Generating from pos %d...\n", (int)pos_max + 1);

            std::string am_output = generate_tokens(ctx, model, vocab, params, pos_max + 1, n_gen);
            LOG_INF("\nAttention Matching output:\n%s\n", am_output.c_str());

            // Summary comparison
            LOG_INF("\n=== Summary ===\n");
            LOG_INF("Compression: %d → %d tokens (%.1fx)\n", n_tokens, t, (float)n_tokens / t);
            LOG_INF("Compaction time: %.1f ms\n", compact_ms);
            LOG_INF("\nFull cache output (first 200 chars):\n  %.200s\n", full_output.c_str());
            if (do_eviction) {
                LOG_INF("\nToken eviction output (first 200 chars):\n  %.200s\n", evict_output.c_str());
            }
            LOG_INF("\nAttention Matching output (first 200 chars):\n  %.200s\n", am_output.c_str());
        }
    } else {
        LOG_INF("\n--- Skipping write-back (--no-writeback) ---\n");
        LOG_INF("To enable: remove --no-writeback flag\n");
    }

    // ---- Cleanup ----
    LOG_INF("\n=== Done ===\n");
    llama_batch_free(batch);
    llama_backend_free();

    return 0;
}
