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

#include "kv-compact-api.h"
#include "kv-compact-math.h"
#include "kv-compact-state.h"

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
    LOG("  -m  MODEL             path to model file\n");
    LOG("  -p  PROMPT            input context to compact\n");
    LOG("  -f  FILE              read context from file\n");
    LOG("  -c  N                 context size (default: 2048)\n");
    LOG("  --compact-ratio R     compaction ratio (default: 0.2, meaning keep 20%%)\n");
    LOG("  --auto-ratio-mem M    auto-compute ratio from memory budget (MB) and --n-parallel\n");
    LOG("  --n-parallel N        target parallel agents/slots (default: 1, used with --auto-ratio-mem)\n");
    LOG("  --n-ref-queries N     reference queries (default: 0 = last quarter of context)\n");
    LOG("  -n  N                 tokens to generate after compaction (default: 128)\n");
    LOG("  --no-writeback        skip write-back (demo mode: compute quality metrics only)\n");
    LOG("  --no-eviction         skip token eviction baseline\n");
    LOG("\n");
    LOG("Compact ratio tuning guide:\n\n");
    LOG("  The --compact-ratio controls the quality/memory tradeoff.  Lower ratio = more\n");
    LOG("  compression but lower quality.  The --auto-ratio-mem flag auto-computes the\n");
    LOG("  best ratio to fit your memory budget.\n\n");
    LOG("  ratio   compression   quality (cos_sim)   typical use case\n");
    LOG("  -----   -----------   -----------------   ----------------\n");
    LOG("  1.0     none          perfect             no compaction needed\n");
    LOG("  0.50    2x            >0.999              conservative, maximum quality\n");
    LOG("  0.25    4x            >0.97               good balance for most use cases\n");
    LOG("  0.10    10x           >0.90               aggressive, long context on small devices\n");
    LOG("  0.05    20x           ~0.80               extreme, quality loss noticeable\n\n");
    LOG("  On bandwidth-limited hardware (APUs like AMD Strix Halo, integrated GPUs):\n");
    LOG("  - KV cache reads dominate memory bandwidth during token generation\n");
    LOG("  - Smaller KV = less bandwidth per decode step = higher tg/s\n");
    LOG("  - Smaller KV = less memory per agent = more parallel agents\n\n");
    LOG("  Example: AMD Strix Halo, 8GB KV budget, Qwen-7B, 4096 ctx, 8 agents:\n");
    LOG("    %s -m qwen7b.gguf -c 4096 --auto-ratio-mem 8192 --n-parallel 8\n\n", argv[0]);
    LOG("  The tool will auto-compute: KV/seq ≈ 1GB → 8 agents need 8GB → ratio ≈ 1.0 (fits!)\n");
    LOG("  If you want 16 agents: --n-parallel 16 → ratio ≈ 0.50 (2x compression)\n");
    LOG("\n");
}

int main(int argc, char ** argv) {
    common_params params;

    // Custom parameters
    float compact_ratio    = 0.2f;
    float auto_ratio_mem   = 0.0f;  // 0 = disabled; >0 = memory budget in MB
    int   n_parallel       = 1;     // target parallel agents/slots
    int   n_ref_queries    = 0;     // 0 = auto (last quarter)
    bool  do_writeback     = true;
    bool  do_eviction      = true;

    // Parse custom args first and build a filtered argv for llama.cpp
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--compact-ratio") == 0 && i + 1 < argc) {
            compact_ratio = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--auto-ratio-mem") == 0 && i + 1 < argc) {
            auto_ratio_mem = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--n-parallel") == 0 && i + 1 < argc) {
            n_parallel = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--n-ref-queries") == 0 && i + 1 < argc) {
            n_ref_queries = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-writeback") == 0) {
            do_writeback = false;
        } else if (strcmp(argv[i], "--no-eviction") == 0) {
            do_eviction = false;
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }

    // Parse standard params (with custom args stripped)
    int filtered_argc = (int) filtered_argv.size();
    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMPLETION, print_usage)) {
        return 1;
    }

    if (compact_ratio <= 0.0f || compact_ratio >= 1.0f) {
        LOG_ERR("compact-ratio must be between 0 and 1 (exclusive)\n");
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

    // ---- Auto-ratio from memory budget ----
    if (auto_ratio_mem > 0.0f) {
        float suggested = kv_compact_suggest_ratio(model, n_ctx, auto_ratio_mem, n_parallel);
        if (suggested < 0.0f) {
            LOG_ERR("Failed to compute auto ratio (invalid model?)\n");
            return 1;
        }
        if (suggested >= 1.0f) {
            LOG_INF("Auto-ratio: all %d agents fit in %.0f MB budget — no compaction needed\n",
                    n_parallel, auto_ratio_mem);
            LOG_INF("(Overriding compact_ratio to 0.99 for demonstration)\n");
            compact_ratio = 0.99f;
        } else {
            LOG_INF("Auto-ratio: %.0f MB budget / %d parallel / %d ctx → compact_ratio = %.2f (%.1fx compression)\n",
                    auto_ratio_mem, n_parallel, n_ctx, suggested, 1.0f / suggested);
            compact_ratio = suggested;
        }
    }

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
    std::vector<int> shared_selected;
    std::vector<std::vector<std::vector<float>>> cv_all(sd.n_layer);
    std::vector<std::vector<std::vector<float>>> beta_all(sd.n_layer);

    // For the state writer, we need a single shared selection across ALL layers
    // (because cell positions must be consistent across layers in the state format)
    // Strategy: compute importance per layer (streaming), aggregate, then select globally

    LOG_INF("Computing global key importance across %u layers × %d heads...\n",
            sd.n_layer, n_head_kv);

    // Global importance: max across all layers and heads
    const int T = (int) sd.cell_count;
    const float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);
    std::vector<float> global_importance(T, 0.0f);

    // Streaming importance scoring — process one layer at a time, discard after scoring
    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];

        for (int h = 0; h < n_head_kv; h++) {
            // Compute softmax attention weights for this head
            std::vector<float> attn_weights(n_ref_queries * T);
            for (int qi = 0; qi < n_ref_queries; qi++) {
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
            softmax_rows(attn_weights.data(), n_ref_queries, T);

            // Per-key max attention across queries → global importance
            for (int j = 0; j < T; j++) {
                float max_w = 0.0f;
                for (int qi = 0; qi < n_ref_queries; qi++) {
                    float w = attn_weights[qi * T + j];
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

    // Select top-t globally
    {
        std::vector<int> indices(T);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                          [&](int a, int b) { return global_importance[a] > global_importance[b]; });

        shared_selected.assign(indices.begin(), indices.begin() + t);
        std::sort(shared_selected.begin(), shared_selected.end());
    }

    LOG_INF("Selected %d / %d positions globally\n", t, T);

    // Per-layer, per-head NNLS (beta) and least-squares (C_v)
    // C_v is fitted with un-biased softmax so it works at inference time
    // (betas are not stored in the state format)
    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];
        cv_all[l].resize(n_head_kv);
        beta_all[l].resize(n_head_kv);

        for (int h = 0; h < n_head_kv; h++) {
            auto rr = refit_head_values(
                ld.K.data(), ld.V.data(),
                T, n_embd_k_gqa, n_embd_v_gqa,
                h, d_k, d_v,
                n_ref_queries, ref_start,
                shared_selected,
                false /* use_beta_for_cv: false since betas aren't stored */);

            beta_all[l][h] = std::move(rr.beta);
            cv_all[l][h]   = std::move(rr.C_v);
        }

        if ((l + 1) % 8 == 0 || l + 1 == sd.n_layer) {
            LOG_INF("  Compacted %u / %u layers\n", l + 1, sd.n_layer);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double compact_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    LOG_INF("Compaction took %.1f ms (%.1f ms/layer)\n", compact_ms, compact_ms / sd.n_layer);

    // ============================================================
    // Phase 5: Quality metrics (sample layers/heads)
    // ============================================================
    LOG_INF("\n--- Phase 5: Quality metrics ---\n");

    // Evaluate on 3 representative layers — shows quality WITHOUT beta
    // (which is what inference will see after state round-trip)
    int eval_layers[] = { 0, (int)sd.n_layer / 2, (int)sd.n_layer - 1 };
    for (int li = 0; li < 3; li++) {
        int l = eval_layers[li];
        if (l < 0 || l >= (int)sd.n_layer) continue;

        const auto & ld = sd.layers[l];

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

        // Compacted output (C_v without beta — matches inference behavior)
        std::vector<float> comp_scores(t);
        for (int j = 0; j < t; j++) {
            float dot = 0.0f;
            const float * k_row = ld.K.data() + shared_selected[j] * n_embd_k_gqa;
            for (int d = 0; d < d_k; d++) dot += q_test[d] * k_row[d];
            comp_scores[j] = dot * inv_sqrt_dk;
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
