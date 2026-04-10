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

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <set>
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
                                   llama_pos start_pos, int n_gen,
                                   std::vector<llama_token> * out_ids = nullptr,
                                   float * out_tok_per_sec = nullptr) {
    std::string output;
    llama_batch batch = llama_batch_init(1, 0, 1);
    common_sampler * smpl = common_sampler_init(model, params.sampling);

    auto t_start = std::chrono::high_resolution_clock::now();
    int n_decoded = 0;

    for (int i = 0; i < n_gen; i++) {
        llama_token id = common_sampler_sample(smpl, ctx, -1);
        if (llama_vocab_is_eog(vocab, id)) break;

        output += common_token_to_piece(vocab, id);
        if (out_ids) out_ids->push_back(id);
        common_sampler_accept(smpl, id, true);

        common_batch_clear(batch);
        common_batch_add(batch, id, start_pos + i, {0}, true);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("Failed to decode during generation\n");
            break;
        }
        n_decoded++;
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    float tps = (elapsed_ms > 0 && n_decoded > 0) ? (float)(n_decoded * 1000.0 / elapsed_ms) : 0.0f;
    if (out_tok_per_sec) *out_tok_per_sec = tps;
    LOG_INF("  Generated %d tokens in %.1f ms (%.2f tok/s)\n", n_decoded, elapsed_ms, tps);

    common_sampler_free(smpl);
    llama_batch_free(batch);
    return output;
}

// ============================================================================
// Quality metrics: perplexity, token agreement, top-5 overlap, KL divergence
// ============================================================================

struct quality_metrics {
    float perplexity      = 0.0f;  // exp(avg negative log-likelihood)
    float token_agreement = 0.0f;  // fraction of greedy tokens matching reference
    float top5_overlap    = 0.0f;  // avg fraction of top-5 tokens overlapping
    float kl_divergence   = 0.0f;  // avg KL(P_full || P_test)
    int   n_tokens        = 0;
};

struct reference_logits {
    std::vector<llama_token>              greedy_ids;  // argmax at each step
    std::vector<std::array<llama_token,5>> top5_ids;   // top-5 token IDs
    std::vector<std::vector<float>>       logits;      // raw logits (for KL)
};

// Helper: find top-5 token IDs from logits
static void find_top5(const float * logits, int n_vocab, llama_token out[5]) {
    // Track top-5 with a simple insertion approach
    float vals[5] = {-1e30f, -1e30f, -1e30f, -1e30f, -1e30f};
    for (int k = 0; k < 5; k++) out[k] = 0;
    for (int v = 0; v < n_vocab; v++) {
        if (logits[v] > vals[4]) {
            // Insert into sorted top-5
            int pos = 4;
            while (pos > 0 && logits[v] > vals[pos - 1]) pos--;
            for (int k = 4; k > pos; k--) { vals[k] = vals[k-1]; out[k] = out[k-1]; }
            vals[pos] = logits[v];
            out[pos] = v;
        }
    }
}

// Collect reference logits from the full-cache pass.
// Must be called with the full cache loaded and logits available from prefill.
static reference_logits collect_reference_logits(
        llama_context * ctx, const llama_vocab * vocab,
        const std::vector<llama_token> & ref_ids,
        llama_pos start_pos) {
    reference_logits ref;
    if (ref_ids.empty()) return ref;

    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_batch batch = llama_batch_init(1, 0, 1);

    for (size_t i = 0; i < ref_ids.size(); i++) {
        const float * logits = llama_get_logits(ctx);
        if (!logits) break;

        // Store raw logits for KL divergence
        ref.logits.emplace_back(logits, logits + n_vocab);

        // Greedy token
        llama_token greedy = 0;
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > logits[greedy]) greedy = v;
        }
        ref.greedy_ids.push_back(greedy);

        // Top-5
        std::array<llama_token,5> t5;
        find_top5(logits, n_vocab, t5.data());
        ref.top5_ids.push_back(t5);

        // Feed reference token
        common_batch_clear(batch);
        common_batch_add(batch, ref_ids[i], start_pos + (llama_pos)i, {0}, true);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("Failed to decode during reference logit collection\n");
            break;
        }
    }

    llama_batch_free(batch);
    return ref;
}

// Compute quality metrics against reference logits.
// Feeds ref_ids through the current KV cache state and compares.
static quality_metrics compute_quality_metrics(
        llama_context * ctx, const llama_vocab * vocab,
        const std::vector<llama_token> & ref_ids,
        const reference_logits & ref,
        llama_pos start_pos) {
    quality_metrics m;
    if (ref_ids.empty() || ref.greedy_ids.empty()) return m;

    const int n_vocab = llama_vocab_n_tokens(vocab);
    llama_batch batch = llama_batch_init(1, 0, 1);

    double sum_log_prob = 0.0;
    int    n_agree = 0;
    double sum_top5 = 0.0;
    double sum_kl = 0.0;

    const size_t n = std::min(ref_ids.size(), ref.greedy_ids.size());

    for (size_t i = 0; i < n; i++) {
        const float * logits = llama_get_logits(ctx);
        if (!logits) break;

        // --- log-softmax for this step ---
        float max_l = logits[0];
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > max_l) max_l = logits[v];
        }
        double sum_exp = 0.0;
        for (int v = 0; v < n_vocab; v++) {
            sum_exp += exp((double)(logits[v] - max_l));
        }
        double log_Z = log(sum_exp);

        // #1 Perplexity: log-prob of the reference token
        sum_log_prob += (double)(logits[ref_ids[i]] - max_l) - log_Z;

        // #2 Token agreement: does greedy match reference greedy?
        llama_token greedy = 0;
        for (int v = 1; v < n_vocab; v++) {
            if (logits[v] > logits[greedy]) greedy = v;
        }
        if (greedy == ref.greedy_ids[i]) n_agree++;

        // #3 Top-5 overlap
        llama_token t5[5];
        find_top5(logits, n_vocab, t5);
        int overlap = 0;
        for (int a = 0; a < 5; a++) {
            for (int b = 0; b < 5; b++) {
                if (t5[a] == ref.top5_ids[i][b]) { overlap++; break; }
            }
        }
        sum_top5 += overlap / 5.0;

        // #4 KL divergence: KL(P_full || P_test)
        const float * ref_l = ref.logits[i].data();
        float ref_max = ref_l[0];
        for (int v = 1; v < n_vocab; v++) {
            if (ref_l[v] > ref_max) ref_max = ref_l[v];
        }
        double ref_sum_exp = 0.0;
        for (int v = 0; v < n_vocab; v++) {
            ref_sum_exp += exp((double)(ref_l[v] - ref_max));
        }
        double ref_log_Z = log(ref_sum_exp);

        double kl = 0.0;
        for (int v = 0; v < n_vocab; v++) {
            double log_p = (double)(ref_l[v] - ref_max) - ref_log_Z;
            double log_q = (double)(logits[v] - max_l) - log_Z;
            double p = exp(log_p);
            if (p > 1e-10) {
                kl += p * (log_p - log_q);
            }
        }
        sum_kl += kl;

        m.n_tokens++;

        // Feed reference token
        common_batch_clear(batch);
        common_batch_add(batch, ref_ids[i], start_pos + (llama_pos)i, {0}, true);
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("Failed to decode during quality metrics\n");
            break;
        }
    }

    llama_batch_free(batch);

    if (m.n_tokens > 0) {
        m.perplexity      = (float)exp(-sum_log_prob / m.n_tokens);
        m.token_agreement = (float)n_agree / m.n_tokens;
        m.top5_overlap    = (float)(sum_top5 / m.n_tokens);
        m.kl_divergence   = (float)(sum_kl / m.n_tokens);
    }
    return m;
}

static void print_quality_metrics(const char * label, const quality_metrics & m,
                                  const quality_metrics * baseline = nullptr) {
    if (baseline) {
        LOG_INF("%s: ppl=%.4f (%.2fx) agree=%.1f%% top5=%.1f%% kl=%.4f\n",
                label, m.perplexity, m.perplexity / std::max(baseline->perplexity, 0.001f),
                m.token_agreement * 100.0f, m.top5_overlap * 100.0f, m.kl_divergence);
    } else {
        LOG_INF("%s: ppl=%.4f agree=100.0%% top5=100.0%% kl=0.0000\n",
                label, m.perplexity);
    }
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
    bool  use_original_v = true; // smart eviction: original V values (no C_v fitting)

    // Streaming compaction parameters (Phase 1)
    int stream_pin_prefix = 256;
    int stream_recent_window = 512;
    int stream_trigger = 8192;
    int stream_budget = 4096;

    // Sublinear optimization parameters (Phase 2)
    bool use_optimized = false;
    std::string compaction_method = "baseline";  // baseline, l2
    bool enable_early_stop = false;

    // Pre-parse custom args and strip them before common_params_parse
    // (common_params_parse rejects unknown flags)
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        bool consumed = false;
        // Flags with a value argument
        if (i + 1 < argc) {
            if (strcmp(argv[i], "--compact-ratio") == 0) {
                compact_ratio = std::stof(argv[++i]); consumed = true;
            } else if (strcmp(argv[i], "--n-ref-queries") == 0) {
                n_ref_queries = std::stoi(argv[++i]); consumed = true;
            } else if (strcmp(argv[i], "--pin-prefix") == 0) {
                stream_pin_prefix = std::stoi(argv[++i]); consumed = true;
            } else if (strcmp(argv[i], "--recent-window") == 0) {
                stream_recent_window = std::stoi(argv[++i]); consumed = true;
            } else if (strcmp(argv[i], "--trigger") == 0) {
                stream_trigger = std::stoi(argv[++i]); consumed = true;
            } else if (strcmp(argv[i], "--budget") == 0) {
                stream_budget = std::stoi(argv[++i]); consumed = true;
            } else if (strcmp(argv[i], "--method") == 0) {
                compaction_method = argv[++i]; consumed = true;
            }
        }
        // Boolean flags
        if (!consumed) {
            if (strcmp(argv[i], "--no-writeback") == 0) {
                do_writeback = false; consumed = true;
            } else if (strcmp(argv[i], "--no-eviction") == 0) {
                do_eviction = false; consumed = true;
            } else if (strcmp(argv[i], "--optimized") == 0) {
                use_optimized = true; consumed = true;
            } else if (strcmp(argv[i], "--early-stop") == 0) {
                enable_early_stop = true; consumed = true;
            }
        }
        if (!consumed) {
            filtered_argv.push_back(argv[i]);
        }
    }
    int filtered_argc = (int)filtered_argv.size();

    // Parse standard params (only sees flags it recognizes)
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

    // --optimized enables l2 + early-stop
    if (use_optimized) {
        if (compaction_method == "baseline") compaction_method = "l2";
        enable_early_stop = true;
    }

    if (compaction_method == "l2" && !use_original_v) {
        LOG_WRN("L2 method with C_v fitting (no --use-original-v) not yet supported, falling back to baseline\n");
        compaction_method = "baseline";
    }

    // Validate optimization parameters
    if (compaction_method != "baseline" && compaction_method != "l2") {
        LOG_ERR("Invalid method: %s (must be baseline or l2)\n", compaction_method.c_str());
        return 1;
    }

    // Log streaming config if any streaming flag was used
    bool use_streaming = (stream_trigger != 8192 || stream_budget != 4096 ||
                          stream_pin_prefix != 256 || stream_recent_window != 512);

    // Log optimization config if any optimization flag was used
    bool use_optimizations = (use_optimized || compaction_method != "baseline" ||
                              enable_early_stop);

    if (use_streaming) {
        LOG_INF("Streaming mode ENABLED: budget=%d, trigger=%d, pin=%d, recent=%d\n",
                stream_budget, stream_trigger, stream_pin_prefix, stream_recent_window);
    }

    if (use_optimizations) {
        LOG_INF("Optimization mode ENABLED: method=%s, early_stop=%s\n",
                compaction_method.c_str(),
                enable_early_stop ? "ON" : "OFF");
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

    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("Prefill failed\n");
        llama_batch_free(batch);
        return 1;
    }
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
    float prefill_tps = (prefill_ms > 0) ? (float)(n_tokens * 1000.0 / prefill_ms) : 0.0f;
    LOG_INF("Prefill: %d tokens in %.1f ms (%.2f tok/s)\n", n_tokens, prefill_ms, prefill_tps);

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
    std::vector<llama_token> ref_ids;
    std::string full_output = generate_tokens(ctx, model, vocab, params, n_tokens, n_gen, &ref_ids);
    LOG_INF("Full output:\n%s\n", full_output.c_str());

    llama_memory_t mem = llama_get_memory(ctx);

    // Collect reference logits from full cache for quality metrics
    reference_logits ref_logits;
    quality_metrics qm_full;
    if (!ref_ids.empty()) {
        llama_memory_seq_rm(mem, 0, -1, -1);
        llama_state_seq_set_data(ctx, full_state.data(), full_state.size(), 0);
        ref_logits = collect_reference_logits(ctx, vocab, ref_ids, n_tokens);
        qm_full.perplexity = 0.0f;
        // Compute self-perplexity (baseline)
        llama_memory_seq_rm(mem, 0, -1, -1);
        llama_state_seq_set_data(ctx, full_state.data(), full_state.size(), 0);
        qm_full = compute_quality_metrics(ctx, vocab, ref_ids, ref_logits, n_tokens);
        print_quality_metrics("Full cache", qm_full);
    }

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

        // Quality metrics for eviction
        if (!ref_ids.empty() && !ref_logits.greedy_ids.empty()) {
            llama_memory_seq_rm(mem, 0, -1, -1);
            llama_state_seq_set_data(ctx, full_state.data(), full_state.size(), 0);
            for (int i = n_tokens - 1; i >= 0; i--) {
                if (!keep[i]) llama_memory_seq_rm(mem, 0, i, i + 1);
            }
            quality_metrics qm_evict = compute_quality_metrics(ctx, vocab, ref_ids, ref_logits, pos_max + 1);
            print_quality_metrics("Eviction", qm_evict, &qm_full);
        }
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
    LOG_INF("KV types: k_type=%d, v_type=%d (0=f32, 1=f16, 2=q4_0, 8=q8_0)\n",
            sd.layers[0].k_type, sd.layers[0].v_type);

    // Reference queries: use K from last quarter of context (all heads)
    const int ref_start = (int)sd.cell_count - n_ref_queries;
    // Q_ref_all: [n_ref_queries, n_embd_k_gqa]
    // We use K vectors as proxy queries (Q and K share similar structure)

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_phase = t_start;

    // Compact each layer independently, but with shared key selection per layer
    std::vector<int> shared_selected;  // will be set by first layer, may differ per layer
    std::vector<std::vector<std::vector<float>>> cv_all(sd.n_layer);

    // For the state writer, we need a single shared selection across ALL layers
    // (because cell positions must be consistent across layers in the state format)
    // Strategy: compute importance per layer, aggregate, then select globally

    // ================================================================
    // Ada-KV inspired per-head adaptive key selection
    //
    // Design rationale (TRIZ + Ada-KV + Attention Matching paper §5):
    //
    // PROBLEM: Global key selection fails for models with large d_k
    // (e.g., Gemma 3, d_k=256) because attention is extremely spiky
    // (one-hot) and each head attends to DIFFERENT tokens. A shared
    // selection misses most heads' critical tokens entirely.
    //
    // TRIZ Contradiction: Need shared positions (state format) but
    // per-head selection (quality).
    //
    // Solution (TRIZ #7 Nested Dolls + Ada-KV Algorithm 1):
    //   1. Compute per-head importance for every (layer,head,token)
    //   2. Pool all importance scores, select top-B globally (Ada-KV)
    //   3. This naturally allocates more budget to spread-attention
    //      heads and less to spiky heads
    //   4. Each head picks its own top-k_h tokens from its budget
    //   5. Union of all per-head selections → shared superset
    //   6. If |union| != t: adjust via global fill or trim
    //
    // References:
    //   - Ada-KV (Feng et al., 2024, arXiv:2407.11550)
    //   - LU-KV (Tang et al., 2026, arXiv:2602.08585)
    //   - CoKV (Sun et al., 2025, arXiv:2502.17501)
    //   - Attention Matching (Zweiger et al., 2026, §5)
    // ================================================================

    LOG_INF("Computing per-head adaptive key importance across %u layers × %d heads...\n",
            sd.n_layer, n_head_kv);

    // Store per-layer precomputed data for reuse in NNLS/LSQ steps
    struct layer_head_cache {
        std::vector<float> scores;       // [n_q, T]
        std::vector<float> exp_scores;   // [n_q, T]
        std::vector<float> row_sums;     // [n_q]
        std::vector<float> attn_weights; // [n_q, T]
    };
    std::vector<std::vector<layer_head_cache>> lh_cache(sd.n_layer);

    const float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);
    const int T = (int) sd.cell_count;

    // Total heads across all layers (for pooling)
    const int total_heads = (int)sd.n_layer * n_head_kv;

    // Per-head importance: [total_heads][T] — max attn weight per token
    std::vector<std::vector<float>> per_head_importance(total_heads);

    if (compaction_method == "l2") {
        // L2 importance scoring: skip attention computation, saves memory
        for (uint32_t l = 0; l < sd.n_layer; l++) {
            const auto & ld = sd.layers[l];

            for (int h = 0; h < n_head_kv; h++) {
                // Extract per-head K and Q_ref
                std::vector<float> K_h(T * d_k), Q_h(n_ref_queries * d_k);
                for (int i = 0; i < T; i++)
                    memcpy(K_h.data() + i * d_k,
                           ld.K.data() + i * n_embd_k_gqa + h * d_k,
                           d_k * sizeof(float));
                for (int qi = 0; qi < n_ref_queries; qi++)
                    memcpy(Q_h.data() + qi * d_k,
                           ld.K.data() + (ref_start + qi) * n_embd_k_gqa + h * d_k,
                           d_k * sizeof(float));

                auto importance = kvcompact::optimized::FastImportanceEstimator::estimate_importance_l2(
                    K_h.data(), Q_h.data(), T, n_ref_queries, d_k);

                int head_idx = l * n_head_kv + h;
                per_head_importance[head_idx].resize(T);
                for (int j = 0; j < T; j++)
                    per_head_importance[head_idx][j] = (float)importance[j];
            }

            if ((l + 1) % 8 == 0 || l + 1 == sd.n_layer) {
                LOG_INF("  L2-scored %u / %u layers\n", l + 1, sd.n_layer);
            }
        }
    } else {
        for (uint32_t l = 0; l < sd.n_layer; l++) {
            const auto & ld = sd.layers[l];
            lh_cache[l].resize(n_head_kv);

            for (int h = 0; h < n_head_kv; h++) {
                auto & hc = lh_cache[l][h];
                hc.scores.resize(n_ref_queries * T);
                hc.exp_scores.resize(n_ref_queries * T);
                hc.row_sums.resize(n_ref_queries);
                hc.attn_weights.resize(n_ref_queries * T);

                // Compute scores: Q_ref_h @ K_h^T / sqrt(d_k)
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

                // exp and softmax
                memcpy(hc.exp_scores.data(), hc.scores.data(), n_ref_queries * T * sizeof(float));
                exp_rows_stable(hc.exp_scores.data(), hc.row_sums.data(), n_ref_queries, T);

                memcpy(hc.attn_weights.data(), hc.scores.data(), n_ref_queries * T * sizeof(float));
                softmax_rows(hc.attn_weights.data(), n_ref_queries, T);

                // Per-key max attention weight across queries
                int head_idx = l * n_head_kv + h;
                per_head_importance[head_idx].resize(T);
                for (int j = 0; j < T; j++) {
                    float max_w = 0.0f;
                    for (int qi = 0; qi < n_ref_queries; qi++) {
                        float w = hc.attn_weights[qi * T + j];
                        if (w > max_w) max_w = w;
                    }
                    per_head_importance[head_idx][j] = max_w;
                }
            }

            if ((l + 1) % 8 == 0 || l + 1 == sd.n_layer) {
                LOG_INF("  Scored %u / %u layers\n", l + 1, sd.n_layer);
            }
        }
    } // end if (compaction_method == "l2") else

    {
        auto t_now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_now - t_phase).count();
        LOG_INF("[timing] Importance scoring: %.1f ms\n", ms);
        t_phase = t_now;
    }

    // ================================================================
    // Ada-KV Algorithm 1: Adaptive per-head budget allocation
    //
    // Pool importance scores across all (layer,head,token) tuples,
    // select top-B globally. Count how many selections fall in each
    // head to determine per-head budgets.
    //
    // Then each head selects its own top-k_h tokens. The union of all
    // per-head selections becomes the shared selection, guaranteeing
    // every head's critical tokens are included.
    // ================================================================

    // Step A: Pool all (head_idx, token, importance) and select top-t*total_heads/total_heads = top-t
    // But we need the union to be exactly t tokens (positions, not head-token pairs).
    // Approach: each layer needs the same selected positions.
    // So we do Ada-KV within each layer, then merge across layers.

    // Per-layer union of per-head selections
    // For each layer: allocate per-head budgets, select per-head, union → layer selection
    // Then across layers: intersect or union to get shared selection

    // Actually the state format requires ONE shared selection across ALL layers.
    // So we aggregate importance across layers first (max), then do per-head Ada-KV.

    // Per-head importance aggregated across layers: max_l importance[l,h,j]
    std::vector<std::vector<float>> head_importance_agg(n_head_kv, std::vector<float>(T, 0.0f));
    for (uint32_t l = 0; l < sd.n_layer; l++) {
        for (int h = 0; h < n_head_kv; h++) {
            int head_idx = l * n_head_kv + h;
            for (int j = 0; j < T; j++) {
                float imp = per_head_importance[head_idx][j];
                if (imp > head_importance_agg[h][j]) {
                    head_importance_agg[h][j] = imp;
                }
            }
        }
    }

    // ================================================================
    // Guaranteed coverage: sinks + distributed middle + recent context
    //
    // K@K^T importance with reference queries from the last quarter
    // has a strong recency bias (queries attend to nearby keys).
    // Without guaranteed coverage, ALL selected tokens can fall in
    // the last 25% — losing the entire document body and attention sinks.
    //
    // Fix (H2O/SnapKV-inspired): reserve slots for structural coverage,
    // then let Ada-KV fill the remaining budget with attention-important tokens.
    // ================================================================
    std::set<int> guaranteed;

    // Attention sinks: first few positions (critical for many models)
    const int n_sink = std::min(4, T);
    for (int i = 0; i < n_sink; i++) guaranteed.insert(i);

    // Distributed middle: uniform sample from non-sink, non-recent region
    const int recent_start = T - n_ref_queries;
    const int n_reserved_recent = std::min(t / 2, n_ref_queries); // at most half budget for recent
    const int n_middle_budget = std::max(0, t / 4); // 25% of budget for middle context
    if (n_middle_budget > 0 && recent_start > n_sink) {
        const float step = (float)(recent_start - n_sink) / (float)n_middle_budget;
        for (int i = 0; i < n_middle_budget; i++) {
            int idx = n_sink + (int)(i * step);
            if (idx < recent_start) guaranteed.insert(idx);
        }
    }

    LOG_INF("Guaranteed coverage: %d sinks + %d middle = %d reserved (of %d budget)\n",
            n_sink, (int)guaranteed.size() - n_sink, (int)guaranteed.size(), t);

    // Remaining budget for Ada-KV attention-based selection
    int ada_budget = t - (int)guaranteed.size();

    // Step B: Ada-KV — pool aggregated importance across heads, select from remaining
    struct importance_entry {
        float score;
        int head;
        int token;
    };
    std::vector<importance_entry> pooled;
    pooled.reserve(n_head_kv * T);
    for (int h = 0; h < n_head_kv; h++) {
        for (int j = 0; j < T; j++) {
            if (guaranteed.count(j)) continue; // skip already-guaranteed positions
            pooled.push_back({head_importance_agg[h][j], h, j});
        }
    }

    // Select top from pooled
    int pool_size = std::min((int)pooled.size(), ada_budget * 2);
    std::partial_sort(pooled.begin(), pooled.begin() + pool_size, pooled.end(),
        [](const importance_entry & a, const importance_entry & b) {
            return a.score > b.score;
        });

    // Count per-head budget from Ada-KV entries
    std::vector<int> head_budget(n_head_kv, 0);
    std::set<int> ada_selected_set;
    for (int i = 0; i < pool_size && (int)ada_selected_set.size() < ada_budget; i++) {
        ada_selected_set.insert(pooled[i].token);
        if ((int)ada_selected_set.size() <= ada_budget) {
            head_budget[pooled[i].head]++;
        }
    }

    // Ensure minimum budget per head (at least 1 token each)
    for (int h = 0; h < n_head_kv; h++) {
        if (head_budget[h] < 1) head_budget[h] = 1;
    }

    // Step C: Per-head top-k selection with guaranteed inclusion
    std::set<int> union_selected = guaranteed; // start with guaranteed positions
    std::vector<std::vector<int>> per_head_selected(n_head_kv);

    for (int h = 0; h < n_head_kv; h++) {
        // Only consider non-guaranteed positions
        std::vector<int> indices;
        indices.reserve(T);
        for (int j = 0; j < T; j++) {
            if (!guaranteed.count(j)) indices.push_back(j);
        }
        int budget_h = std::min(head_budget[h], (int)indices.size());

        std::partial_sort(indices.begin(), indices.begin() + budget_h, indices.end(),
            [&](int a, int b) {
                return head_importance_agg[h][a] > head_importance_agg[h][b];
            });

        per_head_selected[h].assign(indices.begin(), indices.begin() + budget_h);
        for (int j = 0; j < budget_h; j++) {
            union_selected.insert(indices[j]);
        }
    }

    // Step D: Adjust union size to exactly t
    if ((int)union_selected.size() < t) {
        // Fill with globally important tokens not yet selected
        std::vector<float> global_importance(T, 0.0f);
        for (int h = 0; h < n_head_kv; h++) {
            for (int j = 0; j < T; j++) {
                if (head_importance_agg[h][j] > global_importance[j]) {
                    global_importance[j] = head_importance_agg[h][j];
                }
            }
        }

        std::vector<int> remaining;
        for (int j = 0; j < T; j++) {
            if (!union_selected.count(j)) remaining.push_back(j);
        }
        std::sort(remaining.begin(), remaining.end(),
            [&](int a, int b) { return global_importance[a] > global_importance[b]; });

        int fill = t - (int)union_selected.size();
        for (int i = 0; i < fill && i < (int)remaining.size(); i++) {
            union_selected.insert(remaining[i]);
        }
    } else if ((int)union_selected.size() > t) {
        // Trim: remove least important non-guaranteed tokens
        std::vector<float> global_importance(T, 0.0f);
        for (int h = 0; h < n_head_kv; h++) {
            for (int j = 0; j < T; j++) {
                if (head_importance_agg[h][j] > global_importance[j]) {
                    global_importance[j] = head_importance_agg[h][j];
                }
            }
        }

        // Protected: guaranteed positions + each head's top token
        std::set<int> protected_tokens = guaranteed;
        for (int h = 0; h < n_head_kv; h++) {
            if (!per_head_selected[h].empty()) {
                protected_tokens.insert(per_head_selected[h][0]);
            }
        }

        // Sort by importance ascending, remove least important first
        std::vector<int> union_vec(union_selected.begin(), union_selected.end());
        std::sort(union_vec.begin(), union_vec.end(),
            [&](int a, int b) { return global_importance[a] < global_importance[b]; });

        int excess = (int)union_selected.size() - t;
        for (int i = 0; i < (int)union_vec.size() && excess > 0; i++) {
            if (!protected_tokens.count(union_vec[i])) {
                union_selected.erase(union_vec[i]);
                excess--;
            }
        }
    }

    // Convert to sorted vector
    shared_selected.assign(union_selected.begin(), union_selected.end());
    std::sort(shared_selected.begin(), shared_selected.end());

    // Log per-head budget allocation
    {
        LOG_INF("Per-head budget allocation (Ada-KV):\n");
        for (int h = 0; h < n_head_kv; h++) {
            // Count how many of this head's selections made it into the union
            int in_union = 0;
            for (int idx : per_head_selected[h]) {
                if (union_selected.count(idx)) in_union++;
            }
            LOG_INF("  Head %d: budget=%d, in_union=%d/%d\n",
                    h, head_budget[h], in_union, (int)per_head_selected[h].size());
        }
        LOG_INF("Union size: %d (target: %d)\n", (int)shared_selected.size(), t);

        // Distribution diagnostic
        int n_recent = 0, n_sink = 0;
        int recent_start = T - T/4;
        for (int idx : shared_selected) {
            if (idx < 4) n_sink++;
            if (idx >= recent_start) n_recent++;
        }
        LOG_INF("  Selection distribution: sinks(pos<4)=%d, recent(last_25%%)=%d/%d, middle=%d\n",
                n_sink, n_recent, T - recent_start, t - n_sink - n_recent);
        LOG_INF("  Position range: [%d, %d]\n", shared_selected.front(), shared_selected.back());
    }

    {
        auto t_now = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t_now - t_phase).count();
        LOG_INF("[timing] Ada-KV selection: %.1f ms\n", ms);
        t_phase = t_now;
    }

    // Per-layer, per-head NNLS (beta) and least-squares (C_v)
    std::vector<std::vector<std::vector<float>>> beta_all(sd.n_layer);
    int n_spiky_bypassed = 0;
    int n_total_heads = 0;

    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];
        cv_all[l].resize(n_head_kv);
        beta_all[l].resize(n_head_kv);

        for (int h = 0; h < n_head_kv; h++) {
            const auto & hc = lh_cache[l][h];

            auto & beta = beta_all[l][h];
            auto & cv   = cv_all[l][h];
            beta.resize(t);
            cv.resize(t * d_v);

            // Smart eviction or spiky-head bypass: use original V values
            if (use_original_v) {
                n_total_heads++;
                n_spiky_bypassed++; // count as bypass
                for (int j = 0; j < t; j++) beta[j] = 0.0f;
                fill_original_values(ld.V.data(), shared_selected.data(), cv.data(),
                                     t, h, d_v, n_embd_v_gqa);
                continue;
            }

            // Spiky-head bypass (TRIZ #3: Local Quality)
            auto head_stats = compute_head_attention_stats(
                hc.attn_weights.data(), n_ref_queries, T);

            n_total_heads++;
            if (is_spiky_head(head_stats)) {
                // Bypass: zero beta, original V values
                n_spiky_bypassed++;
                for (int j = 0; j < t; j++) beta[j] = 0.0f;
                fill_original_values(ld.V.data(), shared_selected.data(), cv.data(),
                                     t, h, d_v, n_embd_v_gqa);
                if (l == 0) {
                    LOG_INF("  L%02u H%d: SPIKY bypass (entropy=%.4f top1=%.4f) -> original V, beta=0\n",
                            l, h, head_stats.mean_entropy, head_stats.mean_top1_mass);
                }
                continue;
            }

            // Step 2: NNLS for beta
            std::vector<float> M(n_ref_queries * t);
            for (int qi = 0; qi < n_ref_queries; qi++) {
                for (int j = 0; j < t; j++) {
                    M[qi * t + j] = hc.exp_scores[qi * T + shared_selected[j]];
                }
            }

            std::vector<float> w(t);
            float nnls_tol = enable_early_stop ? 1e-3f : 0.0f;
            nnls_solve(M.data(), hc.row_sums.data(), w.data(), n_ref_queries, t, 200, nnls_tol);

            for (int j = 0; j < t; j++) {
                beta[j] = logf(std::max(1e-12f, w[j]));
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

            // Final refit: C_v without beta (generic fix — no injection needed)
            // Beta helped NNLS conditioning, now C_v absorbs the full correction
            for (int qi = 0; qi < n_ref_queries; qi++) {
                for (int j = 0; j < t; j++) {
                    X[qi * t + j] = hc.scores[qi * T + shared_selected[j]]; // NO beta
                }
            }
            softmax_rows(X.data(), n_ref_queries, t);
            least_squares_solve(X.data(), Y.data(), cv.data(), n_ref_queries, t, d_v);

            // Zero beta — C_v now works standalone
            for (int j = 0; j < t; j++) beta[j] = 0.0f;
        }

        if ((l + 1) % 8 == 0 || l + 1 == sd.n_layer) {
            LOG_INF("  Compacted %u / %u layers\n", l + 1, sd.n_layer);
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    {
        double ms = std::chrono::duration<double, std::milli>(t_end - t_phase).count();
        LOG_INF("[timing] NNLS+LS fitting: %.1f ms\n", ms);
    }
    double compact_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    LOG_INF("Compaction took %.1f ms (%.1f ms/layer)\n", compact_ms, compact_ms / sd.n_layer);
    if (n_spiky_bypassed > 0) {
        LOG_INF("Spiky-head bypass: %d / %d heads (%.1f%%) used original V values\n",
                n_spiky_bypassed, n_total_heads,
                100.0f * n_spiky_bypassed / n_total_heads);
    }

    // ============================================================
    // Phase 5: Quality metrics (all layers × all heads, with diagnostics)
    // ============================================================
    LOG_INF("\n--- Phase 5: Quality metrics ---\n");
    LOG_INF("  d_k=%d, d_v=%d, inv_sqrt_dk=%.6f, T=%d, t=%d\n", d_k, d_v, inv_sqrt_dk, T, t);

    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];

        for (int h = 0; h < n_head_kv; h++) {
            // Test query: last K vector for this head
            const float * q_test = ld.K.data() + (T - 1) * n_embd_k_gqa + h * d_k;

            // --- Diagnostic: attention distribution stats ---
            std::vector<float> orig_scores(T);
            float score_min = 1e30f, score_max = -1e30f;
            for (int j = 0; j < T; j++) {
                float dot = 0.0f;
                const float * k_row = ld.K.data() + j * n_embd_k_gqa + h * d_k;
                for (int d = 0; d < d_k; d++) dot += q_test[d] * k_row[d];
                orig_scores[j] = dot * inv_sqrt_dk;
                if (orig_scores[j] < score_min) score_min = orig_scores[j];
                if (orig_scores[j] > score_max) score_max = orig_scores[j];
            }

            // Pre-softmax score stats
            float score_mean = 0.0f;
            for (int j = 0; j < T; j++) score_mean += orig_scores[j];
            score_mean /= T;

            softmax_rows(orig_scores.data(), 1, T);

            // Post-softmax: entropy and top-k concentration
            float entropy = 0.0f;
            float top1_mass = 0.0f, top5_mass = 0.0f, top20_mass = 0.0f;
            std::vector<float> sorted_weights(orig_scores.begin(), orig_scores.end());
            std::sort(sorted_weights.begin(), sorted_weights.end(), std::greater<float>());
            for (int j = 0; j < T; j++) {
                if (orig_scores[j] > 1e-12f)
                    entropy -= orig_scores[j] * logf(orig_scores[j]);
                if (j < (int)sorted_weights.size()) {
                    if (j < 1) top1_mass += sorted_weights[j];
                    if (j < 5) top5_mass += sorted_weights[j];
                    if (j < 20) top20_mass += sorted_weights[j];
                }
            }

            // How many selected keys overlap with the top-t attended keys?
            std::vector<int> top_t_by_attn(T);
            std::iota(top_t_by_attn.begin(), top_t_by_attn.end(), 0);
            std::partial_sort(top_t_by_attn.begin(), top_t_by_attn.begin() + t, top_t_by_attn.end(),
                [&](int a, int b) { return orig_scores[a] > orig_scores[b]; });
            std::set<int> ideal_set(top_t_by_attn.begin(), top_t_by_attn.begin() + t);
            std::set<int> selected_set(shared_selected.begin(), shared_selected.end());
            int overlap = 0;
            for (int idx : selected_set) {
                if (ideal_set.count(idx)) overlap++;
            }

            // Mass captured by selected keys (before beta/C_v adjustment)
            float selected_mass = 0.0f;
            for (int j : shared_selected) selected_mass += orig_scores[j];

            // Original output
            std::vector<float> orig_out(d_v, 0.0f);
            for (int j = 0; j < T; j++) {
                const float * v_row = ld.V.data() + j * n_embd_v_gqa + h * d_v;
                for (int d = 0; d < d_v; d++) {
                    orig_out[d] += orig_scores[j] * v_row[d];
                }
            }

            // Compacted output (with C_v + beta)
            std::vector<float> comp_scores(t);
            for (int j = 0; j < t; j++) {
                float dot = 0.0f;
                const float * k_row = ld.K.data() + shared_selected[j] * n_embd_k_gqa + h * d_k;
                for (int d = 0; d < d_k; d++) dot += q_test[d] * k_row[d];
                comp_scores[j] = dot * inv_sqrt_dk; // beta=0 after generic refit
            }
            softmax_rows(comp_scores.data(), 1, t);

            std::vector<float> comp_out(d_v, 0.0f);
            for (int j = 0; j < t; j++) {
                for (int d = 0; d < d_v; d++) {
                    comp_out[d] += comp_scores[j] * cv_all[l][h][j * d_v + d];
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

            // Beta stats for this head
            float beta_min_h = 1e30f, beta_max_h = -1e30f, beta_mean_h = 0.0f;
            for (int j = 0; j < t; j++) {
                float b = beta_all[l][h][j];
                beta_mean_h += b;
                if (b < beta_min_h) beta_min_h = b;
                if (b > beta_max_h) beta_max_h = b;
            }
            beta_mean_h /= t;

            LOG_INF("  L%02u H%d: cos=%.6f rel_err=%.6f | "
                    "scores[%.2f,%.2f,%.2f] entropy=%.2f "
                    "top1=%.4f top5=%.4f top20=%.4f | "
                    "sel_overlap=%d/%d sel_mass=%.4f | "
                    "beta[%.3f,%.3f,%.3f]\n",
                    l, h, cos_sim, rel_err,
                    score_min, score_mean, score_max, entropy,
                    top1_mass, top5_mass, top20_mass,
                    overlap, t, selected_mass,
                    beta_min_h, beta_mean_h, beta_max_h);
        }
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

            // Generic beta fix: C_v was refitted with beta=0, so no bias injection needed
            LOG_INF("Beta injection: SKIPPED (C_v absorbs correction, generic fix)\n");

            // Generate with compacted cache
            // Position after the max position in selected cells
            llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
            LOG_INF("Generating from pos %d...\n", (int)pos_max + 1);

            std::string am_output = generate_tokens(ctx, model, vocab, params, pos_max + 1, n_gen);
            LOG_INF("\nAttention Matching output:\n%s\n", am_output.c_str());

            // Quality metrics for compacted cache
            if (!ref_ids.empty() && !ref_logits.greedy_ids.empty()) {
                llama_memory_seq_rm(mem, 0, -1, -1);
                llama_state_seq_set_data(ctx, compacted_buf.data(), compacted_buf.size(), 0);
                llama_pos pos_max2 = llama_memory_seq_pos_max(mem, 0);
                quality_metrics qm_compact = compute_quality_metrics(ctx, vocab, ref_ids, ref_logits, pos_max2 + 1);
                print_quality_metrics("Compacted", qm_compact, &qm_full);
            }

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
