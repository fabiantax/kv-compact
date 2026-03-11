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

#ifdef __linux__
#include <fstream>
#endif

#include "kv-compact-api.h"
#include "kv-compact-math.h"
#include "kv-compact-state.h"

// ============================================================================
// Profiling helpers
// ============================================================================

struct phase_timer {
    using clock = std::chrono::high_resolution_clock;

    struct entry {
        std::string name;
        double      ms;
        size_t      mem_bytes;  // estimated peak memory for this phase
    };

    std::vector<entry>  entries;
    clock::time_point   phase_start;
    std::string         current_name;

    void begin(const std::string & name) {
        current_name = name;
        phase_start  = clock::now();
    }

    void end(size_t mem_bytes = 0) {
        double ms = std::chrono::duration<double, std::milli>(clock::now() - phase_start).count();
        entries.push_back({current_name, ms, mem_bytes});
    }

    void print_summary() const {
        double total_ms = 0.0;
        for (const auto & e : entries) total_ms += e.ms;

        LOG_INF("\n┌─────────────────────────────────┬───────────┬─────────┬────────────┐\n");
        LOG_INF("│ Phase                           │  Time(ms) │     %%   │  Mem (MB)  │\n");
        LOG_INF("├─────────────────────────────────┼───────────┼─────────┼────────────┤\n");
        for (const auto & e : entries) {
            double pct = total_ms > 0 ? (e.ms / total_ms * 100.0) : 0.0;
            if (e.mem_bytes > 0) {
                LOG_INF("│ %-31s │ %9.1f │ %5.1f%%  │ %8.2f   │\n",
                        e.name.c_str(), e.ms, pct, e.mem_bytes / (1024.0 * 1024.0));
            } else {
                LOG_INF("│ %-31s │ %9.1f │ %5.1f%%  │        -   │\n",
                        e.name.c_str(), e.ms, pct);
            }
        }
        LOG_INF("├─────────────────────────────────┼───────────┼─────────┼────────────┤\n");
        LOG_INF("│ TOTAL                           │ %9.1f │ 100.0%%  │            │\n", total_ms);
        LOG_INF("└─────────────────────────────────┴───────────┴─────────┴────────────┘\n");
    }
};

// Read peak RSS from /proc/self/status (Linux only)
static size_t get_peak_rss_bytes() {
#ifdef __linux__
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, 6, "VmHWM:") == 0) {
            // Format: "VmHWM:    12345 kB"
            size_t kb = 0;
            sscanf(line.c_str(), "VmHWM: %zu", &kb);
            return kb * 1024;
        }
    }
#endif
    return 0;
}

// Read current RSS
static size_t get_current_rss_bytes() {
#ifdef __linux__
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.compare(0, 5, "VmRSS") == 0) {
            size_t kb = 0;
            sscanf(line.c_str(), "VmRSS: %zu", &kb);
            return kb * 1024;
        }
    }
#endif
    return 0;
}

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
    LOG("  --sensitivity-budget  weight key selection by per-head sensitivity\n");
    LOG("  --attention-interval N  for hybrid models: attention every Nth layer (e.g., 4 for Qwen 3.5)\n");
    LOG("  --attention-layers L    comma-separated list of attention layer indices\n");
    LOG("\n");
}

int main(int argc, char ** argv) {
    common_params params;

    // Custom parameters
    float compact_ratio = 0.2f;
    int   n_ref_queries = 0;   // 0 = auto (last quarter)
    bool  do_writeback  = true;
    bool  do_eviction   = true;
    bool  sensitivity_budget = false;  // per-head sensitivity-weighted budget
    int   attention_interval = 0;      // hybrid models: attention every N layers (0=all)
    std::vector<int> attention_layers; // explicit list of attention layer indices

    // Parse standard params
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION, print_usage)) {
        return 1;
    }

    // Parse custom args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--compact-ratio") == 0 && i + 1 < argc) {
            compact_ratio = std::stof(argv[++i]);
        } else if (strcmp(argv[i], "--n-ref-queries") == 0 && i + 1 < argc) {
            n_ref_queries = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--no-writeback") == 0) {
            do_writeback = false;
        } else if (strcmp(argv[i], "--no-eviction") == 0) {
            do_eviction = false;
        } else if (strcmp(argv[i], "--sensitivity-budget") == 0) {
            sensitivity_budget = true;
        } else if (strcmp(argv[i], "--attention-interval") == 0 && i + 1 < argc) {
            attention_interval = std::stoi(argv[++i]);
        } else if (strcmp(argv[i], "--attention-layers") == 0 && i + 1 < argc) {
            // Parse comma-separated list: "3,7,11,15,19,23,27,31,35,39"
            std::string list_str = argv[++i];
            size_t pos = 0;
            while (pos < list_str.size()) {
                size_t comma = list_str.find(',', pos);
                if (comma == std::string::npos) comma = list_str.size();
                attention_layers.push_back(std::stoi(list_str.substr(pos, comma - pos)));
                pos = comma + 1;
            }
            std::sort(attention_layers.begin(), attention_layers.end());
        }
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

    // Profiling
    phase_timer prof;
    size_t rss_baseline = get_current_rss_bytes();
    LOG_INF("Baseline RSS: %.2f MB\n", rss_baseline / (1024.0 * 1024.0));

    // ============================================================
    // Phase 1: Prefill
    // ============================================================
    LOG_INF("\n--- Phase 1: Prefill ---\n");
    prof.begin("Prefill + state save");

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
    prof.end(saved);

    // ============================================================
    // Phase 2: Generate with full cache (reference output)
    // ============================================================
    LOG_INF("\n--- Phase 2: Full cache generation (reference) ---\n");
    prof.begin("Reference generation");

    const int n_gen = std::min(params.n_predict > 0 ? params.n_predict : 128, n_ctx - n_tokens);
    std::string full_output = generate_tokens(ctx, model, vocab, params, n_tokens, n_gen);
    LOG_INF("Full output:\n%s\n", full_output.c_str());

    llama_memory_t mem = llama_get_memory(ctx);
    prof.end();

    // ============================================================
    // Phase 3: Token eviction baseline
    // ============================================================
    std::string evict_output;
    if (do_eviction) {
        LOG_INF("\n--- Phase 3: Token eviction baseline ---\n");
        prof.begin("Eviction baseline");

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
        prof.end();
    }

    // ============================================================
    // Phase 4: Attention Matching compaction (all layers × heads)
    // ============================================================
    LOG_INF("\n--- Phase 4: Attention Matching compaction ---\n");

    // Parse the saved state
    prof.begin("State parse");
    parsed_kv_state kv_state;
    if (!kv_state.parse(full_state.data(), saved, n_pos_per_embd)) {
        LOG_ERR("Failed to parse state buffer\n");
        llama_batch_free(batch);
        return 1;
    }
    prof.end(saved);  // parsed state holds ~same bytes as raw

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

    // ---- Set up layer filter for hybrid architectures ----
    kv_layer_filter_fn layer_filter = NULL;
    void * layer_filter_data = NULL;
    kv_layer_list explicit_layer_list = {};

    if (!attention_layers.empty()) {
        // Explicit layer list provided via --attention-layers
        explicit_layer_list.layers = attention_layers.data();
        explicit_layer_list.n_layers = (int)attention_layers.size();
        layer_filter = kv_layer_filter_explicit;
        layer_filter_data = &explicit_layer_list;
        LOG_INF("Layer filter: explicit list of %d attention layers\n",
                explicit_layer_list.n_layers);
    } else if (attention_interval > 0) {
        // Periodic filter via --attention-interval
        layer_filter = kv_layer_filter_periodic;
        layer_filter_data = (void *)(intptr_t)attention_interval;
        int n_attn = kv_compact_count_layers(layer_filter, layer_filter_data, sd.n_layer);
        LOG_INF("Layer filter: every %d layers → %d / %u attention layers\n",
                attention_interval, n_attn, sd.n_layer);
    } else if (kv_state.is_hybrid_model()) {
        // Auto-detect: model has varying K dimensions across layers
        auto compactable = kv_state.get_compactable_layers(0, n_head_kv);
        if ((int)compactable.size() < (int)sd.n_layer) {
            attention_layers = compactable;
            explicit_layer_list.layers = attention_layers.data();
            explicit_layer_list.n_layers = (int)attention_layers.size();
            layer_filter = kv_layer_filter_explicit;
            layer_filter_data = &explicit_layer_list;
            LOG_INF("Layer filter: auto-detected hybrid model, %d / %u compactable layers\n",
                    explicit_layer_list.n_layers, sd.n_layer);
        }
    }

    int n_compact_layers = layer_filter
        ? kv_compact_count_layers(layer_filter, layer_filter_data, sd.n_layer)
        : (int)sd.n_layer;
    LOG_INF("Compacting %d / %u layers (%d skipped)\n",
            n_compact_layers, sd.n_layer, (int)sd.n_layer - n_compact_layers);

    // Reference queries: use K from last quarter of context (all heads)
    const int ref_start = (int)sd.cell_count - n_ref_queries;
    // Q_ref_all: [n_ref_queries, n_embd_k_gqa]
    // We use K vectors as proxy queries (Q and K share similar structure)

    prof.begin("Attention scoring");

    // Compact each layer independently, but with shared key selection per layer
    std::vector<int> shared_selected;  // will be set by first layer, may differ per layer
    std::vector<std::vector<std::vector<float>>> cv_all(sd.n_layer);

    // For the state writer, we need a single shared selection across ALL layers
    // (because cell positions must be consistent across layers in the state format)
    // Strategy: compute importance per layer, aggregate, then select globally

    LOG_INF("Computing global key importance across %u layers × %d heads...\n",
            sd.n_layer, n_head_kv);

    // Global importance: aggregated across all layers and heads
    std::vector<float> global_importance(sd.cell_count, 0.0f);

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

    // For sensitivity-weighted budget: collect per-head importance vectors
    // and sensitivities, then aggregate at the end
    std::vector<std::vector<float>> per_layer_head_imp;  // [n_heads_total][T]
    std::vector<float> all_sensitivities;                // [n_heads_total]
    if (sensitivity_budget) {
        per_layer_head_imp.reserve((size_t)sd.n_layer * n_head_kv);
        all_sensitivities.reserve((size_t)sd.n_layer * n_head_kv);
    }

    int layers_scored = 0;
    for (uint32_t l = 0; l < sd.n_layer; l++) {
        // Skip non-attention layers (hybrid architecture support)
        if (layer_filter && !layer_filter(l, sd.n_layer, layer_filter_data)) {
            continue;
        }

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

            // Per-key max attention across queries
            std::vector<float> head_importance(T);
            for (int j = 0; j < T; j++) {
                float max_w = 0.0f;
                for (int qi = 0; qi < n_ref_queries; qi++) {
                    float w = hc.attn_weights[qi * T + j];
                    if (w > max_w) max_w = w;
                }
                head_importance[j] = max_w;
            }

            if (sensitivity_budget) {
                // Store per-head importance and sensitivity for weighted aggregation
                float sens = compute_head_sensitivity(hc.attn_weights.data(), n_ref_queries, T);
                per_layer_head_imp.push_back(std::move(head_importance));
                all_sensitivities.push_back(sens);
            } else {
                // Original: max across all heads
                for (int j = 0; j < T; j++) {
                    if (head_importance[j] > global_importance[j]) {
                        global_importance[j] = head_importance[j];
                    }
                }
            }
        }

        layers_scored++;
        if (layers_scored % 8 == 0 || l + 1 == sd.n_layer) {
            LOG_INF("  Scored %d / %d attention layers (layer %u / %u)\n",
                    layers_scored, n_compact_layers, l + 1, sd.n_layer);
        }
    }

    // Sensitivity-weighted aggregation
    if (sensitivity_budget) {
        int total_heads = (int)all_sensitivities.size();
        accumulate_weighted_importance(per_layer_head_imp, all_sensitivities,
                                       T, total_heads, global_importance.data());

        // Log sensitivity statistics
        float min_s = all_sensitivities[0], max_s = all_sensitivities[0], sum_s = 0.0f;
        for (float s : all_sensitivities) {
            if (s < min_s) min_s = s;
            if (s > max_s) max_s = s;
            sum_s += s;
        }
        LOG_INF("Head sensitivity: min=%.1f, max=%.1f, mean=%.1f (%.1fx spread)\n",
                min_s, max_s, sum_s / total_heads, max_s / (min_s + 1e-8f));
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

    // Estimate memory for lh_cache: n_layer * n_head_kv * (4 arrays of n_q*T floats + n_q floats)
    {
        size_t cache_bytes = (size_t)sd.n_layer * n_head_kv *
            ((size_t)n_ref_queries * T * 4 + n_ref_queries) * sizeof(float);
        prof.end(cache_bytes);
    }

    // Per-layer, per-head NNLS (beta) and least-squares (C_v)
    prof.begin("NNLS + least-squares");
    std::vector<std::vector<std::vector<float>>> beta_all(sd.n_layer);

    int layers_compacted = 0;
    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];
        cv_all[l].resize(n_head_kv);
        beta_all[l].resize(n_head_kv);

        // Skip non-attention layers: passthrough original V values
        if (layer_filter && !layer_filter(l, sd.n_layer, layer_filter_data)) {
            for (int h = 0; h < n_head_kv; h++) {
                beta_all[l][h].assign(t, 0.0f);
                cv_all[l][h].resize(t * d_v);
                for (int j = 0; j < t; j++) {
                    const float * src = ld.V.data() + shared_selected[j] * n_embd_v_gqa + h * d_v;
                    memcpy(cv_all[l][h].data() + j * d_v, src, d_v * sizeof(float));
                }
            }
            continue;
        }

        for (int h = 0; h < n_head_kv; h++) {
            const auto & hc = lh_cache[l][h];

            auto & beta = beta_all[l][h];
            auto & cv   = cv_all[l][h];
            beta.resize(t);
            cv.resize(t * d_v);

            // Step 2: NNLS for beta
            std::vector<float> M(n_ref_queries * t);
            for (int qi = 0; qi < n_ref_queries; qi++) {
                for (int j = 0; j < t; j++) {
                    M[qi * t + j] = hc.exp_scores[qi * T + shared_selected[j]];
                }
            }

            std::vector<float> w(t);
            nnls_solve(M.data(), hc.row_sums.data(), w.data(), n_ref_queries, t);

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
        }

        layers_compacted++;
        if (layers_compacted % 8 == 0 || l + 1 == sd.n_layer) {
            LOG_INF("  Compacted %d / %d attention layers (layer %u / %u)\n",
                    layers_compacted, n_compact_layers, l + 1, sd.n_layer);
        }
    }

    {
        size_t nnls_mem = (size_t)sd.n_layer * n_head_kv *
            ((size_t)t + (size_t)t * d_v) * sizeof(float);
        prof.end(nnls_mem);  // beta_all + cv_all
    }

    // Compute beta directions for K-modification (per layer, per head)
    prof.begin("Beta direction computation");
    // Direction v satisfies: Q_ref_h @ v ≈ 1, so that
    // q @ (k + beta * sqrt(d_k) * v) / sqrt(d_k) ≈ q @ k / sqrt(d_k) + beta
    std::vector<std::vector<std::vector<float>>> beta_dirs(sd.n_layer);
    for (uint32_t l = 0; l < sd.n_layer; l++) {
        const auto & ld = sd.layers[l];
        beta_dirs[l].resize(n_head_kv);

        // Skip non-attention layers: zero beta direction (beta=0, so direction is irrelevant)
        if (layer_filter && !layer_filter(l, sd.n_layer, layer_filter_data)) {
            for (int h = 0; h < n_head_kv; h++) {
                beta_dirs[l][h].assign(d_k, 0.0f);
            }
            continue;
        }

        for (int h = 0; h < n_head_kv; h++) {
            beta_dirs[l][h].resize(d_k);
            // Extract Q_ref for this head
            std::vector<float> Q_ref_head(n_ref_queries * d_k);
            for (int qi = 0; qi < n_ref_queries; qi++) {
                memcpy(Q_ref_head.data() + qi * d_k,
                       ld.K.data() + (ref_start + qi) * n_embd_k_gqa + h * d_k,
                       d_k * sizeof(float));
            }
            compute_beta_direction(Q_ref_head.data(), n_ref_queries, d_k,
                                   beta_dirs[l][h].data());
        }
    }
    LOG_INF("Computed beta directions for %d / %u layers × %d heads\n",
            n_compact_layers, sd.n_layer, n_head_kv);
    {
        size_t dir_mem = (size_t)sd.n_layer * n_head_kv * d_k * sizeof(float);
        prof.end(dir_mem);
    }

    // ============================================================
    // Phase 5: Quality metrics (sample layers/heads)
    // ============================================================
    LOG_INF("\n--- Phase 5: Quality metrics ---\n");
    prof.begin("Quality metrics");

    // Evaluate on up to 3 representative attention layers
    std::vector<int> candidate_layers;
    if (layer_filter) {
        for (uint32_t l = 0; l < sd.n_layer; l++) {
            if (layer_filter(l, sd.n_layer, layer_filter_data)) {
                candidate_layers.push_back(l);
            }
        }
    } else {
        for (uint32_t l = 0; l < sd.n_layer; l++) {
            candidate_layers.push_back(l);
        }
    }
    // Pick first, middle, last from the attention layers
    int eval_layers[3];
    int n_eval_layers = 0;
    if (!candidate_layers.empty()) {
        eval_layers[n_eval_layers++] = candidate_layers.front();
        if (candidate_layers.size() > 2)
            eval_layers[n_eval_layers++] = candidate_layers[candidate_layers.size() / 2];
        if (candidate_layers.size() > 1)
            eval_layers[n_eval_layers++] = candidate_layers.back();
    }
    for (int li = 0; li < n_eval_layers; li++) {
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
    prof.end();

    // ============================================================
    // Phase 6: Write back compacted state and generate
    // ============================================================
    if (do_writeback) {
        LOG_INF("\n--- Phase 6: Write-back and generation ---\n");

        // Build compacted state buffer (with beta folded into K vectors)
        prof.begin("State build + load");
        auto compacted_buf = build_compacted_state(
            kv_state, shared_selected, cv_all, n_head_kv, d_k, d_v, n_pos_per_embd,
            beta_all, beta_dirs);

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
            prof.end(compacted_buf.size());
        } else {
            LOG_INF("Loaded compacted state: %zu bytes\n", loaded);
            prof.end(compacted_buf.size());

            // Generate with compacted cache
            prof.begin("Compacted generation");
            // Position after the max position in selected cells
            llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
            LOG_INF("Generating from pos %d...\n", (int)pos_max + 1);

            std::string am_output = generate_tokens(ctx, model, vocab, params, pos_max + 1, n_gen);
            LOG_INF("\nAttention Matching output:\n%s\n", am_output.c_str());

            // Summary comparison
            LOG_INF("\n=== Summary ===\n");
            LOG_INF("Compression: %d → %d tokens (%.1fx)\n", n_tokens, t, (float)n_tokens / t);
            LOG_INF("\nFull cache output (first 200 chars):\n  %.200s\n", full_output.c_str());
            if (do_eviction) {
                LOG_INF("\nToken eviction output (first 200 chars):\n  %.200s\n", evict_output.c_str());
            }
            LOG_INF("\nAttention Matching output (first 200 chars):\n  %.200s\n", am_output.c_str());
            prof.end();
        }
    } else {
        LOG_INF("\n--- Skipping write-back (--no-writeback) ---\n");
        LOG_INF("To enable: remove --no-writeback flag\n");
    }

    // ============================================================
    // Profiling summary
    // ============================================================
    prof.print_summary();

    size_t peak_rss = get_peak_rss_bytes();
    size_t curr_rss = get_current_rss_bytes();
    if (peak_rss > 0) {
        LOG_INF("\nMemory: current RSS=%.2f MB, peak RSS=%.2f MB (delta=%.2f MB over baseline)\n",
                curr_rss / (1024.0 * 1024.0),
                peak_rss / (1024.0 * 1024.0),
                (peak_rss - rss_baseline) / (1024.0 * 1024.0));
    }

    // ---- Cleanup ----
    LOG_INF("\n=== Done ===\n");
    llama_batch_free(batch);
    llama_backend_free();

    return 0;
}
