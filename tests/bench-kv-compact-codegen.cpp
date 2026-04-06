// Code generation quality benchmark after KV cache compaction at scale
//
// Fills context with real TypeScript code, compacts at various ratios,
// then asks the model to generate new code. Measures tok/s and compares
// outputs between full cache and compacted versions.
//
// Usage:
//   bench-kv-compact-codegen -m <model.gguf> --code-dir <path> [-c <ctx>] [-ngl <n>]
//
// The code-dir should point to a TypeScript project (reads *.ts files recursively).

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "kv-compact-api.h"
#include "kv-compact-math.h"
#include "kv-compact-state.h"

using clock_type = std::chrono::high_resolution_clock;

// ============================================================================
// Code generation instruction (appended after code context)
// ============================================================================

static const char * CODE_GEN_INSTRUCTION =
    "\n\nBased on the codebase above, write a complete TypeScript Fastify service "
    "called 'TaskSchedulingService' that includes:\n"
    "1. TypeScript interfaces for Task (with id, userId, title, description, priority, "
    "status, dueDate, assignedSlotId), TaskAssignment, and SchedulingConflict\n"
    "2. A service class with methods: createTask, assignToTimeSlot, detectConflicts, "
    "suggestOptimalSlots, and getTasksByPriority\n"
    "3. Fastify routes: POST /api/users/:userId/tasks, GET /api/users/:userId/tasks, "
    "POST /api/users/:userId/tasks/:taskId/assign, GET /api/users/:userId/tasks/conflicts\n"
    "Follow the exact same patterns, TypeScript style, error handling, and structure "
    "as the existing code. Use the same DatabaseService pattern.\n\n"
    "Here is the implementation:\n\n";

// ============================================================================
// Helpers
// ============================================================================

static std::string read_file(const std::string & path) {
    std::ifstream f(path);
    if (!f.is_open()) return "";
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Recursively read all .ts files from a directory
static std::string read_all_ts_files(const std::string & dir) {
    std::string result;
    namespace fs = std::filesystem;
    for (const auto & entry : fs::recursive_directory_iterator(dir)) {
        if (entry.is_regular_file() && entry.path().extension() == ".ts") {
            std::string content = read_file(entry.path().string());
            if (!content.empty()) {
                result += "// File: " + entry.path().filename().string() + "\n";
                result += content;
                result += "\n\n";
            }
        }
    }
    return result;
}

// Copy original V values for selected positions across all layers (same as bench-kv-compact-model)
static void copy_v_for_selected(
        const parsed_kv_state::stream_data & sd,
        const std::vector<int> & selected,
        int n_head_kv, int d_v,
        std::vector<std::vector<std::vector<float>>> & cv_all) {

    const int n_layer = (int)sd.n_layer;
    const int t = (int)selected.size();
    const int n_embd_v = n_head_kv * d_v;

    cv_all.resize(n_layer);
    for (int l = 0; l < n_layer; l++) {
        cv_all[l].resize(n_head_kv);
        for (int h = 0; h < n_head_kv; h++) {
            cv_all[l][h].resize(t * d_v);
            for (int j = 0; j < t; j++) {
                const float * src = sd.layers[l].V.data() + selected[j] * n_embd_v + h * d_v;
                memcpy(cv_all[l][h].data() + j * d_v, src, d_v * sizeof(float));
            }
        }
    }
}

// ============================================================================
// Main benchmark
// ============================================================================

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    common_params params;

    // Parse --code-dir from args and strip it before passing to llama.cpp's parser
    std::string code_dir;
    std::vector<char *> filtered_argv;
    filtered_argv.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--code-dir") == 0 && i + 1 < argc) {
            code_dir = argv[++i];
        } else {
            filtered_argv.push_back(argv[i]);
        }
    }
    int filtered_argc = (int)filtered_argv.size();

    if (!common_params_parse(filtered_argc, filtered_argv.data(), params, LLAMA_EXAMPLE_COMPLETION)) {
        fprintf(stderr, "Usage: %s -m <model.gguf> --code-dir <path> [-c <ctx>] [-ngl <n>]\n", argv[0]);
        return 1;
    }

    if (code_dir.empty()) {
        fprintf(stderr, "ERROR: --code-dir is required (path to TypeScript source)\n");
        return 1;
    }

    // Ensure context and batch are large enough
    if (params.n_ctx < 2048) params.n_ctx = 2048;
    if (params.n_batch < (int)params.n_ctx) params.n_batch = params.n_ctx;

    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_context * ctx   = llama_init->context();
    llama_model   * model = llama_init->model();
    assert(ctx && "Failed to create context");

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const enum llama_rope_type rope_type = llama_model_rope_type(model);
    const int n_layer   = llama_model_n_layer(model);
    const int n_head_kv = llama_model_n_head_kv(model);
    const int n_embd    = llama_model_n_embd(model);
    const int n_head    = llama_model_n_head(model);
    const uint32_t n_pos_per_embd = (rope_type == LLAMA_ROPE_TYPE_MROPE ||
                                      rope_type == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;

    char desc_buf[256];
    llama_model_desc(model, desc_buf, sizeof(desc_buf));

    printf("\n=== Code Generation Benchmark ===\n");
    printf("  Model: %s\n", desc_buf);
    printf("  Layers: %d, KV heads: %d, n_embd: %d\n", n_layer, n_head_kv, n_embd);
    printf("  Code dir: %s\n\n", code_dir.c_str());

    llama_memory_t mem = llama_get_memory(ctx);
    const int n_ctx = llama_n_ctx(ctx);

    // Read all TypeScript source files
    std::string code = read_all_ts_files(code_dir);
    if (code.empty()) {
        fprintf(stderr, "ERROR: No .ts files found in %s\n", code_dir.c_str());
        llama_backend_free();
        return 1;
    }
    printf("  Read %zu bytes of TypeScript source\n\n", code.size());

    // Config
    int ctx_sizes[] = {50000, 100000, 200000};
    float comp_ratios[] = {1.0f, 0.2f, 0.1f, 0.05f, 0.02f};  // full, 5x, 10x, 20x, 50x
    const int n_gen = 200;

    // Pre-tokenize the instruction
    std::vector<llama_token> instr_tokens = common_tokenize(vocab, CODE_GEN_INSTRUCTION, false, false);
    int n_instr = (int)instr_tokens.size();

    for (int target_ctx : ctx_sizes) {
        if (target_ctx >= n_ctx) {
            printf("  --- ctx=%d: exceeds -c %d, skipping ---\n\n", target_ctx, n_ctx);
            continue;
        }

        int budget = target_ctx - 64 - n_instr;
        if (budget <= 0) continue;

        // Build the code portion by repeating until we fill the budget
        std::string code_prompt;
        while ((int)code_prompt.size() < budget * 3) {
            code_prompt += code;
        }

        std::vector<llama_token> code_tokens = common_tokenize(vocab, code_prompt, true, false);
        if ((int)code_tokens.size() > budget) {
            code_tokens.resize(budget);
        }

        // Full prompt = code tokens + instruction tokens
        std::vector<llama_token> all_tokens = code_tokens;
        all_tokens.insert(all_tokens.end(), instr_tokens.begin(), instr_tokens.end());
        int n_tok = (int)all_tokens.size();

        printf("  === Context: %d tokens (code=%d, instruction=%d) ===\n\n",
               n_tok, (int)code_tokens.size(), n_instr);

        // ---- Prefill and save state ----
        llama_memory_seq_rm(mem, 0, -1, -1);
        llama_batch pb = llama_batch_init(n_tok, 0, 1);
        for (int i = 0; i < n_tok; i++)
            common_batch_add(pb, all_tokens[i], i, {0}, (i == n_tok - 1));
        int rc = llama_decode(ctx, pb);
        llama_batch_free(pb);
        if (rc != 0) {
            printf("  PREFILL FAILED (rc=%d)\n\n", rc);
            continue;
        }

        // Save full state
        size_t ss = llama_state_seq_get_size(ctx, 0);
        std::vector<uint8_t> sb(ss);
        size_t saved_bytes = llama_state_seq_get_data(ctx, sb.data(), sb.size(), 0);
        double full_mb = saved_bytes / (1024.0 * 1024.0);

        // Parse state
        parsed_kv_state ks;
        ks.parse(sb.data(), saved_bytes, n_pos_per_embd);
        const auto & sd = ks.streams[0];
        int adk = sd.layers[0].n_embd_k_gqa() / n_head_kv;
        int adv = sd.layers[0].n_embd_v_gqa_computed() / n_head_kv;

        printf("  State: %.2f MB for %d tokens\n\n", full_mb, n_tok);

        // ---- Table header ----
        printf("  %-8s  %8s  %9s  %9s  %10s  %10s  %10s  %s\n",
               "ratio", "kept", "full(MB)", "comp(MB)", "compact(ms)", "gen(t/s)", "cum(t/s)", "output");
        printf("  %-8s  %8s  %9s  %9s  %10s  %10s  %10s  %s\n",
               "--------", "--------", "---------", "---------", "----------", "----------", "----------",
               std::string(80, '-').c_str());

        // Store per-ratio results for agent capacity table
        struct ratio_result {
            float ratio;
            double comp_mb;
            double compact_ms;
            double gen_tps;
            double cum_tps;
        };
        std::vector<ratio_result> results;

        // ---- Test each compression ratio ----
        for (float ratio : comp_ratios) {
            // Reload full state each time
            llama_memory_seq_rm(mem, 0, -1, -1);
            llama_state_seq_set_data(ctx, sb.data(), sb.size(), 0);

            size_t comp_size = saved_bytes;
            double compact_ms = 0.0;

            if (ratio < 1.0f) {
                // Compact
                auto t0 = clock_type::now();

                kv_compact_params cp = kv_compact_params_default();
                cp.target_ratio = ratio;
                cp.use_cheap_qref = 1;

                kv_compact_result cr = {};
                rc = kv_compact(sd.layers[0].K.data(), sd.layers[0].V.data(), NULL,
                                n_tok, 0, n_head_kv, adk, adv, &cp, &cr);
                if (rc != 0) {
                    printf("  %-8.0f%%  FAILED (rc=%d)\n", ratio * 100, rc);
                    continue;
                }

                int t = cr.t;
                std::vector<int> sel(cr.selected_indices, cr.selected_indices + t);

                // Copy original V for all layers at selected positions
                std::vector<std::vector<std::vector<float>>> cv_all;
                copy_v_for_selected(sd, sel, n_head_kv, adv, cv_all);

                auto cb = build_compacted_state(ks, sel, cv_all, n_head_kv, adk, adv,
                                                 n_pos_per_embd);
                comp_size = cb.size();
                compact_ms = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();

                // Load compacted state
                llama_memory_seq_rm(mem, 0, -1, -1);
                size_t loaded = llama_state_seq_set_data(ctx, cb.data(), cb.size(), 0);
                if (loaded == 0) {
                    printf("  %-8.0f%%  LOAD FAILED\n", ratio * 100);
                    kv_compact_result_free(&cr);
                    continue;
                }
                kv_compact_result_free(&cr);
            }

            // Generate tokens
            llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
            common_sampler * smpl = common_sampler_init(model, params.sampling);
            llama_batch gb = llama_batch_init(1, 0, 1);
            std::string output;
            int generated = 0;

            auto gen_start = clock_type::now();
            for (int i = 0; i < n_gen; i++) {
                llama_token id = common_sampler_sample(smpl, ctx, -1);
                if (llama_vocab_is_eog(vocab, id)) break;
                output += common_token_to_piece(vocab, id);
                common_sampler_accept(smpl, id, true);
                generated++;
                common_batch_clear(gb);
                common_batch_add(gb, id, pos_max + 1 + i, {0}, true);
                if (llama_decode(ctx, gb) != 0) break;
            }
            double gen_ms = std::chrono::duration<double, std::milli>(clock_type::now() - gen_start).count();
            double gen_tps = (generated > 0) ? generated / (gen_ms / 1000.0) : 0.0;

            common_sampler_free(smpl);
            llama_batch_free(gb);

            double comp_mb = comp_size / (1024.0 * 1024.0);
            int kept = (ratio >= 1.0f) ? n_tok : std::max(1, (int)(n_tok * ratio));

            char ratio_str[32];
            if (ratio >= 1.0f) snprintf(ratio_str, sizeof(ratio_str), "full");
            else snprintf(ratio_str, sizeof(ratio_str), "%.0fx", 1.0/ratio);

            // Cumulative tg/s: effective throughput including compaction overhead
            // This is the rate you'd see if you prefill -> compact -> generate as one pipeline
            double total_time_ms = compact_ms + gen_ms;
            double cum_tps = (generated > 0 && total_time_ms > 0) ? generated / (total_time_ms / 1000.0) : 0.0;

            // Store for agent capacity table
            results.push_back({ratio, comp_mb, compact_ms, gen_tps, cum_tps});

            // Truncate output for display, replace newlines/tabs
            std::string display = output.substr(0, std::min((size_t)80, output.size()));
            for (auto & c : display) { if (c == '\n') c = ' '; if (c == '\t') c = ' '; }

            printf("  %-8s  %5d/%-5d  %8.2f  %8.2f  %10.1f  %10.2f  %10.2f  %.80s\n",
                   ratio_str, kept, n_tok, full_mb, comp_mb, compact_ms, gen_tps, cum_tps, display.c_str());

            // Print full output for detailed comparison
            printf("         Full output (%d tokens):\n", generated);
            size_t pos = 0;
            int line_count = 0;
            while (pos < output.size() && line_count < 40) {
                size_t nl = output.find('\n', pos);
                if (nl == std::string::npos) nl = output.size();
                std::string line = output.substr(pos, nl - pos);
                printf("           %s\n", line.c_str());
                pos = nl + 1;
                line_count++;
            }
            if (pos < output.size()) printf("           ... (%zu more chars)\n", output.size() - pos);
            printf("\n");
        }

        // ---- Agent capacity analysis ----
        // How many concurrent coding agents fit in memory at each compression level
        printf("  --- Agent capacity at %d tok context (128 GB APU, ~18 GB model) ---\n", n_tok);
        printf("  %-8s  %9s  %7s  %12s  %10s  %10s\n",
               "ratio", "mem/agent", "agents", "cum(t/s)*", "gen(t/s)", "cum(t/s)");
        printf("  %-8s  %9s  %7s  %12s  %10s  %10s\n",
               "--------", "---------", "-------", "------------", "----------", "----------");

        // Available memory: 128 GB total - 18 GB model - 2 GB overhead
        const double available_mb = (128.0 - 18.0 - 2.0) * 1024.0;

        for (const auto & rr : results) {
            int n_agents = (int)(available_mb / rr.comp_mb);
            if (n_agents < 1) n_agents = 1;

            // Theoretical cumulative = agents * per-agent gen speed
            // (assumes perfect batching, no contention — upper bound)
            double theoretical_cum = n_agents * rr.gen_tps;

            char ratio_str[32];
            if (rr.ratio >= 1.0f) snprintf(ratio_str, sizeof(ratio_str), "full");
            else snprintf(ratio_str, sizeof(ratio_str), "%.0fx", 1.0/rr.ratio);

            printf("  %-8s  %8.1fMB  %5d   %10.0f  %10.2f  %10.2f\n",
                   ratio_str, rr.comp_mb, n_agents, theoretical_cum,
                   rr.gen_tps, rr.cum_tps);
        }
        printf("  * cumulative = agents x per-agent gen(t/s), theoretical upper bound\n\n");
    }

    llama_backend_free();
    return 0;
}
