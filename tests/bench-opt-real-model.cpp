// Real-model validation gate for optimization variants
//
// Tests that optimization candidates (expected attention, SnapKV window,
// value-norm filter) don't degrade quality on real model data.
// Uses same infrastructure as bench-kv-compact-model.cpp.
//
// Usage:
//   bench-opt-real-model -m <model.gguf> [-c <ctx_size>] [-ngl <n_gpu_layers>]
//
// Pass criteria:
//   - PPL ratio delta < 2% for any single optimization
//   - PPL ratio delta < 5% for combined optimizations
//   - Needle retrieval: all methods must retrieve "BLUE-FALCON-42" at 50% and 20%

#include "arg.h"
#include "common.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "kv-compact-api.h"
#include "kv-compact-math.h"
#include "kv-compact-state.h"

using clock_type = std::chrono::high_resolution_clock;

// ============================================================================
// Prompts (same as bench-kv-compact-model)
// ============================================================================

static const char * BENCH_PROMPT =
    "The history of artificial intelligence began in antiquity, with myths, stories "
    "and rumors of artificial beings endowed with intelligence or consciousness by "
    "master craftsmen. The seeds of modern AI were planted by philosophers who "
    "attempted to describe the process of human thinking as the mechanical manipulation "
    "of symbols. This work culminated in the invention of the programmable digital "
    "computer in the 1940s, a machine based on the abstract essence of mathematical "
    "reasoning. This device and the ideas behind it inspired a handful of scientists "
    "to begin seriously discussing the possibility of building an electronic brain. "
    "The field of AI research was founded at a workshop held on the campus of "
    "Dartmouth College during the summer of 1956. Those who attended would become "
    "the leaders of AI research for decades. Many of them predicted that a machine "
    "as intelligent as a human being would exist in no more than a generation, and "
    "they were given millions of dollars to make this vision come true. Eventually, "
    "it became obvious that commercial developers and researchers had grossly "
    "underestimated the difficulty of the project. In 1974, in response to the "
    "criticism of Sir James Lighthill and ongoing pressure from the US Congress, "
    "both the US and British governments cut off exploratory research in AI. "
    "The next few years would later be called an AI winter, a period when obtaining "
    "funding for AI projects was difficult. In the early 1980s, AI research was "
    "revived by the commercial success of expert systems, a form of AI program that "
    "simulated the knowledge and analytical skills of human experts. By 1985, "
    "the market for AI had reached over a billion dollars. At the same time, "
    "Japan's fifth generation computer project inspired the US and British "
    "governments to restore funding for academic research. However, beginning "
    "with the collapse of the Lisp Machine market in 1987, AI once again fell "
    "into disrepute, and a second, longer-lasting winter began. Many researchers "
    "began to doubt that the symbolic approach would ever be able to imitate all "
    "the processes of human cognition, especially perception, robotics, learning "
    "and pattern recognition. A number of researchers began to look into sub-symbolic "
    "approaches to specific AI problems. Robotics researchers, such as Rodney Brooks, "
    "rejected symbolic AI and focused on the basic engineering problems that would "
    "allow robots to move, survive, and learn their environment.";

static const char * CONTINUATION =
    "Neural networks research had been abandoned by AI and computer science around "
    "the same time. This line of research was revived by David Rumelhart and others "
    "in the middle of the 1980s. These and other sub-symbolic approaches, such as "
    "fuzzy systems, evolutionary computation and many statistical approaches to AI "
    "were attempted. The field of AI, now more than half a century old, finally "
    "achieved some of its oldest goals. It began to be used successfully throughout "
    "the technology industry, although somewhat behind the scenes.";

static const char * NEEDLE = "The secret code is BLUE-FALCON-42.";
static const char * HAYSTACK_BEFORE =
    "The history of mathematics spans many centuries. Ancient civilizations "
    "developed counting systems and basic arithmetic. The Egyptians used a "
    "base-10 system and the Babylonians used base-60. Greek mathematicians "
    "like Euclid and Archimedes made foundational contributions to geometry. "
    "During the Islamic Golden Age, algebra was formalized by al-Khwarizmi. "
    "The Renaissance brought advances in trigonometry and calculus. Newton "
    "and Leibniz independently developed calculus in the 17th century. "
    "Modern mathematics includes fields like topology, abstract algebra, "
    "and mathematical logic. Set theory, developed by Cantor, provides "
    "the foundation for much of modern mathematics. ";
static const char * HAYSTACK_AFTER =
    "Computer science emerged in the 20th century with contributions from "
    "Turing, Church, and von Neumann. Programming languages evolved from "
    "machine code to assembly to high-level languages like Fortran and C. "
    "The development of the internet transformed global communication. "
    "Database systems organize and retrieve information efficiently. "
    "Operating systems manage hardware resources for applications. "
    "Artificial neural networks were inspired by biological neurons. "
    "Cryptography ensures secure communication over public channels. "
    "Software engineering applies systematic methods to development. "
    "Cloud computing provides on-demand computational resources. ";
static const char * NIAH_QUESTION = " What is the secret code mentioned above?";

// ============================================================================
// Helpers
// ============================================================================

struct model_info {
    llama_context * ctx;
    llama_model   * model;
    const llama_vocab * vocab;
    int n_layer;
    int n_head;
    int n_head_kv;
    int n_embd;
    int d_k;
    int d_v;
    uint32_t n_pos_per_embd;
};

static std::vector<float> get_logits_at(llama_context * ctx, int idx, int n_vocab) {
    const float * logits = llama_get_logits_ith(ctx, idx);
    return std::vector<float>(logits, logits + n_vocab);
}

static void softmax_inplace(std::vector<float> & v) {
    float max_val = *std::max_element(v.begin(), v.end());
    float sum = 0.0f;
    for (auto & x : v) {
        x = expf(x - max_val);
        sum += x;
    }
    for (auto & x : v) x /= sum;
}

static double kl_divergence(const std::vector<float> & p, const std::vector<float> & q) {
    double kl = 0.0;
    for (size_t i = 0; i < p.size(); i++) {
        double pi = p[i] + 1e-12;
        double qi = q[i] + 1e-12;
        kl += pi * log(pi / qi);
    }
    return std::max(0.0, kl);
}

// Copy original V values for selected positions in one layer
static void compact_layer_into(
        const float * V_layer,
        const std::vector<int> & selected,
        int T, int n_head_kv, int d_v,
        std::vector<std::vector<float>> & beta_out,
        std::vector<std::vector<float>> & cv_out) {

    const int t = (int)selected.size();
    const int n_embd_v = n_head_kv * d_v;

    for (int h = 0; h < n_head_kv; h++) {
        for (int j = 0; j < t; j++) {
            beta_out[h][j] = 0.0f;
            const float * src = V_layer + selected[j] * n_embd_v + h * d_v;
            memcpy(cv_out[h].data() + j * d_v, src, d_v * sizeof(float));
        }
    }
}

// Compact all layers with shared key selection
static void compact_all_layers(
        const parsed_kv_state::stream_data & sd,
        const std::vector<int> & selected,
        int n_head_kv, int d_k, int d_v,
        std::vector<std::vector<std::vector<float>>> & cv_all,
        std::vector<std::vector<std::vector<float>>> & beta_all,
        std::vector<std::vector<std::vector<float>>> & dirs_all) {

    const int n_layer = (int)sd.n_layer;
    const int t = (int)selected.size();

    cv_all.resize(n_layer);
    beta_all.resize(n_layer);
    dirs_all.clear();

    for (int l = 0; l < n_layer; l++) {
        cv_all[l].resize(n_head_kv);
        beta_all[l].resize(n_head_kv);
        for (int h = 0; h < n_head_kv; h++) {
            cv_all[l][h].resize(t * d_v);
            beta_all[l][h].resize(t);
        }

        compact_layer_into(sd.layers[l].V.data(), selected,
                          sd.cell_count, n_head_kv, d_v,
                          beta_all[l], cv_all[l]);
    }
}

// ============================================================================
// Optimization variant: run compaction with specific method
// ============================================================================

enum compact_method {
    METHOD_BASELINE,       // Current: use_cheap_qref=1
    METHOD_EXPECTED_ATTN,  // Opt A: pass q_mean as single Q_ref
    METHOD_SNAPKV_W32,     // Opt B: pass last 32 K rows as Q_ref
    METHOD_COMBINED,       // Opt A+B: q_mean from last 32 rows
};

static const char * method_name(compact_method m) {
    switch (m) {
        case METHOD_BASELINE:      return "baseline";
        case METHOD_EXPECTED_ATTN: return "exp_attn";
        case METHOD_SNAPKV_W32:    return "snapkv_w32";
        case METHOD_COMBINED:      return "combined";
    }
    return "unknown";
}

// Run compaction with the specified method, return selected indices
static bool run_compact_method(
        compact_method method,
        const float * K_layer0, const float * V_layer0,
        int T, int n_head_kv, int d_k, int d_v,
        float ratio,
        std::vector<int> & selected_out) {

    int n_embd_k = n_head_kv * d_k;

    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = ratio;
    p.skip_beta = 1;

    kv_compact_result r = {};

    switch (method) {
        case METHOD_BASELINE:
            p.use_cheap_qref = 1;
            if (kv_compact(K_layer0, V_layer0, NULL, T, 0,
                           n_head_kv, d_k, d_v, &p, &r) != 0)
                return false;
            break;

        case METHOD_EXPECTED_ATTN: {
            // Compute q_mean from K rows (K-as-Q, averaged)
            std::vector<float> Q_mean(n_embd_k, 0.0f);
            int n_q_use = std::min(T, 64);
            for (int qi = T - n_q_use; qi < T; qi++)
                for (int h = 0; h < n_head_kv; h++)
                    for (int d = 0; d < d_k; d++)
                        Q_mean[h * d_k + d] += K_layer0[qi * n_embd_k + h * d_k + d];
            for (int j = 0; j < n_embd_k; j++)
                Q_mean[j] /= (float)n_q_use;

            p.use_cheap_qref = 0;
            if (kv_compact(K_layer0, V_layer0, Q_mean.data(), T, 1,
                           n_head_kv, d_k, d_v, &p, &r) != 0)
                return false;
            break;
        }

        case METHOD_SNAPKV_W32: {
            // Use last 32 K rows as Q_ref
            int W = std::min(32, T);
            std::vector<float> Q_window(W * n_embd_k);
            for (int qi = 0; qi < W; qi++) {
                int src = T - W + qi;
                memcpy(Q_window.data() + qi * n_embd_k,
                       K_layer0 + src * n_embd_k,
                       n_embd_k * sizeof(float));
            }

            p.use_cheap_qref = 0;
            if (kv_compact(K_layer0, V_layer0, Q_window.data(), T, W,
                           n_head_kv, d_k, d_v, &p, &r) != 0)
                return false;
            break;
        }

        case METHOD_COMBINED: {
            // q_mean from last 32 K rows only
            int W = std::min(32, T);
            std::vector<float> Q_mean(n_embd_k, 0.0f);
            for (int qi = T - W; qi < T; qi++)
                for (int h = 0; h < n_head_kv; h++)
                    for (int d = 0; d < d_k; d++)
                        Q_mean[h * d_k + d] += K_layer0[qi * n_embd_k + h * d_k + d];
            for (int j = 0; j < n_embd_k; j++)
                Q_mean[j] /= (float)W;

            p.use_cheap_qref = 0;
            if (kv_compact(K_layer0, V_layer0, Q_mean.data(), T, 1,
                           n_head_kv, d_k, d_v, &p, &r) != 0)
                return false;
            break;
        }
    }

    selected_out.assign(r.selected_indices, r.selected_indices + r.t);
    kv_compact_result_free(&r);
    return true;
}

// ============================================================================
// Test 1: PPL and KL comparison across methods
// ============================================================================

struct eval_result {
    double ppl;
    double ppl_ratio;
    double avg_kl;
    double max_kl;
    double top1_pct;
    double top5_pct;
    double compact_ms;
    int t;
};

static bool evaluate_method(
        model_info & mi,
        compact_method method,
        const parsed_kv_state & kv_state,
        const std::vector<uint8_t> & orig_state,
        const std::vector<llama_token> & all_tokens,
        const std::vector<llama_token> & prompt_tokens,
        const std::vector<llama_token> & cont_tokens,
        const std::vector<std::vector<float>> & ref_probs,
        double ref_ppl,
        float ratio,
        eval_result & out) {

    const auto & sd = kv_state.streams[0];
    int T = (int)prompt_tokens.size();
    int n_vocab = llama_vocab_n_tokens(mi.vocab);
    llama_memory_t mem = llama_get_memory(mi.ctx);

    auto t_start = clock_type::now();

    // Run selected method for key selection
    std::vector<int> selected;
    if (sd.layers.empty()) {
        fprintf(stderr, "    ERROR: no KV layers in state\n");
        return false;
    }
    int actual_d_k = sd.layers[0].n_embd_k_gqa() / mi.n_head_kv;
    int actual_d_v = sd.layers[0].n_embd_v_gqa_computed() / mi.n_head_kv;

    if (!run_compact_method(method, sd.layers[0].K.data(), sd.layers[0].V.data(),
                            T, mi.n_head_kv, actual_d_k, actual_d_v,
                            ratio, selected)) {
        return false;
    }

    // Compact all layers
    std::vector<std::vector<std::vector<float>>> cv_all, beta_all, dirs_all;
    compact_all_layers(sd, selected, mi.n_head_kv, actual_d_k, actual_d_v,
                      cv_all, beta_all, dirs_all);

    auto compacted_buf = build_compacted_state(kv_state, selected, cv_all,
                                                mi.n_head_kv, actual_d_k, actual_d_v,
                                                mi.n_pos_per_embd);

    out.compact_ms = std::chrono::duration<double, std::milli>(
        clock_type::now() - t_start).count();
    out.t = (int)selected.size();

    // Load compacted state
    llama_memory_seq_rm(mem, 0, -1, -1);
    size_t loaded = llama_state_seq_set_data(mi.ctx, compacted_buf.data(),
                                              compacted_buf.size(), 0);
    if (loaded == 0) return false;

    // Decode continuation tokens
    llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);
    int n_cont = (int)cont_tokens.size();
    int n_prompt = (int)prompt_tokens.size();

    llama_batch eval_batch = llama_batch_init(n_cont, 0, 1);
    for (int i = 0; i < n_cont; i++) {
        common_batch_add(eval_batch, all_tokens[n_prompt + i], pos_max + 1 + i, {0}, true);
    }

    int rc = llama_decode(mi.ctx, eval_batch);
    llama_batch_free(eval_batch);
    if (rc != 0) return false;

    // Evaluate
    double comp_log_prob = 0.0;
    double sum_kl = 0.0, max_kl = 0.0;
    int top1_match = 0, top5_match = 0;
    int n_eval = n_cont - 1;

    for (int i = 0; i < n_eval; i++) {
        std::vector<float> comp_probs = get_logits_at(mi.ctx, i, n_vocab);
        softmax_inplace(comp_probs);

        llama_token target = all_tokens[n_prompt + i + 1];
        comp_log_prob += log(comp_probs[target] + 1e-12);

        if (i + 1 < n_cont) {
            const auto & ref = ref_probs[i + 1];
            double kl = kl_divergence(ref, comp_probs);
            sum_kl += kl;
            if (kl > max_kl) max_kl = kl;

            auto argmax_fn = [](const std::vector<float> & v) {
                return (int)std::distance(v.begin(), std::max_element(v.begin(), v.end()));
            };
            if (argmax_fn(ref) == argmax_fn(comp_probs)) top1_match++;

            std::vector<int> top5_idx(n_vocab);
            std::iota(top5_idx.begin(), top5_idx.end(), 0);
            std::partial_sort(top5_idx.begin(), top5_idx.begin() + 5, top5_idx.end(),
                [&comp_probs](int a, int b) { return comp_probs[a] > comp_probs[b]; });
            int ref_top = argmax_fn(ref);
            for (int k = 0; k < 5; k++) {
                if (top5_idx[k] == ref_top) { top5_match++; break; }
            }
        }
    }

    out.ppl = exp(-comp_log_prob / n_eval);
    out.ppl_ratio = out.ppl / ref_ppl;
    out.avg_kl = sum_kl / n_eval;
    out.max_kl = max_kl;
    out.top1_pct = 100.0 * top1_match / n_eval;
    out.top5_pct = 100.0 * top5_match / n_eval;

    return true;
}

static void test_ppl_quality(model_info & mi, common_params & params) {
    printf("\n=== Test 1: PPL Quality Across Optimization Variants ===\n");
    char desc_buf[256];
    llama_model_desc(mi.model, desc_buf, sizeof(desc_buf));
    printf("  Model: %s\n", desc_buf);
    printf("  Layers: %d, KV heads: %d, d_k: %d\n\n", mi.n_layer, mi.n_head_kv, mi.d_k);

    llama_memory_t mem = llama_get_memory(mi.ctx);

    // Tokenize
    std::vector<llama_token> prompt_tokens = common_tokenize(mi.vocab, BENCH_PROMPT, true, false);
    std::vector<llama_token> cont_tokens = common_tokenize(mi.vocab, CONTINUATION, false, false);
    int n_prompt = (int)prompt_tokens.size();
    int n_cont = (int)cont_tokens.size();
    int n_vocab = llama_vocab_n_tokens(mi.vocab);

    std::vector<llama_token> all_tokens;
    all_tokens.insert(all_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
    all_tokens.insert(all_tokens.end(), cont_tokens.begin(), cont_tokens.end());
    int n_all = (int)all_tokens.size();

    // Full prefill for reference logits
    printf("  [1/3] Full prefill (%d tokens)...\n", n_all);
    llama_memory_seq_rm(mem, 0, -1, -1);
    llama_batch batch = llama_batch_init(n_all, 0, 1);
    for (int i = 0; i < n_all; i++) {
        bool need_logits = (i >= n_prompt - 1);
        common_batch_add(batch, all_tokens[i], i, {0}, need_logits);
    }
    int rc = llama_decode(mi.ctx, batch);
    if (rc != 0) {
        printf("  ERROR: prefill failed (rc=%d)\n", rc);
        llama_batch_free(batch);
        return;
    }
    llama_batch_free(batch);

    std::vector<std::vector<float>> ref_probs(n_cont);
    double ref_log_prob = 0.0;
    for (int i = 0; i < n_cont; i++) {
        ref_probs[i] = get_logits_at(mi.ctx, n_prompt - 1 + i, n_vocab);
        softmax_inplace(ref_probs[i]);
        ref_log_prob += log(ref_probs[i][all_tokens[n_prompt + i]] + 1e-12);
    }
    double ref_ppl = exp(-ref_log_prob / n_cont);
    printf("  Reference PPL: %.4f\n\n", ref_ppl);

    // Re-prefill prompt only for compaction
    llama_memory_seq_rm(mem, 0, -1, -1);
    llama_batch pb = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; i++)
        common_batch_add(pb, prompt_tokens[i], i, {0}, (i == n_prompt - 1));
    llama_decode(mi.ctx, pb);
    llama_batch_free(pb);

    // Save and parse state
    size_t state_size = llama_state_seq_get_size(mi.ctx, 0);
    std::vector<uint8_t> state_buf(state_size);
    size_t saved = llama_state_seq_get_data(mi.ctx, state_buf.data(), state_buf.size(), 0);
    assert(saved > 0);

    parsed_kv_state kv_state;
    bool parse_ok = kv_state.parse(state_buf.data(), saved, mi.n_pos_per_embd);
    assert(parse_ok);
    assert(!kv_state.streams.empty() && "No KV streams parsed from state!");

    printf("  State: %.2f MB for %d tokens\n\n", saved / (1024.0 * 1024.0), n_prompt);

    // Test all methods at two ratios
    float test_ratios[] = {0.5f, 0.2f};
    compact_method methods[] = {METHOD_BASELINE, METHOD_EXPECTED_ATTN, METHOD_SNAPKV_W32, METHOD_COMBINED};

    printf("  %-8s  %-12s  %5s  %12s  %12s  %12s  %12s  %7s  %7s  %10s\n",
           "ratio", "method", "t/T", "ppl", "ppl_ratio", "avg_KL", "max_KL", "top1%", "top5%", "time_ms");
    printf("  %-8s  %-12s  %5s  %12s  %12s  %12s  %12s  %7s  %7s  %10s\n",
           "------", "--------", "-----", "----------", "----------", "----------", "----------",
           "-----", "-----", "--------");

    int total_pass = 0, total_tests = 0;

    for (float ratio : test_ratios) {
        // Get baseline first for delta comparison
        eval_result baseline_res = {};
        bool baseline_ok = evaluate_method(mi, METHOD_BASELINE, kv_state, state_buf,
                                           all_tokens, prompt_tokens, cont_tokens,
                                           ref_probs, ref_ppl, ratio, baseline_res);

        if (baseline_ok) {
            printf("  %-8.0f%%  %-12s  %3d/%-3d  %12.4f  %12.4f  %12.6f  %12.6f  %6.1f%%  %6.1f%%  %10.1f\n",
                   ratio * 100, method_name(METHOD_BASELINE), baseline_res.t, n_prompt,
                   baseline_res.ppl, baseline_res.ppl_ratio, baseline_res.avg_kl,
                   baseline_res.max_kl, baseline_res.top1_pct, baseline_res.top5_pct,
                   baseline_res.compact_ms);
        }

        for (int m = 1; m < 4; m++) {
            eval_result res = {};
            bool ok = evaluate_method(mi, methods[m], kv_state, state_buf,
                                      all_tokens, prompt_tokens, cont_tokens,
                                      ref_probs, ref_ppl, ratio, res);

            total_tests++;
            if (!ok) {
                printf("  %-8.0f%%  %-12s  FAILED\n", ratio * 100, method_name(methods[m]));
                continue;
            }

            // Compare against reference PPL (1.0 = no degradation from compaction)
            double ppl_delta_pct = fabs(res.ppl_ratio - 1.0) * 100.0;

            const char * status = "OK";
            if (ppl_delta_pct > 2.0f) status = "WARN>2%";

            if (ppl_delta_pct <= 2.0f) total_pass++;

            printf("  %-8.0f%%  %-12s  %3d/%-3d  %12.4f  %12.4f  %12.6f  %12.6f  %6.1f%%  %6.1f%%  %10.1f  %s dPPL=%.2f%%\n",
                   ratio * 100, method_name(methods[m]), res.t, n_prompt,
                   res.ppl, res.ppl_ratio, res.avg_kl, res.max_kl,
                   res.top1_pct, res.top5_pct, res.compact_ms, status, ppl_delta_pct);
        }
        printf("\n");
    }

    printf("  PPL quality gate: %d/%d passed (dPPL < 2%%)\n\n", total_pass, total_tests);
}

// ============================================================================
// Test 2: Needle-in-a-Haystack retrieval
// ============================================================================

static void test_needle_retrieval(model_info & mi, common_params & params) {
    printf("\n=== Test 2: Needle-in-a-Haystack Retrieval ===\n\n");

    llama_memory_t mem = llama_get_memory(mi.ctx);

    std::string niah_prompt = std::string(HAYSTACK_BEFORE) + NEEDLE + " " +
                               std::string(HAYSTACK_AFTER) + NIAH_QUESTION;

    std::vector<llama_token> niah_tokens = common_tokenize(mi.vocab, niah_prompt, true, false);
    int n_niah = (int)niah_tokens.size();

    if (n_niah + 64 > (int)llama_n_ctx(mi.ctx)) {
        printf("  SKIP: NIAH prompt (%d tokens) too large for context\n", n_niah);
        return;
    }

    // Full-cache baseline
    llama_memory_seq_rm(mem, 0, -1, -1);
    llama_batch nb = llama_batch_init(n_niah, 0, 1);
    for (int i = 0; i < n_niah; i++)
        common_batch_add(nb, niah_tokens[i], i, {0}, (i == n_niah - 1));
    llama_decode(mi.ctx, nb);
    llama_batch_free(nb);

    // Generate baseline answer — allow enough tokens for thinking model
    auto generate_answer = [&](llama_pos start_pos, int max_len) -> std::string {
        std::string answer;
        common_sampler * smpl = common_sampler_init(mi.model, params.sampling);
        llama_batch gb = llama_batch_init(1, 0, 1);
        for (int i = 0; i < max_len; i++) {
            llama_token id = common_sampler_sample(smpl, mi.ctx, -1);
            if (llama_vocab_is_eog(mi.vocab, id)) break;
            answer += common_token_to_piece(mi.vocab, id);
            common_sampler_accept(smpl, id, true);
            common_batch_clear(gb);
            common_batch_add(gb, id, start_pos + 1 + i, {0}, true);
            if (llama_decode(mi.ctx, gb) != 0) break;
        }
        common_sampler_free(smpl);
        llama_batch_free(gb);
        return answer;
    };

    // Qwen 3.5 is a thinking model — generate 200 tokens to get past <think/> block
    std::string full_answer = generate_answer(n_niah - 1, 200);

    compact_method methods[] = {METHOD_BASELINE, METHOD_EXPECTED_ATTN, METHOD_SNAPKV_W32, METHOD_COMBINED};
    float niah_ratios[] = {0.5f, 0.2f};

    printf("  %-8s  %-12s  %-55s  %s\n", "ratio", "method", "answer", "found?");
    printf("  %-8s  %-12s  %-55s  %s\n", "------", "--------",
           std::string(55, '-').c_str(), "------");
    printf("  %-8s  %-12s  \"%.52s\"  %s\n", "full", "-",
           full_answer.c_str(),
           (full_answer.find("BLUE-FALCON-42") != std::string::npos) ? "YES" : "no");

    int total_pass = 0, total_tests = 0;

    for (float r : niah_ratios) {
        for (compact_method m : methods) {
            // Prefill
            llama_memory_seq_rm(mem, 0, -1, -1);
            llama_batch pb = llama_batch_init(n_niah, 0, 1);
            for (int i = 0; i < n_niah; i++)
                common_batch_add(pb, niah_tokens[i], i, {0}, (i == n_niah - 1));
            llama_decode(mi.ctx, pb);
            llama_batch_free(pb);

            // Save and parse
            size_t ss = llama_state_seq_get_size(mi.ctx, 0);
            std::vector<uint8_t> sb(ss);
            llama_state_seq_get_data(mi.ctx, sb.data(), sb.size(), 0);

            parsed_kv_state ks;
            ks.parse(sb.data(), ss, mi.n_pos_per_embd);

            const auto & sd = ks.streams[0];
            int dk = sd.layers[0].n_embd_k_gqa() / mi.n_head_kv;
            int dv = sd.layers[0].n_embd_v_gqa_computed() / mi.n_head_kv;

            // Run method
            std::vector<int> selected;
            if (!run_compact_method(m, sd.layers[0].K.data(), sd.layers[0].V.data(),
                                    n_niah, mi.n_head_kv, dk, dv, r, selected)) {
                printf("  %-8.0f%%  %-12s  FAILED\n", r * 100, method_name(m));
                continue;
            }

            // Compact all layers
            std::vector<std::vector<std::vector<float>>> cv, beta, dirs;
            compact_all_layers(sd, selected, mi.n_head_kv, dk, dv, cv, beta, dirs);

            auto cb = build_compacted_state(ks, selected, cv, mi.n_head_kv, dk, dv,
                                             mi.n_pos_per_embd);

            llama_memory_seq_rm(mem, 0, -1, -1);
            llama_state_seq_set_data(mi.ctx, cb.data(), cb.size(), 0);

            // Generate answer
            llama_pos pm = llama_memory_seq_pos_max(mem, 0);
            // Prime with space token
            {
                llama_batch prim = llama_batch_init(1, 0, 1);
                std::vector<llama_token> space_tok = common_tokenize(mi.vocab, " ", false, false);
                if (!space_tok.empty()) {
                    common_batch_add(prim, space_tok[0], pm + 1, {0}, true);
                    llama_decode(mi.ctx, prim);
                }
                llama_batch_free(prim);
            }

            std::string answer = generate_answer(pm + 1, 200);
            bool found = answer.find("BLUE-FALCON-42") != std::string::npos;

            total_tests++;
            if (found) total_pass++;

            // Show last 80 chars of answer (past thinking block)
            std::string display = answer.size() > 80 ? answer.substr(answer.size() - 80) : answer;
            printf("  %-8.0f%%  %-12s  \"%.77s\"  %s\n", r * 100, method_name(m),
                   display.c_str(), found ? "YES" : "no");
        }
    }

    printf("\n  Needle retrieval gate: %d/%d passed\n\n", total_pass, total_tests);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION)) {
        fprintf(stderr, "Usage: %s -m <model.gguf> [-c <ctx_size>] [-ngl <n_gpu_layers>]\n", argv[0]);
        return 1;
    }

    if (params.n_ctx < 1024) params.n_ctx = 1024;
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

    model_info mi;
    mi.ctx = ctx;
    mi.model = model;
    mi.vocab = vocab;
    mi.n_layer = llama_model_n_layer(model);
    mi.n_head = llama_model_n_head(model);
    mi.n_head_kv = llama_model_n_head_kv(model);
    mi.n_embd = llama_model_n_embd(model);
    mi.d_k = mi.n_embd / mi.n_head;
    mi.d_v = mi.d_k;
    mi.n_pos_per_embd = (rope_type == LLAMA_ROPE_TYPE_MROPE ||
                          rope_type == LLAMA_ROPE_TYPE_IMROPE) ? 4 : 1;

    printf("bench-opt-real-model\n");
    printf("====================\n");
    printf("Optimization validation gate on real model data\n");

    test_ppl_quality(mi, params);
    test_needle_retrieval(mi, params);

    printf("Validation complete.\n");

    llama_backend_free();
    return 0;
}
