// Real-model quality benchmark for KV cache compaction
//
// Loads an actual GGUF model, prefills a long prompt, captures the KV cache,
// then measures perplexity / KL divergence / generation quality before and
// after compaction at multiple ratios.
//
// Usage:
//   bench-kv-compact-model -m <model.gguf> [-c <ctx_size>] [-ngl <n_gpu_layers>]
//
// Recommended models: Qwen3-0.6B, SmolLM2-1.7B, Llama-3.2-1B (any small GGUF)

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
// Prompts for benchmarking
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

// Continuation prompt for perplexity evaluation (ground truth text after the prefix)
static const char * CONTINUATION =
    "Neural networks research had been abandoned by AI and computer science around "
    "the same time. This line of research was revived by David Rumelhart and others "
    "in the middle of the 1980s. These and other sub-symbolic approaches, such as "
    "fuzzy systems, evolutionary computation and many statistical approaches to AI "
    "were attempted. The field of AI, now more than half a century old, finally "
    "achieved some of its oldest goals. It began to be used successfully throughout "
    "the technology industry, although somewhat behind the scenes.";

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

// ============================================================================
// Thread-parallel per-layer NNLS beta + LS value refit
// ============================================================================
//
// Pre-allocates one workspace per thread (zero allocation in hot path),
// uses mat_mul_ABt for the O(n_q·T·d_k) scoring bottleneck, and distributes
// layers across hardware threads.

// Copy original V values for selected positions in one layer.
// The LS value refit from the paper requires real Q vectors to build the
// softmax design matrix. With only the KV cache (no Q), K-as-Q_ref
// produces degenerate attention (K@K^T scores ~500-1500 after scaling,
// softmax becomes one-hot, NNLS collapses to 1 weight). Original V
// is the correct baseline when real Q vectors aren't available.
static void compact_layer_into(
        const float * V_layer,
        const std::vector<int> & selected,
        int T, int n_head_kv, int d_v,
        std::vector<std::vector<float>> & beta_out,
        std::vector<std::vector<float>> & cv_out) {

    const int t = (int)selected.size();
    const int n_embd_v = n_head_kv * d_v;

    // Copy original V values for selected positions.
    // The LS value refit requires real Q vectors (not K-as-proxy) to produce
    // accurate C_v. With K-as-Q_ref, K@K^T attention is too peaky and
    // LS produces wildly wrong values. Original V is the correct baseline
    // when Q vectors are unavailable.
    for (int h = 0; h < n_head_kv; h++) {
        for (int j = 0; j < t; j++) {
            beta_out[h][j] = 0.0f;
            const float * src = V_layer + selected[j] * n_embd_v + h * d_v;
            memcpy(cv_out[h].data() + j * d_v, src, d_v * sizeof(float));
        }
    }
}

// Compact all layers with shared key selection.
// Copies original V values for selected positions (no LS refit).
// LS value refit requires real Q vectors which aren't available from
// the KV cache state alone.
//
// layer_filter: optional filter to skip non-attention layers in hybrid models.
//   When NULL, all layers are compacted. When provided, skipped layers still
//   get their original V values copied (passthrough) to maintain state format.
static void compact_all_layers(
        const parsed_kv_state::stream_data & sd,
        const std::vector<int> & selected,
        int n_head_kv, int d_k, int d_v,
        std::vector<std::vector<std::vector<float>>> & cv_all,
        std::vector<std::vector<std::vector<float>>> & beta_all,
        std::vector<std::vector<std::vector<float>>> & dirs_all,
        kv_layer_filter_fn layer_filter = NULL,
        void * layer_filter_data = NULL) {

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

        // All layers get original V passthrough (same behavior for
        // attention and non-attention layers in this V-copy mode)
        compact_layer_into(sd.layers[l].V.data(), selected,
                          sd.cell_count, n_head_kv, d_v,
                          beta_all[l], cv_all[l]);
    }
}

// ============================================================================
// Benchmark: Full-model perplexity and KL divergence
// ============================================================================

static void bench_model_quality(model_info & mi, common_params & params) {
    printf("\n=== Real-Model Quality Benchmark ===\n");
    char desc_buf[256];
    llama_model_desc(mi.model, desc_buf, sizeof(desc_buf));
    printf("  Model: %s\n", desc_buf);
    printf("  Layers: %d, KV heads: %d, d_k: %d, n_embd: %d\n",
           mi.n_layer, mi.n_head_kv, mi.d_k, mi.n_embd);
    printf("  n_pos_per_embd: %u\n\n", mi.n_pos_per_embd);

    // Tokenize prompt and continuation
    std::vector<llama_token> prompt_tokens = common_tokenize(mi.vocab, BENCH_PROMPT, true, false);
    std::vector<llama_token> cont_tokens = common_tokenize(mi.vocab, CONTINUATION, false, false);

    int n_prompt = (int)prompt_tokens.size();
    int n_cont = (int)cont_tokens.size();
    int n_vocab = llama_vocab_n_tokens(mi.vocab);

    printf("  Prompt tokens: %d, Continuation tokens: %d, Vocab: %d\n\n", n_prompt, n_cont, n_vocab);

    // Combine for full prefill
    std::vector<llama_token> all_tokens;
    all_tokens.insert(all_tokens.end(), prompt_tokens.begin(), prompt_tokens.end());
    all_tokens.insert(all_tokens.end(), cont_tokens.begin(), cont_tokens.end());
    int n_all = (int)all_tokens.size();

    // ---- Step 1: Full prefill to get reference logits ----
    printf("  [1/5] Full prefill (%d tokens)...\n", n_all);

    llama_memory_t mem = llama_get_memory(mi.ctx);
    llama_memory_seq_rm(mem, 0, -1, -1);

    llama_batch batch = llama_batch_init(n_all, 0, 1);
    for (int i = 0; i < n_all; i++) {
        // Request logits for all continuation positions
        bool need_logits = (i >= n_prompt - 1);
        common_batch_add(batch, all_tokens[i], i, {0}, need_logits);
    }
    int rc = llama_decode(mi.ctx, batch);
    if (rc != 0) {
        printf("  ERROR: prefill failed (rc=%d). Context too small? Try -c %d\n", rc, n_all + 64);
        llama_batch_free(batch);
        return;
    }

    // Collect reference logits for each continuation token.
    // Logit at batch position (n_prompt-1+i) predicts token all_tokens[n_prompt+i].
    // ref_probs[i] = P(· | all_tokens[0..n_prompt-1+i])
    std::vector<std::vector<float>> ref_probs(n_cont);
    double ref_log_prob = 0.0;

    for (int i = 0; i < n_cont; i++) {
        ref_probs[i] = get_logits_at(mi.ctx, n_prompt - 1 + i, n_vocab);
        softmax_inplace(ref_probs[i]);

        // Log prob of actual next token
        llama_token target = all_tokens[n_prompt + i];
        ref_log_prob += log(ref_probs[i][target] + 1e-12);
    }
    double ref_ppl = exp(-ref_log_prob / n_cont);
    printf("  Reference perplexity: %.4f (over %d continuation tokens)\n\n", ref_ppl, n_cont);

    // ---- Step 2: Save state ----
    printf("  [2/5] Saving KV state...\n");

    // Clear and re-prefill just the prompt portion for compaction
    llama_memory_seq_rm(mem, 0, -1, -1);

    llama_batch prompt_batch = llama_batch_init(n_prompt, 0, 1);
    for (int i = 0; i < n_prompt; i++) {
        common_batch_add(prompt_batch, prompt_tokens[i], i, {0}, (i == n_prompt - 1));
    }
    rc = llama_decode(mi.ctx, prompt_batch);
    assert(rc == 0);
    llama_batch_free(prompt_batch);

    const size_t state_size = llama_state_seq_get_size(mi.ctx, 0);
    std::vector<uint8_t> state_buf(state_size);
    size_t saved = llama_state_seq_get_data(mi.ctx, state_buf.data(), state_buf.size(), 0);
    assert(saved > 0);
    printf("  State: %zu bytes (%.2f MB)\n\n", saved, saved / (1024.0 * 1024.0));

    // Parse state
    parsed_kv_state kv_state;
    bool parsed = kv_state.parse(state_buf.data(), saved, mi.n_pos_per_embd);
    assert(parsed);

    const auto & sd = kv_state.streams[0];
    int actual_d_k = sd.layers[0].n_embd_k_gqa() / mi.n_head_kv;
    int actual_d_v = sd.layers[0].n_embd_v_gqa_computed() / mi.n_head_kv;

    // ---- Step 3: Compact at multiple ratios and measure quality ----
    printf("  [3/6] Compacting and evaluating...\n\n");

    // Per-token KV cost: attention layers only have KV; SSM layers have recurrent state.
    // We calculate from the actual state size.
    const double kv_bytes_per_token = (double)saved / n_prompt;
    printf("  Full state: %.2f MB for %d tokens = %.1f bytes/tok\n\n",
           saved / (1024.0 * 1024.0), n_prompt, kv_bytes_per_token);

    float ratios[] = {0.8f, 0.5f, 0.3f, 0.2f, 0.1f, 0.05f, 0.033f, 0.025f, 0.02f};

    printf("  %-8s  %8s  %9s  %9s  %7s  %12s  %12s  %12s  %12s  %7s  %7s  %7s  %10s\n",
           "ratio", "t/T", "KV(MB)", "comp(MB)", "batch", "ppl", "ppl_ratio", "avg_KL", "max_KL", "top1%", "top5%", "dPPL%%", "time_ms");
    printf("  %-8s  %8s  %9s  %9s  %7s  %12s  %12s  %12s  %12s  %7s  %7s  %7s  %10s\n",
           "--------", "--------", "---------", "---------", "-------",
           "------------", "------------", "------------", "------------", "-------", "-------", "-------", "----------");

    for (float ratio : ratios) {
        auto t_start = clock_type::now();

        // Use the C API for compaction on layer 0 to get key selection,
        // then thread-parallel NNLS + LS across all layers
        int T = n_prompt;

        kv_compact_params p = kv_compact_params_default();
        p.target_ratio = ratio;
        p.use_cheap_qref = 1;  // API used for key selection only

        kv_compact_result compact_result = {};
        rc = kv_compact(sd.layers[0].K.data(), sd.layers[0].V.data(), NULL,
                        T, 0, mi.n_head_kv, actual_d_k, actual_d_v,
                        &p, &compact_result);

        if (rc != 0) {
            printf("  %-8.0f%%  FAILED (rc=%d)\n", ratio * 100, rc);
            continue;
        }

        int t = compact_result.t;

        // Build compacted state for ALL layers using shared selection
        // Thread-parallel NNLS beta + LS value refit per layer
        std::vector<int> selected(compact_result.selected_indices,
                                   compact_result.selected_indices + t);

        // Copy original V values for all layers at selected positions
        std::vector<std::vector<std::vector<float>>> cv_all, beta_all, beta_dirs;
        compact_all_layers(sd, selected,
                          mi.n_head_kv, actual_d_k, actual_d_v,
                          cv_all, beta_all, beta_dirs);

        auto compacted_buf = build_compacted_state(kv_state, selected, cv_all,
                                                    mi.n_head_kv, actual_d_k, actual_d_v,
                                                    mi.n_pos_per_embd);

        double compact_ms = std::chrono::duration<double, std::milli>(
            clock_type::now() - t_start).count();

        // Load compacted state
        llama_memory_seq_rm(mem, 0, -1, -1);
        size_t loaded = llama_state_seq_set_data(mi.ctx, compacted_buf.data(), compacted_buf.size(), 0);
        if (loaded == 0) {
            printf("  %-8.0f%%  LOAD FAILED\n", ratio * 100);
            kv_compact_result_free(&compact_result);
            continue;
        }

        // Decode continuation tokens to get compacted logits.
        // After loading compacted state, pos_max is the highest position in the
        // compacted cache. We continue from pos_max + 1, which is the position
        // after the last cached token. The continuation tokens start here.
        //
        // Note: The first continuation token's logit (predicting cont_token[0])
        // came from the last prompt position. After compaction, the last prompt
        // position is still in the cache, so we can directly decode continuation
        // tokens starting at pos_max + 1. The logits at position i predict
        // the token at position i+1.
        llama_pos pos_max = llama_memory_seq_pos_max(mem, 0);

        double comp_log_prob = 0.0;
        double sum_kl = 0.0, max_kl = 0.0;
        int top1_match = 0, top5_match = 0;

        // Decode all continuation tokens at once, starting from pos_max + 1
        llama_batch eval_batch = llama_batch_init(n_cont, 0, 1);
        for (int i = 0; i < n_cont; i++) {
            common_batch_add(eval_batch, all_tokens[n_prompt + i], pos_max + 1 + i, {0}, true);
        }

        rc = llama_decode(mi.ctx, eval_batch);
        if (rc != 0) {
            printf("  %-8.0f%%  EVAL FAILED (rc=%d)\n", ratio * 100, rc);
            llama_batch_free(eval_batch);
            kv_compact_result_free(&compact_result);
            continue;
        }

        // Logit at batch index i predicts the NEXT token (i+1).
        // eval_batch[0] is at pos_max+1, its logits predict the token at pos_max+2.
        // We want: P(cont_token[i+1] | context up to cont_token[i]).
        // So logits at batch index i → target = all_tokens[n_prompt + i + 1]
        // We can evaluate n_cont - 1 such predictions (the last position has no target).
        int n_eval = n_cont - 1;
        for (int i = 0; i < n_eval; i++) {
            std::vector<float> comp_probs = get_logits_at(mi.ctx, i, n_vocab);
            softmax_inplace(comp_probs);

            llama_token target = all_tokens[n_prompt + i + 1];
            comp_log_prob += log(comp_probs[target] + 1e-12);

            // Compare against reference (ref_probs[i] predicted all_tokens[n_prompt+i])
            // We need ref_probs[i+1] which predicted all_tokens[n_prompt+i+1]
            if (i + 1 < n_cont) {
                const auto & ref = ref_probs[i + 1];
                double kl = kl_divergence(ref, comp_probs);
                sum_kl += kl;
                if (kl > max_kl) max_kl = kl;

                // Top-1 and top-5 token agreement
                auto argmax = [](const std::vector<float> & v) {
                    return (int)std::distance(v.begin(), std::max_element(v.begin(), v.end()));
                };
                if (argmax(ref) == argmax(comp_probs)) top1_match++;

                // Top-5: check if reference argmax appears in compacted top-5
                std::vector<int> top5_idx(n_vocab);
                std::iota(top5_idx.begin(), top5_idx.end(), 0);
                std::partial_sort(top5_idx.begin(), top5_idx.begin() + 5, top5_idx.end(),
                    [&comp_probs](int a, int b) { return comp_probs[a] > comp_probs[b]; });
                int ref_top = argmax(ref);
                for (int k = 0; k < 5; k++) {
                    if (top5_idx[k] == ref_top) { top5_match++; break; }
                }
            }
        }

        double comp_ppl = exp(-comp_log_prob / n_eval);
        double ppl_ratio = comp_ppl / ref_ppl;
        double avg_kl = sum_kl / n_eval;
        double top1_pct = 100.0 * top1_match / n_eval;
        double top5_pct = 100.0 * top5_match / n_eval;

        double comp_state_mb = compacted_buf.size() / (1024.0 * 1024.0);
        double full_state_mb = saved / (1024.0 * 1024.0);
        // Batch capacity: how many sequences fit in same memory as 1 full-cache seq
        int batch_capacity = (int)(full_state_mb / comp_state_mb);
        double d_ppl_pct = (ppl_ratio - 1.0) * 100.0;

        printf("  %-8.0f%%  %4d/%-3d  %8.2f  %8.2f  %5d   %12.4f  %12.4f  %12.6f  %12.6f  %6.1f%%  %6.1f%%  %+5.1f%%  %10.1f\n",
               ratio * 100, t, T, full_state_mb, comp_state_mb, batch_capacity,
               comp_ppl, ppl_ratio, avg_kl, max_kl, top1_pct, top5_pct, d_ppl_pct, compact_ms);

        llama_batch_free(eval_batch);
        kv_compact_result_free(&compact_result);
    }

    // ---- Step 4: Needle-in-a-Haystack retrieval ----
    printf("\n  [4/6] Needle-in-a-Haystack retrieval test...\n\n");
    {
        // Insert a distinctive fact ("needle") in the middle of a padded prompt,
        // compact, then check if the model can retrieve it.
        const char * needle = "The secret code is BLUE-FALCON-42.";
        const char * haystack_before =
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
        const char * haystack_after =
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
        const char * question = " What is the secret code mentioned above?";

        std::string niah_prompt = std::string(haystack_before) + needle + " " +
                                   std::string(haystack_after) + question;

        std::vector<llama_token> niah_tokens = common_tokenize(mi.vocab, niah_prompt, true, false);
        int n_niah = (int)niah_tokens.size();

        if (n_niah + 64 > (int)llama_n_ctx(mi.ctx)) {
            printf("  SKIP: NIAH prompt (%d tokens) too large for context\n", n_niah);
        } else {
            // Full-cache baseline: prefill and generate
            llama_memory_seq_rm(mem, 0, -1, -1);
            llama_batch nb = llama_batch_init(n_niah, 0, 1);
            for (int i = 0; i < n_niah; i++)
                common_batch_add(nb, niah_tokens[i], i, {0}, (i == n_niah - 1));
            llama_decode(mi.ctx, nb);
            llama_batch_free(nb);

            std::string full_answer;
            {
                common_sampler * smpl = common_sampler_init(mi.model, params.sampling);
                llama_batch gb = llama_batch_init(1, 0, 1);
                for (int i = 0; i < 30; i++) {
                    llama_token id = common_sampler_sample(smpl, mi.ctx, -1);
                    if (llama_vocab_is_eog(mi.vocab, id)) break;
                    full_answer += common_token_to_piece(mi.vocab, id);
                    common_sampler_accept(smpl, id, true);
                    common_batch_clear(gb);
                    common_batch_add(gb, id, n_niah + i, {0}, true);
                    if (llama_decode(mi.ctx, gb) != 0) break;
                }
                common_sampler_free(smpl);
                llama_batch_free(gb);
            }

            // Test at multiple compression ratios
            float niah_ratios[] = {0.5f, 0.3f, 0.2f};
            printf("  %-8s  %-60s  %s\n", "ratio", "answer", "found?");
            printf("  %-8s  %-60s  %s\n", "--------", std::string(60, '-').c_str(), "------");
            printf("  %-8s  \"%.57s\"  %s\n", "full",
                   full_answer.c_str(),
                   (full_answer.find("BLUE-FALCON-42") != std::string::npos) ? "YES" : "no");

            for (float r : niah_ratios) {
                // Prefill, save, compact, reload
                llama_memory_seq_rm(mem, 0, -1, -1);
                llama_batch pb2 = llama_batch_init(n_niah, 0, 1);
                for (int i = 0; i < n_niah; i++)
                    common_batch_add(pb2, niah_tokens[i], i, {0}, (i == n_niah - 1));
                llama_decode(mi.ctx, pb2);
                llama_batch_free(pb2);

                size_t ss = llama_state_seq_get_size(mi.ctx, 0);
                std::vector<uint8_t> sbuf(ss);
                llama_state_seq_get_data(mi.ctx, sbuf.data(), sbuf.size(), 0);

                parsed_kv_state ks_n;
                ks_n.parse(sbuf.data(), ss, mi.n_pos_per_embd);

                kv_compact_params pn = kv_compact_params_default();
                pn.target_ratio = r;
                pn.use_cheap_qref = 1;

                int niah_d_k = ks_n.streams[0].layers[0].n_embd_k_gqa() / mi.n_head_kv;
                int niah_d_v = ks_n.streams[0].layers[0].n_embd_v_gqa_computed() / mi.n_head_kv;

                kv_compact_result rn = {};
                kv_compact(ks_n.streams[0].layers[0].K.data(),
                           ks_n.streams[0].layers[0].V.data(), NULL,
                           n_niah, 0, mi.n_head_kv, niah_d_k, niah_d_v,
                           &pn, &rn);

                std::vector<int> sel_n(rn.selected_indices, rn.selected_indices + rn.t);
                std::vector<std::vector<std::vector<float>>> cv_n, beta_n, dirs_n;
                compact_all_layers(ks_n.streams[0], sel_n,
                                  mi.n_head_kv, niah_d_k, niah_d_v,
                                  cv_n, beta_n, dirs_n);

                auto cb_n = build_compacted_state(ks_n, sel_n, cv_n,
                                                   mi.n_head_kv, niah_d_k, niah_d_v,
                                                   mi.n_pos_per_embd);

                llama_memory_seq_rm(mem, 0, -1, -1);
                llama_state_seq_set_data(mi.ctx, cb_n.data(), cb_n.size(), 0);
                kv_compact_result_free(&rn);

                llama_pos pm = llama_memory_seq_pos_max(mem, 0);
                std::string comp_answer;
                {
                    common_sampler * smpl = common_sampler_init(mi.model, params.sampling);
                    llama_batch gb = llama_batch_init(1, 0, 1);
                    // Prime with a dummy decode to get first logits
                    // Use a space token as neutral primer
                    std::vector<llama_token> space_tok = common_tokenize(mi.vocab, " ", false, false);
                    if (!space_tok.empty()) {
                        common_batch_add(gb, space_tok[0], pm + 1, {0}, true);
                        llama_decode(mi.ctx, gb);
                        common_batch_clear(gb);
                    }
                    for (int i = 0; i < 30; i++) {
                        llama_token id = common_sampler_sample(smpl, mi.ctx, -1);
                        if (llama_vocab_is_eog(mi.vocab, id)) break;
                        comp_answer += common_token_to_piece(mi.vocab, id);
                        common_sampler_accept(smpl, id, true);
                        common_batch_clear(gb);
                        common_batch_add(gb, id, pm + 2 + i, {0}, true);
                        if (llama_decode(mi.ctx, gb) != 0) break;
                    }
                    common_sampler_free(smpl);
                    llama_batch_free(gb);
                }

                bool found = comp_answer.find("BLUE-FALCON-42") != std::string::npos;
                printf("  %-8.0f%%  \"%.57s\"  %s\n", r * 100,
                       comp_answer.c_str(), found ? "YES" : "no");
            }
        }
    }

    // ---- Step 5: Generation throughput (tok/s) at various compression ratios ----
    printf("\n  [5/6] Generation throughput (tok/s)...\n\n");

    const int n_gen = 100;

    // Helper lambda: prefill, compact at given ratio, generate, return tok/s and output
    // Also returns compacted state size via reference
    auto bench_gen = [&](float ratio, std::string & output, size_t & out_state_size) -> double {
        // Prefill
        llama_memory_seq_rm(mem, 0, -1, -1);
        llama_batch pb = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++)
            common_batch_add(pb, prompt_tokens[i], i, {0}, (i == n_prompt - 1));
        llama_decode(mi.ctx, pb);
        llama_batch_free(pb);

        if (ratio >= 1.0f) {
            // Full cache — no compaction
            out_state_size = saved;
        } else {
            // Save state
            size_t ss = llama_state_seq_get_size(mi.ctx, 0);
            std::vector<uint8_t> sb(ss);
            llama_state_seq_get_data(mi.ctx, sb.data(), sb.size(), 0);

            parsed_kv_state ks;
            ks.parse(sb.data(), ss, mi.n_pos_per_embd);

            kv_compact_params pc = kv_compact_params_default();
            pc.target_ratio = ratio;
            pc.use_cheap_qref = 1;

            kv_compact_result rc2 = {};
            int rr = kv_compact(ks.streams[0].layers[0].K.data(),
                                ks.streams[0].layers[0].V.data(), NULL,
                                n_prompt, 0, mi.n_head_kv, actual_d_k, actual_d_v,
                                &pc, &rc2);
            if (rr != 0) { out_state_size = 0; return 0.0; }

            std::vector<int> sel(rc2.selected_indices, rc2.selected_indices + rc2.t);

            std::vector<std::vector<std::vector<float>>> cv_r, beta_r, dirs_r;
            compact_all_layers(ks.streams[0], sel,
                              mi.n_head_kv, actual_d_k, actual_d_v,
                              cv_r, beta_r, dirs_r);

            auto cb = build_compacted_state(ks, sel, cv_r, mi.n_head_kv, actual_d_k, actual_d_v,
                                             mi.n_pos_per_embd);
            out_state_size = cb.size();

            llama_memory_seq_rm(mem, 0, -1, -1);
            llama_state_seq_set_data(mi.ctx, cb.data(), cb.size(), 0);
            kv_compact_result_free(&rc2);
        }

        llama_pos pm = llama_memory_seq_pos_max(mem, 0);

        // Seed with first continuation token
        llama_batch seed_b = llama_batch_init(1, 0, 1);
        common_batch_add(seed_b, cont_tokens[0], pm + 1, {0}, true);
        llama_decode(mi.ctx, seed_b);
        llama_batch_free(seed_b);

        output = common_token_to_piece(mi.vocab, cont_tokens[0]);

        // Timed generation
        auto gen_start = clock_type::now();
        common_sampler * smpl = common_sampler_init(mi.model, params.sampling);
        llama_batch gb = llama_batch_init(1, 0, 1);
        int generated = 0;
        for (int i = 0; i < n_gen - 1; i++) {
            llama_token id = common_sampler_sample(smpl, mi.ctx, -1);
            if (llama_vocab_is_eog(mi.vocab, id)) break;
            output += common_token_to_piece(mi.vocab, id);
            common_sampler_accept(smpl, id, true);
            common_batch_clear(gb);
            common_batch_add(gb, id, pm + 2 + i, {0}, true);
            if (llama_decode(mi.ctx, gb) != 0) break;
            generated++;
        }
        double gen_ms = std::chrono::duration<double, std::milli>(clock_type::now() - gen_start).count();
        common_sampler_free(smpl);
        llama_batch_free(gb);

        // total tokens = 1 (seed) + generated
        return (1 + generated) / (gen_ms / 1000.0);
    };

    printf("  %-12s  %8s  %9s  %9s  %7s  %10s  %s\n",
           "compression", "kept", "full(MB)", "comp(MB)", "batch", "tok/s", "output");
    printf("  %-12s  %8s  %9s  %9s  %7s  %10s  %s\n",
           "------------", "--------", "---------", "---------", "-------", "----------",
           std::string(60, '-').c_str());

    float gen_ratios[] = {1.0f, 0.5f, 0.2f, 0.1f, 0.05f, 0.033f, 0.025f, 0.02f};
    for (float r : gen_ratios) {
        std::string out;
        size_t comp_size = 0;
        double tps = bench_gen(r, out, comp_size);
        int kept = (r >= 1.0f) ? n_prompt : std::max(1, (int)(n_prompt * r));
        double compression = (double)n_prompt / kept;
        double full_mb = saved / (1024.0 * 1024.0);
        double comp_mb = comp_size / (1024.0 * 1024.0);
        int batch = (comp_size > 0) ? (int)(saved / comp_size) : 1;

        char ratio_str[32];
        if (r >= 1.0f) snprintf(ratio_str, sizeof(ratio_str), "full");
        else snprintf(ratio_str, sizeof(ratio_str), "%.0fx (%.1f%%)", compression, r * 100);

        printf("  %-12s  %4d/%-3d  %8.2f  %8.2f  %5d   %10.2f  \"%.55s\"\n",
               ratio_str, kept, n_prompt, full_mb, comp_mb, batch, tps, out.c_str());
    }

    // ---- Step 6: Batching capacity summary ----
    printf("\n  [6/6] Batching capacity (same %.0f MB budget):\n\n", saved / (1024.0 * 1024.0));
    printf("  %-12s  %7s  %9s  %12s\n", "compression", "seqs", "tok/seq", "total_tok");
    printf("  %-12s  %7s  %9s  %12s\n", "------------", "-------", "---------", "------------");
    for (float r : gen_ratios) {
        int kept = (r >= 1.0f) ? n_prompt : std::max(1, (int)(n_prompt * r));
        double comp_state_bytes = saved * ((double)kept / n_prompt);
        int seqs = (int)(saved / comp_state_bytes);
        if (seqs < 1) seqs = 1;
        long total_tok = (long)seqs * n_prompt;
        double compression = (double)n_prompt / kept;

        char ratio_str[32];
        if (r >= 1.0f) snprintf(ratio_str, sizeof(ratio_str), "full");
        else snprintf(ratio_str, sizeof(ratio_str), "%.0fx", compression);

        printf("  %-12s  %5d   %7d   %10ld\n", ratio_str, seqs, n_prompt, total_tok);
    }

    llama_batch_free(batch);
    printf("\n  Done.\n");
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char ** argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMPLETION)) {
        fprintf(stderr, "Usage: %s -m <model.gguf> [-c <ctx_size>]\n", argv[0]);
        return 1;
    }

    // Ensure enough context for our benchmark prompt
    if (params.n_ctx < 1024) {
        params.n_ctx = 1024;
    }

    // Ensure batch size is large enough for bulk prefills (5k–100k tokens)
    if (params.n_batch < (int)params.n_ctx) {
        params.n_batch = params.n_ctx;
    }

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

    bench_model_quality(mi, params);

    // ---- Scaling benchmark: compaction at 5k, 10k, 20k, 50k, 75k, 100k ----
    {
        printf("\n\n=== Scaling Benchmark ===\n");
        char desc_buf[256];
        llama_model_desc(mi.model, desc_buf, sizeof(desc_buf));
        printf("  Model: %s\n\n", desc_buf);

        llama_memory_t mem = llama_get_memory(mi.ctx);
        const int n_ctx = llama_n_ctx(mi.ctx);

        // Generate a long base text by repeating BENCH_PROMPT
        std::string base_text = BENCH_PROMPT;
        std::string long_text;
        while (long_text.size() < 500000) long_text += base_text;
        std::vector<llama_token> all_tokens = common_tokenize(mi.vocab, long_text, true, false);

        int ctx_sizes[] = {5000, 10000, 20000, 50000, 75000, 100000};
        float comp_ratios[] = {0.5f, 0.2f, 0.1f, 0.05f, 0.02f};

        printf("  %-8s  %8s  %9s", "ctx", "ratio", "kept");
        for (float cr : comp_ratios) {
            printf("  %10s", cr >= 1.0f ? "full(MB)" :
                   cr == 0.5f ? "2x(MB)" :
                   cr == 0.2f ? "5x(MB)" :
                   cr == 0.1f ? "10x(MB)" :
                   cr == 0.05f ? "20x(MB)" : "50x(MB)");
        }
        printf("  %10s  %10s\n", "compact(ms)", "tok/s");
        printf("  %-8s  %8s  %9s", "--------", "--------", "--------");
        for (int i = 0; i < (int)(sizeof(comp_ratios)/sizeof(comp_ratios[0])); i++)
            printf("  %10s", "----------");
        printf("  %10s  %10s\n", "----------", "----------");

        for (int target_ctx : ctx_sizes) {
            if (target_ctx >= n_ctx) {
                printf("  %-8d  (exceeds -c %d, skip)\n", target_ctx, n_ctx);
                continue;
            }

            int n_tok = std::min(target_ctx - 64, (int)all_tokens.size());
            if (n_tok <= 0) continue;

            // Prefill
            llama_memory_seq_rm(mem, 0, -1, -1);
            llama_batch pb = llama_batch_init(n_tok, 0, 1);
            for (int i = 0; i < n_tok; i++)
                common_batch_add(pb, all_tokens[i], i, {0}, (i == n_tok - 1));
            int rc = llama_decode(mi.ctx, pb);
            llama_batch_free(pb);
            if (rc != 0) {
                printf("  %-8d  PREFILL FAILED (rc=%d)\n", n_tok, rc);
                continue;
            }

            // Save state
            size_t ss = llama_state_seq_get_size(mi.ctx, 0);
            std::vector<uint8_t> sb(ss);
            size_t saved_bytes = llama_state_seq_get_data(mi.ctx, sb.data(), sb.size(), 0);
            double full_mb = saved_bytes / (1024.0 * 1024.0);

            // Parse state
            parsed_kv_state ks;
            ks.parse(sb.data(), saved_bytes, mi.n_pos_per_embd);
            const auto & sd = ks.streams[0];
            int adk = sd.layers[0].n_embd_k_gqa() / mi.n_head_kv;
            int adv = sd.layers[0].n_embd_v_gqa_computed() / mi.n_head_kv;

            // Full state column
            printf("  %-8d  %8s  %9d  %10.2f", n_tok, "full", n_tok, full_mb);

            // Compact at each ratio
            for (float cr : comp_ratios) {
                auto t0 = clock_type::now();

                kv_compact_params cp = kv_compact_params_default();
                cp.target_ratio = cr;
                cp.use_cheap_qref = 1;

                kv_compact_result cr2 = {};
                int rr = kv_compact(sd.layers[0].K.data(), sd.layers[0].V.data(), NULL,
                                    n_tok, 0, mi.n_head_kv, adk, adv, &cp, &cr2);
                if (rr != 0) {
                    printf("  %10s", "FAIL");
                    continue;
                }

                std::vector<int> sel(cr2.selected_indices, cr2.selected_indices + cr2.t);
                std::vector<std::vector<std::vector<float>>> cv_r, beta_r, dirs_r;
                compact_all_layers(sd, sel, mi.n_head_kv, adk, adv, cv_r, beta_r, dirs_r);

                auto cb = build_compacted_state(ks, sel, cv_r, mi.n_head_kv, adk, adv,
                                                 mi.n_pos_per_embd);

                double ms = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
                double comp_mb = cb.size() / (1024.0 * 1024.0);
                double saved_pct = (1.0 - (double)cb.size() / saved_bytes) * 100.0;

                // Only print size for the first ratio, print time for all
                printf("  %10.2f", comp_mb);

                kv_compact_result_free(&cr2);
            }

            // Re-run just one ratio (20x) to get clean timing
            {
                kv_compact_params cp = kv_compact_params_default();
                cp.target_ratio = 0.05f;
                cp.use_cheap_qref = 1;
                kv_compact_result cr2 = {};
                auto t0 = clock_type::now();
                int rr = kv_compact(sd.layers[0].K.data(), sd.layers[0].V.data(), NULL,
                                    n_tok, 0, mi.n_head_kv, adk, adv, &cp, &cr2);
                double ms = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
                if (rr == 0) {
                    printf("  %10.1f  %10.1f", ms, n_tok / (ms / 1000.0));
                    kv_compact_result_free(&cr2);
                }
            }

            printf("\n");
        }

        // Batch capacity table
        printf("\n  Batching capacity (sequences fitting in full-cache memory of one 50k context):\n\n");
        {
            // Estimate from 50k row if available
            int ref_ctx = 50000;
            if (ref_ctx >= n_ctx) ref_ctx = n_ctx - 64;
            int n_tok = std::min(ref_ctx, (int)all_tokens.size());

            llama_memory_seq_rm(mem, 0, -1, -1);
            llama_batch pb = llama_batch_init(n_tok, 0, 1);
            for (int i = 0; i < n_tok; i++)
                common_batch_add(pb, all_tokens[i], i, {0}, (i == n_tok - 1));
            llama_decode(mi.ctx, pb);
            llama_batch_free(pb);

            size_t ss = llama_state_seq_get_size(mi.ctx, 0);
            std::vector<uint8_t> sb(ss);
            size_t ref_bytes = llama_state_seq_get_data(mi.ctx, sb.data(), sb.size(), 0);
            double ref_mb = ref_bytes / (1024.0 * 1024.0);

            parsed_kv_state ks;
            ks.parse(sb.data(), ref_bytes, mi.n_pos_per_embd);
            const auto & sd = ks.streams[0];
            int adk = sd.layers[0].n_embd_k_gqa() / mi.n_head_kv;
            int adv = sd.layers[0].n_embd_v_gqa_computed() / mi.n_head_kv;

            printf("  Reference: %d tokens, %.2f MB state\n\n", n_tok, ref_mb);
            printf("  %-12s  %9s  %7s  %9s  %12s\n", "compression", "comp(MB)", "seqs", "tok/seq", "total_tok");
            printf("  %-12s  %9s  %7s  %9s  %12s\n", "------------", "---------", "-------", "---------", "------------");

            float bratios[] = {1.0f, 0.5f, 0.2f, 0.1f, 0.05f, 0.033f, 0.025f, 0.02f};
            for (float r : bratios) {
                kv_compact_params cp = kv_compact_params_default();
                cp.target_ratio = r;
                cp.use_cheap_qref = 1;

                kv_compact_result cr2 = {};
                int rr = kv_compact(sd.layers[0].K.data(), sd.layers[0].V.data(), NULL,
                                    n_tok, 0, mi.n_head_kv, adk, adv, &cp, &cr2);
                if (rr != 0) continue;

                std::vector<int> sel(cr2.selected_indices, cr2.selected_indices + cr2.t);
                std::vector<std::vector<std::vector<float>>> cv_r, beta_r, dirs_r;
                compact_all_layers(sd, sel, mi.n_head_kv, adk, adv, cv_r, beta_r, dirs_r);
                auto cb = build_compacted_state(ks, sel, cv_r, mi.n_head_kv, adk, adv,
                                                 mi.n_pos_per_embd);

                double comp_mb = cb.size() / (1024.0 * 1024.0);
                int seqs = (int)(ref_bytes / cb.size());
                if (seqs < 1) seqs = 1;
                long total = (long)seqs * n_tok;
                double compression = (double)n_tok / (int)(n_tok * r);

                char label[32];
                if (r >= 1.0f) snprintf(label, sizeof(label), "full");
                else snprintf(label, sizeof(label), "%.0fx", compression);

                printf("  %-12s  %9.2f  %5d   %7d   %10ld\n", label, comp_mb, seqs, n_tok, total);
                kv_compact_result_free(&cr2);
            }
        }
    }

    llama_backend_free();
    return 0;
}
