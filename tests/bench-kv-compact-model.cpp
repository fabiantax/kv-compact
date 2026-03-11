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
// Per-layer NNLS beta + LS value refit using shared selected indices
// ============================================================================
//
// Given pre-selected key indices (from layer 0), runs the full 3-step pipeline
// (Steps 2-3 from the paper) on each layer independently:
//   - Generate cheap Q_ref from layer's K
//   - NNLS solve for per-head beta (Section 3.2)
//   - LS solve for per-head C_v (Section 3.3)

struct per_layer_compact_result {
    std::vector<std::vector<float>> beta;  // [n_head_kv][t]
    std::vector<std::vector<float>> C_v;   // [n_head_kv][t * d_v]
};

static per_layer_compact_result compact_layer_with_shared_selection(
        const float * K_layer,  // [T × n_embd_k] all heads concatenated
        const float * V_layer,  // [T × n_embd_v] all heads concatenated
        const std::vector<int> & selected,
        int T, int n_head_kv, int d_k, int d_v,
        const kv_compact_params & p) {

    const int t = (int)selected.size();
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;
    const float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    // Generate cheap Q_ref from this layer's K
    int n_q = T / 2;
    if (n_q > 64) n_q = 64;
    if (n_q < 4) n_q = 4;
    if (n_q > T) n_q = T;

    std::vector<float> Q_ref((size_t)n_q * n_embd_k);
    for (int qi = 0; qi < n_q; qi++) {
        float frac = (float)(qi + 1) / (float)(n_q + 1);
        frac = frac * frac;  // bias toward recent tokens
        int pos = (int)(frac * (T - 1));
        if (pos >= T) pos = T - 1;
        if (pos < 0) pos = 0;
        memcpy(Q_ref.data() + qi * n_embd_k,
               K_layer + pos * n_embd_k,
               n_embd_k * sizeof(float));
    }

    per_layer_compact_result out;
    out.beta.resize(n_head_kv);
    out.C_v.resize(n_head_kv);

    for (int h = 0; h < n_head_kv; h++) {
        out.beta[h].resize(t);
        out.C_v[h].resize(t * d_v);

        // Compute attention scores: Q_ref[h] @ K[h]^T / sqrt(d_k)
        std::vector<float> scores(n_q * T);
        std::vector<float> exp_scores(n_q * T);
        std::vector<float> row_sums(n_q, 0.0f);

        for (int qi = 0; qi < n_q; qi++) {
            const float * q = Q_ref.data() + qi * n_embd_k + h * d_k;
            float max_s = -1e30f;
            for (int j = 0; j < T; j++) {
                const float * k = K_layer + j * n_embd_k + h * d_k;
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++) dot += q[d] * k[d];
                scores[qi * T + j] = dot * inv_sqrt_dk;
                if (scores[qi * T + j] > max_s) max_s = scores[qi * T + j];
            }
            for (int j = 0; j < T; j++) {
                exp_scores[qi * T + j] = expf(scores[qi * T + j] - max_s);
                row_sums[qi] += exp_scores[qi * T + j];
            }
        }

        // Attention weights (softmax)
        std::vector<float> attn_weights(n_q * T);
        for (int qi = 0; qi < n_q; qi++) {
            for (int j = 0; j < T; j++) {
                attn_weights[qi * T + j] = exp_scores[qi * T + j] / row_sums[qi];
            }
        }

        // NNLS for beta: M * w ≈ row_sums
        std::vector<float> M(n_q * t);
        for (int qi = 0; qi < n_q; qi++) {
            for (int j = 0; j < t; j++) {
                M[qi * t + j] = exp_scores[qi * T + selected[j]];
            }
        }

        std::vector<float> w(t);
        nnls_solve(M.data(), row_sums.data(), w.data(), n_q, t, p.nnls_max_iter);

        for (int j = 0; j < t; j++) {
            out.beta[h][j] = logf(std::max(1e-12f, w[j]));
        }

        // LS for C_v: X * C_v ≈ Y
        std::vector<float> X(n_q * t);
        for (int qi = 0; qi < n_q; qi++) {
            for (int j = 0; j < t; j++) {
                X[qi * t + j] = scores[qi * T + selected[j]] + out.beta[h][j];
            }
        }
        softmax_rows(X.data(), n_q, t);

        // Y = original attention output
        std::vector<float> Y(n_q * d_v, 0.0f);
        for (int qi = 0; qi < n_q; qi++) {
            for (int ki = 0; ki < T; ki++) {
                float w_ij = attn_weights[qi * T + ki];
                const float * v_row = V_layer + ki * n_embd_v + h * d_v;
                for (int d = 0; d < d_v; d++)
                    Y[qi * d_v + d] += w_ij * v_row[d];
            }
        }

        least_squares_solve(X.data(), Y.data(), out.C_v[h].data(),
                           n_q, t, d_v, p.ridge);
    }

    return out;
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
    printf("  [1/4] Full prefill (%d tokens)...\n", n_all);

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
    printf("  [2/4] Saving KV state...\n");

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
    printf("  [3/4] Compacting and evaluating...\n\n");

    float ratios[] = {0.8f, 0.5f, 0.3f, 0.2f};

    printf("  %-8s  %8s  %12s  %12s  %12s  %12s  %10s\n",
           "ratio", "t/T", "ppl", "ppl_ratio", "avg_KL", "max_KL", "time_ms");
    printf("  %-8s  %8s  %12s  %12s  %12s  %12s  %10s\n",
           "--------", "--------", "------------", "------------",
           "------------", "------------", "----------");

    for (float ratio : ratios) {
        auto t_start = clock_type::now();

        // Use the C API for compaction on layer 0 (representative)
        // For full benchmark, we compact all layers via state manipulation
        int T = n_prompt;
        int n_embd_k = mi.n_head_kv * actual_d_k;
        int n_embd_v = mi.n_head_kv * actual_d_v;

        // Use kv_compact API on the first layer to get selected indices
        int n_ref_q = std::min(T / 2, 64);
        if (n_ref_q < 4) n_ref_q = 4;

        kv_compact_params p = kv_compact_params_default();
        p.target_ratio = ratio;
        p.use_cheap_qref = 1;  // no separate Q_ref, generate from K

        kv_compact_result compact_result = {};
        rc = kv_compact(sd.layers[0].K.data(), sd.layers[0].V.data(), NULL,
                        T, 0, mi.n_head_kv, actual_d_k, actual_d_v,
                        &p, &compact_result);

        if (rc != 0) {
            printf("  %-8.0f%%  FAILED (rc=%d)\n", ratio * 100, rc);
            continue;
        }

        int t = compact_result.t;

        // Build compacted state for ALL layers using the shared selection
        std::vector<int> selected(compact_result.selected_indices,
                                   compact_result.selected_indices + t);

        // Full NNLS beta + LS value refit per layer (not simplified eviction)
        std::vector<std::vector<std::vector<float>>> cv_all(sd.n_layer);
        std::vector<std::vector<std::vector<float>>> beta_all(sd.n_layer);
        std::vector<std::vector<std::vector<float>>> beta_dirs(sd.n_layer);

        for (uint32_t l = 0; l < sd.n_layer; l++) {
            const auto & ld = sd.layers[l];

            if (l == 0) {
                // Layer 0: use results from kv_compact() directly
                cv_all[l].resize(mi.n_head_kv);
                beta_all[l].resize(mi.n_head_kv);
                beta_dirs[l].resize(mi.n_head_kv);
                for (int h = 0; h < mi.n_head_kv; h++) {
                    cv_all[l][h].assign(compact_result.C_v[h],
                                         compact_result.C_v[h] + t * actual_d_v);
                    beta_all[l][h].assign(compact_result.beta[h],
                                           compact_result.beta[h] + t);
                    // Beta direction: normalized mean of Q_ref (generated from K)
                    beta_dirs[l][h].resize(actual_d_k);
                    int n_q_ref = std::min(T / 2, 64);
                    if (n_q_ref < 4) n_q_ref = 4;
                    std::fill(beta_dirs[l][h].begin(), beta_dirs[l][h].end(), 0.0f);
                    for (int qi = 0; qi < n_q_ref; qi++) {
                        float frac = (float)(qi + 1) / (float)(n_q_ref + 1);
                        frac = frac * frac;
                        int pos = std::min((int)(frac * (T - 1)), T - 1);
                        const float * k = ld.K.data() + pos * n_embd_k + h * actual_d_k;
                        for (int d = 0; d < actual_d_k; d++) beta_dirs[l][h][d] += k[d];
                    }
                    float norm = 0.0f;
                    for (int d = 0; d < actual_d_k; d++)
                        norm += beta_dirs[l][h][d] * beta_dirs[l][h][d];
                    norm = sqrtf(norm + 1e-12f);
                    for (int d = 0; d < actual_d_k; d++) beta_dirs[l][h][d] /= norm;
                }
            } else {
                // Other layers: run full NNLS + LS with shared selection
                auto lr = compact_layer_with_shared_selection(
                    ld.K.data(), ld.V.data(), selected,
                    T, mi.n_head_kv, actual_d_k, actual_d_v, p);

                cv_all[l] = std::move(lr.C_v);
                beta_all[l] = std::move(lr.beta);

                // Compute beta direction per head (normalized mean Q_ref)
                beta_dirs[l].resize(mi.n_head_kv);
                int n_q_ref = std::min(T / 2, 64);
                if (n_q_ref < 4) n_q_ref = 4;
                for (int h = 0; h < mi.n_head_kv; h++) {
                    beta_dirs[l][h].resize(actual_d_k);
                    std::fill(beta_dirs[l][h].begin(), beta_dirs[l][h].end(), 0.0f);
                    for (int qi = 0; qi < n_q_ref; qi++) {
                        float frac = (float)(qi + 1) / (float)(n_q_ref + 1);
                        frac = frac * frac;
                        int pos = std::min((int)(frac * (T - 1)), T - 1);
                        const float * k = ld.K.data() + pos * n_embd_k + h * actual_d_k;
                        for (int d = 0; d < actual_d_k; d++) beta_dirs[l][h][d] += k[d];
                    }
                    float norm2 = 0.0f;
                    for (int d = 0; d < actual_d_k; d++)
                        norm2 += beta_dirs[l][h][d] * beta_dirs[l][h][d];
                    norm2 = sqrtf(norm2 + 1e-12f);
                    for (int d = 0; d < actual_d_k; d++) beta_dirs[l][h][d] /= norm2;
                }
            }
        }

        auto compacted_buf = build_compacted_state(kv_state, selected, cv_all,
                                                    mi.n_head_kv, actual_d_k, actual_d_v,
                                                    mi.n_pos_per_embd,
                                                    beta_all, beta_dirs);

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
                double kl = kl_divergence(ref_probs[i + 1], comp_probs);
                sum_kl += kl;
                if (kl > max_kl) max_kl = kl;
            }
        }

        double comp_ppl = exp(-comp_log_prob / n_eval);
        double ppl_ratio = comp_ppl / ref_ppl;
        double avg_kl = sum_kl / n_eval;

        printf("  %-8.0f%%  %4d/%-3d  %12.4f  %12.4f  %12.6f  %12.6f  %10.1f\n",
               ratio * 100, t, T, comp_ppl, ppl_ratio, avg_kl, max_kl, compact_ms);

        llama_batch_free(eval_batch);
        kv_compact_result_free(&compact_result);
    }

    // ---- Step 4: Generation comparison ----
    printf("\n  [4/4] Generation comparison...\n\n");

    const int n_gen = 50;

    // Generate with full cache
    llama_memory_seq_rm(mem, 0, -1, -1);
    {
        llama_batch pb = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++)
            common_batch_add(pb, prompt_tokens[i], i, {0}, (i == n_prompt - 1));
        llama_decode(mi.ctx, pb);
        llama_batch_free(pb);
    }

    std::string full_gen;
    {
        // Decode first continuation token as seed (same as compacted path)
        llama_batch sb = llama_batch_init(1, 0, 1);
        common_batch_add(sb, cont_tokens[0], n_prompt, {0}, true);
        llama_decode(mi.ctx, sb);
        llama_batch_free(sb);

        full_gen += common_token_to_piece(mi.vocab, cont_tokens[0]);

        common_sampler * smpl = common_sampler_init(mi.model, params.sampling);
        llama_batch gb = llama_batch_init(1, 0, 1);
        for (int i = 0; i < n_gen - 1; i++) {
            llama_token id = common_sampler_sample(smpl, mi.ctx, -1);
            if (llama_vocab_is_eog(mi.vocab, id)) break;
            full_gen += common_token_to_piece(mi.vocab, id);
            common_sampler_accept(smpl, id, true);
            common_batch_clear(gb);
            common_batch_add(gb, id, n_prompt + 1 + i, {0}, true);
            if (llama_decode(mi.ctx, gb) != 0) break;
        }
        common_sampler_free(smpl);
        llama_batch_free(gb);
    }

    // Generate with 50% compacted cache
    {
        llama_memory_seq_rm(mem, 0, -1, -1);
        llama_batch pb = llama_batch_init(n_prompt, 0, 1);
        for (int i = 0; i < n_prompt; i++)
            common_batch_add(pb, prompt_tokens[i], i, {0}, (i == n_prompt - 1));
        llama_decode(mi.ctx, pb);
        llama_batch_free(pb);
    }

    // Save, compact at 50%, reload
    {
        size_t ss = llama_state_seq_get_size(mi.ctx, 0);
        std::vector<uint8_t> sb(ss);
        llama_state_seq_get_data(mi.ctx, sb.data(), sb.size(), 0);

        parsed_kv_state ks2;
        ks2.parse(sb.data(), ss, mi.n_pos_per_embd);

        kv_compact_params p50 = kv_compact_params_default();
        p50.target_ratio = 0.5f;
        p50.use_cheap_qref = 1;

        kv_compact_result r50 = {};
        kv_compact(ks2.streams[0].layers[0].K.data(),
                   ks2.streams[0].layers[0].V.data(), NULL,
                   n_prompt, 0, mi.n_head_kv, actual_d_k, actual_d_v,
                   &p50, &r50);

        std::vector<int> sel50(r50.selected_indices, r50.selected_indices + r50.t);
        const auto & sd2 = ks2.streams[0];
        int n_embd_k2 = mi.n_head_kv * actual_d_k;

        // Full NNLS beta + LS value refit per layer
        std::vector<std::vector<std::vector<float>>> cv50(sd2.n_layer);
        std::vector<std::vector<std::vector<float>>> beta50(sd2.n_layer);
        std::vector<std::vector<std::vector<float>>> dirs50(sd2.n_layer);

        for (uint32_t l = 0; l < sd2.n_layer; l++) {
            const auto & ld = sd2.layers[l];

            if (l == 0) {
                // Use layer 0 results directly
                cv50[l].resize(mi.n_head_kv);
                beta50[l].resize(mi.n_head_kv);
                dirs50[l].resize(mi.n_head_kv);
                for (int h = 0; h < mi.n_head_kv; h++) {
                    cv50[l][h].assign(r50.C_v[h], r50.C_v[h] + r50.t * actual_d_v);
                    beta50[l][h].assign(r50.beta[h], r50.beta[h] + r50.t);
                    // Beta direction: normalized mean Q_ref
                    dirs50[l][h].resize(actual_d_k);
                    int nqr = std::min(n_prompt / 2, 64);
                    if (nqr < 4) nqr = 4;
                    std::fill(dirs50[l][h].begin(), dirs50[l][h].end(), 0.0f);
                    for (int qi = 0; qi < nqr; qi++) {
                        float frac = (float)(qi + 1) / (float)(nqr + 1);
                        frac = frac * frac;
                        int pos = std::min((int)(frac * (n_prompt - 1)), n_prompt - 1);
                        const float * k = ld.K.data() + pos * n_embd_k2 + h * actual_d_k;
                        for (int d = 0; d < actual_d_k; d++) dirs50[l][h][d] += k[d];
                    }
                    float norm = 0.0f;
                    for (int d = 0; d < actual_d_k; d++) norm += dirs50[l][h][d] * dirs50[l][h][d];
                    norm = sqrtf(norm + 1e-12f);
                    for (int d = 0; d < actual_d_k; d++) dirs50[l][h][d] /= norm;
                }
            } else {
                auto lr = compact_layer_with_shared_selection(
                    ld.K.data(), ld.V.data(), sel50,
                    n_prompt, mi.n_head_kv, actual_d_k, actual_d_v, p50);
                cv50[l] = std::move(lr.C_v);
                beta50[l] = std::move(lr.beta);

                dirs50[l].resize(mi.n_head_kv);
                int nqr = std::min(n_prompt / 2, 64);
                if (nqr < 4) nqr = 4;
                for (int h = 0; h < mi.n_head_kv; h++) {
                    dirs50[l][h].resize(actual_d_k);
                    std::fill(dirs50[l][h].begin(), dirs50[l][h].end(), 0.0f);
                    for (int qi = 0; qi < nqr; qi++) {
                        float frac = (float)(qi + 1) / (float)(nqr + 1);
                        frac = frac * frac;
                        int pos = std::min((int)(frac * (n_prompt - 1)), n_prompt - 1);
                        const float * k = ld.K.data() + pos * n_embd_k2 + h * actual_d_k;
                        for (int d = 0; d < actual_d_k; d++) dirs50[l][h][d] += k[d];
                    }
                    float norm = 0.0f;
                    for (int d = 0; d < actual_d_k; d++) norm += dirs50[l][h][d] * dirs50[l][h][d];
                    norm = sqrtf(norm + 1e-12f);
                    for (int d = 0; d < actual_d_k; d++) dirs50[l][h][d] /= norm;
                }
            }
        }

        auto cb = build_compacted_state(ks2, sel50, cv50, mi.n_head_kv, actual_d_k, actual_d_v,
                                         mi.n_pos_per_embd, beta50, dirs50);

        llama_memory_seq_rm(mem, 0, -1, -1);
        llama_state_seq_set_data(mi.ctx, cb.data(), cb.size(), 0);

        kv_compact_result_free(&r50);
    }

    llama_pos pos_max2 = llama_memory_seq_pos_max(mem, 0);
    std::string compact_gen;
    {
        // Decode a "seed" continuation token at pos_max+1 to prime logits
        // (we can't re-decode pos_max since it's already in the cache)
        llama_batch pb = llama_batch_init(1, 0, 1);
        common_batch_add(pb, cont_tokens[0], pos_max2 + 1, {0}, true);
        llama_decode(mi.ctx, pb);
        llama_batch_free(pb);

        compact_gen += common_token_to_piece(mi.vocab, cont_tokens[0]);

        common_sampler * smpl = common_sampler_init(mi.model, params.sampling);
        llama_batch gb = llama_batch_init(1, 0, 1);
        for (int i = 0; i < n_gen - 1; i++) {
            llama_token id = common_sampler_sample(smpl, mi.ctx, -1);
            if (llama_vocab_is_eog(mi.vocab, id)) break;
            compact_gen += common_token_to_piece(mi.vocab, id);
            common_sampler_accept(smpl, id, true);
            common_batch_clear(gb);
            common_batch_add(gb, id, pos_max2 + 2 + i, {0}, true);
            if (llama_decode(mi.ctx, gb) != 0) break;
        }
        common_sampler_free(smpl);
        llama_batch_free(gb);
    }

    printf("  Full cache:  \"%.100s...\"\n", full_gen.c_str());
    printf("  50%% compact: \"%.100s...\"\n", compact_gen.c_str());

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

    llama_backend_free();
    return 0;
}
