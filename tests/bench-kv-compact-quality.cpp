// Quality benchmarks for KV cache compaction
//
// Measures preservation quality across compression ratios using metrics
// that go beyond basic cosine similarity:
//
//   1. Perplexity preservation  — simulated next-token log-prob shift
//   2. KL divergence            — output distribution divergence
//   3. Attention mass error     — partition function preservation (paper §3.2)
//   4. Throughput scaling       — tokens/sec vs T and compression ratio
//
// These are best-practice benchmarks for evaluating KV cache compression
// methods. No model weights required — uses synthetic data that exercises
// the same code paths as real inference.

#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#include "kv-compact-api.h"
#include "kv-compact-math.h"

using clock_type = std::chrono::high_resolution_clock;

// ============================================================================
// Data generation
// ============================================================================

static void gen_data(float * out, int n, int seed) {
    for (int i = 0; i < n; i++) {
        out[i] = sinf((float)(i * 7 + seed) * 0.31f)
               + 0.3f * cosf((float)(i * 3 + seed + 17) * 0.53f);
    }
}

// Generate a random "vocabulary projection" matrix W_vocab [d_v × vocab_size]
// Used to simulate logit computation: logits = attn_output @ W_vocab
static void gen_vocab_proj(float * out, int d_v, int vocab_size, int seed) {
    for (int i = 0; i < d_v * vocab_size; i++) {
        out[i] = 0.1f * sinf((float)(i * 13 + seed) * 0.17f)
               + 0.05f * cosf((float)(i * 11 + seed + 31) * 0.41f);
    }
}

// ============================================================================
// Helper: compute attention output for a single query
// ============================================================================

// Original attention output: softmax(q·K^T/√d) · V
static void compute_original_output(
        const float * q, const float * K, const float * V,
        int T, int d_k, int d_v, float * out) {
    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    // Compute scores
    std::vector<float> scores(T);
    for (int j = 0; j < T; j++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) dot += q[d] * K[j * d_k + d];
        scores[j] = dot * inv_sqrt_dk;
    }

    // Softmax
    softmax_rows(scores.data(), 1, T);

    // Weighted sum of V
    memset(out, 0, d_v * sizeof(float));
    for (int j = 0; j < T; j++) {
        for (int d = 0; d < d_v; d++) {
            out[d] += scores[j] * V[j * d_v + d];
        }
    }
}

// Compacted attention output: softmax(q·C_k^T/√d + β) · C_v
static void compute_compacted_output(
        const float * q, const float * K, const float * beta,
        const float * C_v, const int * selected, int t,
        int d_k, int d_v, float * out) {
    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    std::vector<float> scores(t);
    for (int j = 0; j < t; j++) {
        float dot = 0.0f;
        const float * k = K + selected[j] * d_k;
        for (int d = 0; d < d_k; d++) dot += q[d] * k[d];
        scores[j] = dot * inv_sqrt_dk + beta[j];
    }

    softmax_rows(scores.data(), 1, t);

    memset(out, 0, d_v * sizeof(float));
    for (int j = 0; j < t; j++) {
        for (int d = 0; d < d_v; d++) {
            out[d] += scores[j] * C_v[j * d_v + d];
        }
    }
}

// ============================================================================
// Benchmark 1: Perplexity Preservation
// ============================================================================
// Simulates next-token prediction by projecting attention output through
// a vocabulary matrix, computing log-probabilities, then comparing
// perplexity between original and compacted caches.
//
// Lower perplexity ratio (compacted/original) = better preservation.
// A ratio of 1.0 means perfect preservation.

static void bench_perplexity_preservation() {
    printf("=== Perplexity Preservation ===\n\n");

    const int n_head_kv = 4, d_k = 64, d_v = 64;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;
    const int vocab_size = 256;  // small simulated vocab
    const int n_eval_queries = 64;

    // Vocabulary projection: maps d_v → vocab_size per head
    std::vector<float> W_vocab(d_v * vocab_size);
    gen_vocab_proj(W_vocab.data(), d_v, vocab_size, 42);

    int T_sizes[] = {128, 256, 512, 1024, 2048, 4096, 10240};
    float ratios[] = {0.5f, 0.2f, 0.1f};

    printf("  %-6s  %-8s  %12s  %12s  %12s\n",
           "T", "ratio", "ppl_orig", "ppl_compact", "ppl_ratio");
    printf("  %-6s  %-8s  %12s  %12s  %12s\n",
           "------", "--------", "------------", "------------", "------------");

    for (int T : T_sizes) {
        std::vector<float> K(T * n_embd_k), V(T * n_embd_v);
        std::vector<float> Q_ref(n_eval_queries * n_embd_k);
        std::vector<float> Q_eval(n_eval_queries * n_embd_k);

        gen_data(K.data(), T * n_embd_k, 100 + T);
        gen_data(V.data(), T * n_embd_v, 200 + T);
        gen_data(Q_ref.data(), n_eval_queries * n_embd_k, 300 + T);
        gen_data(Q_eval.data(), n_eval_queries * n_embd_k, 400 + T);

        for (float ratio : ratios) {
            kv_compact_params p = kv_compact_params_default();
            p.target_ratio = ratio;

            kv_compact_result result = {};
            int rc = kv_compact(K.data(), V.data(), Q_ref.data(),
                                T, n_eval_queries, n_head_kv, d_k, d_v,
                                &p, &result);
            assert(rc == 0);

            // Compute perplexity for head 0 (representative)
            double log_prob_orig = 0.0, log_prob_comp = 0.0;
            int h = 0;

            for (int qi = 0; qi < n_eval_queries; qi++) {
                const float * q = Q_eval.data() + qi * n_embd_k + h * d_k;

                // Extract per-head K, V for original
                std::vector<float> K_h(T * d_k), V_h(T * d_v);
                for (int j = 0; j < T; j++) {
                    memcpy(K_h.data() + j * d_k, K.data() + j * n_embd_k + h * d_k, d_k * sizeof(float));
                    memcpy(V_h.data() + j * d_v, V.data() + j * n_embd_v + h * d_v, d_v * sizeof(float));
                }

                // Original output
                std::vector<float> orig_out(d_v);
                compute_original_output(q, K_h.data(), V_h.data(), T, d_k, d_v, orig_out.data());

                // Compacted output
                std::vector<float> comp_out(d_v);
                compute_compacted_output(q, K_h.data(), result.beta[h],
                                         result.C_v[h], result.selected_indices,
                                         result.t, d_k, d_v, comp_out.data());

                // Project to vocab logits: logits = out @ W_vocab^T  [vocab_size]
                std::vector<float> logits_orig(vocab_size), logits_comp(vocab_size);
                for (int v = 0; v < vocab_size; v++) {
                    float dot_o = 0.0f, dot_c = 0.0f;
                    for (int d = 0; d < d_v; d++) {
                        dot_o += orig_out[d] * W_vocab[d * vocab_size + v];
                        dot_c += comp_out[d] * W_vocab[d * vocab_size + v];
                    }
                    logits_orig[v] = dot_o;
                    logits_comp[v] = dot_c;
                }

                // Softmax to get probabilities
                softmax_rows(logits_orig.data(), 1, vocab_size);
                softmax_rows(logits_comp.data(), 1, vocab_size);

                // Use the original's argmax as the "correct" token
                int target = 0;
                for (int v = 1; v < vocab_size; v++) {
                    if (logits_orig[v] > logits_orig[target]) target = v;
                }

                // Log probability of the target token under each distribution
                log_prob_orig += log(logits_orig[target] + 1e-12);
                log_prob_comp += log(logits_comp[target] + 1e-12);
            }

            double ppl_orig = exp(-log_prob_orig / n_eval_queries);
            double ppl_comp = exp(-log_prob_comp / n_eval_queries);
            double ppl_ratio = ppl_comp / ppl_orig;

            printf("  %-6d  %-8.0f%%  %12.4f  %12.4f  %12.4f\n",
                   T, ratio * 100, ppl_orig, ppl_comp, ppl_ratio);

            // Sanity: perplexity should be finite and ratio shouldn't explode
            assert(std::isfinite(ppl_orig));
            assert(std::isfinite(ppl_comp));
            assert(ppl_ratio < 100.0);  // very loose bound

            kv_compact_result_free(&result);
        }
    }
    printf("\n");
}

// ============================================================================
// Benchmark 2: KL Divergence
// ============================================================================
// Measures the KL divergence D_KL(P_orig || P_compact) of the output
// distribution after projecting through a vocabulary matrix.
//
// KL divergence quantifies information lost when approximating the original
// distribution with the compacted one. Lower = better. Zero = identical.

static void bench_kl_divergence() {
    printf("=== KL Divergence ===\n\n");

    const int T = 256, n_head_kv = 4, d_k = 64, d_v = 64;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;
    const int vocab_size = 256;
    const int n_eval = 64;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v);
    std::vector<float> Q_ref(n_eval * n_embd_k), Q_eval(n_eval * n_embd_k);
    std::vector<float> W_vocab(d_v * vocab_size);

    gen_data(K.data(), T * n_embd_k, 500);
    gen_data(V.data(), T * n_embd_v, 600);
    gen_data(Q_ref.data(), n_eval * n_embd_k, 700);
    gen_data(Q_eval.data(), n_eval * n_embd_k, 800);
    gen_vocab_proj(W_vocab.data(), d_v, vocab_size, 900);

    float ratios[] = {0.8f, 0.5f, 0.2f, 0.1f, 0.05f};

    printf("  %-8s  %12s  %12s  %12s\n",
           "ratio", "avg_KL", "max_KL", "KL<0.01");
    printf("  %-8s  %12s  %12s  %12s\n",
           "--------", "------------", "------------", "--------");

    for (float ratio : ratios) {
        kv_compact_params p = kv_compact_params_default();
        p.target_ratio = ratio;

        kv_compact_result result = {};
        int rc = kv_compact(K.data(), V.data(), Q_ref.data(),
                            T, n_eval, n_head_kv, d_k, d_v, &p, &result);
        assert(rc == 0);

        double sum_kl = 0.0, max_kl = 0.0;
        int low_kl_count = 0;

        // Evaluate across heads 0-1 and all queries
        for (int h = 0; h < 2; h++) {
            std::vector<float> K_h(T * d_k), V_h(T * d_v);
            for (int j = 0; j < T; j++) {
                memcpy(K_h.data() + j * d_k, K.data() + j * n_embd_k + h * d_k, d_k * sizeof(float));
                memcpy(V_h.data() + j * d_v, V.data() + j * n_embd_v + h * d_v, d_v * sizeof(float));
            }

            for (int qi = 0; qi < n_eval; qi++) {
                const float * q = Q_eval.data() + qi * n_embd_k + h * d_k;

                std::vector<float> orig_out(d_v), comp_out(d_v);
                compute_original_output(q, K_h.data(), V_h.data(), T, d_k, d_v, orig_out.data());
                compute_compacted_output(q, K_h.data(), result.beta[h],
                                         result.C_v[h], result.selected_indices,
                                         result.t, d_k, d_v, comp_out.data());

                // Project to logits
                std::vector<float> logits_o(vocab_size), logits_c(vocab_size);
                for (int v = 0; v < vocab_size; v++) {
                    float dot_o = 0.0f, dot_c = 0.0f;
                    for (int d = 0; d < d_v; d++) {
                        dot_o += orig_out[d] * W_vocab[d * vocab_size + v];
                        dot_c += comp_out[d] * W_vocab[d * vocab_size + v];
                    }
                    logits_o[v] = dot_o;
                    logits_c[v] = dot_c;
                }

                softmax_rows(logits_o.data(), 1, vocab_size);
                softmax_rows(logits_c.data(), 1, vocab_size);

                // KL(P || Q) = sum_i P(i) * log(P(i) / Q(i))
                double kl = 0.0;
                for (int v = 0; v < vocab_size; v++) {
                    double p_v = logits_o[v] + 1e-12;
                    double q_v = logits_c[v] + 1e-12;
                    kl += p_v * log(p_v / q_v);
                }
                if (kl < 0.0) kl = 0.0;  // numerical floor

                sum_kl += kl;
                if (kl > max_kl) max_kl = kl;
                if (kl < 0.01) low_kl_count++;
            }
        }

        int total = 2 * n_eval;
        double avg_kl = sum_kl / total;

        printf("  %-8.0f%%  %12.6f  %12.6f  %8d/%d\n",
               ratio * 100, avg_kl, max_kl, low_kl_count, total);

        assert(std::isfinite(avg_kl));

        kv_compact_result_free(&result);
    }
    printf("\n");
}

// ============================================================================
// Benchmark 3: Attention Mass Preservation
// ============================================================================
// Directly measures how well the partition function (total attention mass)
// is preserved after compaction. This is the core invariant from Section 3.2
// of the paper — if mass is wrong, the model over/under-attends to cached
// context vs. new tokens during generation.
//
// Reports relative mass error: |mass_compact - mass_orig| / mass_orig

static void bench_mass_preservation() {
    printf("=== Attention Mass Preservation (Section 3.2) ===\n\n");

    const int T = 256, n_head_kv = 4, d_k = 64, d_v = 64;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;
    const int n_eval = 64;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v);
    std::vector<float> Q_ref(n_eval * n_embd_k), Q_eval(n_eval * n_embd_k);

    gen_data(K.data(), T * n_embd_k, 1000);
    gen_data(V.data(), T * n_embd_v, 1100);
    gen_data(Q_ref.data(), n_eval * n_embd_k, 1200);
    gen_data(Q_eval.data(), n_eval * n_embd_k, 1300);

    float ratios[] = {0.8f, 0.5f, 0.2f, 0.1f, 0.05f};

    printf("  %-8s  %14s  %14s  %14s\n",
           "ratio", "avg_rel_error", "max_rel_error", "< 1%% error");
    printf("  %-8s  %14s  %14s  %14s\n",
           "--------", "--------------", "--------------", "-----------");

    for (float ratio : ratios) {
        kv_compact_params p = kv_compact_params_default();
        p.target_ratio = ratio;

        kv_compact_result result = {};
        int rc = kv_compact(K.data(), V.data(), Q_ref.data(),
                            T, n_eval, n_head_kv, d_k, d_v, &p, &result);
        assert(rc == 0);

        double sum_rel_err = 0.0, max_rel_err = 0.0;
        int low_err_count = 0;
        int total = 0;

        float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

        for (int h = 0; h < n_head_kv; h++) {
            for (int qi = 0; qi < n_eval; qi++) {
                const float * q = Q_eval.data() + qi * n_embd_k + h * d_k;

                // Original mass: sum_j exp(q·k_j / √d)
                // Use max-shift for stability
                float max_score = -1e30f;
                std::vector<float> scores(T);
                for (int j = 0; j < T; j++) {
                    float dot = 0.0f;
                    const float * k = K.data() + j * n_embd_k + h * d_k;
                    for (int d = 0; d < d_k; d++) dot += q[d] * k[d];
                    scores[j] = dot * inv_sqrt_dk;
                    if (scores[j] > max_score) max_score = scores[j];
                }
                double mass_orig = 0.0;
                for (int j = 0; j < T; j++) {
                    mass_orig += exp(scores[j] - max_score);
                }

                // Compacted mass: sum_j exp(q·k_sel[j] / √d + beta_j)
                double mass_comp = 0.0;
                for (int j = 0; j < result.t; j++) {
                    float dot = 0.0f;
                    const float * k = K.data() + result.selected_indices[j] * n_embd_k + h * d_k;
                    for (int d = 0; d < d_k; d++) dot += q[d] * k[d];
                    float score = dot * inv_sqrt_dk + result.beta[h][j];
                    mass_comp += exp(score - max_score);
                }

                double rel_err = fabs(mass_comp - mass_orig) / (mass_orig + 1e-12);
                sum_rel_err += rel_err;
                if (rel_err > max_rel_err) max_rel_err = rel_err;
                if (rel_err < 0.01) low_err_count++;
                total++;
            }
        }

        double avg_rel_err = sum_rel_err / total;

        printf("  %-8.0f%%  %14.6f  %14.6f  %10d/%d\n",
               ratio * 100, avg_rel_err, max_rel_err, low_err_count, total);

        assert(std::isfinite(avg_rel_err));

        kv_compact_result_free(&result);
    }
    printf("\n");
}

// ============================================================================
// Benchmark 4: Throughput Scaling
// ============================================================================
// Measures compaction throughput (tokens processed per second) across
// different context lengths and compression ratios. Helps identify
// computational bottlenecks and verify O(n_q·T·d_k) scaling.

static void bench_throughput_scaling() {
    printf("=== Throughput Scaling ===\n\n");

    const int n_head_kv = 4, d_k = 64, d_v = 64;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;
    const int n_q = 64;

    // NOTE: T>10k at 50% retention causes OOM in LS normal equations
    // (t^2 matrix = 10GB+ at t=51200). Needs iterative solver or GPU for 100k.
    int T_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 10240};
    float ratios[] = {0.5f, 0.2f};

    printf("  %-6s  %-8s  %8s  %12s  %14s\n",
           "T", "ratio", "t", "time_ms", "tokens/sec");
    printf("  %-6s  %-8s  %8s  %12s  %14s\n",
           "------", "--------", "--------", "------------", "--------------");

    for (int T : T_sizes) {
        std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
        gen_data(K.data(), T * n_embd_k, 2000 + T);
        gen_data(V.data(), T * n_embd_v, 3000 + T);
        gen_data(Q.data(), n_q * n_embd_k, 4000 + T);

        for (float ratio : ratios) {
            kv_compact_params p = kv_compact_params_default();
            p.target_ratio = ratio;

            // Warmup (skip for large T to save time)
            if (T <= 2048) {
                kv_compact_result warmup = {};
                kv_compact(K.data(), V.data(), Q.data(),
                           T, n_q, n_head_kv, d_k, d_v, &p, &warmup);
                kv_compact_result_free(&warmup);
            }

            // Timed run (fewer runs for large T)
            const int n_runs = (T <= 2048) ? 3 : 1;
            double total_ms = 0.0;
            int final_t = 0;

            for (int run = 0; run < n_runs; run++) {
                kv_compact_result result = {};
                auto t0 = clock_type::now();
                kv_compact(K.data(), V.data(), Q.data(),
                           T, n_q, n_head_kv, d_k, d_v, &p, &result);
                double ms = std::chrono::duration<double, std::milli>(
                    clock_type::now() - t0).count();
                total_ms += ms;
                final_t = result.t;
                kv_compact_result_free(&result);
            }

            double avg_ms = total_ms / n_runs;
            double tokens_per_sec = T / (avg_ms / 1000.0);

            printf("  %-6d  %-8.0f%%  %8d  %12.2f  %14.0f\n",
                   T, ratio * 100, final_t, avg_ms, tokens_per_sec);

            assert(avg_ms > 0.0);
        }
    }
    printf("\n");
}

// ============================================================================
// Benchmark 5: Eviction vs Compaction Comparison
// ============================================================================
// Directly compares naive token eviction (drop lowest-attention tokens,
// keep original V) against full compaction (NNLS beta + LS value refit).
// This demonstrates the MSE improvement factor reported in the paper.

static void bench_eviction_vs_compaction() {
    printf("=== Eviction vs Compaction (quality comparison) ===\n\n");

    const int T = 256, n_head_kv = 4, d_k = 64, d_v = 64;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;
    const int n_q = 64;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    std::vector<float> Q_eval(n_q * n_embd_k);

    gen_data(K.data(), T * n_embd_k, 5000);
    gen_data(V.data(), T * n_embd_v, 5100);
    gen_data(Q.data(), n_q * n_embd_k, 5200);
    gen_data(Q_eval.data(), n_q * n_embd_k, 5300);

    float ratios[] = {0.5f, 0.2f, 0.1f};

    printf("  %-8s  %12s  %12s  %12s  %12s  %12s\n",
           "ratio", "evict_cos", "compact_cos", "evict_mse", "compact_mse", "MSE_ratio");
    printf("  %-8s  %12s  %12s  %12s  %12s  %12s\n",
           "--------", "------------", "------------", "------------", "------------", "------------");

    for (float ratio : ratios) {
        // Full compaction
        kv_compact_params p = kv_compact_params_default();
        p.target_ratio = ratio;

        kv_compact_result result = {};
        int rc = kv_compact(K.data(), V.data(), Q.data(),
                            T, n_q, n_head_kv, d_k, d_v, &p, &result);
        assert(rc == 0);

        // Evaluate both eviction and compaction on head 0
        int h = 0;
        float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

        double evict_cos_sum = 0.0, evict_mse_sum = 0.0;
        double comp_cos_sum = 0.0, comp_mse_sum = 0.0;

        for (int qi = 0; qi < n_q; qi++) {
            const float * q = Q_eval.data() + qi * n_embd_k + h * d_k;

            // Original output
            std::vector<float> orig_scores(T);
            for (int j = 0; j < T; j++) {
                float dot = 0.0f;
                const float * k = K.data() + j * n_embd_k + h * d_k;
                for (int d = 0; d < d_k; d++) dot += q[d] * k[d];
                orig_scores[j] = dot * inv_sqrt_dk;
            }
            softmax_rows(orig_scores.data(), 1, T);
            std::vector<float> orig_out(d_v, 0.0f);
            for (int j = 0; j < T; j++) {
                const float * v = V.data() + j * n_embd_v + h * d_v;
                for (int d = 0; d < d_v; d++) orig_out[d] += orig_scores[j] * v[d];
            }

            // Eviction output: same selected keys, no beta, original V
            std::vector<float> evict_scores(result.t);
            for (int j = 0; j < result.t; j++) {
                float dot = 0.0f;
                const float * k = K.data() + result.selected_indices[j] * n_embd_k + h * d_k;
                for (int d = 0; d < d_k; d++) dot += q[d] * k[d];
                evict_scores[j] = dot * inv_sqrt_dk;  // no beta
            }
            softmax_rows(evict_scores.data(), 1, result.t);
            std::vector<float> evict_out(d_v, 0.0f);
            for (int j = 0; j < result.t; j++) {
                const float * v = V.data() + result.selected_indices[j] * n_embd_v + h * d_v;
                for (int d = 0; d < d_v; d++) evict_out[d] += evict_scores[j] * v[d];
            }

            // Compaction output
            std::vector<float> comp_scores(result.t);
            for (int j = 0; j < result.t; j++) {
                float dot = 0.0f;
                const float * k = K.data() + result.selected_indices[j] * n_embd_k + h * d_k;
                for (int d = 0; d < d_k; d++) dot += q[d] * k[d];
                comp_scores[j] = dot * inv_sqrt_dk + result.beta[h][j];
            }
            softmax_rows(comp_scores.data(), 1, result.t);
            std::vector<float> comp_out(d_v, 0.0f);
            for (int j = 0; j < result.t; j++) {
                for (int d = 0; d < d_v; d++)
                    comp_out[d] += comp_scores[j] * result.C_v[h][j * d_v + d];
            }

            // Cosine similarity
            auto cosine_sim = [&](const float * a, const float * b, int n) {
                float dot = 0, na = 0, nb = 0;
                for (int d = 0; d < n; d++) {
                    dot += a[d] * b[d]; na += a[d] * a[d]; nb += b[d] * b[d];
                }
                return dot / (sqrtf(na * nb) + 1e-8f);
            };

            evict_cos_sum += cosine_sim(orig_out.data(), evict_out.data(), d_v);
            comp_cos_sum += cosine_sim(orig_out.data(), comp_out.data(), d_v);

            // MSE
            double e_mse = 0, c_mse = 0;
            for (int d = 0; d < d_v; d++) {
                double de = orig_out[d] - evict_out[d];
                double dc = orig_out[d] - comp_out[d];
                e_mse += de * de;
                c_mse += dc * dc;
            }
            evict_mse_sum += e_mse / d_v;
            comp_mse_sum += c_mse / d_v;
        }

        double evict_cos = evict_cos_sum / n_q;
        double comp_cos = comp_cos_sum / n_q;
        double evict_mse = evict_mse_sum / n_q;
        double comp_mse = comp_mse_sum / n_q;
        double mse_ratio = (comp_mse > 1e-20) ? evict_mse / comp_mse : INFINITY;

        printf("  %-8.0f%%  %12.6f  %12.6f  %12.2e  %12.2e  %12.0fx\n",
               ratio * 100, evict_cos, comp_cos, evict_mse, comp_mse, mse_ratio);

        // Compaction should always beat eviction
        assert(comp_mse <= evict_mse * 1.1);  // allow small numerical margin

        kv_compact_result_free(&result);
    }
    printf("\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("bench-kv-compact-quality\n");
    printf("========================\n\n");

    bench_perplexity_preservation();
    bench_kl_divergence();
    bench_mass_preservation();
    bench_throughput_scaling();
    bench_eviction_vs_compaction();

    printf("All quality benchmarks completed.\n");
    return 0;
}
