// Synthetic benchmark: KV compaction quality across all modes
//
// Uses Qwen3.5-0.8B dimensions (full-attention layers only):
//   n_head_q = 8, n_head_kv = 2, d_head = 256
//   6 full-attention layers out of 24 total
//
// Measures per mode × beta fitting × compression ratio:
//   1. Attention output MSE:  ||softmax(qC_k+β)C_v - softmax(qK)V||²
//   2. Mass error:            |Σexp(qC_k+β) - Σexp(qK)| / Σexp(qK)
//   3. Wall-clock time (µs)
//
// Data is pseudo-random (deterministic LCG) to ensure reproducibility.

#include "kv-compact-math.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// Qwen3.5-0.8B full-attention layer dimensions
static constexpr int N_HEAD_Q   = 8;
static constexpr int N_HEAD_KV  = 2;
static constexpr int D_HEAD     = 256;  // head_dim for full attention
static constexpr int GQA_RATIO  = N_HEAD_Q / N_HEAD_KV;  // 4 query heads per KV head

// Benchmark parameters
static constexpr int N_Q_REF    = 32;   // reference queries
static constexpr int N_Q_EVAL   = 64;   // evaluation queries (unseen)

// Deterministic pseudo-random float in [-1, 1]
static float lcg_float(uint32_t & seed) {
    seed = seed * 1103515245u + 12345u;
    return (float)((int)(seed >> 16) % 2000) / 1000.0f - 1.0f;
}

// Generate pseudo-random data with controlled structure
static void generate_data(
        std::vector<float> & K, std::vector<float> & V,
        std::vector<float> & Q_ref, std::vector<float> & Q_eval,
        int T, int n_head_kv, int d_k, int d_v,
        int n_q_ref, int n_q_eval) {

    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;

    K.resize(T * n_embd_k);
    V.resize(T * n_embd_v);
    Q_ref.resize(n_q_ref * n_embd_k);
    Q_eval.resize(n_q_eval * n_embd_k);

    uint32_t seed = 42;

    // Keys: mix of structured (some dominant directions) + noise
    for (int i = 0; i < T; i++) {
        for (int h = 0; h < n_head_kv; h++) {
            // Add some structure: token i has a "topic" based on position
            int topic = (i * 7) % 20;
            for (int d = 0; d < d_k; d++) {
                float structured = sinf((float)(topic * d_k + d) * 0.1f) * 0.5f;
                float noise = lcg_float(seed) * 0.3f;
                K[i * n_embd_k + h * d_k + d] = structured + noise;
            }
        }
    }

    // Values: correlated with key topics
    for (int i = 0; i < T; i++) {
        for (int h = 0; h < n_head_kv; h++) {
            int topic = (i * 7) % 20;
            for (int d = 0; d < d_v; d++) {
                float structured = cosf((float)(topic * d_v + d) * 0.15f) * 0.4f;
                float noise = lcg_float(seed) * 0.2f;
                V[i * n_embd_v + h * d_v + d] = structured + noise;
            }
        }
    }

    // Reference queries: used for optimization
    for (int i = 0; i < n_q_ref * n_embd_k; i++) {
        Q_ref[i] = lcg_float(seed) * 0.5f;
    }

    // Evaluation queries: unseen, tests generalization
    seed = 12345;  // different seed
    for (int i = 0; i < n_q_eval * n_embd_k; i++) {
        Q_eval[i] = lcg_float(seed) * 0.5f;
    }
}

// Compute ground-truth attention output for one head:
//   output[qi] = softmax(Q[qi] · K^T / sqrt(d_k)) · V
// Also returns total mass per query.
static void compute_ground_truth(
        const float * K_head, const float * V_head, const float * Q_head,
        int T, int n_q, int d_k, int d_v,
        std::vector<float> & output, std::vector<float> & mass) {

    output.resize(n_q * d_v, 0.0f);
    mass.resize(n_q, 0.0f);
    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    for (int qi = 0; qi < n_q; qi++) {
        // Compute scores
        std::vector<float> scores(T);
        float max_score = -1e30f;
        for (int j = 0; j < T; j++) {
            float dot = 0.0f;
            for (int d = 0; d < d_k; d++) {
                dot += Q_head[qi * d_k + d] * K_head[j * d_k + d];
            }
            scores[j] = dot * inv_sqrt_dk;
            if (scores[j] > max_score) max_score = scores[j];
        }

        // Softmax + mass
        float sum_exp = 0.0f;
        std::vector<float> weights(T);
        for (int j = 0; j < T; j++) {
            weights[j] = expf(scores[j] - max_score);
            sum_exp += weights[j];
        }
        mass[qi] = sum_exp * expf(max_score);

        for (int j = 0; j < T; j++) {
            weights[j] /= sum_exp;
        }

        // Output = weights · V
        for (int j = 0; j < T; j++) {
            for (int d = 0; d < d_v; d++) {
                output[qi * d_v + d] += weights[j] * V_head[j * d_v + d];
            }
        }
    }
}

// Compute compacted attention output for one head:
//   output[qi] = softmax(Q[qi] · C_k^T / sqrt(d_k) + beta) · C_v
// Also returns total mass per query.
static void compute_compacted_output(
        const float * C_k, const float * C_v, const float * beta,
        const float * Q_head,
        int t, int n_q, int d_k, int d_v,
        std::vector<float> & output, std::vector<float> & mass) {

    output.resize(n_q * d_v, 0.0f);
    mass.resize(n_q, 0.0f);
    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    for (int qi = 0; qi < n_q; qi++) {
        std::vector<float> scores(t);
        float max_score = -1e30f;
        for (int j = 0; j < t; j++) {
            float dot = 0.0f;
            for (int d = 0; d < d_k; d++) {
                dot += Q_head[qi * d_k + d] * C_k[j * d_k + d];
            }
            scores[j] = dot * inv_sqrt_dk + beta[j];
            if (scores[j] > max_score) max_score = scores[j];
        }

        float sum_exp = 0.0f;
        std::vector<float> weights(t);
        for (int j = 0; j < t; j++) {
            weights[j] = expf(scores[j] - max_score);
            sum_exp += weights[j];
        }
        mass[qi] = sum_exp * expf(max_score);

        for (int j = 0; j < t; j++) {
            weights[j] /= sum_exp;
        }

        for (int j = 0; j < t; j++) {
            for (int d = 0; d < d_v; d++) {
                output[qi * d_v + d] += weights[j] * C_v[j * d_v + d];
            }
        }
    }
}

struct bench_result {
    const char * mode_name;
    const char * beta_name;
    int    T;
    int    t;
    float  ratio;
    float  mse_ref;      // MSE on reference queries (training)
    float  mse_eval;     // MSE on evaluation queries (generalization)
    float  mass_err_ref; // relative mass error on ref queries
    float  mass_err_eval;// relative mass error on eval queries
    double time_us;      // wall-clock time in microseconds
};

static bench_result run_one(
        const std::vector<float> & K, const std::vector<float> & V,
        const std::vector<float> & Q_ref, const std::vector<float> & Q_eval,
        int T, int n_head_kv, int d_k, int d_v,
        int t, key_select_mode sel_mode, beta_fit_mode fit_mode,
        const char * mode_name, const char * beta_name) {

    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;

    bench_result res;
    res.mode_name = mode_name;
    res.beta_name = beta_name;
    res.T = T;
    res.t = t;
    res.ratio = (float)T / t;

    // Time the compaction
    compaction_config cfg;
    cfg.select_mode = sel_mode;
    cfg.fit_mode = fit_mode;
    cfg.n_alt_rounds = 2;

    auto t0 = std::chrono::high_resolution_clock::now();
    auto result = compact_layer_all_heads(
        K.data(), V.data(), Q_ref.data(),
        T, N_Q_REF, n_head_kv, d_k, d_v, t, cfg);
    auto t1 = std::chrono::high_resolution_clock::now();
    res.time_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // Evaluate per head, accumulate MSE and mass error
    float total_mse_ref = 0.0f, total_mse_eval = 0.0f;
    float total_mass_err_ref = 0.0f, total_mass_err_eval = 0.0f;

    for (int h = 0; h < n_head_kv; h++) {
        // Extract per-head K and V
        std::vector<float> K_h(T * d_k), V_h(T * d_v);
        for (int i = 0; i < T; i++) {
            memcpy(K_h.data() + i * d_k, K.data() + i * n_embd_k + h * d_k, d_k * sizeof(float));
            memcpy(V_h.data() + i * d_v, V.data() + i * n_embd_v + h * d_v, d_v * sizeof(float));
        }

        // Get compacted K for this head
        const float * C_k_h;
        std::vector<float> C_k_from_selected;
        if (!result.C_k.empty() && !result.C_k[h].empty()) {
            C_k_h = result.C_k[h].data();
        } else {
            // Subset selection: extract K at selected indices
            C_k_from_selected.resize(t * d_k);
            for (int j = 0; j < t; j++) {
                memcpy(C_k_from_selected.data() + j * d_k,
                       K_h.data() + result.selected_indices[j] * d_k,
                       d_k * sizeof(float));
            }
            C_k_h = C_k_from_selected.data();
        }

        // Extract per-head Q_ref and Q_eval
        std::vector<float> Q_ref_h(N_Q_REF * d_k), Q_eval_h(N_Q_EVAL * d_k);
        for (int qi = 0; qi < N_Q_REF; qi++) {
            memcpy(Q_ref_h.data() + qi * d_k,
                   Q_ref.data() + qi * n_embd_k + h * d_k,
                   d_k * sizeof(float));
        }
        for (int qi = 0; qi < N_Q_EVAL; qi++) {
            memcpy(Q_eval_h.data() + qi * d_k,
                   Q_eval.data() + qi * n_embd_k + h * d_k,
                   d_k * sizeof(float));
        }

        // Ground truth
        std::vector<float> gt_ref, gt_eval, gt_mass_ref, gt_mass_eval;
        compute_ground_truth(K_h.data(), V_h.data(), Q_ref_h.data(),
                           T, N_Q_REF, d_k, d_v, gt_ref, gt_mass_ref);
        compute_ground_truth(K_h.data(), V_h.data(), Q_eval_h.data(),
                           T, N_Q_EVAL, d_k, d_v, gt_eval, gt_mass_eval);

        // Compacted output
        std::vector<float> comp_ref, comp_eval, comp_mass_ref, comp_mass_eval;
        compute_compacted_output(C_k_h, result.C_v[h].data(), result.beta[h].data(),
                               Q_ref_h.data(), t, N_Q_REF, d_k, d_v,
                               comp_ref, comp_mass_ref);
        compute_compacted_output(C_k_h, result.C_v[h].data(), result.beta[h].data(),
                               Q_eval_h.data(), t, N_Q_EVAL, d_k, d_v,
                               comp_eval, comp_mass_eval);

        // MSE
        float mse_ref = 0.0f, mse_eval = 0.0f;
        for (int i = 0; i < N_Q_REF * d_v; i++) {
            float diff = comp_ref[i] - gt_ref[i];
            mse_ref += diff * diff;
        }
        mse_ref /= (N_Q_REF * d_v);

        for (int i = 0; i < N_Q_EVAL * d_v; i++) {
            float diff = comp_eval[i] - gt_eval[i];
            mse_eval += diff * diff;
        }
        mse_eval /= (N_Q_EVAL * d_v);

        total_mse_ref += mse_ref;
        total_mse_eval += mse_eval;

        // Relative mass error: mean |mass_comp - mass_gt| / mass_gt
        float me_ref = 0.0f, me_eval = 0.0f;
        for (int qi = 0; qi < N_Q_REF; qi++) {
            me_ref += fabsf(comp_mass_ref[qi] - gt_mass_ref[qi]) / (gt_mass_ref[qi] + 1e-12f);
        }
        me_ref /= N_Q_REF;

        for (int qi = 0; qi < N_Q_EVAL; qi++) {
            me_eval += fabsf(comp_mass_eval[qi] - gt_mass_eval[qi]) / (gt_mass_eval[qi] + 1e-12f);
        }
        me_eval /= N_Q_EVAL;

        total_mass_err_ref += me_ref;
        total_mass_err_eval += me_eval;
    }

    res.mse_ref = total_mse_ref / n_head_kv;
    res.mse_eval = total_mse_eval / n_head_kv;
    res.mass_err_ref = total_mass_err_ref / n_head_kv;
    res.mass_err_eval = total_mass_err_eval / n_head_kv;

    return res;
}

int main() {
    printf("=== KV Compaction Synthetic Benchmark ===\n");
    printf("Model: Qwen3.5-0.8B (full-attention layers)\n");
    printf("  n_head_kv=%d, d_head=%d, GQA_ratio=%d\n", N_HEAD_KV, D_HEAD, GQA_RATIO);
    printf("  n_q_ref=%d, n_q_eval=%d\n\n", N_Q_REF, N_Q_EVAL);

    // Test at multiple sequence lengths and compression ratios
    struct test_case {
        int T;
        std::vector<int> targets;  // compacted sizes
    };

    std::vector<test_case> cases = {
        { 256,  {128, 64, 32, 16, 8, 5} },      // 2x, 4x, 8x, 16x, 32x, ~50x
        { 512,  {256, 128, 64, 32, 16, 10} },    // 2x, 4x, 8x, 16x, 32x, ~50x
        { 1024, {512, 256, 128, 64, 32, 20} },   // 2x, 4x, 8x, 16x, 32x, ~50x
        { 4096, {2048, 1024, 256, 128, 82} },    // 2x, 4x, 16x, 32x, ~50x
    };

    struct mode_spec {
        key_select_mode sel;
        beta_fit_mode   fit;
        const char *    sel_name;
        const char *    fit_name;
    };

    std::vector<mode_spec> modes = {
        { KEY_SELECT_MAX_ATTN,    BETA_FIT_NNLS,     "max_attn",    "nnls"     },
        { KEY_SELECT_MAX_ATTN,    BETA_FIT_SINKHORN,  "max_attn",    "sinkhorn" },
        { KEY_SELECT_SUBMODULAR,  BETA_FIT_NNLS,     "submodular",  "nnls"     },
        { KEY_SELECT_SUBMODULAR,  BETA_FIT_SINKHORN,  "submodular",  "sinkhorn" },
        { KEY_SELECT_TOKEN_MERGE, BETA_FIT_NNLS,     "token_merge", "nnls"     },
        { KEY_SELECT_KMEANS,      BETA_FIT_NNLS,     "kmeans",      "nnls"     },
    };

    // Print header
    printf("%-14s %-10s %5s %5s %5s  %12s %12s  %10s %10s  %10s\n",
           "mode", "beta", "T", "t", "ratio",
           "MSE(ref)", "MSE(eval)",
           "mass_e(ref)", "mass_e(eval)",
           "time_us");
    printf("%-14s %-10s %5s %5s %5s  %12s %12s  %10s %10s  %10s\n",
           "--------------", "----------", "-----", "-----", "-----",
           "------------", "------------",
           "----------", "----------",
           "----------");

    for (auto & tc : cases) {
        std::vector<float> K, V, Q_ref, Q_eval;
        generate_data(K, V, Q_ref, Q_eval, tc.T, N_HEAD_KV, D_HEAD, D_HEAD,
                     N_Q_REF, N_Q_EVAL);

        for (int t : tc.targets) {
            for (auto & m : modes) {
                // Skip token_merge for large T (O(T^2) is too slow)
                if (m.sel == KEY_SELECT_TOKEN_MERGE && tc.T > 512) continue;
                // Skip submodular for large T (too slow)
                if (m.sel == KEY_SELECT_SUBMODULAR && tc.T > 1024) continue;

                auto res = run_one(K, V, Q_ref, Q_eval,
                                  tc.T, N_HEAD_KV, D_HEAD, D_HEAD,
                                  t, m.sel, m.fit, m.sel_name, m.fit_name);

                printf("%-14s %-10s %5d %5d %5.1f  %12.8f %12.8f  %10.6f %10.6f  %10.0f\n",
                       res.mode_name, res.beta_name,
                       res.T, res.t, res.ratio,
                       res.mse_ref, res.mse_eval,
                       res.mass_err_ref, res.mass_err_eval,
                       res.time_us);
            }
            printf("\n");
        }
        printf("---\n\n");
    }

    // Carathéodory budget analysis
    printf("=== Carathéodory Budget Analysis ===\n");
    for (auto & tc : cases) {
        std::vector<float> K, V, Q_ref, Q_eval;
        generate_data(K, V, Q_ref, Q_eval, tc.T, N_HEAD_KV, D_HEAD, D_HEAD,
                     N_Q_REF, N_Q_EVAL);

        auto budgets = compute_caratheodory_budgets(V.data(), tc.T, N_HEAD_KV, D_HEAD);
        printf("T=%4d:", tc.T);
        for (int h = 0; h < N_HEAD_KV; h++) {
            printf("  head%d_min_budget=%d", h, budgets[h]);
        }
        printf("  (d_v+1=%d)\n", D_HEAD + 1);
    }

    printf("\nBenchmark complete.\n");
    return 0;
}
