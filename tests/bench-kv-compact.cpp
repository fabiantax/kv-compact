// Automated quality benchmarks for KV cache compaction
//
// Tests compaction quality at multiple compression ratios across various
// model-like configurations. Reports cosine similarity, MSE, and token-level
// agreement rate. Outputs both human-readable and JSON summaries.
//
// US-10: "Automated quality benchmarks"
//   - Perplexity-proxy metrics (cosine sim, MSE, agreement rate)
//   - Multiple compression ratios (20%, 50%, 80% retention)
//   - Multi-layer, multi-head scenarios
//   - Machine-parseable JSON output

#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include "kv-compact-math.h"

// ============================================================================
// Benchmark utilities
// ============================================================================

struct bench_metrics {
    float cosine_sim;      // cosine similarity of attention output
    float mse;             // mean squared error of attention output
    float agreement_rate;  // fraction of queries where argmax output dim agrees
    float max_abs_err;     // maximum absolute error across all output dims
    double elapsed_ms;     // wall-clock time for compaction
};

// Generate deterministic pseudo-random data with structure
// (sine/cosine patterns give correlations like real KV data)
static void gen_structured_data(float * out, int n, int seed_offset, float scale = 1.0f) {
    for (int i = 0; i < n; i++) {
        out[i] = sinf((float)(i * 7 + seed_offset) * 0.31f) * scale
               + 0.3f * cosf((float)(i * 3 + seed_offset + 17) * 0.53f) * scale;
    }
}

// Compute attention output for a single query against full/compacted KV
static void compute_attn_output(
        const float * q, const float * K, const float * V,
        int T, int d_k, int d_v, float * out) {
    float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);

    // scores = q @ K^T / sqrt(d_k)
    std::vector<float> scores(T);
    for (int j = 0; j < T; j++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) dot += q[d] * K[j * d_k + d];
        scores[j] = dot * inv_sqrt_dk;
    }
    softmax_rows(scores.data(), 1, T);

    // out = scores @ V
    memset(out, 0, d_v * sizeof(float));
    for (int j = 0; j < T; j++) {
        for (int d = 0; d < d_v; d++) {
            out[d] += scores[j] * V[j * d_v + d];
        }
    }
}

// Compute attention output with beta bias (compacted cache)
static void compute_attn_output_biased(
        const float * q, const float * K_selected, const float * C_v,
        const float * beta, int t, int d_k, int d_v, float * out) {
    float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);

    std::vector<float> scores(t);
    for (int j = 0; j < t; j++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) dot += q[d] * K_selected[j * d_k + d];
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

// Compute metrics comparing original vs compacted output across test queries
static bench_metrics evaluate_head(
        const float * K, const float * V, const float * Q_ref,
        int T, int n_q, int d_k, int d_v,
        const compacted_head & ch) {
    bench_metrics m = {};
    int t = (int) ch.selected_indices.size();

    // Extract selected K rows for biased computation
    std::vector<float> K_sel(t * d_k);
    for (int j = 0; j < t; j++) {
        memcpy(K_sel.data() + j * d_k, K + ch.selected_indices[j] * d_k,
               d_k * sizeof(float));
    }

    float sum_cos = 0.0f, sum_mse = 0.0f, sum_max_err = 0.0f;
    int agree_count = 0;

    for (int qi = 0; qi < n_q; qi++) {
        const float * q = Q_ref + qi * d_k;

        std::vector<float> orig_out(d_v), comp_out(d_v);
        compute_attn_output(q, K, V, T, d_k, d_v, orig_out.data());
        compute_attn_output_biased(q, K_sel.data(), ch.C_v.data(),
                                   ch.beta.data(), t, d_k, d_v, comp_out.data());

        // Cosine similarity
        float dot = 0.0f, no = 0.0f, nc = 0.0f;
        for (int d = 0; d < d_v; d++) {
            dot += orig_out[d] * comp_out[d];
            no += orig_out[d] * orig_out[d];
            nc += comp_out[d] * comp_out[d];
        }
        sum_cos += dot / (sqrtf(no * nc) + 1e-8f);

        // MSE
        float mse = 0.0f;
        for (int d = 0; d < d_v; d++) {
            float diff = orig_out[d] - comp_out[d];
            mse += diff * diff;
        }
        sum_mse += mse / d_v;

        // Max abs error
        float max_err = 0.0f;
        for (int d = 0; d < d_v; d++) {
            float err = fabsf(orig_out[d] - comp_out[d]);
            if (err > max_err) max_err = err;
        }
        if (max_err > sum_max_err) sum_max_err = max_err;

        // Token agreement (argmax of output dims matches)
        int argmax_orig = 0, argmax_comp = 0;
        for (int d = 1; d < d_v; d++) {
            if (orig_out[d] > orig_out[argmax_orig]) argmax_orig = d;
            if (comp_out[d] > comp_out[argmax_comp]) argmax_comp = d;
        }
        if (argmax_orig == argmax_comp) agree_count++;
    }

    m.cosine_sim = sum_cos / n_q;
    m.mse = sum_mse / n_q;
    m.max_abs_err = sum_max_err;
    m.agreement_rate = (float) agree_count / n_q;
    return m;
}

// ============================================================================
// Benchmark scenarios
// ============================================================================

struct bench_config {
    const char * name;
    int T;          // sequence length
    int n_q;        // reference queries
    int n_head_kv;  // KV heads
    int d_k;        // key dim per head
    int d_v;        // value dim per head
    int n_layer;    // layers to simulate
};

static const bench_config configs[] = {
    {"small-1L",     64,  32, 4,  64,  64, 1},
    {"medium-2L",   128,  64, 8,  64,  64, 2},
    {"large-4L",    256,  64, 8, 128, 128, 4},
    {"wide-heads",  128,  32, 32, 64,  64, 1},
    {"1k-8h",      1024,  64, 8,  128, 128, 1},
    {"5k-4h",      5120,  32, 4,   64,  64, 1},
    {"10k-4h",    10240,  32, 4,   64,  64, 1},
    {"10k-8h",    10240,  32, 8,  128, 128, 1},
};
static const int n_configs = sizeof(configs) / sizeof(configs[0]);

static const float retention_ratios[] = {0.20f, 0.50f, 0.80f};
static const int n_ratios = sizeof(retention_ratios) / sizeof(retention_ratios[0]);

// ============================================================================
// Main benchmark runner
// ============================================================================

int main(int argc, char ** argv) {
    bool json_output = false;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--json") == 0) json_output = true;
    }

    printf("bench-kv-compact: Automated quality benchmarks\n");
    printf("================================================\n\n");

    // Track global pass/fail
    int total_tests = 0;
    int passed_tests = 0;

    // JSON accumulator
    std::string json_str = "[\n";

    bool first_json = true;

    for (int ci = 0; ci < n_configs; ci++) {
        const auto & cfg = configs[ci];
        printf("--- Config: %s (T=%d, n_q=%d, heads=%d, d_k=%d, layers=%d) ---\n",
               cfg.name, cfg.T, cfg.n_q, cfg.n_head_kv, cfg.d_k, cfg.n_layer);

        for (int ri = 0; ri < n_ratios; ri++) {
            float ratio = retention_ratios[ri];
            int t = std::max(1, (int)(cfg.T * ratio));

            printf("  Retention %.0f%% (t=%d/%d):\n", ratio * 100, t, cfg.T);

            // Aggregate metrics across layers and heads
            float avg_cos = 0.0f, avg_mse = 0.0f, avg_agree = 0.0f;
            float worst_cos = 1.0f, worst_mse = 0.0f;
            double total_ms = 0.0;
            int head_count = 0;

            for (int layer = 0; layer < cfg.n_layer; layer++) {
                // Generate data for this layer
                int seed_k = layer * 1000 + 1;
                int seed_v = layer * 1000 + 500;
                int seed_q = layer * 1000 + 900;

                const int n_embd_k = cfg.n_head_kv * cfg.d_k;
                const int n_embd_v = cfg.n_head_kv * cfg.d_v;

                std::vector<float> K(cfg.T * n_embd_k);
                std::vector<float> V(cfg.T * n_embd_v);
                std::vector<float> Q(cfg.n_q * n_embd_k);

                gen_structured_data(K.data(), cfg.T * n_embd_k, seed_k);
                gen_structured_data(V.data(), cfg.T * n_embd_v, seed_v);
                gen_structured_data(Q.data(), cfg.n_q * n_embd_k, seed_q);

                // Time the compaction
                auto t0 = std::chrono::high_resolution_clock::now();
                auto result = compact_layer_all_heads(
                    K.data(), V.data(), Q.data(),
                    cfg.T, cfg.n_q, cfg.n_head_kv, cfg.d_k, cfg.d_v, t);
                auto t1 = std::chrono::high_resolution_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
                total_ms += ms;

                // Evaluate each head
                for (int h = 0; h < cfg.n_head_kv; h++) {
                    // Extract per-head K, V, Q
                    std::vector<float> Kh(cfg.T * cfg.d_k);
                    std::vector<float> Vh(cfg.T * cfg.d_v);
                    std::vector<float> Qh(cfg.n_q * cfg.d_k);

                    for (int i = 0; i < cfg.T; i++) {
                        memcpy(Kh.data() + i * cfg.d_k,
                               K.data() + i * n_embd_k + h * cfg.d_k,
                               cfg.d_k * sizeof(float));
                        memcpy(Vh.data() + i * cfg.d_v,
                               V.data() + i * n_embd_v + h * cfg.d_v,
                               cfg.d_v * sizeof(float));
                    }
                    for (int i = 0; i < cfg.n_q; i++) {
                        memcpy(Qh.data() + i * cfg.d_k,
                               Q.data() + i * n_embd_k + h * cfg.d_k,
                               cfg.d_k * sizeof(float));
                    }

                    // Build a compacted_head from the layer result
                    compacted_head ch;
                    ch.selected_indices = result.selected_indices;
                    ch.beta = result.beta[h];
                    ch.C_v = result.C_v[h];

                    bench_metrics bm = evaluate_head(
                        Kh.data(), Vh.data(), Qh.data(),
                        cfg.T, cfg.n_q, cfg.d_k, cfg.d_v, ch);

                    avg_cos += bm.cosine_sim;
                    avg_mse += bm.mse;
                    avg_agree += bm.agreement_rate;
                    if (bm.cosine_sim < worst_cos) worst_cos = bm.cosine_sim;
                    if (bm.mse > worst_mse) worst_mse = bm.mse;
                    head_count++;
                }
            }

            avg_cos /= head_count;
            avg_mse /= head_count;
            avg_agree /= head_count;

            printf("    Avg cosine sim:    %.6f\n", avg_cos);
            printf("    Avg MSE:           %.8f\n", avg_mse);
            printf("    Avg agreement:     %.1f%%\n", avg_agree * 100.0f);
            printf("    Worst cosine sim:  %.6f\n", worst_cos);
            printf("    Worst MSE:         %.8f\n", worst_mse);
            printf("    Compaction time:   %.2f ms (%d layers)\n", total_ms, cfg.n_layer);

            // Quality thresholds (assertions)
            // These are conservative — they should always pass for reasonable data
            bool pass = true;
            total_tests++;

            if (ratio >= 0.5f) {
                // At 50%+ retention, expect high quality
                if (avg_cos < 0.85f) {
                    printf("    FAIL: avg cosine sim %.4f < 0.85 at %.0f%% retention\n",
                           avg_cos, ratio * 100);
                    pass = false;
                }
                if (avg_agree < 0.5f) {
                    printf("    FAIL: agreement rate %.1f%% < 50%% at %.0f%% retention\n",
                           avg_agree * 100, ratio * 100);
                    pass = false;
                }
            } else {
                // At 20% retention, looser thresholds
                if (avg_cos < 0.5f) {
                    printf("    FAIL: avg cosine sim %.4f < 0.5 at %.0f%% retention\n",
                           avg_cos, ratio * 100);
                    pass = false;
                }
            }

            // All values must be finite
            if (!std::isfinite(avg_cos) || !std::isfinite(avg_mse)) {
                printf("    FAIL: non-finite metrics\n");
                pass = false;
            }

            if (pass) {
                printf("    PASS\n");
                passed_tests++;
            }

            // JSON entry
            char jbuf[1024];
            if (!first_json) json_str += ",\n";
            first_json = false;
            snprintf(jbuf, sizeof(jbuf),
                "  {\"config\": \"%s\", \"retention\": %.2f, \"t\": %d, \"T\": %d, "
                "\"avg_cosine_sim\": %.6f, \"avg_mse\": %.8f, "
                "\"avg_agreement\": %.4f, \"worst_cosine_sim\": %.6f, "
                "\"worst_mse\": %.8f, \"compaction_ms\": %.2f, "
                "\"pass\": %s}",
                cfg.name, ratio, t, cfg.T,
                avg_cos, avg_mse, avg_agree, worst_cos,
                worst_mse, total_ms, pass ? "true" : "false");
            json_str += jbuf;

            printf("\n");
        }
    }

    json_str += "\n]\n";

    printf("================================================\n");
    printf("Results: %d/%d passed\n", passed_tests, total_tests);

    if (json_output) {
        printf("\n--- JSON ---\n");
        printf("%s", json_str.c_str());
    }

    return (passed_tests == total_tests) ? 0 : 1;
}
