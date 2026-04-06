// Validation test: SnapKV Observation Window (arXiv:2404.14469)
//
// Hypothesis: Using only the last W=32 Q_ref vectors produces quality within
// 0.5% of using all 64, because recent tokens predict generation-time attention
// best (SnapKV observation window principle).
//
// Pass criteria:
//   - Cosine sim delta < 0.005 between W=64 and W=32
//   - At W=16: cosine sim delta < 0.02
//   - Speedup proportional to n_q reduction (2x at W=32)

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>
#include <algorithm>
#include <chrono>

#include "kv-compact-api.h"
#include "kv-compact-math.h"

using clock_type = std::chrono::high_resolution_clock;

// ============================================================================
// Data generation (reused from bench-kv-compact-quality)
// ============================================================================

static void gen_data(float * out, int n, int seed) {
    for (int i = 0; i < n; i++) {
        out[i] = sinf((float)(i * 7 + seed) * 0.31f)
               + 0.3f * cosf((float)(i * 3 + seed + 17) * 0.53f);
    }
}

static void gen_spiky_data(float * K, float * V, float * Q,
                           int T, int n_q, int n_head_kv, int d_k, int d_v,
                           int seed) {
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    gen_data(V, T * n_embd_v, seed + 500);

    std::vector<float> base_dir(d_k);
    for (int d = 0; d < d_k; d++)
        base_dir[d] = sinf((float)(d * 7 + seed) * 0.31f);

    for (int t = 0; t < T; t++) {
        for (int h = 0; h < n_head_kv; h++) {
            float * row = K + t * n_embd_k + h * d_k;
            if (t == 0) {
                for (int d = 0; d < d_k; d++)
                    row[d] = 3.0f * cosf((float)(d * 13 + seed) * 0.17f);
            } else if (t == 1 || t == T/4 || t == T/2) {
                for (int d = 0; d < d_k; d++)
                    row[d] = 2.5f * sinf((float)(d * 11 + t * 37 + seed) * 0.23f);
            } else if (t > T - T/10) {
                for (int d = 0; d < d_k; d++) {
                    float noise = 0.1f * sinf((float)(t * 31 + d * 7 + h * 53 + seed) * 0.41f);
                    row[d] = 1.5f * (base_dir[d] + noise);
                }
            } else {
                for (int d = 0; d < d_k; d++) {
                    float noise = 0.05f * sinf((float)(t * 31 + d * 7 + h * 53 + seed) * 0.41f);
                    row[d] = base_dir[d] + noise;
                }
            }
        }
    }

    for (int q = 0; q < n_q; q++) {
        for (int h = 0; h < n_head_kv; h++) {
            float * qrow = Q + q * n_embd_k + h * d_k;
            float sink_w = 0.7f + 0.3f * sinf((float)(q * 13 + h * 7 + seed) * 0.29f);
            for (int d = 0; d < d_k; d++) {
                float sink_dir = cosf((float)(d * 13 + seed) * 0.17f);
                qrow[d] = sink_w * sink_dir + (1.0f - sink_w) * base_dir[d];
                qrow[d] += 0.2f * sinf((float)(q * 41 + d * 3 + h * 19 + seed) * 0.37f);
            }
        }
    }
}

// ============================================================================
// Helper: compute attention output for cosine sim comparison
// ============================================================================

static void compute_original_output(
        const float * q, const float * K, const float * V,
        int T, int d_k, int d_v, float * out) {
    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
    std::vector<float> scores(T);
    for (int j = 0; j < T; j++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) dot += q[d] * K[j * d_k + d];
        scores[j] = dot * inv_sqrt_dk;
    }
    softmax_rows(scores.data(), 1, T);
    memset(out, 0, d_v * sizeof(float));
    for (int j = 0; j < T; j++)
        for (int d = 0; d < d_v; d++)
            out[d] += scores[j] * V[j * d_v + d];
}

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
    for (int j = 0; j < t; j++)
        for (int d = 0; d < d_v; d++)
            out[d] += scores[j] * C_v[j * d_v + d];
}

// ============================================================================
// Bench 2a: Window Size vs Quality
// ============================================================================

static void bench_window_quality() {
    printf("=== Bench 2a: Window Size vs Quality ===\n\n");

    const int n_head_kv = 4, d_k = 64, d_v = 64;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;
    const int n_q_full = 64;

    int T_sizes[] = {256, 1024, 4096};
    float ratios[] = {0.5f, 0.2f, 0.1f};
    int windows[] = {64, 32, 16, 8, 4};

    printf("  %-6s  %-8s  %-6s  %12s  %12s  %12s\n",
           "T", "ratio", "W", "cos_sim", "mse", "cos_delta");
    printf("  %-6s  %-8s  %-6s  %12s  %12s  %12s\n",
           "------", "--------", "------", "------------", "------------", "------------");

    int total_pass = 0, total_tests = 0;

    for (int T : T_sizes) {
        std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q_full(n_q_full * n_embd_k);
        gen_spiky_data(K.data(), V.data(), Q_full.data(), T, n_q_full, n_head_kv, d_k, d_v, 3000 + T);

        // Baseline: n_q=64
        float baseline_cos = 0.0f, baseline_mse = 0.0f;
        {
            kv_compact_params p = kv_compact_params_default();
            p.target_ratio = 0.2f;
            p.use_cheap_qref = 0;
            p.skip_beta = 1;
            p.chunk_size = -1;

            kv_compact_result r = {};
            kv_compact(K.data(), V.data(), Q_full.data(), T, n_q_full,
                       n_head_kv, d_k, d_v, &p, &r);

            // Evaluate on head 0
            int h = 0;
            double cos_sum = 0.0, mse_sum = 0.0;
            int n_eval = std::min(n_q_full, 32);

            std::vector<float> K_h(T * d_k), V_h(T * d_v);
            for (int j = 0; j < T; j++) {
                memcpy(K_h.data() + j * d_k, K.data() + j * n_embd_k + h * d_k, d_k * sizeof(float));
                memcpy(V_h.data() + j * d_v, V.data() + j * n_embd_v + h * d_v, d_v * sizeof(float));
            }

            for (int qi = 0; qi < n_eval; qi++) {
                const float * q = K_h.data() + qi * d_k;  // use first K rows as test queries
                std::vector<float> orig_out(d_v), comp_out(d_v);
                compute_original_output(q, K_h.data(), V_h.data(), T, d_k, d_v, orig_out.data());
                compute_compacted_output(q, K_h.data(), r.beta[h], r.C_v[h],
                                         r.selected_indices, r.t, d_k, d_v, comp_out.data());
                float dot = 0, no = 0, nc = 0;
                for (int d = 0; d < d_v; d++) {
                    dot += orig_out[d] * comp_out[d];
                    no += orig_out[d] * orig_out[d];
                    nc += comp_out[d] * comp_out[d];
                    float diff = orig_out[d] - comp_out[d];
                    mse_sum += diff * diff;
                }
                cos_sum += dot / (sqrtf(no * nc) + 1e-8f);
            }
            baseline_cos = (float)(cos_sum / n_eval);
            baseline_mse = (float)(mse_sum / n_eval);
            kv_compact_result_free(&r);
        }

        for (float ratio : ratios) {
            for (int W : windows) {
                // Extract last W queries from the full set
                std::vector<float> Q_window(W * n_embd_k);
                for (int qi = 0; qi < W; qi++) {
                    int src_qi = n_q_full - W + qi;
                    memcpy(Q_window.data() + qi * n_embd_k,
                           Q_full.data() + src_qi * n_embd_k,
                           n_embd_k * sizeof(float));
                }

                kv_compact_params p = kv_compact_params_default();
                p.target_ratio = ratio;
                p.use_cheap_qref = 0;
                p.skip_beta = 1;
                p.chunk_size = -1;

                kv_compact_result r = {};
                kv_compact(K.data(), V.data(), Q_window.data(), T, W,
                           n_head_kv, d_k, d_v, &p, &r);

                // Evaluate on head 0
                int h = 0;
                double cos_sum = 0.0, mse_sum = 0.0;
                int n_eval = std::min(n_q_full, 32);

                std::vector<float> K_h(T * d_k), V_h(T * d_v);
                for (int j = 0; j < T; j++) {
                    memcpy(K_h.data() + j * d_k, K.data() + j * n_embd_k + h * d_k, d_k * sizeof(float));
                    memcpy(V_h.data() + j * d_v, V.data() + j * n_embd_v + h * d_v, d_v * sizeof(float));
                }

                for (int qi = 0; qi < n_eval; qi++) {
                    const float * q = K_h.data() + qi * d_k;
                    std::vector<float> orig_out(d_v), comp_out(d_v);
                    compute_original_output(q, K_h.data(), V_h.data(), T, d_k, d_v, orig_out.data());
                    compute_compacted_output(q, K_h.data(), r.beta[h], r.C_v[h],
                                             r.selected_indices, r.t, d_k, d_v, comp_out.data());
                    float dot = 0, no = 0, nc = 0;
                    for (int d = 0; d < d_v; d++) {
                        dot += orig_out[d] * comp_out[d];
                        no += orig_out[d] * orig_out[d];
                        nc += comp_out[d] * comp_out[d];
                        float diff = orig_out[d] - comp_out[d];
                        mse_sum += diff * diff;
                    }
                    cos_sum += dot / (sqrtf(no * nc) + 1e-8f);
                }
                float cos_sim = (float)(cos_sum / n_eval);
                float mse = (float)(mse_sum / n_eval);
                float cos_delta = fabsf(cos_sim - baseline_cos);

                const char * marker = "";
                if (W == 32 && cos_delta > 0.005f) marker = " FAIL";
                if (W == 16 && cos_delta > 0.02f) marker = " FAIL";

                printf("  %-6d  %-8.0f%%  %-6d  %12.6f  %12.2e  %12.6f%s\n",
                       T, ratio * 100, W, cos_sim, mse, cos_delta, marker);

                total_tests++;
                bool pass = true;
                if (W == 32 && cos_delta >= 0.005f) pass = false;
                if (W == 16 && cos_delta >= 0.02f) pass = false;
                if (pass) total_pass++;

                kv_compact_result_free(&r);
            }
        }
    }

    printf("\n  Window quality: %d/%d passed\n\n", total_pass, total_tests);
}

// ============================================================================
// Bench 2b: Scoring Speedup
// ============================================================================

static void bench_scoring_speedup() {
    printf("=== Bench 2b: Scoring Speedup ===\n\n");

    const int n_head_kv = 4, d_k = 64, d_v = 64;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;
    const int n_q_full = 64;

    int T_sizes[] = {256, 512, 1024, 2048, 4096, 10240};
    int windows[] = {64, 32, 16};

    printf("  %-6s  ", "T");
    for (int W : windows) printf("  W=%-6d", W);
    printf("\n  %-6s  ", "------");
    for (int i = 0; i < 3; i++) printf("  %8s", "--------");
    printf("\n");

    for (int T : T_sizes) {
        std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q_full(n_q_full * n_embd_k);
        gen_spiky_data(K.data(), V.data(), Q_full.data(), T, n_q_full, n_head_kv, d_k, d_v, 4000 + T);

        printf("  %-6d  ", T);

        for (int W : windows) {
            // Extract last W queries
            std::vector<float> Q_window(W * n_embd_k);
            for (int qi = 0; qi < W; qi++) {
                int src_qi = n_q_full - W + qi;
                memcpy(Q_window.data() + qi * n_embd_k,
                       Q_full.data() + src_qi * n_embd_k,
                       n_embd_k * sizeof(float));
            }

            kv_compact_params p = kv_compact_params_default();
            p.target_ratio = 0.2f;
            p.use_cheap_qref = 0;
            p.skip_beta = 1;
            p.chunk_size = -1;

            // Warmup
            {
                kv_compact_result r = {};
                kv_compact(K.data(), V.data(), Q_window.data(), T, W,
                           n_head_kv, d_k, d_v, &p, &r);
                kv_compact_result_free(&r);
            }

            const int n_runs = 3;
            double total_ms = 0.0;
            for (int run = 0; run < n_runs; run++) {
                kv_compact_result r = {};
                auto t0 = clock_type::now();
                kv_compact(K.data(), V.data(), Q_window.data(), T, W,
                           n_head_kv, d_k, d_v, &p, &r);
                total_ms += std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
                kv_compact_result_free(&r);
            }
            double avg_ms = total_ms / n_runs;
            printf("%8.2f  ", avg_ms);
        }

        // Speedup W=32 vs W=64
        {
            std::vector<float> Q64(n_q_full * n_embd_k);
            memcpy(Q64.data(), Q_full.data(), n_q_full * n_embd_k * sizeof(float));
            std::vector<float> Q32(32 * n_embd_k);
            for (int qi = 0; qi < 32; qi++) {
                memcpy(Q32.data() + qi * n_embd_k,
                       Q_full.data() + (n_q_full - 32 + qi) * n_embd_k,
                       n_embd_k * sizeof(float));
            }

            kv_compact_params p = kv_compact_params_default();
            p.target_ratio = 0.2f;
            p.use_cheap_qref = 0;
            p.skip_beta = 1;
            p.chunk_size = -1;

            // Warmup
            {
                kv_compact_result r1 = {}, r2 = {};
                kv_compact(K.data(), V.data(), Q64.data(), T, n_q_full, n_head_kv, d_k, d_v, &p, &r1);
                kv_compact(K.data(), V.data(), Q32.data(), T, 32, n_head_kv, d_k, d_v, &p, &r2);
                kv_compact_result_free(&r1);
                kv_compact_result_free(&r2);
            }

            double ms64 = 0.0, ms32 = 0.0;
            const int n_runs = 3;
            for (int run = 0; run < n_runs; run++) {
                kv_compact_result r = {};
                auto t0 = clock_type::now();
                kv_compact(K.data(), V.data(), Q64.data(), T, n_q_full, n_head_kv, d_k, d_v, &p, &r);
                ms64 += std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
                kv_compact_result_free(&r);
            }
            for (int run = 0; run < n_runs; run++) {
                kv_compact_result r = {};
                auto t0 = clock_type::now();
                kv_compact(K.data(), V.data(), Q32.data(), T, 32, n_head_kv, d_k, d_v, &p, &r);
                ms32 += std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
                kv_compact_result_free(&r);
            }
            ms64 /= n_runs;
            ms32 /= n_runs;
            printf("  speedup=%.1fx", ms64 / (ms32 + 1e-12));
        }

        printf("\n");
    }

    printf("\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("bench-opt-snapkv-window\n");
    printf("=======================\n");
    printf("Validating: arXiv:2404.14469 — SnapKV Observation Window\n\n");

    bench_window_quality();
    bench_scoring_speedup();

    printf("Validation complete.\n");
    return 0;
}
