// Validation test: Value-Norm Pre-Filter (arXiv:2406.12335 — VATP)
//
// Hypothesis: Tokens with low ||V[j]||₁ contribute negligibly to attention
// output. Pre-filtering bottom 30% has <0.1% quality impact and correlates
// with attention importance (Spearman >= 0.3).
//
// Pass criteria:
//   - At 30% pre-filter: cosine sim delta < 0.01
//   - Spearman correlation >= 0.3 between value-norm and attention importance
//   - Throughput improvement >= 20% when combined with scoring reduction

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
// Data generation
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
// Value norm computation
// ============================================================================

// Compute ||V[j]||₁ for each position j, for a single head
static std::vector<float> compute_value_norms(const float * V, int T, int d_v) {
    std::vector<float> norms(T, 0.0f);
    for (int j = 0; j < T; j++) {
        float sum = 0.0f;
        for (int d = 0; d < d_v; d++)
            sum += fabsf(V[j * d_v + d]);
        norms[j] = sum;
    }
    return norms;
}

// Filter: return indices of top (1-filter_pct)% positions by value norm
static std::vector<int> filter_by_value_norm(const std::vector<float> & norms,
                                              float filter_pct) {
    int T = (int)norms.size();
    int n_keep = T - (int)(T * filter_pct);

    std::vector<std::pair<float, int>> scored(T);
    for (int j = 0; j < T; j++)
        scored[j] = {norms[j], j};

    // Sort descending by norm
    std::partial_sort(scored.begin(), scored.begin() + n_keep, scored.end(),
                      [](const auto & a, const auto & b) { return a.first > b.first; });

    std::vector<int> indices(n_keep);
    for (int j = 0; j < n_keep; j++)
        indices[j] = scored[j].second;
    std::sort(indices.begin(), indices.end());

    return indices;
}

// ============================================================================
// Spearman rank correlation
// ============================================================================

static float spearman_correlation(const std::vector<float> & x,
                                   const std::vector<float> & y) {
    int n = (int)x.size();
    assert(n == (int)y.size());

    // Compute ranks
    auto compute_ranks = [](const std::vector<float> & v) -> std::vector<float> {
        int n = (int)v.size();
        std::vector<std::pair<float, int>> sorted(n);
        for (int i = 0; i < n; i++) sorted[i] = {v[i], i};
        std::sort(sorted.begin(), sorted.end());

        std::vector<float> ranks(n);
        for (int i = 0; i < n; ) {
            int j = i;
            while (j < n && sorted[j].first == sorted[i].first) j++;
            float avg_rank = (i + j - 1) / 2.0f + 1.0f;
            for (int k = i; k < j; k++) ranks[sorted[k].second] = avg_rank;
            i = j;
        }
        return ranks;
    };

    auto rx = compute_ranks(x);
    auto ry = compute_ranks(y);

    float sum_d2 = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = rx[i] - ry[i];
        sum_d2 += d * d;
    }

    return 1.0f - 6.0f * sum_d2 / (n * (float)(n * n - 1));
}

// ============================================================================
// Helper: compute attention output
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
// Bench 3a: Filter Quality Impact
// ============================================================================

static void bench_filter_quality() {
    printf("=== Bench 3a: Filter Quality Impact ===\n\n");

    const int n_head_kv = 4, d_k = 64, d_v = 64;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;
    const int n_q = 64;

    int T_sizes[] = {256, 1024, 4096};
    float filter_pcts[] = {0.2f, 0.3f, 0.4f};
    float compaction_ratios[] = {0.5f, 0.2f, 0.1f};

    printf("  %-6s  %-6s  %-8s  %12s  %12s  %12s\n",
           "T", "filter", "compact", "cos_delta", "mse_delta", "status");
    printf("  %-6s  %-6s  %-8s  %12s  %12s  %12s\n",
           "------", "------", "--------", "------------", "------------", "------------");

    int total_pass = 0, total_tests = 0;

    for (int T : T_sizes) {
        std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
        gen_spiky_data(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, 5000 + T);

        for (float ratio : compaction_ratios) {
            // Baseline: full data
            kv_compact_params p_base = kv_compact_params_default();
            p_base.target_ratio = ratio;
            p_base.use_cheap_qref = 0;
            p_base.skip_beta = 1;
            p_base.chunk_size = -1;

            kv_compact_result r_base = {};
            kv_compact(K.data(), V.data(), Q.data(), T, n_q,
                       n_head_kv, d_k, d_v, &p_base, &r_base);

            // Evaluate baseline on head 0
            int h = 0;
            std::vector<float> K_h(T * d_k), V_h(T * d_v);
            for (int j = 0; j < T; j++) {
                memcpy(K_h.data() + j * d_k, K.data() + j * n_embd_k + h * d_k, d_k * sizeof(float));
                memcpy(V_h.data() + j * d_v, V.data() + j * n_embd_v + h * d_v, d_v * sizeof(float));
            }

            double base_cos_sum = 0.0, base_mse_sum = 0.0;
            int n_eval = std::min(n_q, 32);
            for (int qi = 0; qi < n_eval; qi++) {
                const float * q = K_h.data() + qi * d_k;
                std::vector<float> orig_out(d_v), comp_out(d_v);
                compute_original_output(q, K_h.data(), V_h.data(), T, d_k, d_v, orig_out.data());
                compute_compacted_output(q, K_h.data(), r_base.beta[h], r_base.C_v[h],
                                         r_base.selected_indices, r_base.t, d_k, d_v, comp_out.data());
                float dot = 0, no = 0, nc = 0;
                for (int d = 0; d < d_v; d++) {
                    dot += orig_out[d] * comp_out[d];
                    no += orig_out[d] * orig_out[d];
                    nc += comp_out[d] * comp_out[d];
                    float diff = orig_out[d] - comp_out[d];
                    base_mse_sum += diff * diff;
                }
                base_cos_sum += dot / (sqrtf(no * nc) + 1e-8f);
            }
            float base_cos = (float)(base_cos_sum / n_eval);
            float base_mse = (float)(base_mse_sum / n_eval);

            kv_compact_result_free(&r_base);

            for (float filter_pct : filter_pcts) {
                // Compute value norms for head 0
                std::vector<float> v_norms = compute_value_norms(V_h.data(), T, d_v);

                // Filter
                std::vector<int> kept = filter_by_value_norm(v_norms, filter_pct);
                int T_filtered = (int)kept.size();

                // Build filtered K, V (all heads)
                std::vector<float> K_filt(T_filtered * n_embd_k);
                std::vector<float> V_filt(T_filtered * n_embd_v);
                for (int j = 0; j < T_filtered; j++) {
                    memcpy(K_filt.data() + j * n_embd_k,
                           K.data() + kept[j] * n_embd_k, n_embd_k * sizeof(float));
                    memcpy(V_filt.data() + j * n_embd_v,
                           V.data() + kept[j] * n_embd_v, n_embd_v * sizeof(float));
                }

                // Compact filtered data
                kv_compact_params p_filt = kv_compact_params_default();
                p_filt.target_ratio = ratio;
                p_filt.use_cheap_qref = 0;
                p_filt.skip_beta = 1;
                p_filt.chunk_size = -1;

                kv_compact_result r_filt = {};
                kv_compact(K_filt.data(), V_filt.data(), Q.data(), T_filtered, n_q,
                           n_head_kv, d_k, d_v, &p_filt, &r_filt);

                // Build filtered per-head data for eval
                std::vector<float> K_filt_h(T_filtered * d_k), V_filt_h(T_filtered * d_v);
                for (int j = 0; j < T_filtered; j++) {
                    memcpy(K_filt_h.data() + j * d_k,
                           K_filt.data() + j * n_embd_k + h * d_k, d_k * sizeof(float));
                    memcpy(V_filt_h.data() + j * d_v,
                           V_filt.data() + j * n_embd_v + h * d_v, d_v * sizeof(float));
                }

                // Evaluate filtered compaction on original full data
                double filt_cos_sum = 0.0, filt_mse_sum = 0.0;
                for (int qi = 0; qi < n_eval; qi++) {
                    const float * q = K_h.data() + qi * d_k;
                    std::vector<float> orig_out(d_v), filt_out(d_v);
                    compute_original_output(q, K_h.data(), V_h.data(), T, d_k, d_v, orig_out.data());

                    // Compacted from filtered: need to map selected indices back to original positions
                    std::vector<int> orig_selected(r_filt.t);
                    for (int j = 0; j < r_filt.t; j++)
                        orig_selected[j] = kept[r_filt.selected_indices[j]];

                    compute_compacted_output(q, K_h.data(), r_filt.beta[h], r_filt.C_v[h],
                                             orig_selected.data(), r_filt.t, d_k, d_v, filt_out.data());
                    float dot = 0, no = 0, nc = 0;
                    for (int d = 0; d < d_v; d++) {
                        dot += orig_out[d] * filt_out[d];
                        no += orig_out[d] * orig_out[d];
                        nc += filt_out[d] * filt_out[d];
                        float diff = orig_out[d] - filt_out[d];
                        filt_mse_sum += diff * diff;
                    }
                    filt_cos_sum += dot / (sqrtf(no * nc) + 1e-8f);
                }
                float filt_cos = (float)(filt_cos_sum / n_eval);
                float filt_mse = (float)(filt_mse_sum / n_eval);
                float cos_delta = fabsf(filt_cos - base_cos);

                const char * status = "OK";
                total_tests++;
                if (filter_pct <= 0.3f && cos_delta >= 0.01f) {
                    status = "FAIL";
                } else {
                    total_pass++;
                }

                printf("  %-6d  %4.0f%%   %7.0f%%  %12.6f  %12.2e  %s\n",
                       T, filter_pct * 100, ratio * 100, cos_delta, filt_mse - base_mse, status);

                kv_compact_result_free(&r_filt);
            }
        }
    }

    printf("\n  Filter quality: %d/%d passed (cos_delta < 0.01 at filter <= 30%%)\n\n",
           total_pass, total_tests);
}

// ============================================================================
// Bench 3b: Correlation Analysis
// ============================================================================

static void bench_correlation_analysis() {
    printf("=== Bench 3b: Value-Norm vs Attention Importance Correlation ===\n\n");

    const int n_head_kv = 4, d_k = 64, d_v = 64;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;
    const int n_q = 64;

    int T_sizes[] = {256, 1024, 4096};

    printf("  %-6s  %-6s  %12s\n", "T", "head", "spearman");
    printf("  %-6s  %-6s  %12s\n", "------", "------", "------------");

    float min_spearman = 1e30f;

    for (int T : T_sizes) {
        std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
        gen_spiky_data(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, 6000 + T);

        // Compute attention importance (full scoring) for each head
        float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

        for (int h = 0; h < n_head_kv; h++) {
            // Extract per-head V
            std::vector<float> V_h(T * d_v);
            for (int j = 0; j < T; j++)
                memcpy(V_h.data() + j * d_v, V.data() + j * n_embd_v + h * d_v, d_v * sizeof(float));

            // Value norms
            std::vector<float> v_norms = compute_value_norms(V_h.data(), T, d_v);

            // Attention importance: max score across queries
            std::vector<float> importance(T, -1e30f);
            for (int qi = 0; qi < n_q; qi++) {
                const float * q = Q.data() + qi * n_embd_k + h * d_k;
                for (int j = 0; j < T; j++) {
                    float dot = 0.0f;
                    const float * k = K.data() + j * n_embd_k + h * d_k;
                    for (int d = 0; d < d_k; d++) dot += q[d] * k[d];
                    float score = dot * inv_sqrt_dk;
                    if (score > importance[j]) importance[j] = score;
                }
            }

            float spearman = spearman_correlation(v_norms, importance);
            if (spearman < min_spearman) min_spearman = spearman;

            printf("  %-6d  %-6d  %12.4f\n", T, h, spearman);
        }
    }

    printf("\n  Min Spearman: %.4f (threshold >= 0.3)\n", min_spearman);
    if (min_spearman < 0.3f) {
        printf("  WARN: correlation below threshold — value-norm may not be a reliable proxy\n");
    } else {
        printf("  PASS: correlation sufficient for pre-filtering\n");
    }
    printf("\n");
}

// ============================================================================
// Bench 3c: Throughput Impact
// ============================================================================

static void bench_throughput_impact() {
    printf("=== Bench 3c: Throughput Impact ===\n\n");

    const int n_head_kv = 4, d_k = 64, d_v = 64;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;
    const int n_q = 64;

    int T_sizes[] = {256, 1024, 4096};
    float filter_pct = 0.3f;

    printf("  %-6s  %12s  %12s  %12s\n",
           "T", "base_ms", "filtered_ms", "speedup");
    printf("  %-6s  %12s  %12s  %12s\n",
           "------", "------------", "------------", "------------");

    for (int T : T_sizes) {
        std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
        gen_spiky_data(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, 7000 + T);

        // Baseline: full compaction
        kv_compact_params p = kv_compact_params_default();
        p.target_ratio = 0.2f;
        p.use_cheap_qref = 0;
        p.skip_beta = 1;
        p.chunk_size = -1;

        // Warmup
        {
            kv_compact_result r = {};
            kv_compact(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, &p, &r);
            kv_compact_result_free(&r);
        }

        double base_ms = 0.0;
        const int n_runs = 3;
        for (int run = 0; run < n_runs; run++) {
            kv_compact_result r = {};
            auto t0 = clock_type::now();
            kv_compact(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, &p, &r);
            base_ms += std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
            kv_compact_result_free(&r);
        }
        base_ms /= n_runs;

        // Filtered: compute value norms, filter, then compact
        // Use head 0 value norms for filtering (simple approach)
        int h = 0;
        std::vector<float> V_h(T * d_v);
        for (int j = 0; j < T; j++)
            memcpy(V_h.data() + j * d_v, V.data() + j * n_embd_v + h * d_v, d_v * sizeof(float));
        std::vector<float> v_norms = compute_value_norms(V_h.data(), T, d_v);
        std::vector<int> kept = filter_by_value_norm(v_norms, filter_pct);
        int T_filtered = (int)kept.size();

        std::vector<float> K_filt(T_filtered * n_embd_k);
        std::vector<float> V_filt(T_filtered * n_embd_v);
        for (int j = 0; j < T_filtered; j++) {
            memcpy(K_filt.data() + j * n_embd_k,
                   K.data() + kept[j] * n_embd_k, n_embd_k * sizeof(float));
            memcpy(V_filt.data() + j * n_embd_v,
                   V.data() + kept[j] * n_embd_v, n_embd_v * sizeof(float));
        }

        // Warmup
        {
            kv_compact_result r = {};
            kv_compact(K_filt.data(), V_filt.data(), Q.data(), T_filtered, n_q,
                       n_head_kv, d_k, d_v, &p, &r);
            kv_compact_result_free(&r);
        }

        double filt_ms = 0.0;
        for (int run = 0; run < n_runs; run++) {
            kv_compact_result r = {};
            auto t0 = clock_type::now();
            kv_compact(K_filt.data(), V_filt.data(), Q.data(), T_filtered, n_q,
                       n_head_kv, d_k, d_v, &p, &r);
            filt_ms += std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
            kv_compact_result_free(&r);
        }
        filt_ms /= n_runs;

        double speedup = base_ms / (filt_ms + 1e-12);

        printf("  %-6d  %12.2f  %12.2f  %12.1fx\n",
               T, base_ms, filt_ms, speedup);

        assert(base_ms > 0.0);
        assert(filt_ms > 0.0);
    }

    printf("\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("bench-opt-value-norm-filter\n");
    printf("===========================\n");
    printf("Validating: arXiv:2406.12335 — Value-Norm Pre-Filter (VATP)\n\n");

    bench_filter_quality();
    bench_correlation_analysis();
    bench_throughput_impact();

    printf("Validation complete.\n");
    return 0;
}
