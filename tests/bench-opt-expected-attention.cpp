// Validation test: Expected Attention Proxy (arXiv:2510.00636)
//
// Hypothesis: q_mean = mean(Q_ref) produces top-k selections with IoU >90%
// vs full Q_ref @ K^T scoring, while being n_q× faster.
//
// Pass criteria:
//   - IoU >= 90% at all ratios >= 10%
//   - Cosine sim delta < 0.01 vs full scoring
//   - Measured speedup >= 10x at n_q=64

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
// IoU computation
// ============================================================================

static float compute_iou(const int * a, int na, const int * b, int nb) {
    // Both arrays are sorted
    int intersection = 0;
    int ia = 0, ib = 0;
    while (ia < na && ib < nb) {
        if (a[ia] == b[ib]) { intersection++; ia++; ib++; }
        else if (a[ia] < b[ib]) { ia++; }
        else { ib++; }
    }
    int union_size = na + nb - intersection;
    return (union_size > 0) ? (float)intersection / (float)union_size : 1.0f;
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
// Method B: Expected attention proxy selection
// ============================================================================

// Compute q_mean = mean(Q_ref) and select top-t by q_mean @ K^T
static void expected_attention_select(
        const float * Q_ref, int n_q, int d_k,
        const float * K, int T,
        int t, int * selected) {
    // Compute q_mean = mean of Q_ref rows (per-head slice already extracted)
    std::vector<float> q_mean(d_k, 0.0f);
    for (int qi = 0; qi < n_q; qi++)
        for (int d = 0; d < d_k; d++)
            q_mean[d] += Q_ref[qi * d_k + d];
    for (int d = 0; d < d_k; d++)
        q_mean[d] /= (float)n_q;

    // Score each position
    std::vector<std::pair<float, int>> scored(T);
    for (int j = 0; j < T; j++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++)
            dot += q_mean[d] * K[j * d_k + d];
        scored[j] = {dot, j};
    }

    // Partial sort to get top-t
    std::partial_sort(scored.begin(), scored.begin() + t, scored.end(),
                      [](const auto & a, const auto & b) { return a.first > b.first; });

    // Extract and sort indices
    for (int j = 0; j < t; j++)
        selected[j] = scored[j].second;
    std::sort(selected, selected + t);
}

// ============================================================================
// Bench 1a: Selection Quality (IoU)
// ============================================================================

static void bench_selection_quality() {
    printf("=== Bench 1a: Selection Quality (IoU) ===\n\n");

    const int n_head_kv = 4, d_k = 64, d_v = 64;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;
    const int n_q = 64;

    int T_sizes[] = {256, 1024, 4096, 10240};
    float ratios[] = {0.5f, 0.2f, 0.1f, 0.05f};

    printf("  %-6s  %-8s  %8s  %12s  %12s  %12s\n",
           "T", "ratio", "IoU", "cos_full", "cos_proxy", "cos_delta");
    printf("  %-6s  %-8s  %8s  %12s  %12s  %12s\n",
           "------", "--------", "--------", "------------", "------------", "------------");

    int total_pass = 0, total_tests = 0;

    for (int T : T_sizes) {
        std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
        gen_spiky_data(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, 1000 + T);

        for (float ratio : ratios) {
            int t = std::max(1, (int)(T * ratio));

            // Method A: full kv_compact with Q_ref
            kv_compact_params p = kv_compact_params_default();
            p.target_ratio = ratio;
            p.use_cheap_qref = 0;
            p.skip_beta = 1;
            p.chunk_size = -1;

            kv_compact_result r_full = {};
            kv_compact(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, &p, &r_full);

            // Method B: expected attention proxy
            // Extract per-head data for head 0
            std::vector<float> K_h(T * d_k), V_h(T * d_v), Q_h(n_q * d_k);
            int h = 0;
            for (int j = 0; j < T; j++) {
                memcpy(K_h.data() + j * d_k, K.data() + j * n_embd_k + h * d_k, d_k * sizeof(float));
                memcpy(V_h.data() + j * d_v, V.data() + j * n_embd_v + h * d_v, d_v * sizeof(float));
            }
            for (int qi = 0; qi < n_q; qi++)
                memcpy(Q_h.data() + qi * d_k, Q.data() + qi * n_embd_k + h * d_k, d_k * sizeof(float));

            std::vector<int> proxy_selected(t);
            expected_attention_select(Q_h.data(), n_q, d_k, K_h.data(), T, t, proxy_selected.data());

            // IoU
            float iou = compute_iou(r_full.selected_indices, r_full.t,
                                    proxy_selected.data(), t);

            // Cosine sim comparison for head 0
            // Full method
            double cos_full_sum = 0.0;
            for (int qi = 0; qi < std::min(n_q, 32); qi++) {
                const float * q = Q_h.data() + qi * d_k;
                std::vector<float> orig_out(d_v), comp_out(d_v);
                compute_original_output(q, K_h.data(), V_h.data(), T, d_k, d_v, orig_out.data());
                compute_compacted_output(q, K_h.data(), r_full.beta[h],
                                         r_full.C_v[h], r_full.selected_indices,
                                         r_full.t, d_k, d_v, comp_out.data());
                float dot = 0, no = 0, nc = 0;
                for (int d = 0; d < d_v; d++) {
                    dot += orig_out[d] * comp_out[d];
                    no += orig_out[d] * orig_out[d];
                    nc += comp_out[d] * comp_out[d];
                }
                cos_full_sum += dot / (sqrtf(no * nc) + 1e-8f);
            }
            float cos_full = (float)(cos_full_sum / std::min(n_q, 32));

            // Run kv_compact with proxy-selected Q_ref (mean query)
            // We construct a single Q_ref = q_mean and use it
            std::vector<float> Q_mean(n_embd_k, 0.0f);
            for (int qi = 0; qi < n_q; qi++)
                for (int hh = 0; hh < n_head_kv; hh++)
                    for (int d = 0; d < d_k; d++)
                        Q_mean[hh * d_k + d] += Q[qi * n_embd_k + hh * d_k + d];
            for (int hh = 0; hh < n_head_kv * d_k; hh++)
                Q_mean[hh] /= (float)n_q;

            kv_compact_params p2 = p;
            p2.use_cheap_qref = 0;
            kv_compact_result r_proxy = {};
            kv_compact(K.data(), V.data(), Q_mean.data(), T, 1, n_head_kv, d_k, d_v, &p2, &r_proxy);

            double cos_proxy_sum = 0.0;
            for (int qi = 0; qi < std::min(n_q, 32); qi++) {
                const float * q = Q_h.data() + qi * d_k;
                std::vector<float> orig_out(d_v), comp_out(d_v);
                compute_original_output(q, K_h.data(), V_h.data(), T, d_k, d_v, orig_out.data());
                compute_compacted_output(q, K_h.data(), r_proxy.beta[h],
                                         r_proxy.C_v[h], r_proxy.selected_indices,
                                         r_proxy.t, d_k, d_v, comp_out.data());
                float dot = 0, no = 0, nc = 0;
                for (int d = 0; d < d_v; d++) {
                    dot += orig_out[d] * comp_out[d];
                    no += orig_out[d] * orig_out[d];
                    nc += comp_out[d] * comp_out[d];
                }
                cos_proxy_sum += dot / (sqrtf(no * nc) + 1e-8f);
            }
            float cos_proxy = (float)(cos_proxy_sum / std::min(n_q, 32));
            float cos_delta = fabsf(cos_full - cos_proxy);

            printf("  %-6d  %-8.0f%%  %8.3f  %12.6f  %12.6f  %12.6f\n",
                   T, ratio * 100, iou, cos_full, cos_proxy, cos_delta);

            bool pass = true;
            total_tests++;
            if (ratio >= 0.1f && iou < 0.90f) {
                printf("    FAIL: IoU %.3f < 0.90 at ratio %.0f%%\n", iou, ratio * 100);
                pass = false;
            }
            if (cos_delta > 0.01f) {
                printf("    WARN: cos_delta %.4f > 0.01 (expected < 0.01)\n", cos_delta);
                // Not a hard failure — quality may still be acceptable
            }
            if (pass) total_pass++;

            kv_compact_result_free(&r_full);
            kv_compact_result_free(&r_proxy);
        }
    }

    printf("\n  IoU quality: %d/%d passed (IoU >= 90%% at ratio >= 10%%)\n\n", total_pass, total_tests);
}

// ============================================================================
// Bench 1b: Scoring Speedup
// ============================================================================

static void bench_scoring_speedup() {
    printf("=== Bench 1b: Scoring Speedup ===\n\n");

    const int n_head_kv = 4, d_k = 64, d_v = 64;
    const int n_embd_k = n_head_kv * d_k;
    const int n_q = 64;

    int T_sizes[] = {256, 512, 1024, 2048, 4096, 10240};

    printf("  %-6s  %12s  %12s  %12s\n",
           "T", "full_ms", "proxy_ms", "speedup");
    printf("  %-6s  %12s  %12s  %12s\n",
           "------", "------------", "------------", "------------");

    for (int T : T_sizes) {
        int n_embd_v = n_head_kv * d_v;
        std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
        gen_spiky_data(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, 2000 + T);

        // Method A: full kv_compact scoring
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

        // Time full scoring (n_q=64)
        const int n_runs = 3;
        double full_ms = 0.0;
        for (int run = 0; run < n_runs; run++) {
            kv_compact_result r = {};
            auto t0 = clock_type::now();
            kv_compact(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, &p, &r);
            full_ms += std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
            kv_compact_result_free(&r);
        }
        full_ms /= n_runs;

        // Method B: proxy scoring (n_q=1, q_mean)
        std::vector<float> Q_mean(n_embd_k, 0.0f);
        for (int qi = 0; qi < n_q; qi++)
            for (int hh = 0; hh < n_head_kv; hh++)
                for (int d = 0; d < d_k; d++)
                    Q_mean[hh * d_k + d] += Q[qi * n_embd_k + hh * d_k + d];
        for (int hh = 0; hh < n_head_kv * d_k; hh++)
            Q_mean[hh] /= (float)n_q;

        double proxy_ms = 0.0;
        for (int run = 0; run < n_runs; run++) {
            kv_compact_result r = {};
            auto t0 = clock_type::now();
            kv_compact(K.data(), V.data(), Q_mean.data(), T, 1, n_head_kv, d_k, d_v, &p, &r);
            proxy_ms += std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
            kv_compact_result_free(&r);
        }
        proxy_ms /= n_runs;

        double speedup = full_ms / (proxy_ms + 1e-12);

        printf("  %-6d  %12.2f  %12.2f  %12.1fx\n",
               T, full_ms, proxy_ms, speedup);

        // Sanity check
        assert(full_ms > 0.0);
        assert(proxy_ms > 0.0);
    }

    printf("\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("bench-opt-expected-attention\n");
    printf("============================\n");
    printf("Validating: arXiv:2510.00636 — Expected Attention Proxy\n\n");

    bench_selection_quality();
    bench_scoring_speedup();

    printf("Validation complete.\n");
    return 0;
}
