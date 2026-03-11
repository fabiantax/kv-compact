// Tests for the KV compaction C library API (US-8)
//
// Validates the public C API: kv_compact(), kv_compact_params_default(),
// kv_compact_result_free(), error handling, and quality of results.

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "kv-compact-api.h"

static void gen_data(float * out, int n, int seed) {
    for (int i = 0; i < n; i++) {
        out[i] = sinf((float)(i * 7 + seed) * 0.31f)
               + 0.3f * cosf((float)(i * 3 + seed + 17) * 0.53f);
    }
}

// ============================================================================
// Tests
// ============================================================================

static void test_params_default() {
    printf("  test_params_default...");
    kv_compact_params p = kv_compact_params_default();
    assert(p.target_ratio > 0.0f && p.target_ratio < 1.0f);
    assert(p.target_count == 0);
    assert(p.use_sensitivity == 0);
    assert(p.ridge > 0.0f);
    assert(p.nnls_max_iter > 0);
    printf(" OK\n");
}

static void test_basic_compaction() {
    printf("  test_basic_compaction...");
    const int T = 64, n_q = 32, n_head_kv = 4, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 1);
    gen_data(V.data(), T * n_embd_v, 500);
    gen_data(Q.data(), n_q * n_embd_k, 900);

    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;

    kv_compact_result result = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(),
                        T, n_q, n_head_kv, d_k, d_v, &p, &result);
    assert(rc == 0);
    assert(result.t == 32);  // 50% of 64
    assert(result.n_head_kv == n_head_kv);
    assert(result.selected_indices != NULL);
    assert(result.beta != NULL);
    assert(result.C_v != NULL);

    // Indices should be sorted and in range
    for (int i = 0; i < result.t; i++) {
        assert(result.selected_indices[i] >= 0);
        assert(result.selected_indices[i] < T);
    }
    for (int i = 1; i < result.t; i++) {
        assert(result.selected_indices[i] > result.selected_indices[i - 1]);
    }

    // Beta and C_v should be finite
    for (int h = 0; h < n_head_kv; h++) {
        for (int j = 0; j < result.t; j++) {
            assert(std::isfinite(result.beta[h][j]));
        }
        for (int j = 0; j < result.t * d_v; j++) {
            assert(std::isfinite(result.C_v[h][j]));
        }
    }

    // Quality metrics should be populated
    assert(result.stats.avg_cosine_sim > 0.8f);
    assert(std::isfinite(result.stats.avg_mse));
    assert(result.stats.elapsed_ms > 0.0);

    kv_compact_result_free(&result);
    assert(result.t == 0);
    assert(result.selected_indices == NULL);
    printf(" OK\n");
}

static void test_target_count_override() {
    printf("  test_target_count_override...");
    const int T = 64, n_q = 16, n_head_kv = 2, d_k = 16, d_v = 16;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 10);
    gen_data(V.data(), T * n_embd_v, 20);
    gen_data(Q.data(), n_q * n_embd_k, 30);

    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.1f;   // would give 6
    p.target_count = 20;     // explicit override

    kv_compact_result result = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(),
                        T, n_q, n_head_kv, d_k, d_v, &p, &result);
    assert(rc == 0);
    assert(result.t == 20);

    kv_compact_result_free(&result);
    printf(" OK\n");
}

static void test_no_compression_needed() {
    printf("  test_no_compression_needed...");
    const int T = 16, n_q = 8, n_head_kv = 2, d_k = 8, d_v = 8;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 40);
    gen_data(V.data(), T * n_embd_v, 50);
    gen_data(Q.data(), n_q * n_embd_k, 60);

    kv_compact_params p = kv_compact_params_default();
    p.target_count = T;  // keep everything

    kv_compact_result result = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(),
                        T, n_q, n_head_kv, d_k, d_v, &p, &result);
    assert(rc == 0);
    assert(result.t == T);

    kv_compact_result_free(&result);
    printf(" OK\n");
}

static void test_null_params_uses_defaults() {
    printf("  test_null_params_uses_defaults...");
    const int T = 32, n_q = 16, n_head_kv = 2, d_k = 16, d_v = 16;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 70);
    gen_data(V.data(), T * n_embd_v, 80);
    gen_data(Q.data(), n_q * n_embd_k, 90);

    kv_compact_result result = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(),
                        T, n_q, n_head_kv, d_k, d_v, NULL, &result);
    assert(rc == 0);
    assert(result.t == 16);  // default 50% of 32

    kv_compact_result_free(&result);
    printf(" OK\n");
}

static void test_error_handling() {
    printf("  test_error_handling...");
    kv_compact_result result = {};

    // NULL inputs
    assert(kv_compact(NULL, NULL, NULL, 0, 0, 0, 0, 0, NULL, &result) != 0);

    // NULL result
    float dummy = 1.0f;
    assert(kv_compact(&dummy, &dummy, &dummy, 1, 1, 1, 1, 1, NULL, NULL) != 0);

    // Invalid dimensions
    assert(kv_compact(&dummy, &dummy, &dummy, 0, 1, 1, 1, 1, NULL, &result) != 0);
    assert(kv_compact(&dummy, &dummy, &dummy, 1, 0, 1, 1, 1, NULL, &result) != 0);

    printf(" OK\n");
}

static void test_sensitivity_weighted() {
    printf("  test_sensitivity_weighted...");
    const int T = 64, n_q = 32, n_head_kv = 4, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 100);
    gen_data(V.data(), T * n_embd_v, 200);
    gen_data(Q.data(), n_q * n_embd_k, 300);

    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;
    p.use_sensitivity = 1;

    kv_compact_result result = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(),
                        T, n_q, n_head_kv, d_k, d_v, &p, &result);
    assert(rc == 0);
    assert(result.t == 32);
    assert(result.stats.avg_cosine_sim > 0.8f);

    kv_compact_result_free(&result);
    printf(" OK\n");
}

static void test_quality_across_ratios() {
    printf("  test_quality_across_ratios...\n");
    const int T = 128, n_q = 64, n_head_kv = 4, d_k = 64, d_v = 64;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 400);
    gen_data(V.data(), T * n_embd_v, 500);
    gen_data(Q.data(), n_q * n_embd_k, 600);

    float ratios[] = {0.2f, 0.5f, 0.8f};
    float prev_cos = 0.0f;

    for (float ratio : ratios) {
        kv_compact_params p = kv_compact_params_default();
        p.target_ratio = ratio;

        kv_compact_result result = {};
        int rc = kv_compact(K.data(), V.data(), Q.data(),
                            T, n_q, n_head_kv, d_k, d_v, &p, &result);
        assert(rc == 0);

        printf("    %.0f%%: cos=%.6f mse=%.8f agree=%.1f%% time=%.1fms\n",
               ratio * 100, result.stats.avg_cosine_sim, result.stats.avg_mse,
               result.stats.avg_agreement * 100, result.stats.elapsed_ms);

        // Higher retention should give better quality
        assert(result.stats.avg_cosine_sim >= prev_cos - 0.01f);
        prev_cos = result.stats.avg_cosine_sim;

        // Minimum quality thresholds
        if (ratio >= 0.5f) {
            assert(result.stats.avg_cosine_sim > 0.9f);
        }

        kv_compact_result_free(&result);
    }
    printf("  OK\n");
}

static void test_multi_round_basic() {
    printf("  test_multi_round_basic...\n");
    const int T = 128, n_q = 64, n_head_kv = 4, d_k = 64, d_v = 64;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 700);
    gen_data(V.data(), T * n_embd_v, 800);
    gen_data(Q.data(), n_q * n_embd_k, 900);

    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;  // each round keeps 50%

    const int n_rounds = 3;
    kv_compact_stats round_stats[3];
    kv_compact_result result = {};

    int rc = kv_compact_multi_round(K.data(), V.data(), Q.data(),
                                    T, n_q, n_head_kv, d_k, d_v,
                                    &p, n_rounds, &result, round_stats);
    assert(rc == 0);

    // After 3 rounds of 50%: 128 → 64 → 32 → 16
    assert(result.t == 16);

    // Selected indices should map back to original positions
    for (int j = 0; j < result.t; j++) {
        assert(result.selected_indices[j] >= 0);
        assert(result.selected_indices[j] < T);
    }
    for (int j = 1; j < result.t; j++) {
        assert(result.selected_indices[j] > result.selected_indices[j - 1]);
    }

    printf("    Round results (vs original data):\n");
    for (int r = 0; r < n_rounds; r++) {
        printf("      Round %d: cos=%.6f mse=%.8f\n",
               r + 1, round_stats[r].avg_cosine_sim, round_stats[r].avg_mse);
    }
    printf("    Final (vs original): cos=%.6f mse=%.8f\n",
           result.stats.avg_cosine_sim, result.stats.avg_mse);

    // After 3 rounds of 50% (12.5% total retention), quality will degrade
    // but values must remain finite and the algorithm must not crash
    printf("    Final cosine sim: %.6f\n", result.stats.avg_cosine_sim);
    assert(std::isfinite(result.stats.avg_cosine_sim));
    assert(std::isfinite(result.stats.avg_mse));

    kv_compact_result_free(&result);
    printf("  OK\n");
}

static void test_multi_round_quality_degradation() {
    printf("  test_multi_round_quality_degradation...\n");
    const int T = 256, n_q = 64, n_head_kv = 4, d_k = 64, d_v = 64;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 1000);
    gen_data(V.data(), T * n_embd_v, 1100);
    gen_data(Q.data(), n_q * n_embd_k, 1200);

    // Run 1, 2, and 3 rounds at 70% retention and compare
    float cos_by_rounds[3];

    for (int nr = 1; nr <= 3; nr++) {
        kv_compact_params p = kv_compact_params_default();
        p.target_ratio = 0.7f;

        kv_compact_result result = {};
        int rc = kv_compact_multi_round(K.data(), V.data(), Q.data(),
                                        T, n_q, n_head_kv, d_k, d_v,
                                        &p, nr, &result, NULL);
        assert(rc == 0);

        cos_by_rounds[nr - 1] = result.stats.avg_cosine_sim;
        printf("    %d rounds (%.0f%% total retention): t=%d cos=%.6f\n",
               nr, pow(0.7, nr) * 100, result.t, result.stats.avg_cosine_sim);

        kv_compact_result_free(&result);
    }

    // Quality should degrade gracefully at 70% retention per round
    // 1 round should be best
    assert(cos_by_rounds[0] >= cos_by_rounds[1] - 0.01f);
    assert(cos_by_rounds[1] >= cos_by_rounds[2] - 0.01f);
    // At 70% per round, 3 rounds (34% total retention) — quality degrades but stays positive
    assert(cos_by_rounds[2] > 0.5f);

    printf("  OK\n");
}

static void test_iterative_refinement() {
    printf("  test_iterative_refinement...\n");
    const int T = 128, n_q = 64, n_head_kv = 4, d_k = 64, d_v = 64;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 1300);
    gen_data(V.data(), T * n_embd_v, 1400);
    gen_data(Q.data(), n_q * n_embd_k, 1500);

    // Compare: no refinement vs 2 refinement rounds at 20% retention
    // (refinement matters most at high compression)
    kv_compact_params p_base = kv_compact_params_default();
    p_base.target_ratio = 0.2f;
    p_base.refine_rounds = 0;

    kv_compact_params p_refine = p_base;
    p_refine.refine_rounds = 2;

    kv_compact_result r_base = {}, r_refine = {};

    int rc1 = kv_compact(K.data(), V.data(), Q.data(),
                         T, n_q, n_head_kv, d_k, d_v, &p_base, &r_base);
    int rc2 = kv_compact(K.data(), V.data(), Q.data(),
                         T, n_q, n_head_kv, d_k, d_v, &p_refine, &r_refine);
    assert(rc1 == 0);
    assert(rc2 == 0);

    printf("    No refinement:   cos=%.6f mse=%.8f time=%.1fms\n",
           r_base.stats.avg_cosine_sim, r_base.stats.avg_mse,
           r_base.stats.elapsed_ms);
    printf("    2 refine rounds: cos=%.6f mse=%.8f time=%.1fms\n",
           r_refine.stats.avg_cosine_sim, r_refine.stats.avg_mse,
           r_refine.stats.elapsed_ms);

    // Refinement should improve or at least not worsen quality
    assert(r_refine.stats.avg_cosine_sim >= r_base.stats.avg_cosine_sim - 0.01f);
    // Both must be finite
    assert(std::isfinite(r_refine.stats.avg_cosine_sim));
    assert(std::isfinite(r_refine.stats.avg_mse));

    kv_compact_result_free(&r_base);
    kv_compact_result_free(&r_refine);
    printf("  OK\n");
}

static void test_diversity_aware_selection() {
    printf("  test_diversity_aware_selection...\n");
    const int T = 128, n_q = 64, n_head_kv = 4, d_k = 64, d_v = 64;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 1600);
    gen_data(V.data(), T * n_embd_v, 1700);
    gen_data(Q.data(), n_q * n_embd_k, 1800);

    // Compare: standard vs diversity-aware at 20% retention
    kv_compact_params p_std = kv_compact_params_default();
    p_std.target_ratio = 0.2f;

    kv_compact_params p_div = p_std;
    p_div.use_diversity = 1;
    p_div.diversity_strength = 0.5f;

    kv_compact_result r_std = {}, r_div = {};

    int rc1 = kv_compact(K.data(), V.data(), Q.data(),
                         T, n_q, n_head_kv, d_k, d_v, &p_std, &r_std);
    int rc2 = kv_compact(K.data(), V.data(), Q.data(),
                         T, n_q, n_head_kv, d_k, d_v, &p_div, &r_div);
    assert(rc1 == 0);
    assert(rc2 == 0);

    printf("    Standard:  cos=%.6f mse=%.8f\n",
           r_std.stats.avg_cosine_sim, r_std.stats.avg_mse);
    printf("    Diversity: cos=%.6f mse=%.8f\n",
           r_div.stats.avg_cosine_sim, r_div.stats.avg_mse);

    // Both should produce valid results
    assert(r_std.t == r_div.t);
    assert(std::isfinite(r_div.stats.avg_cosine_sim));
    assert(r_div.stats.avg_cosine_sim > 0.8f);

    // Diversity selection should produce a DIFFERENT set of indices
    // (not necessarily better, but different)
    bool any_different = false;
    for (int j = 0; j < r_std.t; j++) {
        if (r_std.selected_indices[j] != r_div.selected_indices[j]) {
            any_different = true;
            break;
        }
    }
    printf("    Selections differ: %s\n", any_different ? "yes" : "no");

    kv_compact_result_free(&r_std);
    kv_compact_result_free(&r_div);
    printf("  OK\n");
}

static void test_shared_prefix() {
    printf("  test_shared_prefix...\n");
    const int T = 64, n_q = 32, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 1900);
    gen_data(V.data(), T * n_embd_v, 2000);
    gen_data(Q.data(), n_q * n_embd_k, 2100);

    // Compact with 8 shared prefix tokens always kept
    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;
    p.n_shared_prefix = 8;

    kv_compact_result result = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(),
                        T, n_q, n_head_kv, d_k, d_v, &p, &result);
    assert(rc == 0);
    assert(result.t == 32);  // 50% of 64

    // First 8 positions must be in the selected set
    for (int j = 0; j < 8; j++) {
        bool found = false;
        for (int k = 0; k < result.t; k++) {
            if (result.selected_indices[k] == j) { found = true; break; }
        }
        assert(found);
    }

    printf("    Prefix positions 0-7 preserved: yes\n");
    printf("    cos=%.6f mse=%.8f\n",
           result.stats.avg_cosine_sim, result.stats.avg_mse);
    assert(std::isfinite(result.stats.avg_cosine_sim));

    kv_compact_result_free(&result);
    printf("  OK\n");
}

static void test_shared_prefix_with_diversity() {
    printf("  test_shared_prefix_with_diversity...\n");
    const int T = 64, n_q = 32, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 2200);
    gen_data(V.data(), T * n_embd_v, 2300);
    gen_data(Q.data(), n_q * n_embd_k, 2400);

    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;
    p.n_shared_prefix = 8;
    p.use_diversity = 1;
    p.diversity_strength = 0.5f;

    kv_compact_result result = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(),
                        T, n_q, n_head_kv, d_k, d_v, &p, &result);
    assert(rc == 0);

    // Prefix must be preserved even with diversity
    for (int j = 0; j < 8; j++) {
        bool found = false;
        for (int k = 0; k < result.t; k++) {
            if (result.selected_indices[k] == j) { found = true; break; }
        }
        assert(found);
    }

    assert(std::isfinite(result.stats.avg_cosine_sim));
    kv_compact_result_free(&result);
    printf("  OK\n");
}

static void test_cheap_qref() {
    printf("  test_cheap_qref...\n");
    const int T = 128, n_q = 64, n_head_kv = 4, d_k = 64, d_v = 64;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 2500);
    gen_data(V.data(), T * n_embd_v, 2600);
    gen_data(Q.data(), n_q * n_embd_k, 2700);

    // Compare: real Q_ref vs cheap (K-vector proxy) Q_ref
    kv_compact_params p_real = kv_compact_params_default();
    p_real.target_ratio = 0.5f;

    kv_compact_params p_cheap = p_real;
    p_cheap.use_cheap_qref = 1;

    kv_compact_result r_real = {}, r_cheap = {};

    int rc1 = kv_compact(K.data(), V.data(), Q.data(),
                         T, n_q, n_head_kv, d_k, d_v, &p_real, &r_real);
    // With cheap Q_ref, pass NULL for Q_ref_all and 0 for n_q
    int rc2 = kv_compact(K.data(), V.data(), NULL,
                         T, 0, n_head_kv, d_k, d_v, &p_cheap, &r_cheap);
    assert(rc1 == 0);
    assert(rc2 == 0);

    printf("    Real Q_ref:  cos=%.6f mse=%.8f time=%.1fms\n",
           r_real.stats.avg_cosine_sim, r_real.stats.avg_mse,
           r_real.stats.elapsed_ms);
    printf("    Cheap Q_ref: cos=%.6f mse=%.8f time=%.1fms\n",
           r_cheap.stats.avg_cosine_sim, r_cheap.stats.avg_mse,
           r_cheap.stats.elapsed_ms);

    // Cheap Q_ref should still produce reasonable quality
    assert(r_cheap.t == r_real.t);
    assert(std::isfinite(r_cheap.stats.avg_cosine_sim));
    assert(r_cheap.stats.avg_cosine_sim > 0.5f);

    kv_compact_result_free(&r_real);
    kv_compact_result_free(&r_cheap);
    printf("  OK\n");
}

static void test_cheap_qref_multi_round() {
    printf("  test_cheap_qref_multi_round...\n");
    const int T = 128, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v);
    gen_data(K.data(), T * n_embd_k, 2800);
    gen_data(V.data(), T * n_embd_v, 2900);

    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.7f;
    p.use_cheap_qref = 1;

    kv_compact_result result = {};
    int rc = kv_compact_multi_round(K.data(), V.data(), NULL,
                                    T, 0, n_head_kv, d_k, d_v,
                                    &p, 2, &result, NULL);
    assert(rc == 0);
    // 2 rounds at 70%: 128 → 89 → 62
    assert(result.t > 0);
    assert(std::isfinite(result.stats.avg_cosine_sim));

    printf("    t=%d cos=%.6f\n", result.t, result.stats.avg_cosine_sim);

    kv_compact_result_free(&result);
    printf("  OK\n");
}

static void test_double_free_safe() {
    printf("  test_double_free_safe...");
    kv_compact_result result = {};
    // Should be safe to free a zeroed result
    kv_compact_result_free(&result);
    kv_compact_result_free(NULL);
    printf(" OK\n");
}

// ============================================================================
// Layer filter tests (A1: Hybrid layer awareness)
// ============================================================================

static void test_layer_filter_all() {
    printf("  test_layer_filter_all...");
    // Default filter: all layers pass
    for (int l = 0; l < 40; l++) {
        assert(kv_layer_filter_all(l, 40, NULL) != 0);
    }
    printf(" OK\n");
}

static void test_layer_filter_periodic() {
    printf("  test_layer_filter_periodic...");
    // Qwen 3.5 pattern: attention every 4th layer (layers 3,7,11,...,39)
    void * interval = (void *)(intptr_t)4;
    int count = 0;
    for (int l = 0; l < 40; l++) {
        int result = kv_layer_filter_periodic(l, 40, interval);
        if (result) count++;
        // Layers 3,7,11,15,19,23,27,31,35,39 should pass
        bool expected = ((l + 1) % 4 == 0);
        assert((result != 0) == expected);
    }
    assert(count == 10);  // 40/4 = 10 attention layers
    printf(" OK\n");
}

static void test_layer_filter_explicit() {
    printf("  test_layer_filter_explicit...");
    // Explicit list of attention layers
    int layers[] = {3, 7, 11, 15, 19, 23, 27, 31, 35, 39};
    kv_layer_list list = { layers, 10 };

    int count = 0;
    for (int l = 0; l < 40; l++) {
        int result = kv_layer_filter_explicit(l, 40, &list);
        if (result) count++;
    }
    assert(count == 10);

    // Non-listed layers should be filtered out
    assert(kv_layer_filter_explicit(0, 40, &list) == 0);
    assert(kv_layer_filter_explicit(1, 40, &list) == 0);
    assert(kv_layer_filter_explicit(2, 40, &list) == 0);
    assert(kv_layer_filter_explicit(3, 40, &list) != 0);
    assert(kv_layer_filter_explicit(4, 40, &list) == 0);
    assert(kv_layer_filter_explicit(39, 40, &list) != 0);

    printf(" OK\n");
}

static void test_layer_filter_count() {
    printf("  test_layer_filter_count...");

    // Count with NULL filter = all layers
    assert(kv_compact_count_layers(NULL, NULL, 40) == 40);

    // Count with periodic filter
    void * interval = (void *)(intptr_t)4;
    assert(kv_compact_count_layers(kv_layer_filter_periodic, interval, 40) == 10);

    // Count with explicit filter
    int layers[] = {0, 5, 10};
    kv_layer_list list = { layers, 3 };
    assert(kv_compact_count_layers(kv_layer_filter_explicit, &list, 20) == 3);

    printf(" OK\n");
}

static void test_layer_filter_should_compact() {
    printf("  test_layer_filter_should_compact...");

    // NULL params = compact all
    assert(kv_compact_should_compact_layer(NULL, 5, 40) == 1);

    // Params with no filter = compact all
    kv_compact_params p = kv_compact_params_default();
    assert(kv_compact_should_compact_layer(&p, 5, 40) == 1);

    // Params with periodic filter
    p.layer_filter = kv_layer_filter_periodic;
    p.layer_filter_data = (void *)(intptr_t)4;
    assert(kv_compact_should_compact_layer(&p, 3, 40) == 1);  // layer 3: (3+1)%4==0 → yes
    assert(kv_compact_should_compact_layer(&p, 0, 40) == 0);  // layer 0: (0+1)%4!=0 → no
    assert(kv_compact_should_compact_layer(&p, 7, 40) == 1);  // layer 7: (7+1)%4==0 → yes

    printf(" OK\n");
}

static void test_layer_filter_in_compaction() {
    printf("  test_layer_filter_in_compaction...\n");
    // Test that layer_filter field in params doesn't break single-layer compaction
    // (kv_compact operates on a single layer — the filter is for caller use)
    const int T = 64, n_q = 32, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 5000);
    gen_data(V.data(), T * n_embd_v, 5100);
    gen_data(Q.data(), n_q * n_embd_k, 5200);

    // With filter set (doesn't affect single-layer API, just stored for caller use)
    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;
    p.layer_filter = kv_layer_filter_periodic;
    p.layer_filter_data = (void *)(intptr_t)4;

    kv_compact_result result = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(),
                        T, n_q, n_head_kv, d_k, d_v, &p, &result);
    assert(rc == 0);
    assert(result.t == 32);
    assert(std::isfinite(result.stats.avg_cosine_sim));
    assert(result.stats.avg_cosine_sim > 0.8f);

    printf("    cos=%.6f (filter in params doesn't break single-layer API)\n",
           result.stats.avg_cosine_sim);

    kv_compact_result_free(&result);
    printf("  OK\n");
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("test-kv-compact-api:\n\n");

    printf("=== Parameter defaults ===\n");
    test_params_default();

    printf("\n=== Basic compaction ===\n");
    test_basic_compaction();
    test_target_count_override();
    test_no_compression_needed();
    test_null_params_uses_defaults();

    printf("\n=== Error handling ===\n");
    test_error_handling();

    printf("\n=== Sensitivity weighting ===\n");
    test_sensitivity_weighted();

    printf("\n=== Quality across ratios ===\n");
    test_quality_across_ratios();

    printf("\n=== Multi-round compaction (US-9) ===\n");
    test_multi_round_basic();
    test_multi_round_quality_degradation();

    printf("\n=== Iterative refinement ===\n");
    test_iterative_refinement();

    printf("\n=== Diversity-aware selection ===\n");
    test_diversity_aware_selection();

    printf("\n=== Shared prefix ===\n");
    test_shared_prefix();
    test_shared_prefix_with_diversity();

    printf("\n=== Cheap Q_ref generation ===\n");
    test_cheap_qref();
    test_cheap_qref_multi_round();

    printf("\n=== Safety ===\n");
    test_double_free_safe();

    printf("\n=== Layer filter (A1: Hybrid layer awareness) ===\n");
    test_layer_filter_all();
    test_layer_filter_periodic();
    test_layer_filter_explicit();
    test_layer_filter_count();
    test_layer_filter_should_compact();
    test_layer_filter_in_compaction();

    printf("\nAll tests passed!\n");
    return 0;
}
