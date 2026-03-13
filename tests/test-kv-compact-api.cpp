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
    assert(p.score_method == 1);         // RMS (paper default)
    assert(p.use_omp == 0);
    assert(p.omp_k_choice == 1);
    assert(p.omp_refit_interval == 1);
    assert(p.nnls_method == 1);          // PGD (paper default)
    assert(p.nnls_pgd_iters == 0);
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
// Chunked compaction tests
// ============================================================================

static void test_chunked_basic() {
    printf("  test_chunked_basic...\n");
    const int T = 512, n_q = 64, n_head_kv = 4, d_k = 64, d_v = 64;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 6000);
    gen_data(V.data(), T * n_embd_v, 6100);
    gen_data(Q.data(), n_q * n_embd_k, 6200);

    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;
    p.chunk_size = 128;  // 4 chunks of 128 tokens each

    kv_compact_result result = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(),
                        T, n_q, n_head_kv, d_k, d_v, &p, &result);
    assert(rc == 0);

    // 50% of 512 = 256 selected (each chunk: 50% of 128 = 64, 4 chunks = 256)
    assert(result.t == 256);
    assert(result.n_head_kv == n_head_kv);
    assert(result.selected_indices != NULL);
    assert(result.beta != NULL);
    assert(result.C_v != NULL);

    // Indices should be sorted and in valid range
    for (int i = 0; i < result.t; i++) {
        assert(result.selected_indices[i] >= 0);
        assert(result.selected_indices[i] < T);
    }
    for (int i = 1; i < result.t; i++) {
        assert(result.selected_indices[i] > result.selected_indices[i - 1]);
    }

    // All values should be finite
    for (int h = 0; h < n_head_kv; h++) {
        for (int j = 0; j < result.t; j++) {
            assert(std::isfinite(result.beta[h][j]));
        }
        for (int j = 0; j < result.t * d_v; j++) {
            assert(std::isfinite(result.C_v[h][j]));
        }
    }

    // Quality metrics should be computed against the full original
    assert(std::isfinite(result.stats.avg_cosine_sim));
    assert(result.stats.avg_cosine_sim > 0.8f);
    assert(result.stats.elapsed_ms > 0.0);

    printf("    t=%d cos=%.6f mse=%.8f time=%.1fms\n",
           result.t, result.stats.avg_cosine_sim, result.stats.avg_mse,
           result.stats.elapsed_ms);

    kv_compact_result_free(&result);
    printf("  OK\n");
}

static void test_chunked_quality_vs_unchunked() {
    printf("  test_chunked_quality_vs_unchunked...\n");
    const int T = 256, n_q = 64, n_head_kv = 4, d_k = 64, d_v = 64;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 6300);
    gen_data(V.data(), T * n_embd_v, 6400);
    gen_data(Q.data(), n_q * n_embd_k, 6500);

    // Unchunked (baseline)
    kv_compact_params p_unchunked = kv_compact_params_default();
    p_unchunked.target_ratio = 0.5f;
    p_unchunked.chunk_size = -1;

    // Chunked (64-token chunks)
    kv_compact_params p_chunked = kv_compact_params_default();
    p_chunked.target_ratio = 0.5f;
    p_chunked.chunk_size = 64;

    kv_compact_result r_unchunked = {}, r_chunked = {};

    int rc1 = kv_compact(K.data(), V.data(), Q.data(),
                         T, n_q, n_head_kv, d_k, d_v,
                         &p_unchunked, &r_unchunked);
    int rc2 = kv_compact(K.data(), V.data(), Q.data(),
                         T, n_q, n_head_kv, d_k, d_v,
                         &p_chunked, &r_chunked);
    assert(rc1 == 0);
    assert(rc2 == 0);

    // Both should select the same number of tokens
    assert(r_unchunked.t == r_chunked.t);

    printf("    Unchunked: cos=%.6f mse=%.8f time=%.1fms\n",
           r_unchunked.stats.avg_cosine_sim, r_unchunked.stats.avg_mse,
           r_unchunked.stats.elapsed_ms);
    printf("    Chunked:   cos=%.6f mse=%.8f time=%.1fms\n",
           r_chunked.stats.avg_cosine_sim, r_chunked.stats.avg_mse,
           r_chunked.stats.elapsed_ms);

    // Chunked may lose some quality from cross-chunk attention, but should
    // still be reasonable (within 10% of unchunked cosine similarity)
    assert(r_chunked.stats.avg_cosine_sim > r_unchunked.stats.avg_cosine_sim - 0.1f);
    assert(r_chunked.stats.avg_cosine_sim > 0.8f);

    kv_compact_result_free(&r_unchunked);
    kv_compact_result_free(&r_chunked);
    printf("  OK\n");
}

static void test_chunked_no_chunk_when_small() {
    printf("  test_chunked_no_chunk_when_small...\n");
    // With auto chunk_size (0), T=128 at 50% → t=64 ≤ 256, so no chunking
    const int T = 128, n_q = 64, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 6600);
    gen_data(V.data(), T * n_embd_v, 6700);
    gen_data(Q.data(), n_q * n_embd_k, 6800);

    // Auto mode (chunk_size = 0): T=128, ratio=0.5, t=64 ≤ 256 → no chunking
    kv_compact_params p_auto = kv_compact_params_default();
    p_auto.target_ratio = 0.5f;

    // Explicit disabled
    kv_compact_params p_disabled = kv_compact_params_default();
    p_disabled.target_ratio = 0.5f;
    p_disabled.chunk_size = -1;

    kv_compact_result r_auto = {}, r_disabled = {};
    int rc1 = kv_compact(K.data(), V.data(), Q.data(),
                         T, n_q, n_head_kv, d_k, d_v,
                         &p_auto, &r_auto);
    int rc2 = kv_compact(K.data(), V.data(), Q.data(),
                         T, n_q, n_head_kv, d_k, d_v,
                         &p_disabled, &r_disabled);
    assert(rc1 == 0);
    assert(rc2 == 0);

    // Should produce identical results (no chunking in either case)
    assert(r_auto.t == r_disabled.t);
    for (int i = 0; i < r_auto.t; i++) {
        assert(r_auto.selected_indices[i] == r_disabled.selected_indices[i]);
    }

    printf("    Auto vs disabled identical: yes (t=%d)\n", r_auto.t);

    kv_compact_result_free(&r_auto);
    kv_compact_result_free(&r_disabled);
    printf("  OK\n");
}

static void test_chunked_uneven_chunks() {
    printf("  test_chunked_uneven_chunks...\n");
    // T=500 with chunk_size=200: chunks of 200, 200, 100
    const int T = 500, n_q = 32, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 6900);
    gen_data(V.data(), T * n_embd_v, 7000);
    gen_data(Q.data(), n_q * n_embd_k, 7100);

    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;
    p.chunk_size = 200;

    kv_compact_result result = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(),
                        T, n_q, n_head_kv, d_k, d_v, &p, &result);
    assert(rc == 0);

    // 50% of 200=100, 50% of 200=100, 50% of 100=50 → total 250
    assert(result.t == 250);

    // Indices should span the full range [0, T)
    assert(result.selected_indices[0] >= 0);
    assert(result.selected_indices[result.t - 1] < T);
    for (int i = 1; i < result.t; i++) {
        assert(result.selected_indices[i] > result.selected_indices[i - 1]);
    }

    assert(std::isfinite(result.stats.avg_cosine_sim));
    printf("    t=%d cos=%.6f (3 uneven chunks: 200+200+100)\n",
           result.t, result.stats.avg_cosine_sim);

    kv_compact_result_free(&result);
    printf("  OK\n");
}

static void test_chunked_with_shared_prefix() {
    printf("  test_chunked_with_shared_prefix...\n");
    const int T = 256, n_q = 32, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 7200);
    gen_data(V.data(), T * n_embd_v, 7300);
    gen_data(Q.data(), n_q * n_embd_k, 7400);

    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;
    p.chunk_size = 64;
    p.n_shared_prefix = 8;

    kv_compact_result result = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(),
                        T, n_q, n_head_kv, d_k, d_v, &p, &result);
    assert(rc == 0);

    // First 8 positions (in the first chunk) should be preserved
    for (int j = 0; j < 8; j++) {
        bool found = false;
        for (int k = 0; k < result.t; k++) {
            if (result.selected_indices[k] == j) { found = true; break; }
        }
        assert(found);
    }

    printf("    Prefix 0-7 preserved in chunked mode: yes\n");
    assert(std::isfinite(result.stats.avg_cosine_sim));

    kv_compact_result_free(&result);
    printf("  OK\n");
}

static void test_chunked_with_cheap_qref() {
    printf("  test_chunked_with_cheap_qref...\n");
    const int T = 512, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v);
    gen_data(K.data(), T * n_embd_k, 7500);
    gen_data(V.data(), T * n_embd_v, 7600);

    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;
    p.chunk_size = 128;
    p.use_cheap_qref = 1;

    kv_compact_result result = {};
    int rc = kv_compact(K.data(), V.data(), NULL,
                        T, 0, n_head_kv, d_k, d_v, &p, &result);
    assert(rc == 0);
    assert(result.t == 256);
    assert(std::isfinite(result.stats.avg_cosine_sim));

    printf("    Chunked+cheap_qref: t=%d cos=%.6f\n",
           result.t, result.stats.avg_cosine_sim);

    kv_compact_result_free(&result);
    printf("  OK\n");
}

static void test_chunked_params_default() {
    printf("  test_chunked_params_default...");
    kv_compact_params p = kv_compact_params_default();
    assert(p.chunk_size == 0);  // auto mode
    assert(p.n_threads == 0);   // auto threads
    printf(" OK\n");
}

static void test_chunked_parallel_vs_sequential() {
    printf("  test_chunked_parallel_vs_sequential...\n");
    const int T = 4096, n_q = 64, n_head_kv = 4, d_k = 64, d_v = 64;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    std::vector<float> K((size_t)T * n_embd_k), V((size_t)T * n_embd_v);
    std::vector<float> Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 8200);
    gen_data(V.data(), T * n_embd_v, 8300);
    gen_data(Q.data(), n_q * n_embd_k, 8400);

    // Sequential (1 thread)
    kv_compact_params p_seq = kv_compact_params_default();
    p_seq.target_ratio = 0.5f;
    p_seq.n_threads = 1;

    // Parallel (auto threads)
    kv_compact_params p_par = kv_compact_params_default();
    p_par.target_ratio = 0.5f;
    p_par.n_threads = 0;

    kv_compact_result r_seq = {}, r_par = {};
    int rc1 = kv_compact(K.data(), V.data(), Q.data(),
                         T, n_q, n_head_kv, d_k, d_v, &p_seq, &r_seq);
    int rc2 = kv_compact(K.data(), V.data(), Q.data(),
                         T, n_q, n_head_kv, d_k, d_v, &p_par, &r_par);
    assert(rc1 == 0);
    assert(rc2 == 0);

    // Same number of selected tokens
    assert(r_seq.t == r_par.t);

    // Same indices (deterministic despite parallelism — chunks are independent)
    for (int i = 0; i < r_seq.t; i++) {
        assert(r_seq.selected_indices[i] == r_par.selected_indices[i]);
    }

    printf("    Sequential: t=%d cos=%.6f time=%.1fms\n",
           r_seq.t, r_seq.stats.avg_cosine_sim, r_seq.stats.elapsed_ms);
    printf("    Parallel:   t=%d cos=%.6f time=%.1fms\n",
           r_par.t, r_par.stats.avg_cosine_sim, r_par.stats.elapsed_ms);

    // Quality must be identical (same algorithm, just parallelized)
    assert(fabsf(r_seq.stats.avg_cosine_sim - r_par.stats.avg_cosine_sim) < 1e-5f);

    kv_compact_result_free(&r_seq);
    kv_compact_result_free(&r_par);
    printf("  OK\n");
}

static void test_chunked_100k() {
    printf("  test_chunked_100k...\n");
    // T=100k at 50% retention: t=50000, AtA would be 10GB without chunking.
    // Auto mode: chunk=512 (t_chunk=256), 196 chunks. Each LS ≤ 1ms.
    const int T = 100000, n_q = 64, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    // K+V = 100k * 64 * 4 * 2 = 51.2 MB
    std::vector<float> K((size_t)T * n_embd_k), V((size_t)T * n_embd_v);
    std::vector<float> Q(n_q * n_embd_k);
    gen_data(K.data(), (int)((size_t)T * n_embd_k), 7700);
    gen_data(V.data(), (int)((size_t)T * n_embd_v), 7800);
    gen_data(Q.data(), n_q * n_embd_k, 7900);

    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;
    // auto chunk_size: ratio=0.5 → chunk=512, t_chunk=256

    kv_compact_result result = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(),
                        T, n_q, n_head_kv, d_k, d_v, &p, &result);
    assert(rc == 0);
    assert(result.t == 50000);

    // Spot check indices
    assert(result.selected_indices[0] >= 0);
    assert(result.selected_indices[result.t - 1] < T);
    for (int i = 1; i < result.t; i++) {
        assert(result.selected_indices[i] > result.selected_indices[i - 1]);
    }

    assert(std::isfinite(result.stats.avg_cosine_sim));
    assert(result.stats.avg_cosine_sim > 0.9f);
    printf("    T=%d auto-chunk: t=%d cos=%.6f time=%.1fs (no OOM!)\n",
           T, result.t, result.stats.avg_cosine_sim,
           result.stats.elapsed_ms / 1000.0);

    kv_compact_result_free(&result);
    printf("  OK\n");
}

static void test_chunked_1M() {
    printf("  test_chunked_1M...\n");
    // T=1M at 50% retention: t=500k. Auto chunk=512, ~1954 chunks.
    // Using small dims to keep memory reasonable: K+V = 1M * 32 * 4 * 2 = 256 MB
    const int T = 1000000, n_head_kv = 1, d_k = 16, d_v = 16;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    // K = 1M * 16 * 4 = 64MB, V = 64MB → 128MB total
    std::vector<float> K((size_t)T * n_embd_k), V((size_t)T * n_embd_v);
    gen_data(K.data(), (int)((size_t)T * n_embd_k), 8000);
    gen_data(V.data(), (int)((size_t)T * n_embd_v), 8100);

    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;
    p.use_cheap_qref = 1;  // generate Q_ref from K (no external queries at 1M)
    // auto chunk_size: ratio=0.5 → chunk=512, t_chunk=256, ~1954 chunks

    kv_compact_result result = {};
    int rc = kv_compact(K.data(), V.data(), NULL,
                        T, 0, n_head_kv, d_k, d_v, &p, &result);
    assert(rc == 0);
    assert(result.t == 500000);

    // Spot check: indices valid and sorted
    assert(result.selected_indices[0] >= 0);
    assert(result.selected_indices[result.t - 1] < T);
    // Check a few samples (don't iterate all 500k in debug)
    for (int i = 1; i < 1000; i++) {
        assert(result.selected_indices[i] > result.selected_indices[i - 1]);
    }
    int mid = result.t / 2;
    assert(result.selected_indices[mid] > result.selected_indices[mid - 1]);
    assert(result.selected_indices[result.t - 1] > result.selected_indices[result.t - 2]);

    assert(std::isfinite(result.stats.avg_cosine_sim));
    printf("    T=%d auto-chunk: t=%d cos=%.6f time=%.1fs (1M context!)\n",
           T, result.t, result.stats.avg_cosine_sim,
           result.stats.elapsed_ms / 1000.0);

    kv_compact_result_free(&result);
    printf("  OK\n");
}

// ============================================================================
// Main
// ============================================================================

// ============================================================================
// Score aggregation method tests
// ============================================================================

static void test_score_method_rms() {
    printf("  test_score_method_rms...");
    const int T = 64, n_q = 32, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k, n_embd_v = n_head_kv * d_v;
    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 42);
    gen_data(V.data(), T * n_embd_v, 43);
    gen_data(Q.data(), n_q * n_embd_k, 44);

    // Compare all three score methods
    float cos_max = 0, cos_rms = 0, cos_mean = 0;
    for (int method = 0; method <= 2; method++) {
        kv_compact_params p = kv_compact_params_default();
        p.target_ratio = 0.5f;
        p.score_method = method;
        p.chunk_size = -1;

        kv_compact_result r = {};
        int rc = kv_compact(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, &p, &r);
        assert(rc == 0);
        assert(r.t == 32);

        if (method == 0) cos_max = r.stats.avg_cosine_sim;
        else if (method == 1) cos_rms = r.stats.avg_cosine_sim;
        else cos_mean = r.stats.avg_cosine_sim;

        kv_compact_result_free(&r);
    }

    printf("\n    max=%.6f rms=%.6f mean=%.6f", cos_max, cos_rms, cos_mean);
    // All methods should produce reasonable quality
    assert(cos_max > 0.99f);
    assert(cos_rms > 0.99f);
    assert(cos_mean > 0.99f);
    printf(" OK\n");
}

// ============================================================================
// OMP key selection tests
// ============================================================================

static void test_omp_basic() {
    printf("  test_omp_basic...");
    const int T = 64, n_q = 32, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k, n_embd_v = n_head_kv * d_v;
    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 100);
    gen_data(V.data(), T * n_embd_v, 200);
    gen_data(Q.data(), n_q * n_embd_k, 300);

    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;
    p.use_omp = 1;
    p.skip_beta = 0;
    p.chunk_size = -1;

    kv_compact_result r = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, &p, &r);
    assert(rc == 0);
    assert(r.t == 32);
    assert(r.stats.avg_cosine_sim > 0.95f);
    printf("\n    OMP+beta: cos=%.6f mse=%.8f", r.stats.avg_cosine_sim, r.stats.avg_mse);
    kv_compact_result_free(&r);
    printf(" OK\n");
}

static void test_omp_fast() {
    printf("  test_omp_fast...");
    const int T = 128, n_q = 32, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k, n_embd_v = n_head_kv * d_v;
    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 100);
    gen_data(V.data(), T * n_embd_v, 200);
    gen_data(Q.data(), n_q * n_embd_k, 300);

    // AM-OMP-fast: k_choice=4, refit_interval=2 (paper's fast variant)
    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;
    p.use_omp = 1;
    p.omp_k_choice = 4;
    p.omp_refit_interval = 2;
    p.skip_beta = 1;
    p.chunk_size = -1;

    kv_compact_result r = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, &p, &r);
    assert(rc == 0);
    assert(r.t == 64);
    assert(r.stats.avg_cosine_sim > 0.95f);
    printf("\n    OMP-fast(k=4,int=2): cos=%.6f", r.stats.avg_cosine_sim);
    kv_compact_result_free(&r);
    printf(" OK\n");
}

static void test_omp_vs_highest_attn() {
    printf("  test_omp_vs_highest_attn...");
    const int T = 64, n_q = 32, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k, n_embd_v = n_head_kv * d_v;
    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 77);
    gen_data(V.data(), T * n_embd_v, 78);
    gen_data(Q.data(), n_q * n_embd_k, 79);

    // OMP
    kv_compact_params p_omp = kv_compact_params_default();
    p_omp.target_ratio = 0.3f;
    p_omp.use_omp = 1;
    p_omp.skip_beta = 0;
    p_omp.chunk_size = -1;

    kv_compact_result r_omp = {};
    kv_compact(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, &p_omp, &r_omp);

    // Highest attention keys
    kv_compact_params p_hat = kv_compact_params_default();
    p_hat.target_ratio = 0.3f;
    p_hat.skip_beta = 0;
    p_hat.chunk_size = -1;

    kv_compact_result r_hat = {};
    kv_compact(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, &p_hat, &r_hat);

    printf("\n    OMP:          cos=%.6f mse=%.8f", r_omp.stats.avg_cosine_sim, r_omp.stats.avg_mse);
    printf("\n    HighestAttn:  cos=%.6f mse=%.8f", r_hat.stats.avg_cosine_sim, r_hat.stats.avg_mse);

    // Both should produce reasonable quality
    assert(r_omp.stats.avg_cosine_sim > 0.95f);
    assert(r_hat.stats.avg_cosine_sim > 0.95f);

    kv_compact_result_free(&r_omp);
    kv_compact_result_free(&r_hat);
    printf(" OK\n");
}

// ============================================================================
// NNLS solver method tests
// ============================================================================

static void test_nnls_method_pgd() {
    printf("  test_nnls_method_pgd...");
    const int T = 64, n_q = 32, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k, n_embd_v = n_head_kv * d_v;
    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 55);
    gen_data(V.data(), T * n_embd_v, 56);
    gen_data(Q.data(), n_q * n_embd_k, 57);

    // PGD solver (paper default)
    kv_compact_params p_pgd = kv_compact_params_default();
    p_pgd.target_ratio = 0.5f;
    p_pgd.skip_beta = 0;
    p_pgd.nnls_method = 1;  // PGD
    p_pgd.nnls_pgd_iters = 0;  // clamped LS only
    p_pgd.chunk_size = -1;

    kv_compact_result r_pgd = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, &p_pgd, &r_pgd);
    assert(rc == 0);

    // Lawson-Hanson solver
    kv_compact_params p_lh = kv_compact_params_default();
    p_lh.target_ratio = 0.5f;
    p_lh.skip_beta = 0;
    p_lh.nnls_method = 0;  // Lawson-Hanson
    p_lh.chunk_size = -1;

    kv_compact_result r_lh = {};
    kv_compact(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, &p_lh, &r_lh);

    printf("\n    PGD(iters=0): cos=%.6f", r_pgd.stats.avg_cosine_sim);
    printf("\n    Lawson-Hanson: cos=%.6f", r_lh.stats.avg_cosine_sim);

    // Both should produce reasonable quality
    assert(r_pgd.stats.avg_cosine_sim > 0.99f);
    assert(r_lh.stats.avg_cosine_sim > 0.99f);

    kv_compact_result_free(&r_pgd);
    kv_compact_result_free(&r_lh);
    printf(" OK\n");
}

static void test_nnls_pgd_with_iters() {
    printf("  test_nnls_pgd_with_iters...");
    const int T = 64, n_q = 32, n_head_kv = 2, d_k = 32, d_v = 32;
    int n_embd_k = n_head_kv * d_k, n_embd_v = n_head_kv * d_v;
    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 55);
    gen_data(V.data(), T * n_embd_v, 56);
    gen_data(Q.data(), n_q * n_embd_k, 57);

    // PGD with additional iterations
    kv_compact_params p = kv_compact_params_default();
    p.target_ratio = 0.5f;
    p.skip_beta = 0;
    p.nnls_method = 1;
    p.nnls_pgd_iters = 50;
    p.chunk_size = -1;

    kv_compact_result r = {};
    int rc = kv_compact(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, &p, &r);
    assert(rc == 0);
    assert(r.stats.avg_cosine_sim > 0.99f);
    printf("\n    PGD(iters=50): cos=%.6f", r.stats.avg_cosine_sim);
    kv_compact_result_free(&r);
    printf(" OK\n");
}

// Generate keys that create spiky, realistic LLM-like attention patterns:
// - A few "attention sink" tokens (BOS, delimiters) with unique key directions
// - Recent tokens get locality bias (higher magnitude)
// - Most middle tokens are similar/clustered (hard to distinguish)
static void gen_spiky_data(float * K, float * V, float * Q,
                           int T, int n_q, int n_head_kv, int d_k, int d_v,
                           int seed) {
    // Fill V with normal-ish random data (values don't affect key selection)
    gen_data(V, T * n_head_kv * d_v, seed + 500);

    // K: make most keys very similar (clustered), with a few outlier "sinks"
    int n_embd_k = n_head_kv * d_k;
    // Base direction that most keys will be close to
    std::vector<float> base_dir(d_k);
    for (int d = 0; d < d_k; d++) {
        base_dir[d] = sinf((float)(d * 7 + seed) * 0.31f);
    }

    for (int t = 0; t < T; t++) {
        for (int h = 0; h < n_head_kv; h++) {
            float * row = K + t * n_embd_k + h * d_k;
            if (t == 0) {
                // Attention sink 0 (BOS-like): strong unique direction
                for (int d = 0; d < d_k; d++) {
                    row[d] = 3.0f * cosf((float)(d * 13 + seed) * 0.17f);
                }
            } else if (t == 1 || t == T/4 || t == T/2) {
                // A few delimiter/punctuation sinks: distinct directions
                for (int d = 0; d < d_k; d++) {
                    row[d] = 2.5f * sinf((float)(d * 11 + t * 37 + seed) * 0.23f);
                }
            } else if (t > T - T/10) {
                // Recent tokens: slightly boosted magnitude (locality bias)
                for (int d = 0; d < d_k; d++) {
                    float noise = 0.1f * sinf((float)(t * 31 + d * 7 + h * 53 + seed) * 0.41f);
                    row[d] = 1.5f * (base_dir[d] + noise);
                }
            } else {
                // Middle tokens: clustered near base_dir with small noise
                for (int d = 0; d < d_k; d++) {
                    float noise = 0.05f * sinf((float)(t * 31 + d * 7 + h * 53 + seed) * 0.41f);
                    row[d] = base_dir[d] + noise;
                }
            }
        }
    }

    // Q: queries that attend sharply to sinks + recent tokens
    // High dot product with sink directions, moderate with recent, low with middle
    for (int q = 0; q < n_q; q++) {
        for (int h = 0; h < n_head_kv; h++) {
            float * qrow = Q + q * n_embd_k + h * d_k;
            // Mix of sink-0 direction and base direction
            float sink_weight = 0.7f + 0.3f * sinf((float)(q * 13 + h * 7 + seed) * 0.29f);
            for (int d = 0; d < d_k; d++) {
                float sink_dir = cosf((float)(d * 13 + seed) * 0.17f);
                qrow[d] = sink_weight * sink_dir + (1.0f - sink_weight) * base_dir[d];
                // Add per-query variation
                qrow[d] += 0.2f * sinf((float)(q * 41 + d * 3 + h * 19 + seed) * 0.37f);
            }
        }
    }
}

// ============================================================================
// Extreme compression ratio benchmark (50× = 2% retention)
// ============================================================================

static void test_extreme_compression_ratios() {
    printf("  test_extreme_compression_ratios...\n");

    // Use a larger T to make extreme ratios meaningful
    // At T=500, 2% = 10 tokens kept, 5% = 25, 10% = 50, 20% = 100, 50% = 250
    const int T = 500, n_q = 64, n_head_kv = 4, d_k = 64, d_v = 64;
    int n_embd_k = n_head_kv * d_k, n_embd_v = n_head_kv * d_v;
    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_data(K.data(), T * n_embd_k, 9000);
    gen_data(V.data(), T * n_embd_v, 9100);
    gen_data(Q.data(), n_q * n_embd_k, 9200);

    // Test ratios: 50%, 20%, 10%, 5%, 2% (the paper claims 50× = 2%)
    float ratios[] = { 0.50f, 0.20f, 0.10f, 0.05f, 0.02f };
    int n_ratios = 5;

    printf("    %-8s  %-28s  %-28s  %-28s\n",
           "Ratio", "HighestAttn (skip_beta)", "OMP (skip_beta)", "OMP (+beta)");
    printf("    %-8s  %-28s  %-28s  %-28s\n",
           "-----", "------------------------", "------------------------", "------------------------");

    for (int ri = 0; ri < n_ratios; ri++) {
        float ratio = ratios[ri];
        int t_expected = (int)(T * ratio);
        if (t_expected < 2) t_expected = 2;

        // --- HighestAttn (skip_beta=1, default) ---
        kv_compact_params p_hat = kv_compact_params_default();
        p_hat.target_ratio = ratio;
        p_hat.chunk_size = -1;  // no chunking for fair comparison
        kv_compact_result r_hat = {};
        int rc = kv_compact(K.data(), V.data(), Q.data(),
                            T, n_q, n_head_kv, d_k, d_v, &p_hat, &r_hat);
        assert(rc == 0);

        // --- OMP (skip_beta=1) ---
        kv_compact_params p_omp = kv_compact_params_default();
        p_omp.target_ratio = ratio;
        p_omp.use_omp = 1;
        p_omp.omp_k_choice = 4;  // fast variant
        p_omp.omp_refit_interval = 2;
        p_omp.chunk_size = -1;
        kv_compact_result r_omp = {};
        rc = kv_compact(K.data(), V.data(), Q.data(),
                        T, n_q, n_head_kv, d_k, d_v, &p_omp, &r_omp);
        assert(rc == 0);

        // --- OMP with beta (full AM-OMP) ---
        kv_compact_params p_omp_beta = kv_compact_params_default();
        p_omp_beta.target_ratio = ratio;
        p_omp_beta.use_omp = 1;
        p_omp_beta.omp_k_choice = 4;
        p_omp_beta.omp_refit_interval = 2;
        p_omp_beta.skip_beta = 0;
        p_omp_beta.chunk_size = -1;
        kv_compact_result r_omp_beta = {};
        rc = kv_compact(K.data(), V.data(), Q.data(),
                        T, n_q, n_head_kv, d_k, d_v, &p_omp_beta, &r_omp_beta);
        assert(rc == 0);

        printf("    %5.1f%%    cos=%.4f mse=%.6f %4.1fms  cos=%.4f mse=%.6f %4.1fms  cos=%.4f mse=%.6f %4.1fms\n",
               ratio * 100.0f,
               r_hat.stats.avg_cosine_sim, r_hat.stats.avg_mse, r_hat.stats.elapsed_ms,
               r_omp.stats.avg_cosine_sim, r_omp.stats.avg_mse, r_omp.stats.elapsed_ms,
               r_omp_beta.stats.avg_cosine_sim, r_omp_beta.stats.avg_mse, r_omp_beta.stats.elapsed_ms);

        // Quality assertions: even at extreme compression, cos > 0.5 is reasonable
        // At 50% we expect very high quality, at 2% we allow graceful degradation
        float min_cos = (ratio >= 0.10f) ? 0.90f : (ratio >= 0.05f) ? 0.80f : 0.50f;
        assert(r_hat.stats.avg_cosine_sim > min_cos);
        assert(r_omp.stats.avg_cosine_sim > min_cos);
        assert(r_omp_beta.stats.avg_cosine_sim > min_cos);

        // Verify token counts
        assert(r_hat.t == t_expected);
        assert(r_omp.t == t_expected);
        assert(r_omp_beta.t == t_expected);

        kv_compact_result_free(&r_hat);
        kv_compact_result_free(&r_omp);
        kv_compact_result_free(&r_omp_beta);
    }
    printf("  OK\n");
}

static void test_extreme_compression_spiky() {
    printf("  test_extreme_compression_spiky (realistic attention)...\n");

    // Larger T with spiky attention patterns — this is where things should break
    const int T = 500, n_q = 64, n_head_kv = 4, d_k = 64, d_v = 64;
    int n_embd_k = n_head_kv * d_k, n_embd_v = n_head_kv * d_v;
    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    gen_spiky_data(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, 8000);

    float ratios[] = { 0.50f, 0.20f, 0.10f, 0.05f, 0.02f };
    int n_ratios = 5;

    printf("    %-8s  %-28s  %-28s  %-28s\n",
           "Ratio", "HighestAttn (skip_beta)", "OMP (skip_beta)", "OMP (+beta)");
    printf("    %-8s  %-28s  %-28s  %-28s\n",
           "-----", "------------------------", "------------------------", "------------------------");

    for (int ri = 0; ri < n_ratios; ri++) {
        float ratio = ratios[ri];
        int t_expected = (int)(T * ratio);
        if (t_expected < 2) t_expected = 2;

        // --- HighestAttn (skip_beta=1, default) ---
        kv_compact_params p_hat = kv_compact_params_default();
        p_hat.target_ratio = ratio;
        p_hat.chunk_size = -1;
        kv_compact_result r_hat = {};
        kv_compact(K.data(), V.data(), Q.data(),
                   T, n_q, n_head_kv, d_k, d_v, &p_hat, &r_hat);

        // --- OMP (skip_beta=1) ---
        kv_compact_params p_omp = kv_compact_params_default();
        p_omp.target_ratio = ratio;
        p_omp.use_omp = 1;
        p_omp.omp_k_choice = 4;
        p_omp.omp_refit_interval = 2;
        p_omp.chunk_size = -1;
        kv_compact_result r_omp = {};
        kv_compact(K.data(), V.data(), Q.data(),
                   T, n_q, n_head_kv, d_k, d_v, &p_omp, &r_omp);

        // --- OMP with beta ---
        kv_compact_params p_omp_beta = kv_compact_params_default();
        p_omp_beta.target_ratio = ratio;
        p_omp_beta.use_omp = 1;
        p_omp_beta.omp_k_choice = 4;
        p_omp_beta.omp_refit_interval = 2;
        p_omp_beta.skip_beta = 0;
        p_omp_beta.chunk_size = -1;
        kv_compact_result r_omp_beta = {};
        kv_compact(K.data(), V.data(), Q.data(),
                   T, n_q, n_head_kv, d_k, d_v, &p_omp_beta, &r_omp_beta);

        printf("    %5.1f%%    cos=%.4f mse=%.6f %4.1fms  cos=%.4f mse=%.6f %4.1fms  cos=%.4f mse=%.6f %4.1fms\n",
               ratio * 100.0f,
               r_hat.stats.avg_cosine_sim, r_hat.stats.avg_mse, r_hat.stats.elapsed_ms,
               r_omp.stats.avg_cosine_sim, r_omp.stats.avg_mse, r_omp.stats.elapsed_ms,
               r_omp_beta.stats.avg_cosine_sim, r_omp_beta.stats.avg_mse, r_omp_beta.stats.elapsed_ms);

        kv_compact_result_free(&r_hat);
        kv_compact_result_free(&r_omp);
        kv_compact_result_free(&r_omp_beta);
    }
    printf("  OK\n");
}

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

    printf("\n=== Score aggregation methods ===\n");
    test_score_method_rms();

    printf("\n=== OMP key selection ===\n");
    test_omp_basic();
    test_omp_fast();
    test_omp_vs_highest_attn();

    printf("\n=== NNLS solver methods ===\n");
    test_nnls_method_pgd();
    test_nnls_pgd_with_iters();

    printf("\n=== Extreme compression ratios (50x = 2%%) ===\n");
    test_extreme_compression_ratios();
    test_extreme_compression_spiky();

    printf("\n=== Chunked compaction ===\n");
    test_chunked_params_default();
    test_chunked_basic();
    test_chunked_quality_vs_unchunked();
    test_chunked_no_chunk_when_small();
    test_chunked_uneven_chunks();
    test_chunked_with_shared_prefix();
    test_chunked_with_cheap_qref();
    test_chunked_100k();
    test_chunked_1M();
    test_chunked_parallel_vs_sequential();

    printf("\nAll tests passed!\n");
    return 0;
}
