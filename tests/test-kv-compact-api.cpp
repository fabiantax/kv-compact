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
    // At 70% per round, even 3 rounds (34% total) should maintain decent quality
    assert(cos_by_rounds[2] > 0.7f);

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

    printf("\n=== Safety ===\n");
    test_double_free_safe();

    printf("\nAll tests passed!\n");
    return 0;
}
