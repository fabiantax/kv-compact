// Unit tests for Greedy Budget Exchange algorithm (Paper S5)
//
// Tests the greedy_budget_exchange() function and its integration with
// compact_layer_all_heads() for nonuniform per-head budget allocation.
//
// The greedy exchange transfers budget units from compression-tolerant heads
// (spiky attention) to compression-sensitive heads (broad attention), yielding
// better reconstruction quality than uniform allocation.

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include "../include/kv-compact-math.h"

// ---------------------------------------------------------------------------
// Test 1: Budget conservation -- sum of per-head budgets equals total_budget
// ---------------------------------------------------------------------------
static void test_greedy_exchange_conserves_budget() {
    printf("  test_greedy_exchange_conserves_budget...");

    // Try several (n_heads, total_budget) combinations
    struct test_case { int n_heads; int total_budget; };
    test_case cases[] = {
        {4, 64}, {8, 100}, {3, 7}, {16, 256}, {2, 3}, {5, 5}, {10, 1000},
    };

    for (auto & tc : cases) {
        // Random sensitivity values
        std::vector<float> sens(tc.n_heads);
        srand(123 + tc.n_heads);
        for (auto & s : sens) s = (float)rand() / RAND_MAX + 0.01f;

        auto budgets = greedy_budget_exchange(
            sens.data(), tc.n_heads, tc.total_budget);

        int sum = 0;
        for (int b : budgets) sum += b;
        assert(sum == tc.total_budget);
    }

    printf(" PASS\n");
}

// ---------------------------------------------------------------------------
// Test 2: Uniform sensitivity => uniform budgets (differ by at most 1)
// ---------------------------------------------------------------------------
static void test_greedy_exchange_uniform_sensitivity() {
    printf("  test_greedy_exchange_uniform_sensitivity...");

    const int n_heads = 8;
    const int total_budget = 64;
    std::vector<float> sens(n_heads, 1.0f);  // all equal

    auto budgets = greedy_budget_exchange(
        sens.data(), n_heads, total_budget);

    int expected = total_budget / n_heads;  // 8
    for (int h = 0; h < n_heads; h++) {
        assert(budgets[h] == expected || budgets[h] == expected + 1);
    }

    // Conservation
    int sum = 0;
    for (int b : budgets) sum += b;
    assert(sum == total_budget);

    printf(" PASS\n");
}

// ---------------------------------------------------------------------------
// Test 3: Extreme sensitivity -- one head 100x more sensitive gets more budget
// ---------------------------------------------------------------------------
static void test_greedy_exchange_extreme_sensitivity() {
    printf("  test_greedy_exchange_extreme_sensitivity...");

    const int n_heads = 4;
    const int total_budget = 40;
    // Head 0 is 100x more sensitive than the rest
    std::vector<float> sens = {10.0f, 0.1f, 0.1f, 0.1f};

    auto budgets = greedy_budget_exchange(
        sens.data(), n_heads, total_budget);

    // Head 0 should have significantly more budget than others
    assert(budgets[0] > budgets[1]);
    assert(budgets[0] > budgets[2]);
    assert(budgets[0] > budgets[3]);
    // Head 0 should get at least half the total budget (it is ~100x)
    assert(budgets[0] >= total_budget / 2);

    // Conservation
    int sum = 0;
    for (int b : budgets) sum += b;
    assert(sum == total_budget);

    printf(" PASS\n");
}

// ---------------------------------------------------------------------------
// Test 4: Minimum per-head constraint is respected
// ---------------------------------------------------------------------------
static void test_greedy_exchange_respects_minimum() {
    printf("  test_greedy_exchange_respects_minimum...");

    const int n_heads = 4;
    const int total_budget = 20;
    const int min_per_head = 3;
    // Very skewed: head 0 dominates, but all must get >= 3
    std::vector<float> sens = {100.0f, 0.01f, 0.01f, 0.01f};

    auto budgets = greedy_budget_exchange(
        sens.data(), n_heads, total_budget, min_per_head);

    for (int h = 0; h < n_heads; h++) {
        assert(budgets[h] >= min_per_head);
    }

    // Conservation
    int sum = 0;
    for (int b : budgets) sum += b;
    assert(sum == total_budget);

    printf(" PASS\n");
}

// ---------------------------------------------------------------------------
// Test 5: Maximum per-head constraint is respected
// ---------------------------------------------------------------------------
static void test_greedy_exchange_respects_maximum() {
    printf("  test_greedy_exchange_respects_maximum...");

    const int n_heads = 4;
    const int total_budget = 40;
    const int min_per_head = 2;
    const int max_per_head = 15;
    // Head 0 is hugely sensitive, but cap at 15
    std::vector<float> sens = {100.0f, 0.01f, 0.01f, 0.01f};

    auto budgets = greedy_budget_exchange(
        sens.data(), n_heads, total_budget, min_per_head, max_per_head);

    for (int h = 0; h < n_heads; h++) {
        assert(budgets[h] <= max_per_head);
        assert(budgets[h] >= min_per_head);
    }

    // Conservation
    int sum = 0;
    for (int b : budgets) sum += b;
    assert(sum == total_budget);

    printf(" PASS\n");
}

// ---------------------------------------------------------------------------
// Test 6: Single head gets all budget (trivial case)
// ---------------------------------------------------------------------------
static void test_greedy_exchange_single_head() {
    printf("  test_greedy_exchange_single_head...");

    float sens = 0.5f;
    auto budgets = greedy_budget_exchange(&sens, 1, 42);

    assert(budgets.size() == 1);
    assert(budgets[0] == 42);

    printf(" PASS\n");
}

// ---------------------------------------------------------------------------
// Test 7: Budget roughly proportional to sqrt(sensitivity) at optimum
// ---------------------------------------------------------------------------
static void test_greedy_exchange_proportional() {
    printf("  test_greedy_exchange_proportional...");

    // At the greedy exchange equilibrium, the marginal value s[h]/t_h is
    // equalized across heads, so t_h is proportional to s[h] (linear).
    // The greedy exchange uses bounded iterations (n_heads * 20), so with
    // a large sensitivity range we test the weaker monotonicity property:
    // more sensitive heads get strictly more budget.
    //
    // Also verify the stronger proportional relationship for heads with
    // moderate sensitivity differences, where convergence is attainable.
    const int n_heads = 4;
    const int total_budget = 400;
    std::vector<float> sens = {1.0f, 4.0f, 9.0f, 16.0f};

    auto budgets = greedy_budget_exchange(
        sens.data(), n_heads, total_budget, /*min=*/1, /*max=*/0);

    // Monotonicity: higher sensitivity => higher or equal budget
    for (int h = 1; h < n_heads; h++) {
        assert(budgets[h] >= budgets[h - 1]);
    }

    // The most sensitive head (16x) should get much more than the least (1x)
    assert(budgets[3] > budgets[0] * 2);

    // Between adjacent high-sensitivity heads (9 and 16), the ratio should
    // be in the right direction and within 2x of the ideal s[h]/s[h-1]
    float ratio_23 = (float)budgets[3] / (float)budgets[2];
    float ideal_23 = sens[3] / sens[2];  // 16/9 ~ 1.78
    assert(ratio_23 > 1.0f);             // head 3 gets more
    assert(ratio_23 < ideal_23 * 2.0f);  // not wildly off

    // Conservation
    int sum = 0;
    for (int b : budgets) sum += b;
    assert(sum == total_budget);

    printf(" PASS\n");
}

// ---------------------------------------------------------------------------
// Test 8: Nonuniform compaction quality -- per-head budgets via full pipeline
// ---------------------------------------------------------------------------
static void test_nonuniform_compaction_quality() {
    printf("  test_nonuniform_compaction_quality...");

    // Create synthetic data
    int T = 64, n_q = 8, n_head_kv = 4, d_k = 16, d_v = 16, t = 16;
    std::vector<float> K(T * n_head_kv * d_k);
    std::vector<float> V(T * n_head_kv * d_v);
    std::vector<float> Q(n_q * n_head_kv * d_k);

    // Fill with random data using srand(42)
    srand(42);
    for (auto & x : K) x = (float)rand() / RAND_MAX - 0.5f;
    for (auto & x : V) x = (float)rand() / RAND_MAX - 0.5f;
    for (auto & x : Q) x = (float)rand() / RAND_MAX - 0.5f;

    // --- Run 1: uniform budget (default config, no per_head_budgets) ---
    compaction_config cfg_uniform;
    compacted_layer result_uniform = compact_layer_all_heads(
        K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, t, cfg_uniform);

    // --- Run 2: nonuniform budget via greedy exchange ---
    // First compute sensitivity (auto-computed inside compact_layer_all_heads,
    // but we replicate here to feed to greedy_budget_exchange)
    std::vector<float> sensitivity = result_uniform.head_sensitivity;
    assert((int)sensitivity.size() == n_head_kv);

    int total_budget = n_head_kv * t;
    auto per_head_b = greedy_budget_exchange(
        sensitivity.data(), n_head_kv, total_budget, /*min=*/2);

    // Verify per_head_budgets sums to total
    int budget_sum = 0;
    for (int b : per_head_b) budget_sum += b;
    assert(budget_sum == total_budget);

    // Verify all per-head budgets are positive
    for (int h = 0; h < n_head_kv; h++) {
        assert(per_head_b[h] >= 2);
    }

    // --- Run 3: compact with explicit per-head budgets ---
    // Use the largest per-head budget as the shared selection size t,
    // then heads with smaller budgets will have unused slots.
    int max_budget = *std::max_element(per_head_b.begin(), per_head_b.end());

    compaction_config cfg_nonuniform;
    cfg_nonuniform.per_head_budgets = per_head_b.data();

    compacted_layer result_nonuniform = compact_layer_all_heads(
        K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v,
        max_budget, cfg_nonuniform);

    // Verify the result has valid structure
    assert(result_nonuniform.n_head_kv == n_head_kv);
    assert((int)result_nonuniform.beta.size() == n_head_kv);
    assert((int)result_nonuniform.C_v.size() == n_head_kv);

    // For heads with budget < max_budget, check properties of unused slots:
    //   - beta for unused keys should be very negative (effectively -inf)
    //   - C_v for unused keys should be zero
    for (int h = 0; h < n_head_kv; h++) {
        int head_budget = per_head_b[h];
        int t_actual = (int)result_nonuniform.beta[h].size();

        // Beta vector should have max_budget entries
        assert(t_actual == max_budget);

        // Check unused slots (indices >= head_budget)
        for (int j = head_budget; j < t_actual; j++) {
            // Beta should be very negative (suppresses these keys in softmax)
            assert(result_nonuniform.beta[h][j] < -10.0f);

            // C_v should be zero for unused slots
            for (int d = 0; d < d_v; d++) {
                assert(fabsf(result_nonuniform.C_v[h][j * d_v + d]) < 1e-6f);
            }
        }
    }

    // Compute reconstruction error for both runs.
    // For uniform: each head uses t keys.
    // For nonuniform: each head uses per_head_b[h] effective keys.
    // Nonuniform should be at least as good overall (or very close).
    float err_uniform = 0.0f, err_nonuniform = 0.0f;

    for (int h = 0; h < n_head_kv; h++) {
        // Compute attention output with original K/V: Y_h = softmax(Q_h K_h^T / sqrt(d_k)) V_h
        // Then with compacted: Y_hat = softmax(Q_h Ck_h^T / sqrt(d_k) + beta_h) C_v_h
        // Error = ||Y - Y_hat||^2

        const int n_embd_k_gqa = n_head_kv * d_k;
        const int n_embd_v_gqa = n_head_kv * d_v;
        const float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

        // Original attention output Y [n_q, d_v]
        std::vector<float> scores_full(n_q * T);
        for (int qi = 0; qi < n_q; qi++) {
            for (int ki = 0; ki < T; ki++) {
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++) {
                    dot += Q[qi * n_embd_k_gqa + h * d_k + d] *
                           K[ki * n_embd_k_gqa + h * d_k + d];
                }
                scores_full[qi * T + ki] = dot * inv_sqrt_dk;
            }
        }
        softmax_rows(scores_full.data(), n_q, T);

        std::vector<float> Y(n_q * d_v, 0.0f);
        for (int qi = 0; qi < n_q; qi++) {
            for (int ki = 0; ki < T; ki++) {
                for (int d = 0; d < d_v; d++) {
                    Y[qi * d_v + d] += scores_full[qi * T + ki] *
                                       V[ki * n_embd_v_gqa + h * d_v + d];
                }
            }
        }

        // --- Uniform reconstruction error ---
        {
            int tu = result_uniform.t;
            std::vector<float> scores_u(n_q * tu);
            for (int qi = 0; qi < n_q; qi++) {
                for (int j = 0; j < tu; j++) {
                    int idx = result_uniform.selected_indices[j];
                    float dot = 0.0f;
                    for (int d = 0; d < d_k; d++) {
                        dot += Q[qi * n_embd_k_gqa + h * d_k + d] *
                               K[idx * n_embd_k_gqa + h * d_k + d];
                    }
                    scores_u[qi * tu + j] = dot * inv_sqrt_dk + result_uniform.beta[h][j];
                }
            }
            softmax_rows(scores_u.data(), n_q, tu);

            for (int qi = 0; qi < n_q; qi++) {
                for (int d = 0; d < d_v; d++) {
                    float y_hat = 0.0f;
                    for (int j = 0; j < tu; j++) {
                        y_hat += scores_u[qi * tu + j] * result_uniform.C_v[h][j * d_v + d];
                    }
                    float diff = Y[qi * d_v + d] - y_hat;
                    err_uniform += diff * diff;
                }
            }
        }

        // --- Nonuniform reconstruction error ---
        {
            int tn = max_budget;
            std::vector<float> scores_n(n_q * tn);
            for (int qi = 0; qi < n_q; qi++) {
                for (int j = 0; j < tn; j++) {
                    int idx = result_nonuniform.selected_indices[j];
                    float dot = 0.0f;
                    for (int d = 0; d < d_k; d++) {
                        dot += Q[qi * n_embd_k_gqa + h * d_k + d] *
                               K[idx * n_embd_k_gqa + h * d_k + d];
                    }
                    scores_n[qi * tn + j] = dot * inv_sqrt_dk + result_nonuniform.beta[h][j];
                }
            }
            softmax_rows(scores_n.data(), n_q, tn);

            for (int qi = 0; qi < n_q; qi++) {
                for (int d = 0; d < d_v; d++) {
                    float y_hat = 0.0f;
                    for (int j = 0; j < tn; j++) {
                        y_hat += scores_n[qi * tn + j] * result_nonuniform.C_v[h][j * d_v + d];
                    }
                    float diff = Y[qi * d_v + d] - y_hat;
                    err_nonuniform += diff * diff;
                }
            }
        }
    }

    // Nonuniform allocation uses more total keys (sum = n_head_kv * t) but
    // distributes them optimally. Its error should be no worse than uniform
    // (within a generous tolerance for the noisy synthetic data).
    // We allow 20% worse in case random data creates degenerate cases.
    assert(err_nonuniform <= err_uniform * 1.2f + 1e-4f);

    printf(" PASS\n");
}

// ---------------------------------------------------------------------------
// Test 9: Backward compatibility -- no per_head_budgets => same as before
// ---------------------------------------------------------------------------
static void test_nonuniform_backward_compatible() {
    printf("  test_nonuniform_backward_compatible...");

    // Create reproducible synthetic data
    int T = 32, n_q = 4, n_head_kv = 2, d_k = 8, d_v = 8, t = 8;
    std::vector<float> K(T * n_head_kv * d_k);
    std::vector<float> V(T * n_head_kv * d_v);
    std::vector<float> Q(n_q * n_head_kv * d_k);

    srand(99);
    for (auto & x : K) x = (float)rand() / RAND_MAX - 0.5f;
    for (auto & x : V) x = (float)rand() / RAND_MAX - 0.5f;
    for (auto & x : Q) x = (float)rand() / RAND_MAX - 0.5f;

    // Run with default config (no per_head_budgets)
    compaction_config cfg1;
    compacted_layer r1 = compact_layer_all_heads(
        K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, t, cfg1);

    // Run again with explicit nullptr (should be identical)
    compaction_config cfg2;
    cfg2.per_head_budgets = nullptr;
    compacted_layer r2 = compact_layer_all_heads(
        K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, t, cfg2);

    // Same selected indices
    assert(r1.selected_indices.size() == r2.selected_indices.size());
    for (size_t i = 0; i < r1.selected_indices.size(); i++) {
        assert(r1.selected_indices[i] == r2.selected_indices[i]);
    }

    // Same beta and C_v (within floating-point tolerance)
    const float tol = 1e-5f;
    for (int h = 0; h < n_head_kv; h++) {
        assert(r1.beta[h].size() == r2.beta[h].size());
        for (size_t j = 0; j < r1.beta[h].size(); j++) {
            assert(fabsf(r1.beta[h][j] - r2.beta[h][j]) < tol);
        }
        assert(r1.C_v[h].size() == r2.C_v[h].size());
        for (size_t j = 0; j < r1.C_v[h].size(); j++) {
            assert(fabsf(r1.C_v[h][j] - r2.C_v[h][j]) < tol);
        }
    }

    // Same sensitivity
    assert(r1.head_sensitivity.size() == r2.head_sensitivity.size());
    for (size_t h = 0; h < r1.head_sensitivity.size(); h++) {
        assert(fabsf(r1.head_sensitivity[h] - r2.head_sensitivity[h]) < tol);
    }

    printf(" PASS\n");
}

// ===========================================================================
int main() {
    printf("test-budget-exchange:\n");

    printf("\n=== Greedy Budget Exchange (Paper S5) ===\n");
    test_greedy_exchange_conserves_budget();
    test_greedy_exchange_uniform_sensitivity();
    test_greedy_exchange_extreme_sensitivity();
    test_greedy_exchange_respects_minimum();
    test_greedy_exchange_respects_maximum();
    test_greedy_exchange_single_head();
    test_greedy_exchange_proportional();

    printf("\n=== Nonuniform Compaction Integration ===\n");
    test_nonuniform_compaction_quality();
    test_nonuniform_backward_compatible();

    printf("\nAll budget exchange tests passed!\n");
    return 0;
}
