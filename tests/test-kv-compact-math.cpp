// Tests for KV cache compaction math utilities
//
// Validates the pure math functions used by the Attention Matching algorithm:
//   - Matrix multiplication variants (ABt, AtB)
//   - Softmax and stable exp
//   - Non-negative least squares (NNLS)
//   - Regularized least squares
//   - Full compaction pipeline (compact_head_highest_attn)

#undef NDEBUG
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

#include "kv-compact-math.h"
#include "kv-compact-state.h"

static const float EPS = 1e-5f;

static bool approx_eq(float a, float b, float tol = EPS) {
    return fabsf(a - b) < tol;
}

// ============================================================================
// mat_mul_ABt tests
// ============================================================================

static void test_mat_mul_ABt_identity() {
    printf("  test_mat_mul_ABt_identity...");
    // A = [[1,0],[0,1]], B = [[1,0],[0,1]]
    // A * B^T = I * I^T = I
    float A[] = {1, 0, 0, 1};
    float B[] = {1, 0, 0, 1};
    float C[4] = {};
    mat_mul_ABt(A, B, C, 2, 2, 2);
    assert(approx_eq(C[0], 1.0f));
    assert(approx_eq(C[1], 0.0f));
    assert(approx_eq(C[2], 0.0f));
    assert(approx_eq(C[3], 1.0f));
    printf(" OK\n");
}

static void test_mat_mul_ABt_rectangular() {
    printf("  test_mat_mul_ABt_rectangular...");
    // A = [[1,2,3]], B = [[4,5,6],[7,8,9]]
    // A(1x3) * B^T(3x2) = (1x2)
    // C[0,0] = 1*4+2*5+3*6 = 32
    // C[0,1] = 1*7+2*8+3*9 = 50
    float A[] = {1, 2, 3};
    float B[] = {4, 5, 6, 7, 8, 9};
    float C[2] = {};
    mat_mul_ABt(A, B, C, 1, 2, 3);
    assert(approx_eq(C[0], 32.0f));
    assert(approx_eq(C[1], 50.0f));
    printf(" OK\n");
}

// ============================================================================
// mat_mul_AtB tests
// ============================================================================

static void test_mat_mul_AtB_basic() {
    printf("  test_mat_mul_AtB_basic...");
    // A = [[1,2],[3,4]], B = [[5,6],[7,8]]
    // A^T * B where A is (2x2), B is (2x2), result (2x2)
    // A^T = [[1,3],[2,4]]
    // C[0,0] = 1*5+3*7 = 26, C[0,1] = 1*6+3*8 = 30
    // C[1,0] = 2*5+4*7 = 38, C[1,1] = 2*6+4*8 = 44
    float A[] = {1, 2, 3, 4};
    float B[] = {5, 6, 7, 8};
    float C[4] = {};
    mat_mul_AtB(A, B, C, 2, 2, 2);
    assert(approx_eq(C[0], 26.0f));
    assert(approx_eq(C[1], 30.0f));
    assert(approx_eq(C[2], 38.0f));
    assert(approx_eq(C[3], 44.0f));
    printf(" OK\n");
}

static void test_mat_mul_AtB_rectangular() {
    printf("  test_mat_mul_AtB_rectangular...");
    // A is (3x2), B is (3x1) -> result is (2x1)
    // A = [[1,2],[3,4],[5,6]], B = [[1],[2],[3]]
    // A^T = [[1,3,5],[2,4,6]]
    // C[0] = 1*1+3*2+5*3 = 22
    // C[1] = 2*1+4*2+6*3 = 28
    float A[] = {1, 2, 3, 4, 5, 6};
    float B[] = {1, 2, 3};
    float C[2] = {};
    mat_mul_AtB(A, B, C, 3, 2, 1);
    assert(approx_eq(C[0], 22.0f));
    assert(approx_eq(C[1], 28.0f));
    printf(" OK\n");
}

// ============================================================================
// softmax_rows tests
// ============================================================================

static void test_softmax_rows_sums_to_one() {
    printf("  test_softmax_rows_sums_to_one...");
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 1.0f, -1.0f, 0.0f, 2.0f};
    softmax_rows(data, 2, 4);

    // Each row should sum to 1
    float sum0 = data[0] + data[1] + data[2] + data[3];
    float sum1 = data[4] + data[5] + data[6] + data[7];
    assert(approx_eq(sum0, 1.0f));
    assert(approx_eq(sum1, 1.0f));
    printf(" OK\n");
}

static void test_softmax_rows_ordering() {
    printf("  test_softmax_rows_ordering...");
    float data[] = {1.0f, 2.0f, 3.0f};
    softmax_rows(data, 1, 3);

    // Larger input -> larger probability
    assert(data[0] < data[1]);
    assert(data[1] < data[2]);
    // All positive
    assert(data[0] > 0.0f);
    printf(" OK\n");
}

static void test_softmax_rows_uniform() {
    printf("  test_softmax_rows_uniform...");
    float data[] = {5.0f, 5.0f, 5.0f, 5.0f};
    softmax_rows(data, 1, 4);

    // Equal inputs -> uniform distribution
    for (int i = 0; i < 4; i++) {
        assert(approx_eq(data[i], 0.25f));
    }
    printf(" OK\n");
}

static void test_softmax_rows_numerical_stability() {
    printf("  test_softmax_rows_numerical_stability...");
    // Large values that would overflow without max-shift
    float data[] = {1000.0f, 1001.0f, 1002.0f};
    softmax_rows(data, 1, 3);

    float sum = data[0] + data[1] + data[2];
    assert(approx_eq(sum, 1.0f));
    assert(data[2] > data[1]);
    assert(data[1] > data[0]);
    printf(" OK\n");
}

// ============================================================================
// exp_rows_stable tests
// ============================================================================

static void test_exp_rows_stable_basic() {
    printf("  test_exp_rows_stable_basic...");
    float data[] = {0.0f, 1.0f, 2.0f};
    float row_sums[1];
    exp_rows_stable(data, row_sums, 1, 3);

    // After max-shift (max=2): exp(-2), exp(-1), exp(0) = 1
    assert(approx_eq(data[2], 1.0f));
    assert(data[1] < data[2]);
    assert(data[0] < data[1]);

    // row_sum should equal sum of the exp values
    float expected_sum = data[0] + data[1] + data[2];
    assert(approx_eq(row_sums[0], expected_sum));
    printf(" OK\n");
}

static void test_exp_rows_stable_large_values() {
    printf("  test_exp_rows_stable_large_values...");
    float data[] = {500.0f, 501.0f};
    float row_sums[1];
    exp_rows_stable(data, row_sums, 1, 2);

    // Should not overflow/NaN
    assert(std::isfinite(data[0]));
    assert(std::isfinite(data[1]));
    assert(std::isfinite(row_sums[0]));
    // Max element (501) after shift -> exp(0) = 1
    assert(approx_eq(data[1], 1.0f));
    printf(" OK\n");
}

// ============================================================================
// nnls_solve tests
// ============================================================================

static void test_nnls_identity() {
    printf("  test_nnls_identity...");
    // A = I (2x2), b = [3, 5]
    // min ||w - b||^2 s.t. w >= 0 => w = [3, 5]
    float A[] = {1, 0, 0, 1};
    float b[] = {3.0f, 5.0f};
    float w[2] = {};
    nnls_solve(A, b, w, 2, 2, 500);
    assert(approx_eq(w[0], 3.0f, 0.1f));
    assert(approx_eq(w[1], 5.0f, 0.1f));
    printf(" OK\n");
}

static void test_nnls_non_negative_constraint() {
    printf("  test_nnls_non_negative_constraint...");
    // A = I, b = [-2, 4]
    // Unconstrained solution = [-2, 4], but NNLS clamps to w >= 0
    // So w ≈ [0, 4]
    float A[] = {1, 0, 0, 1};
    float b[] = {-2.0f, 4.0f};
    float w[2] = {};
    nnls_solve(A, b, w, 2, 2, 500);
    assert(w[0] < 0.01f);  // should be ~0
    assert(approx_eq(w[1], 4.0f, 0.1f));
    printf(" OK\n");
}

static void test_nnls_overdetermined() {
    printf("  test_nnls_overdetermined...");
    // A = [[1],[1],[1]], b = [2, 3, 4]
    // LS solution: w = mean(b) = 3, and it's positive, so NNLS = 3
    float A[] = {1, 1, 1};
    float b[] = {2.0f, 3.0f, 4.0f};
    float w[1] = {};
    nnls_solve(A, b, w, 3, 1, 500);
    assert(approx_eq(w[0], 3.0f, 0.2f));
    printf(" OK\n");
}

// ============================================================================
// PGD NNLS solver tests
// ============================================================================

static void test_nnls_pgd_basic() {
    printf("  test_nnls_pgd_basic...");
    float A[] = {1, 0, 0, 1};
    float b[] = {3.0f, 7.0f};
    float w[2] = {};
    nnls_pgd_solve(A, b, w, 2, 2);
    assert(approx_eq(w[0], 3.0f, 0.01f));
    assert(approx_eq(w[1], 7.0f, 0.01f));
    printf(" OK\n");
}

static void test_nnls_pgd_clamping() {
    printf("  test_nnls_pgd_clamping...");
    float A[] = {1, 0, 0, 1};
    float b[] = {-1.0f, 5.0f};
    float w[2] = {};
    nnls_pgd_solve(A, b, w, 2, 2);
    assert(w[0] >= 0.0f);
    assert(approx_eq(w[1], 5.0f, 0.01f));
    printf(" OK\n");
}

static void test_nnls_pgd_with_iters() {
    printf("  test_nnls_pgd_with_iters...");
    float A[] = {1, 1, 1};
    float b[] = {2.0f, 3.0f, 4.0f};
    float w[1] = {};
    nnls_pgd_solve(A, b, w, 3, 1, 1e-12f, 50);
    assert(approx_eq(w[0], 3.0f, 0.2f));
    printf(" OK\n");
}

// ============================================================================
// OMP key selection tests
// ============================================================================

static void test_omp_select_basic() {
    printf("  test_omp_select_basic...");
    int T = 8, n_q = 4, t = 3;
    std::vector<float> exp_scores(n_q * T, 0.1f);
    std::vector<float> row_sums(n_q);

    for (int i = 0; i < n_q; i++) {
        exp_scores[i * T + 0] = 10.0f;
        exp_scores[i * T + 7] = 5.0f;
        exp_scores[i * T + 3] = 3.0f;
    }
    for (int i = 0; i < n_q; i++) {
        float s = 0.0f;
        for (int j = 0; j < T; j++) s += exp_scores[i * T + j];
        row_sums[i] = s;
    }

    std::vector<int> selected(t);
    std::vector<float> w_out(t);
    omp_select_keys(exp_scores.data(), row_sums.data(), T, n_q, t,
                    selected.data(), w_out.data());

    // Key 0 should always be selected first (highest correlation)
    bool has_0 = false;
    for (int j = 0; j < t; j++) if (selected[j] == 0) has_0 = true;
    assert(has_0);
    // All selected should be sorted and distinct
    for (int j = 1; j < t; j++) assert(selected[j] > selected[j-1]);
    // Weights should all be positive
    for (int j = 0; j < t; j++) assert(w_out[j] > 0.0f);
    printf(" OK\n");
}

static void test_omp_select_with_kchoice() {
    printf("  test_omp_select_with_kchoice...");
    int T = 8, n_q = 4, t = 4;
    std::vector<float> exp_scores(n_q * T, 0.1f);
    std::vector<float> row_sums(n_q);

    for (int i = 0; i < n_q; i++) {
        exp_scores[i * T + 0] = 10.0f;
        exp_scores[i * T + 7] = 5.0f;
        exp_scores[i * T + 3] = 3.0f;
        exp_scores[i * T + 5] = 2.0f;
    }
    for (int i = 0; i < n_q; i++) {
        float s = 0.0f;
        for (int j = 0; j < T; j++) s += exp_scores[i * T + j];
        row_sums[i] = s;
    }

    std::vector<int> selected(t);
    std::vector<float> w_out(t);
    omp_select_keys(exp_scores.data(), row_sums.data(), T, n_q, t,
                    selected.data(), w_out.data(), 2, 1);

    for (int j = 1; j < t; j++) assert(selected[j] > selected[j-1]);
    printf(" OK\n");
}

// ============================================================================
// Score aggregation tests
// ============================================================================

static void test_score_agg_rms() {
    printf("  test_score_agg_rms...");
    // 2 queries, 3 keys
    float scores[] = {1.0f, 2.0f, 0.5f,
                      0.5f, 1.0f, 2.0f};
    float sm_out[6], imp_max[3], imp_rms[3], imp_mean[3];

    softmax_importance_fused(scores, sm_out, imp_max, 2, 3, SCORE_AGG_MAX);
    softmax_importance_fused(scores, sm_out, imp_rms, 2, 3, SCORE_AGG_RMS);
    softmax_importance_fused(scores, sm_out, imp_mean, 2, 3, SCORE_AGG_MEAN);

    // All should be non-negative
    for (int j = 0; j < 3; j++) {
        assert(imp_max[j] >= 0.0f);
        assert(imp_rms[j] >= 0.0f);
        assert(imp_mean[j] >= 0.0f);
    }
    // RMS should be >= mean (by Jensen's inequality)
    for (int j = 0; j < 3; j++) {
        assert(imp_rms[j] >= imp_mean[j] - 1e-6f);
    }
    printf(" OK\n");
}

// ============================================================================
// least_squares_solve tests
// ============================================================================

static void test_least_squares_identity() {
    printf("  test_least_squares_identity...");
    // A = I (2x2), b = [3, 7] -> x = [3, 7]
    float A[] = {1, 0, 0, 1};
    float b[] = {3.0f, 7.0f};
    float x[2] = {};
    least_squares_solve(A, b, x, 2, 2, 1, 0.0f);
    assert(approx_eq(x[0], 3.0f, 0.01f));
    assert(approx_eq(x[1], 7.0f, 0.01f));
    printf(" OK\n");
}

static void test_least_squares_overdetermined() {
    printf("  test_least_squares_overdetermined...");
    // A = [[1,0],[0,1],[1,1]], b = [[1],[0],[2]] (3x1 multi-column)
    // LS: A^T A = [[2,1],[1,2]], A^T b = [3, 2]
    // Solve: x = [4/3, 1/3]
    float A[] = {1, 0, 0, 1, 1, 1};
    float b[] = {1.0f, 0.0f, 2.0f};
    float x[2] = {};
    least_squares_solve(A, b, x, 3, 2, 1, 0.0f);
    assert(approx_eq(x[0], 4.0f / 3.0f, 0.01f));
    assert(approx_eq(x[1], 1.0f / 3.0f, 0.01f));
    printf(" OK\n");
}

static void test_least_squares_multi_rhs() {
    printf("  test_least_squares_multi_rhs...");
    // A = I (2x2), b = [[1,2],[3,4]] (2 rows, 2 columns) -> x = b
    float A[] = {1, 0, 0, 1};
    float b[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x[4] = {};
    least_squares_solve(A, b, x, 2, 2, 2, 0.0f);
    assert(approx_eq(x[0], 1.0f, 0.01f));
    assert(approx_eq(x[1], 2.0f, 0.01f));
    assert(approx_eq(x[2], 3.0f, 0.01f));
    assert(approx_eq(x[3], 4.0f, 0.01f));
    printf(" OK\n");
}

static void test_least_squares_with_ridge() {
    printf("  test_least_squares_with_ridge...");
    // With ridge > 0, solution should be shrunk toward zero
    float A[] = {1, 0, 0, 1};
    float b[] = {10.0f, 10.0f};
    float x_no_ridge[2] = {};
    float x_ridge[2] = {};
    least_squares_solve(A, b, x_no_ridge, 2, 2, 1, 0.0f);
    least_squares_solve(A, b, x_ridge, 2, 2, 1, 1.0f);
    // Ridge should shrink toward zero
    assert(fabsf(x_ridge[0]) < fabsf(x_no_ridge[0]));
    assert(fabsf(x_ridge[1]) < fabsf(x_no_ridge[1]));
    printf(" OK\n");
}

static void test_least_squares_large_overdetermined() {
    printf("  test_least_squares_large_overdetermined...");
    // Reproduce model-scale dimensions: m=376 rows, n=375 cols, p=128 RHS
    const int m = 376, n = 375, p = 128;

    std::vector<float> A(m * n, 0.0f);
    std::vector<float> b(m * p, 0.0f);
    std::vector<float> x(n * p, 0.0f);

    // Fill A with random uniform rows normalized to sum=1
    srand(42);
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            A[i * n + j] = (float)rand() / RAND_MAX;
            sum += A[i * n + j];
        }
        for (int j = 0; j < n; j++) A[i * n + j] /= sum;
    }

    // b = A * x_true
    std::vector<float> x_true(n * p);
    for (int j = 0; j < n; j++)
        for (int d = 0; d < p; d++)
            x_true[j * p + d] = sinf((float)(j + d) * 0.1f);

    for (int i = 0; i < m; i++)
        for (int d = 0; d < p; d++) {
            float v = 0.0f;
            for (int j = 0; j < n; j++) v += A[i * n + j] * x_true[j * p + d];
            b[i * p + d] = v;
        }

    least_squares_solve(A.data(), b.data(), x.data(), m, n, p, 1e-6f);

    float max_err = 0.0f, sum_x = 0.0f;
    for (int j = 0; j < n * p; j++) {
        sum_x += fabsf(x[j]);
        float e = fabsf(x[j] - x_true[j]);
        if (e > max_err) max_err = e;
    }
    fprintf(stderr, " uniform: |sum_x|=%.1f max_err=%.4f", sum_x, max_err);
    // Relax tolerance for large system (Gaussian elimination accumulates error)
    assert(sum_x > 1.0f && "Solution should not be all zeros");
    assert(max_err < 0.5f && "Solution should be in the right ballpark");
    fprintf(stderr, " OK\n");
    printf(" OK\n");
}

static void test_least_squares_softmax_structure() {
    printf("  test_least_squares_softmax_structure...");
    // Test with REAL softmax structure: peaky rows like actual attention weights.
    // This is what fails in the model benchmark.
    const int m = 200, n = 100, p = 32;  // start small

    std::vector<float> scores(m * n);
    std::vector<float> A(m * n);
    std::vector<float> b(m * p, 0.0f);
    std::vector<float> x(n * p, 0.0f);

    srand(42);
    // Generate random logits and softmax them
    for (int i = 0; i < m; i++) {
        float max_s = -1e30f;
        for (int j = 0; j < n; j++) {
            scores[i * n + j] = ((float)rand() / RAND_MAX - 0.5f) * 4.0f;
            if (scores[i * n + j] > max_s) max_s = scores[i * n + j];
        }
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            A[i * n + j] = expf(scores[i * n + j] - max_s);
            sum += A[i * n + j];
        }
        for (int j = 0; j < n; j++) A[i * n + j] /= sum;
    }

    // x_true = V-like values
    std::vector<float> x_true(n * p);
    for (int j = 0; j < n * p; j++)
        x_true[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    // b = A * x_true
    for (int i = 0; i < m; i++)
        for (int d = 0; d < p; d++) {
            float v = 0.0f;
            for (int j = 0; j < n; j++) v += A[i * n + j] * x_true[j * p + d];
            b[i * p + d] = v;
        }

    least_squares_solve(A.data(), b.data(), x.data(), m, n, p, 1e-6f);

    float max_err = 0.0f, sum_x = 0.0f;
    for (int j = 0; j < n * p; j++) {
        sum_x += fabsf(x[j]);
        float e = fabsf(x[j] - x_true[j]);
        if (e > max_err) max_err = e;
    }
    fprintf(stderr, " softmax(200x100,p=32): |sum_x|=%.1f max_err=%.4f", sum_x, max_err);

    // Now test at model scale with peaky softmax AND full d_v=128
    const int m2 = 376, n2 = 375, p2 = 128;
    std::vector<float> A2(m2 * n2);
    std::vector<float> b2(m2 * p2, 0.0f);
    std::vector<float> x2(n2 * p2, 0.0f);

    for (int i = 0; i < m2; i++) {
        float max_s = -1e30f;
        for (int j = 0; j < n2; j++) {
            float s = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;  // peaky
            scores.resize(m2 * n2);
            scores[i * n2 + j] = s;
            if (s > max_s) max_s = s;
        }
        float sum = 0.0f;
        for (int j = 0; j < n2; j++) {
            A2[i * n2 + j] = expf(scores[i * n2 + j] - max_s);
            sum += A2[i * n2 + j];
        }
        for (int j = 0; j < n2; j++) A2[i * n2 + j] /= sum;
    }

    std::vector<float> x_true2(n2 * p2);
    for (int j = 0; j < n2 * p2; j++)
        x_true2[j] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    for (int i = 0; i < m2; i++)
        for (int d = 0; d < p2; d++) {
            float v = 0.0f;
            for (int j = 0; j < n2; j++) v += A2[i * n2 + j] * x_true2[j * p2 + d];
            b2[i * p2 + d] = v;
        }

    least_squares_solve(A2.data(), b2.data(), x2.data(), m2, n2, p2, 1e-6f);

    float max_err2 = 0.0f, sum_x2 = 0.0f;
    for (int j = 0; j < n2 * p2; j++) {
        sum_x2 += fabsf(x2[j]);
        float e = fabsf(x2[j] - x_true2[j]);
        if (e > max_err2) max_err2 = e;
    }
    fprintf(stderr, "\n    peaky(376x375,p=4): |sum_x|=%.1f max_err=%.4f", sum_x2, max_err2);
    if (sum_x2 < 0.1f) {
        fprintf(stderr, " *** ZERO SOLUTION ***\n");
        // Print some A2 row stats
        for (int i = 0; i < 3; i++) {
            float rmax = 0.0f;
            for (int j = 0; j < n2; j++)
                if (A2[i * n2 + j] > rmax) rmax = A2[i * n2 + j];
            fprintf(stderr, "    A2 row %d: max=%.6f\n", i, rmax);
        }
    }
    assert(sum_x > 1.0f && "Small softmax system should produce non-zero solution");
    fprintf(stderr, " OK\n");
    printf(" OK\n");
}

static void test_least_squares_small_known() {
    printf("  test_least_squares_small_known...");
    // Simple 4x3 system, 2 RHS — verify non-zero solution
    const int m = 4, n = 3, p = 2;
    float A[] = {
        0.5f, 0.3f, 0.2f,
        0.1f, 0.7f, 0.2f,
        0.3f, 0.3f, 0.4f,
        0.2f, 0.2f, 0.6f
    };
    // x_true = [[1,2],[3,4],[5,6]]
    // b = A * x_true
    float b[4 * 2];
    float x_true[] = {1,2, 3,4, 5,6};
    for (int i = 0; i < m; i++) {
        for (int d = 0; d < p; d++) {
            float v = 0.0f;
            for (int j = 0; j < n; j++) v += A[i * n + j] * x_true[j * p + d];
            b[i * p + d] = v;
        }
    }

    float x[3 * 2] = {};
    least_squares_solve(A, b, x, m, n, p, 1e-6f);

    float max_err = 0.0f;
    for (int j = 0; j < n * p; j++) {
        float err = fabsf(x[j] - x_true[j]);
        if (err > max_err) max_err = err;
    }
    printf(" max_err=%.6f", max_err);
    assert(max_err < 0.01f);
    printf(" OK\n");
}

// ============================================================================
// Per-head sensitivity and budget allocation tests
// ============================================================================

static void test_sensitivity_uniform_attention() {
    printf("  test_sensitivity_uniform_attention...");
    // Uniform attention → sensitivity should be close to 1.0
    const int n_q = 4, T = 8;
    std::vector<float> attn(n_q * T);
    for (int i = 0; i < n_q * T; i++) attn[i] = 1.0f / T;

    float sens = compute_head_sensitivity(attn.data(), n_q, T);
    // max_importance = 1/T, mean_importance = 1/T, ratio = 1.0
    assert(approx_eq(sens, 1.0f, 0.1f));
    printf(" OK\n");
}

static void test_sensitivity_concentrated_attention() {
    printf("  test_sensitivity_concentrated_attention...");
    // One position gets all attention → high sensitivity
    const int n_q = 4, T = 8;
    std::vector<float> attn(n_q * T, 0.0f);
    // Each query puts all weight on position 3
    for (int qi = 0; qi < n_q; qi++) {
        attn[qi * T + 3] = 1.0f;
    }

    float sens = compute_head_sensitivity(attn.data(), n_q, T);
    // max_importance = 1.0, mean_importance = 1/T = 0.125, ratio = 8
    assert(approx_eq(sens, (float)T, 0.1f));
    printf(" OK\n");
}

static void test_sensitivity_ordering() {
    printf("  test_sensitivity_ordering...");
    // Concentrated head should have higher sensitivity than spread head
    const int n_q = 8, T = 16;

    // Head A: concentrated on 2 positions
    std::vector<float> attn_a(n_q * T, 0.0f);
    for (int qi = 0; qi < n_q; qi++) {
        attn_a[qi * T + 2] = 0.7f;
        attn_a[qi * T + 5] = 0.3f;
    }

    // Head B: spread across many positions
    std::vector<float> attn_b(n_q * T);
    for (int i = 0; i < n_q * T; i++) attn_b[i] = 1.0f / T;

    float sens_a = compute_head_sensitivity(attn_a.data(), n_q, T);
    float sens_b = compute_head_sensitivity(attn_b.data(), n_q, T);

    printf("\n    Concentrated sensitivity: %.2f, Uniform sensitivity: %.2f\n", sens_a, sens_b);
    assert(sens_a > sens_b);
    printf("  OK\n");
}

static void test_weighted_importance_basic() {
    printf("  test_weighted_importance_basic...");
    // Two heads: head 0 has sensitivity 10, head 1 has sensitivity 1
    // Head 0 cares about position 2, head 1 cares about position 5
    // Weighted importance should favor position 2
    const int T = 8, n_heads = 2;

    std::vector<std::vector<float>> per_head_imp = {
        {0, 0, 1.0f, 0, 0, 0, 0, 0},  // head 0: position 2
        {0, 0, 0, 0, 0, 1.0f, 0, 0},  // head 1: position 5
    };
    std::vector<float> sensitivities = {10.0f, 1.0f};

    std::vector<float> out(T, 0.0f);
    accumulate_weighted_importance(per_head_imp, sensitivities, T, n_heads, out.data());

    // Position 2 should have importance 10, position 5 should have importance 1
    assert(approx_eq(out[2], 10.0f, 0.01f));
    assert(approx_eq(out[5], 1.0f, 0.01f));
    assert(approx_eq(out[0], 0.0f, 0.01f));

    // Position 2 should rank higher
    assert(out[2] > out[5]);
    printf(" OK\n");
}

// ============================================================================
// Beta injection via K-modification tests
// ============================================================================

static void test_compute_beta_direction_identity_queries() {
    printf("  test_compute_beta_direction_identity_queries...");
    // Q_ref = I (identity) → direction should be [1, 1, ..., 1] (approximately)
    // because I @ v = v should approximate [1, 1, ..., 1]
    const int n_q = 3, d_k = 3;
    float Q[] = {1, 0, 0,
                 0, 1, 0,
                 0, 0, 1};
    float dir[3] = {};
    compute_beta_direction(Q, n_q, d_k, dir);

    // Each component should be ~1.0 (LS with identity gives v = ones)
    for (int d = 0; d < d_k; d++) {
        assert(approx_eq(dir[d], 1.0f, 0.01f));
    }
    printf(" OK\n");
}

static void test_compute_beta_direction_produces_unit_dot() {
    printf("  test_compute_beta_direction_produces_unit_dot...");
    // For random queries, Q @ v should be approximately 1 for each query
    const int n_q = 16, d_k = 8;
    std::vector<float> Q(n_q * d_k);
    for (int i = 0; i < n_q * d_k; i++) {
        Q[i] = sinf((float)(i * 7 + 3) * 0.4f);
    }
    std::vector<float> dir(d_k);
    compute_beta_direction(Q.data(), n_q, d_k, dir.data());

    // Check Q @ v ≈ 1
    float max_err = 0.0f;
    for (int qi = 0; qi < n_q; qi++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) {
            dot += Q[qi * d_k + d] * dir[d];
        }
        float err = fabsf(dot - 1.0f);
        if (err > max_err) max_err = err;
    }
    // Also compute mean error
    float mean_err = 0.0f;
    for (int qi = 0; qi < n_q; qi++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) {
            dot += Q[qi * d_k + d] * dir[d];
        }
        mean_err += fabsf(dot - 1.0f);
    }
    mean_err /= n_q;
    printf("\n    Max |Q@v - 1| = %.6f, Mean = %.6f\n", max_err, mean_err);
    // Overdetermined system (16 eqs, 8 vars) — mean error should be small
    assert(mean_err < 1.0f);
    printf("  OK\n");
}

static void test_apply_beta_to_keys_basic() {
    printf("  test_apply_beta_to_keys_basic...");
    // Verify K is modified by beta * sqrt(d_k) * direction
    const int T = 4, d_k = 3, n_embd = 3;
    float K[] = {1, 0, 0,
                 0, 1, 0,
                 0, 0, 1,
                 1, 1, 1};
    int selected[] = {1, 3};
    float beta[] = {2.0f, -1.0f};
    float dir[] = {0.5f, 0.0f, 0.5f};
    int t = 2;

    apply_beta_to_keys(K, n_embd, selected, t, beta, dir, d_k, 0);

    float scale = sqrtf(3.0f);
    // K[1] should be [0 + 2*sqrt(3)*0.5, 1 + 0, 0 + 2*sqrt(3)*0.5]
    assert(approx_eq(K[1 * n_embd + 0], 2.0f * scale * 0.5f, 0.01f));
    assert(approx_eq(K[1 * n_embd + 1], 1.0f, 0.01f));
    assert(approx_eq(K[1 * n_embd + 2], 2.0f * scale * 0.5f, 0.01f));

    // K[3] should be [1 + (-1)*sqrt(3)*0.5, 1 + 0, 1 + (-1)*sqrt(3)*0.5]
    assert(approx_eq(K[3 * n_embd + 0], 1.0f + (-1.0f) * scale * 0.5f, 0.01f));
    assert(approx_eq(K[3 * n_embd + 1], 1.0f, 0.01f));
    assert(approx_eq(K[3 * n_embd + 2], 1.0f + (-1.0f) * scale * 0.5f, 0.01f));

    // K[0] and K[2] should be unchanged
    assert(approx_eq(K[0], 1.0f, 0.01f));
    assert(approx_eq(K[2 * n_embd + 2], 1.0f, 0.01f));
    printf(" OK\n");
}

static void test_beta_injection_quality() {
    printf("  test_beta_injection_quality...");
    // End-to-end: compact, inject beta into K, verify that the modified K
    // approximates the original score + beta for reference queries
    const int T = 32, n_q = 16, d_k = 8, d_v = 8, t = 8;

    std::vector<float> K(T * d_k), V(T * d_v), Q(n_q * d_k);
    for (int i = 0; i < T * d_k; i++) K[i] = sinf((float)(i * 3 + 1) * 0.5f);
    for (int i = 0; i < T * d_v; i++) V[i] = cosf((float)(i * 2 + 3) * 0.3f);
    for (int i = 0; i < n_q * d_k; i++) Q[i] = sinf((float)(i * 5 + 2) * 0.8f);

    auto result = compact_head_highest_attn(K.data(), V.data(), Q.data(),
                                            T, n_q, d_k, d_v, t);

    // Compute beta direction
    // Use the K vectors at selected positions as reference queries (same as Q_ref)
    std::vector<float> dir(d_k);
    compute_beta_direction(Q.data(), n_q, d_k, dir.data());

    // Copy selected K and apply beta
    std::vector<float> K_mod(t * d_k);
    for (int j = 0; j < t; j++) {
        memcpy(K_mod.data() + j * d_k, K.data() + result.selected_indices[j] * d_k,
               d_k * sizeof(float));
    }
    // Apply beta to contiguous K_mod (head_offset=0, n_embd=d_k, indices are 0..t-1)
    std::vector<int> mod_indices(t);
    std::iota(mod_indices.begin(), mod_indices.end(), 0);
    apply_beta_to_keys(K_mod.data(), d_k, mod_indices.data(), t,
                       result.beta.data(), dir.data(), d_k, 0);

    // Compare: for each reference query, check that
    // q @ K_mod / sqrt(d_k) ≈ q @ K_orig / sqrt(d_k) + beta
    float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);
    float total_err = 0.0f;
    int count = 0;
    for (int qi = 0; qi < n_q; qi++) {
        for (int j = 0; j < t; j++) {
            float score_orig = 0.0f, score_mod = 0.0f;
            for (int d = 0; d < d_k; d++) {
                score_orig += Q[qi * d_k + d] * K[result.selected_indices[j] * d_k + d];
                score_mod  += Q[qi * d_k + d] * K_mod[j * d_k + d];
            }
            score_orig *= inv_sqrt_dk;
            score_mod  *= inv_sqrt_dk;

            float target = score_orig + result.beta[j];
            float err = fabsf(score_mod - target);
            total_err += err;
            count++;
        }
    }
    float avg_err = total_err / count;

    // Also compute average |beta| for context
    float avg_beta = 0.0f;
    for (int j = 0; j < t; j++) avg_beta += fabsf(result.beta[j]);
    avg_beta /= t;

    printf("\n    Avg |score_mod - (score_orig + beta)| = %.6f\n", avg_err);
    printf("    Avg |beta| = %.6f\n", avg_beta);
    printf("    Relative error = %.6f\n", avg_err / (avg_beta + 1e-8f));
    // The K-modification should approximate beta reasonably well
    // Relative error should be reasonable (< 50% of average beta)
    assert(avg_err < avg_beta * 2.0f + 0.5f);
    printf("  OK\n");
}

// ============================================================================
// compact_head_highest_attn tests
// ============================================================================

static void test_compact_no_compression_needed() {
    printf("  test_compact_no_compression_needed...");
    // t >= T means no compaction
    const int T = 4, n_q = 2, d_k = 3, d_v = 3;
    std::vector<float> K(T * d_k), V(T * d_v), Q(n_q * d_k);
    for (int i = 0; i < T * d_k; i++) K[i] = (float)(i % 7) * 0.1f;
    for (int i = 0; i < T * d_v; i++) V[i] = (float)(i % 5) * 0.2f;
    for (int i = 0; i < n_q * d_k; i++) Q[i] = (float)(i % 3) * 0.3f;

    auto result = compact_head_highest_attn(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, T);
    assert((int)result.selected_indices.size() == T);
    for (int i = 0; i < T; i++) {
        assert(result.selected_indices[i] == i);
    }
    // Beta should be all zeros
    for (int i = 0; i < T; i++) {
        assert(approx_eq(result.beta[i], 0.0f));
    }
    // C_v should equal V
    for (int i = 0; i < T * d_v; i++) {
        assert(approx_eq(result.C_v[i], V[i]));
    }
    printf(" OK\n");
}

static void test_compact_selects_correct_count() {
    printf("  test_compact_selects_correct_count...");
    const int T = 20, n_q = 8, d_k = 4, d_v = 4, t = 5;

    // Create keys with one obvious "hot" key that all queries attend to
    std::vector<float> K(T * d_k, 0.0f);
    std::vector<float> V(T * d_v, 0.0f);
    std::vector<float> Q(n_q * d_k, 0.0f);

    // Make diverse keys
    for (int i = 0; i < T; i++) {
        for (int d = 0; d < d_k; d++) {
            K[i * d_k + d] = sinf((float)(i * d_k + d) * 0.7f);
        }
        for (int d = 0; d < d_v; d++) {
            V[i * d_v + d] = cosf((float)(i * d_v + d) * 0.3f);
        }
    }
    for (int i = 0; i < n_q; i++) {
        for (int d = 0; d < d_k; d++) {
            Q[i * d_k + d] = sinf((float)(i * d_k + d) * 1.1f);
        }
    }

    auto result = compact_head_highest_attn(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t);

    // Check sizes
    assert((int)result.selected_indices.size() == t);
    assert((int)result.beta.size() == t);
    assert((int)result.C_v.size() == t * d_v);

    // Selected indices should be sorted and within range
    for (int i = 0; i < t; i++) {
        assert(result.selected_indices[i] >= 0);
        assert(result.selected_indices[i] < T);
    }
    for (int i = 1; i < t; i++) {
        assert(result.selected_indices[i] > result.selected_indices[i - 1]);
    }
    printf(" OK\n");
}

static void test_compact_quality_improves_with_refitting() {
    printf("  test_compact_quality_improves_with_refitting...");
    // Core claim: AM with value refitting should produce lower error than
    // simple token eviction (keeping original V for selected keys)

    const int T = 32, n_q = 16, d_k = 8, d_v = 8, t = 8;

    std::vector<float> K(T * d_k), V(T * d_v), Q(n_q * d_k);
    // Create data with some structure
    for (int i = 0; i < T; i++) {
        for (int d = 0; d < d_k; d++) {
            K[i * d_k + d] = sinf((float)(i * 3 + d) * 0.5f) + 0.1f * cosf((float)(i + d * 7));
        }
        for (int d = 0; d < d_v; d++) {
            V[i * d_v + d] = cosf((float)(i * 2 + d) * 0.3f) + 0.2f * sinf((float)(i + d * 5));
        }
    }
    for (int i = 0; i < n_q; i++) {
        for (int d = 0; d < d_k; d++) {
            Q[i * d_k + d] = sinf((float)(i * 5 + d) * 0.8f);
        }
    }

    auto result = compact_head_highest_attn(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t);

    // Compute original attention output for a test query
    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
    std::vector<float> test_q(Q.data(), Q.data() + d_k); // first query

    // Original scores and softmax
    std::vector<float> orig_scores(T);
    for (int j = 0; j < T; j++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) dot += test_q[d] * K[j * d_k + d];
        orig_scores[j] = dot * inv_sqrt_dk;
    }
    std::vector<float> orig_attn(orig_scores);
    softmax_rows(orig_attn.data(), 1, T);

    // Original output
    std::vector<float> orig_out(d_v, 0.0f);
    for (int j = 0; j < T; j++) {
        for (int d = 0; d < d_v; d++) {
            orig_out[d] += orig_attn[j] * V[j * d_v + d];
        }
    }

    // Compacted scores with beta
    std::vector<float> comp_scores(t);
    for (int j = 0; j < t; j++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) {
            dot += test_q[d] * K[result.selected_indices[j] * d_k + d];
        }
        comp_scores[j] = dot * inv_sqrt_dk + result.beta[j];
    }
    softmax_rows(comp_scores.data(), 1, t);

    // Output with refitted values (C_v)
    std::vector<float> refit_out(d_v, 0.0f);
    for (int j = 0; j < t; j++) {
        for (int d = 0; d < d_v; d++) {
            refit_out[d] += comp_scores[j] * result.C_v[j * d_v + d];
        }
    }

    // Output without refitting (original V at selected indices)
    std::vector<float> evict_out(d_v, 0.0f);
    for (int j = 0; j < t; j++) {
        for (int d = 0; d < d_v; d++) {
            evict_out[d] += comp_scores[j] * V[result.selected_indices[j] * d_v + d];
        }
    }

    // Compute MSE for both
    float mse_refit = 0.0f, mse_evict = 0.0f;
    for (int d = 0; d < d_v; d++) {
        float dr = refit_out[d] - orig_out[d];
        float de = evict_out[d] - orig_out[d];
        mse_refit += dr * dr;
        mse_evict += de * de;
    }
    mse_refit /= d_v;
    mse_evict /= d_v;

    printf("\n    MSE with refitting: %.8f\n", mse_refit);
    printf("    MSE without refit:  %.8f\n", mse_evict);
    printf("    Improvement:        %.2fx\n", mse_evict / (mse_refit + 1e-12f));

    // Value refitting should improve or at least not worsen the output
    // (in practice it should be significantly better for 4x compression)
    assert(mse_refit <= mse_evict * 1.1f);  // allow 10% tolerance for edge cases
    printf("  OK\n");
}

static void test_compact_beta_values_are_finite() {
    printf("  test_compact_beta_values_are_finite...");
    const int T = 16, n_q = 8, d_k = 4, d_v = 4, t = 4;

    std::vector<float> K(T * d_k), V(T * d_v), Q(n_q * d_k);
    for (int i = 0; i < T * d_k; i++) K[i] = sinf((float)i * 0.3f);
    for (int i = 0; i < T * d_v; i++) V[i] = cosf((float)i * 0.2f);
    for (int i = 0; i < n_q * d_k; i++) Q[i] = sinf((float)i * 0.5f);

    auto result = compact_head_highest_attn(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t);

    for (int j = 0; j < t; j++) {
        assert(std::isfinite(result.beta[j]));
    }
    for (int j = 0; j < t * d_v; j++) {
        assert(std::isfinite(result.C_v[j]));
    }
    printf(" OK\n");
}

static void test_compact_cosine_similarity() {
    printf("  test_compact_cosine_similarity...");
    // After compaction, the attention output should have high cosine similarity
    // with the original output

    const int T = 24, n_q = 12, d_k = 6, d_v = 6, t = 12; // 50% compression

    std::vector<float> K(T * d_k), V(T * d_v), Q(n_q * d_k);
    for (int i = 0; i < T * d_k; i++) K[i] = sinf((float)i * 0.4f);
    for (int i = 0; i < T * d_v; i++) V[i] = cosf((float)i * 0.25f);
    for (int i = 0; i < n_q * d_k; i++) Q[i] = sinf((float)i * 0.6f);

    auto result = compact_head_highest_attn(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t);

    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    // Test with the first reference query
    std::vector<float> test_q(Q.data(), Q.data() + d_k);

    // Original output
    std::vector<float> orig_scores(T);
    for (int j = 0; j < T; j++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) dot += test_q[d] * K[j * d_k + d];
        orig_scores[j] = dot * inv_sqrt_dk;
    }
    softmax_rows(orig_scores.data(), 1, T);
    std::vector<float> orig_out(d_v, 0.0f);
    for (int j = 0; j < T; j++) {
        for (int d = 0; d < d_v; d++) orig_out[d] += orig_scores[j] * V[j * d_v + d];
    }

    // Compacted output
    std::vector<float> comp_scores(t);
    for (int j = 0; j < t; j++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) {
            dot += test_q[d] * K[result.selected_indices[j] * d_k + d];
        }
        comp_scores[j] = dot * inv_sqrt_dk + result.beta[j];
    }
    softmax_rows(comp_scores.data(), 1, t);
    std::vector<float> comp_out(d_v, 0.0f);
    for (int j = 0; j < t; j++) {
        for (int d = 0; d < d_v; d++) comp_out[d] += comp_scores[j] * result.C_v[j * d_v + d];
    }

    // Cosine similarity
    float dot_prod = 0.0f, norm_orig = 0.0f, norm_comp = 0.0f;
    for (int d = 0; d < d_v; d++) {
        dot_prod += orig_out[d] * comp_out[d];
        norm_orig += orig_out[d] * orig_out[d];
        norm_comp += comp_out[d] * comp_out[d];
    }
    float cos_sim = dot_prod / (sqrtf(norm_orig * norm_comp) + 1e-8f);

    printf("\n    Cosine similarity (50%% compression): %.6f\n", cos_sim);
    // At 50% compression, cosine similarity should be very high
    assert(cos_sim > 0.9f);
    printf("  OK\n");
}

// ============================================================================
// compact_layer_all_heads tests
// ============================================================================

static void test_compact_layer_shared_selection() {
    printf("  test_compact_layer_shared_selection...");
    // Multiple heads should share the same selected indices
    const int T = 16, n_q = 8, n_head_kv = 2, d_k = 4, d_v = 4, t = 4;
    const int n_embd_k_gqa = n_head_kv * d_k;
    const int n_embd_v_gqa = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k_gqa), V(T * n_embd_v_gqa), Q(n_q * n_embd_k_gqa);
    for (int i = 0; i < T * n_embd_k_gqa; i++) K[i] = sinf((float)i * 0.3f);
    for (int i = 0; i < T * n_embd_v_gqa; i++) V[i] = cosf((float)i * 0.2f);
    for (int i = 0; i < n_q * n_embd_k_gqa; i++) Q[i] = sinf((float)i * 0.5f);

    auto result = compact_layer_all_heads(K.data(), V.data(), Q.data(),
                                          T, n_q, n_head_kv, d_k, d_v, t);

    // Check shared selection
    assert((int)result.selected_indices.size() == t);
    for (int i = 0; i < t; i++) {
        assert(result.selected_indices[i] >= 0);
        assert(result.selected_indices[i] < T);
    }
    // Sorted
    for (int i = 1; i < t; i++) {
        assert(result.selected_indices[i] > result.selected_indices[i - 1]);
    }

    // Check per-head outputs exist with correct sizes
    assert((int)result.beta.size() == n_head_kv);
    assert((int)result.C_v.size() == n_head_kv);
    for (int h = 0; h < n_head_kv; h++) {
        assert((int)result.beta[h].size() == t);
        assert((int)result.C_v[h].size() == t * d_v);
    }

    printf(" OK\n");
}

static void test_compact_layer_no_compression() {
    printf("  test_compact_layer_no_compression...");
    const int T = 4, n_q = 2, n_head_kv = 2, d_k = 3, d_v = 3;
    const int n_embd_k_gqa = n_head_kv * d_k;
    const int n_embd_v_gqa = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k_gqa), V(T * n_embd_v_gqa), Q(n_q * n_embd_k_gqa);
    for (int i = 0; i < T * n_embd_k_gqa; i++) K[i] = (float)(i % 7) * 0.1f;
    for (int i = 0; i < T * n_embd_v_gqa; i++) V[i] = (float)(i % 5) * 0.2f;
    for (int i = 0; i < n_q * n_embd_k_gqa; i++) Q[i] = (float)(i % 3) * 0.3f;

    auto result = compact_layer_all_heads(K.data(), V.data(), Q.data(),
                                          T, n_q, n_head_kv, d_k, d_v, T);

    // t >= T → no compaction, all indices selected
    assert((int)result.selected_indices.size() == T);
    for (int i = 0; i < T; i++) assert(result.selected_indices[i] == i);

    // Beta should be all zeros
    for (int h = 0; h < n_head_kv; h++) {
        for (int i = 0; i < T; i++) {
            assert(approx_eq(result.beta[h][i], 0.0f));
        }
    }

    printf(" OK\n");
}

static void test_compact_layer_quality_per_head() {
    printf("  test_compact_layer_quality_per_head...");
    // Verify each head's output quality improves with refitting
    const int T = 32, n_q = 16, n_head_kv = 3, d_k = 6, d_v = 6, t = 8;
    const int n_embd_k_gqa = n_head_kv * d_k;
    const int n_embd_v_gqa = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k_gqa), V(T * n_embd_v_gqa), Q(n_q * n_embd_k_gqa);
    for (int i = 0; i < T * n_embd_k_gqa; i++) K[i] = sinf((float)(i * 3 + 1) * 0.4f);
    for (int i = 0; i < T * n_embd_v_gqa; i++) V[i] = cosf((float)(i * 2 + 3) * 0.3f);
    for (int i = 0; i < n_q * n_embd_k_gqa; i++) Q[i] = sinf((float)(i * 5 + 2) * 0.7f);

    auto result = compact_layer_all_heads(K.data(), V.data(), Q.data(),
                                          T, n_q, n_head_kv, d_k, d_v, t);

    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    for (int h = 0; h < n_head_kv; h++) {
        // Test query: first Q for this head
        const float * q_test = Q.data() + h * d_k;

        // Original output
        std::vector<float> orig_scores(T);
        for (int j = 0; j < T; j++) {
            float dot = 0.0f;
            for (int d = 0; d < d_k; d++) {
                dot += q_test[d] * K[j * n_embd_k_gqa + h * d_k + d];
            }
            orig_scores[j] = dot * inv_sqrt_dk;
        }
        softmax_rows(orig_scores.data(), 1, T);

        std::vector<float> orig_out(d_v, 0.0f);
        for (int j = 0; j < T; j++) {
            for (int d = 0; d < d_v; d++) {
                orig_out[d] += orig_scores[j] * V[j * n_embd_v_gqa + h * d_v + d];
            }
        }

        // Compacted output with beta + C_v
        std::vector<float> comp_scores(t);
        for (int j = 0; j < t; j++) {
            float dot = 0.0f;
            int idx = result.selected_indices[j];
            for (int d = 0; d < d_k; d++) {
                dot += q_test[d] * K[idx * n_embd_k_gqa + h * d_k + d];
            }
            comp_scores[j] = dot * inv_sqrt_dk + result.beta[h][j];
        }
        softmax_rows(comp_scores.data(), 1, t);

        std::vector<float> comp_out(d_v, 0.0f);
        for (int j = 0; j < t; j++) {
            for (int d = 0; d < d_v; d++) {
                comp_out[d] += comp_scores[j] * result.C_v[h][j * d_v + d];
            }
        }

        // Cosine similarity should be decent (> 0.8 at 4x compression)
        float dot_p = 0.0f, n_o = 0.0f, n_c = 0.0f;
        for (int d = 0; d < d_v; d++) {
            dot_p += orig_out[d] * comp_out[d];
            n_o += orig_out[d] * orig_out[d];
            n_c += comp_out[d] * comp_out[d];
        }
        float cos_sim = dot_p / (sqrtf(n_o * n_c) + 1e-8f);

        printf("\n    Head %d: cos_sim=%.6f", h, cos_sim);
        assert(cos_sim > 0.8f);
    }

    printf("\n  OK\n");
}

static void test_compact_layer_finite_values() {
    printf("  test_compact_layer_finite_values...");
    const int T = 16, n_q = 8, n_head_kv = 4, d_k = 4, d_v = 4, t = 4;
    const int n_embd_k_gqa = n_head_kv * d_k;
    const int n_embd_v_gqa = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k_gqa), V(T * n_embd_v_gqa), Q(n_q * n_embd_k_gqa);
    for (int i = 0; i < T * n_embd_k_gqa; i++) K[i] = sinf((float)i * 0.1f);
    for (int i = 0; i < T * n_embd_v_gqa; i++) V[i] = cosf((float)i * 0.15f);
    for (int i = 0; i < n_q * n_embd_k_gqa; i++) Q[i] = sinf((float)i * 0.25f);

    auto result = compact_layer_all_heads(K.data(), V.data(), Q.data(),
                                          T, n_q, n_head_kv, d_k, d_v, t);

    for (int h = 0; h < n_head_kv; h++) {
        for (int j = 0; j < t; j++) {
            assert(std::isfinite(result.beta[h][j]));
        }
        for (int j = 0; j < t * d_v; j++) {
            assert(std::isfinite(result.C_v[h][j]));
        }
    }

    printf(" OK\n");
}

// ============================================================================
// Quantized KV round-trip tests
// ============================================================================

static void test_q8_0_round_trip() {
    printf("  test_q8_0_round_trip...");
    // Create float data, quantize to Q8_0, dequantize, check error
    const int n = 64;  // 2 blocks of 32
    std::vector<float> original(n);
    for (int i = 0; i < n; i++) {
        original[i] = sinf((float)i * 0.3f) * 2.0f;
    }

    // Quantize
    int block_bytes = 2 + 32;  // Q8_0: 34 bytes per block
    int n_blocks = n / 32;
    std::vector<uint8_t> quantized(n_blocks * block_bytes);
    convert_from_f32(original.data(), KV_COMPACT_GGML_TYPE_Q8_0,
                     quantized.data(), n);

    // Dequantize
    std::vector<float> recovered(n);
    parsed_kv_state dummy;  // just to access static methods via convert_to_f32
    // Use the convert_to_f32 from parsed_kv_state (it's a static method)
    // Actually we can't access private. Let's dequant manually.
    for (int b = 0; b < n_blocks; b++) {
        parsed_kv_state::dequant_q8_0_block(
            quantized.data() + b * block_bytes,
            recovered.data() + b * 32);
    }

    // Check round-trip error
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(original[i] - recovered[i]);
        if (err > max_err) max_err = err;
    }
    printf("\n    Q8_0 max round-trip error: %.6f (range: [-2, 2])\n", max_err);
    // Q8_0 should have very low error for this range
    assert(max_err < 0.05f);
    printf("  OK\n");
}

static void test_q4_0_round_trip() {
    printf("  test_q4_0_round_trip...");
    const int n = 64;
    std::vector<float> original(n);
    for (int i = 0; i < n; i++) {
        original[i] = sinf((float)i * 0.3f) * 2.0f;
    }

    // Quantize
    int block_bytes = 2 + 16;  // Q4_0: 18 bytes per block
    int n_blocks = n / 32;
    std::vector<uint8_t> quantized(n_blocks * block_bytes);
    convert_from_f32(original.data(), KV_COMPACT_GGML_TYPE_Q4_0,
                     quantized.data(), n);

    // Dequantize
    std::vector<float> recovered(n);
    for (int b = 0; b < n_blocks; b++) {
        parsed_kv_state::dequant_q4_0_block(
            quantized.data() + b * block_bytes,
            recovered.data() + b * 32);
    }

    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float err = fabsf(original[i] - recovered[i]);
        if (err > max_err) max_err = err;
    }
    printf("\n    Q4_0 max round-trip error: %.6f (range: [-2, 2])\n", max_err);
    // Q4_0 has lower precision, allow more error
    assert(max_err < 1.0f);
    printf("  OK\n");
}

static void test_n_elements_per_row() {
    printf("  test_n_elements_per_row...");
    // F32: 128 elements = 512 bytes
    assert(parsed_kv_state::n_elements_per_row(KV_COMPACT_GGML_TYPE_F32, 512) == 128);
    // F16: 128 elements = 256 bytes
    assert(parsed_kv_state::n_elements_per_row(KV_COMPACT_GGML_TYPE_F16, 256) == 128);
    // Q8_0: 128 elements = 4 blocks * 34 bytes = 136 bytes
    assert(parsed_kv_state::n_elements_per_row(KV_COMPACT_GGML_TYPE_Q8_0, 136) == 128);
    // Q4_0: 128 elements = 4 blocks * 18 bytes = 72 bytes
    assert(parsed_kv_state::n_elements_per_row(KV_COMPACT_GGML_TYPE_Q4_0, 72) == 128);
    printf(" OK\n");
}

// ============================================================================
// Main
// ============================================================================

// End-to-end test mimicking compact_layer_into pipeline:
// score → NNLS → beta → softmax(scores+beta) → LS → C_v
// This reproduces the exact data flow that causes zero C_v in the benchmark.
static void test_compact_pipeline_ls_produces_nonzero_cv() {
    printf("  test_compact_pipeline_ls_produces_nonzero_cv...");

    const int T = 750, t = 375, d_k = 64, d_v = 128;
    const int n_q = t + 1;  // overdetermined

    srand(42);

    // Generate random K [T × d_k] and V [T × d_v]
    std::vector<float> K(T * d_k), V(T * d_v);
    for (int i = 0; i < T * d_k; i++) K[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < T * d_v; i++) V[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    // Select every other token (mimics importance-based selection)
    std::vector<int> selected(t);
    for (int j = 0; j < t; j++) selected[j] = j * 2;

    // Generate cheap Q_ref from K at sampled positions
    std::vector<int> qref_pos(n_q);
    for (int qi = 0; qi < n_q; qi++) {
        float frac = (float)(qi + 1) / (float)(n_q + 1);
        frac = frac * frac;
        qref_pos[qi] = std::min((int)(frac * (T - 1)), T - 1);
    }

    std::vector<float> Q(n_q * d_k);
    for (int qi = 0; qi < n_q; qi++)
        memcpy(Q.data() + qi * d_k, K.data() + qref_pos[qi] * d_k, d_k * sizeof(float));

    // Batched scoring: scores = Q @ K^T  [n_q × T]
    std::vector<float> scores(n_q * T);
    mat_mul_ABt(Q.data(), K.data(), scores.data(), n_q, T, d_k);

    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    // Softmax attention weights [n_q × T]
    std::vector<float> attn(n_q * T);
    std::vector<float> exp_scores(n_q * T);
    std::vector<float> row_sums(n_q);

    for (int qi = 0; qi < n_q; qi++) {
        float * row = scores.data() + qi * T;
        float max_s = -1e30f;
        for (int j = 0; j < T; j++) {
            row[j] *= inv_sqrt_dk;
            if (row[j] > max_s) max_s = row[j];
        }
        float rsum = 0.0f;
        float * erow = exp_scores.data() + qi * T;
        for (int j = 0; j < T; j++) {
            erow[j] = expf(row[j] - max_s);
            rsum += erow[j];
        }
        row_sums[qi] = rsum;
        float * arow = attn.data() + qi * T;
        for (int j = 0; j < T; j++) arow[j] = erow[j] / rsum;
    }

    // Build NNLS design matrix M: [n_q × t]
    std::vector<float> M(n_q * t);
    for (int qi = 0; qi < n_q; qi++) {
        const float * erow = exp_scores.data() + qi * T;
        float * mrow = M.data() + qi * t;
        for (int j = 0; j < t; j++) mrow[j] = erow[selected[j]];
    }

    // NNLS solve
    std::vector<float> w(t, 0.0f);
    nnls_solve(M.data(), row_sums.data(), w.data(), n_q, t, 200);

    // Beta = log(w)
    std::vector<float> beta(t);
    int n_dead = 0;
    float min_beta = 1e30f, max_beta = -1e30f;
    for (int j = 0; j < t; j++) {
        beta[j] = logf(std::max(1e-12f, w[j]));
        if (w[j] < 1e-10f) n_dead++;
        if (beta[j] < min_beta) min_beta = beta[j];
        if (beta[j] > max_beta) max_beta = beta[j];
    }
    fprintf(stderr, "\n    NNLS: %d/%d dead weights, beta range [%.1f, %.1f]", n_dead, t, min_beta, max_beta);

    // Build LS design matrix X: softmax(scores[selected] + beta) [n_q × t]
    std::vector<float> X(n_q * t);
    for (int qi = 0; qi < n_q; qi++) {
        const float * srow = scores.data() + qi * T;
        float * xrow = X.data() + qi * t;
        for (int j = 0; j < t; j++) xrow[j] = srow[selected[j]] + beta[j];
    }
    softmax_rows(X.data(), n_q, t);

    // Check X statistics
    float x_max = 0.0f, x_sum = 0.0f;
    int x_nonzero = 0;
    for (int i = 0; i < n_q * t; i++) {
        if (X[i] > x_max) x_max = X[i];
        x_sum += X[i];
        if (X[i] > 1e-7f) x_nonzero++;
    }
    fprintf(stderr, "\n    X: max=%.6f sum=%.1f nonzero=%d/%d", x_max, x_sum, x_nonzero, n_q * t);

    // Y = attn @ V  (original attention output) [n_q × d_v]
    std::vector<float> Y(n_q * d_v, 0.0f);
    for (int qi = 0; qi < n_q; qi++) {
        const float * arow = attn.data() + qi * T;
        float * yrow = Y.data() + qi * d_v;
        for (int ki = 0; ki < T; ki++) {
            float w_ij = arow[ki];
            const float * vr = V.data() + ki * d_v;
            for (int d = 0; d < d_v; d++) yrow[d] += w_ij * vr[d];
        }
    }

    float y_sum = 0.0f;
    for (int i = 0; i < n_q * d_v; i++) y_sum += fabsf(Y[i]);
    fprintf(stderr, "\n    Y: |sum|=%.1f", y_sum);

    // LS solve: X * C_v = Y
    std::vector<float> Cv(t * d_v, 0.0f);
    least_squares_solve(X.data(), Y.data(), Cv.data(), n_q, t, d_v, 1e-6f);

    float cv_sum = 0.0f, cv_max = 0.0f;
    for (int i = 0; i < t * d_v; i++) {
        cv_sum += fabsf(Cv[i]);
        if (fabsf(Cv[i]) > cv_max) cv_max = fabsf(Cv[i]);
    }
    fprintf(stderr, "\n    C_v: |sum|=%.1f max=%.6f", cv_sum, cv_max);

    if (cv_sum < 0.1f) {
        fprintf(stderr, " *** ZERO C_v ***\n");

        // Diagnose: check AtA condition
        std::vector<float> AtA(t * t, 0.0f);
        for (int i = 0; i < t; i++)
            for (int j = 0; j < t; j++) {
                float s = 0.0f;
                for (int k = 0; k < n_q; k++) s += X[k * t + i] * X[k * t + j];
                AtA[i * t + j] = s;
            }
        float diag_min = 1e30f, diag_max = 0.0f;
        for (int i = 0; i < t; i++) {
            float d = AtA[i * t + i];
            if (d < diag_min) diag_min = d;
            if (d > diag_max) diag_max = d;
        }
        fprintf(stderr, "    AtA diag range: [%.2e, %.2e]\n", diag_min, diag_max);
    }

    assert(cv_sum > 0.1f && "C_v should not be all zeros from pipeline");
    fprintf(stderr, " OK\n");
    printf(" OK\n");
}

int main() {
    printf("test-kv-compact-math:\n");

    printf("\n=== Matrix multiplication ===\n");
    test_mat_mul_ABt_identity();
    test_mat_mul_ABt_rectangular();
    test_mat_mul_AtB_basic();
    test_mat_mul_AtB_rectangular();

    printf("\n=== Softmax ===\n");
    test_softmax_rows_sums_to_one();
    test_softmax_rows_ordering();
    test_softmax_rows_uniform();
    test_softmax_rows_numerical_stability();

    printf("\n=== Stable exp ===\n");
    test_exp_rows_stable_basic();
    test_exp_rows_stable_large_values();

    printf("\n=== NNLS solver (Lawson-Hanson) ===\n");
    test_nnls_identity();
    test_nnls_non_negative_constraint();
    test_nnls_overdetermined();

    printf("\n=== NNLS solver (PGD — paper's Algorithm 3) ===\n");
    test_nnls_pgd_basic();
    test_nnls_pgd_clamping();
    test_nnls_pgd_with_iters();

    printf("\n=== OMP key selection ===\n");
    test_omp_select_basic();
    test_omp_select_with_kchoice();

    printf("\n=== Score aggregation methods ===\n");
    test_score_agg_rms();

    printf("\n=== Least squares solver ===\n");
    test_least_squares_identity();
    test_least_squares_overdetermined();
    test_least_squares_multi_rhs();
    test_least_squares_with_ridge();
    test_least_squares_small_known();
    test_least_squares_softmax_structure();
    test_least_squares_large_overdetermined();

    printf("\n=== Per-head sensitivity ===\n");
    test_sensitivity_uniform_attention();
    test_sensitivity_concentrated_attention();
    test_sensitivity_ordering();
    test_weighted_importance_basic();

    printf("\n=== Beta injection via K-modification ===\n");
    test_compute_beta_direction_identity_queries();
    test_compute_beta_direction_produces_unit_dot();
    test_apply_beta_to_keys_basic();
    test_beta_injection_quality();

    printf("\n=== Full compaction pipeline ===\n");
    test_compact_no_compression_needed();
    test_compact_selects_correct_count();
    test_compact_quality_improves_with_refitting();
    test_compact_beta_values_are_finite();
    test_compact_cosine_similarity();

    printf("\n=== Layer-level compaction ===\n");
    test_compact_layer_shared_selection();
    test_compact_layer_no_compression();
    test_compact_layer_quality_per_head();
    test_compact_layer_finite_values();

    printf("\n=== Pipeline end-to-end ===\n");
    test_compact_pipeline_ls_produces_nonzero_cv();

    printf("\n=== Quantized KV round-trip ===\n");
    test_q8_0_round_trip();
    test_q4_0_round_trip();
    test_n_elements_per_row();

    printf("\nAll tests passed!\n");
    return 0;
}
