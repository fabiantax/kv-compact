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
// Main
// ============================================================================

// ============================================================================
// Quantization round-trip tests
// ============================================================================

#include "kv-compact-state.h"

// Helper: compute max absolute error between two float arrays
static float max_abs_error(const float * a, const float * b, int n) {
    float err = 0.0f;
    for (int i = 0; i < n; i++) {
        float d = fabsf(a[i] - b[i]);
        if (d > err) err = d;
    }
    return err;
}

// Helper: compute relative error (max |a-b| / max |a|)
static float relative_error(const float * a, const float * b, int n) {
    float max_val = 0.0f;
    float max_err = 0.0f;
    for (int i = 0; i < n; i++) {
        float av = fabsf(a[i]);
        if (av > max_val) max_val = av;
        float d = fabsf(a[i] - b[i]);
        if (d > max_err) max_err = d;
    }
    return (max_val > 0.0f) ? max_err / max_val : 0.0f;
}

static void test_q8_0_round_trip() {
    printf("  test_q8_0_round_trip...");
    // 64 floats = 2 blocks of Q8_0
    const int n = 64;
    float src[n];
    for (int i = 0; i < n; i++) {
        src[i] = (float)(i - 32) * 0.1f; // range [-3.2, 3.1]
    }

    // Quantize
    const size_t buf_size = (n / KV_COMPACT_QK) * KV_COMPACT_Q8_0_BLOCK_SIZE;
    std::vector<uint8_t> buf(buf_size);
    quantize_q8_0(src, buf.data(), n);

    // Dequantize
    float dst[n];
    parsed_kv_state::convert_to_f32(buf.data(), KV_COMPACT_GGML_TYPE_Q8_0, dst, n);

    // Q8_0 should have very low error (127 levels)
    float rel_err = relative_error(src, dst, n);
    printf(" rel_err=%.6f", rel_err);
    assert(rel_err < 0.02f); // < 2% relative error
    printf(" OK\n");
}

static void test_q4_0_round_trip() {
    printf("  test_q4_0_round_trip...");
    const int n = 64;
    float src[n];
    for (int i = 0; i < n; i++) {
        src[i] = (float)(i - 32) * 0.1f;
    }

    const size_t buf_size = (n / KV_COMPACT_QK) * KV_COMPACT_Q4_0_BLOCK_SIZE;
    std::vector<uint8_t> buf(buf_size);
    quantize_q4_0(src, buf.data(), n);

    float dst[n];
    parsed_kv_state::convert_to_f32(buf.data(), KV_COMPACT_GGML_TYPE_Q4_0, dst, n);

    // Q4_0 has only 16 levels, so higher error expected
    float rel_err = relative_error(src, dst, n);
    printf(" rel_err=%.6f", rel_err);
    assert(rel_err < 0.20f); // < 20% relative error
    printf(" OK\n");
}

static void test_q4_1_round_trip() {
    printf("  test_q4_1_round_trip...");
    const int n = 64;
    float src[n];
    for (int i = 0; i < n; i++) {
        src[i] = (float)i * 0.05f; // range [0, 3.15] — asymmetric
    }

    const size_t buf_size = (n / KV_COMPACT_QK) * KV_COMPACT_Q4_1_BLOCK_SIZE;
    std::vector<uint8_t> buf(buf_size);
    quantize_q4_1(src, buf.data(), n);

    float dst[n];
    parsed_kv_state::convert_to_f32(buf.data(), KV_COMPACT_GGML_TYPE_Q4_1, dst, n);

    float rel_err = relative_error(src, dst, n);
    printf(" rel_err=%.6f", rel_err);
    assert(rel_err < 0.20f);
    printf(" OK\n");
}

static void test_q5_0_round_trip() {
    printf("  test_q5_0_round_trip...");
    const int n = 64;
    float src[n];
    for (int i = 0; i < n; i++) {
        src[i] = (float)(i - 32) * 0.1f;
    }

    const size_t buf_size = (n / KV_COMPACT_QK) * KV_COMPACT_Q5_0_BLOCK_SIZE;
    std::vector<uint8_t> buf(buf_size);
    quantize_q5_0(src, buf.data(), n);

    float dst[n];
    parsed_kv_state::convert_to_f32(buf.data(), KV_COMPACT_GGML_TYPE_Q5_0, dst, n);

    float rel_err = relative_error(src, dst, n);
    printf(" rel_err=%.6f", rel_err);
    assert(rel_err < 0.10f); // 32 levels — between Q4 and Q8
    printf(" OK\n");
}

static void test_q5_1_round_trip() {
    printf("  test_q5_1_round_trip...");
    const int n = 64;
    float src[n];
    for (int i = 0; i < n; i++) {
        src[i] = (float)i * 0.05f;
    }

    const size_t buf_size = (n / KV_COMPACT_QK) * KV_COMPACT_Q5_1_BLOCK_SIZE;
    std::vector<uint8_t> buf(buf_size);
    quantize_q5_1(src, buf.data(), n);

    float dst[n];
    parsed_kv_state::convert_to_f32(buf.data(), KV_COMPACT_GGML_TYPE_Q5_1, dst, n);

    float rel_err = relative_error(src, dst, n);
    printf(" rel_err=%.6f", rel_err);
    assert(rel_err < 0.10f);
    printf(" OK\n");
}

static void test_q8_1_round_trip() {
    printf("  test_q8_1_round_trip...");
    const int n = 64;
    float src[n];
    for (int i = 0; i < n; i++) {
        src[i] = (float)(i - 32) * 0.1f;
    }

    const size_t buf_size = (n / KV_COMPACT_QK) * KV_COMPACT_Q8_1_BLOCK_SIZE;
    std::vector<uint8_t> buf(buf_size);
    quantize_q8_1(src, buf.data(), n);

    float dst[n];
    parsed_kv_state::convert_to_f32(buf.data(), KV_COMPACT_GGML_TYPE_Q8_1, dst, n);

    float rel_err = relative_error(src, dst, n);
    printf(" rel_err=%.6f", rel_err);
    assert(rel_err < 0.02f);
    printf(" OK\n");
}

static void test_q8_0_zeros() {
    printf("  test_q8_0_zeros...");
    const int n = 32;
    float src[n] = {};

    std::vector<uint8_t> buf((n / KV_COMPACT_QK) * KV_COMPACT_Q8_0_BLOCK_SIZE);
    quantize_q8_0(src, buf.data(), n);

    float dst[n];
    parsed_kv_state::convert_to_f32(buf.data(), KV_COMPACT_GGML_TYPE_Q8_0, dst, n);

    for (int i = 0; i < n; i++) {
        assert(dst[i] == 0.0f);
    }
    printf(" OK\n");
}

static void test_elements_per_row() {
    printf("  test_elements_per_row...");
    // F32: 128 bytes = 32 floats
    assert(parsed_kv_state::elements_per_row(KV_COMPACT_GGML_TYPE_F32, 128) == 32);
    // F16: 128 bytes = 64 halfs
    assert(parsed_kv_state::elements_per_row(KV_COMPACT_GGML_TYPE_F16, 128) == 64);
    // Q8_0: 1 block = 34 bytes = 32 elements
    assert(parsed_kv_state::elements_per_row(KV_COMPACT_GGML_TYPE_Q8_0, 34) == 32);
    assert(parsed_kv_state::elements_per_row(KV_COMPACT_GGML_TYPE_Q8_0, 68) == 64);
    // Q4_0: 1 block = 18 bytes = 32 elements
    assert(parsed_kv_state::elements_per_row(KV_COMPACT_GGML_TYPE_Q4_0, 18) == 32);
    assert(parsed_kv_state::elements_per_row(KV_COMPACT_GGML_TYPE_Q4_0, 36) == 64);
    printf(" OK\n");
}

static void test_row_bytes_for() {
    printf("  test_row_bytes_for...");
    assert(parsed_kv_state::row_bytes_for(KV_COMPACT_GGML_TYPE_F32, 32) == 128);
    assert(parsed_kv_state::row_bytes_for(KV_COMPACT_GGML_TYPE_F16, 32) == 64);
    assert(parsed_kv_state::row_bytes_for(KV_COMPACT_GGML_TYPE_Q8_0, 32) == 34);
    assert(parsed_kv_state::row_bytes_for(KV_COMPACT_GGML_TYPE_Q8_0, 64) == 68);
    assert(parsed_kv_state::row_bytes_for(KV_COMPACT_GGML_TYPE_Q4_0, 32) == 18);
    assert(parsed_kv_state::row_bytes_for(KV_COMPACT_GGML_TYPE_Q4_0, 64) == 36);
    printf(" OK\n");
}

// Test that parse + build_compacted_state round-trips correctly for Q8_0
static void test_state_round_trip_q8_0() {
    printf("  test_state_round_trip_q8_0...");

    // Build a synthetic state buffer with Q8_0 K and V
    // 1 stream, 1 layer, 32 cells (1 block), 32-dim embeddings, non-transposed
    const int cell_count = 32;
    const int n_embd = 32;  // exactly 1 block per row

    std::vector<uint8_t> state_buf;
    auto write_val = [&](const void * v, size_t sz) {
        const uint8_t * p = (const uint8_t *)v;
        state_buf.insert(state_buf.end(), p, p + sz);
    };

    // n_stream = 1
    uint32_t n_stream = 1;
    write_val(&n_stream, 4);

    // cell_count
    uint32_t cc = cell_count;
    write_val(&cc, 4);

    // Cell metadata: pos=i, n_seq_id=1, seq_id=0
    for (int i = 0; i < cell_count; i++) {
        int32_t pos = i;
        write_val(&pos, 4);
        uint32_t n_seq = 1;
        write_val(&n_seq, 4);
        int32_t sid = 0;
        write_val(&sid, 4);
    }

    // v_trans=0, n_layer=1
    uint32_t v_trans = 0;
    write_val(&v_trans, 4);
    uint32_t n_layer = 1;
    write_val(&n_layer, 4);

    // K layer: type=Q8_0, size_row=34 (1 block for 32 elements)
    int32_t k_type = KV_COMPACT_GGML_TYPE_Q8_0;
    write_val(&k_type, 4);
    uint64_t k_size_row = KV_COMPACT_Q8_0_BLOCK_SIZE; // 34 bytes for 32 floats
    write_val(&k_size_row, 8);

    // Generate K data: quantize known float values
    std::vector<float> k_f32(cell_count * n_embd);
    for (int i = 0; i < cell_count * n_embd; i++) {
        k_f32[i] = sinf((float)i * 0.1f);
    }
    std::vector<uint8_t> k_quant(cell_count * KV_COMPACT_Q8_0_BLOCK_SIZE);
    for (int r = 0; r < cell_count; r++) {
        quantize_q8_0(k_f32.data() + r * n_embd,
                       k_quant.data() + r * KV_COMPACT_Q8_0_BLOCK_SIZE, n_embd);
    }
    write_val(k_quant.data(), k_quant.size());

    // V layer: type=Q8_0
    int32_t v_type = KV_COMPACT_GGML_TYPE_Q8_0;
    write_val(&v_type, 4);
    uint64_t v_size_row = KV_COMPACT_Q8_0_BLOCK_SIZE;
    write_val(&v_size_row, 8);

    std::vector<float> v_f32(cell_count * n_embd);
    for (int i = 0; i < cell_count * n_embd; i++) {
        v_f32[i] = cosf((float)i * 0.1f);
    }
    std::vector<uint8_t> v_quant(cell_count * KV_COMPACT_Q8_0_BLOCK_SIZE);
    for (int r = 0; r < cell_count; r++) {
        quantize_q8_0(v_f32.data() + r * n_embd,
                       v_quant.data() + r * KV_COMPACT_Q8_0_BLOCK_SIZE, n_embd);
    }
    write_val(v_quant.data(), v_quant.size());

    // Parse the state
    parsed_kv_state pstate;
    bool ok = pstate.parse(state_buf.data(), state_buf.size());
    assert(ok);
    assert(pstate.streams.size() == 1);
    assert(pstate.streams[0].cell_count == (uint32_t)cell_count);
    assert(pstate.streams[0].layers[0].k_type == KV_COMPACT_GGML_TYPE_Q8_0);
    assert((int)pstate.streams[0].layers[0].K.size() == cell_count * n_embd);

    // Verify parsed K values are close to originals (within Q8_0 quantization error)
    // Note: k_f32 is the original, but the state was written from quantized data,
    // so we should compare against dequantized values
    float k_dequant[cell_count * n_embd];
    for (int r = 0; r < cell_count; r++) {
        parsed_kv_state::convert_to_f32(
            k_quant.data() + r * KV_COMPACT_Q8_0_BLOCK_SIZE,
            KV_COMPACT_GGML_TYPE_Q8_0,
            k_dequant + r * n_embd, n_embd);
    }
    float k_err = max_abs_error(pstate.streams[0].layers[0].K.data(), k_dequant, cell_count * n_embd);
    assert(k_err < 1e-6f); // Should be exact match (same dequantize path)

    printf(" parse_ok rel_err_K<1e-6");
    printf(" OK\n");
}

// Test convert_from_f32 dispatches correctly
static void test_convert_from_f32_dispatch() {
    printf("  test_convert_from_f32_dispatch...");
    const int n = 32;
    float src[n];
    for (int i = 0; i < n; i++) src[i] = (float)(i - 16) * 0.1f;

    // Test Q8_0
    {
        std::vector<uint8_t> buf(KV_COMPACT_Q8_0_BLOCK_SIZE);
        convert_from_f32(src, KV_COMPACT_GGML_TYPE_Q8_0, buf.data(), n);
        float dst[n];
        parsed_kv_state::convert_to_f32(buf.data(), KV_COMPACT_GGML_TYPE_Q8_0, dst, n);
        assert(relative_error(src, dst, n) < 0.02f);
    }

    // Test Q4_0
    {
        std::vector<uint8_t> buf(KV_COMPACT_Q4_0_BLOCK_SIZE);
        convert_from_f32(src, KV_COMPACT_GGML_TYPE_Q4_0, buf.data(), n);
        float dst[n];
        parsed_kv_state::convert_to_f32(buf.data(), KV_COMPACT_GGML_TYPE_Q4_0, dst, n);
        assert(relative_error(src, dst, n) < 0.20f);
    }

    // Test F32
    {
        std::vector<uint8_t> buf(n * 4);
        convert_from_f32(src, KV_COMPACT_GGML_TYPE_F32, buf.data(), n);
        float dst[n];
        parsed_kv_state::convert_to_f32(buf.data(), KV_COMPACT_GGML_TYPE_F32, dst, n);
        assert(max_abs_error(src, dst, n) < 1e-7f);
    }

    printf(" OK\n");
}

// ============================================================================
// Beta averaging and injection tests
// ============================================================================

static void test_beta_averaging_across_heads() {
    printf("  test_beta_averaging_across_heads...");

    // Use compact_layer_all_heads to get per-head betas, then verify averaging
    const int T = 16, n_q = 8, d_k = 4, d_v = 4, t = 6;
    const int n_head_kv = 3;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;

    // Create concatenated K, V, Q data
    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
    for (size_t i = 0; i < K.size(); i++) K[i] = sinf((float)i * 0.3f);
    for (size_t i = 0; i < V.size(); i++) V[i] = cosf((float)i * 0.2f);
    for (size_t i = 0; i < Q.size(); i++) Q[i] = sinf((float)i * 0.5f);

    auto result = compact_layer_all_heads(K.data(), V.data(), Q.data(),
                                          T, n_q, n_head_kv, d_k, d_v, t);

    // Verify per-head beta is available
    assert((int)result.beta.size() == n_head_kv);
    for (int h = 0; h < n_head_kv; h++) {
        assert((int)result.beta[h].size() == t);
    }

    // Compute averaged beta (same logic as CLI)
    std::vector<float> beta_avg(t, 0.0f);
    for (int h = 0; h < n_head_kv; h++) {
        for (int j = 0; j < t; j++) {
            beta_avg[j] += result.beta[h][j];
        }
    }
    for (int j = 0; j < t; j++) {
        beta_avg[j] /= n_head_kv;
    }

    // Verify averaged beta is finite and reasonable
    for (int j = 0; j < t; j++) {
        assert(std::isfinite(beta_avg[j]));
    }

    // Verify averaged beta is between min and max of per-head betas
    for (int j = 0; j < t; j++) {
        float b_min = result.beta[0][j], b_max = result.beta[0][j];
        for (int h = 1; h < n_head_kv; h++) {
            if (result.beta[h][j] < b_min) b_min = result.beta[h][j];
            if (result.beta[h][j] > b_max) b_max = result.beta[h][j];
        }
        assert(beta_avg[j] >= b_min - EPS && beta_avg[j] <= b_max + EPS);
    }

    printf(" OK\n");
}

static void test_beta_improves_attention_output() {
    printf("  test_beta_improves_attention_output...");

    // Test that applying beta to compacted attention logits improves output
    // compared to not using beta (beta=0)
    const int T = 24, n_q = 12, d_k = 6, d_v = 6, t = 8;

    std::vector<float> K(T * d_k), V(T * d_v), Q(n_q * d_k);
    for (int i = 0; i < T * d_k; i++) K[i] = sinf((float)i * 0.4f);
    for (int i = 0; i < T * d_v; i++) V[i] = cosf((float)i * 0.25f);
    for (int i = 0; i < n_q * d_k; i++) Q[i] = sinf((float)i * 0.6f);

    auto result = compact_head_highest_attn(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t);

    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    // Test with a new query
    std::vector<float> q_test(d_k);
    for (int d = 0; d < d_k; d++) q_test[d] = sinf((float)d * 1.7f);

    // Original output (full cache)
    std::vector<float> orig_scores(T);
    for (int j = 0; j < T; j++) {
        float dot = 0.0f;
        for (int d = 0; d < d_k; d++) dot += q_test[d] * K[j * d_k + d];
        orig_scores[j] = dot * inv_sqrt_dk;
    }
    softmax_rows(orig_scores.data(), 1, T);
    std::vector<float> orig_out(d_v, 0.0f);
    for (int j = 0; j < T; j++) {
        for (int d = 0; d < d_v; d++) {
            orig_out[d] += orig_scores[j] * V[j * d_v + d];
        }
    }

    // Compacted output WITH beta
    std::vector<float> scores_beta(t);
    for (int j = 0; j < t; j++) {
        float dot = 0.0f;
        const float * k_row = K.data() + result.selected_indices[j] * d_k;
        for (int d = 0; d < d_k; d++) dot += q_test[d] * k_row[d];
        scores_beta[j] = dot * inv_sqrt_dk + result.beta[j]; // WITH beta
    }
    softmax_rows(scores_beta.data(), 1, t);
    std::vector<float> out_beta(d_v, 0.0f);
    for (int j = 0; j < t; j++) {
        for (int d = 0; d < d_v; d++) {
            out_beta[d] += scores_beta[j] * result.C_v[j * d_v + d];
        }
    }

    // Compacted output WITHOUT beta
    std::vector<float> scores_nobeta(t);
    for (int j = 0; j < t; j++) {
        float dot = 0.0f;
        const float * k_row = K.data() + result.selected_indices[j] * d_k;
        for (int d = 0; d < d_k; d++) dot += q_test[d] * k_row[d];
        scores_nobeta[j] = dot * inv_sqrt_dk; // NO beta
    }
    softmax_rows(scores_nobeta.data(), 1, t);
    std::vector<float> out_nobeta(d_v, 0.0f);
    for (int j = 0; j < t; j++) {
        for (int d = 0; d < d_v; d++) {
            out_nobeta[d] += scores_nobeta[j] * result.C_v[j * d_v + d];
        }
    }

    // Compute MSE vs original for both
    float mse_beta = 0.0f, mse_nobeta = 0.0f;
    for (int d = 0; d < d_v; d++) {
        float diff_b = orig_out[d] - out_beta[d];
        float diff_n = orig_out[d] - out_nobeta[d];
        mse_beta += diff_b * diff_b;
        mse_nobeta += diff_n * diff_n;
    }
    mse_beta /= d_v;
    mse_nobeta /= d_v;

    printf("\n    MSE with beta:    %.8f\n", mse_beta);
    printf("    MSE without beta: %.8f\n", mse_nobeta);
    printf("    Improvement:      %.2fx\n", mse_nobeta / (mse_beta + 1e-12f));

    // Beta should improve or at least not worsen the output
    // (with refitted C_v, beta should help)
    assert(mse_beta <= mse_nobeta * 1.1f); // allow 10% tolerance

    printf("  OK\n");
}

// ============================================================================
// Non-uniform head budget tests
// ============================================================================

static void test_head_sensitivity_computation() {
    printf("  test_head_sensitivity_computation...");

    // Create two heads: one with sharp attention (low sensitivity),
    // one with broad attention (high sensitivity)
    const int T = 20, n_q = 8, d_k = 4, d_v = 4, t = 5;
    const int n_head_kv = 2;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;

    // Head 0: keys are very distinct (sharp attention)
    // Head 1: keys are similar (broad, uniform attention)
    std::vector<float> K(T * n_embd_k, 0.0f), V(T * n_embd_v, 0.0f);
    std::vector<float> Q(n_q * n_embd_k, 0.0f);

    for (int i = 0; i < T; i++) {
        // Head 0: each key is a strong unit-ish vector in a different direction
        for (int d = 0; d < d_k; d++) {
            K[i * n_embd_k + d] = (d == (i % d_k)) ? 3.0f : 0.0f;
        }
        // Head 1: all keys are nearly identical
        for (int d = 0; d < d_k; d++) {
            K[i * n_embd_k + d_k + d] = 1.0f + 0.01f * sinf((float)(i * d_k + d));
        }
        // Values: arbitrary
        for (int d = 0; d < n_embd_v; d++) {
            V[i * n_embd_v + d] = cosf((float)(i * n_embd_v + d) * 0.3f);
        }
    }

    for (int qi = 0; qi < n_q; qi++) {
        // Head 0 queries: also sharp (match specific keys)
        for (int d = 0; d < d_k; d++) {
            Q[qi * n_embd_k + d] = (d == (qi % d_k)) ? 2.0f : 0.0f;
        }
        // Head 1 queries: all similar
        for (int d = 0; d < d_k; d++) {
            Q[qi * n_embd_k + d_k + d] = 1.0f;
        }
    }

    auto result = compact_layer_all_heads(K.data(), V.data(), Q.data(),
                                          T, n_q, n_head_kv, d_k, d_v, t);

    assert((int)result.head_sensitivity.size() == n_head_kv);

    printf("\n    Head 0 (sharp):  sensitivity=%.4f\n", result.head_sensitivity[0]);
    printf("    Head 1 (broad):  sensitivity=%.4f\n", result.head_sensitivity[1]);

    // Head 1 (broad attention) should be MORE sensitive than head 0 (sharp)
    // because broad attention spreads mass across many keys, so losing keys hurts more
    assert(result.head_sensitivity[1] > result.head_sensitivity[0]);

    printf("  OK\n");
}

static void test_sensitivity_weighted_selection() {
    printf("  test_sensitivity_weighted_selection...");

    // Test that sensitivity-weighted selection improves worst-head quality
    // compared to uniform weighting (all sensitivities = 1.0)
    const int T = 30, n_q = 10, d_k = 4, d_v = 4, t = 6;
    const int n_head_kv = 3;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;

    // Create heads with varying attention patterns
    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);

    // Head 0: sharp retrieval head (focuses on a few keys)
    // Head 1: broad attention head (spreads across many keys)
    // Head 2: medium attention head
    for (int i = 0; i < T; i++) {
        for (int d = 0; d < d_k; d++) {
            // Head 0: very distinct keys
            K[i * n_embd_k + d] = (d == (i % d_k)) ? 5.0f : -0.5f;
            // Head 1: all keys similar, slight variation
            K[i * n_embd_k + d_k + d] = 1.0f + 0.02f * sinf((float)(i * 7 + d));
            // Head 2: moderate distinction
            K[i * n_embd_k + 2*d_k + d] = sinf((float)(i * d_k + d) * 0.5f);
        }
        for (int d = 0; d < n_embd_v; d++) {
            V[i * n_embd_v + d] = cosf((float)(i * n_embd_v + d) * 0.2f);
        }
    }

    for (int qi = 0; qi < n_q; qi++) {
        for (int d = 0; d < d_k; d++) {
            Q[qi * n_embd_k + d] = (d == (qi % d_k)) ? 3.0f : 0.0f;
            Q[qi * n_embd_k + d_k + d] = 1.0f + 0.1f * cosf((float)(qi * d));
            Q[qi * n_embd_k + 2*d_k + d] = sinf((float)(qi * d_k + d) * 0.7f);
        }
    }

    // Sensitivity-weighted (auto-computed)
    auto result_weighted = compact_layer_all_heads(
        K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, t);

    // Uniform weighting (all sensitivity = 1.0)
    std::vector<float> uniform_sens(n_head_kv, 1.0f);
    auto result_uniform = compact_layer_all_heads(
        K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, t,
        uniform_sens.data());

    // Compute per-head cosine similarity for a test query
    std::vector<float> q_test(n_embd_k);
    for (int d = 0; d < n_embd_k; d++) q_test[d] = sinf((float)d * 1.3f);

    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    auto compute_cos_sim = [&](const compacted_layer & res, int h) -> float {
        // Original output
        std::vector<float> orig_scores(T);
        for (int j = 0; j < T; j++) {
            float dot = 0.0f;
            for (int d = 0; d < d_k; d++)
                dot += q_test[h * d_k + d] * K[j * n_embd_k + h * d_k + d];
            orig_scores[j] = dot * inv_sqrt_dk;
        }
        softmax_rows(orig_scores.data(), 1, T);
        std::vector<float> orig_out(d_v, 0.0f);
        for (int j = 0; j < T; j++)
            for (int d = 0; d < d_v; d++)
                orig_out[d] += orig_scores[j] * V[j * n_embd_v + h * d_v + d];

        // Compacted output
        std::vector<float> comp_scores(t);
        for (int j = 0; j < t; j++) {
            float dot = 0.0f;
            int idx = res.selected_indices[j];
            for (int d = 0; d < d_k; d++)
                dot += q_test[h * d_k + d] * K[idx * n_embd_k + h * d_k + d];
            comp_scores[j] = dot * inv_sqrt_dk + res.beta[h][j];
        }
        softmax_rows(comp_scores.data(), 1, t);
        std::vector<float> comp_out(d_v, 0.0f);
        for (int j = 0; j < t; j++)
            for (int d = 0; d < d_v; d++)
                comp_out[d] += comp_scores[j] * res.C_v[h][j * d_v + d];

        // Cosine similarity
        float dot_p = 0.0f, n_o = 0.0f, n_c = 0.0f;
        for (int d = 0; d < d_v; d++) {
            dot_p += orig_out[d] * comp_out[d];
            n_o += orig_out[d] * orig_out[d];
            n_c += comp_out[d] * comp_out[d];
        }
        return dot_p / (sqrtf(n_o * n_c) + 1e-8f);
    };

    printf("\n    Sensitivities:");
    for (int h = 0; h < n_head_kv; h++)
        printf(" h%d=%.4f", h, result_weighted.head_sensitivity[h]);
    printf("\n");

    float worst_weighted = 1.0f, worst_uniform = 1.0f;
    float sum_weighted = 0.0f, sum_uniform = 0.0f;
    for (int h = 0; h < n_head_kv; h++) {
        float cs_w = compute_cos_sim(result_weighted, h);
        float cs_u = compute_cos_sim(result_uniform, h);
        printf("    Head %d:  weighted=%.6f  uniform=%.6f\n", h, cs_w, cs_u);
        if (cs_w < worst_weighted) worst_weighted = cs_w;
        if (cs_u < worst_uniform) worst_uniform = cs_u;
        sum_weighted += cs_w;
        sum_uniform += cs_u;
    }

    float avg_weighted = sum_weighted / n_head_kv;
    float avg_uniform = sum_uniform / n_head_kv;
    printf("    Worst:   weighted=%.6f  uniform=%.6f\n", worst_weighted, worst_uniform);
    printf("    Average: weighted=%.6f  uniform=%.6f\n", avg_weighted, avg_uniform);

    // Sensitivity weighting should improve worst-case or average quality
    // (it prioritizes keys for the heads that need them most)
    assert(worst_weighted >= worst_uniform - 0.01f ||
           avg_weighted >= avg_uniform - 0.001f);

    printf("  OK\n");
}

// ============================================================================
// Alternating minimization tests
// ============================================================================

static void test_alternating_minimization_single_head() {
    printf("  test_alternating_minimization_single_head...");

    // Generate a scenario where single-pass is suboptimal: many keys, moderate compression
    const int T = 40, n_q = 16, d_k = 8, d_v = 8, t = 8;

    std::vector<float> K(T * d_k), V(T * d_v), Q(n_q * d_k);

    // Keys with correlated structure (makes beta/C_v interaction matter)
    for (int i = 0; i < T; i++) {
        for (int d = 0; d < d_k; d++) {
            K[i * d_k + d] = sinf((float)(i * 3 + d) * 0.4f) + 0.5f * cosf((float)(i + d * 7) * 0.2f);
        }
        for (int d = 0; d < d_v; d++) {
            V[i * d_v + d] = cosf((float)(i * d_v + d) * 0.3f) + 0.3f * sinf((float)(i * 2 + d) * 0.7f);
        }
    }
    for (int qi = 0; qi < n_q; qi++) {
        for (int d = 0; d < d_k; d++) {
            Q[qi * d_k + d] = sinf((float)(qi * d_k + d) * 0.5f);
        }
    }

    // Single-pass (n_alt_rounds=1)
    auto r1 = compact_head_highest_attn(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t, 1);

    // With alternation (n_alt_rounds=2)
    auto r2 = compact_head_highest_attn(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t, 2);

    // With more alternation (n_alt_rounds=3)
    auto r3 = compact_head_highest_attn(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t, 3);

    // Compute MSE for each: compare output on test queries
    auto compute_mse = [&](const compacted_head & res) -> float {
        float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
        float total_err = 0.0f;
        for (int qi = 0; qi < n_q; qi++) {
            // Original output
            std::vector<float> orig_scores(T);
            for (int j = 0; j < T; j++) {
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++)
                    dot += Q[qi * d_k + d] * K[j * d_k + d];
                orig_scores[j] = dot * inv_sqrt_dk;
            }
            softmax_rows(orig_scores.data(), 1, T);
            std::vector<float> orig_out(d_v, 0.0f);
            for (int j = 0; j < T; j++)
                for (int d = 0; d < d_v; d++)
                    orig_out[d] += orig_scores[j] * V[j * d_v + d];

            // Compacted output
            std::vector<float> comp_scores(t);
            for (int j = 0; j < t; j++) {
                float dot = 0.0f;
                int idx = res.selected_indices[j];
                for (int d = 0; d < d_k; d++)
                    dot += Q[qi * d_k + d] * K[idx * d_k + d];
                comp_scores[j] = dot * inv_sqrt_dk + res.beta[j];
            }
            softmax_rows(comp_scores.data(), 1, t);
            std::vector<float> comp_out(d_v, 0.0f);
            for (int j = 0; j < t; j++)
                for (int d = 0; d < d_v; d++)
                    comp_out[d] += comp_scores[j] * res.C_v[j * d_v + d];

            for (int d = 0; d < d_v; d++) {
                float e = orig_out[d] - comp_out[d];
                total_err += e * e;
            }
        }
        return total_err / (n_q * d_v);
    };

    float mse1 = compute_mse(r1);
    float mse2 = compute_mse(r2);
    float mse3 = compute_mse(r3);

    printf("\n    MSE: 1-pass=%.8f  2-pass=%.8f  3-pass=%.8f\n",
           mse1, mse2, mse3);
    printf("    Improvement: 2-pass=%.2fx  3-pass=%.2fx over 1-pass\n",
           mse1 / (mse2 + 1e-15f), mse1 / (mse3 + 1e-15f));

    // Alternation should not make things worse
    assert(mse2 <= mse1 * 1.01f);  // allow tiny numerical tolerance
    // More rounds should be at least as good
    assert(mse3 <= mse2 * 1.01f);

    printf("  OK\n");
}

static void test_alternating_minimization_multi_head() {
    printf("  test_alternating_minimization_multi_head...");

    const int T = 30, n_q = 12, d_k = 4, d_v = 4, t = 6;
    const int n_head_kv = 3;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);

    for (int i = 0; i < T; i++) {
        for (int d = 0; d < n_embd_k; d++)
            K[i * n_embd_k + d] = sinf((float)(i * n_embd_k + d) * 0.3f);
        for (int d = 0; d < n_embd_v; d++)
            V[i * n_embd_v + d] = cosf((float)(i * n_embd_v + d) * 0.2f);
    }
    for (int qi = 0; qi < n_q; qi++)
        for (int d = 0; d < n_embd_k; d++)
            Q[qi * n_embd_k + d] = sinf((float)(qi * n_embd_k + d) * 0.5f);

    // 1-pass
    auto r1 = compact_layer_all_heads(
        K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, t,
        nullptr, 1);

    // 2-pass (default)
    auto r2 = compact_layer_all_heads(
        K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, t,
        nullptr, 2);

    // Compute per-head MSE
    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
    float total_mse1 = 0.0f, total_mse2 = 0.0f;

    for (int h = 0; h < n_head_kv; h++) {
        float mse1 = 0.0f, mse2 = 0.0f;
        for (int qi = 0; qi < n_q; qi++) {
            // Original output for head h
            std::vector<float> orig_scores(T);
            for (int j = 0; j < T; j++) {
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++)
                    dot += Q[qi * n_embd_k + h * d_k + d] * K[j * n_embd_k + h * d_k + d];
                orig_scores[j] = dot * inv_sqrt_dk;
            }
            softmax_rows(orig_scores.data(), 1, T);
            std::vector<float> orig_out(d_v, 0.0f);
            for (int j = 0; j < T; j++)
                for (int d = 0; d < d_v; d++)
                    orig_out[d] += orig_scores[j] * V[j * n_embd_v + h * d_v + d];

            // Compacted output for each result
            auto comp_out = [&](const compacted_layer & res) {
                std::vector<float> scores(t), out(d_v, 0.0f);
                for (int j = 0; j < t; j++) {
                    float dot = 0.0f;
                    int idx = res.selected_indices[j];
                    for (int d = 0; d < d_k; d++)
                        dot += Q[qi * n_embd_k + h * d_k + d] * K[idx * n_embd_k + h * d_k + d];
                    scores[j] = dot * inv_sqrt_dk + res.beta[h][j];
                }
                softmax_rows(scores.data(), 1, t);
                for (int j = 0; j < t; j++)
                    for (int d = 0; d < d_v; d++)
                        out[d] += scores[j] * res.C_v[h][j * d_v + d];
                return out;
            };

            auto out1 = comp_out(r1);
            auto out2 = comp_out(r2);

            for (int d = 0; d < d_v; d++) {
                float e1 = orig_out[d] - out1[d];
                float e2 = orig_out[d] - out2[d];
                mse1 += e1 * e1;
                mse2 += e2 * e2;
            }
        }
        mse1 /= (n_q * d_v);
        mse2 /= (n_q * d_v);
        total_mse1 += mse1;
        total_mse2 += mse2;
        printf("\n    Head %d: 1-pass=%.8f  2-pass=%.8f", h, mse1, mse2);
    }

    float avg_mse1 = total_mse1 / n_head_kv;
    float avg_mse2 = total_mse2 / n_head_kv;
    printf("\n    Average: 1-pass=%.8f  2-pass=%.8f  improvement=%.2fx\n",
           avg_mse1, avg_mse2, avg_mse1 / (avg_mse2 + 1e-15f));

    // Alternation should not make things worse
    assert(avg_mse2 <= avg_mse1 * 1.01f);

    printf("  OK\n");
}

// ============================================================================
// Submodular key selection tests
// ============================================================================

static void test_submodular_selects_correct_count() {
    printf("  test_submodular_selects_correct_count...");

    const int T = 20, d_k = 4, t = 7;
    std::vector<float> K(T * d_k), importance(T);
    for (int i = 0; i < T * d_k; i++) K[i] = sinf((float)i * 0.3f);
    for (int i = 0; i < T; i++) importance[i] = 1.0f / (1.0f + i);

    auto selected = submodular_key_select(K.data(), importance.data(), T, d_k, t);

    assert((int)selected.size() == t);
    // Check sorted and unique
    for (int i = 1; i < t; i++) assert(selected[i] > selected[i-1]);
    // Check valid range
    for (int i = 0; i < t; i++) assert(selected[i] >= 0 && selected[i] < T);

    printf(" OK\n");
}

static void test_submodular_prefers_diverse_keys() {
    printf("  test_submodular_prefers_diverse_keys...");

    // Create 8 nearly-identical keys and 4 orthogonal keys
    const int T = 12, d_k = 4, t = 4;
    std::vector<float> K(T * d_k, 0.0f);
    std::vector<float> importance(T, 1.0f);

    // Keys 0-7: all nearly identical (same direction, tiny perturbation)
    for (int i = 0; i < 8; i++) {
        K[i * d_k + 0] = 1.0f;
        K[i * d_k + 1] = 0.5f;
        K[i * d_k + 2] = 0.01f * i;
    }
    // Keys 8-11: each in a distinct orthogonal direction
    for (int i = 8; i < 12; i++) {
        K[i * d_k + (i - 8)] = 2.0f;
    }

    auto selected = submodular_key_select(K.data(), importance.data(), T, d_k, t, 0.5f);

    // Count how many from the redundant cluster (0-7) vs diverse set (8-11)
    int redundant_count = 0;
    for (int idx : selected) {
        if (idx < 8) redundant_count++;
    }

    printf(" redundant=%d/4 diverse=%d/4", redundant_count, 4 - redundant_count);
    // Submodular should NOT select more than 2 from the redundant cluster
    // (diminishing returns: after 1, the rest are already covered)
    assert(redundant_count <= 2);

    printf(" OK\n");
}

static void test_submodular_respects_importance() {
    printf("  test_submodular_respects_importance...");

    // All keys equally spaced but with varying importance
    const int T = 16, d_k = 4, t = 4;
    std::vector<float> K(T * d_k);
    std::vector<float> importance(T);

    for (int i = 0; i < T; i++) {
        for (int d = 0; d < d_k; d++) {
            K[i * d_k + d] = sinf((float)(i * d_k + d) * 0.7f);
        }
        // First 4 keys have 10x importance
        importance[i] = (i < 4) ? 10.0f : 1.0f;
    }

    auto selected = submodular_key_select(K.data(), importance.data(), T, d_k, t, 0.9f);

    // With high lambda (coverage-dominant) and high importance on first 4,
    // should select mostly from the important keys
    int important_count = 0;
    for (int idx : selected) {
        if (idx < 4) important_count++;
    }

    printf(" important=%d/4", important_count);
    assert(important_count >= 2);

    printf(" OK\n");
}

static void test_submodular_vs_max_attn_quality() {
    printf("  test_submodular_vs_max_attn_quality...");

    // Scenario designed to show submodular advantage: clustered keys where
    // max-attention picks redundant keys from the same cluster
    const int T = 30, n_q = 10, d_k = 4, d_v = 4, t = 6;
    const int n_head_kv = 2;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;

    std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);

    // Create 3 clusters of keys, with one cluster having very high attention scores
    for (int i = 0; i < T; i++) {
        int cluster = i % 3;
        for (int h = 0; h < n_head_kv; h++) {
            for (int d = 0; d < d_k; d++) {
                float base = (cluster == 0) ? 3.0f : ((cluster == 1) ? -2.0f : 0.5f);
                K[i * n_embd_k + h * d_k + d] = base + 0.1f * sinf((float)(i * d_k + d + h * 100));
            }
        }
        for (int d = 0; d < n_embd_v; d++) {
            V[i * n_embd_v + d] = cosf((float)(i * n_embd_v + d) * 0.2f);
        }
    }

    // Queries that attend to cluster 0 heavily
    for (int qi = 0; qi < n_q; qi++) {
        for (int h = 0; h < n_head_kv; h++) {
            for (int d = 0; d < d_k; d++) {
                Q[qi * n_embd_k + h * d_k + d] = 2.0f + 0.3f * sinf((float)(qi * d + h));
            }
        }
    }

    // Max-attention selection
    auto r_max = compact_layer_all_heads(
        K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, t,
        nullptr, 2, KEY_SELECT_MAX_ATTN);

    // Submodular selection
    auto r_sub = compact_layer_all_heads(
        K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, t,
        nullptr, 2, KEY_SELECT_SUBMODULAR);

    // Check diversity of selection: count unique clusters
    auto count_clusters = [](const std::vector<int> & sel) {
        bool has[3] = {false, false, false};
        for (int idx : sel) has[idx % 3] = true;
        return (int)has[0] + (int)has[1] + (int)has[2];
    };

    int clusters_max = count_clusters(r_max.selected_indices);
    int clusters_sub = count_clusters(r_sub.selected_indices);

    printf("\n    Clusters covered: max_attn=%d  submodular=%d\n", clusters_max, clusters_sub);

    // Submodular should cover at least as many clusters
    assert(clusters_sub >= clusters_max);

    // Compute per-head MSE for both
    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
    for (int h = 0; h < n_head_kv; h++) {
        auto compute_mse = [&](const compacted_layer & res) {
            float mse = 0.0f;
            for (int qi = 0; qi < n_q; qi++) {
                std::vector<float> orig_scores(T);
                for (int j = 0; j < T; j++) {
                    float dot = 0.0f;
                    for (int d = 0; d < d_k; d++)
                        dot += Q[qi * n_embd_k + h * d_k + d] * K[j * n_embd_k + h * d_k + d];
                    orig_scores[j] = dot * inv_sqrt_dk;
                }
                softmax_rows(orig_scores.data(), 1, T);
                std::vector<float> orig_out(d_v, 0.0f);
                for (int j = 0; j < T; j++)
                    for (int d = 0; d < d_v; d++)
                        orig_out[d] += orig_scores[j] * V[j * n_embd_v + h * d_v + d];

                std::vector<float> comp_scores(t);
                for (int j = 0; j < t; j++) {
                    float dot = 0.0f;
                    int idx = res.selected_indices[j];
                    for (int d = 0; d < d_k; d++)
                        dot += Q[qi * n_embd_k + h * d_k + d] * K[idx * n_embd_k + h * d_k + d];
                    comp_scores[j] = dot * inv_sqrt_dk + res.beta[h][j];
                }
                softmax_rows(comp_scores.data(), 1, t);
                std::vector<float> comp_out(d_v, 0.0f);
                for (int j = 0; j < t; j++)
                    for (int d = 0; d < d_v; d++)
                        comp_out[d] += comp_scores[j] * res.C_v[h][j * d_v + d];

                for (int d = 0; d < d_v; d++) {
                    float e = orig_out[d] - comp_out[d];
                    mse += e * e;
                }
            }
            return mse / (n_q * d_v);
        };

        float mse_max = compute_mse(r_max);
        float mse_sub = compute_mse(r_sub);
        printf("    Head %d: max_attn_mse=%.8f  submodular_mse=%.8f\n", h, mse_max, mse_sub);
    }

    printf("  OK\n");
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

    printf("\n=== NNLS solver ===\n");
    test_nnls_identity();
    test_nnls_non_negative_constraint();
    test_nnls_overdetermined();

    printf("\n=== Least squares solver ===\n");
    test_least_squares_identity();
    test_least_squares_overdetermined();
    test_least_squares_multi_rhs();
    test_least_squares_with_ridge();

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

    printf("\n=== Quantization round-trip ===\n");
    test_q8_0_round_trip();
    test_q4_0_round_trip();
    test_q4_1_round_trip();
    test_q5_0_round_trip();
    test_q5_1_round_trip();
    test_q8_1_round_trip();
    test_q8_0_zeros();
    test_elements_per_row();
    test_row_bytes_for();
    test_state_round_trip_q8_0();
    test_convert_from_f32_dispatch();

    printf("\n=== Beta averaging and injection ===\n");
    test_beta_averaging_across_heads();
    test_beta_improves_attention_output();

    printf("\n=== Non-uniform head budgets ===\n");
    test_head_sensitivity_computation();
    test_sensitivity_weighted_selection();

    printf("\n=== Alternating minimization ===\n");
    test_alternating_minimization_single_head();
    test_alternating_minimization_multi_head();

    printf("\n=== Submodular key selection ===\n");
    test_submodular_selects_correct_count();
    test_submodular_prefers_diverse_keys();
    test_submodular_respects_importance();
    test_submodular_vs_max_attn_quality();

    printf("\nAll tests passed!\n");
    return 0;
}
