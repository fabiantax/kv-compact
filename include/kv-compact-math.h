// KV Cache Compaction via Attention Matching — Math Utilities
//
// Pure CPU float32 linear algebra routines for the compaction algorithm from:
//   "Fast KV Compaction via Attention Matching" (Zweiger et al., 2026)
//   https://arxiv.org/abs/2602.16284
//
// Implements the three-step compaction pipeline (paper Section 3):
//   Step 1: Key selection via max softmax attention (Section 3.1)
//   Step 2: NNLS bias fitting for attention mass preservation (Section 3.2)
//   Step 3: Least-squares value refitting for output preservation (Section 3.3)
//
// See docs/algorithms.md for detailed algorithm descriptions with equations.
// Header-only, no dependencies — extracted for standalone testing.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#if defined(__GNUC__) || defined(__clang__)
#define KV_COMPACT_UNUSED __attribute__((unused))
#else
#define KV_COMPACT_UNUSED
#endif

// ============================================================================
// Linear algebra utilities (CPU-side, float32)
// ============================================================================

// Compute C = A * B^T  where A is (m x k), B is (n x k), result is (m x n)
KV_COMPACT_UNUSED static void mat_mul_ABt(const float * A, const float * B, float * C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[j * k + l];
            }
            C[i * n + j] = sum;
        }
    }
}

// Compute C = A^T * B  where A is (m x k), B is (m x n), result is (k x n)
KV_COMPACT_UNUSED static void mat_mul_AtB(const float * A, const float * B, float * C, int m, int k, int n) {
    // zero out C
    memset(C, 0, k * n * sizeof(float));
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            for (int l = 0; l < n; l++) {
                C[j * n + l] += A[i * k + j] * B[i * n + l];
            }
        }
    }
}

// Softmax over rows: softmax(x_i) = exp(x_i - max) / sum(exp(x_j - max))
// Uses max-shift for numerical stability (paper Section 3.1, algorithms.md §7.1)
static void softmax_rows(float * data, int m, int n) {
    for (int i = 0; i < m; i++) {
        float * row = data + i * n;
        float max_val = row[0];
        for (int j = 1; j < n; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }
        float inv_sum = 1.0f / (sum + 1e-12f);
        for (int j = 0; j < n; j++) {
            row[j] *= inv_sum;
        }
    }
}

// Row-wise exp with max-shift for numerical stability: input (m x n)
// Returns exp(data - max_per_row) and stores the sum per row in row_sums
static void exp_rows_stable(float * data, float * row_sums, int m, int n) {
    for (int i = 0; i < m; i++) {
        float * row = data + i * n;
        float max_val = row[0];
        for (int j = 1; j < n; j++) {
            if (row[j] > max_val) max_val = row[j];
        }
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            row[j] = expf(row[j] - max_val);
            sum += row[j];
        }
        row_sums[i] = sum;
    }
}

// Non-negative least squares via projected gradient descent (paper Section 3.2)
//   min_{w >= 0} ||A*w - b||^2
//
// Used for attention mass matching: M @ w ≈ row_sums, where w = exp(beta).
// M[i,j] = exp(q_i · k_selected_j / √d_k), b[i] = sum_j exp(q_i · k_j / √d_k).
// Step size: 1/trace(A^T A), a conservative bound on 1/λ_max (algorithms.md §4.4).
// Floor at 1e-12 prevents log(0) when converting w → beta = log(w) (§7.5).
//
// A is (m x n), b is (m), w is (n). Returns solution in w.
static void nnls_solve(const float * A, const float * b, float * w, int m, int n, int max_iter = 200) {
    // Precompute A^T * A and A^T * b
    std::vector<float> AtA(n * n);
    std::vector<float> Atb(n);

    // AtA = A^T * A
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += A[k * n + i] * A[k * n + j];
            }
            AtA[i * n + j] = sum;
        }
    }

    // Atb = A^T * b
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int k = 0; k < m; k++) {
            sum += A[k * n + i] * b[k];
        }
        Atb[i] = sum;
    }

    // Initialize w to unconstrained least squares, clamped to >= 0
    // Simple init: w = max(0, (A^T A)^{-1} A^T b) via gradient descent from w=1
    for (int i = 0; i < n; i++) {
        w[i] = 1.0f;
    }

    // Compute step size: 1 / (max eigenvalue of AtA) ≈ 1 / (trace(AtA))
    float trace = 0.0f;
    for (int i = 0; i < n; i++) {
        trace += AtA[i * n + i];
    }
    float step = 1.0f / (trace + 1e-8f);

    // Projected gradient descent
    std::vector<float> grad(n);
    for (int iter = 0; iter < max_iter; iter++) {
        // grad = AtA * w - Atb
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += AtA[i * n + j] * w[j];
            }
            grad[i] = sum - Atb[i];
        }

        // w = max(0, w - step * grad)
        for (int i = 0; i < n; i++) {
            w[i] = std::max(0.0f, w[i] - step * grad[i]);
        }
    }
}

// Least squares for value refitting via regularized normal equations (paper Section 3.3)
//   min ||A*x - b||^2  →  x = (A^T A + λI)^{-1} A^T b
//
// Used to find C_v such that softmax(q·K_selected/√d_k) · C_v ≈ original output.
// A = compacted attention weights [n_q, t], b = original attention output [n_q, d_v].
// Gaussian elimination with partial pivoting (algorithms.md §5.4).
// Ridge λ=1e-6 stabilizes ill-conditioned systems without distortion (§5.5).
//
// A is (m x n), b is (m x p), x is (n x p).
static void least_squares_solve(const float * A, const float * b, float * x,
                                int m, int n, int p, float ridge = 1e-6f) {
    // Compute AtA = A^T * A  (n x n)
    std::vector<float> AtA(n * n, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += A[k * n + i] * A[k * n + j];
            }
            AtA[i * n + j] = sum;
        }
    }

    // Add ridge regularization
    for (int i = 0; i < n; i++) {
        AtA[i * n + i] += ridge;
    }

    // Compute Atb = A^T * b  (n x p)
    std::vector<float> Atb(n * p, 0.0f);
    for (int i = 0; i < n; i++) {
        for (int l = 0; l < p; l++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += A[k * n + i] * b[k * p + l];
            }
            Atb[i * p + l] = sum;
        }
    }

    // Solve AtA * x = Atb via Gaussian elimination with partial pivoting
    // Augmented matrix [AtA | Atb]
    std::vector<float> aug(n * (n + p));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aug[i * (n + p) + j] = AtA[i * n + j];
        }
        for (int j = 0; j < p; j++) {
            aug[i * (n + p) + n + j] = Atb[i * p + j];
        }
    }

    // Forward elimination with partial pivoting
    for (int col = 0; col < n; col++) {
        // Find pivot
        int max_row = col;
        float max_val = fabsf(aug[col * (n + p) + col]);
        for (int row = col + 1; row < n; row++) {
            float val = fabsf(aug[row * (n + p) + col]);
            if (val > max_val) {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows
        if (max_row != col) {
            for (int j = 0; j < n + p; j++) {
                std::swap(aug[col * (n + p) + j], aug[max_row * (n + p) + j]);
            }
        }

        float pivot = aug[col * (n + p) + col];
        if (fabsf(pivot) < 1e-12f) {
            continue; // skip singular column
        }

        // Eliminate below
        for (int row = col + 1; row < n; row++) {
            float factor = aug[row * (n + p) + col] / pivot;
            for (int j = col; j < n + p; j++) {
                aug[row * (n + p) + j] -= factor * aug[col * (n + p) + j];
            }
        }
    }

    // Back substitution
    for (int col = n - 1; col >= 0; col--) {
        float pivot = aug[col * (n + p) + col];
        if (fabsf(pivot) < 1e-12f) {
            for (int j = 0; j < p; j++) {
                x[col * p + j] = 0.0f;
            }
            continue;
        }
        for (int j = 0; j < p; j++) {
            float val = aug[col * (n + p) + n + j];
            for (int row = col + 1; row < n; row++) {
                val -= aug[col * (n + p) + row] * x[row * p + j];
            }
            x[col * p + j] = val / pivot;
        }
    }
}

// ============================================================================
// Compaction algorithm types and implementation
// ============================================================================

// Result of refitting a single head's values given pre-selected key indices
struct refit_head_result {
    std::vector<float> beta;  // [t] attention mass biases (NNLS)
    std::vector<float> C_v;   // [t * d_v] refitted values
};

// Refit a single head's values for a given set of selected key positions.
// Implements paper Steps 2-3 (algorithms.md §4-5) for one head.
//
// Given global key selection (shared across heads from Step 1), computes:
//   1. NNLS beta: attention mass biases to approximate full softmax partition (§4)
//   2. C_v: least-squares value refitting to minimize attention output error (§5)
//
// Design note: use_beta_for_cv=false (default) fits C_v with un-biased softmax
// because llama.cpp's state format has no mechanism to store per-token attention
// biases. At inference, the model computes softmax(q·K/√d_k)·V without beta,
// so C_v must be correct under that distribution. (See algorithms.md §12.1)
//
// NOT IMPLEMENTED (paper §4.1, algorithms.md §12.1): beta injection during
// inference. Would require a bias hook in ggml_flash_attn_ext or attention
// graph modification. If implemented, use_beta_for_cv=true becomes correct
// and C_v quality improves since beta compensates for missing attention mass.
//
// Parameters:
//   K_all:          [T, n_embd_k_gqa] keys for all heads concatenated
//   V_all:          [T, n_embd_v_gqa] values for all heads concatenated
//   T:              total number of KV positions
//   n_embd_k_gqa:   total K embedding size across all heads
//   n_embd_v_gqa:   total V embedding size across all heads
//   h:              head index
//   d_k, d_v:       per-head key/value dimensions
//   n_ref:          number of reference queries
//   ref_start:      starting index of reference queries in K
//   selected:       [t] selected position indices (sorted)
//   use_beta_for_cv: if false (default), compute C_v using un-biased softmax
//                    so values are correct at inference time (betas are not
//                    stored in the state). If true, include beta in the C_v
//                    fitting (only useful for in-memory evaluation).
//
KV_COMPACT_UNUSED static refit_head_result refit_head_values(
        const float * K_all, const float * V_all,
        int T, int n_embd_k_gqa, int n_embd_v_gqa,
        int h, int d_k, int d_v,
        int n_ref, int ref_start,
        const std::vector<int> & selected,
        bool use_beta_for_cv = false) {

    const int t = (int)selected.size();
    const float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);

    refit_head_result result;
    result.beta.resize(t);
    result.C_v.resize(t * d_v);

    // Compute attention scores: Q_ref_h @ K_h^T / sqrt(d_k)
    std::vector<float> scores(n_ref * T);
    for (int qi = 0; qi < n_ref; qi++) {
        const float * q_row = K_all + (ref_start + qi) * n_embd_k_gqa + h * d_k;
        for (int ki = 0; ki < T; ki++) {
            const float * k_row = K_all + ki * n_embd_k_gqa + h * d_k;
            float dot = 0.0f;
            for (int d = 0; d < d_k; d++) {
                dot += q_row[d] * k_row[d];
            }
            scores[qi * T + ki] = dot * inv_sqrt_dk;
        }
    }

    // Compute exp(scores) for NNLS mass matching
    std::vector<float> exp_scores(scores);
    std::vector<float> row_sums(n_ref);
    exp_rows_stable(exp_scores.data(), row_sums.data(), n_ref, T);

    // Softmax attention weights for value refitting target
    std::vector<float> attn_weights(scores);
    softmax_rows(attn_weights.data(), n_ref, T);

    // Step 1: NNLS for beta (attention mass matching)
    std::vector<float> M(n_ref * t);
    for (int qi = 0; qi < n_ref; qi++) {
        for (int j = 0; j < t; j++) {
            M[qi * t + j] = exp_scores[qi * T + selected[j]];
        }
    }

    std::vector<float> w(t);
    nnls_solve(M.data(), row_sums.data(), w.data(), n_ref, t);

    for (int j = 0; j < t; j++) {
        result.beta[j] = logf(std::max(1e-12f, w[j]));
    }

    // Step 2: Least-squares for C_v (value refitting)
    // X_ij = softmax over selected positions, with or without beta adjustment
    std::vector<float> X(n_ref * t);
    for (int qi = 0; qi < n_ref; qi++) {
        for (int j = 0; j < t; j++) {
            X[qi * t + j] = scores[qi * T + selected[j]];
            if (use_beta_for_cv) {
                X[qi * t + j] += result.beta[j];
            }
        }
    }
    softmax_rows(X.data(), n_ref, t);

    // Y = original attention output: attn_weights @ V_head  [n_ref, d_v]
    std::vector<float> Y(n_ref * d_v, 0.0f);
    for (int qi = 0; qi < n_ref; qi++) {
        for (int ki = 0; ki < T; ki++) {
            float w_ij = attn_weights[qi * T + ki];
            const float * v_row = V_all + ki * n_embd_v_gqa + h * d_v;
            for (int d = 0; d < d_v; d++) {
                Y[qi * d_v + d] += w_ij * v_row[d];
            }
        }
    }

    least_squares_solve(X.data(), Y.data(), result.C_v.data(), n_ref, t, d_v);

    return result;
}

struct compacted_head {
    std::vector<int>   selected_indices;  // which original tokens were selected
    std::vector<float> beta;              // attention mass biases [t]
    std::vector<float> C_v;               // refit values [t * d_v]
};

// Result of compacting all heads within a single layer
struct compacted_layer {
    std::vector<int>   selected_indices;  // [t] shared token selection across all heads
    int                n_head_kv;         // number of KV heads
    int                t;                 // compacted size
    int                d_k;               // key dimension per head
    int                d_v;               // value dimension per head

    // Per-head results: beta[h] is [t], C_v[h] is [t * d_v]
    std::vector<std::vector<float>> beta;  // [n_head_kv][t]
    std::vector<std::vector<float>> C_v;   // [n_head_kv][t * d_v]
};

// Compact a single KV head using the Highest Attention Keys method (paper Section 3.1)
//
// Full pipeline: key selection → NNLS bias fitting → least-squares value refitting.
// Key importance = max softmax attention weight across reference queries (§3.3).
// C_v is fitted with un-biased softmax to match inference behavior (§5.1).
//
// NOT IMPLEMENTED (paper ideas):
//   - OMP key selection (paper §5.2, algorithms.md §12.2): Orthogonal Matching Pursuit
//     selects keys by iterative greedy residual reduction. Higher quality but ~100x slower.
//   - Direct key optimization (paper §5.5, algorithms.md §12.4): allowing C_k to be
//     arbitrary vectors (not a subset of original K) could improve results but makes
//     optimization non-convex.
//
//   K:       [T, d_k] original keys for this head
//   V:       [T, d_v] original values for this head
//   Q_ref:   [n_q, d_k] reference queries
//   t:       target compacted size
//   d_k:     key dimension
//   d_v:     value dimension
//
// Returns compacted_head with selected indices, beta, and C_v.
// C_v is fitted with un-biased softmax (no beta) to match inference behavior,
// since betas cannot be stored in the llama.cpp state format.
KV_COMPACT_UNUSED static compacted_head compact_head_highest_attn(
        const float * K, const float * V, const float * Q_ref,
        int T, int n_q, int d_k, int d_v, int t) {

    compacted_head result;
    result.selected_indices.resize(t);
    result.beta.resize(t);
    result.C_v.resize(t * d_v);

    if (t >= T) {
        // No compaction needed
        for (int i = 0; i < T; i++) result.selected_indices[i] = i;
        std::fill(result.beta.begin(), result.beta.end(), 0.0f);
        memcpy(result.C_v.data(), V, T * d_v * sizeof(float));
        return result;
    }

    // Step 1: Compute attention scores Q_ref @ K^T / sqrt(d_k)
    const float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);
    std::vector<float> scores(n_q * T);
    mat_mul_ABt(Q_ref, K, scores.data(), n_q, T, d_k);
    for (int i = 0; i < n_q * T; i++) {
        scores[i] *= inv_sqrt_dk;
    }

    // Softmax attention weights for key scoring and value target
    std::vector<float> attn_weights(scores);
    softmax_rows(attn_weights.data(), n_q, T);

    // Score each key: max attention weight across queries
    std::vector<float> key_scores(T, 0.0f);
    for (int j = 0; j < T; j++) {
        float max_score = 0.0f;
        for (int i = 0; i < n_q; i++) {
            float w = attn_weights[i * T + j];
            if (w > max_score) max_score = w;
        }
        key_scores[j] = max_score;
    }

    // Select top-t keys by score
    std::vector<int> indices(T);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                      [&](int a, int b) { return key_scores[a] > key_scores[b]; });

    std::vector<int> selected(indices.begin(), indices.begin() + t);
    std::sort(selected.begin(), selected.end());
    result.selected_indices = selected;

    // Step 2: NNLS for beta (attention mass matching)
    std::vector<float> exp_scores(scores);
    std::vector<float> row_sums(n_q);
    exp_rows_stable(exp_scores.data(), row_sums.data(), n_q, T);

    std::vector<float> M(n_q * t);
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < t; j++) {
            M[i * t + j] = exp_scores[i * T + selected[j]];
        }
    }

    std::vector<float> w(t);
    nnls_solve(M.data(), row_sums.data(), w.data(), n_q, t);

    for (int j = 0; j < t; j++) {
        result.beta[j] = logf(std::max(1e-12f, w[j]));
    }

    // Step 3: Least squares for C_v (un-biased: no beta in softmax)
    // At inference, model computes softmax(q·K_selected/√d) · C_v without beta,
    // so C_v must be fitted against the un-biased distribution.
    std::vector<float> X(n_q * t);
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < t; j++) {
            X[i * t + j] = scores[i * T + selected[j]];
        }
    }
    softmax_rows(X.data(), n_q, t);

    std::vector<float> Y(n_q * d_v, 0.0f);
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < T; j++) {
            float w_ij = attn_weights[i * T + j];
            for (int d = 0; d < d_v; d++) {
                Y[i * d_v + d] += w_ij * V[j * d_v + d];
            }
        }
    }

    least_squares_solve(X.data(), Y.data(), result.C_v.data(), n_q, t, d_v);

    return result;
}

// Compact all KV heads within a single layer using shared key selection
//
// Implements the full pipeline from paper Section 3 at layer granularity:
//   1. Global key selection: max importance across all heads (§3.1, §3.3)
//   2. Per-head NNLS bias fitting + least-squares value refitting (§4-5)
//      via refit_head_values()
//
// Key positions are shared across heads within a layer because the llama.cpp
// state format requires consistent cell positions across all heads per layer.
//
// NOT IMPLEMENTED (paper ideas):
//   - Non-uniform per-head budgets (paper §6.2, algorithms.md §12.2): head sensitivity
//     varies — pre-computed sensitivity curves could allocate more budget to critical
//     heads and less to redundant ones. Currently all heads share the same t.
//   - OMP key selection (paper §5.2): see compact_head_highest_attn comments.
//
//   K_all:     [T, n_embd_k_gqa] all heads concatenated, row-major
//   V_all:     [T, n_embd_v_gqa] all heads concatenated, row-major
//   Q_ref_all: [n_q, n_embd_k_gqa] reference queries (all heads concatenated)
//   T:         number of tokens (cache positions)
//   n_q:       number of reference queries
//   n_head_kv: number of KV heads
//   d_k:       key dimension per head
//   d_v:       value dimension per head
//   t:         target compacted size
//
// Algorithm:
//   1. For each head, compute attention scores and per-key importance
//   2. Global key selection: max importance across heads for each position
//   3. Per-head NNLS (beta) and least-squares (C_v) on shared selection
//
KV_COMPACT_UNUSED static compacted_layer compact_layer_all_heads(
        const float * K_all, const float * V_all, const float * Q_ref_all,
        int T, int n_q, int n_head_kv, int d_k, int d_v, int t) {

    compacted_layer result;
    result.n_head_kv = n_head_kv;
    result.t = t;
    result.d_k = d_k;
    result.d_v = d_v;
    result.beta.resize(n_head_kv);
    result.C_v.resize(n_head_kv);

    const int n_embd_k_gqa = n_head_kv * d_k;
    const int n_embd_v_gqa = n_head_kv * d_v;

    if (t >= T) {
        // No compaction needed
        result.selected_indices.resize(T);
        for (int i = 0; i < T; i++) result.selected_indices[i] = i;
        for (int h = 0; h < n_head_kv; h++) {
            result.beta[h].assign(T, 0.0f);
            result.C_v[h].resize(T * d_v);
            for (int i = 0; i < T; i++) {
                memcpy(result.C_v[h].data() + i * d_v,
                       V_all + i * n_embd_v_gqa + h * d_v,
                       d_v * sizeof(float));
            }
        }
        return result;
    }

    // ---- Step 1: Global key selection via max importance across heads ----

    // Build combined K buffer: [K_all (T rows), Q_ref_all (n_q rows)]
    // so refit_head_values can use K_combined[T:] as reference queries.
    const int ref_start = T; // queries start right after original K data
    std::vector<float> K_combined((size_t)(T + n_q) * n_embd_k_gqa);
    memcpy(K_combined.data(), K_all, (size_t)T * n_embd_k_gqa * sizeof(float));
    memcpy(K_combined.data() + (size_t)T * n_embd_k_gqa, Q_ref_all,
           (size_t)n_q * n_embd_k_gqa * sizeof(float));

    // Compute per-head key importance scores, then take max across heads
    std::vector<float> global_scores(T, 0.0f);
    const float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);

    for (int h = 0; h < n_head_kv; h++) {
        std::vector<float> attn_weights(n_q * T);

        for (int qi = 0; qi < n_q; qi++) {
            const float * q_row = Q_ref_all + qi * n_embd_k_gqa + h * d_k;
            for (int ki = 0; ki < T; ki++) {
                const float * k_row = K_all + ki * n_embd_k_gqa + h * d_k;
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++) {
                    dot += q_row[d] * k_row[d];
                }
                attn_weights[qi * T + ki] = dot * inv_sqrt_dk;
            }
        }

        softmax_rows(attn_weights.data(), n_q, T);

        for (int j = 0; j < T; j++) {
            float max_w = 0.0f;
            for (int qi = 0; qi < n_q; qi++) {
                float w = attn_weights[qi * T + j];
                if (w > max_w) max_w = w;
            }
            if (max_w > global_scores[j]) {
                global_scores[j] = max_w;
            }
        }
    }

    // Select top-t positions globally
    std::vector<int> indices(T);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                      [&](int a, int b) { return global_scores[a] > global_scores[b]; });

    std::vector<int> selected(indices.begin(), indices.begin() + t);
    std::sort(selected.begin(), selected.end());
    result.selected_indices = selected;

    // ---- Steps 2-3: Per-head NNLS (beta) and least squares (C_v) ----
    // Delegate to refit_head_values using the combined K buffer with Q_ref
    // appended at position T. Pass T (not T_combined) so refit_head_values
    // only iterates keys 0..T-1 (original positions) while reading queries
    // from K_combined[T..T+n_q-1] (the appended Q_ref data).

    for (int h = 0; h < n_head_kv; h++) {
        auto rr = refit_head_values(
            K_combined.data(), V_all,
            T, n_embd_k_gqa, n_embd_v_gqa,
            h, d_k, d_v,
            n_q, ref_start,
            selected,
            false /* use_beta_for_cv */);

        result.beta[h] = std::move(rr.beta);
        result.C_v[h]  = std::move(rr.C_v);
    }

    return result;
}

// ============================================================================
// Bandwidth-aware ratio computation (pure math, no llama.cpp dependency)
// ============================================================================

// Compute suggested compact_ratio given model dimensions and memory budget.
//
// Parameters:
//   n_layer          - number of attention layers
//   n_embd           - embedding dimension
//   n_head           - number of attention heads
//   n_head_kv        - number of KV heads (may differ from n_head in GQA)
//   ctx_size         - context window per sequence
//   mem_budget_mb    - total memory budget for KV caches (MB)
//   n_parallel       - number of parallel sequences
//   bytes_per_elem_k - bytes per element for K cache (e.g. 2.0 for F16, 1.0625 for Q8_0)
//   bytes_per_elem_v - bytes per element for V cache
//
// Returns ratio in [0.05, 1.0], or -1.0 on invalid input.
KV_COMPACT_UNUSED static float compute_suggest_ratio(
    int n_layer, int n_embd, int n_head, int n_head_kv,
    int ctx_size, float mem_budget_mb, int n_parallel,
    float bytes_per_elem_k = 2.0f, float bytes_per_elem_v = 2.0f)
{
    if (n_head == 0 || n_head_kv == 0 || n_layer <= 0) return -1.0f;
    if (ctx_size <= 0 || mem_budget_mb <= 0.0f || n_parallel <= 0) return -1.0f;
    if (bytes_per_elem_k <= 0.0f || bytes_per_elem_v <= 0.0f) return -1.0f;

    const int d_head = n_embd / n_head;
    const int n_embd_kv_gqa = d_head * n_head_kv;

    // K + V per token per layer, respecting quantization type
    const float bytes_per_token_per_layer =
        n_embd_kv_gqa * bytes_per_elem_k + n_embd_kv_gqa * bytes_per_elem_v;

    const float kv_bytes_per_seq = bytes_per_token_per_layer * n_layer * ctx_size;
    const float total_kv_bytes = kv_bytes_per_seq * n_parallel;
    const float budget_bytes = mem_budget_mb * 1024.0f * 1024.0f;

    if (total_kv_bytes <= budget_bytes) return 1.0f;

    float ratio = budget_bytes / total_kv_bytes;
    if (ratio < 0.05f) ratio = 0.05f;

    return ratio;
}
