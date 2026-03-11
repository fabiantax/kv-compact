// KV Cache Compaction via Attention Matching - Math Utilities
//
// Pure CPU float32 linear algebra routines implementing the compaction
// algorithm from:
//
//   "Fast KV Compaction via Attention Matching" (Zweiger et al., 2026)
//   https://arxiv.org/abs/2602.16284
//
// The paper's 3-step compaction pipeline is implemented here:
//   Step 1 — Key Selection (Section 3.1): "Highest Attention Keys" scores
//            each key by max attention weight across reference queries,
//            then selects the top-t keys.
//   Step 2 — Mass Matching / Beta (Section 3.2, Eq. 4): Solves NNLS to find
//            per-key weights w such that the attention mass (partition
//            function) over selected keys matches the original:
//              sum_j exp(q@k_j/sqrt(d)) * w_j ≈ sum_j exp(q@K_j/sqrt(d))
//            Beta = log(w) are additive biases applied during attention.
//   Step 3 — Value Refitting / C_v (Section 3.3, Eq. 6): Solves least-squares
//            to find replacement values C_v such that the attention output
//            with compacted keys+beta closely matches the original:
//              softmax(q@C_k/sqrt(d) + beta) @ C_v ≈ softmax(q@K/sqrt(d)) @ V
//
// Extensions beyond the paper:
//   - Lawson-Hanson active-set NNLS solver (replaces projected gradient descent)
//   - Per-head sensitivity weighting for key selection (Section 4 ablation)
//   - Beta injection via K-vector modification (our approach for runtime use)
//   - Multi-head shared key selection with per-head beta/C_v (Section 3.4)
//
// Extracted into a header-only library for testability.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

// ============================================================================
// Linear algebra utilities (CPU-side, float32)
// ============================================================================

// Compute C = A * B^T  where A is (m x k), B is (n x k), result is (m x n)
static void mat_mul_ABt(const float * A, const float * B, float * C, int m, int n, int k) {
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
static void mat_mul_AtB(const float * A, const float * B, float * C, int m, int k, int n) {
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

// Softmax over rows: input (m x n), output (m x n), in-place safe
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

// Solve non-negative least squares via Lawson-Hanson active-set method:
//   min_{w >= 0} ||A*w - b||^2
// A is (m x n), b is (m), w is (n)
//
// Used in Step 2 (Section 3.2, Eq. 4) to solve for attention mass weights w_j
// such that exp(beta_j) = w_j and the partition function is preserved:
//   M * w ≈ m   where M_ij = exp(q_i @ k_{sel[j]} / sqrt(d))
//                      m_i  = sum_j exp(q_i @ K_j / sqrt(d))
//
// The Lawson-Hanson algorithm (Lawson & Hanson, 1974) converges in finitely
// many iterations. It maintains a passive set P (unconstrained variables) and
// a zero set Z (variables fixed at 0). Each outer iteration moves one variable
// from Z to P, inner iterations handle infeasible solutions by interpolating
// back to the boundary. This replaces the original projected gradient descent
// from our initial implementation, providing exact solutions for the small
// dense NNLS problems encountered here (t ~ 10-500 variables).
//
// Returns solution in w
static void nnls_solve(const float * A, const float * b, float * w, int m, int n, int max_iter = 200) {
    // Precompute A^T * A and A^T * b
    std::vector<float> AtA(n * n);
    std::vector<float> Atb(n);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < m; k++) {
                sum += A[k * n + i] * A[k * n + j];
            }
            AtA[i * n + j] = sum;
        }
    }

    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int k = 0; k < m; k++) {
            sum += A[k * n + i] * b[k];
        }
        Atb[i] = sum;
    }

    // Initialize: all variables in zero set
    std::vector<bool> in_P(n, false);  // passive set membership
    for (int i = 0; i < n; i++) w[i] = 0.0f;

    // Compute initial gradient: grad = AtA * w - Atb = -Atb (since w=0)
    std::vector<float> grad(n);
    for (int i = 0; i < n; i++) grad[i] = -Atb[i];

    // Temporary for sub-problem solution
    std::vector<float> s(n, 0.0f);

    for (int outer = 0; outer < max_iter; outer++) {
        // Find the index in Z with the most negative gradient (largest dual variable)
        int t_idx = -1;
        float max_dual = 0.0f;
        for (int i = 0; i < n; i++) {
            if (!in_P[i] && (-grad[i]) > max_dual) {
                max_dual = -grad[i];
                t_idx = i;
            }
        }

        // KKT check: if no zero-set variable has negative gradient, we're optimal
        if (t_idx < 0 || max_dual < 1e-10f) break;

        // Move t_idx from Z to P
        in_P[t_idx] = true;

        // Inner loop: solve unconstrained LS on P, fix infeasible variables
        for (int inner = 0; inner < max_iter; inner++) {
            // Count passive set size
            int p_count = 0;
            std::vector<int> p_indices;
            for (int i = 0; i < n; i++) {
                if (in_P[i]) p_indices.push_back(i);
            }
            p_count = (int) p_indices.size();

            // Solve the sub-problem: min ||A_P * s_P - b||^2
            // Using normal equations: (A_P^T A_P) s_P = A_P^T b
            // We already have AtA and Atb, so extract submatrices
            std::vector<float> sub_AtA(p_count * p_count);
            std::vector<float> sub_Atb(p_count);

            for (int i = 0; i < p_count; i++) {
                sub_Atb[i] = Atb[p_indices[i]];
                for (int j = 0; j < p_count; j++) {
                    sub_AtA[i * p_count + j] = AtA[p_indices[i] * n + p_indices[j]];
                }
                // Add small ridge for numerical stability
                sub_AtA[i * p_count + i] += 1e-10f;
            }

            // Solve via Gaussian elimination with partial pivoting
            std::vector<float> aug(p_count * (p_count + 1));
            for (int i = 0; i < p_count; i++) {
                for (int j = 0; j < p_count; j++) {
                    aug[i * (p_count + 1) + j] = sub_AtA[i * p_count + j];
                }
                aug[i * (p_count + 1) + p_count] = sub_Atb[i];
            }

            for (int col = 0; col < p_count; col++) {
                int max_row = col;
                float max_val = fabsf(aug[col * (p_count + 1) + col]);
                for (int row = col + 1; row < p_count; row++) {
                    float val = fabsf(aug[row * (p_count + 1) + col]);
                    if (val > max_val) { max_val = val; max_row = row; }
                }
                if (max_row != col) {
                    for (int j = 0; j < p_count + 1; j++)
                        std::swap(aug[col * (p_count + 1) + j],
                                  aug[max_row * (p_count + 1) + j]);
                }
                float pivot = aug[col * (p_count + 1) + col];
                if (fabsf(pivot) < 1e-12f) continue;
                for (int row = col + 1; row < p_count; row++) {
                    float factor = aug[row * (p_count + 1) + col] / pivot;
                    for (int j = col; j < p_count + 1; j++)
                        aug[row * (p_count + 1) + j] -= factor * aug[col * (p_count + 1) + j];
                }
            }

            std::vector<float> s_P(p_count, 0.0f);
            for (int col = p_count - 1; col >= 0; col--) {
                float pivot = aug[col * (p_count + 1) + col];
                if (fabsf(pivot) < 1e-12f) continue;
                float val = aug[col * (p_count + 1) + p_count];
                for (int row = col + 1; row < p_count; row++)
                    val -= aug[col * (p_count + 1) + row] * s_P[row];
                s_P[col] = val / pivot;
            }

            // Map s_P back to full s
            for (int i = 0; i < n; i++) s[i] = 0.0f;
            for (int i = 0; i < p_count; i++) s[p_indices[i]] = s_P[i];

            // Check feasibility: all s[P] >= 0
            bool feasible = true;
            for (int i = 0; i < p_count; i++) {
                if (s_P[i] <= 0.0f) { feasible = false; break; }
            }

            if (feasible) {
                // Accept the solution
                for (int i = 0; i < n; i++) w[i] = s[i];
                break;
            }

            // Find alpha: step size to boundary
            float alpha = 1.0f;
            for (int i = 0; i < p_count; i++) {
                int idx = p_indices[i];
                if (s[idx] <= 0.0f) {
                    float a = w[idx] / (w[idx] - s[idx] + 1e-30f);
                    if (a < alpha) alpha = a;
                }
            }

            // Interpolate: w = w + alpha * (s - w)
            for (int i = 0; i < n; i++) {
                w[i] += alpha * (s[i] - w[i]);
            }

            // Move variables at zero back to Z
            for (int i = 0; i < p_count; i++) {
                int idx = p_indices[i];
                if (fabsf(w[idx]) < 1e-12f) {
                    w[idx] = 0.0f;
                    in_P[idx] = false;
                }
            }
        }

        // Update gradient: grad = AtA * w - Atb
        for (int i = 0; i < n; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) sum += AtA[i * n + j] * w[j];
            grad[i] = sum - Atb[i];
        }
    }

    // Ensure no negative values (numerical safety)
    for (int i = 0; i < n; i++) {
        if (w[i] < 1e-12f) w[i] = 1e-12f;
    }
}

// Solve least squares: min ||A*x - b||^2 via normal equations
// A is (m x n), b is (m x p), x is (n x p)
//
// Used in Step 3 (Section 3.3, Eq. 6) to solve for refitted values C_v:
//   X * C_v = Y   where X_ij = softmax(q_i @ k_{sel[j]} / sqrt(d) + beta_j)
//                       Y_i  = softmax(q_i @ K / sqrt(d)) @ V
//
// Also used for beta direction computation (see compute_beta_direction).
// Solves via regularized normal equations: x = (A^T A + ridge*I)^{-1} A^T b
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
// Per-head sensitivity and budget allocation
// ============================================================================

// Compute per-head sensitivity (concentration ratio) from attention weights.
//
// Related to Section 4 (Ablation Study) of Zweiger et al., which shows that
// non-uniform budget allocation across heads improves quality. Heads that
// concentrate attention on fewer positions are more sensitive to key loss.
//
// For each head, the "importance" of position j is max_q(attn_weights[q,j]).
// Sensitivity = max(importance) / mean(importance).
// A high ratio means the head concentrates on few positions → more sensitive
// to losing those positions → should have more influence on key selection.
//
// attn_weights: [n_q × T] softmax attention weights for one head
// n_q:          number of reference queries
// T:            number of positions
//
// Returns sensitivity scalar (>= 1.0)
static float compute_head_sensitivity(const float * attn_weights, int n_q, int T) {
    // Per-position importance: max attention weight across queries
    float sum_imp = 0.0f;
    float max_imp = 0.0f;
    for (int j = 0; j < T; j++) {
        float max_w = 0.0f;
        for (int qi = 0; qi < n_q; qi++) {
            float w = attn_weights[qi * T + j];
            if (w > max_w) max_w = w;
        }
        sum_imp += max_w;
        if (max_w > max_imp) max_imp = max_w;
    }
    float mean_imp = sum_imp / (float) T;
    return max_imp / (mean_imp + 1e-12f);
}

// Compute sensitivity-weighted global importance across multiple heads.
//
// For each position j, the weighted importance is the sum across heads of
// sensitivity[h] * per_head_importance[h][j]. This gives more influence to
// heads that concentrate attention on fewer positions.
//
// per_head_importance: [n_heads][T] max-attention-weight per position per head
// sensitivities:       [n_heads] sensitivity scalars
// T:                   number of positions
// n_heads:             number of heads
// out:                 [T] output weighted importance (accumulated, not zeroed)
static void accumulate_weighted_importance(
        const std::vector<std::vector<float>> & per_head_importance,
        const std::vector<float> & sensitivities,
        int T, int n_heads, float * out) {
    for (int h = 0; h < n_heads; h++) {
        const float s = sensitivities[h];
        const float * imp = per_head_importance[h].data();
        for (int j = 0; j < T; j++) {
            out[j] += s * imp[j];
        }
    }
}

// ============================================================================
// Beta injection via K-vector modification
// ============================================================================

// Compute the optimal direction vector for encoding beta into K vectors.
//
// This is our approach for runtime beta injection (US-1). The paper applies
// beta as a separate additive bias in the attention score computation
// (Section 3.2, Eq. 3): score_ij = q_i @ k_j / sqrt(d) + beta_j
//
// Since modifying the attention kernel is invasive, we instead encode beta
// directly into the K vectors by finding a direction v such that q @ v ≈ 1
// for all reference queries, then setting k_j' = k_j + beta_j * sqrt(d_k) * v.
// This gives: q @ k_j' / sqrt(d_k) = q @ k_j / sqrt(d_k) + beta_j * (q @ v)
// For queries where q @ v ≈ 1, this exactly reproduces the paper's beta bias
// without any attention kernel modifications.
static void compute_beta_direction(const float * Q_ref, int n_q, int d_k,
                                   float * direction, float ridge = 1e-6f) {
    std::vector<float> ones(n_q, 1.0f);
    least_squares_solve(Q_ref, ones.data(), direction, n_q, d_k, 1, ridge);
}

// Apply beta biases to K vectors by modifying keys in-place.
//
// For each selected key j: k_j[head_offset..+d_k] += beta_j * sqrt(d_k) * direction
//
// K_all:            [cell_count × n_embd_k_gqa] — modified in-place
// n_embd_k_gqa:     total K embedding size across all heads
// selected_indices:  [t] which rows to modify
// beta:             [t] per-selected-position bias for this head
// direction:        [d_k] beta encoding direction for this head
// d_k:              key dimension per head
// head_offset:      starting column for this head's dimensions in K rows
static void apply_beta_to_keys(float * K_all, int n_embd_k_gqa,
                               const int * selected_indices, int t,
                               const float * beta, const float * direction,
                               int d_k, int head_offset) {
    const float scale = sqrtf((float) d_k);
    for (int j = 0; j < t; j++) {
        float * k_row = K_all + selected_indices[j] * n_embd_k_gqa + head_offset;
        const float b_scaled = beta[j] * scale;
        for (int d = 0; d < d_k; d++) {
            k_row[d] += b_scaled * direction[d];
        }
    }
}

// ============================================================================
// Compaction algorithm types and implementation
// ============================================================================

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

// Compact a single KV head using the Highest Attention Keys method.
//
// Implements the full 3-step pipeline from Zweiger et al., Section 3:
//   Step 1 (Section 3.1): Score keys by max attention weight across Q_ref,
//           select top-t. This is the "Highest Attention Keys" variant
//           (vs. OMP in Section 3.1.2).
//   Step 2 (Section 3.2, Eq. 4): NNLS mass matching → beta biases
//   Step 3 (Section 3.3, Eq. 6): Least-squares value refitting → C_v
//
//   K:       [T, d_k] original keys for this head
//   V:       [T, d_v] original values for this head
//   Q_ref:   [n_q, d_k] reference queries (Section 2.2 — can be from
//            repeat-prefill or K-vector proxy)
//   t:       target compacted size
//   d_k:     key dimension
//   d_v:     value dimension
//
// Returns compacted_head with selected indices, beta, and C_v
static compacted_head compact_head_highest_attn(
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

    // ---- Step 1: Key Selection (Section 3.1 — "Highest Attention Keys") ----
    // Compute attention scores: S_ij = q_i @ k_j / sqrt(d_k)  [Eq. 1]
    const float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);
    std::vector<float> scores(n_q * T);
    mat_mul_ABt(Q_ref, K, scores.data(), n_q, T, d_k);
    for (int i = 0; i < n_q * T; i++) {
        scores[i] *= inv_sqrt_dk;
    }

    // Compute exp(scores) with max-shift for mass computation
    std::vector<float> exp_scores(scores); // copy
    std::vector<float> row_sums(n_q);
    exp_rows_stable(exp_scores.data(), row_sums.data(), n_q, T);

    // Compute softmax attention weights for key scoring
    std::vector<float> attn_weights(scores);
    softmax_rows(attn_weights.data(), n_q, T);

    // Score each key by max attention weight across queries (Section 3.1):
    //   importance(j) = max_i softmax(S)_{i,j}
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

    // Sort selected indices for cache locality
    std::vector<int> selected(indices.begin(), indices.begin() + t);
    std::sort(selected.begin(), selected.end());
    result.selected_indices = selected;

    // ---- Step 2: Mass Matching / Beta (Section 3.2, Eq. 4) ----
    // Solve NNLS: M * w ≈ m  where w_j = exp(beta_j)
    //   M_ij = exp(q_i @ k_{sel[j]} / sqrt(d))  — design matrix
    //   m_i  = sum_j exp(q_i @ K_j / sqrt(d))    — target partition function
    // This preserves the total attention mass (partition function) so that
    // softmax over selected keys + beta matches the original distribution.

    std::vector<float> M(n_q * t);
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < t; j++) {
            M[i * t + j] = exp_scores[i * T + selected[j]];
        }
    }

    // Target mass: already computed as row_sums
    std::vector<float> w(t);
    nnls_solve(M.data(), row_sums.data(), w.data(), n_q, t);

    // beta = log(w)
    for (int j = 0; j < t; j++) {
        result.beta[j] = logf(std::max(1e-12f, w[j]));
    }

    // ---- Step 3: Value Refitting / C_v (Section 3.3, Eq. 6) ----
    // Solve least squares: X * C_v = Y
    //   X_ij = softmax(q_i @ k_{sel[j]} / sqrt(d) + beta_j)  — compacted attn weights
    //   Y_i  = softmax(q_i @ K / sqrt(d)) @ V                — original attn output
    // This finds replacement values C_v that minimize the output error
    // between the compacted and original attention computation.

    // Compute X: attention weights with compacted keys + bias
    std::vector<float> X(n_q * t);
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < t; j++) {
            // scores[] already contains q@k/sqrt(d), so just add beta
            X[i * t + j] = scores[i * T + selected[j]] + result.beta[j];
        }
    }
    softmax_rows(X.data(), n_q, t);

    // Compute Y: original attention output = attn_weights @ V  [n_q, d_v]
    std::vector<float> Y(n_q * d_v, 0.0f);
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < T; j++) {
            float w_ij = attn_weights[i * T + j];
            for (int d = 0; d < d_v; d++) {
                Y[i * d_v + d] += w_ij * V[j * d_v + d];
            }
        }
    }

    // Solve: X * C_v = Y  =>  C_v = (X^T X)^{-1} X^T Y
    least_squares_solve(X.data(), Y.data(), result.C_v.data(), n_q, t, d_v);

    return result;
}

// Compact all KV heads within a single layer using shared key selection.
//
// Extends the per-head algorithm (Section 3) to multi-head with shared
// position selection (Section 3.4). All heads share the same selected
// token positions but have independent beta and C_v values. This is
// required because llama.cpp stores KV as contiguous rows across heads,
// so we can only remove entire token positions (not per-head subsets).
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
//      (each position's score = max over all heads of its importance)
//   3. Per-head NNLS (beta) and least-squares (C_v) on shared selection
//
static compacted_layer compact_layer_all_heads(
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

    // ---- Step 1: Global key selection (Section 3.1 + Section 3.4) ----
    // Compute per-head importance, then take max across heads per position

    // Compute per-head key importance scores, then take max across heads
    std::vector<float> global_scores(T, 0.0f);

    // Per-head temporary data for reuse in steps 2-3
    struct head_data {
        std::vector<float> scores;      // [n_q, T] scaled attention logits
        std::vector<float> exp_scores;  // [n_q, T] exp with max-shift
        std::vector<float> row_sums;    // [n_q] sum of exp per query
        std::vector<float> attn_weights;// [n_q, T] softmax attention
    };
    std::vector<head_data> hdata(n_head_kv);

    const float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);

    for (int h = 0; h < n_head_kv; h++) {
        auto & hd = hdata[h];
        hd.scores.resize(n_q * T);
        hd.exp_scores.resize(n_q * T);
        hd.row_sums.resize(n_q);
        hd.attn_weights.resize(n_q * T);

        // Extract per-head K and Q_ref slices
        // K_head[i] = K_all[i * n_embd_k_gqa + h * d_k ... + (h+1)*d_k]
        // Instead of extracting, compute Q_ref_h @ K_h^T directly

        // Compute scores: Q_ref_h @ K_h^T / sqrt(d_k)
        for (int qi = 0; qi < n_q; qi++) {
            const float * q_row = Q_ref_all + qi * n_embd_k_gqa + h * d_k;
            for (int ki = 0; ki < T; ki++) {
                const float * k_row = K_all + ki * n_embd_k_gqa + h * d_k;
                float dot = 0.0f;
                for (int d = 0; d < d_k; d++) {
                    dot += q_row[d] * k_row[d];
                }
                hd.scores[qi * T + ki] = dot * inv_sqrt_dk;
            }
        }

        // Compute exp(scores) for mass computation
        memcpy(hd.exp_scores.data(), hd.scores.data(), n_q * T * sizeof(float));
        exp_rows_stable(hd.exp_scores.data(), hd.row_sums.data(), n_q, T);

        // Compute softmax for key scoring
        memcpy(hd.attn_weights.data(), hd.scores.data(), n_q * T * sizeof(float));
        softmax_rows(hd.attn_weights.data(), n_q, T);

        // Per-key max attention weight across queries
        for (int j = 0; j < T; j++) {
            float max_w = 0.0f;
            for (int qi = 0; qi < n_q; qi++) {
                float w = hd.attn_weights[qi * T + j];
                if (w > max_w) max_w = w;
            }
            // Global score = max across heads
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

    // ---- Steps 2-3: Per-head NNLS + LS (Section 3.2-3.3) ----
    // Each head gets its own beta and C_v on the shared key selection

    for (int h = 0; h < n_head_kv; h++) {
        const auto & hd = hdata[h];

        result.beta[h].resize(t);
        result.C_v[h].resize(t * d_v);

        // Step 2: NNLS for beta (Section 3.2, Eq. 4)
        // M_ij = exp(q_i @ k_{sel[j]} / sqrt(d)), target = partition function
        std::vector<float> M(n_q * t);
        for (int qi = 0; qi < n_q; qi++) {
            for (int j = 0; j < t; j++) {
                M[qi * t + j] = hd.exp_scores[qi * T + selected[j]];
            }
        }

        std::vector<float> w(t);
        nnls_solve(M.data(), hd.row_sums.data(), w.data(), n_q, t);

        for (int j = 0; j < t; j++) {
            result.beta[h][j] = logf(std::max(1e-12f, w[j]));
        }

        // Step 3: Least squares for C_v (Section 3.3, Eq. 6)
        // X_ij = softmax(score[qi, sel[j]] + beta[j]), Y = original attn output
        std::vector<float> X(n_q * t);
        for (int qi = 0; qi < n_q; qi++) {
            for (int j = 0; j < t; j++) {
                X[qi * t + j] = hd.scores[qi * T + selected[j]] + result.beta[h][j];
            }
        }
        softmax_rows(X.data(), n_q, t);

        // Y = original attention output: attn_weights @ V_head  [n_q, d_v]
        std::vector<float> Y(n_q * d_v, 0.0f);
        for (int qi = 0; qi < n_q; qi++) {
            for (int ki = 0; ki < T; ki++) {
                float w_ij = hd.attn_weights[qi * T + ki];
                const float * v_row = V_all + ki * n_embd_v_gqa + h * d_v;
                for (int d = 0; d < d_v; d++) {
                    Y[qi * d_v + d] += w_ij * v_row[d];
                }
            }
        }

        least_squares_solve(X.data(), Y.data(), result.C_v[h].data(), n_q, t, d_v);
    }

    return result;
}
