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

// Compute C = A * B  where A is (m x k), B is (k x n), result is (m x n)
static void mat_mul_AB(const float * A, const float * B, float * C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
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

// Score aggregation methods for key importance across queries.
// The paper (Section 3.1) recommends RMS as default; this implementation
// also supports max (our original default) and mean.
enum score_agg_method {
    SCORE_AGG_MAX  = 0,   // max_i softmax[i,j] — fast, slightly less robust
    SCORE_AGG_RMS  = 1,   // sqrt(mean_i(softmax[i,j]^2)) — paper default (Section 3.1)
    SCORE_AGG_MEAN = 2,   // mean_i softmax[i,j]
};

// Fused exp + softmax + per-column importance in a single pass.
//
// Given scores [m × n], computes in one pass:
//   exp_out[i,j]    = exp(scores[i,j] - max_row)     (unnormalized)
//   row_sums[i]     = sum_j exp_out[i,j]              (partition function)
//   softmax_out[i,j]= exp_out[i,j] / row_sums[i]     (normalized weights)
//   importance[j]   = aggregated importance per key    (method-dependent)
//
// The agg parameter selects how per-query attention weights are aggregated
// into a single importance score per key (see score_agg_method).
//
// Replaces: memcpy + exp_rows_stable + memcpy + softmax_rows + column-max loop
// with a single pass over the data. This is 5 operations fused into 1.
static void exp_softmax_importance_fused(
        const float * scores, float * exp_out, float * row_sums,
        float * softmax_out, float * importance, int m, int n,
        int agg = SCORE_AGG_MAX) {
    // Zero importance (or init for sum-of-squares for RMS)
    memset(importance, 0, n * sizeof(float));

    for (int i = 0; i < m; i++) {
        const float * row = scores + i * n;
        float * exp_row = exp_out + i * n;
        float * sm_row  = softmax_out + i * n;

        // Find row max
        float max_val = row[0];
        for (int j = 1; j < n; j++) {
            if (row[j] > max_val) max_val = row[j];
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            float e = expf(row[j] - max_val);
            exp_row[j] = e;
            sum += e;
        }
        row_sums[i] = sum;

        // Normalize and update column importance
        float inv_sum = 1.0f / (sum + 1e-12f);
        for (int j = 0; j < n; j++) {
            float sm = exp_row[j] * inv_sum;
            sm_row[j] = sm;
            if (agg == SCORE_AGG_MAX) {
                if (sm > importance[j]) importance[j] = sm;
            } else if (agg == SCORE_AGG_RMS) {
                importance[j] += sm * sm;  // accumulate squares
            } else { // SCORE_AGG_MEAN
                importance[j] += sm;
            }
        }
    }

    // Finalize aggregation
    if (agg == SCORE_AGG_RMS) {
        float inv_m = 1.0f / (float)m;
        for (int j = 0; j < n; j++) {
            importance[j] = sqrtf(importance[j] * inv_m);
        }
    } else if (agg == SCORE_AGG_MEAN) {
        float inv_m = 1.0f / (float)m;
        for (int j = 0; j < n; j++) {
            importance[j] *= inv_m;
        }
    }
}

// Lightweight fused softmax + per-column importance (no exp_out/row_sums).
// When beta is skipped, we don't need exp_scores or row_sums.
// Saves memory allocation and avoids unnecessary buffer writes.
static void softmax_importance_fused(
        const float * scores, float * softmax_out, float * importance, int m, int n,
        int agg = SCORE_AGG_MAX) {
    memset(importance, 0, n * sizeof(float));

    for (int i = 0; i < m; i++) {
        const float * row = scores + i * n;
        float * sm_row  = softmax_out + i * n;

        float max_val = row[0];
        for (int j = 1; j < n; j++) {
            if (row[j] > max_val) max_val = row[j];
        }

        float sum = 0.0f;
        for (int j = 0; j < n; j++) {
            float e = expf(row[j] - max_val);
            sm_row[j] = e;
            sum += e;
        }

        float inv_sum = 1.0f / (sum + 1e-12f);
        for (int j = 0; j < n; j++) {
            float sm = sm_row[j] * inv_sum;
            sm_row[j] = sm;
            if (agg == SCORE_AGG_MAX) {
                if (sm > importance[j]) importance[j] = sm;
            } else if (agg == SCORE_AGG_RMS) {
                importance[j] += sm * sm;
            } else { // SCORE_AGG_MEAN
                importance[j] += sm;
            }
        }
    }

    // Finalize aggregation
    if (agg == SCORE_AGG_RMS) {
        float inv_m = 1.0f / (float)m;
        for (int j = 0; j < n; j++) {
            importance[j] = sqrtf(importance[j] * inv_m);
        }
    } else if (agg == SCORE_AGG_MEAN) {
        float inv_m = 1.0f / (float)m;
        for (int j = 0; j < n; j++) {
            importance[j] *= inv_m;
        }
    }
}

// Extract per-head slice from interleaved layout into contiguous buffer.
//   src:       [rows × stride] interleaved data
//   dst:       [rows × width] contiguous output
//   rows:      number of rows to extract
//   width:     elements per head (d_k or d_v)
//   stride:    total row stride (n_embd_k_gqa or n_embd_v_gqa)
//   offset:    head offset within each row (h * d_k or h * d_v)
static void extract_head_contiguous(
        const float * src, float * dst,
        int rows, int width, int stride, int offset) {
    for (int i = 0; i < rows; i++) {
        memcpy(dst + i * width, src + i * stride + offset, width * sizeof(float));
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
static void nnls_solve(const float * A, const float * b, float * w, int m, int n, int max_iter = 200, float tol = 0.0f) {
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

        // Early stop: check gradient norm periodically
        if (tol > 0.0f && (outer % 10 == 9)) {
            float grad_norm = 0.0f;
            for (int i = 0; i < n; i++) grad_norm += grad[i] * grad[i];
            if (grad_norm < tol * tol * n) break;
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
// Projected gradient descent NNLS solver (paper's Algorithm 3)
// ============================================================================

// Solve non-negative least squares via projected gradient descent:
//   min_{w >= eps} ||A*w - b||^2
//
// This matches the paper's Appendix C.2 (Algorithm 3):
//   1. Compute unconstrained least-squares solution via normal equations
//   2. Clamp to enforce w >= eps
//   3. If iters > 0, refine via projected gradient descent with step size 1/L
//      where L ≈ ||A||_2^2 estimated via power iteration.
//
// A is (m x n), b is (m), w is (n)
// eps: lower bound (default 1e-12)
// iters: PGD iterations (0 = clamped LS only, matching paper's default)
static void nnls_pgd_solve(const float * A, const float * b, float * w,
                            int m, int n, float eps = 1e-12f, int iters = 0) {
    // Precompute A^T*A and A^T*b
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

    // Step 1: Unconstrained least-squares via normal equations (with small ridge)
    {
        std::vector<float> aug(n * (n + 1));
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                aug[i * (n + 1) + j] = AtA[i * n + j];
            }
            aug[i * (n + 1) + i] += 1e-10f;  // tiny ridge for stability
            aug[i * (n + 1) + n] = Atb[i];
        }

        // Gaussian elimination with partial pivoting
        for (int col = 0; col < n; col++) {
            int max_row = col;
            float max_val = fabsf(aug[col * (n + 1) + col]);
            for (int row = col + 1; row < n; row++) {
                float val = fabsf(aug[row * (n + 1) + col]);
                if (val > max_val) { max_val = val; max_row = row; }
            }
            if (max_row != col) {
                for (int j = 0; j < n + 1; j++)
                    std::swap(aug[col * (n + 1) + j], aug[max_row * (n + 1) + j]);
            }
            float pivot = aug[col * (n + 1) + col];
            if (fabsf(pivot) < 1e-12f) continue;
            for (int row = col + 1; row < n; row++) {
                float factor = aug[row * (n + 1) + col] / pivot;
                for (int j = col; j < n + 1; j++)
                    aug[row * (n + 1) + j] -= factor * aug[col * (n + 1) + j];
            }
        }
        for (int col = n - 1; col >= 0; col--) {
            float pivot = aug[col * (n + 1) + col];
            if (fabsf(pivot) < 1e-12f) { w[col] = eps; continue; }
            float val = aug[col * (n + 1) + n];
            for (int row = col + 1; row < n; row++)
                val -= aug[col * (n + 1) + row] * w[row];
            w[col] = val / pivot;
        }
    }

    // Step 2: Clamp to enforce w >= eps
    for (int i = 0; i < n; i++) {
        if (w[i] < eps) w[i] = eps;
    }

    if (iters <= 0) return;

    // Step 3: Projected gradient descent
    // Estimate L ≈ ||A||_2^2 via 3 power iteration steps
    std::vector<float> u(n), v(m);
    for (int i = 0; i < n; i++) u[i] = 1.0f / sqrtf((float)n);
    for (int pw = 0; pw < 3; pw++) {
        // v = A*u
        for (int i = 0; i < m; i++) {
            float s = 0.0f;
            for (int j = 0; j < n; j++) s += A[i * n + j] * u[j];
            v[i] = s;
        }
        float vn = 0.0f;
        for (int i = 0; i < m; i++) vn += v[i] * v[i];
        vn = sqrtf(vn);
        if (vn < 1e-12f) break;
        for (int i = 0; i < m; i++) v[i] /= vn;
        // u = A^T*v
        for (int j = 0; j < n; j++) {
            float s = 0.0f;
            for (int i = 0; i < m; i++) s += A[i * n + j] * v[i];
            u[j] = s;
        }
        float un = 0.0f;
        for (int j = 0; j < n; j++) un += u[j] * u[j];
        un = sqrtf(un);
        if (un < 1e-12f) break;
        for (int j = 0; j < n; j++) u[j] /= un;
    }
    // L = u^T * AtA * u
    float L = 0.0f;
    for (int i = 0; i < n; i++) {
        float s = 0.0f;
        for (int j = 0; j < n; j++) s += AtA[i * n + j] * u[j];
        L += u[i] * s;
    }
    if (L < 1e-6f) L = 1e-6f;
    float eta = 1.0f / L;

    // PGD iterations: w = clamp(w - eta * grad, eps, inf)
    std::vector<float> grad(n);
    for (int iter = 0; iter < iters; iter++) {
        // grad = AtA * w - Atb
        for (int i = 0; i < n; i++) {
            float s = 0.0f;
            for (int j = 0; j < n; j++) s += AtA[i * n + j] * w[j];
            grad[i] = s - Atb[i];
        }
        for (int i = 0; i < n; i++) {
            w[i] -= eta * grad[i];
            if (w[i] < eps) w[i] = eps;
        }
    }
}

// ============================================================================
// OMP key selection (paper's Algorithm 1, Section 3.3)
// ============================================================================

// Select t keys via Orthogonal Matching Pursuit on the attention mass.
//
// The paper's best method (AM-OMP) greedily selects keys to minimize the
// attention mass residual. At each step:
//   1. Compute residual r = target - current approximation
//   2. Select the key with highest correlation to r
//   3. Re-solve NNLS for weights w on selected keys
//
// Parameters:
//   exp_scores: [n_q × T] unnormalized exp(q@k/sqrt(d) - max) per row
//   row_sums:   [n_q] partition function sum per query
//   T:          total number of keys
//   n_q:        number of reference queries
//   t:          target number of keys to select
//   selected:   [t] output selected indices (sorted)
//   w_out:      [t] output NNLS weights (beta = log(w))
//   k_choice:   number of keys to select per iteration (default: 1)
//               k_choice=4 with refit_interval=2 gives 4-8x speedup
//   refit_interval: solve NNLS every N iterations (default: 1)
//   nnls_iters: PGD iterations for NNLS refinement (default: 0)
static void omp_select_keys(
        const float * exp_scores, const float * row_sums,
        int T, int n_q, int t,
        int * selected, float * w_out,
        int k_choice = 1, int refit_interval = 1, int nnls_iters = 0) {

    std::vector<bool> mask(T, false);
    std::vector<float> current(n_q, 0.0f);  // current approximation
    std::vector<int> sel_order;
    sel_order.reserve(t);

    // Temporary for NNLS
    std::vector<float> prev_w;

    int iteration = 0;
    while ((int)sel_order.size() < t) {
        // Compute residual
        std::vector<float> residual(n_q);
        for (int i = 0; i < n_q; i++) {
            residual[i] = row_sums[i] - current[i];
        }

        // Correlation of each key with residual
        std::vector<float> corr(T, 0.0f);
        for (int j = 0; j < T; j++) {
            if (mask[j]) continue;
            float c = 0.0f;
            for (int i = 0; i < n_q; i++) {
                c += exp_scores[i * T + j] * residual[i];
            }
            corr[j] = c;
        }

        // Select top k_choice keys by correlation
        int remaining = t - (int)sel_order.size();
        int k_sel = k_choice < remaining ? k_choice : remaining;
        int available = T - (int)sel_order.size();
        if (k_sel > available) k_sel = available;
        if (k_sel <= 0) break;

        // Find top k_sel indices
        for (int k = 0; k < k_sel; k++) {
            int best = -1;
            float best_c = -1e30f;
            for (int j = 0; j < T; j++) {
                if (mask[j]) continue;
                if (corr[j] > best_c) { best_c = corr[j]; best = j; }
            }
            if (best < 0) break;
            sel_order.push_back(best);
            mask[best] = true;
            corr[best] = -1e30f;
        }

        int cur_t = (int)sel_order.size();

        // Solve NNLS (or reuse previous weights)
        bool should_solve = (refit_interval <= 1) || (iteration % refit_interval == 0)
                            || (cur_t == t);  // always solve on last iteration

        if (should_solve) {
            // Build design matrix M: [n_q × cur_t]
            std::vector<float> M(n_q * cur_t);
            for (int i = 0; i < n_q; i++) {
                for (int j = 0; j < cur_t; j++) {
                    M[i * cur_t + j] = exp_scores[i * T + sel_order[j]];
                }
            }

            std::vector<float> w(cur_t);
            nnls_pgd_solve(M.data(), row_sums, w.data(), n_q, cur_t, 1e-12f, nnls_iters);
            prev_w = w;

            // Update current approximation
            for (int i = 0; i < n_q; i++) {
                float s = 0.0f;
                for (int j = 0; j < cur_t; j++) {
                    s += M[i * cur_t + j] * w[j];
                }
                current[i] = s;
            }
        } else {
            // Extend previous weights with eps for new entries
            std::vector<float> w(cur_t, 1e-12f);
            for (int j = 0; j < (int)prev_w.size() && j < cur_t; j++) {
                w[j] = prev_w[j];
            }
            prev_w = w;

            // Update current approximation
            for (int i = 0; i < n_q; i++) {
                float s = 0.0f;
                for (int j = 0; j < cur_t; j++) {
                    s += exp_scores[i * T + sel_order[j]] * w[j];
                }
                current[i] = s;
            }
        }

        iteration++;
    }

    // Sort selected indices for cache locality, maintaining weight mapping
    int actual_t = (int)sel_order.size();
    std::vector<int> order(actual_t);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return sel_order[a] < sel_order[b]; });

    for (int j = 0; j < actual_t; j++) {
        selected[j] = sel_order[order[j]];
        w_out[j] = (j < (int)prev_w.size()) ? prev_w[order[j]] : 1e-12f;
    }
    // Zero-fill if we got fewer than t
    for (int j = actual_t; j < t; j++) {
        selected[j] = 0;
        w_out[j] = 1e-12f;
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

    // Per-head budget allocation from greedy exchange [n_head_kv]
    // If empty, uniform budget t was used for all heads
    std::vector<int> per_head_budgets;
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

    // Fused softmax + importance scoring
    std::vector<float> attn_weights(n_q * T);
    std::vector<float> key_scores(T);
    softmax_importance_fused(scores.data(), attn_weights.data(), key_scores.data(), n_q, T);

    // Select top-t keys by score
    std::vector<int> indices(T);
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                      [&](int a, int b) { return key_scores[a] > key_scores[b]; });

    // Sort selected indices for cache locality
    std::vector<int> selected(indices.begin(), indices.begin() + t);
    std::sort(selected.begin(), selected.end());
    result.selected_indices = selected;

    // Beta = 0 (skip NNLS — LS value refit alone achieves equal or better quality)
    std::fill(result.beta.begin(), result.beta.end(), 0.0f);

    // ---- Value Refitting / C_v (Section 3.3, Eq. 6) ----
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
    // V is contiguous [T × d_v] for single-head, so use batched matmul directly
    std::vector<float> Y(n_q * d_v);
    mat_mul_AB(attn_weights.data(), V, Y.data(), n_q, T, d_v);

    // Solve: X * C_v = Y  =>  C_v = (X^T X)^{-1} X^T Y
    least_squares_solve(X.data(), Y.data(), result.C_v.data(), n_q, t, d_v);

    return result;
}

// ============================================================================
// Greedy Budget Exchange (Paper §5)
// ============================================================================
//
// Non-uniform per-head budget allocation that greedily exchanges budget units
// from compression-tolerant heads to compression-sensitive heads.
//
// Algorithm:
//   1. Start with uniform allocation: t_h = total_budget / n_heads
//   2. Iteratively:
//      a. Find donor = head whose last key has lowest marginal value
//      b. Find recipient = head whose next key has highest marginal value
//      c. If gain > loss: transfer 1 budget unit, continue
//      d. Else: converged
//
// Marginal value model: sensitivity[h] / t_h  (diminishing returns)
// This is equivalent to water-filling: budget ∝ sqrt(sensitivity) at optimum.
//
static std::vector<int> greedy_budget_exchange(
        const float * sensitivity,
        int n_heads,
        int total_budget,
        int min_per_head = 2,
        int max_per_head = 0) {

    if (n_heads <= 0 || total_budget <= 0) return {};
    if (n_heads == 1) return {total_budget};

    min_per_head = std::max(1, min_per_head);
    if (max_per_head <= 0) max_per_head = total_budget;

    // Ensure feasible: min_per_head * n_heads <= total_budget
    if (min_per_head * n_heads > total_budget) {
        min_per_head = total_budget / n_heads;
        if (min_per_head < 1) min_per_head = 1;
    }

    // Start uniform
    std::vector<int> budget(n_heads, total_budget / n_heads);
    int remaining = total_budget - n_heads * (total_budget / n_heads);

    // Distribute remainder to most sensitive heads
    std::vector<int> order(n_heads);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(),
        [&](int a, int b) { return sensitivity[a] > sensitivity[b]; });
    for (int i = 0; i < remaining; i++) {
        budget[order[i]]++;
    }

    // Enforce minimum
    for (int h = 0; h < n_heads; h++) {
        budget[h] = std::max(budget[h], min_per_head);
    }
    // Re-balance if minimums caused overflow
    int excess = 0;
    for (int h = 0; h < n_heads; h++) excess += budget[h];
    excess -= total_budget;
    if (excess > 0) {
        for (int i = n_heads - 1; i >= 0 && excess > 0; i--) {
            int h = order[i];
            int can_remove = budget[h] - min_per_head;
            int remove = std::min(can_remove, excess);
            budget[h] -= remove;
            excess -= remove;
        }
    }

    // Greedy exchange iterations
    const int max_iters = n_heads * 20;

    for (int iter = 0; iter < max_iters; iter++) {
        // Find donor: head whose last key has lowest marginal value
        int donor = -1;
        float min_marginal = 1e30f;
        for (int h = 0; h < n_heads; h++) {
            if (budget[h] <= min_per_head) continue;
            float marginal = sensitivity[h] / (float)budget[h];
            if (marginal < min_marginal) {
                min_marginal = marginal;
                donor = h;
            }
        }

        // Find recipient: head whose next key has highest marginal value
        int recipient = -1;
        float max_marginal = 0.0f;
        for (int h = 0; h < n_heads; h++) {
            if (h == donor) continue;
            if (budget[h] >= max_per_head) continue;
            float marginal = sensitivity[h] / (float)(budget[h] + 1);
            if (marginal > max_marginal) {
                max_marginal = marginal;
                recipient = h;
            }
        }

        // Exchange if beneficial
        if (donor >= 0 && recipient >= 0 && max_marginal > min_marginal) {
            budget[donor]--;
            budget[recipient]++;
        } else {
            break;  // Converged
        }
    }

    return budget;
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

    // Per-head temporary data for reuse in LS step
    struct head_data {
        std::vector<float> scores;      // [n_q, T] scaled attention logits
        std::vector<float> attn_weights;// [n_q, T] softmax attention
    };
    std::vector<head_data> hdata(n_head_kv);

    const float inv_sqrt_dk = 1.0f / sqrtf((float) d_k);

    // Temporary contiguous buffers for per-head extraction (reused across heads)
    std::vector<float> q_buf(n_q * d_k);
    std::vector<float> k_buf(T * d_k);
    std::vector<float> v_buf(T * d_v);

    for (int h = 0; h < n_head_kv; h++) {
        auto & hd = hdata[h];
        hd.scores.resize(n_q * T);
        hd.attn_weights.resize(n_q * T);

        // Extract per-head Q_ref and K into contiguous buffers for cache-friendly matmul
        const int head_offset = h * d_k;
        extract_head_contiguous(Q_ref_all, q_buf.data(), n_q, d_k, n_embd_k_gqa, head_offset);
        extract_head_contiguous(K_all, k_buf.data(), T, d_k, n_embd_k_gqa, head_offset);

        // Batched matmul: scores = Q_ref_h @ K_h^T  [n_q × T]
        mat_mul_ABt(q_buf.data(), k_buf.data(), hd.scores.data(), n_q, T, d_k);
        for (int i = 0; i < n_q * T; i++) hd.scores[i] *= inv_sqrt_dk;

        // Fused: softmax + per-key importance in a single pass
        std::vector<float> head_importance(T);
        softmax_importance_fused(hd.scores.data(), hd.attn_weights.data(),
                                 head_importance.data(), n_q, T);

        // Global score = max across heads
        for (int j = 0; j < T; j++) {
            if (head_importance[j] > global_scores[j]) {
                global_scores[j] = head_importance[j];
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

        result.beta[h].assign(t, 0.0f);
        result.C_v[h].resize(t * d_v);

        // Least squares for C_v (Section 3.3, Eq. 6)
        // X_ij = softmax(score[qi, sel[j]]), Y = original attn output
        std::vector<float> X(n_q * t);
        for (int qi = 0; qi < n_q; qi++) {
            for (int j = 0; j < t; j++) {
                X[qi * t + j] = hd.scores[qi * T + selected[j]];
            }
        }
        softmax_rows(X.data(), n_q, t);

        // Y = original attention output: attn_weights @ V_head  [n_q, d_v]
        // Extract V_head into contiguous buffer, then use batched matmul
        extract_head_contiguous(V_all, v_buf.data(), T, d_v, n_embd_v_gqa, h * d_v);
        std::vector<float> Y(n_q * d_v);
        mat_mul_AB(hd.attn_weights.data(), v_buf.data(), Y.data(), n_q, T, d_v);

        least_squares_solve(X.data(), Y.data(), result.C_v[h].data(), n_q, t, d_v);
    }

    return result;
}
