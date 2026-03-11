// KV Cache Compaction via Attention Matching - Math Utilities
//
// Pure CPU float32 linear algebra routines used by the compaction algorithm.
// Extracted for testability.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

// Key selection strategy for compaction
enum key_select_mode {
    KEY_SELECT_MAX_ATTN    = 0,  // Original: top-t by max attention weight (fast)
    KEY_SELECT_SUBMODULAR  = 1,  // Greedy submodular: coverage + diversity (better quality)
    KEY_SELECT_TOKEN_MERGE = 2,  // Merge similar tokens instead of evicting (preserves info)
    KEY_SELECT_KMEANS      = 3,  // K-means clustering: centroid keys + averaged values
};

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

// Solve non-negative least squares via projected gradient descent:
//   min_{w >= 0} ||A*w - b||^2
// A is (m x n), b is (m), w is (n)
// Returns solution in w
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
            w[i] = std::max(1e-12f, w[i] - step * grad[i]);
        }
    }
}

// Beta fitting strategy for mass matching (Step 2)
enum beta_fit_mode {
    BETA_FIT_NNLS     = 0,  // Projected gradient NNLS (original)
    BETA_FIT_SINKHORN = 1,  // Sinkhorn multiplicative updates (OT-inspired)
};

// Solve non-negative mass matching via Sinkhorn-like multiplicative updates:
//   Given M (m x n) with M_ij >= 0 and target b (m) with b_i > 0,
//   find w (n) with w_j > 0 such that M * w ≈ b.
//
// Multiplicative update: w ← w ⊙ (M^T (b / (M w))) / m
// This automatically maintains non-negativity and is equivalent to
// minimizing KL(b || M w) — the entropic OT objective.
//
// The entropic regularization (eps) prevents degenerate solutions
// by adding eps * sum(w * log(w)) to the objective.
//
static void sinkhorn_beta_fit(const float * M, const float * b, float * w,
                              int m, int n, int max_iter = 100, float eps = 0.01f) {
    // Initialize w uniformly
    for (int j = 0; j < n; j++) {
        w[j] = 1.0f;
    }

    std::vector<float> Mw(m);
    std::vector<float> ratio(m);

    for (int iter = 0; iter < max_iter; iter++) {
        // Compute M * w
        for (int i = 0; i < m; i++) {
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
                sum += M[i * n + j] * w[j];
            }
            Mw[i] = sum;
        }

        // Compute ratio = b / (M * w)
        for (int i = 0; i < m; i++) {
            ratio[i] = b[i] / (Mw[i] + 1e-12f);
        }

        // Multiplicative update: w ← w * (M^T * ratio) / (M^T * 1)
        // The denominator (column sums of M) normalizes for column scale.
        // Entropic damping: raise the update factor to 1/(1+eps) to
        // prevent degenerate solutions where all mass goes to one key.
        float exp_factor = 1.0f / (1.0f + eps);
        for (int j = 0; j < n; j++) {
            float num = 0.0f, den = 0.0f;
            for (int i = 0; i < m; i++) {
                num += M[i * n + j] * ratio[i];
                den += M[i * n + j];
            }
            float update = num / (den + 1e-12f);
            w[j] *= powf(std::max(1e-12f, update), exp_factor);
            w[j] = std::max(1e-12f, w[j]);
        }
    }
}

// Solve least squares: min ||A*x - b||^2 via normal equations
// A is (m x n), b is (m x p), x is (n x p)
// Uses Cholesky-like approach: x = (A^T A)^{-1} A^T b
// For simplicity, uses pseudo-inverse via regularized normal equations
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

struct compacted_head {
    std::vector<int>   selected_indices;  // which original tokens were selected
    std::vector<float> beta;              // attention mass biases [t]
    std::vector<float> C_v;               // refit values [t * d_v]
    std::vector<float> C_k;               // merged keys [t * d_k] (empty = use original K at selected_indices)
};

// Configuration for compaction pipeline
struct compaction_config {
    key_select_mode select_mode  = KEY_SELECT_MAX_ATTN;
    beta_fit_mode   fit_mode     = BETA_FIT_NNLS;
    int             n_alt_rounds = 2;        // alternating minimization rounds
    const float *   head_sensitivity = nullptr; // [n_head_kv] or nullptr for auto
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

    // Merged keys (token merging only): C_k[h] is [t * d_k]
    // Empty = use original K at selected_indices (subset selection modes)
    std::vector<std::vector<float>> C_k;   // [n_head_kv][t * d_k]

    // Per-head sensitivity used for key selection weighting [n_head_kv]
    // Higher = more influence on which keys are kept
    std::vector<float> head_sensitivity;
};

// ============================================================================
// Submodular key selection (BumbleBee-inspired)
// ============================================================================
//
// Selects t keys by greedily maximizing a mixture of:
//   - Facility location (coverage): each unselected key is "covered" by its
//     most similar selected key, weighted by attention importance
//   - Graph cut (diversity): selected keys should span different regions
//     of the key space
//
// This gives a (1-1/e) approximation guarantee for the combined objective.

// Compute cosine similarity between two key vectors
static float key_cosine_sim(const float * a, const float * b, int d) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (int i = 0; i < d; i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    return dot / (sqrtf(na * nb) + 1e-8f);
}

// Greedy submodular key selection
//
//   K:             [T, d_k] key vectors (single head or concatenated)
//   importance:    [T] per-key importance scores (e.g., max attention weight)
//   T:             number of tokens
//   d_k:           key dimension
//   t:             target selection size
//   lambda:        mixture weight in [0,1]: 1.0 = pure coverage, 0.0 = pure diversity
//
// Returns: sorted indices of selected keys
//
// Maximum T for submodular selection before falling back to top-t.
// T=4096 uses 64MB; beyond this, memory cost dominates any quality gain.
static constexpr int SUBMODULAR_MAX_T = 4096;

static std::vector<int> submodular_key_select(
        const float * K, const float * importance,
        int T, int d_k, int t, float lambda = 0.7f) {

    if (t >= T) {
        std::vector<int> all(T);
        std::iota(all.begin(), all.end(), 0);
        return all;
    }

    // Guard: O(T^2) similarity matrix. For T > SUBMODULAR_MAX_T, fall back
    // to importance-based top-t selection to avoid excessive memory use.
    if (T > SUBMODULAR_MAX_T) {
        std::vector<int> indices(T);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                          [&](int a, int b) { return importance[a] > importance[b]; });
        std::vector<int> result(indices.begin(), indices.begin() + t);
        std::sort(result.begin(), result.end());
        return result;
    }

    // Precompute pairwise similarity matrix (T <= SUBMODULAR_MAX_T)
    // sim[i][j] = cosine(K[i], K[j]) * sqrt(importance[i] * importance[j])
    std::vector<float> sim(T * T);
    for (int i = 0; i < T; i++) {
        sim[i * T + i] = 1.0f * importance[i];
        for (int j = i + 1; j < T; j++) {
            float cs = key_cosine_sim(K + i * d_k, K + j * d_k, d_k);
            float w = sqrtf(importance[i] * importance[j]);
            float s = cs * w;
            sim[i * T + j] = s;
            sim[j * T + i] = s;
        }
    }

    // Greedy selection with incremental state tracking
    //
    // Facility location (coverage): F(S) = sum_{j in U} max_{i in S} sim(j, i)
    //   Marginal gain: sum_{j} max(0, sim(j,k) - nearest[j])
    //
    // Redundancy penalty: -max_{i in S} sim(k, i)
    //   Penalizes selecting keys too similar to already-selected ones

    std::vector<float> nearest(T, 0.0f);  // max sim to selected set, per key
    std::vector<bool>  selected_mask(T, false);

    std::vector<int> result;
    result.reserve(t);

    for (int step = 0; step < t; step++) {
        float best_gain = -1e30f;
        int   best_idx  = -1;

        for (int k = 0; k < T; k++) {
            if (selected_mask[k]) continue;

            // Facility location marginal gain (coverage)
            float fl_gain = 0.0f;
            for (int j = 0; j < T; j++) {
                float improvement = sim[j * T + k] - nearest[j];
                if (improvement > 0.0f) fl_gain += improvement;
            }

            // Redundancy penalty: how similar is k to the closest selected key?
            float redundancy = nearest[k];  // max sim to selected set
            float gain = lambda * fl_gain - (1.0f - lambda) * redundancy;

            if (gain > best_gain) {
                best_gain = gain;
                best_idx  = k;
            }
        }

        result.push_back(best_idx);
        selected_mask[best_idx] = true;

        // Update nearest-selected similarity for all keys
        for (int j = 0; j < T; j++) {
            float s = sim[j * T + best_idx];
            if (s > nearest[j]) nearest[j] = s;
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

// ============================================================================
// Token merging (ToMe-inspired)
// ============================================================================
//
// Reduces T tokens to t by iteratively merging the most similar key pairs.
// Each merge:
//   - Averages key vectors (weighted by cluster size)
//   - Averages value vectors (weighted by cluster size)
//   - Accumulates cluster size for beta = log(cluster_size)
//
// Unlike subset selection, this produces NEW key/value vectors (centroids)
// that preserve information from merged tokens.

// Merge T tokens down to t by iteratively combining most similar pairs.
//
//   K:        [T, d_k] key vectors for one head
//   V:        [T, d_v] value vectors for one head
//   T:        number of tokens
//   d_k:      key dimension
//   d_v:      value dimension
//   t:        target count
//
// Returns:
//   merged_K:       [t, d_k] merged key centroids
//   merged_V:       [t, d_v] merged value centroids
//   merged_beta:    [t] log(cluster_size) for each merged token
//   representative: [t] index of the original token that seeded each cluster
//
struct merged_tokens {
    std::vector<float> K;              // [t * d_k]
    std::vector<float> V;              // [t * d_v]
    std::vector<float> beta;           // [t]
    std::vector<int>   representative; // [t] original token index per cluster
};

static merged_tokens token_merge(
        const float * K, const float * V,
        int T, int d_k, int d_v, int t) {

    if (t >= T) {
        merged_tokens result;
        result.K.assign(K, K + T * d_k);
        result.V.assign(V, V + T * d_v);
        result.beta.assign(T, 0.0f);
        result.representative.resize(T);
        std::iota(result.representative.begin(), result.representative.end(), 0);
        return result;
    }

    // Working copies: each "live" token has K, V, size, and original index
    struct token_info {
        std::vector<float> k;    // [d_k]
        std::vector<float> v;    // [d_v]
        int   size;              // cluster size
        int   orig_idx;          // representative original index
        bool  alive;
    };

    std::vector<token_info> tokens(T);
    for (int i = 0; i < T; i++) {
        tokens[i].k.assign(K + i * d_k, K + (i + 1) * d_k);
        tokens[i].v.assign(V + i * d_v, V + (i + 1) * d_v);
        tokens[i].size = 1;
        tokens[i].orig_idx = i;
        tokens[i].alive = true;
    }

    int n_alive = T;

    // Iteratively merge the most similar pair until t remain
    while (n_alive > t) {
        // Find the most similar alive pair
        float best_sim = -1e30f;
        int   best_i = -1, best_j = -1;

        for (int i = 0; i < T; i++) {
            if (!tokens[i].alive) continue;
            for (int j = i + 1; j < T; j++) {
                if (!tokens[j].alive) continue;
                float cs = key_cosine_sim(tokens[i].k.data(), tokens[j].k.data(), d_k);
                if (cs > best_sim) {
                    best_sim = cs;
                    best_i = i;
                    best_j = j;
                }
            }
        }

        // Merge j into i (weighted average by cluster size)
        float wi = (float)tokens[best_i].size;
        float wj = (float)tokens[best_j].size;
        float wt = wi + wj;

        for (int d = 0; d < d_k; d++) {
            tokens[best_i].k[d] = (wi * tokens[best_i].k[d] + wj * tokens[best_j].k[d]) / wt;
        }
        for (int d = 0; d < d_v; d++) {
            tokens[best_i].v[d] = (wi * tokens[best_i].v[d] + wj * tokens[best_j].v[d]) / wt;
        }
        tokens[best_i].size += tokens[best_j].size;
        tokens[best_j].alive = false;
        n_alive--;
    }

    // Collect surviving tokens
    merged_tokens result;
    result.K.resize(t * d_k);
    result.V.resize(t * d_v);
    result.beta.resize(t);
    result.representative.resize(t);

    int idx = 0;
    for (int i = 0; i < T; i++) {
        if (!tokens[i].alive) continue;
        memcpy(result.K.data() + idx * d_k, tokens[i].k.data(), d_k * sizeof(float));
        memcpy(result.V.data() + idx * d_v, tokens[i].v.data(), d_v * sizeof(float));
        result.beta[idx] = logf((float)tokens[i].size);
        result.representative[idx] = tokens[i].orig_idx;
        idx++;
    }

    return result;
}

// ============================================================================
// K-means clustering of keys
// ============================================================================
//
// Partitions T keys into t clusters using Lloyd's algorithm, producing
// centroid keys and weighted-average values. Unlike token merging (greedy
// pairwise), K-means optimizes a global objective (within-cluster variance).
//
// Returns the same merged_tokens struct as token_merge.

static merged_tokens kmeans_compact(
        const float * K, const float * V,
        int T, int d_k, int d_v, int t, int max_iters = 20) {

    if (t >= T) {
        merged_tokens result;
        result.K.assign(K, K + T * d_k);
        result.V.assign(V, V + T * d_v);
        result.beta.assign(T, 0.0f);
        result.representative.resize(T);
        std::iota(result.representative.begin(), result.representative.end(), 0);
        return result;
    }

    // Initialize centroids: pick t evenly-spaced keys (deterministic, no RNG needed)
    std::vector<float> centroids(t * d_k);
    for (int c = 0; c < t; c++) {
        int idx = (int)((float)c / t * T);
        memcpy(centroids.data() + c * d_k, K + idx * d_k, d_k * sizeof(float));
    }

    std::vector<int> assignment(T, 0);  // cluster assignment for each token

    for (int iter = 0; iter < max_iters; iter++) {
        // E-step: assign each token to nearest centroid
        bool changed = false;
        for (int i = 0; i < T; i++) {
            float best_dist = 1e30f;
            int   best_c = 0;
            for (int c = 0; c < t; c++) {
                float dist = 0.0f;
                for (int d = 0; d < d_k; d++) {
                    float diff = K[i * d_k + d] - centroids[c * d_k + d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    best_c = c;
                }
            }
            if (assignment[i] != best_c) {
                assignment[i] = best_c;
                changed = true;
            }
        }
        if (!changed) break;

        // M-step: recompute centroids
        std::vector<int> counts(t, 0);
        std::fill(centroids.begin(), centroids.end(), 0.0f);
        for (int i = 0; i < T; i++) {
            int c = assignment[i];
            counts[c]++;
            for (int d = 0; d < d_k; d++) {
                centroids[c * d_k + d] += K[i * d_k + d];
            }
        }
        for (int c = 0; c < t; c++) {
            if (counts[c] > 0) {
                for (int d = 0; d < d_k; d++) {
                    centroids[c * d_k + d] /= (float)counts[c];
                }
            } else {
                // Reinitialize empty cluster: pick the token farthest from
                // its assigned centroid (splits the worst-fit cluster)
                float worst_dist = -1.0f;
                int   worst_idx  = 0;
                for (int i = 0; i < T; i++) {
                    int ac = assignment[i];
                    float dist = 0.0f;
                    for (int d = 0; d < d_k; d++) {
                        float diff = K[i * d_k + d] - centroids[ac * d_k + d];
                        dist += diff * diff;
                    }
                    if (dist > worst_dist) {
                        worst_dist = dist;
                        worst_idx = i;
                    }
                }
                memcpy(centroids.data() + c * d_k, K + worst_idx * d_k, d_k * sizeof(float));
            }
        }
    }

    // Build result: centroid K, averaged V, beta = log(cluster_size)
    merged_tokens result;
    result.K = centroids;
    result.V.resize(t * d_v, 0.0f);
    result.beta.resize(t);
    result.representative.resize(t);

    std::vector<int> counts(t, 0);
    for (int i = 0; i < T; i++) {
        int c = assignment[i];
        counts[c]++;
        for (int d = 0; d < d_v; d++) {
            result.V[c * d_v + d] += V[i * d_v + d];
        }
    }

    for (int c = 0; c < t; c++) {
        if (counts[c] > 0) {
            for (int d = 0; d < d_v; d++) {
                result.V[c * d_v + d] /= (float)counts[c];
            }
            result.beta[c] = logf((float)counts[c]);
            // Representative: first token assigned to this cluster
            result.representative[c] = -1;
            for (int i = 0; i < T; i++) {
                if (assignment[i] == c) {
                    result.representative[c] = i;
                    break;
                }
            }
        } else {
            // Empty cluster after convergence: centroid is from reinit,
            // value is zero, beta = -inf (zero weight). Use nearest token.
            result.beta[c] = -10.0f;  // exp(-10) ≈ 0, negligible weight
            float best_dist = 1e30f;
            result.representative[c] = 0;
            for (int i = 0; i < T; i++) {
                float dist = 0.0f;
                for (int d = 0; d < d_k; d++) {
                    float diff = K[i * d_k + d] - result.K[c * d_k + d];
                    dist += diff * diff;
                }
                if (dist < best_dist) {
                    best_dist = dist;
                    result.representative[c] = i;
                }
            }
            // Use that token's value
            memcpy(result.V.data() + c * d_v,
                   V + result.representative[c] * d_v,
                   d_v * sizeof(float));
        }
    }

    return result;
}

// Compact a single KV head using the Highest Attention Keys method
//
//   K:       [T, d_k] original keys for this head
//   V:       [T, d_v] original values for this head
//   Q_ref:   [n_q, d_k] reference queries
//   t:       target compacted size
//   d_k:     key dimension
//   d_v:     value dimension
//
// Returns compacted_head with selected indices, beta, and C_v
static compacted_head compact_head_highest_attn(
        const float * K, const float * V, const float * Q_ref,
        int T, int n_q, int d_k, int d_v, int t,
        int n_alt_rounds = 2,
        key_select_mode select_mode = KEY_SELECT_MAX_ATTN,
        beta_fit_mode   fit_mode   = BETA_FIT_NNLS) {

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

    // Token merge mode: merge similar tokens instead of selecting + NNLS + LS
    if (select_mode == KEY_SELECT_TOKEN_MERGE) {
        auto merged = token_merge(K, V, T, d_k, d_v, t);
        result.selected_indices = merged.representative;
        result.beta = merged.beta;
        result.C_v = merged.V;
        result.C_k = merged.K;
        return result;
    }

    // K-means mode: cluster keys and use centroids
    if (select_mode == KEY_SELECT_KMEANS) {
        auto merged = kmeans_compact(K, V, T, d_k, d_v, t);
        result.selected_indices = merged.representative;
        result.beta = merged.beta;
        result.C_v = merged.V;
        result.C_k = merged.K;
        return result;
    }

    // Step 1: Compute attention scores Q_ref @ K^T / sqrt(d_k)
    //   scores: [n_q, T]
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

    // Select top-t keys
    std::vector<int> selected;
    if (select_mode == KEY_SELECT_SUBMODULAR) {
        selected = submodular_key_select(K, key_scores.data(), T, d_k, t, 0.7f);
    } else {
        std::vector<int> indices(T);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                          [&](int a, int b) { return key_scores[a] > key_scores[b]; });
        selected.assign(indices.begin(), indices.begin() + t);
        std::sort(selected.begin(), selected.end());
    }
    result.selected_indices = selected;

    // Step 2 (initial): Solve NNLS for beta (mass matching)
    //   Design matrix M: M_ij = exp(q_i * K_{selected[j]} / sqrt(d))
    //   Target: m_i = row_sums[i] (already computed)

    std::vector<float> M(n_q * t);
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < t; j++) {
            M[i * t + j] = exp_scores[i * T + selected[j]];
        }
    }

    std::vector<float> w(t);
    if (fit_mode == BETA_FIT_SINKHORN) {
        sinkhorn_beta_fit(M.data(), row_sums.data(), w.data(), n_q, t);
    } else {
        nnls_solve(M.data(), row_sums.data(), w.data(), n_q, t);
    }

    for (int j = 0; j < t; j++) {
        result.beta[j] = logf(std::max(1e-12f, w[j]));
    }

    // Compute Y (original attention output) once: attn_weights @ V [n_q, d_v]
    std::vector<float> Y(n_q * d_v, 0.0f);
    for (int i = 0; i < n_q; i++) {
        for (int j = 0; j < T; j++) {
            float w_ij = attn_weights[i * T + j];
            for (int d = 0; d < d_v; d++) {
                Y[i * d_v + d] += w_ij * V[j * d_v + d];
            }
        }
    }

    // Steps 2-3 alternating minimization: refine beta and C_v jointly
    std::vector<float> X(n_q * t);

    for (int round = 0; round < n_alt_rounds; round++) {
        // Step 3: Build X from current beta, solve LS for C_v
        for (int i = 0; i < n_q; i++) {
            for (int j = 0; j < t; j++) {
                X[i * t + j] = scores[i * T + selected[j]] + result.beta[j];
            }
        }
        softmax_rows(X.data(), n_q, t);

        least_squares_solve(X.data(), Y.data(), result.C_v.data(), n_q, t, d_v);

        // Gradient refinement of beta given C_v (skip on last round)
        if (round < n_alt_rounds - 1) {
            std::vector<float> R(n_q * d_v);
            for (int qi = 0; qi < n_q; qi++) {
                for (int d = 0; d < d_v; d++) {
                    float o = 0.0f;
                    for (int j = 0; j < t; j++) {
                        o += X[qi * t + j] * result.C_v[j * d_v + d];
                    }
                    R[qi * d_v + d] = o - Y[qi * d_v + d];
                }
            }

            std::vector<float> grad_beta(t, 0.0f);
            for (int qi = 0; qi < n_q; qi++) {
                std::vector<float> g(t);
                for (int j = 0; j < t; j++) {
                    float dot = 0.0f;
                    for (int d = 0; d < d_v; d++) {
                        dot += R[qi * d_v + d] * result.C_v[j * d_v + d];
                    }
                    g[j] = 2.0f * dot;
                }

                float g_bar = 0.0f;
                for (int k = 0; k < t; k++) {
                    g_bar += X[qi * t + k] * g[k];
                }

                for (int j = 0; j < t; j++) {
                    grad_beta[j] += X[qi * t + j] * (g[j] - g_bar);
                }
            }

            float grad_norm = 0.0f;
            for (int j = 0; j < t; j++) {
                grad_beta[j] /= n_q;
                grad_norm += grad_beta[j] * grad_beta[j];
            }
            grad_norm = sqrtf(grad_norm + 1e-12f);
            float lr = std::min(0.5f, 1.0f / (grad_norm + 1e-8f));

            for (int j = 0; j < t; j++) {
                result.beta[j] -= lr * grad_beta[j];
            }
        }
    }

    return result;
}

// Compute per-head sensitivity for key selection weighting
//
// Sensitivity measures how much attention mass falls outside the top-t keys.
// Heads that spread attention broadly are harder to compress and should have
// more influence on which keys are selected.
//
//   attn_weights: [n_q, T] softmax attention weights for one head
//   n_q:          number of queries
//   T:            number of tokens
//   t:            target compacted size
//
// Returns: sensitivity in [0, 1] where 1 = maximally sensitive (broad attention)
//
static float compute_head_sensitivity(const float * attn_weights, int n_q, int T, int t) {
    if (t >= T) return 0.0f;

    // For each query, compute the mass NOT covered by the top-t keys
    float total_uncovered = 0.0f;

    std::vector<float> weights(T);
    for (int qi = 0; qi < n_q; qi++) {
        // Copy weights for this query
        for (int j = 0; j < T; j++) {
            weights[j] = attn_weights[qi * T + j];
        }

        // Partial sort to find top-t weights
        std::partial_sort(weights.begin(), weights.begin() + t, weights.end(),
                          [](float a, float b) { return a > b; });

        float covered = 0.0f;
        for (int j = 0; j < t; j++) {
            covered += weights[j];
        }
        total_uncovered += (1.0f - covered);
    }

    return total_uncovered / n_q;
}

// ============================================================================
// Carathéodory-informed budget allocation
// ============================================================================
//
// Carathéodory's theorem: any convex combination of T points in R^d can be
// expressed using at most d+1 points. For attention, the output is a convex
// combination of value vectors in R^{d_v}, so t_min = d_v + 1 per head.
//
// If the value matrix has effective rank r << d_v, then only r+1 entries
// are needed. This gives a tighter, per-head compression floor.
//
// compute_caratheodory_budget: given V for one head, compute effective rank
// and return the minimum budget (effective_rank + 1).

// Compute effective rank of V via singular value thresholding.
// V: [T, d_v], returns effective rank (number of significant singular values).
// Uses the Frobenius norm ratio: rank_eff = (sum(sigma))^2 / sum(sigma^2)
// This is a smooth, parameter-free measure of effective dimensionality.
static int compute_effective_rank(const float * V, int T, int d_v) {
    if (T == 0 || d_v == 0) return 0;

    // Compute V^T V (d_v x d_v) — the Gram matrix of columns.
    // Accumulate trace and Frobenius norm during construction to avoid
    // a second O(d_v^2) pass.
    float trace = 0.0f;    // tr(VtV)  = sum of sigma_i^2
    float trace2 = 0.0f;   // ||VtV||_F^2 = sum of sigma_i^4

    std::vector<float> VtV(d_v * d_v, 0.0f);
    for (int i = 0; i < d_v; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < T; k++) {
                sum += V[k * d_v + i] * V[k * d_v + j];
            }
            VtV[i * d_v + j] = sum;
            VtV[j * d_v + i] = sum;

            if (i == j) {
                trace += sum;
                trace2 += sum * sum;
            } else {
                // Off-diagonal: appears twice in Frobenius norm
                trace2 += 2.0f * sum * sum;
            }
        }
    }

    // Effective rank = tr(VtV)^2 / ||VtV||_F^2
    // = (sum sigma_i^2)^2 / (sum sigma_i^4)

    if (trace2 < 1e-12f) return 1;

    // Effective rank = trace^2 / trace2
    float eff_rank = (trace * trace) / trace2;

    return std::max(1, (int)ceilf(eff_rank));
}

// Compute per-head Carathéodory budget: minimum number of entries needed
// to exactly represent any convex combination of value vectors.
//
// This is a standalone utility — call it before compaction to determine
// per-head target sizes. The compaction pipeline itself uses a uniform t;
// use these budgets to decide what t should be, or to identify heads
// that can be compressed more aggressively.
//
// Example usage:
//   auto budgets = compute_caratheodory_budgets(V_all, T, n_heads, d_v);
//   int t = *std::max_element(budgets.begin(), budgets.end());
//
//   V_all: [T, n_embd_v_gqa] all heads concatenated
//   T:     number of tokens
//   n_head_kv: number of KV heads
//   d_v:   value dimension per head
//
// Returns: [n_head_kv] minimum budget per head (effective_rank + 1)
static std::vector<int> compute_caratheodory_budgets(
        const float * V_all, int T, int n_head_kv, int d_v) {

    const int n_embd_v_gqa = n_head_kv * d_v;
    std::vector<int> budgets(n_head_kv);

    for (int h = 0; h < n_head_kv; h++) {
        // Extract per-head V
        std::vector<float> V_h(T * d_v);
        for (int i = 0; i < T; i++) {
            memcpy(V_h.data() + i * d_v,
                   V_all + i * n_embd_v_gqa + h * d_v,
                   d_v * sizeof(float));
        }

        int eff_rank = compute_effective_rank(V_h.data(), T, d_v);
        budgets[h] = std::min(T, eff_rank + 1);
    }

    return budgets;
}

// Compact all KV heads within a single layer using shared key selection
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
//   head_sensitivity: optional [n_head_kv] weights for key selection (nullptr = auto-compute)
//
// Algorithm:
//   1. For each head, compute attention scores and per-key importance
//   2. Global key selection: max importance across heads for each position
//   3. Per-head NNLS (beta) and least-squares (C_v) on shared selection
//
static compacted_layer compact_layer_all_heads(
        const float * K_all, const float * V_all, const float * Q_ref_all,
        int T, int n_q, int n_head_kv, int d_k, int d_v, int t,
        const compaction_config & cfg = {}) {

    const float *       head_sensitivity = cfg.head_sensitivity;
    const int           n_alt_rounds     = cfg.n_alt_rounds;
    const key_select_mode select_mode    = cfg.select_mode;
    const beta_fit_mode   fit_mode       = cfg.fit_mode;

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

    // Token merge mode: merge per-head independently, use union of representatives
    if (select_mode == KEY_SELECT_TOKEN_MERGE) {
        result.C_k.resize(n_head_kv);

        // Merge each head independently
        std::vector<merged_tokens> per_head_merged(n_head_kv);
        for (int h = 0; h < n_head_kv; h++) {
            // Extract per-head K and V
            std::vector<float> K_h(T * d_k), V_h(T * d_v);
            for (int i = 0; i < T; i++) {
                memcpy(K_h.data() + i * d_k, K_all + i * n_embd_k_gqa + h * d_k, d_k * sizeof(float));
                memcpy(V_h.data() + i * d_v, V_all + i * n_embd_v_gqa + h * d_v, d_v * sizeof(float));
            }
            per_head_merged[h] = token_merge(K_h.data(), V_h.data(), T, d_k, d_v, t);
        }

        // Use head 0's representatives as the shared selection
        // (the actual K/V data comes from C_k/C_v, not from original positions)
        result.selected_indices = per_head_merged[0].representative;

        for (int h = 0; h < n_head_kv; h++) {
            result.beta[h] = per_head_merged[h].beta;
            result.C_v[h] = per_head_merged[h].V;
            result.C_k[h] = per_head_merged[h].K;
        }

        return result;
    }

    // K-means mode: cluster per-head independently
    if (select_mode == KEY_SELECT_KMEANS) {
        result.C_k.resize(n_head_kv);

        std::vector<merged_tokens> per_head_merged(n_head_kv);
        for (int h = 0; h < n_head_kv; h++) {
            std::vector<float> K_h(T * d_k), V_h(T * d_v);
            for (int i = 0; i < T; i++) {
                memcpy(K_h.data() + i * d_k, K_all + i * n_embd_k_gqa + h * d_k, d_k * sizeof(float));
                memcpy(V_h.data() + i * d_v, V_all + i * n_embd_v_gqa + h * d_v, d_v * sizeof(float));
            }
            per_head_merged[h] = kmeans_compact(K_h.data(), V_h.data(), T, d_k, d_v, t);
        }

        result.selected_indices = per_head_merged[0].representative;

        for (int h = 0; h < n_head_kv; h++) {
            result.beta[h] = per_head_merged[h].beta;
            result.C_v[h] = per_head_merged[h].V;
            result.C_k[h] = per_head_merged[h].K;
        }

        return result;
    }

    // ---- Step 1: Global key selection via max importance across heads ----

    // Compute per-head key importance scores, then take max across heads
    std::vector<float> global_scores(T, 0.0f);
    std::vector<float> per_head_importance(n_head_kv * T, 0.0f);

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
            per_head_importance[h * T + j] = max_w;
        }
    }

    // Compute or use provided sensitivity weights
    std::vector<float> sensitivity(n_head_kv, 1.0f);
    if (head_sensitivity) {
        for (int h = 0; h < n_head_kv; h++) {
            sensitivity[h] = head_sensitivity[h];
        }
    } else if (n_head_kv > 1) {
        // Auto-compute: measure how much mass falls outside top-t per head
        for (int h = 0; h < n_head_kv; h++) {
            sensitivity[h] = compute_head_sensitivity(
                hdata[h].attn_weights.data(), n_q, T, t);
            // Clamp minimum to avoid zero-weighting any head
            sensitivity[h] = std::max(0.01f, sensitivity[h]);
        }
    }

    // Sensitivity-weighted scoring: sum_h(sensitivity[h] * importance[h][j])
    // This gives more influence to heads that are harder to compress
    for (int j = 0; j < T; j++) {
        float score = 0.0f;
        for (int h = 0; h < n_head_kv; h++) {
            score += sensitivity[h] * per_head_importance[h * T + j];
        }
        global_scores[j] = score;
    }

    // ---- Key selection (mode-dependent) ----
    std::vector<int> selected;

    if (select_mode == KEY_SELECT_SUBMODULAR) {
        // Submodular selection: use concatenated keys weighted by importance.
        // We average keys across KV heads (weighted by sensitivity) to get
        // a single [T, d_k] representation for similarity computation.
        std::vector<float> K_avg(T * d_k, 0.0f);
        float sens_sum = 0.0f;
        for (int h = 0; h < n_head_kv; h++) sens_sum += sensitivity[h];
        for (int i = 0; i < T; i++) {
            for (int h = 0; h < n_head_kv; h++) {
                float w = sensitivity[h] / sens_sum;
                for (int d = 0; d < d_k; d++) {
                    K_avg[i * d_k + d] += w * K_all[i * n_embd_k_gqa + h * d_k + d];
                }
            }
        }

        selected = submodular_key_select(K_avg.data(), global_scores.data(),
                                         T, d_k, t, 0.7f);
    } else {
        // Default: top-t by global score
        std::vector<int> indices(T);
        std::iota(indices.begin(), indices.end(), 0);
        std::partial_sort(indices.begin(), indices.begin() + t, indices.end(),
                          [&](int a, int b) { return global_scores[a] > global_scores[b]; });
        selected.assign(indices.begin(), indices.begin() + t);
        std::sort(selected.begin(), selected.end());
    }

    result.selected_indices = selected;
    result.head_sensitivity = sensitivity;

    // ---- Steps 2-3: Per-head NNLS (beta) and least squares (C_v) ----
    //
    // With alternating minimization: after the initial NNLS+LS pass,
    // refine beta via gradient descent on ||softmax(s+beta)·C_v - Y||^2,
    // then re-solve LS for C_v. Repeat for n_alt_rounds total.



    for (int h = 0; h < n_head_kv; h++) {
        const auto & hd = hdata[h];

        result.beta[h].resize(t);
        result.C_v[h].resize(t * d_v);

        // Step 2 (initial): NNLS for beta
        std::vector<float> M(n_q * t);
        for (int qi = 0; qi < n_q; qi++) {
            for (int j = 0; j < t; j++) {
                M[qi * t + j] = hd.exp_scores[qi * T + selected[j]];
            }
        }

        std::vector<float> w(t);
        if (fit_mode == BETA_FIT_SINKHORN) {
            sinkhorn_beta_fit(M.data(), hd.row_sums.data(), w.data(), n_q, t);
        } else {
            nnls_solve(M.data(), hd.row_sums.data(), w.data(), n_q, t);
        }

        for (int j = 0; j < t; j++) {
            result.beta[h][j] = logf(std::max(1e-12f, w[j]));
        }

        // Compute Y (original attention output) once: attn_weights @ V_head [n_q, d_v]
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

        // Alternating minimization rounds
        std::vector<float> X(n_q * t);

        for (int round = 0; round < n_alt_rounds; round++) {
            // Step 3: Build X from current beta, solve LS for C_v
            for (int qi = 0; qi < n_q; qi++) {
                for (int j = 0; j < t; j++) {
                    X[qi * t + j] = hd.scores[qi * T + selected[j]] + result.beta[h][j];
                }
            }
            softmax_rows(X.data(), n_q, t);

            least_squares_solve(X.data(), Y.data(), result.C_v[h].data(), n_q, t, d_v);

            // Gradient refinement of beta given C_v (skip on last round)
            if (round < n_alt_rounds - 1) {
                // Compute output residual R = X·C_v - Y  [n_q, d_v]
                std::vector<float> R(n_q * d_v);
                for (int qi = 0; qi < n_q; qi++) {
                    for (int d = 0; d < d_v; d++) {
                        float o = 0.0f;
                        for (int j = 0; j < t; j++) {
                            o += X[qi * t + j] * result.C_v[h][j * d_v + d];
                        }
                        R[qi * d_v + d] = o - Y[qi * d_v + d];
                    }
                }

                // g[qi,j] = 2 * R[qi] · C_v[j]  (per-query, per-key gradient component)
                // dL/d(beta_j) = sum_qi X[qi,j] * (g[qi,j] - g_bar[qi])
                //   where g_bar[qi] = sum_k X[qi,k] * g[qi,k]
                std::vector<float> grad_beta(t, 0.0f);
                for (int qi = 0; qi < n_q; qi++) {
                    // Compute g[qi,j] for all j
                    std::vector<float> g(t);
                    for (int j = 0; j < t; j++) {
                        float dot = 0.0f;
                        for (int d = 0; d < d_v; d++) {
                            dot += R[qi * d_v + d] * result.C_v[h][j * d_v + d];
                        }
                        g[j] = 2.0f * dot;
                    }

                    // g_bar = sum_k X[qi,k] * g[k]
                    float g_bar = 0.0f;
                    for (int k = 0; k < t; k++) {
                        g_bar += X[qi * t + k] * g[k];
                    }

                    // Accumulate gradient
                    for (int j = 0; j < t; j++) {
                        grad_beta[j] += X[qi * t + j] * (g[j] - g_bar);
                    }
                }

                // Gradient step with adaptive learning rate
                float grad_norm = 0.0f;
                for (int j = 0; j < t; j++) {
                    grad_beta[j] /= n_q;
                    grad_norm += grad_beta[j] * grad_beta[j];
                }
                grad_norm = sqrtf(grad_norm + 1e-12f);
                float lr = std::min(0.5f, 1.0f / (grad_norm + 1e-8f));

                for (int j = 0; j < t; j++) {
                    result.beta[h][j] -= lr * grad_beta[j];
                }
            }
        }
    }

    return result;
}
