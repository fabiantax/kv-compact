// Benchmark: Bridge-aware key selection vs standard max-attention selection
//
// Compares quality (cosine similarity) and speed across different:
//   - Context lengths (T)
//   - Compression ratios
//   - Bridge weights (0 = standard, 0.3 = moderate, 0.5 = aggressive)
//
// Also tests with "bridge-heavy" attention patterns where some keys are
// structurally critical but have low individual attention scores.

#include "kv-compact-math.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

using clock_type = std::chrono::high_resolution_clock;

static void gen_data(float * data, int n, int seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
}

// Create attention pattern with deliberate bridge keys
// cluster_A queries attend to keys [0, T/3)
// cluster_B queries attend to keys [2*T/3, T)
// bridge keys at [T/3, 2*T/3) are attended by BOTH clusters (weakly)
static void gen_bridge_data(float * K, float * V, float * Q,
                             int T, int n_q, int d_k, int d_v, int seed) {
    gen_data(K, T * d_k, seed);
    gen_data(V, T * d_v, seed + 1000);

    // Queries are designed so half are similar to keys in cluster A,
    // half are similar to keys in cluster B
    int half_q = n_q / 2;
    for (int qi = 0; qi < half_q; qi++) {
        int src_key = qi % (T / 3); // cluster A key
        for (int d = 0; d < d_k; d++) {
            Q[qi * d_k + d] = K[src_key * d_k + d] + 0.1f * ((float)rand() / RAND_MAX - 0.5f);
        }
    }
    for (int qi = half_q; qi < n_q; qi++) {
        int src_key = 2 * T / 3 + (qi % (T / 3)); // cluster B key
        for (int d = 0; d < d_k; d++) {
            Q[qi * d_k + d] = K[src_key * d_k + d] + 0.1f * ((float)rand() / RAND_MAX - 0.5f);
        }
    }

    // Make bridge keys (middle third) weakly similar to both clusters
    for (int ki = T / 3; ki < 2 * T / 3; ki++) {
        int a_key = ki % (T / 3);
        int b_key = 2 * T / 3 + (ki % (T / 3));
        for (int d = 0; d < d_k; d++) {
            K[ki * d_k + d] = 0.3f * K[a_key * d_k + d] + 0.3f * K[b_key * d_k + d]
                             + 0.4f * K[ki * d_k + d];
        }
    }
}

struct bench_result {
    double cosine_sim;
    double time_ms;
    int    nnz;        // sparse edges (0 for standard)
};

static bench_result run_standard(const float * K, const float * V, const float * Q,
                                  int T, int n_q, int d_k, int d_v, int t) {
    auto t0 = clock_type::now();
    auto result = compact_head_highest_attn(K, V, Q, T, n_q, d_k, d_v, t);
    double ms = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();

    // Evaluate quality
    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
    double total_cos = 0;
    int n_eval = std::min(16, n_q);
    for (int qi = 0; qi < n_eval; qi++) {
        const float * q = Q + qi * d_k;
        std::vector<float> orig_s(T), comp_s(t);

        for (int j = 0; j < T; j++) {
            float dot = 0;
            for (int d = 0; d < d_k; d++) dot += q[d] * K[j * d_k + d];
            orig_s[j] = dot * inv_sqrt_dk;
        }
        softmax_rows(orig_s.data(), 1, T);

        std::vector<float> orig_out(d_v, 0.0f);
        for (int j = 0; j < T; j++)
            for (int d = 0; d < d_v; d++)
                orig_out[d] += orig_s[j] * V[j * d_v + d];

        for (int j = 0; j < t; j++) {
            float dot = 0;
            int idx = result.selected_indices[j];
            for (int d = 0; d < d_k; d++) dot += q[d] * K[idx * d_k + d];
            comp_s[j] = dot * inv_sqrt_dk + result.beta[j];
        }
        softmax_rows(comp_s.data(), 1, t);

        std::vector<float> comp_out(d_v, 0.0f);
        for (int j = 0; j < t; j++)
            for (int d = 0; d < d_v; d++)
                comp_out[d] += comp_s[j] * result.C_v[j * d_v + d];

        float dotp = 0, no = 0, nc = 0;
        for (int d = 0; d < d_v; d++) {
            dotp += orig_out[d] * comp_out[d];
            no += orig_out[d] * orig_out[d];
            nc += comp_out[d] * comp_out[d];
        }
        total_cos += dotp / (sqrtf(no * nc) + 1e-8f);
    }

    return {total_cos / n_eval, ms, 0};
}

static bench_result run_bridge(const float * K, const float * V, const float * Q,
                                int T, int n_q, int d_k, int d_v, int t,
                                float bridge_weight, float percentile) {
    auto t0 = clock_type::now();
    auto result = compact_head_bridge_aware(K, V, Q, T, n_q, d_k, d_v, t,
                                             bridge_weight, percentile);
    double ms = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();

    // Same quality evaluation
    float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
    double total_cos = 0;
    int n_eval = std::min(16, n_q);
    for (int qi = 0; qi < n_eval; qi++) {
        const float * q = Q + qi * d_k;
        std::vector<float> orig_s(T);

        for (int j = 0; j < T; j++) {
            float dot = 0;
            for (int d = 0; d < d_k; d++) dot += q[d] * K[j * d_k + d];
            orig_s[j] = dot * inv_sqrt_dk;
        }
        softmax_rows(orig_s.data(), 1, T);

        std::vector<float> orig_out(d_v, 0.0f);
        for (int j = 0; j < T; j++)
            for (int d = 0; d < d_v; d++)
                orig_out[d] += orig_s[j] * V[j * d_v + d];

        std::vector<float> comp_s(t);
        for (int j = 0; j < t; j++) {
            float dot = 0;
            int idx = result.selected_indices[j];
            for (int d = 0; d < d_k; d++) dot += q[d] * K[idx * d_k + d];
            comp_s[j] = dot * inv_sqrt_dk; // no beta (skip-beta mode)
        }
        softmax_rows(comp_s.data(), 1, t);

        std::vector<float> comp_out(d_v, 0.0f);
        for (int j = 0; j < t; j++)
            for (int d = 0; d < d_v; d++)
                comp_out[d] += comp_s[j] * result.C_v[j * d_v + d];

        float dotp = 0, no = 0, nc = 0;
        for (int d = 0; d < d_v; d++) {
            dotp += orig_out[d] * comp_out[d];
            no += orig_out[d] * orig_out[d];
            nc += comp_out[d] * comp_out[d];
        }
        total_cos += dotp / (sqrtf(no * nc) + 1e-8f);
    }

    return {total_cos / n_eval, ms, 0};
}

int main() {
    printf("Bridge-Aware Key Selection Benchmark\n");
    printf("=====================================\n\n");

    const int d_k = 64, d_v = 64, n_q = 64;

    // ---- Part 1: Random data (baseline) ----
    printf("Part 1: Random Data (uniform attention)\n");
    printf("─────────────────────────────────────────────────────────────────────────────\n");
    printf("  %-6s %-6s  %-12s %-12s %-12s  %-10s %-10s %-10s\n",
           "T", "ratio", "std_cos", "brg0.3_cos", "brg0.5_cos", "std_ms", "brg0.3_ms", "brg0.5_ms");
    printf("  %-6s %-6s  %-12s %-12s %-12s  %-10s %-10s %-10s\n",
           "------", "------", "----------", "----------", "----------",
           "--------", "--------", "--------");

    struct test_case { int T; float ratio; };
    test_case cases[] = {
        {128,  0.50f},
        {128,  0.20f},
        {256,  0.50f},
        {256,  0.20f},
        {512,  0.50f},
        {512,  0.20f},
        {1024, 0.50f},
        {1024, 0.20f},
        {2048, 0.20f},
    };

    for (auto & tc : cases) {
        int T = tc.T;
        int t = (int)(T * tc.ratio);

        std::vector<float> K(T * d_k), V(T * d_v), Q(n_q * d_k);
        gen_data(K.data(), T * d_k, 1000 + T);
        gen_data(V.data(), T * d_v, 2000 + T);
        gen_data(Q.data(), n_q * d_k, 3000 + T);

        auto r_std  = run_standard(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t);
        auto r_b03  = run_bridge(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t, 0.3f, 0.90f);
        auto r_b05  = run_bridge(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t, 0.5f, 0.90f);

        printf("  %-6d %-4.0f%%   %-12.6f %-12.6f %-12.6f  %-10.2f %-10.2f %-10.2f\n",
               T, tc.ratio * 100,
               r_std.cosine_sim, r_b03.cosine_sim, r_b05.cosine_sim,
               r_std.time_ms, r_b03.time_ms, r_b05.time_ms);
    }

    // ---- Part 2: Bridge-heavy data ----
    printf("\nPart 2: Bridge-Heavy Data (clustered attention with bridge keys)\n");
    printf("─────────────────────────────────────────────────────────────────────────────\n");
    printf("  %-6s %-6s  %-12s %-12s %-12s  %-10s %-10s %-10s\n",
           "T", "ratio", "std_cos", "brg0.3_cos", "brg0.5_cos", "std_ms", "brg0.3_ms", "brg0.5_ms");
    printf("  %-6s %-6s  %-12s %-12s %-12s  %-10s %-10s %-10s\n",
           "------", "------", "----------", "----------", "----------",
           "--------", "--------", "--------");

    for (auto & tc : cases) {
        int T = tc.T;
        int t = (int)(T * tc.ratio);

        std::vector<float> K(T * d_k), V(T * d_v), Q(n_q * d_k);
        gen_bridge_data(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, 5000 + T);

        auto r_std  = run_standard(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t);
        auto r_b03  = run_bridge(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t, 0.3f, 0.90f);
        auto r_b05  = run_bridge(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t, 0.5f, 0.90f);

        printf("  %-6d %-4.0f%%   %-12.6f %-12.6f %-12.6f  %-10.2f %-10.2f %-10.2f\n",
               T, tc.ratio * 100,
               r_std.cosine_sim, r_b03.cosine_sim, r_b05.cosine_sim,
               r_std.time_ms, r_b03.time_ms, r_b05.time_ms);
    }

    // ---- Part 3: Extreme compression ----
    printf("\nPart 3: Extreme Compression (5%% retention — where bridges matter most)\n");
    printf("─────────────────────────────────────────────────────────────────────────────\n");
    printf("  %-6s %-6s  %-12s %-12s %-12s  %-10s %-10s %-10s\n",
           "T", "ratio", "std_cos", "brg0.3_cos", "brg0.5_cos", "std_ms", "brg0.3_ms", "brg0.5_ms");
    printf("  %-6s %-6s  %-12s %-12s %-12s  %-10s %-10s %-10s\n",
           "------", "------", "----------", "----------", "----------",
           "--------", "--------", "--------");

    test_case extreme_cases[] = {
        {256,  0.05f},
        {512,  0.05f},
        {1024, 0.05f},
        {2048, 0.05f},
    };

    for (auto & tc : extreme_cases) {
        int T = tc.T;
        int t = std::max(4, (int)(T * tc.ratio));

        std::vector<float> K(T * d_k), V(T * d_v), Q(n_q * d_k);
        gen_bridge_data(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, 7000 + T);

        auto r_std  = run_standard(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t);
        auto r_b03  = run_bridge(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t, 0.3f, 0.90f);
        auto r_b05  = run_bridge(K.data(), V.data(), Q.data(), T, n_q, d_k, d_v, t, 0.5f, 0.90f);

        printf("  %-6d %-4.0f%%   %-12.6f %-12.6f %-12.6f  %-10.2f %-10.2f %-10.2f\n",
               T, tc.ratio * 100,
               r_std.cosine_sim, r_b03.cosine_sim, r_b05.cosine_sim,
               r_std.time_ms, r_b03.time_ms, r_b05.time_ms);
    }

    // ---- Part 4: CSR sparsity stats ----
    printf("\nPart 4: CSR Sparsity Statistics\n");
    printf("───────────────────────────────────────────────────\n");
    printf("  %-6s %-12s %-12s %-10s %-10s\n",
           "T", "dense_edges", "sparse_nnz", "sparsity%", "csr_ms");
    printf("  %-6s %-12s %-12s %-10s %-10s\n",
           "------", "----------", "----------", "--------", "--------");

    int sparsity_sizes[] = {128, 256, 512, 1024, 2048, 4096};
    for (int T : sparsity_sizes) {
        std::vector<float> K(T * d_k), Q(n_q * d_k);
        gen_data(K.data(), T * d_k, 9000 + T);
        gen_data(Q.data(), n_q * d_k, 9500 + T);

        // Compute attention
        float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
        std::vector<float> attn(n_q * T);
        mat_mul_ABt(Q.data(), K.data(), attn.data(), n_q, T, d_k);
        for (int i = 0; i < n_q * T; i++) attn[i] *= inv_sqrt_dk;
        softmax_rows(attn.data(), n_q, T);

        auto t0 = clock_type::now();
        csr_matrix csr = csr_from_threshold(attn.data(), n_q, T, 0.90f);
        double csr_ms = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();

        int dense = n_q * T;
        printf("  %-6d %-12d %-12d %-9.1f%% %-10.3f\n",
               T, dense, csr.nnz(), 100.0 * (1.0 - (double)csr.nnz() / dense), csr_ms);
    }

    printf("\nDone.\n");
    return 0;
}
