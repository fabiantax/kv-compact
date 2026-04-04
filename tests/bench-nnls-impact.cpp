// Quick test: what's the quality impact of skipping NNLS (beta=0)?
// Compare: full pipeline vs beta=0 (LS only) vs unconstrained LS beta
#include "kv-compact-api.h"
#include "kv-compact-math.h"
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using clock_type = std::chrono::high_resolution_clock;

static void gen_data(float * data, int n, int seed) {
    srand(seed);
    for (int i = 0; i < n; i++)
        data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
}

int main() {
    printf("NNLS Impact Analysis\n");
    printf("====================\n\n");

    const int n_head_kv = 4, d_k = 64, d_v = 64;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;
    const int n_q = 64;

    struct test_case { int T; float ratio; };
    test_case cases[] = {
        {256,  0.2f},
        {256,  0.5f},
        {512,  0.2f},
        {1024, 0.2f},
        {2048, 0.2f},
    };

    printf("  %-6s  %-6s  %10s  %10s  %10s  %10s  %10s\n",
           "T", "ratio", "full_cos", "no_beta_cos", "full_ms", "nobeta_ms", "speedup");
    printf("  %-6s  %-6s  %10s  %10s  %10s  %10s  %10s\n",
           "------", "------", "----------", "----------", "----------", "----------", "----------");

    for (auto & tc : cases) {
        int T = tc.T;
        float ratio = tc.ratio;

        std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
        gen_data(K.data(), T * n_embd_k, 2000 + T);
        gen_data(V.data(), T * n_embd_v, 3000 + T);
        gen_data(Q.data(), n_q * n_embd_k, 4000 + T);

        // Full pipeline (with NNLS)
        kv_compact_params p1 = kv_compact_params_default();
        p1.target_ratio = ratio;

        kv_compact_result r1 = {};
        auto t0 = clock_type::now();
        kv_compact(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, &p1, &r1);
        double full_ms = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();

        // Skip NNLS: set nnls_max_iter=0
        kv_compact_params p2 = kv_compact_params_default();
        p2.target_ratio = ratio;
        p2.nnls_max_iter = 0;

        kv_compact_result r2 = {};
        t0 = clock_type::now();
        kv_compact(K.data(), V.data(), Q.data(), T, n_q, n_head_kv, d_k, d_v, &p2, &r2);
        double nobeta_ms = std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();

        // Evaluate quality: compare attention output on eval queries
        float inv_sqrt_dk = 1.0f / sqrtf((float)d_k);
        int h = 0; // head 0
        double cos_full = 0, cos_nobeta = 0;
        int n_eval = std::min(32, n_q);

        for (int qi = 0; qi < n_eval; qi++) {
            const float * q = Q.data() + qi * n_embd_k + h * d_k;

            // Original output
            std::vector<float> orig_scores(T);
            for (int j = 0; j < T; j++) {
                float dot = 0.0f;
                const float * k = K.data() + j * n_embd_k + h * d_k;
                for (int d = 0; d < d_k; d++) dot += q[d] * k[d];
                orig_scores[j] = dot * inv_sqrt_dk;
            }
            softmax_rows(orig_scores.data(), 1, T);
            std::vector<float> orig_out(d_v, 0.0f);
            for (int j = 0; j < T; j++) {
                const float * v = V.data() + j * n_embd_v + h * d_v;
                for (int d = 0; d < d_v; d++) orig_out[d] += orig_scores[j] * v[d];
            }

            // Full compaction output
            auto eval_compact = [&](const kv_compact_result & r) -> double {
                std::vector<float> comp_scores(r.t);
                for (int j = 0; j < r.t; j++) {
                    float dot = 0.0f;
                    const float * k = K.data() + r.selected_indices[j] * n_embd_k + h * d_k;
                    for (int d = 0; d < d_k; d++) dot += q[d] * k[d];
                    comp_scores[j] = dot * inv_sqrt_dk + r.beta[h][j];
                }
                softmax_rows(comp_scores.data(), 1, r.t);
                std::vector<float> comp_out(d_v, 0.0f);
                for (int j = 0; j < r.t; j++) {
                    const float * cv = r.C_v[h] + j * d_v;
                    for (int d = 0; d < d_v; d++) comp_out[d] += comp_scores[j] * cv[d];
                }

                float dotp = 0, no = 0, nc = 0;
                for (int d = 0; d < d_v; d++) {
                    dotp += orig_out[d] * comp_out[d];
                    no += orig_out[d] * orig_out[d];
                    nc += comp_out[d] * comp_out[d];
                }
                return dotp / (sqrtf(no * nc) + 1e-8f);
            };

            cos_full += eval_compact(r1);
            cos_nobeta += eval_compact(r2);
        }

        cos_full /= n_eval;
        cos_nobeta /= n_eval;

        printf("  %-6d  %-4.0f%%   %10.6f  %10.6f  %10.2f  %10.2f  %10.2fx\n",
               T, ratio * 100, cos_full, cos_nobeta, full_ms, nobeta_ms, full_ms / nobeta_ms);

        kv_compact_result_free(&r1);
        kv_compact_result_free(&r2);
    }

    printf("\nDone.\n");
    return 0;
}
