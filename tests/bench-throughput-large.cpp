// Quick throughput benchmark at 10k and 100k context sizes
#include "kv-compact-api.h"
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
    printf("Throughput at large context sizes\n");
    printf("==================================\n\n");

    const int n_head_kv = 4, d_k = 64, d_v = 64;
    const int n_embd_k = n_head_kv * d_k;
    const int n_embd_v = n_head_kv * d_v;
    const int n_q = 64;

    struct test_case { int T; float ratio; };
    test_case cases[] = {
        {4096,   0.5f},
        {4096,   0.2f},
        {10240,  0.5f},
        {10240,  0.2f},
        {10240,  0.1f},
        {102400, 0.2f},
        {102400, 0.1f},
        {102400, 0.05f},
    };

    printf("  %-7s  %-8s  %8s  %12s  %14s\n",
           "T", "ratio", "t", "time_ms", "tokens/sec");
    printf("  %-7s  %-8s  %8s  %12s  %14s\n",
           "-------", "--------", "--------", "------------", "--------------");

    for (auto & tc : cases) {
        int T = tc.T;
        float ratio = tc.ratio;

        std::vector<float> K(T * n_embd_k), V(T * n_embd_v), Q(n_q * n_embd_k);
        gen_data(K.data(), T * n_embd_k, 2000 + T);
        gen_data(V.data(), T * n_embd_v, 3000 + T);
        gen_data(Q.data(), n_q * n_embd_k, 4000 + T);

        kv_compact_params p = kv_compact_params_default();
        p.target_ratio = ratio;

        kv_compact_result result = {};
        auto t0 = clock_type::now();
        kv_compact(K.data(), V.data(), Q.data(),
                   T, n_q, n_head_kv, d_k, d_v, &p, &result);
        double ms = std::chrono::duration<double, std::milli>(
            clock_type::now() - t0).count();

        printf("  %-7d  %-4.0f    %%  %8d  %12.2f  %14.0f\n",
               T, ratio * 100, result.t, ms, T / (ms / 1000.0));
        kv_compact_result_free(&result);
    }

    printf("\nDone.\n");
    return 0;
}
