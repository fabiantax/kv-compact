// Standalone quantized KV throughput benchmark (B4)
// Measures per-stage timing: dequant, compact (scoring + LS), requant

#undef NDEBUG
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "kv-compact-api.h"
#include "kv-compact-state.h"

static void gen_data(float * out, int n, int seed) {
    for (int i = 0; i < n; i++) {
        seed = seed * 1103515245 + 12345;
        out[i] = ((seed >> 16) & 0x7FFF) / 16384.0f - 1.0f;
    }
}

static double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

int main() {
    printf("Quantized KV Throughput Benchmark (B4)\n");
    printf("======================================\n\n");

    const int n_head_kv = 4, d_k = 64, d_v = 64;
    int n_embd_k = n_head_kv * d_k;
    int n_embd_v = n_head_kv * d_v;

    // ---- Throughput table: all types x all sizes ----
    printf("  Throughput (tokens/sec) at 50%% compression, %d heads, d=%d:\n\n", n_head_kv, d_k);
    printf("  T       F32          Q8_0         Q4_0         Q4_1\n");
    printf("  ------  ----------   ----------   ----------   ----------\n");

    int sizes[] = { 256, 512, 1024, 2048, 4096 };
    int all_types[] = { KV_TYPE_F32, KV_TYPE_Q8_0, KV_TYPE_Q4_0, KV_TYPE_Q4_1 };

    for (int T : sizes) {
        int n_q = std::min(T / 4, 64);

        std::vector<float> K_f32(T * n_embd_k), V_f32(T * n_embd_v), Q(n_q * n_embd_k);
        gen_data(K_f32.data(), T * n_embd_k, 5000 + T);
        gen_data(V_f32.data(), T * n_embd_v, 5001 + T);
        gen_data(Q.data(), n_q * n_embd_k, 5002 + T);

        printf("  %-6d  ", T);

        for (int ti = 0; ti < 4; ti++) {
            int type = all_types[ti];
            double best_ms = 1e9;

            if (type == KV_TYPE_F32) {
                kv_compact_params p = kv_compact_params_default();
                p.target_ratio = 0.5f;
                p.chunk_size = -1;

                for (int r = 0; r < 3; r++) {
                    kv_compact_result res = {};
                    double t0 = now_ms();
                    kv_compact(K_f32.data(), V_f32.data(), Q.data(),
                               T, n_q, n_head_kv, d_k, d_v, &p, &res);
                    double elapsed = now_ms() - t0;
                    if (elapsed < best_ms) best_ms = elapsed;
                    kv_compact_result_free(&res);
                }
            } else {
                size_t k_rb = kv_quant_row_bytes(type, n_embd_k);
                size_t v_rb = kv_quant_row_bytes(type, n_embd_v);
                std::vector<uint8_t> K_q(T * k_rb), V_q(T * v_rb);
                for (int i = 0; i < T; i++) {
                    kv_quantize_row(K_f32.data() + i * n_embd_k, type, n_embd_k,
                                    K_q.data() + i * k_rb);
                    kv_quantize_row(V_f32.data() + i * n_embd_v, type, n_embd_v,
                                    V_q.data() + i * v_rb);
                }

                kv_compact_params p = kv_compact_params_default();
                p.target_ratio = 0.5f;
                p.chunk_size = -1;

                for (int r = 0; r < 3; r++) {
                    kv_compact_quant_result res = {};
                    double t0 = now_ms();
                    kv_compact_quantized(K_q.data(), V_q.data(), Q.data(),
                                          type, type, k_rb, v_rb,
                                          T, n_q, n_head_kv, d_k, d_v, &p, &res);
                    double elapsed = now_ms() - t0;
                    if (elapsed < best_ms) best_ms = elapsed;
                    kv_compact_quant_result_free(&res);
                }
            }

            printf("%-13.0f", T / (best_ms / 1000.0));
        }
        printf("\n");
    }

    // ---- Per-stage breakdown at multiple sizes ----
    printf("\n  Per-stage breakdown (Q8_0, 50%% ratio):\n\n");
    printf("  T       dequant_ms   scoring_ms   nnls+ls_ms   requant_ms   total_ms    tok/s\n");
    printf("  ------  ----------   ----------   ----------   ----------   --------    ------\n");

    for (int T : sizes) {
        int n_q = std::min(T / 4, 64);
        size_t k_rb = kv_quant_row_bytes(KV_TYPE_Q8_0, n_embd_k);
        size_t v_rb = kv_quant_row_bytes(KV_TYPE_Q8_0, n_embd_v);

        std::vector<float> K_f32_l(T * n_embd_k), V_f32_l(T * n_embd_v), Q_l(n_q * n_embd_k);
        gen_data(K_f32_l.data(), T * n_embd_k, 6000 + T);
        gen_data(V_f32_l.data(), T * n_embd_v, 6001 + T);
        gen_data(Q_l.data(), n_q * n_embd_k, 6002 + T);

        std::vector<uint8_t> K_q(T * k_rb), V_q(T * v_rb);
        for (int i = 0; i < T; i++) {
            kv_quantize_row(K_f32_l.data() + i * n_embd_k, KV_TYPE_Q8_0, n_embd_k,
                            K_q.data() + i * k_rb);
            kv_quantize_row(V_f32_l.data() + i * n_embd_v, KV_TYPE_Q8_0, n_embd_v,
                            V_q.data() + i * v_rb);
        }

        // Phase 1: Dequant
        std::vector<float> K_deq(T * n_embd_k), V_deq(T * n_embd_v);
        double t0 = now_ms();
        for (int i = 0; i < T; i++) {
            kv_dequantize_row(K_q.data() + i * k_rb, KV_TYPE_Q8_0, k_rb,
                              K_deq.data() + i * n_embd_k, n_embd_k);
            kv_dequantize_row(V_q.data() + i * v_rb, KV_TYPE_Q8_0, v_rb,
                              V_deq.data() + i * n_embd_v, n_embd_v);
        }
        double dequant_ms = now_ms() - t0;

        // Phase 2: Compact (scoring + NNLS/LS)
        kv_compact_params p = kv_compact_params_default();
        p.target_ratio = 0.5f;
        p.chunk_size = -1;
        kv_compact_result result = {};
        t0 = now_ms();
        kv_compact(K_deq.data(), V_deq.data(), Q_l.data(),
                   T, n_q, n_head_kv, d_k, d_v, &p, &result);
        double compact_ms = now_ms() - t0;
        double scoring_ms = result.stats.scoring_ms;
        double nnls_ms = result.stats.nnls_ms;

        // Phase 3: Requant
        int t_kept = result.t;
        std::vector<uint8_t> K_out(t_kept * k_rb), V_out(t_kept * v_rb);
        std::vector<float> v_row(n_embd_v);
        t0 = now_ms();
        for (int j = 0; j < t_kept; j++) {
            int orig = result.selected_indices[j];
            kv_quantize_row(K_deq.data() + orig * n_embd_k, KV_TYPE_Q8_0, n_embd_k,
                            K_out.data() + j * k_rb);
            for (int h = 0; h < n_head_kv; h++) {
                memcpy(v_row.data() + h * d_v, result.C_v[h] + j * d_v, d_v * sizeof(float));
            }
            kv_quantize_row(v_row.data(), KV_TYPE_Q8_0, n_embd_v,
                            V_out.data() + j * v_rb);
        }
        double requant_ms = now_ms() - t0;
        double total = dequant_ms + compact_ms + requant_ms;

        printf("  %-6d  %8.2f     %8.2f     %8.2f     %8.2f     %8.2f   %7.0f\n",
               T, dequant_ms, scoring_ms, nnls_ms, requant_ms, total,
               T / (total / 1000.0));

        kv_compact_result_free(&result);
    }

    // ---- Stage % breakdown at T=2048 ----
    printf("\n  Stage percentage breakdown (T=2048, Q8_0):\n\n");
    {
        int T = 2048, n_q = 64;
        size_t k_rb = kv_quant_row_bytes(KV_TYPE_Q8_0, n_embd_k);
        size_t v_rb = kv_quant_row_bytes(KV_TYPE_Q8_0, n_embd_v);

        std::vector<float> K_f32_l(T * n_embd_k), V_f32_l(T * n_embd_v), Q_l(n_q * n_embd_k);
        gen_data(K_f32_l.data(), T * n_embd_k, 7000);
        gen_data(V_f32_l.data(), T * n_embd_v, 7001);
        gen_data(Q_l.data(), n_q * n_embd_k, 7002);

        std::vector<uint8_t> K_q(T * k_rb), V_q(T * v_rb);
        for (int i = 0; i < T; i++) {
            kv_quantize_row(K_f32_l.data() + i * n_embd_k, KV_TYPE_Q8_0, n_embd_k,
                            K_q.data() + i * k_rb);
            kv_quantize_row(V_f32_l.data() + i * n_embd_v, KV_TYPE_Q8_0, n_embd_v,
                            V_q.data() + i * v_rb);
        }

        // Full pipeline with phase timings
        std::vector<float> K_deq(T * n_embd_k), V_deq(T * n_embd_v);

        double t0 = now_ms();
        for (int i = 0; i < T; i++) {
            kv_dequantize_row(K_q.data() + i * k_rb, KV_TYPE_Q8_0, k_rb,
                              K_deq.data() + i * n_embd_k, n_embd_k);
            kv_dequantize_row(V_q.data() + i * v_rb, KV_TYPE_Q8_0, v_rb,
                              V_deq.data() + i * n_embd_v, n_embd_v);
        }
        double dequant_ms = now_ms() - t0;

        kv_compact_params p = kv_compact_params_default();
        p.target_ratio = 0.5f;
        p.chunk_size = -1;
        kv_compact_result result = {};
        kv_compact(K_deq.data(), V_deq.data(), Q_l.data(),
                   T, n_q, n_head_kv, d_k, d_v, &p, &result);

        int t_kept = result.t;
        std::vector<uint8_t> K_out(t_kept * k_rb), V_out(t_kept * v_rb);
        std::vector<float> v_row(n_embd_v);
        t0 = now_ms();
        for (int j = 0; j < t_kept; j++) {
            int orig = result.selected_indices[j];
            kv_quantize_row(K_deq.data() + orig * n_embd_k, KV_TYPE_Q8_0, n_embd_k,
                            K_out.data() + j * k_rb);
            for (int h = 0; h < n_head_kv; h++) {
                memcpy(v_row.data() + h * d_v, result.C_v[h] + j * d_v, d_v * sizeof(float));
            }
            kv_quantize_row(v_row.data(), KV_TYPE_Q8_0, n_embd_v,
                            V_out.data() + j * v_rb);
        }
        double requant_ms = now_ms() - t0;

        double scoring = result.stats.scoring_ms;
        double nnls = result.stats.nnls_ms;
        double total = dequant_ms + result.stats.elapsed_ms + requant_ms;

        printf("  stage             time_ms    %%_total    analogy\n");
        printf("  ----------------  --------   --------   --------------------------\n");
        printf("  dequant (load)    %7.2f    %5.1f%%     ~ batch loading KV from VRAM\n",
               dequant_ms, 100.0 * dequant_ms / total);
        printf("  scoring           %7.2f    %5.1f%%     ~ prompt processing (Q@K^T)\n",
               scoring, 100.0 * scoring / total);
        printf("  NNLS + LS solve   %7.2f    %5.1f%%     ~ weight optimization\n",
               nnls, 100.0 * nnls / total);
        printf("  requant (store)   %7.2f    %5.1f%%     ~ token generation writeback\n",
               requant_ms, 100.0 * requant_ms / total);
        printf("  TOTAL             %7.2f    100.0%%\n", total);
        printf("  throughput:       %.0f tok/s (end-to-end)\n\n", T / (total / 1000.0));

        // Compare overhead: quantized vs F32
        kv_compact_result r_f32 = {};
        kv_compact(K_deq.data(), V_deq.data(), Q_l.data(),
                   T, n_q, n_head_kv, d_k, d_v, &p, &r_f32);

        printf("  Quantization overhead (Q8_0 vs F32):\n");
        printf("    F32 compact only:  %.2f ms (%.0f tok/s)\n",
               r_f32.stats.elapsed_ms, T / (r_f32.stats.elapsed_ms / 1000.0));
        printf("    Q8_0 end-to-end:   %.2f ms (%.0f tok/s)\n", total, T / (total / 1000.0));
        printf("    overhead:          +%.2f ms (+%.1f%% from dequant+requant)\n",
               dequant_ms + requant_ms,
               100.0 * (dequant_ms + requant_ms) / r_f32.stats.elapsed_ms);

        kv_compact_result_free(&result);
        kv_compact_result_free(&r_f32);
    }

    printf("\nDone.\n");
    return 0;
}
