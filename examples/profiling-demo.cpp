// KV Cache Compaction Profiling Demo
//
// Demonstrates the profiling infrastructure for data-driven optimization.
// RICE Score: 1,520 (Highest priority item)
//
// Build:
//   cmake -DKV_COMPACT_ENABLE_PROFILING=ON ..
//   make profiling-demo
//
// Run:
//   ./profiling-demo

#include <cstdio>
#include <vector>
#include <cstdlib>
#include "kv-compact-math.h"

#ifdef KV_COMPACT_ENABLE_PROFILING
#include "kv-compact-profiling.h"

int main() {
    printf("=== KV Cache Compaction Profiling Demo ===\n\n");

    // Detect GPU capabilities
    GPUInfo gpu_info = detect_gpu_capabilities();
    gpu_info.print_info();

    // Set up test data
    const int T = 8192;      // Original token count
    const int t = 133;       // Compacted token count (example from benchmark)
    const int n_q = 33;      // Number of reference queries
    const int d_k = 256;     // Key dimension (Qwen3.5-0.8B)
    const int d_v = 256;     // Value dimension

    printf("Test configuration:\n");
    printf("  Original tokens (T): %d\n", T);
    printf("  Compacted tokens (t): %d\n", t);
    printf("  Reference queries (n_q): %d\n", n_q);
    printf("  Key dimension (d_k): %d\n", d_k);
    printf("  Value dimension (d_v): %d\n", d_v);
    printf("  Compression ratio: %.1fx\n\n", (float)T / t);

    // Allocate test data
    std::vector<float> K(T * d_k);
    std::vector<float> V(T * d_v);
    std::vector<float> Q_ref(n_q * d_k);

    // Initialize with random data
    for (size_t i = 0; i < K.size(); i++) {
        K[i] = (float)rand() / RAND_MAX;
    }
    for (size_t i = 0; i < V.size(); i++) {
        V[i] = (float)rand() / RAND_MAX;
    }
    for (size_t i = 0; i < Q_ref.size(); i++) {
        Q_ref[i] = (float)rand() / RAND_MAX;
    }

    // Perform compaction with profiling
    KVCompactPerfMetrics perf_metrics;

    printf("Running compaction with profiling...\n");
    compacted_head result = compact_head_highest_attn_profiled(
        K.data(), V.data(), Q_ref.data(),
        T, n_q, d_k, d_v, t,
        2,  // n_alt_rounds
        KEY_SELECT_MAX_ATTN,
        BETA_FIT_CLOSED_FORM,
        &perf_metrics
    );

    // Print performance summary
    perf_metrics.print_summary();

    // Analyze bottlenecks and provide recommendations
    auto recommendations = analyze_performance_bottlenecks(perf_metrics, gpu_info);
    print_recommendations(recommendations);

    // Example: Compare matrix operation timings
    printf("\n=== Matrix Operation Breakdown ===\n");
    printf("mat_mul_AtB (beta computation):  %.2f ms\n", perf_metrics.matmul_atb_ms);
    printf("Attention scores (Q @ K^T):      %.2f ms\n", perf_metrics.attention_score_ms);
    printf("Value aggregation (A @ V):       %.2f ms\n", perf_metrics.value_aggregation_ms);
    printf("Total matmul operations:        %d\n", perf_metrics.matmul_operations);

    if (perf_metrics.matmul_atb_ms > 0) {
        double matmul_percentage = (perf_metrics.matmul_atb_ms / perf_metrics.total_compaction_ms) * 100.0;
        printf("\nBeta computation matmul is %.1f%% of total compaction time\n", matmul_percentage);
        if (matmul_percentage > 20.0 && !gpu_info.has_any_gpu()) {
            printf("  -> RECOMMENDATION: High impact GPU optimization target\n");
        }
    }

    // Memory usage analysis
    printf("\n=== Memory Usage Analysis ===\n");
    size_t input_memory = (T * d_k + T * d_v) * sizeof(float);
    size_t output_memory = (t * d_k + t * d_v) * sizeof(float);
    size_t working_memory = (n_q * T + n_q * t) * sizeof(float);

    printf("Input K+V memory:    %.2f MB\n", input_memory / (1024.0 * 1024.0));
    printf("Output K+V memory:   %.2f MB\n", output_memory / (1024.0 * 1024.0));
    printf("Working memory:      %.2f MB\n", working_memory / (1024.0 * 1024.0));
    printf("Peak memory:         %.2f MB\n", perf_metrics.peak_memory_bytes / (1024.0 * 1024.0));
    printf("Memory saved:        %.2f MB (%.1f%% compression)\n",
           (input_memory - output_memory) / (1024.0 * 1024.0),
           100.0 * (1.0 - (double)output_memory / input_memory));

    if (gpu_info.has_any_gpu()) {
        printf("\nGPU memory capacity: %zu MB\n", gpu_info.total_memory_mb);
        size_t required_gpu_memory = (input_memory + output_memory + working_memory) / (1024 * 1024);
        printf("Required GPU memory:  %zu MB\n", required_gpu_memory);

        if (required_gpu_memory < gpu_info.free_memory_mb) {
            printf("  -> STATUS: Fits in GPU memory with %zu MB to spare\n",
                   gpu_info.free_memory_mb - required_gpu_memory);
        } else {
            printf("  -> WARNING: Exceeds GPU memory by %zu MB\n",
                   required_gpu_memory - gpu_info.free_memory_mb);
        }
    }

    // Performance projections
    printf("\n=== Performance Projections ===\n");
    printf("Current CPU compaction: %.2f ms\n", perf_metrics.total_compaction_ms);

    if (gpu_info.has_any_gpu()) {
        // Project GPU speedup based on matmul operations
        double gpu_matmul_time = perf_metrics.matmul_atb_ms / 20.0;  // 20x speedup estimate
        double gpu_total_time = perf_metrics.total_compaction_ms - perf_metrics.matmul_atb_ms + gpu_matmul_time;

        printf("Projected GPU compaction: %.2f ms (%.1fx speedup)\n",
               gpu_total_time, perf_metrics.total_compaction_ms / gpu_total_time);
    } else {
        printf("GPU not detected - CPU-only optimization recommended\n");
    }

    printf("\n=== Demo Complete ===\n");
    return 0;
}

#else

int main() {
    printf("Profiling is not enabled. Build with:\n");
    printf("  cmake -DKV_COMPACT_ENABLE_PROFILING=ON ..\n");
    return 1;
}

#endif // KV_COMPACT_ENABLE_PROFILING
