// KV Cache Compaction Performance Profiling Infrastructure
//
// Provides performance tracking for compaction operations to enable
// data-driven optimization decisions.
//
// RICE Score: 1,520 (Highest priority item)
//   Reach: 100% (all users benefit from performance visibility)
//   Impact: 4 (Low - enables future optimization)
//   Confidence: 95% (well-understood tools)
//   Effort: 0.25 person-weeks (add perf counters, hooks)

#pragma once

#include <chrono>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>

// ============================================================================
// Performance Metrics Structure
// ============================================================================

struct KVCompactPerfMetrics {
    // Overall compaction time
    double total_compaction_ms = 0.0;

    // Individual operation timings
    double key_selection_ms = 0.0;
    double beta_computation_ms = 0.0;
    double value_refitting_ms = 0.0;
    double state_parsing_ms = 0.0;
    double state_writing_ms = 0.0;

    // Matrix operation timings (for GPU optimization targeting)
    double matmul_atb_ms = 0.0;         // A^T @ B (beta computation)
    double matmul_abt_ms = 0.0;         // A @ B^T (attention scores)
    double attention_score_ms = 0.0;    // Q @ K^T
    double value_aggregation_ms = 0.0;  // A @ V

    // GPU-specific timings (when CUDA/ROCm is enabled)
    double h2d_transfer_ms = 0.0;       // Host to device memory transfer
    double kernel_compute_ms = 0.0;     // GPU kernel execution
    double d2h_transfer_ms = 0.0;       // Device to host transfer
    double gpu_overhead_ms = 0.0;       // GPU initialization/cleanup

    // Memory usage
    size_t peak_memory_bytes = 0;
    size_t gpu_memory_bytes = 0;

    // Operation counts
    int layers_processed = 0;
    int heads_processed = 0;
    int tokens_compacted = 0;
    int matmul_operations = 0;

    // Quality metrics
    double cosine_similarity = 0.0;
    double relative_error = 0.0;

    // Percentage of total inference time
    double percentage_of_inference = 0.0;

    void reset() {
        *this = KVCompactPerfMetrics{};
    }

    void print_summary() const {
        printf("\n=== KV Compaction Performance Summary ===\n");
        printf("Total compaction time: %.2f ms\n", total_compaction_ms);
        printf("  - Key selection:     %.2f ms (%.1f%%)\n",
               key_selection_ms, (key_selection_ms / total_compaction_ms) * 100.0);
        printf("  - Beta computation:  %.2f ms (%.1f%%)\n",
               beta_computation_ms, (beta_computation_ms / total_compaction_ms) * 100.0);
        printf("  - Value refitting:   %.2f ms (%.1f%%)\n",
               value_refitting_ms, (value_refitting_ms / total_compaction_ms) * 100.0);
        printf("  - State parsing:     %.2f ms (%.1f%%)\n",
               state_parsing_ms, (state_parsing_ms / total_compaction_ms) * 100.0);
        printf("  - State writing:     %.2f ms (%.1f%%)\n",
               state_writing_ms, (state_writing_ms / total_compaction_ms) * 100.0);

        if (gpu_overhead_ms > 0.0) {
            printf("\nGPU Operations:\n");
            printf("  - H2D transfer:      %.2f ms\n", h2d_transfer_ms);
            printf("  - Kernel compute:    %.2f ms\n", kernel_compute_ms);
            printf("  - D2H transfer:      %.2f ms\n", d2h_transfer_ms);
            printf("  - GPU overhead:      %.2f ms\n", gpu_overhead_ms);
            printf("  - GPU memory:        %.2f MB\n", gpu_memory_bytes / (1024.0 * 1024.0));
        }

        printf("\nMatrix Operations:\n");
        printf("  - matmul A^T @ B:    %.2f ms (%d ops)\n",
               matmul_atb_ms, matmul_operations);
        printf("  - Attention scores:  %.2f ms\n", attention_score_ms);
        printf("  - Value aggregation: %.2f ms\n", value_aggregation_ms);

        printf("\nStatistics:\n");
        printf("  - Layers processed:  %d\n", layers_processed);
        printf("  - Heads processed:   %d\n", heads_processed);
        printf("  - Tokens compacted:  %d\n", tokens_compacted);
        printf("  - Peak memory:       %.2f MB\n",
               peak_memory_bytes / (1024.0 * 1024.0));

        printf("\nQuality Metrics:\n");
        printf("  - Cosine similarity: %.4f\n", cosine_similarity);
        printf("  - Relative error:    %.4f\n", relative_error);

        if (percentage_of_inference > 0.0) {
            printf("\nInference Impact:\n");
            printf("  - Compaction is %.2f%% of total inference time\n",
                   percentage_of_inference);
        }

        printf("==========================================\n\n");
    }
};

// ============================================================================
// Performance Scoped Timer
// ============================================================================

class PerfTimer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
    const char* name_;
    double& target_ms_;

public:
    PerfTimer(const char* name, double& target_ms)
        : name_(name), target_ms_(target_ms) {
        start_ = std::chrono::high_resolution_clock::now();
    }

    ~PerfTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start_;
        target_ms_ += elapsed.count() * 1000.0;  // Convert to milliseconds
    }

    // Disable copy, enable move
    PerfTimer(const PerfTimer&) = delete;
    PerfTimer& operator=(const PerfTimer&) = delete;
    PerfTimer(PerfTimer&&) = default;
    PerfTimer& operator=(PerfTimer&&) = default;
};

// ============================================================================
// Auto-Detection: GPU Availability
// ============================================================================

struct GPUInfo {
    bool has_cuda = false;
    bool has_rocm = false;
    bool has_metal = false;
    int device_count = 0;
    std::string device_name;
    size_t total_memory_mb = 0;
    size_t free_memory_mb = 0;

    bool has_any_gpu() const {
        return has_cuda || has_rocm || has_metal;
    }

    void print_info() const {
        printf("\n=== GPU Detection ===\n");
        printf("CUDA available:  %s\n", has_cuda ? "Yes" : "No");
        printf("ROCm available:  %s\n", has_rocm ? "Yes" : "No");
        printf("Metal available: %s\n", has_metal ? "Yes" : "No");
        printf("Device count:    %d\n", device_count);

        if (has_any_gpu()) {
            printf("Device name:     %s\n", device_name.c_str());
            printf("Total memory:    %zu MB\n", total_memory_mb);
            printf("Free memory:     %zu MB\n", free_memory_mb);
        }
        printf("====================\n\n");
    }
};

// Detect available GPU backends
inline GPUInfo detect_gpu_capabilities() {
    GPUInfo info;

#ifdef GGML_USE_CUDA
    // Check CUDA availability
    int cuda_device_count = 0;
    cudaError_t cuda_result = cudaGetDeviceCount(&cuda_device_count);
    if (cuda_result == cudaSuccess && cuda_device_count > 0) {
        info.has_cuda = true;
        info.device_count = cuda_device_count;

        // Get device name and memory for first device
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        info.device_name = prop.name;
        info.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);

        // Get free memory
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        info.free_memory_mb = free_mem / (1024 * 1024);
    }
#endif

#ifdef GGML_USE_HIP
    // Check ROCm/HIP availability
    int hip_device_count = 0;
    hipError_t hip_result = hipGetDeviceCount(&hip_device_count);
    if (hip_result == hipSuccess && hip_device_count > 0) {
        info.has_rocm = true;
        if (info.device_count == 0) {
            info.device_count = hip_device_count;
        }

        // Get device name and memory for first device
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, 0);
        if (info.device_name.empty()) {
            info.device_name = prop.name;
        }
        info.total_memory_mb = prop.totalGlobalMem / (1024 * 1024);

        // Get free memory
        size_t free_mem, total_mem;
        hipMemGetInfo(&free_mem, &total_mem);
        if (info.free_memory_mb == 0) {
            info.free_memory_mb = free_mem / (1024 * 1024);
        }
    }
#endif

#ifdef GGML_USE_METAL
    // Check Metal availability
    info.has_metal = true;
    info.device_count = 1;  // Metal typically reports 1 unified GPU
    info.device_name = "Apple Silicon GPU";
#endif

    return info;
}

// ============================================================================
// Performance Recommendations
// ============================================================================

struct OptimizationRecommendation {
    std::string operation;
    std::string recommendation;
    double potential_speedup;
    std::string reasoning;
    int priority;  // 1 = highest, 5 = lowest
};

inline std::vector<OptimizationRecommendation>
analyze_performance_bottlenecks(const KVCompactPerfMetrics& metrics,
                                const GPUInfo& gpu_info) {
    std::vector<OptimizationRecommendation> recommendations;

    // Check if GPU inference should be enabled
    if (gpu_info.has_any_gpu() && metrics.beta_computation_ms > 1.0) {
        // GPU available but not using it for inference
        recommendations.push_back({
            "GPU Inference",
            "Enable GGML CUDA/ROCm backend for llama.cpp inference",
            3.0 * (metrics.total_compaction_ms / metrics.percentage_of_inference),
            "GGML GPU backend typically provides 3-7x speedup on inference",
            1  // Highest priority
        });
    }

    // Check if matrix operations are bottlenecked
    if (metrics.matmul_atb_ms > 2.0 && !gpu_info.has_any_gpu()) {
        recommendations.push_back({
            "GPU Matrix Operations",
            "Implement GGML backend for matmul operations (beta computation)",
            10.0,  // O(n³) → GPU cuBLAS/rocBLAS
            "Matrix multiplication is O(n³) and benefits massively from GPU",
            2
        });
    }

    // Check if attention score computation is significant
    if (metrics.attention_score_ms > 1.0 && gpu_info.has_any_gpu()) {
        recommendations.push_back({
            "GPU Attention Scores",
            "Use llama.cpp flash attention kernels for Q @ K^T",
            5.0,
            "Flash attention provides 5-20x speedup on attention computation",
            2
        });
    }

    // Check if memory transfers are bottleneck (GPU-specific)
    if (metrics.h2d_transfer_ms + metrics.d2h_transfer_ms > metrics.kernel_compute_ms) {
        recommendations.push_back({
            "Memory Transfer Optimization",
            "Implement batch processing or memory pooling to reduce H2D/D2H overhead",
            2.0,
            "Memory transfers dominate kernel compute time",
            3
        });
    }

    // Check if compaction is significant portion of inference
    if (metrics.percentage_of_inference > 5.0) {
        recommendations.push_back({
            "Compaction Algorithm",
            "Profile compaction sub-operations for optimization opportunities",
            1.5,
            "Compaction is >5% of inference time, worth optimizing",
            3
        });
    } else {
        recommendations.push_back({
            "Focus on Inference",
            "Compaction is <5% of total time, focus on inference optimization",
            0.0,
            "GPU compaction has minimal impact when inference dominates",
            5  // Lowest priority
        });
    }

    return recommendations;
}

inline void print_recommendations(const std::vector<OptimizationRecommendation>& recs) {
    if (recs.empty()) {
        printf("\n=== Performance Recommendations ===\n");
        printf("No specific recommendations. System is well-optimized!\n");
        printf("=====================================\n\n");
        return;
    }

    printf("\n=== Performance Recommendations (RICE-Prioritized) ===\n");

    // Sort by priority
    auto sorted = recs;
    std::sort(sorted.begin(), sorted.end(),
              [](const auto& a, const auto& b) { return a.priority < b.priority; });

    for (const auto& rec : sorted) {
        printf("\n[PRIORITY %d] %s\n", rec.priority, rec.operation.c_str());
        printf("  Recommendation: %s\n", rec.recommendation.c_str());
        printf("  Potential speedup: %.1fx\n", rec.potential_speedup);
        printf("  Reasoning: %s\n", rec.reasoning.c_str());
    }

    printf("\n========================================================\n\n");
}

// ============================================================================
// Convenience Macros for Easy Profiling
// ============================================================================>

#define KV_COMPACT_PERF_BEGIN(metrics) \
    KVCompactPerfMetrics perf_metrics_##metrics; \
    auto perf_start_##metrics = clock::now();

#define KV_COMPACT_PERF_END(metrics) \
    do { \
        auto perf_end_##metrics = clock::now(); \
        duration perf_elapsed_##metrics = perf_end_##metrics - perf_start_##metrics; \
        perf_metrics_##metrics.total_compaction_ms = perf_elapsed_##metrics.count() * 1000.0; \
    } while(0);

#define KV_COMPACT_PERF_STAGE(metrics, stage) \
    PerfTimer _timer_##stage(#stage, perf_metrics_##metrics.stage##_ms);

#define KV_COMPACT_PERF_PRINT(metrics) \
    perf_metrics_##metrics.print_summary();
