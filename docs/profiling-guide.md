# KV Cache Compaction Profiling Guide

## Overview

The profiling infrastructure provides **data-driven optimization** for KV cache compaction. This is the **highest priority optimization** according to RICE scoring (1,520 points).

**Why?**
- Reach: 100% (all users benefit from performance visibility)
- Impact: 4 (Low - enables future optimization)
- Confidence: 95% (well-understood tools)
- Effort: 0.25 person-weeks (completed ✅)

## Features

### 1. Performance Metrics Tracking

Tracks all compaction stages:
- Key selection time
- Beta computation time (including matrix operations)
- Value refitting time
- State parsing/writing time
- GPU-specific timings (H2D/D2H transfers, kernel execution)

### 2. GPU Detection

Automatically detects available GPU backends:
- CUDA (NVIDIA)
- ROCm (AMD)
- Metal (Apple Silicon)

### 3. Optimization Recommendations

Analyzes performance data and provides **RICE-prioritized** recommendations:
- GPU inference enablement (Priority 1)
- GPU matrix operations (Priority 2)
- Memory transfer optimization (Priority 3)

## Usage

### Basic Profiling

```cpp
#define KV_COMPACT_ENABLE_PROFILING
#include "kv-compact-math.h"
#include "kv-compact-profiling.h"

int main() {
    // Detect GPU
    GPUInfo gpu_info = detect_gpu_capabilities();
    gpu_info.print_info();

    // Set up data
    const int T = 8192, t = 133, n_q = 33, d_k = 256, d_v = 256;
    std::vector<float> K(T * d_k), V(T * d_v), Q_ref(n_q * d_k);
    // ... initialize data ...

    // Run compaction with profiling
    KVCompactPerfMetrics perf_metrics;
    compacted_head result = compact_head_highest_attn_profiled(
        K.data(), V.data(), Q_ref.data(),
        T, n_q, d_k, d_v, t,
        2,  // n_alt_rounds
        KEY_SELECT_MAX_ATTN,
        BETA_FIT_CLOSED_FORM,
        &perf_metrics  // Pass metrics pointer
    );

    // Print performance summary
    perf_metrics.print_summary();

    // Get optimization recommendations
    auto recommendations = analyze_performance_bottlenecks(perf_metrics, gpu_info);
    print_recommendations(recommendations);

    return 0;
}
```

### Building with Profiling

```bash
# Enable profiling macro
g++ -std=c++11 -DKV_COMPACT_ENABLE_PROFILING \
    -Iinclude \
    src/kv-compact.cpp \
    -o kv-compact-profiled

# Or with CMake
cmake -DKV_COMPACT_ENABLE_PROFILING=ON ..
make
```

## Performance Metrics Structure

```cpp
struct KVCompactPerfMetrics {
    // Overall timing
    double total_compaction_ms;

    // Stage timings
    double key_selection_ms;
    double beta_computation_ms;
    double value_refitting_ms;
    double state_parsing_ms;
    double state_writing_ms;

    // Matrix operation timings
    double matmul_atb_ms;         // A^T @ B (beta computation)
    double matmul_abt_ms;         // A @ B^T (attention scores)
    double attention_score_ms;    // Q @ K^T
    double value_aggregation_ms;  // A @ V

    // GPU-specific timings
    double h2d_transfer_ms;
    double kernel_compute_ms;
    double d2h_transfer_ms;
    double gpu_overhead_ms;
    size_t gpu_memory_bytes;

    // Statistics
    int layers_processed;
    int heads_processed;
    int tokens_compacted;
    int matmul_operations;

    // Quality metrics
    double cosine_similarity;
    double relative_error;
};
```

## Interpreting Results

### Example Output

```
=== KV Compaction Performance Summary ===
Total compaction time: 23.90 ms
  - Key selection:     8.50 ms (35.6%)
  - Beta computation:  12.30 ms (51.5%)
  - Value refitting:   2.80 ms (11.7%)
  - State parsing:     0.20 ms (0.8%)
  - State writing:     0.10 ms (0.4%)

Matrix Operations:
  - matmul A^T @ B:    12.30 ms (1 ops)
  - Attention scores:  8.50 ms
  - Value aggregation: 2.80 ms

Statistics:
  - Layers processed:  6
  - Heads processed:   12
  - Tokens compacted:  133
  - Peak memory:       19.57 MB

Inference Impact:
  - Compaction is 0.15% of total inference time
==========================================
```

### Analysis

**If compaction > 5% of inference:**
- Compaction is a bottleneck
- GPU optimization warranted
- Focus on largest percentage items

**If compaction < 5% of inference:**
- Compaction is negligible
- Focus on inference optimization
- GGML CUDA/ROCm backend recommended

**If beta computation > 30% of compaction:**
- Matrix multiplication is bottleneck
- GPU matmul provides 10-50x speedup
- Implement GGML backend integration

## Optimization Workflow

### Step 1: Profile Current Performance

```bash
./kv-compact-adapter --model ... --perf > baseline.txt
```

### Step 2: Analyze Bottlenecks

```bash
# Check if GPU is available
./kv-compact-adapter --detect-gpu

# Check if compaction is significant
grep "Compaction is" baseline.txt
```

### Step 3: Apply Recommendations

```bash
# If GPU inference recommended:
cmake -DGGML_CUDA=ON ..
make

# If GPU compaction recommended:
# Implement GGML backend integration
# See skills/llama-gpu-integration.md
```

### Step 4: Re-profile

```bash
./kv-compact-adapter --model ... --n-gpu-layers 24 --perf > optimized.txt
diff baseline.txt optimized.txt
```

## RICE-Based Decision Making

The profiling infrastructure provides data for RICE scoring:

| Operation | % of Compaction | GPU Speedup | Impact | Priority |
|-----------|----------------|-------------|--------|----------|
| Beta matmul | 51.5% | 20x | High | 1 |
| Key selection | 35.6% | 5x | Medium | 2 |
| Value refitting | 11.7% | 3x | Low | 3 |

**Decision**: If compaction > 5% of inference, prioritize beta matmul optimization.

## Integration with Existing Code

### Minimal Changes Required

```cpp
// Before:
auto result = compact_head_highest_attn(K, V, Q_ref, T, n_q, d_k, d_v, t);

// After (with profiling):
#ifdef KV_COMPACT_ENABLE_PROFILING
KVCompactPerfMetrics perf_metrics;
auto result = compact_head_highest_attn_profiled(
    K, V, Q_ref, T, n_q, d_k, d_v, t, 2, KEY_SELECT_MAX_ATTN, BETA_FIT_CLOSED_FORM, &perf_metrics);
perf_metrics.print_summary();
#else
auto result = compact_head_highest_attn(K, V, Q_ref, T, n_q, d_k, d_v, t);
#endif
```

## Advanced Usage

### Custom Performance Tracking

```cpp
// Track specific operations
PerfTimer timer("custom_operation", perf_metrics.custom_ms);
// ... do work ...
// timer automatically records elapsed time on destruction
```

### GPU Memory Tracking

```cpp
// In GPU-accelerated code
size_t gpu_memory_before = get_gpu_memory_used();
// ... GPU operations ...
size_t gpu_memory_after = get_gpu_memory_used();
perf_metrics.gpu_memory_bytes = gpu_memory_after - gpu_memory_before;
```

### Batch Processing

```cpp
// Profile multiple compactions
std::vector<KVCompactPerfMetrics> all_metrics;
for (int layer = 0; layer < num_layers; layer++) {
    KVCompactPerfMetrics metrics;
    compact_head_highest_attn_profiled(..., &metrics);
    all_metrics.push_back(metrics);
}

// Aggregate statistics
double avg_time = 0;
for (const auto& m : all_metrics) {
    avg_time += m.total_compaction_ms;
}
avg_time /= all_metrics.size();
printf("Average compaction time: %.2f ms\n", avg_time);
```

## Troubleshooting

### Issue: Profiling overhead is too high

**Solution**: Disable profiling in production builds
```cpp
// Only enable for development
#ifdef DEBUG
#define KV_COMPACT_ENABLE_PROFILING
#endif
```

### Issue: GPU detection fails

**Solution**: Check GGML backend configuration
```bash
# Verify CUDA/ROCm is enabled
cmake -LA | grep GGML

# Rebuild with GPU support
cmake -DGGML_CUDA=ON ..
```

### Issue: Recommendations don't match performance

**Solution**: Verify inference time is being measured
```cpp
// Add to your code:
perf_metrics.percentage_of_inference =
    (perf_metrics.total_compaction_ms / total_inference_time_ms) * 100.0;
```

## Next Steps

1. **Build the demo**: `make profiling-demo`
2. **Run the demo**: `./profiling-demo`
3. **Review recommendations**: Check optimization priorities
4. **Implement high-priority items**: Follow RICE scoring

## References

- **RICE Methodology**: `docs/rice-prioritization.md`
- **GPU Integration**: `skills/llama-gpu-integration.md`
- **Performance Guide**: `skills/gpu-optimization.md`
- **CUDA Agent**: `agents/cuda-optimizer.md`
- **ROCm Agent**: `agents/rocm-specialist.md`

## Status

✅ **Completed** (RICE: 1,520)
- Profiling infrastructure implemented
- GPU detection working
- Recommendations engine functional
- Demo program available

**Next Priority**: Sprint 1, Item 2 - Auto-detection logic (RICE: 800)
