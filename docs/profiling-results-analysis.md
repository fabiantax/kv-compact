# Profiling Results & Analysis

## Test Results (2026-03-13)

### Configuration
- **Model**: Qwen3.5-0.8B-Q4_K_M
- **Hardware**: CPU-only (no GPU detected)
- **Test**: T=8192 → t=133 tokens (61.6x compression)
- **Dimensions**: d_k=256, d_v=256, n_q=33

### Performance Breakdown

```
Total compaction time: 120.37 ms
  - Key selection:     42.13 ms (35.0%)
  - Beta computation:  62.59 ms (52.0%) ← BOTTLENECK
  - Value refitting:   13.24 ms (11.0%)
```

### Matrix Operations

```
matmul A^T @ B:    50.07 ms (41.6% of total) ← PRIMARY TARGET
Attention scores:  25.28 ms (21.0% of total)
Value aggregation:  6.62 ms (5.5% of total)
Total matmuls:     4 operations
```

### Memory Usage

```
Input K+V memory:    16.00 MB
Output K+V memory:   0.26 MB
Working memory:      1.05 MB
Memory saved:        15.74 MB (98.4% compression)
```

## Key Findings

### 1. Beta Computation is the Bottleneck

**52% of compaction time** is spent in beta computation (NNLS or closed-form).

**Why it matters:**
- Contains the largest matrix multiplication (A^T @ B)
- O(n³) complexity with token count
- Scales poorly with sequence length

**GPU Speedup Potential:**
- cuBLAS/rocBLAS: **10-50x** speedup
- Expected compaction time: **2-6 ms** (vs 62.59 ms currently)
- Net impact: Saves 56 ms per compaction

### 2. Matrix Multiplication Dominates

**42% of total compaction time** is spent in matrix operations.

**Breakdown:**
- matmul A^T @ B (beta): 50.07 ms
- Attention scores (Q @ K^T): 25.28 ms
- Value aggregation (A @ V): 6.62 ms

**GPU Speedup Potential:**
- Combined speedup: **10-30x** on all matmuls
- Expected time: **5-8 ms** (vs 81.97 ms currently)
- Net impact: Saves 74 ms per compaction

### 3. No GPU Detected

**Current Status:** CPU-only execution

**Implications:**
- All matmuls run on CPU (naive O(n³) implementation)
- No cuBLAS/rocBLAS acceleration
- Missing GGML GPU backend

**GPU Detection Results:**
```
CUDA available:  No
ROCm available:  No
Metal available: No
Device count:    0
```

## RICE-Based Recommendations

### Priority 1: GPU Inference (RICE: 486)

**Recommendation:** Enable GGML CUDA/ROCm backend for llama.cpp inference

**Potential Impact:**
- **3-7x overall speedup** (not just compaction!)
- Inference: 15,437ms → 2,000-5,000ms
- Compaction becomes: 120ms → 120ms (unchanged)
- **New percentage: 120ms / 2,000ms = 6%** (vs 0.15% currently)

**Why Highest Priority:**
1. **100x more impact** than GPU compaction alone
2. **Affects 99.85% of execution time** (inference)
3. **Easy to implement** (2-4 hours)
4. **Well-supported** in llama.cpp

**Implementation:**
```bash
# Rebuild llama.cpp with CUDA
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# Run with GPU layers
./llama-kv-compact -m model.gguf -c 8192 -n-gpu-layers 24 --perf
```

### Priority 2: GPU Matrix Operations (RICE: 45)

**Recommendation:** Implement GGML backend for compaction matmuls

**Potential Impact:**
- **10-50x speedup** on matmul operations
- Compaction: 120ms → 5-12ms
- **Only worth it AFTER GPU inference** (when compaction is >5% of total)

**Why Secondary Priority:**
1. Only affects 0.15% of current execution time
2. High effort (1-2 weeks implementation)
3. Uncertain real-world benefit (memory transfer overhead)
4. **Precondition:** GPU inference must be enabled first

**Implementation:**
```cpp
// Replace naive matmul with GGML backend
#include <ggml.h>

static void mat_mul_AtB_ggml(const float* A, const float* B, float* C,
                              int m, int k, int n) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    // ... use GGML's optimized matmul
}
```

### Priority 3: Memory Transfer Optimization (RICE: TBD)

**Recommendation:** Implement batch processing or memory pooling

**Potential Impact:**
- **2x speedup** by reducing H2D/D2H overhead
- Only relevant after GPU compaction is implemented

**Preconditions:**
1. GPU inference enabled
2. GPU compaction implemented
3. Profiling shows memory transfer is bottleneck

## Decision Matrix

| Scenario | Current Time | GPU Inference | GPU Compaction | Total Time | Speedup |
|----------|--------------|---------------|----------------|------------|---------|
| **Baseline** | 15,557ms | - | - | 15,557ms | 1.0x |
| **GPU Inference Only** | 15,557ms | 2,000ms | - | 2,120ms | **7.3x** |
| **GPU Both** | 15,557ms | 2,000ms | 5ms | 2,005ms | **7.8x** |
| **GPU Compaction Only** | 15,557ms | - | 5ms | 15,442ms | **1.007x** |

**Conclusion:** GPU compaction alone provides **0.7% speedup**. GPU inference provides **730% speedup**.

## Action Items

### Immediate (This Week)

1. **Detect GPU Hardware**
   ```bash
   nvidia-smi  # Check NVIDIA GPU
   rocm-smi    # Check AMD GPU
   ```

2. **Rebuild with GPU Support**
   ```bash
   cd D:/Projects/kv-compact/kv-compact/llama.cpp
   cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
   cmake --build build --config Release
   ```

3. **Benchmark GPU Inference**
   ```bash
   ./llama-kv-compact -m models/Qwen3.5-0.8B-Q4_K_M.gguf \
     -c 8192 -n 512 -p "..." --n-gpu-layers 24 --perf
   ```

4. **Re-profile After GPU Inference**
   - Check if compaction is now >5% of total
   - If yes, proceed to GPU compaction
   - If no, GPU compaction not worth the effort

### Secondary (If GPU Inference Helps)

5. **Implement GGML Backend for Compaction**
   - Replace `mat_mul_AtB` with GGML call
   - Benchmark single layer
   - Measure H2D/D2H overhead

6. **Optimize Memory Transfers**
   - Batch multiple layers
   - Reuse GPU memory pools
   - Overlap transfers with computation

## Validation Criteria

**Success Metrics:**

1. **GPU Inference Enabled:**
   - ✓ `nvidia-smi` shows GPU utilization
   - ✓ `llama-kv-compact` output shows "CUDA initialized"
   - ✓ tg/s improves by 3-7x (32 → 100-230 tg/s)

2. **Compaction Re-profiled:**
   - ✓ Compaction % increases to 5-10% (from 0.15%)
   - ✓ Matmul operations are clear bottleneck
   - ✓ GPU compaction ROI justified

3. **GPU Compaction Implemented:**
   - ✓ Compaction time: 120ms → 5-12ms
   - ✓ Overall speedup: <2% (expected)
   - ✓ Memory overhead acceptable

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| CUDA not available on system | Medium | High | Fall back to CPU, use ROCm |
| GGML CUDA build fails | Low | Medium | Use pre-built llama.cpp binaries |
| Memory transfer overhead > compute | High | Low | Profile first, batch operations |
| Compaction still <5% after GPU inference | High | Low | Don't implement GPU compaction |

## Next Steps

1. **Confirm GPU Hardware:** Run `nvidia-smi` or `rocm-smi`
2. **Enable GPU Inference:** Rebuild llama.cpp with CUDA/ROCm
3. **Re-benchmark:** Compare before/after GPU inference
4. **Decision Point:** Implement GPU compaction only if >5% threshold met

**Timeline:**
- Day 1-2: GPU inference setup and benchmarking
- Day 3: Analyze results and make decision
- Day 4-5: Implement GPU compaction (if justified)

## References

- **Profiling Guide:** `docs/profiling-guide.md`
- **GPU Integration:** `skills/llama-gpu-integration.md`
- **CUDA Optimizer:** `agents/cuda-optimizer.md`
- **ROCm Specialist:** `agents/rocm-specialist.md`
