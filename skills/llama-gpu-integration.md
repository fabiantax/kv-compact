---
name: llama-gpu-integration
description: Leveraging llama.cpp's GPU backends for kv-compact acceleration
type: user
---

# llama.cpp GPU Backend Integration for kv-compact

This skill explains how to leverage llama.cpp's existing GPU infrastructure (CUDA, ROCm, Metal, Vulkan) to accelerate KV cache compaction.

## llama.cpp GPU Backends Overview

### Available Backends
From [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp):

| Backend | Target | Status | Performance |
|---------|--------|--------|-------------|
| **CUDA** | NVIDIA GPU | ✅ Mature | Best on NVIDIA |
| **HIP** | AMD GPU | ✅ Supported | Good on AMD |
| **Metal** | Apple Silicon | ✅ Optimized | Excellent on M1/M2/M3 |
| **Vulkan** | Cross-platform GPU | ✅ Available | Moderate |
| **SYCL** | Intel GPU | ✅ Experimental | Growing support |

### Key CUDA Kernels in llama.cpp

Located in `ggml/src/ggml-cuda/`:

1. **Flash Attention** (`fattn.cu`, `fattn-tile.cu`, `fattn-mma-f16.cuh`)
   - Optimized attention computation
   - Multiple variants: tile-based, MMA, WMMA
   - Supports FP16 and BF16

2. **Matrix Operations** (`mmq.cu`, `mmvq.cu`)
   - Quantized matrix multiplication
   - Optimized for KV cache operations
   - Supports various quantization formats (Q4_0, Q4_K, Q5_K, Q6_K, Q8_0)

3. **KV Cache Operations** (`getrows.cu`, `cpy.cu`)
   - Efficient KV cache reads/writes
   - Async memory transfers
   - Cache-aware memory access patterns

4. **Specialized Operations**
   - `rope.cu`: Rotary positional embeddings
   - `norm.cu`: Layer normalization
   - `softmax`: Via fused kernels
   - `gated_delta_net.cu`: For Qwen 3.5 DeltaNet layers

## Building kv-compact with GPU Support

### NVIDIA (CUDA)

```bash
# Prerequisites
# - CUDA Toolkit 11.0+ or 12.0+
# - Visual Studio 2022 (Windows) or GCC/Clang (Linux)

cd /d/Projects/kv-compact/kv-compact
rm -rf build
mkdir build && cd build

# Configure with CUDA
cmake .. -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release

# Verify GPU backend is enabled
./build/Release/llama-kv-compact.exe --help | grep -i cuda
```

**Expected Output:**
```
llama_kv_cache:        GPU KV buffer size =   768.00 MiB
ggml_cuda_init: CUDA initialized
GGML_CUDA: version 12.0.0, CUDA Runtime 12.0.0
```

### AMD (ROCm/HIP)

```bash
# Prerequisites
# - ROCm 5.0+ toolkit
# - HIP compiler (hipcc)

cd /d/Projects/kv-compact/kv-compact
rm -rf build
mkdir build && cd build

# Configure with ROCm
cmake .. -DGGML_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release
```

### Apple Silicon (Metal)

```bash
# Prerequisites
# - Xcode Command Line Tools
# - macOS 11+

cd /d/Projects/kv-compact/kv-compact
rm -rf build
mkdir build && cd build

# Configure with Metal
cmake .. -DGGML_METAL=ON -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build . --config Release
```

## GPU Acceleration Strategy for kv-compact

### Phase 1: Leverage Existing llama.cpp GPU Operations

The compaction algorithm can benefit from GPU acceleration even without custom kernels:

#### 1. Attention Score Computation (Q @ K^T)
```cpp
// Current: CPU matrix multiplication
// kv-compact-math.h:55-65

// With GPU: Use llama.cpp's built-in matmul
// This automatically uses cuBLAS/rocBLAS when GPU backend is enabled

struct ggml_tensor * Q = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
struct ggml_tensor * K = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
struct ggml_tensor * scores = ggml_mul_mat(Q, K); // GPU-accelerated!
```

**Benefit**: 10-50x speedup for attention computation

#### 2. Value Aggregation (A @ V)
```cpp
// Weighted sum of values using attention weights
// This is also a matrix multiplication, GPU-accelerated

struct ggml_tensor * A = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
struct ggml_tensor * V = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
struct ggml_tensor * output = ggml_mul_mat(A, V); // GPU!
```

#### 3. KV Cache Operations
```cpp
// Reading KV cache from GPU memory
// Writing compacted KV cache back to GPU

// llama.cpp handles this automatically when GPU layers are enabled
// Just use the llama_context API
```

### Phase 2: GPU Offloading Configuration

```cpp
// In llama-kv-compact.cpp or your main file

// Enable GPU offloading for KV cache
common_params params;
params.n_gpu_layers = -1;  // -1 = all layers to GPU

// For GTX 1050 Ti (4GB VRAM):
params.n_gpu_layers = 24;   // Offload all 24 layers

// For larger models or limited VRAM:
params.n_gpu_layers = 12;   // Offload first 12 layers
```

**VRAM Requirements for Qwen3.5-0.8B:**
- Model weights (Q4_K_M): ~500 MB
- KV cache (8K context): ~100 MB
- Working memory: ~200 MB
- **Total**: ~800 MB per sequence

**With 4GB VRAM (GTX 1050 Ti):**
- Can fit 4-5 sequences with 8K context
- Or 1 sequence with 32K context

### Phase 3: Custom CUDA Kernels (Optional)

For maximum performance, implement custom kernels:

#### 1. Key Selection Kernel
```cuda
__global__ void key_selection_kernel(
    const float* keys,        // [n_tokens, head_dim]
    const float* importance,  // [n_tokens]
    int* selected_indices,    // [target_tokens]
    int n_tokens,
    int head_dim,
    int target_tokens
) {
    // Parallel top-k selection
    // Uses shared memory for reduction
    // Optimizes for memory coalescing
}
```

#### 2. Closed-Form Beta Kernel
```cuda
__global__ void closed_form_beta_kernel(
    const float* attention_scores,  // [n_queries, n_tokens]
    const float* keys,              // [n_tokens, head_dim]
    float* beta,                    // [target_tokens]
    int n_queries,
    int n_tokens,
    int head_dim
) {
    // Vectorized closed-form computation
    // Leverages GPU's parallel reduction
    // Minimal benefit (CPU already fast: 1.4ms)
}
```

**However**: Closed-form beta is already very fast (1.4 ms on CPU). GPU kernels may not provide significant benefit due to kernel launch overhead.

## Performance Expectations

### CPU-Only (Current)
```
Compaction:     1.4 ms  (0.03%)
Inference:      28 tg/s
Memory:         5x compression
```

### With GPU (Expected)
```
Compaction:     0.5-1 ms   (marginal gain)
Inference:      80-150 tg/s (3-5x speedup)
Memory:         5x compression + larger batches
```

### Key Insight
**The compaction algorithm is already optimal** (1.4 ms). The major benefit from GPU is:
1. **Faster inference** (3-5x)
2. **Larger batch sizes** (4-8x more sequences)
3. **Longer contexts** (2-4x within same VRAM)

## Integration Steps

### 1. Update CMakeLists.txt
```cmake
# Add to kv-compact's CMakeLists.txt

option(GGML_CUDA "Enable CUDA backend" OFF)
option(GGML_HIPBLAS "Enable HIP/ROCm backend" OFF)

# Automatically detect and enable GPU
if(EXISTS "/usr/local/cuda")
    set(GGML_CUDA ON)
endif()
```

### 2. Add GPU Backend Detection
```cpp
// In kv-compact.cpp

#include "ggml.h"

bool has_cuda = ggml_cuda_is_enabled();
bool has_rocm = ggml_hip_is_enabled();

if (has_cuda) {
    LOG("GPU: CUDA backend enabled\n");
} else if (has_rocm) {
    LOG("GPU: ROCm/HIP backend enabled\n");
} else {
    LOG("GPU: CPU-only mode\n");
}
```

### 3. Use GGML Tensors for Compaction
```cpp
// Replace naive matmul with GGML operations

// Before (CPU):
mat_mul_ABt(A, B, C, m, n, k);

// After (GPU-accelerated):
struct ggml_context * ctx = ggml_init_context(params);
struct ggml_tensor * A_tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
struct ggml_tensor * B_tensor = ggml_new_tensor(ctx, GGML_TYPE_F32, 2, ne);
ggml_set_tensor(A_tensor, A);
ggml_set_tensor(B_tensor, B);

struct ggml_tensor * C_tensor = ggml_mul_mat(A_tensor, B_tensor);
ggml_build_forward_expand(&gf, C_tensor);
ggml_graph_compute(ctx, gf);

// Result is automatically on GPU if backend enabled!
```

## Benchmarking GPU Performance

### Test Script
```bash
# Run benchmarks with different GPU configurations

# CPU-only (baseline)
./build/Release/llama-kv-compact \
  -m models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -c 8192 -n 512 \
  --perf

# GPU: All layers
./build/Release/llama-kv-compact \
  -m models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -c 8192 -n 512 \
  -ngl 24 \
  --perf

# GPU: Partial layers (for limited VRAM)
./build/Release/llama-kv-compact \
  -m models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -c 8192 -n 512 \
  -ngl 12 \
  --perf

# Parallel inference (2 sequences)
./build/Release/llama-kv-compact \
  -m models/Qwen3.5-0.8B-Q4_K_M.gguf \
  -c 8192 -n 256 \
  -ngl 24 \
  -np 2 \
  --perf
```

### Expected Results
```
Context: 8192 tokens, Generation: 512 tokens

CPU-only:
  Full cache:     22.77 tg/s
  Compacted:      24.73 tg/s (1.09x speedup)

GPU (all layers):
  Full cache:     80-120 tg/s (3.5-5.3x speedup)
  Compacted:      90-130 tg/s (3.9-5.7x speedup)
```

## Troubleshooting

### CUDA Out of Memory
```
Error: CUDA out of memory
```

**Solutions:**
1. Reduce batch size: `-np 1`
2. Reduce context length: `-c 4096`
3. Reduce GPU layers: `-ngl 12`
4. Use smaller model: Q4_K_M instead of Q8_0

### Poor GPU Utilization
```
nvidia-smi shows 0% GPU utilization
```

**Solutions:**
1. Verify CUDA is enabled: Check for "CUDA initialized" in output
2. Increase batch size: `-np 4`
3. Increase context length: More tokens = more GPU work
4. Check GPU layers: `-ngl -1` (all layers)

### ROCm Compilation Errors
```
error: HIP not found
```

**Solutions:**
1. Install ROCm toolkit: https://rocm.docs.amd.com/
2. Set HIP path: `export HIP_PATH=/opt/rocm`
3. Use `rocm-dev` package manager

## Resources

- **llama.cpp CUDA Guide**: https://github.com/ggerganov/llama.cpp#cublas
- **GGML CUDA Source**: https://github.com/ggerganov/llama.cpp/tree/master/ggml/src/ggml-cuda
- **ROCm Documentation**: https://rocm.docs.amd.com/
- **CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/

## When to Use This Skill

Use this skill when:
- Building kv-compact with GPU support
- Configuring GPU offloading
- Troubleshooting GPU issues
- Benchmarking GPU vs CPU performance
- Optimizing for specific GPU architectures

## Related Skills & Agents

- `gpu-optimization.md`: General GPU optimization strategies
- `cuda-optimizer` agent: Deep CUDA kernel optimization
- `rocm-specialist` agent: AMD GPU/ROCm expertise
