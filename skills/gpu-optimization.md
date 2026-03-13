---
name: gpu-optimization
description: GPU acceleration and optimization strategies for LLM inference and KV cache compaction
type: user
---

# GPU Optimization for KV Cache Compaction

This skill provides comprehensive guidance for leveraging GPU acceleration (CUDA/ROCm) in the kv-compact project.

## Quick Start

### Detect GPU Hardware
```bash
# NVIDIA GPU
nvidia-smi

# AMD GPU
rocm-smi
```

### Check llama.cpp GPU Support
```bash
# Check available backends in llama.cpp
cd llama.cpp
cmake -B build -DGGML_CUDA=ON   # NVIDIA
cmake -B build -DGGML_HIPBLAS=ON # AMD/ROCm
```

### Build with GPU Support
```bash
# NVIDIA (CUDA)
cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release

# AMD (ROCm)
cmake -B build -DGGML_HIPBLAS=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## GPU Backend Options

### CUDA (NVIDIA)
- **Supported**: Most widely used, mature ecosystem
- **Tools**: cuBLAS, Nsight Compute, Nsight Systems
- **Performance**: Best on NVIDIA hardware
- **Requirements**: CUDA Toolkit 11.0+ or 12.0+

### ROCm (AMD)
- **Supported**: Growing support, HIP compatibility layer
- **Tools**: rocBLAS, rocprofiler, rocBT
- **Performance**: Good on AMD GPUs (RDNA2/3, CDNA)
- **Requirements**: ROCm 5.0+ toolkit

### Alternative Backends
- **Vulkan**: Cross-platform, limited performance
- **Metal**: Apple Silicon (M1/M2/M3)
- **SYCL**: Intel GPUs (oneAPI)

## Optimization Strategies

### 1. Matrix Operations (Highest Impact)
```cpp
// Current: CPU O(n³) matrix multiply
// kv-compact-math.h:55-65

// Optimized: GPU cuBLAS/rocBLAS
#include <cublas_v2.h>
// or
#include <hipblas/hipblas.h>

// Replace mat_mul_ABt with:
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
           m, n, k, &alpha, A, k, B, k, &beta, C, n);
```

**Expected Speedup**: 10-50x for matrix operations

### 2. Attention Computation
```cuda
// CUDA kernel for attention scores Q @ K^T
__global__ void attention_scores_kernel(
    const float* queries,  // [n_queries, head_dim]
    const float* keys,     // [n_tokens, head_dim]
    float* scores,        // [n_queries, n_tokens]
    int n_queries,
    int n_tokens,
    int head_dim
) {
    // Use shared memory for tiles
    // Optimize for memory coalescing
    // Handle large contexts with tiling
}
```

**Expected Speedup**: 5-20x for attention computation

### 3. Closed-Form Beta (Already Fast)
- Current: 1.4 ms (CPU)
- GPU: Marginal benefit (already fast)
- **Recommendation**: Keep on CPU

### 4. KV Cache Read/Write
```cuda
// Optimize KV cache memory access
// Use texture memory for read-heavy workloads
// Employ async memcpy for overlap
```

**Expected Speedup**: 2-5x for cache operations

## Profiling Tools

### NVIDIA (CUDA)
```bash
# Nsight Compute - kernel profiling
ncu --set full --target-processes-only ./llama-kv-compact

# Nsight Systems - timeline analysis
nsys profile --stats=true ./llama-kv-compact

# Visual profiler
nvprof --print-gpu-trace ./llama-kv-compact
```

### AMD (ROCm)
```bash
# rocprofiler - kernel profiling
rocprof --base-analysis gpucorrelation ./llama-kv-compact

# rocBT - backtrace profiler
rocm-profiler --basenames ./llama-kv-compact
```

## Memory Optimization

### GPU Memory Hierarchy
1. **Registers**: Fastest, per-thread
2. **Shared Memory**: Fast, per-block
3. **L1 Cache**: Per-SM
4. **L2 Cache**: Global
5. **Global Memory**: Slowest, largest

### Optimization Techniques
- **Shared Memory**: Cache frequently accessed data
- **Memory Coalescing**: Align memory accesses
- **Tiling**: Break large matrices into tiles
- **Prefetching**: Overlap memory transfers with computation

## Multi-GPU Scaling

### NCCL (NVIDIA)
```cpp
#include <nccl.h>
// All-reduce for multi-GPU attention
ncclAllReduce(..., ncclComm, stream);
```

### RCCL (AMD/ROCm)
```cpp
#include <rccl.h>
// ROCm collective communication
```

### Scaling Strategy
1. **Data Parallel**: Different sequences per GPU
2. **Tensor Parallel**: Split attention heads across GPUs
3. **Pipeline Parallel**: Different layers per GPU

## Performance Targets

### Current (CPU-only)
- Compaction: 1.4 ms ✓
- Inference: 28 tg/s
- Memory: 5x compression

### With GPU (Expected)
- Compaction: 0.5-1 ms (marginal gain)
- Inference: 80-150 tg/s (3-5x speedup)
- Memory: 5x compression + larger batches

## Troubleshooting

### CUDA Issues
- **Error: "CUDA out of memory"**: Reduce batch size or context length
- **Error: "nvcc not found"**: Add CUDA bin to PATH
- **Poor performance**: Check GPU utilization with nvidia-smi

### ROCm Issues
- **Error: "HIP not found"**: Install ROCm toolkit
- **Compilation errors**: Check HIP version compatibility
- **Poor performance**: Verify ROCm drivers are installed

## Best Practices

1. **Profile First**: Use profilers to identify actual bottlenecks
2. **Start with Libraries**: Use cuBLAS/rocBLAS before custom kernels
3. **Validate Correctness**: Compare GPU output with CPU reference
4. **Optimize Incrementally**: Make one change at a time
5. **Monitor GPU Utilization**: Aim for >80% utilization

## Resources

- **llama.cpp GPU guide**: https://github.com/ggerganov/llama.cpp#cublas
- **CUDA Best Practices**: https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/
- **ROCm Documentation**: https://rocm.docs.amd.com/
- **kv-compact paper**: https://arxiv.org/abs/2602.16284

## When to Use This Skill

Use this skill when:
- Setting up GPU acceleration for kv-compact
- Optimizing CUDA/ROCm kernels
- Profiling GPU performance
- Troubleshooting GPU issues
- Comparing different GPU backends
- Scaling to multiple GPUs

## Related Skills

- `cuda-optimizer` agent: Deep CUDA kernel optimization
- `rocm-specialist` agent: AMD GPU/ROCm expertise
- `performance-profiling`: General profiling strategies
