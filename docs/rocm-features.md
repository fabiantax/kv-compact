# ROCm Features for KV Cache Compaction

AMD ROCm (Radeon Open Compute) platform features relevant to accelerating the kv-compact pipeline.

**Target hardware:** AMD Ryzen AI 395 Pro Max — 8060S RDNA 3.5 iGPU (gfx1151), 128GB unified LPDDR5X

---

## 1. Math Libraries

### rocBLAS / hipBLAS — Dense Linear Algebra

| API | Use in kv-compact | Notes |
|-----|-------------------|-------|
| `rocblas_sgemm` | Q_ref @ K^T attention scoring | Core hotspot: O(n_q × T × d_k) per head |
| `rocblas_sgemm_strided_batched` | Multi-head attention scoring | Batch all heads in one call; stride = head_dim × seq_len |
| `rocblas_sgemv` | Beta injection (K += beta) | Per-head bias application |
| `rocblas_snrm2` / `rocblas_sdot` | Cosine similarity for diversity | Used in key selection scoring |

**Key advantage:** `rocblas_sgemm_strided_batched` can process all KV heads simultaneously — eliminates the current per-head loop.

### hipBLASLt — Lightweight GEMM with Extensions

| Feature | Use in kv-compact | Notes |
|---------|-------------------|-------|
| Grouped GEMM (`GroupedGemm`) | Multi-layer compaction | Different matrix sizes per layer in one dispatch |
| Epilogue fusion | Fused scale + softmax prep | Avoids separate scaling kernel |
| Auto-tuning (`hipblaslt-bench`) | Optimal kernel selection | Tune for specific (m,n,k) sizes in our pipeline |

**Key advantage:** Grouped GEMM handles heterogeneous matrix sizes across layers without padding.

### rocSOLVER / hipSOLVER — Linear Solvers

| API | Use in kv-compact | Notes |
|-----|-------------------|-------|
| `rocsolver_sgels` | Least-squares value refitting (C_v) | Solves min ‖AX - B‖₂ for refitted values |
| `rocsolver_sgels_batched` | Multi-head LS solve | All heads in one call |
| `rocsolver_sgeqrf` | QR factorization for LS | Building block for custom NNLS |
| `rocsolver_strtrs` | Triangular solve after QR | R \ (Q^T b) for LS solution |

**Key advantage:** Batched `sgels` can solve all per-head value refitting problems in parallel.

### rocSPARSE — Sparse Operations

| Feature | Use in kv-compact | Notes |
|---------|-------------------|-------|
| SpMV / SpMM | Attention with sparse keys | After compaction, attention matrix is sparse |
| CSR format | Compact key index storage | Selected indices map naturally to CSR |

**Relevance:** Lower priority — our matrices are small enough that dense ops dominate.

---

## 2. Memory Management

### Unified Memory (APU Zero-Copy)

| Feature | API | Benefit |
|---------|-----|---------|
| Managed allocation | `hipMallocManaged()` | Single pointer for CPU+GPU — no H2D/D2H copies |
| System allocator | `hipHostMalloc(flag=hipHostMallocMapped)` | Map existing host allocations to GPU |
| Coherent access | Automatic on APU | CPU and GPU see same physical LPDDR5X |
| Prefetch hint | `hipMemPrefetchAsync()` | Hint GPU to stage data in L2 before kernel launch |

**APU advantage:** On the Ryzen AI 395 Pro Max, CPU and GPU share 128GB LPDDR5X. `hipMallocManaged` has near-zero overhead — no PCIe transfers. This is our biggest hardware advantage.

### Memory Pools

| Feature | API | Benefit |
|---------|-----|---------|
| Stream-ordered pools | `hipMallocAsync()` / `hipFreeAsync()` | Amortize allocation overhead across compaction rounds |
| Memory pool reuse | `hipMemPoolSetAttribute()` | Pre-allocate workspace for repeated compactions |

**Use case:** Multi-round compaction reuses same buffer sizes — pool eliminates malloc overhead.

---

## 3. Execution Model

### HIP Streams & Concurrency

| Feature | Use in kv-compact | Notes |
|---------|-------------------|-------|
| Multiple streams | Per-layer compaction | Layers 3,7,11,... run on separate streams |
| Stream callbacks | Progress reporting | `hipStreamAddCallback()` for per-layer status |
| Events | Inter-stream sync | `hipEventRecord` / `hipStreamWaitEvent` for dependencies |
| Default stream | Simple single-layer | Current implementation uses default stream |

**Opportunity:** Launch 10 attention layers on 10 streams for full GPU utilization.

### HIP Graphs

| Feature | Use in kv-compact | Notes |
|---------|-------------------|-------|
| Graph capture | `hipStreamBeginCapture()` | Record full compaction pipeline as a graph |
| Graph replay | `hipGraphLaunch()` | Re-execute without re-recording for multi-round |
| Graph update | `hipGraphExecUpdate()` | Change buffer pointers between rounds |

**Key advantage:** For multi-round compaction, capture the 4-phase pipeline once, replay on each trigger. Eliminates kernel launch overhead (~5-10μs per launch × ~20 kernels = 100-200μs saved per round).

---

## 4. Profiling & Analysis

| Tool | Purpose | Command |
|------|---------|---------|
| `rocprof` | Kernel timing, HW counters | `rocprof --stats ./kv-compact-tool` |
| `rocprof --hip-trace` | HIP API tracing | Shows every hipMalloc, hipLaunch, hipSync |
| `omniperf` | Full GPU performance analysis | Occupancy, cache hit rates, memory bandwidth |
| `omnitrace` | Timeline trace (CPU+GPU) | Visualize CPU/GPU overlap, find idle gaps |
| `rocm-smi` | Runtime GPU monitoring | Memory usage, clock speeds, temperature |
| `roctx` markers | Custom profiling regions | `roctxRangePush("phase1_scoring")` in code |

**Profiling strategy:**
1. Use `rocprof --stats` for quick kernel timing
2. Use `omniperf` to identify bottlenecks (memory-bound vs compute-bound)
3. Use `omnitrace` to verify CPU/GPU overlap in multi-stream execution

---

## 5. Kernel Optimization for RDNA 3.5

### Architecture Parameters (gfx1151)

| Parameter | Value | Impact |
|-----------|-------|--------|
| Wavefront size | 32 (RDNA) | Half of CDNA's 64 — smaller work groups viable |
| Compute Units | ~40 CUs (8060S) | ~2560 shader cores |
| L2 cache | 4MB | Fits several attention matrices |
| Shared memory | 64KB per workgroup | Tiled GEMM tile size limited by this |
| Max occupancy | 16 waves/CU | Balance register pressure vs parallelism |
| Memory bandwidth | ~200 GB/s (LPDDR5X) | Shared with CPU — contention possible |

### Optimization Techniques

| Technique | Description | Applicable Phase |
|-----------|-------------|-----------------|
| **Wave32 mode** | RDNA native 32-wide SIMD | All kernels (current code assumes 64) |
| **Tiled GEMM** | Shared memory blocking | Phase 1: attention scoring |
| **Warp shuffle** | `__shfl_xor` for reductions | Phase 2: softmax, cosine similarity |
| **Vectorized loads** | `float4` coalesced reads | Phase 1: K/V matrix reads |
| **Persistent kernels** | Keep CUs busy across phases | Full pipeline |
| **Occupancy tuning** | Register vs shared memory tradeoff | Per-kernel tuning |

### Composable Kernel (CK) Library

| Feature | Use | Notes |
|---------|-----|-------|
| CK-Tile GEMM | Custom fused GEMM kernels | Fuse scale + GEMM + epilogue |
| Device-level API | Fine-grained kernel composition | Build attention scoring as single fused kernel |
| gfx1151 support | RDNA 3.5 backend | Supported since ROCm 6.x |

**Opportunity:** CK can fuse the entire attention scoring phase (Q_ref @ K^T / √d_k → softmax → score accumulation) into a single kernel, eliminating intermediate memory traffic.

---

## 6. Batched Small-Matrix Strategies

Our workload involves many small GEMMs (e.g., 64×128 × 128×1024 per head).

| Strategy | API | When to use |
|----------|-----|-------------|
| Strided batched GEMM | `rocblas_sgemm_strided_batched` | Same (m,n,k) across heads |
| Grouped GEMM | `hipBLASLt GroupedGemm` | Different sizes per head/layer |
| Kernel fusion | Custom HIP kernel | When GEMM + post-processing needed |
| CPU fallback | Direct C code | When matrices are tiny (<32×32) |

**Decision rule:** GPU wins when `m × n × k > ~32K` AND batch count ≥ 4. Below that, CPU may be faster due to launch overhead on APU.

---

## 7. Feature Mapping to Compaction Phases

| Phase | Operation | Current (CPU) | ROCm Acceleration |
|-------|-----------|--------------|-------------------|
| **1. Scoring** | Q_ref @ K^T / √d_k | Manual matmul loop | `rocblas_sgemm_strided_batched` |
| **1. Selection** | Top-k with diversity | Sequential scan | Custom HIP kernel + `rocprim::radix_sort` |
| **2. NNLS** | Active-set solver | Lawson-Hanson (CPU) | `rocsolver_sgels` + custom active-set on GPU |
| **3. Value refit** | min ‖AX-B‖₂ | Manual LS solve | `rocsolver_sgels_batched` |
| **4. Writeback** | Copy compacted K,V | memcpy | `hipMemcpyAsync` (trivial on APU) |

### Expected Speedups (APU, 64K context, 10 layers × 2 heads)

| Phase | CPU time (est.) | GPU time (est.) | Speedup |
|-------|----------------|----------------|---------|
| Scoring | 800ms | 80ms | 10x |
| Selection | 50ms | 20ms | 2.5x |
| NNLS | 200ms | 100ms | 2x |
| Value refit | 400ms | 40ms | 10x |
| **Total** | **1450ms** | **240ms** | **~6x** |

*Estimates for 64K tokens, 10 attention layers, 2 KV heads, d_k=256.*

---

## 8. Implementation Priority

Based on ROI and implementation complexity:

| Priority | Feature | Effort | Impact |
|----------|---------|--------|--------|
| **P0** | `rocblas_sgemm_strided_batched` for scoring | 1 week | 10x scoring speedup |
| **P0** | `rocsolver_sgels_batched` for value refit | 1 week | 10x refit speedup |
| **P1** | Multi-stream per-layer execution | 2 days | Better GPU utilization |
| **P1** | `hipMallocManaged` memory pool | 2 days | Eliminate alloc overhead |
| **P2** | HIP Graph capture for multi-round | 3 days | Faster repeated compaction |
| **P2** | Custom diversity selection kernel | 3 days | GPU-native key selection |
| **P3** | CK fused attention kernel | 2 weeks | Maximum throughput |
| **P3** | `rocprof` + `omniperf` integration | 1 week | Data-driven optimization |
| **P4** | `hipBLASLt` grouped GEMM | 1 week | Multi-layer dispatch |

---

## 9. ROCm Version Requirements

| Feature | Minimum ROCm | Notes |
|---------|-------------|-------|
| HIP basics | 5.0+ | `hipMallocManaged`, kernels |
| rocBLAS batched | 5.0+ | `sgemm_strided_batched` |
| rocSOLVER | 5.0+ | `sgels`, `sgels_batched` |
| hipBLASLt GroupedGemm | 6.0+ | Grouped GEMM extension |
| HIP Graphs | 5.3+ | `hipStreamBeginCapture` |
| gfx1151 support | 6.2+ | RDNA 3.5 ISA support |
| omniperf gfx1151 counters | 7.2+ | Performance counter collection |
| Composable Kernel | 6.0+ | CK-Tile for custom fused kernels |

**Recommended:** ROCm 6.4+ for full gfx1151 support with all libraries.

---

## 10. APU vs Discrete GPU Considerations

| Aspect | APU (Strix Halo) | Discrete GPU (MI300X) |
|--------|-------------------|----------------------|
| Memory model | Unified (zero-copy) | Separate (PCIe transfers) |
| Memory bandwidth | ~200 GB/s (shared) | ~5.3 TB/s (HBM3) |
| Compute | ~40 CUs | ~304 CUs |
| Best for | Latency-sensitive, small batches | Throughput, large batches |
| Our workload | Ideal (small matrices, no copies) | Overkill for compaction |

**Conclusion:** APU is the sweet spot for KV compaction — the zero-copy advantage outweighs the lower compute density because our matrices are small and memory-bandwidth-bound.
