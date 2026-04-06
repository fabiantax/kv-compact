# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

kv-compact implements "Fast KV Compaction via Attention Matching" (arXiv:2602.16284) — compresses transformer KV caches up to 50x with minimal quality loss. C/C++ library with optional ROCm/HIP GPU acceleration. No model retraining required.

## Build Commands

```bash
# Standalone tests only (no llama.cpp)
mkdir -p build && cd build
cmake .. -DKV_COMPACT_BUILD_TOOL=OFF
cmake --build . -j$(nproc)
./test-kv-compact-math    # 12 math unit tests
./test-kv-compact-api     # 34 API integration tests

# Full build with local llama.cpp
cmake .. -DLLAMA_CPP_DIR=/path/to/llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Full build with auto-fetch llama.cpp
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# ROCm/HIP GPU acceleration (AMD gfx1151 / RDNA 3.5)
cmake .. -DKV_COMPACT_HIP=ON
cmake --build . -j$(nproc)

# Build single target
cmake --build . --target test-kv-compact-api
cmake --build . --target bench-kv-compact-model
```

## Architecture

### Three-layer dependency structure

```
kv-compact-math (header-only, zero deps)
    ↑
kv-compact-api (static lib, links math + optional OpenMP + optional HIP)
    ↑
CLI tools / E2E tests (require llama.cpp)
```

- `include/kv-compact-math.h` — Pure CPU float32 linear algebra. Matmul, softmax, NNLS (Lawson-Hanson + PGD), least squares, key selection, beta directions. ~1200 lines, header-only.
- `include/kv-compact-api.h` — C API types and function declarations. `kv_compact()`, `kv_compact_multi_round()`, `kv_compact_quantized()`. Layer filter callbacks for hybrid models. `extern "C"` linkage.
- `include/kv-compact-state.h` — Parses llama.cpp's binary KV state format into per-layer K/V arrays (float32). Handles F32/F16/Q8_0/Q4_0/Q4_1 quantized types, M-RoPE/IMROPE. Writer builds compacted state buffer.
- `include/kv-compact-accel.h` — GPU acceleration interface. When `KV_COMPACT_HIP` is defined: extern HIP functions. Otherwise: inline CPU stubs returning -1.
- `src/kv-compact-api.cpp` — C API implementation. Chunked compaction with OpenMP, GPU dispatch, quality metrics.
- `src/kv-compact-hip.hip` — ROCm kernels. rocBLAS strided-batched GEMM for scoring, custom HIP kernels as fallback. Unified memory (`hipMallocManaged`), persistent buffer pool.
- `src/kv-compact.cpp` — CLI tool. Full pipeline: prefill → save state → parse → per-layer compact → write back → generate. Handles hybrid models via layer filters.

### Compaction pipeline (per layer)

1. **Key selection** — Score all positions using attention to reference queries, select top-k by importance
2. **Beta (NNLS)** — Skipped by default (`skip_beta=1`). LS value refitting alone achieves equal/better quality, 6.75x faster.
3. **Value refit (LS)** — Reconstruct compacted values via least squares to match original attention output

### Hybrid model support

Models like Qwen 3.5 mix attention layers with SSM/Mamba layers. Only attention layers have compactable KV state.

Detection/filtering priority: explicit layer list (`--attention-layers`) > periodic interval (`--attention-interval`) > auto-detect from state geometry > compact all.

Auto-detection works by checking `is_compactable_layer()` — layers where K/V dimensions don't divide evenly by `n_head_kv` are non-attention (SSM).

## Key patterns

- **Beta skipped by default** — The paper's NNLS bias fitting adds cost without quality gain when using K-as-Q_ref. LS value refitting is sufficient.
- **Chunked compaction** — Auto-splits large contexts into chunks of ~256 tokens each, compacted independently in parallel via OpenMP. Bounds LS memory to O(256²), enables 1M+ token contexts.
- **`KV_COMPACT_ACCEL_IMPL`** — Pattern for exactly-one-TU definition of CPU stub functions when HIP is disabled.
- **State format** — Directly reads/writes llama.cpp's binary state format. No dependency on llama.cpp internals for the standalone library.

## Test data patterns

Tests use deterministic sine/cosine generators (`gen_data`, `gen_structured_data`, `gen_spiky_data`). Validation via cosine similarity thresholds, MSE bounds, and argmax agreement rates. Approximate equality uses `approx_eq(a, b, tol=1e-5)`.

## ROCm target hardware

AMD Radeon 8060S (RDNA 3.5, gfx1151), ~40 CUs, wavefront size 32, 128GB unified LPDDR5X. Use `hipMallocManaged()` for zero-copy on APU. GPU dispatch only when total elements > 32K or batch ≥ 4.

## Benchmark targets

- `bench-kv-compact` — Throughput at various sizes (no model needed)
- `bench-kv-compact-quality` — Cosine sim, MSE, agreement metrics (synthetic data)
- `bench-kv-compact-model` — Real-model PPL, KL divergence, generation quality, tok/s. Requires GGUF model: `-m model.gguf -c 4096 -ngl 99`
- `bench-quant-throughput` — Per-stage timing for quantized KV
- `bench-throughput-large` — 4k–100k token contexts
