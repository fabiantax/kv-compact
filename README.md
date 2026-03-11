# kv-compact

Fast KV Cache Compaction via Attention Matching — a C++ implementation of [arXiv:2602.16284](https://arxiv.org/abs/2602.16284).

Compresses transformer KV caches by 50x with minimal quality loss using a closed-form 3-step algorithm:

1. **Key selection** — pick top-t keys by cumulative attention score
2. **NNLS bias fitting** — solve for attention mass biases (β) to match original attention distribution
3. **Least squares value refitting** — compute optimal compacted values (C_v) via ridge regression

## Project structure

```
include/kv-compact-math.h      # Header-only math library (zero dependencies)
include/kv-compact-state.h     # State parser/writer (F32, F16, transposed V)
include/kv-compact-api.h       # Public C API
src/kv-compact-api.cpp         # API implementation (full pipeline)
src/kv-compact.cpp             # CLI demo tool (requires llama.cpp)
tests/test-kv-compact-math.cpp # 31 unit tests
tests/test-kv-compact-e2e.cpp  # E2E integration test
docs/algorithms.md             # Detailed algorithm reference with paper §refs
```

## Quick start — tests only (no dependencies)

```bash
mkdir build && cd build
cmake .. -DKV_COMPACT_BUILD_TOOL=OFF
cmake --build .
./test-kv-compact-math
```

## Full build with llama.cpp

### Option A: Point to local llama.cpp checkout

```bash
cmake .. -DLLAMA_CPP_DIR=/path/to/llama.cpp
cmake --build .
```

### Option B: Auto-fetch from GitHub

```bash
cmake ..
cmake --build .
```

### Usage

```bash
./llama-kv-compact -m model.gguf -p "your context..." --compact-ratio 0.2
```

## Paper

> **Fast KV Compaction via Attention Matching**
> Zweiger et al., 2026 — [arXiv:2602.16284](https://arxiv.org/abs/2602.16284)
>
> Achieves 50x KV cache compression with closed-form solutions (no gradient descent).
> Value refitting reduces MSE by ~4,000,000x compared to naive token eviction.

## Test results

- 31 tests covering matrix ops, softmax, NNLS, least squares, and full pipeline
- Value refitting: ~4M× MSE improvement over token eviction at 4x compression
- Cosine similarity: 0.999999 at 50% compression

## Development timeline & roadmap

```mermaid
timeline
    title kv-compact — Development & Roadmap

    section Completed
        Core Algorithm
            : Header-only math library (NNLS, least squares, softmax)
            : Key selection via max softmax attention (paper §3.1)
            : NNLS bias fitting for attention mass preservation (paper §3.2)
            : Regularized least-squares value refitting C_v (paper §3.3)
        State I/O & Integration
            : llama.cpp binary state parser/writer (F32 + F16)
            : Transposed V storage support
            : IMROPE / M-RoPE position handling
            : kv_compact_sequence() C API
            : llama-server patch auto-apply
        All Heads & Layers
            : Per-head NNLS + C_v via refit_head_values()
            : compact_layer_all_heads() with shared key selection
            : Full all-layer pipeline in kv_compact_sequence()
        Hybrid Model Support
            : Recurrent state save/restore for SSM+attention models
            : Hybrid compaction path (Qwen 3.5 / Mamba-based)
        Robustness
            : Bandwidth-aware auto-ratio (kv_compact_suggest_ratio)
            : Failure recovery — restore original state on error
            : Portable sed for BSD/GNU in apply.sh
            : 31 unit tests + E2E integration test

    section Not Yet Implemented
        Beta Injection (paper §4.1)
            : Inject attention biases during llama_decode
            : Requires ggml_flash_attn_ext bias hook
        Better Reference Queries (paper §7.2)
            : True repeat-prefill or learned query sets
            : Currently uses trailing K vectors as proxies
        Quantized KV Types
            : Support Q8_0, Q4_0 etc.
            : Needs dequantize → compact → requantize path
        OMP Key Selection (paper §5.2)
            : Orthogonal Matching Pursuit — greedy residual reduction
            : Higher quality than max-attention, ~100x slower
        Non-Uniform Per-Head Budgets (paper §6.2)
            : Sensitivity-aware budget allocation across heads
            : Currently all heads share same target t
        Iterative Compaction (paper §6.1)
            : Built-in scheduling for repeated compression
            : Paper shows 6 consecutive compressions without quality loss
        Direct Key Optimization (paper §5.5)
            : Allow C_k to be arbitrary vectors, not just a subset of K
            : Non-convex optimization, potentially higher quality
```

See [`docs/algorithms.md` §12](docs/algorithms.md#12-limitations--future-work) for detailed status of each item with code cross-references.
