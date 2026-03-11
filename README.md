# kv-compact

Fast KV Cache Compaction via Attention Matching — a C++ implementation of [arXiv:2602.16284](https://arxiv.org/abs/2602.16284).

Compresses transformer KV caches by 50x with minimal quality loss using a closed-form 3-step algorithm:

1. **Key selection** — pick top-t keys by cumulative attention score
2. **NNLS bias fitting** — solve for attention mass biases (β) to match original attention distribution
3. **Least squares value refitting** — compute optimal compacted values (C_v) via ridge regression

## Features

- **4 key selection modes**: Max attention, submodular (BumbleBee-inspired), token merging (ToMe-inspired), K-means centroids
- **2 beta fitting modes**: NNLS, Sinkhorn (entropic optimal transport)
- **Per-head sensitivity weighting** with Carathéodory-informed budget allocation
- **Alternating minimization** for joint beta/C_v optimization
- **Adapter abstraction** for GQA, MLA (DeepSeek-V3), and hybrid (Qwen 3.5) architectures
- **Full quantization support**: F32, F16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
- **Attention bias injection** via llama.cpp patch (`llama_memory_set_attn_bias()`)
- **69+ unit tests** covering math ops, adapters, round-trip quantization, and full pipeline

## Project structure

```
include/
  kv-compact-math.h          # Header-only math library (zero dependencies)
  kv-compact-adapter.h       # GQA/MLA/hybrid adapter abstraction
  kv-compact-state.h         # llama.cpp state buffer parser/writer
src/
  kv-compact.cpp              # CLI tool (requires llama.cpp)
tests/
  test-kv-compact-math.cpp    # 22+ math unit tests
  test-kv-compact-adapter.cpp # 20+ adapter tests
  test-kv-compact-e2e.cpp     # End-to-end integration tests
  bench-synthetic.cpp         # Synthetic benchmarks (up to 50x, T=4096)
docs/                         # Research notes and design docs
patches/
  attn-bias.patch             # llama.cpp patch for attention bias support
```

## Quick start — tests only (no dependencies)

```bash
mkdir build && cd build
cmake .. -DKV_COMPACT_BUILD_TOOL=OFF
cmake --build .
./test-kv-compact-math
./test-kv-compact-adapter
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

## Hybrid model support (Qwen 3.5)

Qwen 3.5 uses a Gated DeltaNet + full attention hybrid — only every 4th layer
has a standard KV cache. The adapter layer automatically detects layer types
via `hybrid_classifier` and applies compaction only to full-attention layers.

## Paper

> **Fast KV Compaction via Attention Matching**
> Zweiger et al., 2026 — [arXiv:2602.16284](https://arxiv.org/abs/2602.16284)
>
> Achieves 50x KV cache compression with closed-form solutions (no gradient descent).
> Value refitting reduces MSE by ~4,000,000x compared to naive token eviction.

## Test results

- 69+ tests covering matrix ops, softmax, NNLS, least squares, adapters, quantization round-trips, and full pipeline
- Value refitting: ~4M× MSE improvement over token eviction at 4x compression
- Cosine similarity: 0.999999 at 50% compression
- Benchmarked up to 50x compression at T=4096 with Qwen3.5-0.8B dimensions

## Documentation

- [`docs/timeline.md`](docs/timeline.md) — Development timeline & 5-phase roadmap (Phases 1-5 are TODO)
- [`docs/improvement-tracker.md`](docs/improvement-tracker.md) — Implementation status matrix
- [`plan.md`](plan.md) — Detailed streaming compaction roadmap
- [`docs/attention-matching-paper.md`](docs/attention-matching-paper.md) — Full paper breakdown
