# Serving Benchmarks Session (2026-03-14)

## Setup
- Hardware: AMD Strix Halo (Ryzen AI Max+ 395, Radeon 8060S, 128 GB RAM, UMA)
- Stock llama.cpp: b8334 (Vulkan, from `ggml-org/llama.cpp/releases`)
- Flash-attn fork: build 8305 (**5-11x server regression** — do not use for serving)

## SmolLM3 3B (Pure Attention) — Compaction Impact

### Fixed 16K Memory Budget (flash-attn fork)

| Config | Slots | Ctx/slot | Previous | Current | Change |
|--------|-------|----------|----------|---------|--------|
| Full KV | 4 | 2048 | 28.9 | 32.7 | +13% |
| Full KV | 8 | 2048 | 39.8 | 45.6 | +15% |
| 5x compact | 8 | 512 | 58.3 | 68.2 | +17% |
| 5x compact | 16 | 512 | 94.6 | 102.3 | +8% |
| 5x compact | 32 | 512 | 127.0 | 140.2 | +10% |

### Fixed 24K Budget, 10K Prompt (flash-attn fork)

| Ratio | Slots | Previous | Current | Change |
|-------|-------|----------|---------|--------|
| 1x | 2 | 8.2 | 8.9 | +9% |
| 5x | 10 | 27.9 | 30.9 | +11% |
| 10x | 20 | 55.2 | 61.2 | +11% |
| 20x | 40 | 123.6 | 135.7 | +10% |
| 50x | 80 | 293.4 | 318.2 | +8% |

Consistent **+8-17% improvement** over previous baselines across all configurations.

### Stock llama.cpp Results (from memory, separate session)

| Compact | KV tok | Slots | Agg tok/s | vs 1x |
|---------|--------|-------|-----------|-------|
| 1x | 10500 | 6 | 9.1 | baseline |
| 5x | 2100 | 32 | 52.3 | 5.8x |
| 10x | 1050 | 64 | 111.7 | 12.3x |
| 50x | 210 | 256 | 604.4 | **66.7x** |

## Qwen3-Coder-Next 80B.A3B (Hybrid, Stock Build)

### Per-Slot Speed (1 slot, varying KV size)

| KV Tokens | Gen tok/s | Prefill tok/s |
|-----------|-----------|---------------|
| 461 | 39.9 | 388.5 |
| 897 | 39.4 | 381.8 |
| 1754 | 39.2 | 385.7 |
| 2679 | 39.1 | 399.7 |
| 3319 | 38.4 | 399.2 |

Per-slot speed is **constant at ~39 tok/s** regardless of KV size. GDN layers dominate compute.

### Multi-Agent Scaling (10 slots x 4K ctx)

| Agents | ~500 tok KV | ~1K tok KV |
|--------|------------|------------|
| 1 | 39.5 tok/s | 39.3 tok/s |
| 2 | 23.6 agg | 14.8 agg |
| 5 | 32.8 agg | FAIL |
| 10 | 34.5 agg | FAIL |

10 concurrent agents work at ~500 tok KV (34.5 agg tok/s). Fails at 2K+ with 5+ agents (Vulkan compute pressure on 46 GB model).

### Flash-attn Fork vs Stock Comparison

| Model | Fork tok/s | Stock tok/s | Regression |
|-------|-----------|-------------|------------|
| SmolLM3 3B | 19 | 91.5 | 4.8x slower |
| Gemma 3 4B | CRASH | 67.4 | - |
| Qwen3.5-35B-A3B | 6 | 64.1 | 10.7x slower |
| Qwen3-Coder-Next | ~3* | 39.9 | ~13x slower |

*Estimated from 35B-A3B regression ratio.

## Key Takeaways

1. **Pure attention models**: compaction delivers massive throughput gains (up to 66.7x at 50x compression on stock build)
2. **Hybrid models**: compaction is a memory enabler, not a speed booster. Per-token speed unchanged.
3. **Always use stock llama.cpp** for serving benchmarks — the flash-attn fork has a 5-11x regression
4. **10 agents x 100K is impossible without compaction** (150-262 GB KV). With 200x compaction: trivially fits.
5. **Qwen3-Coder-Next is the best hybrid candidate** (kv_unified=true, n_seq_max=10, 39 tok/s)
