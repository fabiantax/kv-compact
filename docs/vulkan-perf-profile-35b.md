# Vulkan Performance Profile: Qwen3.5-35B-A3B (2026-03-15)

**Model:** Qwen3.5-35B-A3B Q4_K_M, 256 experts, top-8
**Hardware:** Radeon 8060S (RDNA 3.5), 128 GB LPDDR5X
**Build:** Fixed fork (UMA profiler disabled), Vulkan, ngl=99

## Per-Token Kernel Breakdown (tg50, single slot)

| Kernel | Count | Total us | % | Category |
|--------|-------|----------|---|----------|
| `MUL_MAT_VEC q5_K m=8192 n=1 k=2048` | 30 | 5666 | **34.2%** | **Shared expert FFN** |
| `MUL_MAT_VEC q4_K m=2048 n=1 k=4096` | 31 | 3979 | 24.0% | **Attention QKV** |
| `MUL_MAT_ID_VEC q4_K m=512 n=8 k=2048 n_expert=256` | 80 | 2186 | 13.2% | MoE gate_up |
| `MUL_MAT_ID_MUL q6_K m=2048 n=8 k=512 n_expert=256` | 20 | 829 | 5.0% | MoE down (Q6_K) |
| `MUL_MAT_ID_MUL q4_K m=2048 n=8 k=512 n_expert=256` | 20 | 526 | 3.2% | MoE down (Q4_K) |
| `MUL_MAT_VEC q4_K m=2048 n=1 k=512` | 20 | 39 | 0.2% | Expert shared proj |
| `MUL_MAT_VEC q4_K m=8192 n=1 k=2048` | 10 | 523 | 3.2% | Dense FFN |
| `MUL_MAT_VEC f32 m=256 n=1 k=2048` | 40 | 775 | 4.7% | SSM projections |
| `TOPK_MOE_EARLY_SOFTMAX_NORM` | 40 | 297 | 1.8% | Router top-k |
| Other (ADD, NORM, SSM, etc.) | — | ~1720 | 10.5% | — |
| **Total per token** | | **~16,560** | **100%** | **~60 tok/s** |

## Category Summary

| Category | % of Time | Cacheable? | Multi-slot scaling |
|----------|-----------|------------|-------------------|
| **Shared expert FFN** | 34.2% | No (runs every token) | Batches well (GEMM) |
| **Attention QKV** | 24.0% | No (weight read) | Batches well |
| **MoE experts** | 21.4% | **YES — cache-aware routing** | Batches if same experts |
| **SSM/DeltaNet** | ~10% | No (sequential state) | Does NOT batch |
| **Router + overhead** | ~10% | No | Negligible |

## Key Insights

### 1. Shared Expert is the #1 Bottleneck (34.2%)
The shared expert FFN (`q5_K m=8192 n=1 k=2048`) runs on EVERY token regardless
of MoE routing. It's 34% of per-token time. No amount of expert caching helps here.

**Optimization:** Quantize shared experts to Q4_K (from Q5_K) for ~40% bandwidth
reduction. Or fuse the shared expert into the MoE dispatch.

### 2. MoE Experts are Only 21.4% — Less Than Expected
The expert caching plan estimated MoE dispatch as the primary bottleneck.
Profiling shows it's only 1/5 of the time. At 10 concurrent slots, the
MoE fraction grows (10× expert reads) but shared expert also scales (10× batch).

### 3. Expert Dispatch is Already Efficient
`MUL_MAT_ID_VEC` processes all 8 experts in a single dispatch (n=8).
There are only 80 + 20 + 20 = 120 MoE dispatches per token (3 per MoE layer ×
40 layers), not the 4800 estimated in the plan. The plan's dispatch overhead
estimate was wrong — llama.cpp batches experts per layer.

### 4. Multi-Slot Scaling Analysis
At 10 concurrent tokens:
- Shared expert: 5666 us → ~5666 us (batches into GEMM, nearly constant)
- Attention: 3979 us → ~3979 us (batches well for small ctx)
- MoE: 3541 us → ~35,000 us if 10× unique experts (OR ~5,000 us with caching)
- SSM: ~1700 us → ~17,000 us (sequential, does NOT batch)

**Without caching:** SSM + MoE dominate at 10 slots → ~52,000 us → ~19 tok/s agg
**With caching (70% overlap):** MoE drops from 35K to 5K → ~27,000 us → ~37 tok/s agg

### 5. Revised Expert Caching Impact

| Scenario | MoE us | Total us | Agg tok/s (10 slots) | Per-slot |
|----------|--------|----------|---------------------|----------|
| No caching | 35,000 | 62,000 | 16.1 | 1.6 |
| 50% overlap | 17,500 | 44,500 | 22.5 | 2.2 |
| 70% overlap | 10,500 | 37,500 | 26.7 | 2.7 |
| 90% overlap | 3,500 | 30,500 | 32.8 | 3.3 |

Measured: 79.1 agg at 10 slots (8.6 per-slot) — much better than the model
predicts, suggesting batch-level optimizations and weight caching are already
partially effective in the Vulkan backend.

## Fork vs Stock Comparison (llama-bench)

| Test | Fork | Stock | Delta |
|------|------|-------|-------|
| tg50 (generation) | **67.96** | 63.84 | **+6.5%** |
| pp512 (prefill) | 949.99 | **1149.54** | **-17%** |

Fork's RDNA3 tuning helps generation (+6.5%) but hurts prefill (-17%).
The prefill regression may come from the flash attention row_split change.
