# Revised Plan: 30 tok/s × 10 Agents (Post-Profiling)

**Date:** 2026-03-15
**Based on:** Vulkan kernel profiling (GGML_VK_PERF_LOGGER) + multi-slot benchmarks

## What Changed

The original expert caching plan assumed MoE dispatch overhead was the bottleneck
(4800 tiny dispatches per step). Profiling showed:

| Assumption | Reality |
|------------|---------|
| 4800 dispatches/token | **120 dispatches/token** (already batched, n=8) |
| MoE is 50%+ of time | **MoE is 21%** of single-slot time |
| Dispatch overhead dominates | **Shared expert FFN dominates (34%)** |
| Expert caching is the silver bullet | Expert caching helps 21% fraction only |

## Current Performance

| Agents | Agg tok/s | Per-slot | Build |
|--------|-----------|----------|-------|
| 1 | 39.8 | 44.7 | Fork (Coder-Next) |
| 2 | 50.7 | 28.3 | Fork |
| 5 | 82.0 | 17.9 | Fork |
| 10 | 79.1 | 8.6 | Fork |
| **Target** | **300** | **30** | |
| **Gap** | **3.8x** | **3.5x** | |

## Where Time Goes at 10 Slots

Extrapolating from single-slot profiling (Qwen3.5-35B-A3B):

```
Single slot: ~16.5 ms per token (60 tok/s)
10 slots:    ~126 ms per step   (79 agg tok/s)
Overhead:    126 - 16.5 = 110 ms from multi-slot scaling

Breakdown of the 110 ms overhead:
  SSM sequential state: 10 × ~1.7ms = ~17 ms  (must process each seq)
  MoE weight reads:     10 × ~3.5ms = ~35 ms  (10× unique experts worst case)
  Vulkan scheduling:    ~58 ms                 (sync, buffer mgmt, dispatch)
```

The **Vulkan scheduling overhead** (~58 ms) is the biggest multi-slot penalty.
This is the cost of managing 10 concurrent compute graphs, buffer allocations,
and command buffer submissions per step.

## Revised Priority Stack

### P0: Reduce Vulkan Scheduling Overhead (biggest gap)

**Problem:** 58 ms of the 126 ms step is scheduling overhead.
**Target:** Reduce to ~15 ms → step = ~83 ms → 120 agg → 12 per-slot

| Approach | Impact | Effort |
|----------|--------|--------|
| Graph caching (reuse compute graph across steps) | 20-40 ms saved | Medium |
| Reduce buffer reallocations per slot | 10-20 ms saved | Medium |
| Vulkan command buffer batching | 5-10 ms saved | Low |

**How:** llama.cpp already has graph caching (`can_reuse()`). Check if it's
working properly for multi-slot with MoE models. The graph structure is
identical across steps — only input data changes.

### P1: SSM State Optimization (doesn't batch)

**Problem:** SSM/DeltaNet must process each sequence separately (~1.7 ms each).
At 10 slots: ~17 ms sequential overhead that can't be parallelized.
**Target:** Reduce to ~5 ms → save 12 ms per step

| Approach | Impact | Effort |
|----------|--------|--------|
| AIRE-Prune (60% state reduction) | 40% faster SSM → save ~7 ms | Medium |
| Batch SSM across sequences (if state is independent) | 5-10× faster | Hard |
| Quantize SSM state (F32 → F16) | 2× bandwidth → ~50% faster | Low |

**Quick win:** SSM state is currently F32 (see profiling: `llama_memory_recurrent:
R (f32), S (f32)`). Quantizing to F16 halves the 62 MB state and reduces
bandwidth by 2× per sequence. This is a configuration change, not code.

### P2: Shared Expert Optimization (34% of single-slot)

**Problem:** `q5_K m=8192 n=1 k=2048` runs every token, 34% of time.
At 10 slots it batches well (GEMM with n=10), so it's NOT the multi-slot
bottleneck. But reducing it helps all configurations.

| Approach | Impact | Effort |
|----------|--------|--------|
| Re-quantize shared experts Q5_K → Q4_K | ~30% faster (less bandwidth) | Easy (requant) |
| Fuse shared expert into MoE dispatch | Saves 1 dispatch per layer | Medium |

### P3: Expert Caching (21% of single-slot, grows at multi-slot)

**Problem:** 10 tokens route to 10 × 10 = 100 experts per layer (512 total).
With overlap maybe ~60 unique. Each unique expert = 1.7 MB read.
**Target:** Reduce unique experts from 60 → ~20 per layer

| Approach | Impact | Effort |
|----------|--------|--------|
| Cache-aware routing (implemented!) | 50-70% overlap → 3× fewer reads | **Done** |
| Least-stale eviction (SpecMD) | 95% cache hit vs 80% baseline | Days |
| Cross-layer prediction (Fate) | Prefetch during attention | Weeks |

The cache-aware routing API is working. The next step is wiring it into
the server's decode loop so the bias updates dynamically per step.

### P4: Prefill Regression Fix (-17%)

**Problem:** Fork is 17% slower at prefill (950 vs 1150 tok/s).
This affects time-to-first-token for each new agent request.

| Approach | Impact | Effort |
|----------|--------|--------|
| Revert flash attention row_split UMA change | Likely fix | Low |
| Profile prefill separately | Identify specific regression | Low |

## Revised Roadmap

```
Week 1: P0 (Vulkan scheduling) + P1 quick win (SSM F16)
  Target: 10-slot step from 126 ms → ~85 ms → ~118 agg → 11.8 per-slot

Week 2: P3 (dynamic cache bias in server) + P2 (shared expert requant)
  Target: ~95 ms → ~105 agg → 10.5 per-slot
  + MoE expert reads cut by 2-3× at multi-slot

Week 3: P1 (AIRE-Prune SSM) + P0 (graph caching improvements)
  Target: ~65 ms → ~154 agg → 15.4 per-slot

Week 4: P3 (least-stale eviction) + P4 (prefill fix) + tuning
  Target: ~50 ms → ~200 agg → 20 per-slot
```

**Honest assessment:** Reaching 30 tok/s × 10 agents (300 agg) may not be
achievable on this hardware with the current architecture. The fundamental
limit is the 212 GB/s memory bandwidth shared across CPU, GPU, and all
sequences. At 10 concurrent tokens, the weight reads alone (~10 GB/step for
MoE + shared + attention) require 10/212 = 47 ms minimum → 213 agg max.

The realistic target is **~200 agg tok/s (20 per-slot)**, achievable by:
- Eliminating the 58 ms Vulkan scheduling overhead
- Optimizing SSM state processing
- Reducing unique expert reads via cache-aware routing

For 30 tok/s per-slot, the math says **max 7 agents** (not 10):
- 7 × 30 = 210 agg, within the ~213 bandwidth ceiling

## Immediate Action Items

1. **Check graph caching** — Is `can_reuse()` working for MoE multi-slot?
   ```bash
   GGML_VK_PERF_LOGGER=1 llama-server -m model.gguf -np 10 ...
   ```
   Look for "graph reuse" messages in the server log.

2. **Try SSM state F16** — Add `-ctk f16 -ctv f16` and check if SSM state
   respects the KV type flag (it may be hardcoded to F32).

3. **Wire dynamic cache bias into server** — After each decode step, update
   the expert_cache_bias based on which experts were selected.
