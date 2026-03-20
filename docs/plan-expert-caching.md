# Plan: Expert Caching for 30 tok/s × 10 Agents

**Model:** Qwen3-Coder-Next 80B.A3B (Q4_K_M, 46 GB)
**Hardware:** Strix Halo (128 GB LPDDR5X, 212 GB/s, Radeon 8060S)
**Current:** 43.6 tok/s single-slot, 2.6 tok/s per slot at 10 agents
**Target:** 30 tok/s per slot × 10 agents = 300 agg tok/s

---

## Architecture (measured)

```
n_layer              = 48
n_expert             = 512       ← huge expert pool
n_expert_used        = 10        ← top-10 routing per token
expert_ff_length     = 512       ← small experts (2048→512→2048)
shared_expert_ff     = 512       ← always-active shared expert
n_embd               = 2048
n_head/n_head_kv     = 16/2 (GQA 8:1)
```

### Per-Expert Size (Q4_K_M)
```
gate:  2048 × 512 = 1.0M params
up:    2048 × 512 = 1.0M params
down:  512 × 2048 = 1.0M params
Total: 3.0M params × ~0.56 bytes = 1.7 MB per expert
```

### Per-Token Weight Read
```
10 active experts × 1.7 MB      = 17 MB  (MoE FFN)
1 shared expert × 1.7 MB         = 1.7 MB (always read)
Attention/SSM/norm per layer      = ~0.8 MB
Per layer: ~20 MB
48 layers: ~960 MB per token
Plus embeddings/output: ~60 MB
Total: ~1020 MB per token
```

---

## The 10-Agent Bandwidth Problem

### Without Expert Caching (current)

```
10 tokens per step, each picks 10 experts per layer.
512 total experts → collision probability is low.
Worst case: ~80 unique experts per layer (some overlap).

Per step: 48 layers × 80 × 1.7 MB = 6.5 GB expert reads
         + shared/attn/SSM        = 1.3 GB
         Total:                    = 7.8 GB per step

Time: 7.8 / 212 = 36.8 ms → 27.2 steps/s
Aggregate: 272 tok/s → 27.2 per slot

But measured is only 26 agg (2.6/slot) — 10x worse than theory!
The gap is MoE dispatch overhead: 512 tiny matmuls per layer per step.
```

### Why Current 10-Slot is 10x Below Theory

The bottleneck is NOT bandwidth — it's **dispatch overhead**:
- 48 layers × 10 experts = 480 expert matmuls per token
- 10 tokens × 480 = 4800 matmul dispatches per step
- Each dispatch: Vulkan command buffer submission + sync
- At ~5 μs per dispatch: 4800 × 5 = 24 ms overhead alone
- Plus actual compute: ~12 ms
- Total: ~36 ms vs theoretical 7 ms (pure bandwidth)

**Expert caching alone won't help if dispatch overhead dominates.**

---

## Strategy: Cache + Batch + Predict

### Target Budget

```
30 tok/s × 10 slots = 300 agg tok/s
Steps/s = 300 / 10 = 30
Time per step = 33.3 ms
Bandwidth available: 212 × 0.0333 = 7.06 GB per step
Dispatch budget: 33.3 ms (need to fit compute + dispatch)
```

### Three-Pronged Approach

```
                    Current              Target
Expert reads:       6.5 GB/step     →    2.0 GB/step  (cache-aware routing)
Dispatch count:     4800/step       →    ~500/step    (expert batching)
Cache misses:       ~80%            →    ~20%         (prediction + prefetch)
```

---

## Phase 1: Cache-Aware Routing (Days)

**Paper:** arxiv 2412.00099 — training-free, <0.1% accuracy loss

Bias the MoE router to prefer experts already in cache. When 10 agents
run concurrently, all 10 tokens are steered toward the SAME experts.

```cpp
// In MoE routing (before top-k selection):
for (int e = 0; e < n_expert; e++) {
    if (expert_in_cache[layer][e]) {
        logits[e] += cache_bonus;  // small additive bias
    }
}
// Then select top-10 as normal
```

**Impact:**
- With cache_bonus tuned: ~70% expert overlap across 10 tokens
- Expert reads: 80 unique → ~25 unique per layer per step
- Bandwidth: 48 × 25 × 1.7 = 2.0 GB (from 6.5 GB) = **3.3x reduction**

**Implementation:**
1. Add `bool expert_cache[n_layer][n_expert]` tracking loaded experts
2. Before router softmax, add bias for cached experts
3. After routing, update cache tracking
4. Tune `cache_bonus` (start with 0.5, sweep 0.1-2.0)

**Location:** `src/models/delta-net-base.cpp` or wherever MoE routing happens

---

## Phase 2: Expert Batching (Days)

When multiple tokens route to the same expert (thanks to Phase 1),
batch them into a single larger matmul instead of N separate tiny ones.

```
Current:  token1→expert7: matmul(1, 2048, 512)
          token2→expert7: matmul(1, 2048, 512)
          token5→expert7: matmul(1, 2048, 512)

Batched:  tokens{1,2,5}→expert7: matmul(3, 2048, 512)  ← 1 dispatch
```

**Impact:**
- With 70% overlap: average 7 tokens per expert per layer
- Dispatches: 48 × 25 experts × 3 matmuls = 3600 → but batched: 48 × 25 × 1 fused = 1200
- Dispatch overhead: 1200 × 5 μs = 6 ms (from 24 ms) = **4x reduction**

**Implementation:**
1. After routing, group tokens by expert assignment
2. For each unique expert: gather input vectors, run batched matmul, scatter outputs
3. This is what `ggml_mul_mat_id` already does — verify it batches properly

**Location:** `ggml/src/ggml-vulkan/ggml-vulkan.cpp` (mul_mat_id dispatch)

---

## Phase 3: Least-Stale Eviction (Days)

**Paper:** arxiv 2602.03921 (SpecMD) — 85x fewer cache misses than LRU

MoE access patterns are NOT temporally local (LRU is wrong). Expert
access follows periodic/structured patterns. Least-stale eviction tracks
the "staleness" of each expert and evicts the one least likely to be
reused soon.

```
LRU:          evicts most-recently-unused → bad for periodic patterns
Least-Stale:  evicts expert with longest gap since pattern repeat → 85x better
```

**Implementation:**
1. Track per-expert access history: `uint32_t last_used[n_layer][n_expert]`
2. Track access period: `uint32_t period[n_layer][n_expert]`
3. On eviction: choose expert with `staleness = step - last_used - period`
4. Highest staleness = least likely to be needed = evict

**Impact:**
- Cache hit rate: 80% → 95%+ with least-stale
- Reduces cold-start expert reads from 25 → ~5 per layer per step
- Bandwidth: 2.0 GB → 0.8 GB = **2.5x further reduction**

---

## Phase 4: Pre-Attention Predictor (Weeks)

**Paper:** arxiv 2511.10676 — 94.7% accuracy on Qwen3-30B

Train a tiny neural network (2 linear layers) to predict which experts
will be needed, using activations BEFORE the attention block. This lets
you prefetch experts while attention is computing.

```
Timeline:
  [attention computing]  ← GPU busy with attention
  [predict experts]      ← CPU runs tiny predictor in parallel
  [prefetch experts]     ← DMA loads predicted experts
  [expert matmul]        ← experts already in cache, zero miss
```

**Implementation:**
1. Collect routing decisions from 1000+ inference steps
2. Train 2-layer MLP: input=pre-attn hidden state, output=expert probabilities
3. Quantize predictor to INT8 (runs on CPU while GPU does attention)
4. Integrate prefetch: start loading predicted experts during attention

**Impact:**
- Near-zero cache miss latency (prefetch hides load time)
- Combined with Phase 1-3: effective bandwidth → ~0.5 GB/step
- Per-step time: 0.5/212 + compute ≈ 5 ms → 200 steps/s → 2000 agg (saturates at dispatch limit)

---

## Phase 5: Cross-Layer Prediction (Weeks)

**Paper:** arxiv 2502.12224 (Fate) — 99% expert hit rate

Use layer N-1's routing decision to predict layer N's experts.
Adjacent layers often use similar experts. Shallow-favoring cache
allocates more cache to early layers (which have more stable patterns).

**Impact:** Marginal over Phase 4, but gets the last few % of cache misses.

---

## Projected Results

| Phase | Expert reads/step | Dispatches | Per-slot tok/s | Agg (10 slots) |
|-------|-------------------|------------|----------------|-----------------|
| Current | 6.5 GB | 4800 | 2.6 | 26 |
| P1: Cache-aware routing | 2.0 GB | 4800 | ~8 | ~80 |
| P1+P2: + Batching | 2.0 GB | 1200 | ~20 | ~200 |
| P1-P3: + Least-stale | 0.8 GB | 1200 | **~30** | **~300** |
| P1-P4: + Prediction | 0.5 GB | 1200 | ~40 | ~400 |

**Phase 1+2+3 achieves the 30 tok/s target. All three are days-level effort.**

---

## Implementation Roadmap

```
Week 1: Phase 1 (cache-aware routing) + Phase 2 (expert batching)
         Deliverable: 10-agent benchmark at ~200 agg tok/s

Week 2: Phase 3 (least-stale eviction) + tuning
         Deliverable: 10-agent benchmark at ~300 agg tok/s (target met)

Week 3: Phase 4 (predictor training + integration)
         Deliverable: Beyond target, quality validation

Week 4: Quality benchmarks, multi-model testing
         Deliverable: Production-ready expert caching
```

---

## Key Files to Modify

| File | Change |
|------|--------|
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Expert cache tracking, batched dispatch |
| `src/models/delta-net-base.cpp` | Router logit bias for cached experts |
| `ggml/include/ggml.h` | Expert cache API (init, query, update) |
| `common/expert-cache.h` (new) | Cache policy implementation (least-stale) |
| `tools/expert-predictor/` (new) | Training script for pre-attention predictor |

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Cache-aware routing degrades quality | Low | High | Sweep cache_bonus, measure perplexity |
| Expert batching has diminishing returns | Medium | Medium | Profile actual dispatch overhead first |
| 512 experts too many for cache tracking | Low | Low | Bitset: 512 bits = 64 bytes per layer |
| Least-stale needs tuning per model | Medium | Low | Start with LFU, graduate to least-stale |
| Predictor training data insufficient | Medium | Medium | Use 10K+ tokens from coding prompts |
