# Plan: KV Compaction for 200K-Context Agents (Qwen3.5-0.8B)

## Goal

Make `kv-compact` work for agentic workloads with 200K token contexts on
Qwen3.5-0.8B, targeting 50x compression with acceptable quality at interactive
speed (<2s total compaction overhead across a full session).

## Architecture Context

Qwen3.5-0.8B: hybrid model, only 6/24 layers use full attention (KV cache).
- n_head_kv=2, d_head=256, GQA ratio=4
- KV cache at 200K: ~2.5 GB (bf16) across 6 layers
- Target: compress to ~50 MB (50x) → ~4K tokens per layer

## Current Limitations (from benchmark)

1. **No streaming mode** — must compact all T at once
2. **Scalar C++ only** — no SIMD
3. **No token pinning** — can't protect system prompt / tool boundaries
4. **Single-shot compaction** — no incremental merge of compacted + new tokens

---

## Phase 1: Streaming Compaction Engine

**The core change.** Instead of compacting 200K→4K in one shot (impossible
speed-wise), compact incrementally in chunks.

### Step 1.1: Add `streaming_compactor` class to `kv-compact-math.h`

```cpp
struct streaming_config {
    int budget;             // max tokens to retain (e.g. 4096)
    int trigger;            // compact when cache hits this size (e.g. 8192)
    int pin_prefix;         // first N tokens are pinned (system prompt)
    int recent_window;      // last N tokens are never compacted
    key_select_mode select_mode;
    beta_fit_mode   fit_mode;
    int n_alt_rounds;
};

class streaming_compactor {
    streaming_config cfg;
    int current_size;       // tokens currently in cache

    // Accumulated compacted state per layer per head
    struct head_state {
        std::vector<float> C_k;   // [budget * d_k]
        std::vector<float> C_v;   // [budget * d_v]
        std::vector<float> beta;  // [budget]
        int n_compacted;          // how many slots used
    };

public:
    // Called when new tokens are added to cache
    // Returns true if compaction was triggered
    bool maybe_compact(
        const float * K_all, const float * V_all,
        int T_current, int n_layers, int n_head_kv, int d_k, int d_v);

    // Get compacted state for write-back
    const head_state & get_state(int layer, int head) const;
};
```

### Step 1.2: Chunk compaction algorithm

When `current_size >= trigger`:
1. Split cache into zones:
   - `[0, pin_prefix)` → pinned, never touched
   - `[pin_prefix, current_size - recent_window)` → compactable zone
   - `[current_size - recent_window, current_size)` → recent, never touched
2. Compact the compactable zone from `T_compact` → `budget - pin_prefix - recent_window`
3. Use reference queries from recent window (already in cache, free)
4. Write back compacted K/V/beta, shift recent window down

### Step 1.3: Tests

- Unit test: `test_streaming_basic` — verify repeated compaction preserves quality
- Unit test: `test_streaming_cumulative_error` — measure error after N rounds
- Benchmark: `bench_streaming_200k` — simulate 200K tokens arriving in chunks

**Files changed:**
- `include/kv-compact-math.h` — add streaming_compactor
- `tests/test-kv-compact-math.cpp` — add streaming tests
- `tests/bench-synthetic.cpp` — add streaming benchmark

---

## Phase 2: Speed — Fast Paths for Large T

### Step 2.1: Mini-batch k-means

Current k-means is O(T * t * d * iters). At T=8K, t=4K, d=256, 20 iters:
~164 billion FLOPs → ~40s single-threaded.

Fix: mini-batch k-means (Sculley 2010).
- Sample B=256 tokens per iteration instead of all T
- Assignment: O(B * t * d) per iter → 50x faster
- Add `kmeans_mini_batch()` variant

### Step 2.2: Approximate key scoring

For max_attn mode, the bottleneck is `Q_ref @ K^T` which is O(n_q * T * d).
At T=8K, n_q=32, d=256: ~67M FLOPs — already fast.

But NNLS (Step 2) does 200 iterations of `M^T @ M @ w` and `M^T @ b`.
M is [n_q × t]. At t=4K, n_q=32: M is small, this is fine.

**Verdict:** For max_attn + NNLS, the current algorithm is fast enough for
chunk sizes up to ~16K. No SIMD needed in Phase 2.

### Step 2.3: Pre-allocated scratch buffers

Currently every call to `compact_layer_all_heads` allocates vectors.
Add a scratch buffer pool to `streaming_compactor` — allocate once, reuse.

**Files changed:**
- `include/kv-compact-math.h` — add kmeans_mini_batch, scratch pool

---

## Phase 3: Token Pinning

### Step 3.1: Pin mask in compaction config

```cpp
struct compaction_config {
    // ... existing fields ...
    const bool * pin_mask = nullptr;  // [T] true = never evict
};
```

Pinned tokens are excluded from key selection but included in attention
computation. They pass through to the compacted cache unchanged.

### Step 3.2: Agentic pin policy

For tool-use agents, auto-pin:
- BOS token
- System prompt tokens (configurable prefix length)
- Tool call boundary tokens (configurable token IDs)
- Last N tokens (recent window)

### Step 3.3: Tests

- Unit test: verify pinned tokens survive compaction unchanged
- Unit test: verify attention output includes pinned token contributions

**Files changed:**
- `include/kv-compact-math.h` — pin_mask support in key selection
- `src/kv-compact.cpp` — CLI --pin-prefix, --pin-tokens flags

---

## Phase 4: Cumulative Error Control

The key risk of streaming compaction: errors compound across rounds.
Round N compacts already-compacted tokens, amplifying distortion.

### Step 4.1: Error monitoring

After each compaction round, compute:
- MSE between compacted and pre-compaction attention output
- Mass error (beta drift)
- Track cumulative metrics across rounds

### Step 4.2: Adaptive trigger

If cumulative error exceeds threshold, options:
- Increase budget temporarily (compact less aggressively)
- Skip compaction for this round (let cache grow)
- Alert the caller

### Step 4.3: "Re-anchor" mechanism

Every K rounds, use the full compacted cache as reference to recompute
beta from scratch (not incrementally). This corrects accumulated beta drift
without full recomputation.

**Files changed:**
- `include/kv-compact-math.h` — error monitoring in streaming_compactor
- `tests/bench-synthetic.cpp` — cumulative error benchmark

---

## Phase 5: Qwen3.5-0.8B Integration

### Step 5.1: Hybrid model support in CLI

Qwen3.5-0.8B has 24 layers but only layers 3,7,11,15,19,23 use full attention.
The CLI must:
- Detect hybrid architecture (DeltaNet + full attention)
- Only compact full-attention layers
- Pass through DeltaNet layers unchanged

### Step 5.2: Per-head budget for 2-head GQA

With only 2 KV heads, per-head budget allocation is trivial (2 choices).
But the GQA ratio of 4 means errors in one KV head affect 4 query heads.
Add GQA-aware sensitivity weighting: multiply head sensitivity by GQA ratio.

### Step 5.3: E2E test with Qwen3.5-0.8B

- Download Qwen3.5-0.8B-Q4_K_M.gguf
- Prefill 4K tokens of agentic conversation (system prompt + tool calls)
- Compact at 4x, 16x, 50x
- Measure perplexity on continuation
- Compare: full cache vs compacted vs token eviction

**Files changed:**
- `src/kv-compact.cpp` — hybrid layer detection
- `tests/test-kv-compact-e2e.cpp` — Qwen3.5 test case

---

## Implementation Order & Dependencies

```
Phase 1 (streaming) ──→ Phase 4 (error control)
     │                         │
     └──→ Phase 2 (speed) ────┘
                                ╲
Phase 3 (pinning) ──────────────→ Phase 5 (Qwen integration)
```

Phases 1 and 3 are independent and can be parallelized.
Phase 2 unblocks Phase 4 (need fast compaction to test cumulative error).
Phase 5 requires all others.

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Compaction overhead per round | <100ms (max_attn, T=16K→8K) |
| Total overhead for 200K session | <2.5s (25 rounds × 100ms) |
| Memory after compaction | <150 MB (vs 2.5 GB full) |
| Perplexity increase at 50x | <5% vs full cache |
| Cumulative error after 25 rounds | MSE < 10x single-round MSE |
| Token pinning | System prompt + recent 2K preserved exactly |

---

## Risk Assessment

1. **Cumulative error may blow up** — Mitigated by Phase 4 (monitoring +
   re-anchoring). If fundamentally broken, fall back to lower compression.

2. **Hybrid model state format** — DeltaNet layers store state differently.
   Need to verify llama.cpp state save/restore handles this correctly.

3. **Beta averaging across heads** — Current CLI averages beta across all
   heads into one per-cell bias. With streaming, beta values from different
   compaction rounds may be inconsistent. May need per-round beta tracking.

4. **Qwen3.5 attention bias patch** — The llama.cpp patch adds per-cell bias
   to attention mask. Need to verify this works correctly with Qwen3.5's
   hybrid attention (only on full-attention layers, not DeltaNet).
