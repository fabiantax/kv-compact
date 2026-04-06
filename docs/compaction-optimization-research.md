# Compaction Performance Optimization: Research Synthesis

**Date:** 2026-04-05
**Scope:** Bottleneck analysis, arXiv survey (2024-2026), TRIZ + Axiomatic Design ideation

---

## 1. Current Bottleneck Breakdown

For a 50k-token context at 5x compaction (~3.3s total):

| Stage | Time (est.) | % | Why |
|-------|-------------|---|-----|
| State serialization (`llama_state_seq_get_data`) | ~800ms | 24% | memcpy of ~1 GB KV state |
| State parsing (binary → per-layer K/V arrays) | ~300ms | 9% | Deserialization, dequant |
| **Key scoring** (`mat_mul_ABt` per chunk per head) | ~1,000ms | 30% | O(n_q × chunk_size × d_k × n_chunks × n_heads) |
| **Softmax + importance fusion** | ~300ms | 9% | Per-head, per-chunk post-processing |
| **Value refit** (LS solve per head per chunk) | ~500ms | 15% | O(n_q × t²) normal equations + Cholesky |
| State rebuild + deserialize | ~400ms | 13% | Build compacted buffer, `set_data` |

At 200k tokens, state I/O dominates (~60%): 4 GB serialize + 4 GB deserialize + 4 GB rebuild.

### Key observation: Two distinct regimes

- **T ≤ 50k**: Scoring + LS math dominate (~55%). State I/O is ~35%.
- **T ≥ 100k**: State I/O dominates (~60%). Scoring/LS is ~30%.

Optimization strategy must address **both** regimes.

---

## 2. arXiv Survey: Most Relevant Techniques

### 2.1 Sub-quadratic scoring (replaces O(n_q × T × d_k) matmul)

| Technique | Paper | Complexity | Quality | Notes |
|-----------|-------|-----------|---------|-------|
| **FASA: Dominant frequency chunks** | arXiv:2602.03152 | O(T × d_fc) | SOTA | RoPE frequency structure → free proxy for importance. Near-100% of full-KV quality with only 256 tokens. |
| **HISA: Hierarchical block scoring** | arXiv:2603.28458 | O(T/B × d_k + k × B × d_k) | IoU >99% | Coarse-to-fine: score block reps, then refine in top blocks. 2-4x speedup. |
| **Expected attention** | arXiv:2510.00636 | O(T × d_k) | Good pre-filter | score(j) = mean(Q_ref) @ K[j]. 1000x faster than full scoring. Use as coarse filter. |
| **Value-norm pruning** | arXiv:2406.12335 (VATP) | O(T × d_v) | Good pre-filter | Prune tokens with ||V[j]||₁ ≈ 0. Eliminates 20-40% before scoring. |
| **Loki: Low-rank key projection** | arXiv:2406.02542 | O(n_q × T × d') | High | Keys are low-rank — project to d'=16 and score there. 8x reduction. |
| **LSH selection** | arXiv:2412.16187 (HashEvict) | O(T × L × d_k) | Good | SimHash → bucket collision = likely important. O(T) total. |
| **SnapKV observation window** | arXiv:2404.14469 | O(W × T × d_k) | Near-lossless | Use last W=32 tokens as Q_ref. 30x reduction when n_q >> W. |

### 2.2 Non-destructive compression (avoiding information loss)

| Technique | Paper | Method | Quality |
|-----------|-------|--------|---------|
| **ZeroMerge** | arXiv:2503.10714 | Residual merging into retained tokens | Full-cache quality at 5% retention |
| **EMS** | arXiv:2412.08521 | Evict-then-merge with global+local scores | Lowest perplexity |
| **DMC** | arXiv:2403.09636 | Learned online merge gates | 4x compression, requires light training |

### 2.3 Layer-adaptive methods

| Technique | Paper | Insight |
|-----------|-------|---------|
| **PyramidKV** | arXiv:2406.02069 | Lower layers need more KV entries; upper layers need fewer. Monotonically decreasing budget. |
| **RazorAttention** | arXiv:2407.15891 | Most heads are "local" (need only recent tokens). Few "retrieval heads" need full cache. 70% reduction. |

### 2.4 Cross-layer methods

| Technique | Paper | Insight |
|-----------|-------|---------|
| **xKV** | arXiv:2503.18893 | KV singular vectors align across layers → shared low-rank subspace. 6.8x compression. |
| **MiniCache** | arXiv:2405.14366 | Middle-to-deep layer KV states are highly similar. Interpolate directions, preserve magnitudes. |

---

## 3. TRIZ Analysis

### Principle 2: Taking Out

**Current problem:** A separate scoring pass computes Q_ref @ K^T after prefill.
**TRIZ insight:** Remove the scoring pass entirely. The prefill already computes attention — accumulate importance during the forward pass.

**Implementation:** Hook into llama.cpp's attention kernel. After each softmax, accumulate the attention weights into a per-position importance buffer. Zero extra compute; only memory O(T) per layer.

**Impact:** Eliminates 30-55% of compaction time (the entire scoring + softmax stage).

### Principle 10: Prior Action

**Current problem:** Compaction is triggered after the KV cache is full, requiring a full separate pass.
**TRIZ insight:** Pre-compute importance incrementally during prefill itself. When context hits a threshold, importance scores are already ready.

**Implementation:** During prefill, maintain a running `importance[j] += attention_weight[i][j]` accumulator. When compaction is triggered, skip scoring entirely — go straight to key selection.

**Related work:** H2O (arXiv:2306.14048) accumulates scores online during decode. A2SF (arXiv:2407.20485) adds exponential decay. Your chunked approach already processes tokens sequentially.

### Principle 12: Equipotentiality

**Current problem:** Serialize → parse → compact → serialize → deserialize = 4 copies of GB-scale data.
**TRIZ insight:** Eliminate the state round-trip. Modify KV cells in-place where they live.

**Implementation options:**
1. **Clear + contiguous write:** Before `set_data()`, call `llama_memory_seq_rm()` to clear the cache. This makes `find_slot()` allocate contiguously from position 0, hitting the fast-path in `state_read_data()` (single memcpy instead of per-row scatter).
2. **Direct tensor access:** Use `ggml_backend_tensor_set()` to write compacted K/V rows directly into the KV tensors. Requires including llama.cpp internal headers but eliminates 2 of 4 copies.
3. **In-place shuffle:** Compact the K/V tensor rows in-place using a permutation — swap/move rows to their final positions within the same buffer.

**Impact:** Eliminates 35-60% of compaction time (the state I/O).

### Principle 13: Do It In Reverse

**Current problem:** Select which tokens to keep (O(T log T) sort over all T positions).
**TRIZ insight:** At high compression ratios, there are fewer tokens to keep than to discard. Instead of "select top-k to keep," compute "select bottom-(T-k) to discard."

**Application:** For 20x+ compression, only 5% of tokens are kept. Identifying the 95% to discard via cheap heuristics (value norm, recency, key norm) is O(T). Then keep everything else.

### Principle 17: Another Dimension

**Current problem:** Scoring uses the spatial dimension (token positions) against query dimension.
**TRIZ insight:** Exploit the RoPE frequency domain. FASA (arXiv:2602.03152) shows that dominant frequency chunks in RoPE-encoded keys are a "computationally free proxy" for importance.

**Implementation:** Extract the top 2-4 frequency components from each key vector (already available — they're subsets of the existing dimensions). Use their magnitude as importance scores. No attention computation needed.

**Impact:** Reduces scoring from O(n_q × T × d_k) to O(T × d_fc) where d_fc ≈ d_k/8. ~8x speedup on the scoring stage.

### Principle 21: Skipping

**Current problem:** Every chunk goes through full scoring + selection + LS refit.
**TRIZ insight:** Skip entire stages for chunks that are "easy" (uniform importance) or "unimportant" (low max importance).

**Application:** After coarse block scoring, if a chunk's importance variance is below a threshold, use uniform sampling instead of expensive LS solve. For SSM-adjacent chunks in hybrid models, skip entirely.

### Principle 24: Intermediary

**Current problem:** Full-precision F32 scoring on all T tokens.
**TRIZ insight:** Use a low-precision intermediary for the initial filter. Score in FP16 or even INT8 — importance ranking is preserved even with reduced precision.

**Implementation:** Cast K and Q_ref to FP16 before `mat_mul_ABt`. Apple Silicon NEON has 2x throughput for FP16 matmuls. For the coarse filter (expected attention), even INT8 dot products suffice.

### Principle 25: Self-Service

**Current problem:** We generate separate Q_ref vectors (either from repeat-prefill or from K-as-Q).
**TRIZ insight:** The model's own attention during prefill already knows which tokens are important. Use the attention weights that are naturally computed.

**Implementation:** During prefill, the attention softmax produces exact attention distributions. For the last W tokens (the "observation window" per SnapKV), these attention patterns predict generation-time attention with high fidelity. No Q_ref generation needed at all.

### Principle 35: Parameter Changes

**Current problem:** Uniform chunk size and compression ratio across all layers and heads.
**TRIZ insight:** Different layers and heads have different KV importance distributions (PyramidKV, RazorAttention).

**Implementation:** After profiling once per model, set per-layer retention ratios:
- Lower attention layers: keep more (wider attention patterns)
- Higher attention layers: keep less (concentrated on few tokens)
- For Qwen 3.5's 10 attention layers: maybe [0.25, 0.25, 0.22, 0.20, 0.18, 0.15, 0.12, 0.10, 0.08, 0.06]

**Impact:** Better quality at same total KV size, or same quality at smaller KV size.

### Principle 40: Composite Materials

**Current problem:** Pure eviction loses information irreversibly.
**TRIZ insight:** Combine eviction with merging. Evict the least important tokens, but merge their information into retained neighbors (ZeroMerge, EMS).

**Implementation:** After key selection, for each evicted token, distribute its value contribution to the nearest retained tokens weighted by key similarity. This is an O(T × t × d_v) operation but can be approximated.

---

## 4. Axiomatic Design Analysis

### Current functional requirements (coupled):

| FR | Current implementation | Coupled with |
|----|----------------------|--------------|
| FR1: Select important tokens | Attention scoring (Q@K^T) | FR3 (needs same attention for value refit) |
| FR2: Fit compacted values | Least-squares on selected tokens | FR1 (selection determines LS inputs) |
| FR3: Minimize compaction time | Serial pipeline | FR1, FR2 (faster = less accuracy) |
| FR4: Maximize output quality | Full attention scoring + LS | FR3 (more compute = slower) |

### Uncoupled design (target):

**Decouple scoring from selection.** Scoring should be a byproduct of inference, not a separate stage.

```
FR1': Maintain running importance during prefill    (O(1) per token, zero extra time)
FR2': Select top-k from pre-computed importance     (O(T) partial sort, <1ms)
FR3': Refit values via LS on selected tokens        (O(n_q × t²), already fast with chunks)
FR4': Pipeline compaction with generation            (zero critical-path overhead)
```

This eliminates the scoring stage entirely (FR1' replaces FR1) and decouples quality from time (FR4' removes FR3 from the critical path).

### Information axiom: Simplest sufficient design

The simplest design that satisfies all FRs:

1. **Accumulate attention during prefill** — a single FADD per attention weight per token. No separate pass.
2. **Top-k selection** — `std::partial_sort` on importance scores. O(T) for the selection.
3. **LS value refit** — already implemented and chunked. Unchanged.
4. **Skip state round-trip** — clear cache, write contiguously.

---

## 5. Prioritized Implementation Plan

### Tier 1: Quick wins (days, no new dependencies)

| # | Optimization | Stage affected | Speedup | TRIZ principle |
|---|-------------|---------------|---------|----------------|
| 1 | **Clear KV before set_data** | State I/O | 2-3x faster restore | #12 Equipotentiality |
| 2 | **Reduce n_q to observation window (32)** | Scoring | 2x scoring speedup | #25 Self-Service |
| 3 | **Value-norm pre-filter** | Scoring | Eliminates 20-40% of tokens before scoring | #16 Partial action |
| 4 | **FP16 scoring matmul** | Scoring | 2x throughput on NEON | #24 Intermediary |

### Tier 2: Medium effort (weeks)

| # | Optimization | Stage affected | Speedup | TRIZ principle |
|---|-------------|---------------|---------|----------------|
| 5 | **Expected attention pre-filter** | Scoring | 1000x on filter pass, reduces to ~10k candidates | #21 Skipping |
| 6 | **Hierarchical block scoring (HISA)** | Scoring | 5-10x, IoU >99% | #7 Nested doll |
| 7 | **FASA dominant frequency proxy** | Scoring | Replace scoring with free frequency analysis | #17 Another dimension |
| 8 | **Layer-adaptive retention (PyramidKV)** | Quality | Better quality at same or smaller KV | #35 Parameter changes |
| 9 | **Direct tensor access** (bypass serialize) | State I/O | Eliminate 2 of 4 GB-scale copies | #12 Equipotentiality |

### Tier 3: Architecture changes (months)

| # | Optimization | Stage affected | Speedup | TRIZ principle |
|---|-------------|---------------|---------|----------------|
| 10 | **Accumulate importance during prefill** | Scoring | Eliminate scoring entirely | #2 Taking out, #10 Prior action |
| 11 | **Background compaction thread** | Critical path | Compaction disappears from latency | #20 Continuity |
| 12 | **KV merging instead of eviction** | Quality | Better quality at extreme ratios | #40 Composites |
| 13 | **Cross-layer shared subspace (xKV)** | Memory | 6.8x additional compression | #17 Another dimension |

---

## 6. Projected Impact

### At 200k tokens, 5x compaction (current: ~13s)

| After optimizations | Compaction time | Cumulative tok/s (200 tok) | Cumulative tok/s (2000 tok) |
|---------------------|-----------------|---------------------------|----------------------------|
| Current | 13,000ms | 14 t/s | 69 t/s |
| + Tier 1 only | ~5,000ms | 28 t/s | 83 t/s |
| + Tier 1+2 | ~1,500ms | 53 t/s | 88 t/s |
| + Tier 1+2+3 | ~200ms | 91 t/s | 92 t/s |
| Ideal (pipelined) | 0 (background) | 92 t/s | 92 t/s |

### At 50k tokens, 5x compaction (current: ~3.3s)

| After optimizations | Compaction time | Cumulative tok/s (200 tok) | Cumulative tok/s (2000 tok) |
|---------------------|-----------------|---------------------------|----------------------------|
| Current | 3,300ms | 36 t/s | 74 t/s |
| + Tier 1 only | ~1,200ms | 66 t/s | 83 t/s |
| + Tier 1+2 | ~400ms | 111 t/s | 86 t/s |
| + Tier 1+2+3 | ~50ms | 174 t/s | 88 t/s |

---

## 7. Recommended Next Steps

1. **Profile first** — Run `sample` (macOS) on `bench-kv-compact-model` during compaction to get actual flamegraph data. Confirms or refines the time breakdown in Section 1.

2. **Tier 1, item 1: Clear KV before set_data** — One-line change in the CLI tool (`llama_memory_seq_rm()` before `llama_state_seq_set_data()`). Should immediately improve restore speed.

3. **Tier 1, item 2: Reduce n_q** — Change `generate_cheap_qref` to cap at 32 instead of 64. Test quality impact.

4. **Tier 1, item 3: Value-norm pre-filter** — Before scoring, compute `||V[j]||₁` for all j. Drop bottom 30%. Straightforward to add as a pre-step in `kv_compact()`.

5. **Benchmark each change** — Re-run the codegen benchmark at 50k/100k/200k after each optimization to measure actual impact.
