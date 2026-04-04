# Speculative Decoding × KV Compaction — Synergy Analysis

Research notes on how heterogeneous speculative decoding (d-Matrix Corsair
blog) and Speculative Speculative Decoding (SSD/Saguaro, arXiv 2603.03251)
relate to KV cache compaction, and concrete paths to enhancing tg/s.

---

## Source Material

1. **Gimlet Labs blog:** "Low-Latency Speculative Decoding with d-Matrix
   Corsair" — disaggregated inference where a high-bandwidth SRAM accelerator
   (Corsair, 150 TB/s) runs the draft model while GPUs handle prefill +
   verification.

2. **arXiv 2603.03251:** "Speculative Speculative Decoding" (Kumar, Dao, May;
   ICLR 2026) — the Saguaro algorithm parallelizes drafting and verification
   by having the draft model predict verification outcomes during the verify
   step, pre-populating a "speculation cache" so drafts are ready instantly on
   cache hit.

---

## Key Ideas from the Sources

### Corsair / Heterogeneous Inference

| Insight | Detail |
|---------|--------|
| **Drafting is memory-bandwidth-bound** | Small draft models are bottlenecked by DRAM bandwidth, not compute. Corsair's 150 TB/s SRAM solves this. |
| **Longer draft sequences become viable** | When drafting is cheap, you can afford γ=20 tokens instead of γ=5 — even if tail acceptance rates drop, the amortized cost per accepted token falls. |
| **Per-token acceptance ≈ 92-96%** | On coding workloads with a 1.6B draft model vs. 120B target, demonstrating that small drafters can be very accurate. |
| **2-10× speedup over homogeneous GPU spec decode** | At equal or lower energy. The 10× case is energy-optimized. |

### SSD / Saguaro

| Insight | Detail |
|---------|--------|
| **Drafting and verification run simultaneously** | While the verifier processes round T, the speculator pre-computes round T+1 speculations for multiple possible verification outcomes. |
| **Geometric fan-out** | Budget allocation across lookahead positions follows F_k = F_0 × a_p^(k/(1+r)) — more guesses at positions with higher acceptance probability. |
| **Saguaro sampling** | Novel sampling scheme that trades off cache-hit rate vs. acceptance rate by concentrating residual probability on top draft tokens. |
| **Cache-miss fallback** | At low batch sizes, reuse the primary speculator; at high batch sizes, switch to a fast n-gram fallback. |
| **30% faster than optimized SD, up to 5× vs autoregressive** | On Llama-3.1-70B with Llama-3.2-1B drafter. |

---

## Why KV Compaction Is Complementary (Not Competing)

Speculative decoding and KV compaction optimize **different bottlenecks**:

```
Speculative Decoding     →  reduces LATENCY (tokens/sec per request)
  - Generates multiple tokens per verification step
  - Bounded by draft quality and verify throughput

KV Compaction            →  reduces MEMORY (enables higher throughput)
  - Shrinks KV cache by 50×
  - Enables larger batches, longer contexts, more concurrent requests
```

**They compose multiplicatively.** A system using both gets faster individual
requests (spec decode) AND can serve more requests concurrently (compacted
cache). Neither technique crowds out the other because they address orthogonal
resources (compute/bandwidth vs. memory).

---

## Concrete Synergies: How KV Compaction Enhances Spec Decode tg/s

### Synergy 1: Compacted Cache Enables Longer Draft Sequences

The Corsair blog shows that longer draft sequences (γ=20 vs γ=5) yield bigger
speedups — but only if the verifier can process them efficiently. Verification
cost scales as O(γ × T) for the attention over T cached tokens. With KV
compaction reducing T by 50×:

- **Verification of 20 draft tokens against a 50×-compacted cache** costs the
  same as verifying 20 tokens against the original — but the *memory* for the
  KV cache is 50× smaller.
- This frees GPU HBM for larger batch sizes during verification, increasing
  throughput.

**Quantified impact:** If verification is memory-bound (which it is during
decode), 50× KV compression translates to roughly proportional batch size
increase, multiplied by the spec decode speedup.

### Synergy 2: Draft Model KV Cache Compaction

The draft model also maintains a KV cache. For SSD/Saguaro, the speculator
maintains *multiple* cached speculations (the "speculation cache"). Compacting
the draft model's KV cache means:

- More speculation cache entries fit in memory
- Higher cache-hit rates (the geometric fan-out budget B can increase)
- Lower cache-miss penalty (fallback speculation is faster with compact cache)

The draft model is small (1-3B params), so its KV cache is already manageable,
but at long contexts (8k+ input as in the Corsair evaluation) it adds up,
especially with multiple cached speculations.

### Synergy 3: Heterogeneous Compaction Scheduling

The Corsair architecture splits work across hardware. KV compaction can be
scheduled during otherwise-idle cycles:

```
Timeline with Corsair + Compaction:

GPU:     [prefill] [verify₁] [verify₂] ... [verify_n]
Corsair: [draft₁ ] [draft₂ ] [draft₃ ] ... [draft_n ]
GPU:                [compact KV cache in background during verify]
```

Compaction (key selection + NNLS + least squares) is CPU/compute work that can
overlap with memory-bound verification on the GPU. The 3-4s compaction time
(at 60k tokens) can be hidden behind verification steps.

### Synergy 4: Compaction Improves SSD's Verification Speed

In SSD, faster verification = more time for the speculator to pre-compute
futures = higher cache hit rates. From the Saguaro paper's analysis, the
speedup bound is:

```
speedup_SSD ∝ (1 + T_SD) × (E_hit / E_SD)
```

Where T_SD is the verification time. With a compacted cache, attention
computation during verification is cheaper (fewer KV entries), reducing T_SD
and improving the ratio.

### Synergy 5: Compacted Cache Enables Wider Geometric Fan-out

Saguaro's geometric fan-out strategy allocates a "budget" B of pre-computed
speculations. Each speculation requires maintaining state (including KV cache
entries for the draft). Compacting these caches means:

- Budget B can increase without proportional memory increase
- Higher B → higher cache hit rate → less drafting latency
- The 90% cache hit rate reported in the paper could approach 95%+

---

## Concrete Enhancement Paths for kv-compact

### Path A: Compaction-Aware Spec Decode Integration (Highest Value)

Add a mode where kv-compact runs as a **compaction service** in a spec decode
pipeline:

1. After prefill, compact the target model's KV cache
2. Pass compacted cache to the verifier for all subsequent verification steps
3. Verification attention runs against t entries instead of T

**What to build:**
- A streaming compaction API: `compact_layer(layer_id, K, V, Q_ref) → (C_k, β, C_v)`
- Integration point in llama.cpp's `llama_decode` to use compacted cache
  during verification (this requires the beta injection from Epic 1)
- Benchmark: measure verification step latency with and without compaction

**Estimated tg/s impact:** At 50× compression on a 60k context, verification
attention memory reads drop by ~50×. For memory-bound decode, this could yield
2-5× faster verification steps, which directly translates to higher tg/s.

### Path B: Speculation-Cache-Aware Compaction (Novel Research)

Design compaction specifically for the SSD speculation cache pattern:

- Multiple speculations share a common KV cache prefix
- Compact the shared prefix aggressively (it's verified/stable)
- Keep speculation-specific suffixes at full resolution (they're speculative)

This is a **hierarchical compaction** scheme:
```
[heavily compacted shared prefix] + [full-resolution speculation suffix]
```

**What to build:**
- Prefix-aware compaction that identifies stable vs. speculative KV entries
- Incremental compaction: when a speculation is verified, merge its suffix
  into the compacted prefix

### Path C: Draft-Model-Targeted Compaction (Quick Win)

The draft model doesn't need perfect attention — it just needs to predict
tokens accurately enough for the verifier to accept them. This means:

- **More aggressive compression is tolerable** for draft model KV cache
  (100× instead of 50×)
- Quality degradation in the draft model only affects acceptance rate, not
  output quality (the verifier still guarantees correctness)
- This is a direct application of existing kv-compact with relaxed quality
  thresholds

**What to build:**
- Run kv-compact on the draft model's cache with aggressive settings
- Measure acceptance rate degradation vs. compression ratio
- Find the Pareto frontier: maximum compression before acceptance rate
  degrades enough to hurt tg/s

---

## Priority Ranking

| Priority | Path | Impact on tg/s | Feasibility | Depends On |
|----------|------|-----------------|-------------|------------|
| **1** | A: Compaction-aware verification | 2-5× verification speedup | Medium — needs beta injection (Epic 1) | US-1, US-2 |
| **2** | C: Aggressive draft compaction | 10-30% tg/s via memory savings | High — works with current POC | Nothing |
| **3** | B: Hierarchical speculation cache | Novel, potentially large | Low — requires SSD integration | Path A + SSD codebase |

---

## Key Takeaway

**KV compaction and speculative decoding are not alternatives — they're
multipliers.** Spec decode reduces latency per request; compaction reduces
memory per request. Together, a system can serve more concurrent requests,
each faster. The Corsair blog's heterogeneous architecture makes this even
more attractive: compaction can run on idle compute during memory-bound decode
phases.

The most immediate win (Path C) requires no new code — just running kv-compact
on a draft model's cache and measuring acceptance rate vs. compression. Path A
is the real prize but requires completing the beta injection work (Epic 1).

---

## References

- [Low-Latency Speculative Decoding with Corsair](https://gimletlabs.ai/blog/low-latency-spec-decode-corsair) — Gimlet Labs blog
- [Speculative Speculative Decoding](https://arxiv.org/abs/2603.03251) — Kumar, Dao, May (ICLR 2026)
- [Accelerating LLM Decoding with Speculative Execution](https://arxiv.org/abs/2211.17192) — Leviathan et al. (2022)
- [EAGLE: Speculative Sampling](https://arxiv.org/abs/2401.15077) — Li et al. (2024)
- [Fast KV Compaction via Attention Matching](https://arxiv.org/abs/2602.16284) — Zweiger et al. (2026)
