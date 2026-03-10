# RICE/COD Prioritization — Parallel Coding Agent Workload

**Target hardware:** AMD Ryzen AI 395 Pro Max, 128GB unified LPDDR5X, 8060S APU
**Constraint:** Bandwidth-bound (~200-250 GB/s), not capacity-bound
**Workload:** Multiple concurrent Qwen3 instances serving code generation
**Goal:** Maximize aggregate tg/s across N parallel coding agents

---

## Scoring Framework

### RICE Score = (Reach × Impact × Confidence) / Effort

| Factor | Scale | Meaning |
|--------|-------|---------|
| **Reach** | 1-10 | How many concurrent agent sessions benefit |
| **Impact** | 1-10 | tg/s improvement per session × quality preservation |
| **Confidence** | 0.0-1.0 | Probability the improvement materializes as expected |
| **Effort** | 1-10 | Dev weeks (1=days, 5=2-3 weeks, 10=months) |

### COD (Cost of Delay) = Impact if NOT done × Urgency

Captures: "What do we lose by deferring this?"

---

## The Parallel Agent Constraint Landscape

With N concurrent Qwen3 agents sharing 128GB:

```
Memory per agent ≈ 128GB / N  (minus model weights, shared via mmap)

Model weights (Qwen3-8B Q4):     ~4.5 GB (shared, loaded once)
Available for KV per agent:       (128 - 4.5) / N GB

At N=8 agents:  ~15.4 GB KV each → ~64K ctx at FP16 (comfortable)
At N=16 agents: ~7.7 GB KV each  → ~32K ctx at FP16 (tight for repos)
At N=32 agents: ~3.9 GB KV each  → ~16K ctx at FP16 (starving)
```

**With 50x KV compaction:**
```
At N=8 agents:  ~15.4 GB KV → equivalent ~3.2M ctx (unbounded)
At N=16 agents: ~7.7 GB KV  → equivalent ~1.6M ctx (unbounded)
At N=32 agents: ~3.9 GB KV  → equivalent ~800K ctx (huge)
```

Compaction doesn't just save memory — it **reduces bandwidth consumption**
during generation because the attention computation scans fewer KV entries.
On bandwidth-bound hardware, this directly converts to higher tg/s.

```
Bandwidth saving ≈ compression_ratio × n_layers × 2 × d_model bytes/token
At 50x compression: attention reads 50x less KV → massive tg/s gain
```

---

## Scored Feature List

### 1. Core Integration: Beta Injection + C_v Writeback (US-1 + US-2)

**What:** Inject beta biases into attention and write refitted values back.
Without this, compaction is compute-only with no runtime benefit.

| Factor | Score | Rationale |
|--------|-------|-----------|
| Reach | 10 | Every agent session, every decode step |
| Impact | 10 | Unlocks the entire compaction benefit. 0 → 50x compression. |
| Confidence | 0.9 | Well-understood; llama.cpp attention supports bias hooks |
| Effort | 4 | Requires attention graph modification in llama.cpp |

**RICE = (10 × 10 × 0.9) / 4 = 22.5**

**COD = 10** — Nothing else matters without this. Total blocker.

**Parallel agent impact:** This is the gate. With beta injection, each agent's
KV cache shrinks 50x → you run 8-32x more concurrent agents in the same
128GB. Without it, KV compaction is a benchmarking curiosity.

---

### 2. All-Layer/All-Head Compaction (US-3)

**What:** Extend compaction from single-layer demo to full model.

| Factor | Score | Rationale |
|--------|-------|-----------|
| Reach | 10 | Every agent, every layer |
| Impact | 9 | Layer 0 only = ~3% benefit; all layers = 100% benefit |
| Confidence | 0.95 | Per-head math is independent; trivial loop |
| Effort | 2 | Infrastructure exists; just iterate over layers/heads |

**RICE = (10 × 9 × 0.95) / 2 = 42.75**

**COD = 9** — Single-layer compaction is useless for production.

**Parallel agent impact:** Full-model compaction is what actually frees the
memory. The loop is embarrassingly parallel across heads (important for your
8060S which has many CUs).

---

### 3. Per-Head Non-Uniform Budget Allocation (US-5)

**What:** Give more budget to sensitive heads, less to compressible ones.

| Factor | Score | Rationale |
|--------|-------|-----------|
| Reach | 10 | Every agent session benefits from better quality |
| Impact | 8 | Paper's #1 ablation finding; largest quality contributor |
| Confidence | 0.85 | Proven in paper; precomputed per-model (do once for Qwen3) |
| Effort | 3 | Need sensitivity profiling + greedy allocation algorithm |

**RICE = (10 × 8 × 0.85) / 3 = 22.67**

**COD = 7** — Quality at 50x without budgets is significantly worse;
coding agents need high accuracy for correct code generation.

**Parallel agent impact:** Coding agents are quality-sensitive — a wrong
bracket or logic error wastes a full generation cycle. Better budgets =
fewer retries = higher effective throughput.

---

### 4. Quantized KV Cache Support (US-7)

**What:** Handle Q8_0/Q4_0 KV types (dequant for math, requant for storage).

| Factor | Score | Rationale |
|--------|-------|-----------|
| Reach | 10 | You'll run Q4 models; quantized KV is the default |
| Impact | 7 | Q4 KV + 50x compaction = extreme memory efficiency |
| Confidence | 0.8 | ggml has dequant routines; requant needs care |
| Effort | 3 | Dequant path exists in ggml; requant is the work |

**RICE = (10 × 7 × 0.8) / 3 = 18.67**

**COD = 6** — Without this, you'd need FP16 KV which doubles KV memory,
partially negating compaction benefit. Especially painful at N=16+ agents.

**Parallel agent impact:** Q4 KV + compaction means each agent's KV footprint
drops from ~1GB (32K ctx, FP16) to ~10MB (32K ctx, Q4, 50x compact). You
could theoretically run 100+ agents in 128GB (model weights become the
bottleneck, not KV).

---

### 5. Cheap Reference Query Generation (Tier 2, Item 4)

**What:** Generate Q_ref without repeat-prefill (40% of compaction time).

| Factor | Score | Rationale |
|--------|-------|-----------|
| Reach | 10 | Every compaction event across all agents |
| Impact | 7 | Halves compaction latency; matters for interactive agents |
| Confidence | 0.5 | K-vector proxy may be "good enough"; unclear quality delta |
| Effort | 5 | Research risk: need to validate alternatives |

**RICE = (10 × 7 × 0.5) / 5 = 7.0**

**COD = 4** — K-vector proxy works for now. Repeat-prefill quality is better
but the 2x cost is painful. Not blocking but limits how often you compact.

**Parallel agent impact:** Agents compact when context grows (every ~32K
tokens). Faster compaction = less generation stall. At 16 agents, stalls
cascade — one slow compaction blocks a decode batch slot.

---

### 6. C Library API (US-8)

**What:** Expose `llama_kv_compact()` as a callable API, not just CLI.

| Factor | Score | Rationale |
|--------|-------|-----------|
| Reach | 10 | Required for integration into any serving framework |
| Impact | 6 | Enables runtime compaction triggers (auto-compact at threshold) |
| Confidence | 0.9 | Well-defined interface; math library already modular |
| Effort | 3 | Header + wrapper functions + thread safety |

**RICE = (10 × 6 × 0.9) / 3 = 18.0**

**COD = 5** — CLI is fine for testing; API is needed for production serving
where agents auto-compact as context grows.

**Parallel agent impact:** With an API, the serving framework can compact
each agent's KV independently. Without it, you need subprocess spawning per
compaction — slow and clunky for N=16+ agents.

---

### 7. Iterative Refinement (Tier 1, Item 2)

**What:** After Steps 1-2-3, re-score keys and swap out poor ones. 2-3 rounds.

| Factor | Score | Rationale |
|--------|-------|-----------|
| Reach | 10 | Every compaction benefits |
| Impact | 5 | Improves quality at extreme compression (>20x) |
| Confidence | 0.7 | Theoretically sound; paper didn't explore; unknown magnitude |
| Effort | 2 | Simple loop: compact, evaluate, swap, re-compact |

**RICE = (10 × 5 × 0.7) / 2 = 17.5**

**COD = 3** — Quality is already good at moderate compression. Matters more
at 50x where coding accuracy is at risk.

**Parallel agent impact:** At 50x compression (needed for N=32 agents),
iterative refinement may be the difference between usable and broken code
generation quality.

---

### 8. Multi-Round Compaction (US-9)

**What:** Compact an already-compacted cache as conversation grows.

| Factor | Score | Rationale |
|--------|-------|-----------|
| Reach | 8 | Long coding sessions only (not short completions) |
| Impact | 7 | Without this, agents must restart after context fills |
| Confidence | 0.7 | Paper claims 6 rounds on AIME; coding tasks may differ |
| Effort | 3 | Beta accumulation logic; quality monitoring |

**RICE = (8 × 7 × 0.7) / 3 = 13.07**

**COD = 6** — Coding agents have long sessions (debugging loops, multi-file
edits). Without multi-round compaction, you hit context limits and lose
conversation history.

**Parallel agent impact:** Each agent potentially runs for hours. If
compaction is one-shot, agents hit a wall and need expensive re-prefill.
Multi-round keeps agents running indefinitely.

---

### 9. Better NNLS Solver — Lawson-Hanson (Tier 1, Item 3)

**What:** Replace projected gradient descent with active-set NNLS.

| Factor | Score | Rationale |
|--------|-------|-----------|
| Reach | 10 | Every compaction, every head |
| Impact | 3 | Faster convergence, slightly better beta; small overall effect |
| Confidence | 0.9 | Well-studied algorithm; drop-in replacement |
| Effort | 2 | Implement Lawson-Hanson for small dense problems |

**RICE = (10 × 3 × 0.9) / 2 = 13.5**

**COD = 1** — Current solver works. Improvement is incremental.

**Parallel agent impact:** Marginal. Compaction is already fast relative to
generation. Shaving 2ms off NNLS doesn't move the needle when you're
generating thousands of tokens.

---

### 10. Diversity-Aware Key Selection (Tier 2, Item 5)

**What:** Down-weight redundant keys during selection.

| Factor | Score | Rationale |
|--------|-------|-----------|
| Reach | 10 | Every compaction |
| Impact | 4 | Reduces wasted budget slots at high compression |
| Confidence | 0.6 | Simple heuristic; unclear magnitude vs. Highest Attention |
| Effort | 2 | Score modifier: `score *= (1 - max_sim_to_selected)` |

**RICE = (10 × 4 × 0.6) / 2 = 12.0**

**COD = 2** — Highest Attention works well enough; diversity helps at
extreme compression where it's needed.

**Parallel agent impact:** Moderate. At 50x compression, avoiding redundant
keys prevents quality cliffs in code generation.

---

### 11. Automated Quality Benchmarks (US-10)

**What:** CI-friendly benchmark suite: perplexity, token agreement, cosine sim.

| Factor | Score | Rationale |
|--------|-------|-----------|
| Reach | 10 | Validates every other feature on this list |
| Impact | 5 | Prevents quality regressions; guides tuning |
| Confidence | 0.95 | Straightforward tooling |
| Effort | 3 | Scripts, reference datasets, CI integration |

**RICE = (10 × 5 × 0.95) / 3 = 15.83**

**COD = 4** — Can test manually for now; becomes critical as features stack.

**Parallel agent impact:** When running 16+ agents, a quality regression
that causes 5% more code errors multiplied by 16 agents = major productivity
loss. Benchmarks catch this early.

---

### 12. Memory/Latency Profiling (US-11)

**What:** JSON profiling output: memory before/after, phase timing.

| Factor | Score | Rationale |
|--------|-------|-----------|
| Reach | 8 | Developers tuning for the APU setup |
| Impact | 4 | Guides ROI decisions; identifies bottlenecks |
| Confidence | 0.95 | Instrumentation is straightforward |
| Effort | 1 | Timer wrappers + JSON output |

**RICE = (8 × 4 × 0.95) / 1 = 30.4**

**COD = 2** — Nice to have; can use external profiling tools meanwhile.

---

### 13. ROCm/HIP Acceleration for Compaction Math

**What:** Port matrix ops to ROCm for the 8060S integrated GPU.

| Factor | Score | Rationale |
|--------|-------|-----------|
| Reach | 10 | Every compaction event |
| Impact | 6 | 8060S RDNA3.5 has ~16 TFLOPS FP16; CPU is ~1 TFLOPS |
| Confidence | 0.5 | ROCm on APU is immature; driver issues likely |
| Effort | 6 | hipBLAS integration, memory management, testing |

**RICE = (10 × 6 × 0.5) / 6 = 5.0**

**COD = 2** — CPU compaction is fast enough for the problem sizes involved
(t=50-500 per head). GPU wins only at 60K+ context with many heads.

**Parallel agent impact:** When 16 agents all trigger compaction simultaneously,
GPU-accelerated math prevents a CPU bottleneck. But compaction is rare
relative to generation, so the absolute impact is small.

---

### 14. Shared Prompt KV Cache Across Agents

**What:** Multiple agents share KV cache for common system prompt / repo context.

| Factor | Score | Rationale |
|--------|-------|-----------|
| Reach | 10 | Every agent shares system prompt overhead |
| Impact | 8 | Coding agents share 2-4K system prompt tokens; huge N× saving |
| Confidence | 0.6 | llama.cpp supports seq_id sharing; compaction interaction unclear |
| Effort | 4 | Fork KV at system prompt boundary; manage copy-on-write |

**RICE = (10 × 8 × 0.6) / 4 = 12.0**

**COD = 5** — At N=16 agents, shared system prompt saves 16 × 4K × 2 × d
bytes. Meaningful but not transformative compared to compaction.

**Parallel agent impact:** Direct N× multiplier on shared prefix memory.
Combined with compaction: shared prefix stays full-resolution, per-agent
suffix gets compacted. Best of both worlds.

---

## Final Rankings

### By RICE Score (do first = highest ROI)

| Rank | Feature | RICE | COD |
|------|---------|------|-----|
| 1 | All-layer/all-head compaction (US-3) | **42.75** | 9 |
| 2 | Memory/latency profiling (US-11) | **30.4** | 2 |
| 3 | Per-head budget allocation (US-5) | **22.67** | 7 |
| 4 | Beta injection + C_v writeback (US-1+2) | **22.5** | 10 |
| 5 | Quantized KV support (US-7) | **18.67** | 6 |
| 6 | C library API (US-8) | **18.0** | 5 |
| 7 | Iterative refinement | **17.5** | 3 |
| 8 | Automated benchmarks (US-10) | **15.83** | 4 |
| 9 | Better NNLS solver | **13.5** | 1 |
| 10 | Multi-round compaction (US-9) | **13.07** | 6 |
| 11 | Diversity-aware selection | **12.0** | 2 |
| 12 | Shared prompt cache | **12.0** | 5 |
| 13 | Cheap Q_ref generation | **7.0** | 4 |
| 14 | ROCm/HIP acceleration | **5.0** | 2 |

### By COD (cost of NOT doing = urgency)

| Rank | Feature | COD | RICE |
|------|---------|-----|------|
| 1 | Beta injection + C_v writeback (US-1+2) | **10** | 22.5 |
| 2 | All-layer/all-head compaction (US-3) | **9** | 42.75 |
| 3 | Per-head budget allocation (US-5) | **7** | 22.67 |
| 4 | Quantized KV support (US-7) | **6** | 18.67 |
| 4 | Multi-round compaction (US-9) | **6** | 13.07 |
| 6 | Shared prompt cache | **5** | 12.0 |
| 6 | C library API (US-8) | **5** | 18.0 |
| 8 | Automated benchmarks (US-10) | **4** | 15.83 |
| 8 | Cheap Q_ref generation | **4** | 7.0 |
| 10 | Iterative refinement | **3** | 17.5 |

---

## Recommended Implementation Order

Combining RICE (efficiency) and COD (urgency) with dependency analysis:

### Phase 1: Minimum Viable Compaction (weeks 1-2)
*Goal: KV compaction actually works end-to-end*

1. **Beta injection + C_v writeback** (COD=10, blocker for everything)
2. **All-layer/all-head compaction** (RICE=42.75, trivial once #1 works)
3. **Memory/latency profiling** (RICE=30.4, effort=1, validates the above)

**Parallel agent milestone:** Can run compaction on a real model and measure
actual memory savings. First estimate of how many Qwen3 agents fit in 128GB.

### Phase 2: Quality & Production (weeks 3-4)
*Goal: Quality good enough for code generation at high compression*

4. **Per-head budget allocation** (RICE=22.67, biggest quality lever)
5. **Quantized KV support** (RICE=18.67, required for Q4 Qwen3)
6. **Automated benchmarks** (RICE=15.83, validates quality for coding tasks)

**Parallel agent milestone:** Run N=8-16 Qwen3-8B-Q4 agents with compacted
KV caches. Measure aggregate tg/s and code generation quality.

### Phase 3: Scale & Sustain (weeks 5-6)
*Goal: Agents run indefinitely, integrate cleanly*

7. **C library API** (RICE=18.0, needed for serving integration)
8. **Multi-round compaction** (COD=6, agents need long sessions)
9. **Iterative refinement** (RICE=17.5, quality polish at extreme compression)

**Parallel agent milestone:** 16+ agents running persistent coding sessions
with auto-compaction. Full production deployment.

### Phase 4: Optimization (when needed)
*Goal: Squeeze remaining performance*

10. **Shared prompt cache** (synergy with compaction)
11. **Diversity-aware selection** (quality at 50x)
12. **Cheap Q_ref generation** (faster compaction cycles)
13. **Better NNLS solver** (marginal improvement)
14. **ROCm acceleration** (only if CPU compaction becomes bottleneck)

---

## Key Insight: Why This Order Differs From Pure RICE

Pure RICE would put profiling (#2) and NNLS solver (#9) higher than beta
injection (#4). But **COD breaks the tie**: without beta injection, nothing
works. The RICE score for beta injection is "only" 22.5 because the effort
is real (attention graph modification). But COD=10 means everything else is
blocked.

For the parallel coding agent workload specifically:
- **Quantized KV support jumps up** — you'll run Q4 models, so Q4 KV is
  the default path, not an edge case
- **Multi-round compaction jumps up** — coding sessions are long; one-shot
  compaction means agents die at context limits
- **ROCm acceleration drops** — compaction is rare relative to generation;
  CPU is fine for the math given the problem sizes
- **Shared prompt cache becomes relevant** — N agents × shared system prompt
  is a direct N× memory win
